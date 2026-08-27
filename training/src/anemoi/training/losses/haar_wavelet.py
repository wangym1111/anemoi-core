# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Haar wavelet losses."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Literal

import torch
import torch.nn.functional as F
from torch.distributed.distributed_c10d import ProcessGroup

from anemoi.training.losses.base import BaseLoss
from anemoi.training.losses.base import Squash_mode
from anemoi.training.utils.enums import TensorDim

LOGGER = logging.getLogger(__name__)

CoefficientNormalization = Literal["orthonormal", "average"]
LossNormalization = Literal["none", "band", "scale"]
OddSizeMode = Literal["error", "trim", "pad"]
WaveletLoss = Literal["mse", "huber"]

DETAIL_SUBBANDS = ("lh", "hl", "hh")
SUBBANDS = (*DETAIL_SUBBANDS, "ll")


class HaarWaveletLoss(BaseLoss):
    """2-D Haar wavelet multiscale loss for flattened regular grids.

    The input tensors use Anemoi's standard loss shape
    ``(batch, time, ensemble, grid, variables)``. The flattened grid is
    reshaped to ``(y_dim, x_dim)`` before the wavelet decomposition.

    Existing Anemoi scalers are applied to the spatial residual before the
    wavelet transform. For MSE with the default orthonormal Haar coefficients,
    summing all detail bands and the final low-pass band with
    ``normalization="none"`` preserves the weighted spatial squared-error
    energy while allowing scale and subband reweighting.
    """

    name: str = "haar_wavelet"

    def __init__(
        self,
        x_dim: int,
        y_dim: int,
        *,
        num_scales: int | None = None,
        level_weights: list[float] | None = None,
        lowpass_weight: float = 1.0,
        subband_weights: Mapping[str, float] | None = None,
        include_lowpass: bool = True,
        normalization: LossNormalization = "none",
        coefficient_normalization: CoefficientNormalization = "orthonormal",
        odd_size_mode: OddSizeMode = "pad",
        loss: WaveletLoss = "mse",
        delta: float = 1.0,
        ignore_nans: bool = False,
    ) -> None:
        """Initialise a Haar wavelet loss.

        Parameters
        ----------
        x_dim : int
            Size of the regular-grid x dimension.
        y_dim : int
            Size of the regular-grid y dimension.
        num_scales : int | None, optional
            Number of repeated Haar decompositions. Defaults to the maximum
            possible number for ``x_dim``, ``y_dim`` and ``odd_size_mode``.
        level_weights : list[float] | None, optional
            Per-detail-level weights. Length must match ``num_scales``.
        lowpass_weight : float, optional
            Weight for the final ``ll`` approximation band.
        subband_weights : Mapping[str, float] | None, optional
            Optional weights for ``lh``, ``hl``, ``hh`` and ``ll`` subbands.
        include_lowpass : bool, optional
            Include the final low-pass approximation coefficients.
        normalization : {"none", "band", "scale"}, optional
            ``"none"`` keeps Anemoi's scaler/reduction semantics, ``"band"``
            averages each subband over its coefficients, and ``"scale"``
            averages all detail subbands within a decomposition level together.
        coefficient_normalization : {"orthonormal", "average"}, optional
            Haar coefficient normalization. ``"orthonormal"`` uses the
            energy-preserving divisor 2. ``"average"`` uses divisor 4.
        odd_size_mode : {"error", "trim", "pad"}, optional
            How to handle odd spatial dimensions at each level.
        loss : {"mse", "huber"}, optional
            Pointwise loss applied to wavelet residual coefficients.
        delta : float, optional
            Huber threshold when ``loss="huber"``.
        ignore_nans : bool, optional
            Mask NaNs in predictions or targets before computing residuals.
        """
        super().__init__(ignore_nans=ignore_nans)

        if x_dim <= 0 or y_dim <= 0:
            msg = f"x_dim and y_dim must be positive, got x_dim={x_dim}, y_dim={y_dim}."
            raise ValueError(msg)
        if normalization not in ("none", "band", "scale"):
            msg = f"Unknown normalization {normalization!r}; expected one of 'none', 'band' or 'scale'."
            raise ValueError(msg)
        if coefficient_normalization not in ("orthonormal", "average"):
            msg = (
                f"Unknown coefficient_normalization {coefficient_normalization!r}; "
                "expected 'orthonormal' or 'average'."
            )
            raise ValueError(msg)
        if odd_size_mode not in ("error", "trim", "pad"):
            msg = f"Unknown odd_size_mode {odd_size_mode!r}; expected one of 'error', 'trim' or 'pad'."
            raise ValueError(msg)
        if loss not in ("mse", "huber"):
            msg = f"Unknown wavelet loss {loss!r}; expected 'mse' or 'huber'."
            raise ValueError(msg)
        if delta <= 0:
            msg = f"delta must be positive, got {delta}."
            raise ValueError(msg)

        max_scales = self._max_decomposition_levels(y_dim, x_dim, odd_size_mode)
        if num_scales is None:
            num_scales = max_scales
        if num_scales < 1:
            msg = f"num_scales must be at least 1 for HaarWaveletLoss, got {num_scales}."
            raise ValueError(msg)
        if num_scales > max_scales:
            msg = (
                f"num_scales={num_scales} exceeds the {max_scales} available Haar levels "
                f"for grid ({y_dim}, {x_dim}) with odd_size_mode={odd_size_mode!r}."
            )
            raise ValueError(msg)

        if level_weights is None:
            level_weights = [1.0] * num_scales
        if len(level_weights) != num_scales:
            msg = f"Expected {num_scales} level_weights, got {len(level_weights)}."
            raise ValueError(msg)
        if any(weight < 0 for weight in level_weights):
            msg = "level_weights must be non-negative."
            raise ValueError(msg)
        if lowpass_weight < 0:
            msg = "lowpass_weight must be non-negative."
            raise ValueError(msg)

        resolved_subband_weights = dict.fromkeys(SUBBANDS, 1.0)
        if subband_weights is not None:
            unknown = set(subband_weights) - set(SUBBANDS)
            if unknown:
                msg = f"Unknown Haar subband weight keys {sorted(unknown)}; expected keys from {SUBBANDS}."
                raise ValueError(msg)
            resolved_subband_weights.update(subband_weights)
        if any(weight < 0 for weight in resolved_subband_weights.values()):
            msg = "subband_weights must be non-negative."
            raise ValueError(msg)

        self.x_dim = x_dim
        self.y_dim = y_dim
        self.wavelet_scales = num_scales
        self.num_scales = num_scales + int(include_lowpass)
        self.level_weights = [float(weight) for weight in level_weights]
        self.lowpass_weight = float(lowpass_weight)
        self.subband_weights = {name: float(weight) for name, weight in resolved_subband_weights.items()}
        self.include_lowpass = include_lowpass
        self.normalization = normalization
        self.coefficient_normalization = coefficient_normalization
        self.odd_size_mode = odd_size_mode
        self.loss = loss
        self.delta = delta

        # The framework will gather grid-sharded tensors before calling losses
        # that do not support sharding. A 2-D wavelet transform needs the full
        # rectangular grid on each rank.
        self.supports_sharding = False

    @staticmethod
    def _max_decomposition_levels(height: int, width: int, odd_size_mode: OddSizeMode) -> int:
        """Return how many 2-D Haar decomposition levels can be applied."""
        levels = 0
        while height >= 2 and width >= 2:
            if odd_size_mode == "error" and (height % 2 != 0 or width % 2 != 0):
                break
            if odd_size_mode == "pad":
                height = (height + 1) // 2
                width = (width + 1) // 2
            else:
                height = height // 2
                width = width // 2
            levels += 1
        return levels

    def _reshape_grid(self, x: torch.Tensor) -> torch.Tensor:
        expected_grid = self.y_dim * self.x_dim
        if x.shape[TensorDim.GRID] != expected_grid:
            msg = (
                "HaarWaveletLoss expected the flattened grid dimension to equal "
                f"y_dim * x_dim = {self.y_dim} * {self.x_dim} = {expected_grid}, "
                f"got {x.shape[TensorDim.GRID]}."
            )
            raise ValueError(msg)
        return x.unflatten(int(TensorDim.GRID), (self.y_dim, self.x_dim))

    def _prepare_even_grid(self, x: torch.Tensor, level: int) -> torch.Tensor:
        height = x.shape[-3]
        width = x.shape[-2]
        pad_height = height % 2
        pad_width = width % 2

        if pad_height == 0 and pad_width == 0:
            return x

        if self.odd_size_mode == "error":
            msg = (
                "HaarWaveletLoss encountered an odd grid size at decomposition "
                f"level {level}: height={height}, width={width}."
            )
            raise ValueError(msg)

        if self.odd_size_mode == "trim":
            return x[..., : height - pad_height, : width - pad_width, :]

        return F.pad(x, (0, 0, 0, pad_width, 0, pad_height), mode="constant", value=0.0)

    def _haar_step(self, x: torch.Tensor, level: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Apply one separable 2-D Haar decomposition step."""
        x = self._prepare_even_grid(x, level)
        top_left = x[..., 0::2, 0::2, :]
        top_right = x[..., 0::2, 1::2, :]
        bottom_left = x[..., 1::2, 0::2, :]
        bottom_right = x[..., 1::2, 1::2, :]

        divisor = 2.0 if self.coefficient_normalization == "orthonormal" else 4.0
        ll = (top_left + top_right + bottom_left + bottom_right) / divisor
        lh = (top_left - top_right + bottom_left - bottom_right) / divisor
        hl = (top_left + top_right - bottom_left - bottom_right) / divisor
        hh = (top_left - top_right - bottom_left + bottom_right) / divisor
        return ll, {"lh": lh, "hl": hl, "hh": hh}

    @staticmethod
    def _flatten_coefficients(x: torch.Tensor) -> torch.Tensor:
        return x.flatten(start_dim=int(TensorDim.GRID), end_dim=int(TensorDim.GRID) + 1)

    def _coefficient_loss(self, coefficients: torch.Tensor) -> torch.Tensor:
        if self.loss == "mse":
            return torch.square(coefficients)

        diff = torch.abs(coefficients)
        return torch.where(diff < self.delta, 0.5 * torch.square(coefficients), self.delta * (diff - 0.5 * self.delta))

    def _normalize_loss(self, loss_tensor: torch.Tensor, subbands_in_scale: int) -> torch.Tensor:
        if self.normalization == "none":
            return loss_tensor

        coefficient_count = loss_tensor.shape[TensorDim.GRID]
        divisor = coefficient_count if self.normalization == "band" else coefficient_count * subbands_in_scale
        return loss_tensor / divisor

    def _scale_residual(
        self,
        residual: torch.Tensor,
        scaler_indices: tuple[int, ...] | None,
        without_scalers: list[str] | list[int] | None,
        grid_shard_slice: slice | None,
    ) -> torch.Tensor:
        """Apply Anemoi loss scalers to the spatial residual before decomposition."""
        if len(self.scaler) == 0:
            return residual if scaler_indices is None else residual[scaler_indices]

        weights = self.scale(
            torch.ones_like(residual),
            scaler_indices,
            without_scalers=without_scalers,
            grid_shard_slice=grid_shard_slice,
        )
        residual = residual if scaler_indices is None else residual[scaler_indices]
        return residual * torch.sqrt(weights)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        squash: bool = True,
        *,
        scaler_indices: tuple[int, ...] | None = None,
        without_scalers: list[str] | list[int] | None = None,
        grid_shard_slice: slice | None = None,
        group: ProcessGroup | None = None,
        squash_mode: Squash_mode = "avg",
        **_kwargs,
    ) -> torch.Tensor:
        """Compute the weighted Haar wavelet loss."""
        if grid_shard_slice is not None:
            msg = (
                "HaarWaveletLoss requires full-grid tensors. In normal Anemoi training this is handled by "
                "supports_sharding=False before the loss is called."
            )
            raise ValueError(msg)

        del group  # This loss asks the training framework to gather sharded tensors before calling it.

        pred, target = self.mask_nans(pred, target)
        residual = self._scale_residual(
            pred - target,
            scaler_indices=scaler_indices,
            without_scalers=without_scalers,
            grid_shard_slice=grid_shard_slice,
        )
        current = self._reshape_grid(residual)

        weighted_losses: list[torch.Tensor] = []
        for level in range(1, self.wavelet_scales + 1):
            current, details = self._haar_step(current, level)
            for subband_name, coefficients in details.items():
                weight = self.level_weights[level - 1] * self.subband_weights[subband_name]
                if weight == 0.0:
                    continue
                loss_tensor = self._coefficient_loss(self._flatten_coefficients(coefficients))
                loss_tensor = self._normalize_loss(loss_tensor, subbands_in_scale=len(DETAIL_SUBBANDS))
                weighted_losses.append(
                    weight * self.reduce(loss_tensor, squash=squash, group=None, squash_mode=squash_mode),
                )

        if self.include_lowpass and self.lowpass_weight != 0.0 and self.subband_weights["ll"] != 0.0:
            lowpass = self._coefficient_loss(self._flatten_coefficients(current))
            lowpass = self._normalize_loss(lowpass, subbands_in_scale=1)
            weighted_losses.append(
                self.lowpass_weight
                * self.subband_weights["ll"]
                * self.reduce(lowpass, squash=squash, group=None, squash_mode=squash_mode),
            )

        if not weighted_losses:
            zero = pred.new_zeros(())
            if squash:
                return zero
            return pred.new_zeros((residual.shape[TensorDim.VARIABLE],))

        return torch.stack(weighted_losses).sum(dim=0)
