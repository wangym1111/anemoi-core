# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import pytest
import torch
from omegaconf import DictConfig

from anemoi.training.losses import HaarWaveletLoss
from anemoi.training.losses import MSELoss
from anemoi.training.losses import get_loss_function
from anemoi.training.utils.enums import TensorDim


def test_haar_wavelet_loss_matches_mse_for_complete_orthonormal_transform() -> None:
    """A complete orthonormal Haar decomposition preserves squared-error energy."""
    pred = torch.randn(2, 3, 1, 16, 2, dtype=torch.float64)
    target = torch.randn(2, 3, 1, 16, 2, dtype=torch.float64)

    wavelet = HaarWaveletLoss(x_dim=4, y_dim=4, num_scales=2, normalization="none")
    mse = MSELoss()

    torch.testing.assert_close(
        wavelet(pred, target, squash=False),
        mse(pred, target, squash=False),
    )
    torch.testing.assert_close(wavelet(pred, target), mse(pred, target))


def test_haar_wavelet_loss_backpropagates_through_padded_odd_grid() -> None:
    pred = torch.randn(2, 2, 1, 15, 3, requires_grad=True)
    target = torch.randn(2, 2, 1, 15, 3)

    loss = HaarWaveletLoss(
        x_dim=5,
        y_dim=3,
        num_scales=2,
        normalization="band",
        odd_size_mode="pad",
    )

    out = loss(pred, target)
    out.backward()

    assert torch.isfinite(out)
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()
    assert not torch.allclose(pred.grad, torch.zeros_like(pred.grad))


def test_haar_wavelet_loss_rejects_odd_grid_when_configured_to_error() -> None:
    with pytest.raises(ValueError, match="exceeds the 0 available Haar levels"):
        HaarWaveletLoss(x_dim=5, y_dim=3, num_scales=1, odd_size_mode="error")


def test_haar_wavelet_loss_level_weights_are_scale_sensitive() -> None:
    checkerboard = torch.tensor(
        [
            [1.0, -1.0, 1.0, -1.0],
            [-1.0, 1.0, -1.0, 1.0],
            [1.0, -1.0, 1.0, -1.0],
            [-1.0, 1.0, -1.0, 1.0],
        ],
    )
    pred = checkerboard.reshape(1, 1, 1, 16, 1)
    target = torch.zeros_like(pred)

    fine_scale = HaarWaveletLoss(
        x_dim=4,
        y_dim=4,
        num_scales=2,
        level_weights=[1.0, 0.0],
        include_lowpass=False,
    )
    coarse_scale = HaarWaveletLoss(
        x_dim=4,
        y_dim=4,
        num_scales=2,
        level_weights=[0.0, 1.0],
        include_lowpass=False,
    )

    assert fine_scale(pred, target) > 0
    torch.testing.assert_close(coarse_scale(pred, target), torch.tensor(0.0))


def test_haar_wavelet_loss_subband_weights_are_applied() -> None:
    checkerboard = torch.tensor(
        [
            [1.0, -1.0],
            [-1.0, 1.0],
        ],
    )
    pred = checkerboard.reshape(1, 1, 1, 4, 1)
    target = torch.zeros_like(pred)

    keep_hh = HaarWaveletLoss(x_dim=2, y_dim=2, num_scales=1, include_lowpass=False)
    drop_hh = HaarWaveletLoss(
        x_dim=2,
        y_dim=2,
        num_scales=1,
        include_lowpass=False,
        subband_weights={"hh": 0.0},
    )

    assert keep_hh(pred, target) > 0
    torch.testing.assert_close(drop_hh(pred, target), torch.tensor(0.0))


def test_haar_wavelet_loss_preserves_spatial_scaler_semantics() -> None:
    pred = torch.zeros(1, 1, 1, 4, 2)
    pred[..., 0, 0] = 1.0
    pred[..., 1, 0] = 10.0
    pred[..., 2, 1] = 3.0
    target = torch.zeros_like(pred)

    grid_weights = torch.tensor([1.0, 0.0, 1.0, 1.0])
    variable_weights = torch.tensor([2.0, 3.0])

    wavelet = HaarWaveletLoss(x_dim=2, y_dim=2, num_scales=1, normalization="none")
    wavelet.add_scaler(TensorDim.GRID, grid_weights, name="grid")
    wavelet.add_scaler(TensorDim.VARIABLE, variable_weights, name="variable")

    mse = MSELoss()
    mse.add_scaler(TensorDim.GRID, grid_weights, name="grid")
    mse.add_scaler(TensorDim.VARIABLE, variable_weights, name="variable")

    torch.testing.assert_close(wavelet(pred, target, squash=False), mse(pred, target, squash=False))


def test_haar_wavelet_loss_supports_scaler_indices() -> None:
    pred = torch.ones(1, 1, 1, 4, 2)
    target = torch.zeros_like(pred)
    selected_variable = torch.tensor([1])

    loss = HaarWaveletLoss(x_dim=2, y_dim=2, num_scales=1, normalization="none")
    loss.add_scaler(TensorDim.GRID, torch.ones(4), name="grid")
    loss.add_scaler(TensorDim.VARIABLE, torch.tensor([2.0, 3.0]), name="variable")

    out = loss(pred, target, squash=False, scaler_indices=(..., selected_variable))

    assert out.shape == (1,)
    torch.testing.assert_close(out, torch.tensor([12.0]))


def test_haar_wavelet_loss_instantiates_from_loss_config() -> None:
    loss = get_loss_function(
        DictConfig(
            {
                "_target_": "anemoi.training.losses.HaarWaveletLoss",
                "x_dim": 4,
                "y_dim": 4,
                "num_scales": 2,
                "normalization": "scale",
                "scalers": [],
            },
        ),
    )

    assert isinstance(loss, HaarWaveletLoss)
    assert loss.supports_sharding is False
