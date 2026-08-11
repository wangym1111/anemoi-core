# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from __future__ import annotations

import argparse
import logging
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import torch

from anemoi.datasets import open_dataset
from anemoi.training.commands import Command

LOG = logging.getLogger(__name__)


class Latent(Command):
    """Encode and decode latent states for Anemoi autoencoders."""

    @staticmethod
    def add_arguments(command_parser: argparse.ArgumentParser) -> None:
        subparsers = command_parser.add_subparsers(dest="subcommand", required=True)

        encode = subparsers.add_parser(
            "encode-dataset",
            help="Encode an Anemoi Zarr dataset into latent-state Zarr arrays.",
            description="Encode [date, variable, ensemble, grid] Anemoi data into z[sample, node, channel].",
        )
        encode.add_argument("--checkpoint", "-c", required=True, type=Path)
        encode.add_argument("--dataset", "-d", required=True)
        encode.add_argument("--output", "-o", required=True, type=Path)
        encode.add_argument("--dataset-name", default=None)
        encode.add_argument("--batch-size", type=int, default=1)
        encode.add_argument("--member-index", type=int, default=0)
        encode.add_argument("--start-index", type=int, default=0)
        encode.add_argument("--end-index", type=int, default=None)
        encode.add_argument("--device", default="auto")
        encode.add_argument("--zarr-chunk-samples", type=int, default=1)

        decode = subparsers.add_parser(
            "decode-latent",
            help="Decode latent-state Zarr arrays back to reconstructed Anemoi fields.",
            description="Decode z[sample, node, channel] into reconstruction[sample, variable, grid].",
        )
        decode.add_argument("--checkpoint", "-c", required=True, type=Path)
        decode.add_argument("--latent", "-l", required=True, type=Path)
        decode.add_argument("--output", "-o", required=True, type=Path)
        decode.add_argument("--template-dataset", "-d", required=True)
        decode.add_argument("--dataset-name", default=None)
        decode.add_argument("--batch-size", type=int, default=1)
        decode.add_argument("--member-index", type=int, default=0)
        decode.add_argument("--device", default="auto")
        decode.add_argument("--latent-key", default="z")
        decode.add_argument("--zarr-chunk-samples", type=int, default=1)

    @staticmethod
    def run(args: argparse.Namespace) -> None:
        if args.subcommand == "encode-dataset":
            path = encode_dataset(
                checkpoint=args.checkpoint,
                dataset=args.dataset,
                output=args.output,
                dataset_name=args.dataset_name,
                batch_size=args.batch_size,
                member_index=args.member_index,
                start_index=args.start_index,
                end_index=args.end_index,
                device=args.device,
                zarr_chunk_samples=args.zarr_chunk_samples,
            )
            LOG.info("Latent state written to %s", path)
            return

        if args.subcommand == "decode-latent":
            path = decode_latent(
                checkpoint=args.checkpoint,
                latent=args.latent,
                output=args.output,
                template_dataset=args.template_dataset,
                dataset_name=args.dataset_name,
                batch_size=args.batch_size,
                member_index=args.member_index,
                device=args.device,
                latent_key=args.latent_key,
                zarr_chunk_samples=args.zarr_chunk_samples,
            )
            LOG.info("Decoded state written to %s", path)
            return


def encode_dataset(
    *,
    checkpoint: Path,
    dataset: str | dict[str, Any],
    output: Path,
    dataset_name: str | None = None,
    batch_size: int = 1,
    member_index: int = 0,
    start_index: int = 0,
    end_index: int | None = None,
    device: str = "auto",
    zarr_chunk_samples: int = 1,
) -> Path:
    """Encode an Anemoi dataset into ``z``, ``mu`` and ``logvar`` Zarr arrays."""

    zarr = _require_zarr()
    model = _load_anemoi_model(checkpoint, device=device)
    dataset_name = dataset_name or _default_dataset_name(model)
    data = open_dataset(dataset)
    _validate_anemoi_dataset_shape(data, member_index)

    indices = list(_sample_indices(data.shape[0], start_index, end_index))
    if not indices:
        msg = "No samples selected for latent encoding."
        raise ValueError(msg)

    output.parent.mkdir(parents=True, exist_ok=True)
    root = zarr.open_group(str(output), mode="w")
    root.attrs["checkpoint"] = str(checkpoint)
    root.attrs["source_dataset"] = str(dataset)
    root.attrs["source_shape"] = tuple(int(v) for v in data.shape)
    root.attrs["source_layout"] = "date_variable_ensemble_grid"
    root.attrs["model_input_layout"] = "batch_time_grid_variable"
    root.attrs["dataset_name"] = dataset_name
    root.attrs["member_index"] = int(member_index)
    root.attrs["sample_index_start"] = int(indices[0])
    root.attrs["sample_index_end"] = int(indices[-1]) + 1

    z_array = None
    logvar_array = None
    sample_index_array = root.create_dataset(
        "sample_index",
        shape=(len(indices),),
        chunks=(max(1, zarr_chunk_samples),),
        dtype="i8",
    )

    cursor = 0
    for batch_indices in _batched(indices, batch_size):
        batch = _read_anemoi_batch(data, batch_indices, member_index).to(model_device(model))
        latent = model.encode_latent_step({dataset_name: batch})
        z = _as_numpy(latent["z"])
        logvar = np.zeros_like(z, dtype=z.dtype)

        if z_array is None:
            chunks = (max(1, zarr_chunk_samples), z.shape[1], z.shape[2])
            z_array = root.create_dataset(
                "z",
                shape=(len(indices), z.shape[1], z.shape[2]),
                chunks=chunks,
                dtype=z.dtype,
            )
            root.create_dataset("mu", shape=z_array.shape, chunks=chunks, dtype=z.dtype)
            logvar_array = root.create_dataset("logvar", shape=z_array.shape, chunks=chunks, dtype=z.dtype)
            root.attrs["latent_shape"] = tuple(int(v) for v in z_array.shape)

        next_cursor = cursor + z.shape[0]
        z_array[cursor:next_cursor] = z
        root["mu"][cursor:next_cursor] = z
        logvar_array[cursor:next_cursor] = logvar
        sample_index_array[cursor:next_cursor] = np.asarray(batch_indices, dtype=np.int64)
        cursor = next_cursor

    return output


def decode_latent(
    *,
    checkpoint: Path,
    latent: Path,
    output: Path,
    template_dataset: str | dict[str, Any],
    dataset_name: str | None = None,
    batch_size: int = 1,
    member_index: int = 0,
    device: str = "auto",
    latent_key: str = "z",
    zarr_chunk_samples: int = 1,
) -> Path:
    """Decode latent Zarr arrays into reconstructed physical-space fields."""

    zarr = _require_zarr()
    model = _load_anemoi_model(checkpoint, device=device)
    dataset_name = dataset_name or _default_dataset_name(model)
    latent_root = zarr.open_group(str(latent), mode="r")
    if latent_key not in latent_root:
        msg = f"Latent store does not contain {latent_key!r}."
        raise KeyError(msg)
    z_array = latent_root[latent_key]
    if "sample_index" in latent_root:
        sample_indices = np.asarray(latent_root["sample_index"])
    else:
        sample_indices = np.arange(z_array.shape[0])
    data = open_dataset(template_dataset)
    _validate_anemoi_dataset_shape(data, member_index)

    output.parent.mkdir(parents=True, exist_ok=True)
    root = zarr.open_group(str(output), mode="w")
    root.attrs["checkpoint"] = str(checkpoint)
    root.attrs["latent"] = str(latent)
    root.attrs["template_dataset"] = str(template_dataset)
    root.attrs["dataset_name"] = dataset_name
    root.attrs["member_index"] = int(member_index)
    root.attrs["output_layout"] = "sample_variable_grid"

    reconstruction_array = None
    for start in range(0, int(z_array.shape[0]), batch_size):
        stop = min(start + batch_size, int(z_array.shape[0]))
        selected = [int(i) for i in sample_indices[start:stop]]
        template = _read_anemoi_batch(data, selected, member_index).to(model_device(model))
        z = torch.as_tensor(np.asarray(z_array[start:stop]), device=model_device(model))
        decoded = model.decode_latent_step({"z": z}, {dataset_name: template})
        reconstruction = _as_numpy(decoded[dataset_name][:, 0])
        reconstruction = np.moveaxis(reconstruction, 2, 1)

        if reconstruction_array is None:
            chunks = (max(1, zarr_chunk_samples), reconstruction.shape[1], reconstruction.shape[2])
            reconstruction_array = root.create_dataset(
                "reconstruction",
                shape=(int(z_array.shape[0]), reconstruction.shape[1], reconstruction.shape[2]),
                chunks=chunks,
                dtype=reconstruction.dtype,
            )
            root.attrs["reconstruction_shape"] = tuple(int(v) for v in reconstruction_array.shape)

        reconstruction_array[start:stop] = reconstruction

    return output


def _load_anemoi_model(checkpoint: Path, *, device: str) -> torch.nn.Module:
    if device == "auto":
        target_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        target_device = torch.device(device)
    try:
        from anemoi.training.utils.checkpoint import load_and_prepare_model

        model, _metadata = load_and_prepare_model(str(checkpoint))
    except (RuntimeError, TypeError, ValueError, AttributeError, ImportError, FileNotFoundError):
        LOG.debug("Falling back to direct torch.load for %s", checkpoint, exc_info=True)
        model = torch.load(checkpoint, map_location=target_device, weights_only=False)
        if isinstance(model, dict) and "model" in model:
            model = model["model"]
    if not isinstance(model, torch.nn.Module):
        msg = f"Checkpoint {checkpoint} did not load to a torch.nn.Module; got {type(model)!r}."
        raise TypeError(msg)
    model.to(target_device)
    model.eval()
    return model


def _validate_anemoi_dataset_shape(data: Any, member_index: int) -> None:
    if len(data.shape) != 4:
        msg = f"Expected Anemoi dataset shape [date, variable, ensemble, grid], got {data.shape}."
        raise ValueError(msg)
    if member_index < 0 or member_index >= int(data.shape[2]):
        msg = f"member_index {member_index} is out of range for dataset shape {data.shape}."
        raise IndexError(msg)


def _read_anemoi_batch(data: Any, indices: list[int], member_index: int) -> torch.Tensor:
    selection: slice | list[int]
    if indices and indices == list(range(indices[0], indices[-1] + 1)):
        selection = slice(indices[0], indices[-1] + 1)
    else:
        selection = indices
    values = np.asarray(data[selection, :, member_index, :], dtype=np.float32)
    values = np.moveaxis(values, 1, 2)
    values = values[:, None, :, :]
    return torch.from_numpy(values)


def _sample_indices(length: int, start_index: int, end_index: int | None) -> range:
    end = length if end_index is None else min(end_index, length)
    if start_index < 0 or end < start_index:
        msg = f"Invalid sample range start={start_index}, end={end_index}, length={length}."
        raise ValueError(msg)
    return range(start_index, end)


def _batched(values: list[int], batch_size: int) -> Iterable[list[int]]:
    if batch_size < 1:
        msg = f"batch_size must be >= 1, got {batch_size}."
        raise ValueError(msg)
    for start in range(0, len(values), batch_size):
        yield values[start : start + batch_size]


def _as_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()


def model_device(model: torch.nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _default_dataset_name(model: torch.nn.Module) -> str:
    names = getattr(model, "dataset_names", None)
    if names is None and hasattr(model, "model"):
        names = getattr(model.model, "dataset_names", None)
    if not names:
        msg = "Could not infer dataset name from model; pass --dataset-name."
        raise ValueError(msg)
    if len(names) != 1:
        msg = f"Model has multiple datasets {names}; pass --dataset-name."
        raise ValueError(msg)
    return str(names[0])


def _require_zarr() -> Any:
    try:
        import zarr
    except ModuleNotFoundError as exc:
        msg = "The latent command requires zarr. Install anemoi-training dependencies."
        raise RuntimeError(msg) from exc
    return zarr


command = Latent
