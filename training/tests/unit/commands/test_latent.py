# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import numpy as np
import pytest

from anemoi.training.commands.latent import _batched
from anemoi.training.commands.latent import _default_dataset_name
from anemoi.training.commands.latent import _read_anemoi_batch
from anemoi.training.commands.latent import _sample_indices
from anemoi.training.commands.latent import _validate_anemoi_dataset_shape


def test_read_anemoi_batch_converts_dataset_layout() -> None:
    data = np.arange(3 * 4 * 2 * 5, dtype=np.float32).reshape(3, 4, 2, 5)

    batch = _read_anemoi_batch(data, [0, 1], member_index=1)

    assert tuple(batch.shape) == (2, 1, 5, 4)
    np.testing.assert_array_equal(batch[:, 0].numpy(), np.moveaxis(data[0:2, :, 1, :], 1, 2))


def test_sample_indices_validates_range() -> None:
    assert list(_sample_indices(10, 2, 5)) == [2, 3, 4]

    with pytest.raises(ValueError, match="Invalid sample range"):
        list(_sample_indices(10, -1, 5))


def test_batched_requires_positive_batch_size() -> None:
    assert list(_batched([0, 1, 2], 2)) == [[0, 1], [2]]

    with pytest.raises(ValueError, match="batch_size"):
        list(_batched([0], 0))


def test_default_dataset_name() -> None:
    class Model:
        dataset_names = ["data"]

    assert _default_dataset_name(Model()) == "data"


def test_validate_anemoi_dataset_shape() -> None:
    data = np.zeros((2, 3, 1, 4), dtype=np.float32)

    _validate_anemoi_dataset_shape(data, member_index=0)

    with pytest.raises(IndexError, match="member_index"):
        _validate_anemoi_dataset_shape(data, member_index=1)

    with pytest.raises(ValueError, match="Expected Anemoi dataset shape"):
        _validate_anemoi_dataset_shape(np.zeros((2, 3, 4), dtype=np.float32), member_index=0)
