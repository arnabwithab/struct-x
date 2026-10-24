"""voxel_to_model_input: ROI voxel -> (2,B,224,224), longest side 224 + square pad."""

import numpy as np
import pytest
from scipy import ndimage

from voxel import voxel_to_model_input

T224 = 224


def test_output_shape_and_pad():
    vox = np.zeros((10, 105, 1280), dtype=np.float32)
    vox[:, 45:55, 630:650] = 1.0  # block survives downsampling (isolated pixels need not)
    out, meta = voxel_to_model_input(vox)
    assert out.shape == (2, 5, T224, T224)
    assert meta["scale"] == pytest.approx(T224 / 1280)
    assert meta["pad"] == ((103, 103), (0, 0))  # 18 -> 224 symmetric
    assert out.sum() > 0 and np.all(np.isfinite(out))


def test_square_input_no_pad():
    vox = np.random.RandomState(0).rand(10, 200, 200).astype(np.float32)
    out, meta = voxel_to_model_input(vox)
    assert out.shape == (2, 5, T224, T224)
    assert meta["pad"] == ((12, 12), (12, 12))
    assert meta["in_shape"] == (200, 200)


def test_channel_split_order():
    vox = np.zeros((6, 32, 48), dtype=np.float32)
    vox[3] = 7.0  # channel 3 = bin 1, polarity + (b*2+p)
    out, _ = voxel_to_model_input(vox)
    assert out.shape == (2, 3, T224, T224)
    assert out[1, 1].sum() > 0  # [polarity, bin]
    assert out[0, 1].sum() == 0
