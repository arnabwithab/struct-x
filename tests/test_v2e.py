"""v2e synthetic-event validation (F018-1/F018-2).

Loader smoke on a 10-frame clip of the real beam video, stats-kernel unit
tests on hand-computed arrays, 2x gate-logic tests, and a small deterministic
v2e-emulator smoke (downscaled, fixed seed run twice).
"""

import os

import numpy as np
import pytest

from v2e_synth import (
    aggregate_vox_dir,
    event_stats,
    load_clip,
    run_emulator,
    voxel_stats,
    within_2x,
)

VIDEO = "/home/arnab/code/collegeCode/struct-x/data/XJTU-DV-Beam/Supplementary materials/Video of the acquisition process.mp4"  # read-only source, never copied
BEAM_VOX = os.path.join(os.path.dirname(__file__), "..", "beam-vox")
needs_video = pytest.mark.skipif(
    not os.path.exists(VIDEO), reason="beam video absent (fresh clone/CI)"
)


@needs_video
def test_loader_smoke_10_frames():
    frames, fps = load_clip(VIDEO, n_frames=10, max_side=160)
    assert frames.shape[0] == 10
    assert frames.ndim == 3  # (T, H, W) grayscale
    assert frames.dtype == np.uint8
    assert fps == 30.0  # actual file property, not assumed


def test_event_stats_kernel():
    # 2x2 px, 1.0 s: two + at (0,0) 50 us apart (refractory hit), one - at (1,0).
    xs = np.array([0, 0, 1])
    ys = np.array([0, 0, 0])
    ts = np.array([0, 50, 1000])  # us
    ps = np.array([1, 1, -1])
    s = event_stats(xs, ys, ts, ps, height=2, width=2, duration_s=1.0, refractory_us=100)
    assert s["mean_rate_px_s"] == 3 / 4.0
    assert s["polarity_ratio"] == 2.0
    assert s["refr_viol_frac"] == 1 / 3.0


def test_voxel_stats_kernel():
    # n=2 windows, B=2 bins, 4x4 px, window 1.0 s; 64 events, 48 + / 16 -.
    vox = np.zeros((2, 4, 4, 4), dtype=np.float32)
    vox[:, 0::2] = 1.5  # + channels: 2 ch x 16 px x 1.5 = 48/win
    vox[:, 1::2] = 0.5  # - channels: 16/win
    s = voxel_stats(vox, window_s=1.0, refractory_us=100)
    assert s["mean_rate_px_s"] == (64.0 * 2) / (2 * 16 * 1.0)
    assert s["polarity_ratio"] == 3.0
    assert s["refr_viol_frac"] == 0.0  # counts far below refractory capacity


def test_gate_logic_boundaries():
    assert within_2x(1.0, 2.0)[0] is True  # exactly 2x passes
    assert within_2x(1.0, 2.01)[0] is False
    assert within_2x(0.0, 0.0) == (True, 1.0)  # degenerate 0/0 passes as 1.0
    assert within_2x(0.0, 0.5)[0] is False


def test_gate_logic_realistic():
    # polarity ratio near unity must pass; 10x rate mismatch must fail.
    assert within_2x(0.98, 1.02)[0] is True
    assert within_2x(0.48, 4.8)[0] is False


@needs_video
def test_emulator_smoke_deterministic():
    frames, fps = load_clip(VIDEO, n_frames=5, max_side=64)
    kw = {"pos_thr": 0.3, "neg_thr": 0.3, "sigma_thres": 0.03}  # default nonzero seed
    e1 = run_emulator(frames, fps, **kw)
    e2 = run_emulator(frames, fps, **kw)
    assert e1["xs"].shape == e2["xs"].shape  # same seed -> identical output
    assert np.array_equal(e1["ts_us"], e2["ts_us"])
    assert np.array_equal(e1["ps"], e2["ps"])


def test_aggregate_vox_dir(tmp_path):
    vox = np.zeros((3, 4, 2, 2), dtype=np.float32)
    vox[:, 0::2] = 2.0  # + : 2 ch x 4 px x 2.0 = 16/win
    vox[:, 1::2] = 1.0  # - : 8/win
    np.savez(tmp_path / "a.npz", vox=vox, q=np.zeros((3, 1000), dtype=np.float32))
    np.savez(tmp_path / "b.npz", vox=vox, q=np.zeros((3, 1000), dtype=np.float32))
    s = aggregate_vox_dir(str(tmp_path), window_s=1.0)
    assert s["n_files"] == 2
    assert s["mean_rate_px_s"] == 144.0 / (6 * 4 * 1.0)
    assert s["polarity_ratio"] == 2.0
    assert s["refr_viol_frac"] == 0.0
