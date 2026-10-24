"""Real-archive decode + pairing + trigger-anchored windows (F009, design sec 5).

Real-file tests need STRUCTX_BEAM_DIR pointing at the extracted
XJTU-DV-Beam/ folder (or the two dev files); otherwise they skip.
"""

import os

import numpy as np
import pytest

from data_loader import (
    build_trial_windows,
    detect_trigger_ldv,
    load_dat_events,
    pair_trials,
    parse_ldv_txt,
    read_dat_header,
    trial_to_training_pairs,
)

BEAM_DIR = os.environ.get("STRUCTX_BEAM_DIR", "/tmp/opencode/XJTU-DV-Beam")
NEEDS_BEAM = pytest.mark.skipif(
    not os.path.isdir(BEAM_DIR), reason="STRUCTX_BEAM_DIR with extracted archive not present"
)


def write_mini_dat(path, n=200, height=32, width=48):
    header = f"% Height {height}\n% Width {width}\n% end\n".encode("latin-1")
    meta = np.array([0, 8], dtype=np.uint8)
    rng = np.random.RandomState(0)
    rec = np.empty(n, dtype=[("t", "<u4"), ("_", "<i4")])
    rec["t"] = (1000 + 10 * np.arange(n)).astype(np.uint32)
    rec["_"] = (
        rng.randint(0, width, n) | (rng.randint(0, height, n) << 14) | (rng.randint(0, 2, n) << 28)
    ).astype(np.int32)
    with open(path, "wb") as f:
        f.write(header)
        f.write(meta.tobytes())
        rec.tofile(f)


def test_pair_trials_handles_trai4_typo(tmp_path):
    dv = tmp_path / "DV"
    ldv = tmp_path / "LDV"
    dv.mkdir()
    ldv.mkdir()
    (dv / "On_set1_trai4.dat").touch()
    (ldv / "On_set1_trail4.txt").touch()
    (dv / "Off_set2_trail1.dat").touch()
    (ldv / "Off_set2_trail1.txt").touch()
    pairs = pair_trials(str(dv), str(ldv))
    ids = sorted(p[0] for p in pairs)
    assert ids == ["Off_set2_trail1", "On_set1_trail4"]
    assert pairs[0][1].endswith(".dat") and pairs[0][2].endswith(".txt")


def test_mini_dat_roundtrip(tmp_path):
    p = str(tmp_path / "mini.dat")
    write_mini_dat(p)
    h = read_dat_header(p)
    assert (h["height"], h["width"]) == (32, 48)
    assert h["n_events"] == 200
    xs, ys, ts, ps = load_dat_events(p)
    assert set(np.unique(ps).tolist()) <= {-1, 1}
    assert xs.max() < 48 and ys.max() < 32
    assert ts.min() >= 1000 and np.all(np.diff(ts) >= 0)


def test_detect_trigger_synthetic():
    t = np.arange(0, 8, 1 / 256)
    v = np.zeros_like(t)
    v[300:310] = 5.0  # hammer hit at ~1.17 s
    assert abs(detect_trigger_ldv(t, v) - t[300]) < 1e-9


def test_build_trial_windows_math():
    # offsets since trigger: usable = min(2.276, 8.0-0.6) = 2.276
    # grid 0.0 (excluded, <100ms), 0.5, 1.0 (1.5 + 1.0 > 2.276 stops)
    starts = build_trial_windows(trigger_s=0.6, event_span_s=2.276, ldv_end_s=8.0)
    assert starts == pytest.approx([0.5, 1.0])
    assert 0.6 + starts[0] >= 0.7 - 1e-9  # trigger + 100 ms exclusion in absolute time


@NEEDS_BEAM
def test_parse_ldv_txt_real():
    t, v = parse_ldv_txt(os.path.join(BEAM_DIR, "LDV/On_set1_trail1.txt"))
    assert t.shape == (2048,) and v.shape == (2048,)
    assert abs((t[1] - t[0]) - 1 / 256) < 1e-9
    assert abs(v).max() > 0.1  # m/s scale velocity


@NEEDS_BEAM
def test_read_dat_header_real():
    h = read_dat_header(os.path.join(BEAM_DIR, "DV/On_set1_trail2.dat"))
    assert (h["height"], h["width"]) == (720, 1280)  # from header, never hardcoded
    assert h["n_events"] > 1_000_000


@NEEDS_BEAM
def test_detect_trigger_real():
    t, v = parse_ldv_txt(os.path.join(BEAM_DIR, "LDV/On_set1_trail2.txt"))
    trig = detect_trigger_ldv(t, v)
    assert 0.3 <= trig <= 1.5  # authors use the 0.6 s region


@NEEDS_BEAM
def test_trial_pairs_real_end_to_end():
    pairs = trial_to_training_pairs(
        os.path.join(BEAM_DIR, "DV/On_set1_trail2.dat"),
        os.path.join(BEAM_DIR, "LDV/On_set1_trail2.txt"),
        n_bins=5,
    )
    assert len(pairs) >= 2
    vox, q = pairs[0]
    assert vox.shape[0] == 10 and vox.shape[1] > 0 and vox.shape[2] > 0
    assert q.shape == (1000,)
    assert np.all(np.isfinite(vox)) and np.all(np.isfinite(q))
