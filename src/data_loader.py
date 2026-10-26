"""Trial inventory, LDV/event loading, windowing (trial-level, no leakage). See design sec 5."""

import os
import re

import numpy as np
from scipy import signal

FS_TARGET = 1000  # Hz resample for every LDV target
WINDOW_S = 1.0  # window length
STRIDE_S = 0.5  # window stride
EXCLUSION_S = 0.1  # drop windows starting <100 ms after hammer trigger
LP_CUTOFF_HZ = 500.0  # 4th-order Butterworth low-pass at resample Nyquist
LP_ORDER = 4
EPS = 1e-6  # frozen z-score guard
LDV_SKIPROWS = 5  # author .txt header lines (see Code/H-beam_LDV.py)
DAT_PACK_X_MASK = 16383  # Prophesee Event2D bit packing (Code/utils/io/dat_events_tools.py)
DAT_PACK_Y_MASK = 268419072
DAT_PACK_P_MASK = 268435456
TRIAL_ALIAS = {"trai4": "trail4"}  # On_set1_trai4.dat typo in the archive


def is_velocity_unit(unit):
    """True when an LDV header unit (e.g. 'm/s', 'mm/s') means velocity."""
    return "/s" in str(unit)


def velocity_to_displacement(q_vel, dt_s):
    """Integrate velocity once (cumulative trapezoid) then linear-detrend.

    Binding unit convention (design sec 5): if the header shows velocity,
    integrate BEFORE anything else.
    """
    q_vel = np.asarray(q_vel, dtype=np.float64)
    q_disp = np.zeros_like(q_vel)
    q_disp[1:] = np.cumsum((q_vel[1:] + q_vel[:-1]) / 2.0) * dt_s
    return signal.detrend(q_disp, type="linear")


def preprocess_ldv(t_raw_s, q_raw, unit="m", fs_target=FS_TARGET):
    """Per-trial LDV pipeline: integrate-if-velocity -> detrend -> low-pass -> resample.

    Returns (t_out_s, q_out) at fs_target Hz. Z-scoring is separate so frozen
    train stats can be applied (see compute_train_stats / apply_frozen_zscore).
    """
    t_raw_s = np.asarray(t_raw_s, dtype=np.float64)
    q_raw = np.asarray(q_raw, dtype=np.float64)
    dt_raw = float(np.median(np.diff(t_raw_s)))
    fs_raw = 1.0 / dt_raw
    q = velocity_to_displacement(q_raw, dt_raw) if is_velocity_unit(unit) else q_raw
    q = signal.detrend(q, type="linear")
    nyq = fs_raw / 2.0
    cutoff = min(LP_CUTOFF_HZ, 0.9 * nyq)
    sos = signal.butter(LP_ORDER, cutoff, btype="low", fs=fs_raw, output="sos")
    q = signal.sosfiltfilt(sos, q)
    n_out = int(round((t_raw_s[-1] - t_raw_s[0]) * fs_target))
    q_out = signal.resample(q, n_out)
    t_out = t_raw_s[0] + np.arange(n_out) / fs_target
    return t_out, q_out


def compute_train_stats(signals):
    """Mean/std over concatenated TRAIN trials only; frozen for val/test."""
    all_q = np.concatenate([np.asarray(s, dtype=np.float64).ravel() for s in signals])
    return float(all_q.mean()), float(all_q.std())


def apply_frozen_zscore(q, mean, std, eps=EPS):
    """Per-trial z-score with frozen train stats (never per-window stats)."""
    return (np.asarray(q, dtype=np.float64) - mean) / (std + eps)


def window_starts(duration_s, window_s=WINDOW_S, stride_s=STRIDE_S, exclusion_s=EXCLUSION_S):
    """Window start times (s); excludes all starts <100 ms post-trigger."""
    starts = []
    s = 0.0
    while s + window_s <= duration_s + 1e-9:
        if s >= exclusion_s - 1e-9:
            starts.append(round(s, 9))
        s = round(s + stride_s, 9)
    return starts


def count_windows(durations_s, **kwargs):
    """Total windows over trials (design correction: ~90-270 for 30 trials)."""
    return sum(len(window_starts(d, **kwargs)) for d in durations_s)


def make_loco_splits(trial_conditions, seed=0):
    """Deterministic leave-one-condition-out over conditions found (folds=K).

    Per fold held out on condition H: all other conditions pool as matched
    (train/val-matched/test-matched); H splits 50/50 into val-shift (tuning
    only) vs test-shift (final Delta only). Trial-level only — window-level
    shuffle is FORBIDDEN.
    """
    rng = np.random.RandomState(seed)
    conditions = sorted(set(trial_conditions.values()))
    folds = {}
    for held in conditions:
        matched = sorted(t for t, c in trial_conditions.items() if c != held)
        shift = sorted(t for t, c in trial_conditions.items() if c == held)
        rng.shuffle(matched)
        rng.shuffle(shift)
        n = len(matched)
        n_test = max(1, int(round(0.2 * n))) if n >= 3 else 0
        n_val = max(1, int(round(0.2 * n))) if n >= 3 else 0
        n_val = min(n_val, n - n_test) if n >= 3 else 0
        folds[held] = {
            "train": matched[: n - n_val - n_test],
            "val_matched": matched[n - n_val - n_test : n - n_test],
            "test_matched": matched[n - n_test :],
            "val_shift": shift[: len(shift) // 2],
            "test_shift": shift[len(shift) // 2 :],
        }
    return folds


def check_no_leakage(split_to_trials):
    """Raise if any trial appears in more than one split set."""
    seen = {}
    for name, trials in split_to_trials.items():
        for t in trials:
            if t in seen:
                raise AssertionError(f"trial {t} in both {seen[t]} and {name}")
            seen[t] = name
    return True


def assign_windows_to_split(windows, trial_to_split):
    """Map (trial_id, start_s) windows to split sets via their trial id."""
    by_set = {}
    for trial, start in windows:
        by_set.setdefault(trial_to_split[trial], []).append((trial, start))
    return by_set


def pick_mck_fit_trial(train_trial_ids):
    """The ONE trial for m,c,k fit: deterministic pick from TRAIN only."""
    trials = sorted(train_trial_ids)
    if not trials:
        raise ValueError("no train trials to pick m,c,k fit trial from")
    return trials[0]


def parse_ldv_txt(path):
    """Author .txt: 5-line header, cols Time [s] + Vib Velocity [m/s] at 256 Hz."""
    data = np.loadtxt(path, skiprows=LDV_SKIPROWS)
    return np.asarray(data[:, 0], dtype=np.float64), np.asarray(data[:, 1], dtype=np.float64)


def read_dat_header(path):
    """Prophesee .dat header: Height/Width from file (never hardcoded)."""
    with open(path, "rb") as f:
        size = [None, None]
        while True:
            pos = f.tell()
            line = f.readline().decode("latin-1")
            if not line.startswith("% "):
                f.seek(pos)
                break
            parts = line.split()
            if len(parts) > 2 and parts[1] == "Height":
                size[0] = int(parts[2])
            if len(parts) > 2 and parts[1] == "Width":
                size[1] = int(parts[2])
        ev_type = int(np.frombuffer(f.read(1), dtype=np.uint8)[0])
        ev_size = int(np.frombuffer(f.read(1), dtype=np.uint8)[0])
        bod = f.tell()
        f.seek(0, os.SEEK_END)
        n_events = (f.tell() - bod) // ev_size
    if size[0] is None or size[1] is None:
        raise ValueError(f"no sensor size in {path} header")
    return {"height": size[0], "width": size[1], "ev_type": ev_type, "n_events": n_events}


def load_dat_events(path, max_events=None):
    """Full event stream: (xs, ys, ts_us, ps) with p mapped {0,1} -> {-1,+1}.

    Chunked read keeps big trials (~200 MB) out of trouble.
    """
    header = read_dat_header(path)
    xs_list, ys_list, ts_list, ps_list = [], [], [], []
    with open(path, "rb") as f:
        while True:  # skip header lines
            pos = f.tell()
            if not f.readline().decode("latin-1").startswith("% "):
                f.seek(pos)
                break
        f.read(2)  # ev_type + ev_size bytes
        remaining = (
            header["n_events"] if max_events is None else min(max_events, header["n_events"])
        )
        while remaining > 0:
            chunk = np.fromfile(
                f, dtype=[("t", "u4"), ("_", "i4")], count=min(1_000_000, remaining)
            )
            if len(chunk) == 0:
                break
            remaining -= len(chunk)
            xs_list.append((chunk["_"] & DAT_PACK_X_MASK).astype(np.int64))
            ys_list.append(((chunk["_"] & DAT_PACK_Y_MASK) >> 14).astype(np.int64))
            ts_list.append(chunk["t"].astype(np.int64))
            ps_list.append(np.where(((chunk["_"] & DAT_PACK_P_MASK) >> 28) > 0, 1, -1))
    return (
        np.concatenate(xs_list),
        np.concatenate(ys_list),
        np.concatenate(ts_list),
        np.concatenate(ps_list).astype(np.int8),
    )


def pair_trials(dv_dir, ldv_dir):
    """Pair DV .dat with LDV .txt by trial id; applies the trai4 typo alias.

    Returns [(trial_id, dv_path, ldv_path)] sorted; condition = On/Off prefix.
    """

    def stem(name):
        s = os.path.splitext(name)[0]
        for wrong, right in TRIAL_ALIAS.items():
            s = s.replace(wrong, right)
        return s

    ldv_by_id = {}
    for name in os.listdir(ldv_dir):
        if name.endswith(".txt"):
            ldv_by_id[stem(name)] = os.path.join(ldv_dir, name)
    pairs = []
    for name in sorted(os.listdir(dv_dir)):
        if not name.endswith(".dat"):
            continue
        trial_id = stem(name)
        if trial_id not in ldv_by_id:
            raise ValueError(f"no LDV match for DV trial {trial_id}")
        pairs.append((trial_id, os.path.join(dv_dir, name), ldv_by_id[trial_id]))
    return pairs


def condition_of(trial_id):
    """On/Off illumination prefix x set number, e.g. On_set1 -> On_set1."""
    m = re.match(r"(On|Off)_set(\d+)", trial_id)
    if not m:
        raise ValueError(f"unexpected trial id {trial_id}")
    return f"{m.group(1)}_set{m.group(2)}"


def detect_trigger_ldv(t_s, v):
    """Hammer trigger: time of max |velocity| (authors' valid segment starts ~0.6 s)."""
    t_s = np.asarray(t_s, dtype=np.float64)
    v = np.abs(np.asarray(v, dtype=np.float64))
    return float(t_s[int(np.argmax(v))])


def detect_trigger_events(ts_us, bin_us=10_000):
    """Event burst onset: start of the 10 ms bin with max event rate."""
    ts_us = np.asarray(ts_us, dtype=np.int64)
    lo, hi = int(ts_us.min()), int(ts_us.max())
    hist, edges = np.histogram(ts_us, bins=max(1, (hi - lo) // bin_us))
    return float(edges[int(np.argmax(hist))])


def build_trial_windows(
    trigger_s,
    event_span_s,
    ldv_end_s,
    window_s=WINDOW_S,
    stride_s=STRIDE_S,
    exclusion_s=EXCLUSION_S,
):
    """Window start offsets (s since trigger) inside BOTH streams' overlap."""
    usable = min(event_span_s, ldv_end_s - trigger_s)
    return [s for s in window_starts(usable, window_s, stride_s, exclusion_s)]


def trial_to_training_pairs(
    dv_path, ldv_path, n_bins=5, ldv_stats=None, vox_stats=None, standardize_q=True
):
    """One trial -> [(voxel (2B,H,W) ROI-cropped, q (1000,))], trigger-anchored.

    Stats args are (mean, std); when None, per-trial stats are used — dev-only,
    training MUST pass frozen train stats. With standardize_q=False the raw
    resampled q is returned (unscaled) for fold-level frozen standardisation.
    """
    from voxel import (
        events_to_voxel,
        find_roi,
        hot_pixel_mask,
        refractory_filter,
    )

    header = read_dat_header(dv_path)
    xs, ys, ts, ps = load_dat_events(dv_path)
    keep = hot_pixel_mask(xs, ys, ts, header["height"], header["width"])
    xs, ys, ts, ps = xs[keep], ys[keep], ts[keep], ps[keep]
    keep = refractory_filter(xs, ys, ts, ps)
    xs, ys, ts, ps = xs[keep], ys[keep], ts[keep], ps[keep]
    counts_y = np.bincount(ys, minlength=header["height"])
    y0, y1 = find_roi(counts_y)
    in_roi = (ys >= y0) & (ys < y1)
    xs, ys, ts, ps = xs[in_roi], ys[in_roi] - y0, ts[in_roi], ps[in_roi]
    roi_h = y1 - y0
    te_trig = detect_trigger_events(ts)

    t_raw, v_raw = parse_ldv_txt(ldv_path)
    t_trig = detect_trigger_ldv(t_raw, v_raw)
    _, q_1k = preprocess_ldv(t_raw, v_raw, unit="m/s")
    t0 = t_raw[0]
    if ldv_stats is None:
        ldv_stats = (float(q_1k.mean()), float(q_1k.std()))
    if standardize_q:
        q_1k = apply_frozen_zscore(q_1k, *ldv_stats)

    span_s = (float(ts.max()) - te_trig) / 1e6
    pairs = []
    for s in build_trial_windows(0.0, span_s, float(t_raw[-1]) - t_trig):
        t0w, t1w = int(te_trig + s * 1e6), int(te_trig + (s + WINDOW_S) * 1e6)
        in_win = (ts >= t0w) & (ts < t1w)
        vox = events_to_voxel(
            xs[in_win],
            ys[in_win],
            ts[in_win],
            ps[in_win],
            roi_h,
            header["width"],
            n_bins,
            t0_us=t0w,
            t1_us=t1w,
        )
        i0 = int(round((t_trig + s - t0) * FS_TARGET))
        q = q_1k[i0 : i0 + int(WINDOW_S * FS_TARGET)]
        if len(q) != int(WINDOW_S * FS_TARGET):
            continue
        pairs.append((vox, q))
    return pairs
