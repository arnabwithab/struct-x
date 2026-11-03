"""v2e synthetic events + synthetic-vs-real stats (F018). See design sec 8/12.

Wraps the SensorsINI v2e EventEmulator (Hu et al. CVPRW 2021, MIT) to
synthesize threshold/illumination-shifted events from the beam video, and
compares event-rate stats against REAL beam-vox windows. CPU-only by design
(shared 4060); determinism via fixed seed.
"""

import os

import cv2
import numpy as np

REFRACTORY_US = 100  # mirrors voxel.REFRACTORY_US (drop same-pixel ev <100 us apart)
# v2e only seeds its RNGs for NONZERO seed ("fix it to nonzero value to get
# same mismatch every time"); 0 means unseeded. Keep nonzero for determinism.
DEFAULT_SEED = 42


def load_clip(path, n_frames=10, max_side=None):
    """First n_frames of video as uint8 grayscale (T, H, W) + fps.

    max_side downscales longest side (INTER_AREA); aspect preserved.
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video: {path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    frames = []
    for _ in range(n_frames):
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    cap.release()
    if not frames:
        raise ValueError(f"no frames decoded from {path}")
    clip = np.stack(frames)
    if max_side is not None and max(clip.shape[1:]) > max_side:
        s = max_side / max(clip.shape[1:])
        h, w = clip.shape[1:]
        clip = np.stack(
            [
                cv2.resize(f, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
                for f in clip
            ]
        )
    return np.ascontiguousarray(clip), fps


def apply_gain(frames, gain):
    """Illumination shift proxy: scale luma by gain, clip to uint8."""
    return np.clip(frames.astype(np.float32) * gain, 0, 255).astype(np.uint8)


def run_emulator(
    frames,
    fps,
    pos_thr=0.3,
    neg_thr=0.3,
    sigma_thres=0.03,
    refractory_us=REFRACTORY_US,
    leak_rate_hz=0.0,
    shot_noise_rate_hz=0.0,
    seed=DEFAULT_SEED,
):
    """Feed grayscale frames through v2e; return raw events + geometry.

    device=cpu (shared GPU, forward-only CPU is fast enough). First frame
    only initialises the photoreceptor state (emits nothing).
    """
    from v2ecore.emulator import EventEmulator

    emu = EventEmulator(
        pos_thres=pos_thr,
        neg_thres=neg_thr,
        sigma_thres=sigma_thres,
        refractory_period_s=refractory_us * 1e-6,
        leak_rate_hz=leak_rate_hz,
        shot_noise_rate_hz=shot_noise_rate_hz,
        seed=seed,
        device="cpu",
    )
    h, w = frames.shape[1:]
    parts = []
    for i, frame in enumerate(frames):
        ev = emu.generate_events(frame, i / fps)
        if ev is not None:
            parts.append(np.asarray(ev))
    if parts:
        all_ev = np.concatenate(parts)
        ts_us = (all_ev[:, 0] * 1e6).astype(np.int64)
        xs = all_ev[:, 1].astype(np.int64)
        ys = all_ev[:, 2].astype(np.int64)
        ps = all_ev[:, 3].astype(np.int8)
    else:
        xs = ys = ts_us = np.zeros(0, dtype=np.int64)
        ps = np.zeros(0, dtype=np.int8)
    return {
        "xs": xs,
        "ys": ys,
        "ts_us": ts_us,
        "ps": ps.astype(np.int8),
        "height": h,
        "width": w,
        "duration_s": len(frames) / fps,
        "params": {
            "pos_thr": pos_thr,
            "neg_thr": neg_thr,
            "sigma_thres": sigma_thres,
            "refractory_us": refractory_us,
            "leak_rate_hz": leak_rate_hz,
            "shot_noise_rate_hz": shot_noise_rate_hz,
            "seed": seed,
        },
    }


def _polarity_ratio(n_pos, n_neg):
    if n_neg == 0:
        return 1.0 if n_pos == 0 else float("inf")
    return n_pos / n_neg


def event_stats(xs, ys, ts, ps, height, width, duration_s, refractory_us=REFRACTORY_US):
    """Exact stats from raw events: per-pixel rate, polarity ratio, refr fraction."""
    xs, ys, ts, ps = (np.asarray(a) for a in (xs, ys, ts, ps))
    n = len(ts)
    n_pos, n_neg = int((ps > 0).sum()), int((ps < 0).sum())
    viol = 0
    if n:
        order = np.argsort(ts, kind="stable")
        last_t = {}
        for i in order:
            key = (int(xs[i]), int(ys[i]))
            if key in last_t and ts[i] - last_t[key] < refractory_us:
                viol += 1
            else:
                last_t[key] = ts[i]
    return {
        "n_events": n,
        "mean_rate_px_s": n / (height * width * duration_s),
        "polarity_ratio": _polarity_ratio(n_pos, n_neg),
        "refr_viol_frac": viol / n if n else 0.0,
    }


def voxel_stats(vox, window_s=1.0, refractory_us=REFRACTORY_US):
    """Same stats from binned (n, 2B, H, W) voxels (channels alternate -/+).

    Refractory fraction is a LOWER BOUND: bins hide sub-bin timing, so only
    counts exceeding refractory capacity floor(window/ref)+1 per pixel count.
    """
    vox = np.asarray(vox, dtype=np.float64)
    n, _, h, w = vox.shape
    total = float(vox.sum())
    pos = float(vox[:, 0::2].sum())
    neg = float(vox[:, 1::2].sum())
    cap = window_s * 1e6 / refractory_us + 1
    per_px = vox.reshape(n, -1, h * w).sum(axis=1)
    excess = float(np.maximum(0.0, per_px - cap).sum())
    return {
        "n_events": total,
        "mean_rate_px_s": total / (n * h * w * window_s),
        "polarity_ratio": _polarity_ratio(pos, neg),
        "refr_viol_frac": excess / total if total else 0.0,
    }


def within_2x(synth, real):
    """2x gate: pass iff max(s/r, r/s) <= 2. Returns (pass, synth/real)."""
    if synth == 0 and real == 0:
        return True, 1.0
    if synth == 0 or real == 0:
        return False, float("inf")
    ratio = synth / real
    return max(ratio, 1 / ratio) <= 2.0, ratio


def aggregate_vox_dir(vox_dir, window_s=1.0, refractory_us=REFRACTORY_US):
    """Global real-DVS stats over every beam-vox window (read-only glob)."""
    import glob

    total, pos, neg, px_s, excess = 0.0, 0.0, 0.0, 0.0, 0.0
    cap = window_s * 1e6 / refractory_us + 1
    files = sorted(glob.glob(os.path.join(vox_dir, "*.npz")))
    if not files:
        raise FileNotFoundError(f"no .npz windows in {vox_dir}")
    for path in files:
        vox = np.load(path)["vox"].astype(np.float64)
        n, _, h, w = vox.shape
        total += float(vox.sum())
        pos += float(vox[:, 0::2].sum())
        neg += float(vox[:, 1::2].sum())
        px_s += n * h * w * window_s
        per_px = vox.reshape(n, -1, h * w).sum(axis=1)
        excess += float(np.maximum(0.0, per_px - cap).sum())
    return {
        "n_events": total,
        "mean_rate_px_s": total / px_s,
        "polarity_ratio": _polarity_ratio(pos, neg),
        "refr_viol_frac": excess / total if total else 0.0,
        "n_files": len(files),
    }
