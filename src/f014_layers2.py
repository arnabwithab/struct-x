"""F014-2 attribution helpers: divided-attention rollout + bounded IG/occlusion.

Pure functions only (no IO, no checkpoints); the inference driver is
scripts/f014_layers2.py. Covers the ported TimeSformer divided attention
(src/models/timesformer.py): per-window rollout maps over bins (polarity is
mixed by the patch embed, so polarity comes from IG/occlusion instead).

Structural note (pinned by test_timesformer_readout_is_input_independent):
the port's CLS readout never mixes with patch tokens, so waveform outputs are
input-independent per checkpoint. Rollout therefore describes attention routing
among patches (descriptive, not causal); IG/occlusion are the causal check and
are expected ~null on these ckpts. Report both, never rollout alone.
"""

import contextlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


@contextlib.contextmanager
def captured_block_attention(encoder):
    """Yield per-block {"t": (N*P,T,T), "s": (N*T,P,P)} MultiheadAttention weights.

    Hooks only; no forward logic is changed. Caller runs the forward inside the
    context under eval/no-grad (weights are still materialised by the module).
    """
    stores, handles = [], []

    def _hook(store, name):
        def fn(_mod, _inp, out):
            store[name] = out[1].detach()

        return fn

    for blk in encoder.blocks:
        store: dict = {}
        stores.append(store)
        handles.append(blk.t_attn.register_forward_hook(_hook(store, "t")))
        handles.append(blk.s_attn.register_forward_hook(_hook(store, "s")))
    try:
        yield stores
    finally:
        for h in handles:
            h.remove()


def divided_rollout(stores, n_frames, n_patches):
    """Joint rollout over T*P patch tokens from divided (temporal, spatial) weights.

    Per block: temporal step (block-diagonal over patches, 0.5*A+0.5*I residual)
    then spatial step (block-diagonal over frames). CLS is excluded: it never
    attends in the port, so rollout over patch tokens is the full story.
    Returns (N, T*P, T*P) numpy array; rows stay stochastic (sum to 1).
    """
    n_blocks = len(stores)
    n = stores[0]["t"].shape[0] // n_patches
    n_t = stores[0]["t"].shape[1]
    t_all = torch.stack([s["t"] for s in stores]).reshape(n_blocks, n, n_patches, n_t, n_t)
    n_p = stores[0]["s"].shape[1]
    s_all = torch.stack([s["s"] for s in stores]).reshape(n_blocks, n, n_t, n_p, n_p)
    tp = n_frames * n_patches
    dev = t_all.device
    eye_t = torch.eye(t_all.shape[3], dtype=t_all.dtype, device=dev)
    eye_p = torch.eye(n_patches, dtype=s_all.dtype, device=dev)
    r = torch.eye(tp, dtype=torch.float64, device=dev).unsqueeze(0).expand(n, -1, -1).contiguous()
    t_idx = (torch.arange(t_all.shape[3]).unsqueeze(1) * n_patches + torch.arange(n_patches)).to(
        dev
    )
    s_idx = (torch.arange(s_all.shape[2]).unsqueeze(1) * n_patches + torch.arange(n_patches)).to(
        dev
    )
    for b in range(n_blocks):
        a = 0.5 * t_all[b].double() + 0.5 * eye_t.double()  # (N,P,T,T)
        r_t = r.reshape(n, t_all.shape[3], n_patches, tp).permute(0, 2, 1, 3)
        r[:, t_idx.reshape(-1), :] = (
            torch.einsum("npij,npjk->npik", a, r_t).permute(0, 2, 1, 3).reshape(n, tp, tp)
        )
        c = 0.5 * s_all[b].double() + 0.5 * eye_p.double()  # (N,T,P,P)
        r_s = r.reshape(n, s_all.shape[2], n_patches, tp)
        r[:, s_idx.reshape(-1), :] = torch.einsum("ntij,ntjk->ntik", c, r_s).reshape(n, tp, tp)
    return r.cpu().numpy()


def bin_relevance(rollout, n_frames, n_patches):
    """(N,TP,TP) rollout -> (N,T) per-bin relevance (mean over queries), sums to 1."""
    rel = np.asarray(rollout, dtype=np.float64).mean(axis=1).reshape(-1, n_frames, n_patches)
    rel = rel.mean(axis=-1)
    return rel / rel.sum(axis=1, keepdims=True)


def _as_batch(x):
    t = x if isinstance(x, torch.Tensor) else torch.from_numpy(np.ascontiguousarray(x))
    return t.float()


def integrated_gradients(scalar_fn, x, baseline=None, steps=50, alpha_batch=8):
    """Midpoint-rule IG for scalar_fn(x)->(N,) in chunks of alphas (memory-bounded).

    Deterministic (no randomness). Completeness: sum(attrs) == f(x)-f(baseline).
    """
    x = _as_batch(x)
    base = torch.zeros_like(x) if baseline is None else _as_batch(baseline)
    delta = x - base
    attr = torch.zeros_like(x, dtype=torch.float64)
    with torch.enable_grad():
        for start in range(0, steps, alpha_batch):
            alphas = ((torch.arange(start, min(start + alpha_batch, steps)) + 0.5) / steps).to(
                x.device
            )
            xa = (base.unsqueeze(0) + alphas.view(-1, *([1] * x.dim())) * delta).reshape(
                -1, *x.shape[1:]
            )
            xa = xa.float().requires_grad_(True)
            s = scalar_fn(xa)
            (g,) = torch.autograd.grad(s.sum(), xa)
            attr += g.double().reshape(len(alphas), *x.shape).sum(dim=0)
    return (attr * delta.double() / steps).cpu().numpy()


def _batched_scalar(scalar_fn, x, micro=8):
    outs = []
    with torch.no_grad():
        for i in range(0, len(x), micro):
            outs.append(np.asarray(scalar_fn(x[i : i + micro]).detach().cpu()).ravel())
    return np.concatenate(outs)


def occlusion_map(scalar_fn, x, micro_batch=8):
    """Zero each (polarity, bin) slice -> |S_occ - S_full| per window: (N,2,B)."""
    x = _as_batch(x)
    n, _, b, _, _ = x.shape
    full = _batched_scalar(scalar_fn, x, micro_batch)
    out = np.zeros((n, 2, b))
    for p in range(2):
        for q in range(b):
            occ = x.clone()
            occ[:, p, q] = 0.0
            out[:, p, q] = np.abs(_batched_scalar(scalar_fn, occ, micro_batch) - full)
    return out


def select_mixed_windows(errors, n=20, seed=0):
    """n/2 lowest + n-n/2 highest RMSE trial/window ids, shuffled with fixed seed."""
    ids = sorted(errors, key=lambda k: (float(errors[k]), k))
    lo, hi = n // 2, n - n // 2
    pick = ids[:lo] + ids[len(ids) - hi :]
    rng = np.random.RandomState(seed)
    rng.shuffle(pick)
    return list(pick)


def magnitude_spectrum(q, fs=1000.0):
    """Hann magnitude spectrum with 4x zero-pad (same convention as layer 1)."""
    q = np.asarray(q, dtype=float).ravel()
    freqs = np.fft.rfftfreq(4 * q.size, d=1.0 / fs)
    return freqs, np.abs(np.fft.rfft(q * np.hanning(q.size), n=4 * q.size))


def spectra_diff(qa, qb, fs=1000.0):
    """Variant-difference spectra: rel L2 of magnitudes + Hann FFT-peak freqs."""
    from baseline_ridge import window_fft_peak

    _, ma = magnitude_spectrum(qa, fs)
    _, mb = magnitude_spectrum(qb, fs)
    denom = float(np.linalg.norm(ma)) or 1.0
    return {
        "l2_rel": float(np.linalg.norm(ma - mb) / denom),
        "peak_a_hz": float(window_fft_peak(np.asarray(qa, dtype=float).ravel(), fs=fs)),
        "peak_b_hz": float(window_fft_peak(np.asarray(qb, dtype=float).ravel(), fs=fs)),
    }


def rollout_bar_fig(series, title):
    """Grouped mean+/-std bars over bins, one group per variant."""
    variants = list(series)
    b = len(series[variants[0]][0])
    x = np.arange(b)
    w = 0.8 / max(1, len(variants))
    fig, a = plt.subplots(figsize=(8, 3.5))
    for i, v in enumerate(variants):
        mean, std = np.asarray(series[v][0]), np.asarray(series[v][1])
        a.bar(x + (i - (len(variants) - 1) / 2) * w, mean, w, yerr=std, capsize=2, label=v)
    a.set_xticks(x)
    a.set_xticklabels([f"b{i}" for i in range(b)])
    a.set_ylabel("mean relevance")
    a.set_title(title, fontsize=10)
    a.legend(fontsize=8)
    fig.tight_layout()
    return fig


def heatmap_fig(mat, title):
    """(2, B) polarity x bin heatmap."""
    mat = np.asarray(mat, dtype=float)
    fig, a = plt.subplots(figsize=(max(4, mat.shape[1]), 2.5))
    im = a.imshow(mat, aspect="auto", origin="lower")
    a.set_yticks([0, 1])
    a.set_yticklabels(["neg", "pos"])
    a.set_xlabel("bin")
    fig.colorbar(im, ax=a, label="value")
    a.set_title(title, fontsize=10)
    fig.tight_layout()
    return fig


def spectra_overlay_fig(freqs, mag_a, mag_b, peak_a, peak_b, title):
    """Mean magnitude spectra (data_only vs both) with FFT-peak markers."""
    fig, a = plt.subplots(figsize=(8, 3.5))
    a.plot(freqs, mag_a, label="data_only", linewidth=1.0)
    a.plot(freqs, mag_b, label="both", linewidth=1.0, alpha=0.8)
    a.axvline(peak_a, linestyle="--", label="peak data_only")
    a.axvline(peak_b, linestyle=":", label="peak both")
    a.set_xlim(0, 250)
    a.set_xlabel("Hz")
    a.set_title(title, fontsize=10)
    a.legend(fontsize=8)
    fig.tight_layout()
    return fig


def fig_to_png_bytes(fig):
    """Render fig to PNG bytes (Agg) and close it."""
    import io

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100)
    plt.close(fig)
    return buf.getvalue()
