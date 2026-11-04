"""F014-2 attribution: rollout/IG/occlusion invariants + determinism + selection.

Layer 2 (attribution) runs inference-only on existing TimeSformer checkpoints.
The determinism test pins the core guarantee: same ckpt + window + seed ->
byte-identical rollout/IG values. The constant-readout test pins the headline
architectural fact: the ported TimeSformer readout (CLS) never mixes with patch
tokens, so waveform outputs are input-independent (IG/occlusion null expected).
"""

import numpy as np
import torch

from f014_layers2 import (
    bin_relevance,
    captured_block_attention,
    divided_rollout,
    integrated_gradients,
    occlusion_map,
    select_mixed_windows,
    spectra_diff,
)


def _tiny_encoder(frames=3, patches_side=2):
    from models.timesformer import TimeSformerEncoder

    torch.manual_seed(0)
    return TimeSformerEncoder(
        img_size=16 * patches_side,
        patch_size=16,
        num_frames=frames,
        dim=48,
        depth=2,
        heads=3,
    )


def _toy_input(n=2, frames=3, side=32, seed=1):
    rng = np.random.RandomState(seed)
    return torch.from_numpy(rng.normal(size=(n, 2, frames, side, side)).astype(np.float32))


def _rollout_once(enc, x):
    enc.eval()
    with torch.no_grad(), captured_block_attention(enc) as stores:
        enc(x)
    return divided_rollout(stores, enc.num_frames, enc.num_patches)


def test_rollout_bin_relevance_sums_to_one():
    enc = _tiny_encoder()
    x = _toy_input()
    rel = bin_relevance(_rollout_once(enc, x), enc.num_frames, enc.num_patches)
    assert rel.shape == (2, enc.num_frames)
    assert np.allclose(rel.sum(axis=1), 1.0, atol=1e-5)


def test_rollout_values_byte_identical():
    enc = _tiny_encoder()
    x = _toy_input()
    a = _rollout_once(enc, x)
    b = _rollout_once(enc, x)
    assert a.tobytes() == b.tobytes()


def test_ig_completeness_on_linear_probe():
    torch.manual_seed(0)
    probe = torch.nn.Linear(8, 1)
    fn = lambda x: probe(x).squeeze(-1)
    rng = np.random.RandomState(0)
    x = torch.from_numpy(rng.normal(size=(2, 8)).astype(np.float32))
    base = torch.zeros_like(x)
    attr = integrated_gradients(fn, x, base, steps=50)
    assert attr.shape == x.shape
    assert np.allclose(attr.sum(axis=1), (fn(x) - fn(base)).detach().numpy(), rtol=1e-4, atol=1e-5)


def test_ig_zero_baseline_is_zero():
    enc = _tiny_encoder()
    enc.eval()
    fn = lambda x: enc(x).pow(2).mean(dim=-1)
    x = _toy_input()
    attr = integrated_gradients(fn, x, x.clone(), steps=10)
    assert torch.as_tensor(attr).abs().max().item() < 1e-6


def test_ig_values_byte_identical():
    enc = _tiny_encoder()
    enc.eval()
    fn = lambda x: enc(x).sum(dim=-1)
    x = _toy_input()
    a = integrated_gradients(fn, x, None, steps=12)
    b = integrated_gradients(fn, x, None, steps=12)
    assert np.ascontiguousarray(a).tobytes() == np.ascontiguousarray(b).tobytes()


def test_occlusion_map_shape_and_determinism():
    enc = _tiny_encoder()
    enc.eval()
    fn = lambda x: enc(x).pow(2).mean(dim=-1)
    x = _toy_input()
    a = occlusion_map(fn, x, micro_batch=6)
    b = occlusion_map(fn, x, micro_batch=6)
    assert a.shape == (2, 2, enc.num_frames)
    assert a.tobytes() == b.tobytes()


def test_occlusion_zero_on_constant_model():
    class Const(torch.nn.Module):
        def forward(self, x):
            return torch.ones(x.shape[0], 4)

    x = _toy_input()
    got = occlusion_map(lambda t: Const()(t).pow(2).mean(dim=-1), x)
    assert np.abs(got).max() == 0.0


def test_timesformer_readout_is_input_dependent():
    """Pins the fixed readout: mean-pool mixes patch info, outputs vary with input."""
    from models.heads import WaveformFrequencyModel

    enc = _tiny_encoder()
    model = WaveformFrequencyModel(enc, 48)
    model.eval()
    x = _toy_input()
    with torch.no_grad():
        qh, fh = model(x.float())
    assert not torch.allclose(qh[0], qh[1], atol=1e-6)
    assert not torch.allclose(fh[0], fh[1], atol=1e-6)


def test_select_mixed_windows_best_worst_and_deterministic():
    errs = {f"t{i}": float(i) for i in range(10)}
    a = select_mixed_windows(errs, n=6, seed=0)
    b = select_mixed_windows(errs, n=6, seed=0)
    assert a == b
    assert set(a) == {"t0", "t1", "t2", "t7", "t8", "t9"}


def test_spectra_diff_zero_for_identical():
    rng = np.random.RandomState(0)
    q = rng.normal(size=1000)
    out = spectra_diff(q, q)
    assert out["l2_rel"] == 0.0
    assert out["peak_a_hz"] == out["peak_b_hz"]
