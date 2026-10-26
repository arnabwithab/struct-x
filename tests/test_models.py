"""Shape/grad smoke for heads, ResNet3D control, TimeSformer port (design sec 6).

Fast on CPU: small spatial sizes for grad tests; one 224-square case per model
for the binding shape contract (microB,2,B,224,224)->(microB,1000)+(microB,).
"""

import os

import pytest
import torch

from models.heads import FrequencyHead, WaveformFrequencyModel, WaveformHead
from models.resnet3d import ResNet3DEncoder
from models.timesformer import TimeSformerEncoder

T_OUT = 1000


def test_heads_shapes_and_grad():
    torch.manual_seed(0)
    wave, freq = WaveformHead(64, T_OUT), FrequencyHead(64)
    z = torch.randn(2, 64, requires_grad=True)
    q = wave(z)
    f = freq(z)
    assert q.shape == (2, T_OUT)
    assert f.shape == (2,)
    assert torch.isfinite(f).all()
    (q.sum() + f.sum()).backward()
    assert z.grad is not None and torch.isfinite(z.grad).all()


@pytest.mark.parametrize("b", [5, 6, 9])
def test_resnet3d_shapes_small(b):
    torch.manual_seed(0)
    enc = ResNet3DEncoder(latent_dim=64)
    x = torch.randn(1, 2, b, 64, 64)
    z = enc(x)
    assert z.shape == (1, 64) and torch.isfinite(z).all()


def test_resnet3d_binding_shape_224():
    torch.manual_seed(0)
    enc = ResNet3DEncoder(latent_dim=64)
    wave, freq = WaveformHead(64, T_OUT), FrequencyHead(64)
    x = torch.randn(1, 2, 5, 224, 224)
    z = enc(x)
    assert wave(z).shape == (1, T_OUT)
    assert freq(z).shape == (1,)
    (wave(z).sum() + freq(z).sum()).backward()


@pytest.mark.parametrize("b", [5, 9])
def test_timesformer_shapes_small(b):
    torch.manual_seed(0)
    enc = TimeSformerEncoder(img_size=32, patch_size=16, num_frames=b, dim=48, depth=2, heads=2)
    x = torch.randn(1, 2, b, 32, 32)
    z = enc(x)
    assert z.shape == (1, 48) and torch.isfinite(z).all()


def test_timesformer_binding_shape_224():
    torch.manual_seed(0)
    enc = TimeSformerEncoder(img_size=224, patch_size=32, num_frames=5, dim=48, depth=2, heads=2)
    x = torch.randn(1, 2, 5, 224, 224)
    assert enc(x).shape == (1, 48)


def _tiny_hf_state():
    from transformers import TimesformerConfig, TimesformerModel

    cfg = TimesformerConfig(
        image_size=32,
        patch_size=16,
        num_frames=8,
        num_channels=3,
        hidden_size=48,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=192,
    )
    return TimesformerModel(cfg).state_dict()


def test_hf_load_surgery_shapes():
    from models.timesformer import load_hf_timesformer

    torch.manual_seed(0)
    enc = TimeSformerEncoder(img_size=32, patch_size=16, num_frames=5, dim=48, depth=2, heads=2)
    hf = _tiny_hf_state()
    load_hf_timesformer(enc, hf, n_bins=5)
    w = enc.patch_embed.weight
    assert w.shape == (48, 2, 1, 16, 16)
    assert torch.equal(w[:, 0], w[:, 1])  # C=2 mean inflation: identical channels
    assert enc.time_pos.shape == (1, 5, 48)
    x = torch.randn(1, 2, 5, 32, 32)
    z = enc(x)
    assert z.shape == (1, 48) and torch.isfinite(z).all()


def test_hf_load_maps_attention_weights():
    from models.timesformer import load_hf_timesformer

    enc = TimeSformerEncoder(img_size=32, patch_size=16, num_frames=5, dim=48, depth=2, heads=2)
    hf = _tiny_hf_state()
    load_hf_timesformer(enc, hf, n_bins=5)
    key = "encoder.layer.1.attention.attention.qkv.weight"
    assert torch.equal(enc.blocks[1].s_attn.in_proj_weight, hf[key])
    tkey = "encoder.layer.0.temporal_attention.attention.qkv.weight"
    assert torch.equal(enc.blocks[0].t_attn.in_proj_weight, hf[tkey])


KINETICS_BIN = os.environ.get("STRUCTX_KINETICS_BIN", "")
needs_bin = pytest.mark.skipif(
    not (KINETICS_BIN and os.path.isfile(KINETICS_BIN)),
    reason="STRUCTX_KINETICS_BIN with pytorch_model.bin not present",
)


@needs_bin
def test_hf_load_real_bin():
    from models.timesformer import load_hf_timesformer

    enc = TimeSformerEncoder()  # full base: 224/16, 5 frames stub, 768/12
    state = torch.load(KINETICS_BIN, map_location="cpu", weights_only=True)
    load_hf_timesformer(enc, state, n_bins=5)
    assert enc.patch_embed.weight.shape == (768, 2, 1, 16, 16)
    assert torch.equal(enc.norm.weight, state["timesformer.layernorm.weight"])


def test_combined_model_contract():
    torch.manual_seed(0)
    model = WaveformFrequencyModel(ResNet3DEncoder(32), 32)
    x = torch.randn(1, 2, 3, 64, 64)
    q_hat, f_hat = model(x)
    assert q_hat.shape == (1, 1000) and f_hat.shape == (1,)
    assert torch.isfinite(f_hat).all() and (f_hat > 0).all()
    (q_hat.sum() + f_hat.sum()).backward()
