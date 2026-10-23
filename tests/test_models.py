"""Shape/grad smoke for heads, ResNet3D control, TimeSformer port (design sec 6).

Fast on CPU: small spatial sizes for grad tests; one 224-square case per model
for the binding shape contract (microB,2,B,224,224)->(microB,1000)+(microB,).
"""

import pytest
import torch

from models.heads import FrequencyHead, WaveformHead
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
