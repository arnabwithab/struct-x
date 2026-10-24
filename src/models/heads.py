"""Waveform head (T=1000 @1kHz) + frequency head. See design sec 6-7."""

import torch
import torch.nn as nn


class WaveformHead(nn.Module):
    """Linear readout z -> q_hat(t) in R^T."""

    def __init__(self, latent_dim: int, t_out: int = 1000):
        super().__init__()
        self.t_out = t_out
        self.proj = nn.Linear(latent_dim, t_out)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.proj(z)


class FrequencyHead(nn.Module):
    """MLP readout z -> f_hat_n (Hz); softplus keeps it positive."""

    def __init__(self, latent_dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(latent_dim, hidden), nn.GELU(), nn.Linear(hidden, 1))
        self.softplus = nn.Softplus()

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.softplus(self.net(z)).squeeze(-1)


class WaveformFrequencyModel(nn.Module):
    """Encoder + both heads: (N,2,B,224,224) -> (q_hat (N,T), f_hat (N,))."""

    def __init__(self, encoder: nn.Module, latent_dim: int, t_out: int = 1000):
        super().__init__()
        self.encoder = encoder
        self.wave = WaveformHead(latent_dim, t_out)
        self.freq = FrequencyHead(latent_dim)

    def forward(self, x: torch.Tensor):
        z = self.encoder(x)
        return self.wave(z), self.freq(z)
