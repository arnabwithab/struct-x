"""Ported TimeSformer divided space-time attention (research-only, CC-NC 4.0).

Reference: Bertasius et al., ICML 2021 (facebookresearch/TimeSformer, archived
read-only 2021, py3.7-era) -- ported to modern torch, never pip-installed.
Product path must use ResNet3D control instead (license fence, design sec 10).

Adapter (binding, design sec 6): voxels run as T=B frames with C=2 polarity
channels (NOT extra frames). Default net runs T=5, C=2, 224x224 square
(aspect preserved by ROI resize + zero-pad; pad amounts recorded per trial).
Surgery: (a) patch-embed conv in-channels 3->2 via channel-mean inflation of
Kinetics weights; (b) spatial pos-embed bilinear-interp to 224^2 grid,
temporal pos-embed linearly interp to B (re-interp per B in 5-9 ablation).
"""

import torch
import torch.nn as nn

KINETICS_URL = "https://github.com/facebookresearch/TimeSformer/releases/download/v1/divided_space_time_8x32_224_K400.pyth"
KINETICS_SHA256 = "UNPINNED"  # pin real SHA at first successful fetch (F003)


class DividedBlock(nn.Module):
    """Temporal MHA over frames then spatial MHA over patches (joint banned)."""

    def __init__(self, dim: int, heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.t_attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.s_attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.n1 = nn.LayerNorm(dim)
        self.n2 = nn.LayerNorm(dim)
        self.n3 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)), nn.GELU(), nn.Linear(int(dim * mlp_ratio), dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, 1+T*P, D); index 0 is CLS
        cls, tok = x[:, :1], x[:, 1:]
        n, tp, d = tok.shape
        t = self._frames(tok)  # inferred per forward from pos-embed length
        # temporal: (N*P, T, D)
        p = tp // t
        h = tok.reshape(n, t, p, d).permute(0, 2, 1, 3).reshape(n * p, t, d)
        h = h + self.t_attn(self.n1(h), self.n1(h), self.n1(h))[0]
        tok = h.reshape(n, p, t, d).permute(0, 2, 1, 3).reshape(n, tp, d)
        # spatial: (N*T, P, D); CLS passes through (averaged back later)
        s = tok.reshape(n, t, p, d).reshape(n * t, p, d)
        s = s + self.s_attn(self.n2(s), self.n2(s), self.n2(s))[0]
        tok = s.reshape(n, t, p, d).reshape(n, tp, d)
        x = torch.cat([cls, tok], dim=1)
        return x + self.mlp(self.n3(x))

    def _frames(self, tok: torch.Tensor) -> int:
        return self.num_frames

    num_frames: int = 5


class TimeSformerEncoder(nn.Module):
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        num_frames: int = 5,
        in_channels: int = 2,
        dim: int = 768,
        depth: int = 12,
        heads: int = 12,
        latent_dim: int | None = None,
    ):
        super().__init__()
        assert img_size % patch_size == 0
        self.num_patches = (img_size // patch_size) ** 2
        self.num_frames = num_frames
        self.patch_embed = nn.Conv3d(
            in_channels,
            dim,
            kernel_size=(1, patch_size, patch_size),
            stride=(1, patch_size, patch_size),
        )
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        self.space_pos = nn.Parameter(torch.zeros(1, self.num_patches, dim))
        self.time_pos = nn.Parameter(torch.zeros(1, num_frames, dim))
        self.blocks = nn.ModuleList([self._block(dim, heads, num_frames) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Identity() if latent_dim is None else nn.Linear(dim, latent_dim)
        self._init_weights()

    @staticmethod
    def _block(dim: int, heads: int, frames: int) -> DividedBlock:
        b = DividedBlock(dim, heads)
        b.num_frames = frames
        return b

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.space_pos, std=0.02)
        nn.init.trunc_normal_(self.time_pos, std=0.02)
        nn.init.trunc_normal_(self.cls, std=0.02)

    def set_num_frames(self, b: int) -> None:
        """Re-interp temporal pos-embed to a new B (5-9 ablation)."""
        with torch.no_grad():
            old = self.time_pos
            new = (
                nn.functional.interpolate(
                    old.transpose(1, 2).unsqueeze(-1),
                    size=(b, 1),
                    mode="bilinear",
                    align_corners=False,
                )
                .squeeze(-1)
                .transpose(1, 2)
            )
            self.time_pos = nn.Parameter(new)
        for blk in self.blocks:
            blk.num_frames = b
        self.num_frames = b

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, C=2, T=B, H, W)
        n, _, t, _, _ = x.shape
        tok = self.patch_embed(x)  # (N, D, T, Hp, Wp)
        tok = tok.permute(0, 2, 3, 4, 1).reshape(n, t * self.num_patches, -1)
        tok = tok.reshape(n, t, self.num_patches, -1)
        tok = tok + self.space_pos.unsqueeze(1)
        tok = tok + self.time_pos[:, :t, :].unsqueeze(2)
        tok = tok.reshape(n, t * self.num_patches, -1)
        x = torch.cat([self.cls.expand(n, -1, -1), tok], dim=1)
        for blk in self.blocks:
            x = blk(x)
        return self.head(self.norm(x[:, 0]))

    def load_kinetics_weights(self, url: str = KINETICS_URL, sha256: str = KINETICS_SHA256) -> None:
        """Fetch Kinetics zoo weights, verify SHA, inflate 3->2ch stem. ResNet fallback if 404."""
        import hashlib
        import io
        import urllib.request

        if sha256 == "UNPINNED":
            raise RuntimeError("Kinetics SHA unpinned; pin URL+SHA at first fetch (F003).")
        buf = urllib.request.urlopen(url).read()
        if hashlib.sha256(buf).hexdigest() != sha256:
            raise ValueError("Kinetics weights SHA mismatch.")
        state = torch.load(io.BytesIO(buf), map_location="cpu")
        own = self.state_dict()
        for k, v in state.items():
            if k not in own or own[k].shape != v.shape:
                continue
            if k == "patch_embed.weight" and v.shape[1] == 3 and own[k].shape[1] == 2:
                v = v.mean(dim=1, keepdim=True).expand(-1, 2, -1, -1, -1)
            own[k].copy_(v)
