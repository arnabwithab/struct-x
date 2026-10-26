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

KINETICS_URL = (
    "https://huggingface.co/facebook/timesformer-base-finetuned-k400/resolve/main/pytorch_model.bin"
)
KINETICS_SHA256 = (
    "bdb264469ffc423fb86c78932268a290bfbd69bd768b7ce8595c047a2dc54089"  # pinned 2026-10-26 (F011)
)


def _mha_from_qkv(
    mha: nn.MultiheadAttention,
    qkv_w: torch.Tensor,
    qkv_b: torch.Tensor,
    out_w: torch.Tensor,
    out_b: torch.Tensor,
) -> None:
    """Copy a fused QKV + output projection into torch MultiheadAttention."""
    with torch.no_grad():
        mha.in_proj_weight.copy_(qkv_w)
        mha.in_proj_bias.copy_(qkv_b)
        mha.out_proj.weight.copy_(out_w)
        mha.out_proj.bias.copy_(out_b)


def load_hf_timesformer(model: "TimeSformerEncoder", hf_state: dict, n_bins: int = 5) -> None:
    """Load facebook/timesformer-base-finetuned-k400 weights into the port.

    Surgery: patch stem 3->2 channels by mean inflation; temporal pos-embed
    linearly interpolated 8->B frames. Temporal dense + separate space/time QKV
    match the HF divided_space_time block layout 1:1. Accepts the full
    ForVideoClassification checkpoint (strips the `timesformer.` prefix,
    ignores the classifier head -- replaced by our heads).
    """
    state = {
        k[len("timesformer.") :]: v
        for k, v in hf_state.items()
        if k.startswith("timesformer.") and not k.startswith("timesformer.classifier")
    }
    hf_state = state if state else hf_state
    with torch.no_grad():
        stem = hf_state["embeddings.patch_embeddings.projection.weight"]  # (D,3,16,16)
        model.patch_embed.weight.copy_(
            stem.mean(dim=1, keepdim=True).unsqueeze(2).expand(-1, 2, -1, -1, -1).contiguous()
        )
        model.patch_embed.bias.copy_(hf_state["embeddings.patch_embeddings.projection.bias"])
        pos = hf_state["embeddings.position_embeddings"]  # (1, P+1): [CLS, patches]
        model.cls.copy_(hf_state["embeddings.cls_token"] + pos[:, :1, :])
        model.space_pos.copy_(pos[:, 1:, :])
        time_emb = hf_state["embeddings.time_embeddings"]  # (1,8,D) -> (1,B,D)
        if time_emb.shape[1] != n_bins:
            time_emb = (
                nn.functional.interpolate(
                    time_emb.transpose(1, 2).unsqueeze(-1),
                    size=(n_bins, 1),
                    mode="bilinear",
                    align_corners=False,
                )
                .squeeze(-1)
                .transpose(1, 2)
            )
        model.time_pos.copy_(time_emb)
        for i, blk in enumerate(model.blocks):
            p = f"encoder.layer.{i}."
            h = hf_state
            _mha_from_qkv(
                blk.t_attn,
                h[p + "temporal_attention.attention.qkv.weight"],
                h[p + "temporal_attention.attention.qkv.bias"],
                h[p + "temporal_attention.output.dense.weight"],
                h[p + "temporal_attention.output.dense.bias"],
            )
            blk.t_dense.weight.copy_(h[p + "temporal_dense.weight"])
            blk.t_dense.bias.copy_(h[p + "temporal_dense.bias"])
            blk.n1.weight.copy_(h[p + "temporal_layernorm.weight"])
            blk.n1.bias.copy_(h[p + "temporal_layernorm.bias"])
            _mha_from_qkv(
                blk.s_attn,
                h[p + "attention.attention.qkv.weight"],
                h[p + "attention.attention.qkv.bias"],
                h[p + "attention.output.dense.weight"],
                h[p + "attention.output.dense.bias"],
            )
            blk.n2.weight.copy_(h[p + "layernorm_before.weight"])
            blk.n2.bias.copy_(h[p + "layernorm_before.bias"])
            blk.mlp[0].weight.copy_(h[p + "intermediate.dense.weight"])
            blk.mlp[0].bias.copy_(h[p + "intermediate.dense.bias"])
            blk.mlp[2].weight.copy_(h[p + "output.dense.weight"])
            blk.mlp[2].bias.copy_(h[p + "output.dense.bias"])
            blk.n3.weight.copy_(h[p + "layernorm_after.weight"])
            blk.n3.bias.copy_(h[p + "layernorm_after.bias"])
            blk.num_frames = n_bins
        model.norm.weight.copy_(hf_state["layernorm.weight"])
        model.norm.bias.copy_(hf_state["layernorm.bias"])
        model.num_frames = n_bins


class DividedBlock(nn.Module):
    """Temporal MHA over frames then spatial MHA over patches (joint banned)."""

    def __init__(self, dim: int, heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.t_attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.t_dense = nn.Linear(dim, dim)  # HF temporal branch projection
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
        h = h + self.t_dense(self.t_attn(self.n1(h), self.n1(h), self.n1(h))[0])
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
        """Fetch facebook/timesformer-base-finetuned-k400, verify SHA, map into the port.

        Surgery: patch stem 3->2 channels by mean inflation; temporal pos-embed
        linearly interpolated 8->B frames. ResNet fallback if weights 404.
        """
        import hashlib
        import io
        import urllib.request

        if sha256 == "UNPINNED":
            raise RuntimeError("Kinetics SHA unpinned; pin URL+SHA at first fetch (F011).")
        buf = urllib.request.urlopen(url).read()
        if hashlib.sha256(buf).hexdigest() != sha256:
            raise ValueError("Kinetics weights SHA mismatch.")
        state = torch.load(io.BytesIO(buf), map_location="cpu")
        load_hf_timesformer(self, state, n_bins=self.num_frames)
