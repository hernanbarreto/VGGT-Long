"""
VGGT-Omega adapter for the VGGT-Long unified pipeline.
================================================================================
Wraps the (separately vendored, UNMODIFIED) `vggt-omega` repo so VGGT-Long can use
VGGT-Ω as a backbone exactly like VGGT/MapAnything — selected via config
(Weights.model == 'VGGTOmega'). VGGT-Ω is SOTA for camera/pose (CVPR 2026, +77%
on Sintel) and robust to DYNAMIC scenes (moving people/machinery on a worksite) —
our normal mode. It is UP-TO-SCALE (not metric); metric scale is recovered
downstream by aligning to DA3's metric depth.

VGGT-Ω predicts cameras + depth (no native pointmap), so this adapter UNPROJECTS
depth → world_points to feed VGGT-Long's chunk-Sim3 alignment, returning the same
dict shape as VGGTAdapter. The vendored omega package is imported from
../vggt-omega via sys.path (no pip install, no edits to that repo).
"""
import sys
from pathlib import Path

import numpy as np
import torch

from base_models.base_model import Base3DModel


def _omega_pkg_on_path() -> None:
    # ../vggt-omega relative to vendor/VGGT-Long → its package root holds `vggt_omega/`
    root = Path(__file__).resolve().parents[2] / "vggt-omega"
    if not (root / "vggt_omega").exists():
        raise FileNotFoundError(f"vggt-omega repo not found at {root} (clone it into vendor/)")
    p = str(root)
    if p not in sys.path:
        sys.path.insert(0, p)


def _ensure_bsd(x: torch.Tensor, n_lead: int = 2) -> torch.Tensor:
    """Ensure a leading (batch, seq) pair: if the tensor has only `seq` leading,
    add a batch dim. Used to normalise omega outputs to [B,S,...]."""
    return x


class VGGTOmegaAdapter(Base3DModel):
    def load(self):
        """Load VGGT-Omega and its checkpoint."""
        print("Loading VGGT-Omega model...")
        _omega_pkg_on_path()
        from vggt_omega.models import VGGTOmega

        self.image_resolution = int(self.config.get("Model", {}).get("omega_resolution", 512))
        self.preproc_mode = self.config.get("Model", {}).get("omega_mode", "balanced")
        self.model = VGGTOmega()                       # enable_alignment=False (geometry ckpt)
        url = self.config["Weights"]["VGGTOmega"]
        print(f"Loading weights from: {url}")
        state_dict = torch.load(url, map_location="cpu")
        if isinstance(state_dict, dict) and "model" in state_dict and "pose_enc" not in state_dict:
            state_dict = state_dict["model"]           # tolerate {'model': sd} wrappers
        self.model.load_state_dict(state_dict, strict=False)
        self.model.eval()
        self.model = self.model.to(self.device)

    def infer_chunk(self, image_paths: list) -> dict:
        from vggt_omega.utils.load_fn import load_and_preprocess_images
        from vggt_omega.utils.pose_enc import encoding_to_camera

        images = load_and_preprocess_images(
            image_paths, mode=self.preproc_mode, image_resolution=self.image_resolution
        ).to(self.device)                              # [S,3,H,W]
        if images.dim() == 5:
            images = images[0]
        S, _, H, W = images.shape

        torch.cuda.empty_cache()
        with torch.inference_mode():
            with torch.cuda.amp.autocast(dtype=self.dtype):
                pred = self.model(images)
        torch.cuda.empty_cache()

        pose_enc = pred["pose_enc"]
        if pose_enc.dim() == 2:                        # [S,D] → [1,S,D]
            pose_enc = pose_enc[None]
        extr_w2c, intr = encoding_to_camera(pose_enc, (H, W))   # [1,S,3,4] (w2c, OpenCV), [1,S,3,3]
        # → homogeneous 4x4, then invert to C2W (the pipeline's convention)
        ones = torch.tensor([0, 0, 0, 1], dtype=extr_w2c.dtype, device=extr_w2c.device)
        ones = ones.view(1, 1, 1, 4).repeat(extr_w2c.shape[0], extr_w2c.shape[1], 1, 1)
        extr_homo = torch.cat([extr_w2c, ones], dim=2)          # [1,S,4,4]
        c2w = torch.inverse(extr_homo)                          # [1,S,4,4]

        depth = pred["depth"]
        if depth.dim() == 5 and depth.shape[-1] == 1:
            depth = depth[..., 0]                               # [.,S,H,W]
        if depth.dim() == 3:                                    # [S,H,W] → [1,S,H,W]
            depth = depth[None]
        dconf = pred.get("depth_conf")
        if dconf is not None and dconf.dim() == 3:
            dconf = dconf[None]

        world_points = self._unproject(depth, intr, c2w)        # [1,S,H,W,3]

        return {
            "world_points": world_points,
            "world_points_conf": dconf if dconf is not None else torch.ones_like(depth),
            "extrinsic": c2w,                                   # C2W 4x4
            "intrinsic": intr,
            "depth": depth,
            "depth_conf": dconf,
            "images": images[None],
            "mask": None,
        }

    @staticmethod
    def _unproject(depth: torch.Tensor, K: torch.Tensor, c2w: torch.Tensor) -> torch.Tensor:
        """depth [1,S,H,W], K [1,S,3,3], c2w [1,S,4,4] → world_points [1,S,H,W,3]."""
        B, S, H, W = depth.shape
        dev, dt = depth.device, torch.float32
        vv, uu = torch.meshgrid(torch.arange(H, device=dev, dtype=dt),
                                torch.arange(W, device=dev, dtype=dt), indexing="ij")
        out = torch.empty(B, S, H, W, 3, device=dev, dtype=dt)
        for b in range(B):
            for s in range(S):
                d = depth[b, s].to(dt)
                fx, fy = K[b, s, 0, 0], K[b, s, 1, 1]
                cx, cy = K[b, s, 0, 2], K[b, s, 1, 2]
                x = (uu - cx) * d / fx
                y = (vv - cy) * d / fy
                cam = torch.stack([x, y, d], dim=-1)            # [H,W,3]
                R = c2w[b, s, :3, :3].to(dt); t = c2w[b, s, :3, 3].to(dt)
                out[b, s] = cam @ R.T + t
        return out
