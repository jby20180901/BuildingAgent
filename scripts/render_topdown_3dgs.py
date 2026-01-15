#!/usr/bin/env python
"""
Render views of a 3D Gaussian Splatting (.ply) scene using the
official PyTorch/gsplat renderer. Saves PNGs and the 4x4 camera poses.

Usage (example):
  python scripts/render_topdown_3dgs.py \
      --ply path/to/scene.ply \
      --out outputs/topdown.png \
      --pose-out outputs/topdown_pose.json \
      --width 1920 --height 1080 \
      --cam-height 8.0 --fovy 55

Each call now writes six views (±X, ±Y, ±Z). When ``--out`` (resp.
``--pose-out``) points to a file, suffixes like ``_posx`` are appended
before the extension (e.g., ``topdown_posx.png``). If it points to a
folder, files are created inside with those names.

Requirements: torch, gsplat, plyfile, torchvision, numpy (already in requirements.txt).
"""
import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from gsplat.rendering import rasterization
from torchvision.utils import save_image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

try:
    from utils.gs_utils import load_ply
except Exception as exc:  # pragma: no cover - safety net for path issues
    raise ImportError("Failed to import utils.gs_utils.load_ply; check PYTHONPATH.") from exc


def compute_scene_stats(means: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (center, extent, size_norm) from point cloud means."""
    min_xyz, _ = torch.min(means, dim=0)
    max_xyz, _ = torch.max(means, dim=0)
    center = (min_xyz + max_xyz) / 2.0
    extent = (max_xyz - min_xyz).abs()
    size_norm = torch.linalg.norm(extent)
    return center, extent, size_norm


def look_at(camera_pos: torch.Tensor, target: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """Create camera-to-world matrix with given position, look-at target, and up vector.

    Returns a camera-to-world (c2w) matrix with axes:
    - X-axis (right): points to the right of the camera
    - Y-axis (up): points upward
    - Z-axis (forward): points in the direction from camera to target
    """
    forward = torch.nn.functional.normalize(target - camera_pos, dim=-1)
    right = torch.nn.functional.normalize(torch.cross(forward, up), dim=-1)
    true_up = torch.nn.functional.normalize(torch.cross(right, forward), dim=-1)

    c2w = torch.eye(4, device=camera_pos.device, dtype=torch.float32)
    c2w[:3, 0] = right
    c2w[:3, 1] = true_up
    c2w[:3, 2] = forward
    c2w[:3, 3] = camera_pos
    return c2w


def build_intrinsics(width: int, height: int, fovy_deg: float) -> torch.Tensor:
    """Compute pinhole intrinsics from vertical FOV."""
    fovy = torch.deg2rad(torch.tensor(float(fovy_deg)))
    fy = height / (2.0 * torch.tan(fovy / 2.0))
    fx = fy
    cx, cy = width / 2.0, height / 2.0
    return torch.tensor([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=torch.float32)


def save_pose_matrix(pose: torch.Tensor, path: Path) -> None:
    """Save 4x4 pose matrix to JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    pose_list = pose.cpu().numpy().tolist()
    with path.open("w", encoding="utf-8") as f:
        json.dump({"c2w": pose_list}, f, indent=2)


def _resolve_output_path(base: Path, label: str, default_suffix: str) -> Path:
    """Derive a labeled output path from a base path."""
    if base.suffix:
        return base.with_name(f"{base.stem}_{label}{base.suffix}")

    if base.exists() and base.is_dir():
        directory = base
        return directory / f"{label}{default_suffix}"

    if base.name == "":
        return base / f"{label}{default_suffix}"

    return base.parent / f"{base.name}_{label}{default_suffix}"


def render_topdown(
    ply_path: Path,
    out_path: Path,
    pose_path: Path,
    width: int,
    height: int,
    cam_height: Optional[float] = None,
    fovy_deg: float = 55.0,
    down_axis: str = "y",
    device: str = "cuda",
) -> None:
    dev = torch.device(device if (device == "cpu" or torch.cuda.is_available()) else "cpu")

    means, scales, quats, rgbs, opacities = load_ply(str(ply_path), device=dev)
    center, extent, _ = compute_scene_stats(means)

    auto_radius = float(extent.max().item() * 1.2)
    radius = cam_height if cam_height is not None else max(auto_radius, 1.0)

    view_configs: List[Tuple[str, torch.Tensor, torch.Tensor]] = [
        ("posx", torch.tensor([radius, 0.0, 0.0], device=dev), torch.tensor([0.0, 1.0, 0.0], device=dev)),
        ("negx", torch.tensor([-radius, 0.0, 0.0], device=dev), torch.tensor([0.0, 1.0, 0.0], device=dev)),
        ("posy", torch.tensor([0.0, radius, 0.0], device=dev), torch.tensor([0.0, 0.0, 1.0], device=dev)),
        ("negy", torch.tensor([0.0, -radius, 0.0], device=dev), torch.tensor([0.0, 0.0, -1.0], device=dev)),
        ("posz", torch.tensor([0.0, 0.0, radius], device=dev), torch.tensor([0.0, 1.0, 0.0], device=dev)),
        ("negz", torch.tensor([0.0, 0.0, -radius], device=dev), torch.tensor([0.0, 1.0, 0.0], device=dev)),
    ]

    view_labels: List[str] = []
    c2w_mats: List[torch.Tensor] = []
    camera_positions = {}
    for label, offset, up in view_configs:
        camera_pos = center + offset
        c2w = look_at(camera_pos, center, up)
        view_labels.append(label)
        c2w_mats.append(c2w)
        camera_positions[label] = camera_pos

    world_to_view = torch.linalg.inv(torch.stack(c2w_mats)).to(dev)
    K = build_intrinsics(width, height, fovy_deg).to(dev)
    Ks = K.unsqueeze(0).repeat(len(view_labels), 1, 1)

    backgrounds = torch.ones(3, device=dev, dtype=torch.float32)
    try:
        outputs, _, _ = rasterization(
            means=means.float(),
            quats=quats.float(),
            scales=scales.float(),
            opacities=opacities.squeeze(-1).float(),
            colors=rgbs.float(),
            viewmats=world_to_view.float(),
            Ks=Ks.float(),
            height=height,
            width=width,
            render_mode="RGB",
            backgrounds=backgrounds.float(),
        )
    except Exception as exc:
        raise RuntimeError(f"Rasterization failed: {exc}") from exc

    for idx, label in enumerate(view_labels):
        image_path = _resolve_output_path(out_path, label, ".png")
        pose_out_path = _resolve_output_path(pose_path, label, ".json")

        image_path.parent.mkdir(parents=True, exist_ok=True)
        save_image(outputs[idx].permute(2, 0, 1), str(image_path))
        save_pose_matrix(c2w_mats[idx], pose_out_path)

        camera_pos = camera_positions[label]
        print(f"[{label}] Saved image to: {image_path}")
        print(f"[{label}] Saved camera pose to: {pose_out_path}")
        print(f"[{label}] Camera position: {camera_pos.tolist()} | look_at: {center.tolist()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render six axis-aligned 3DGS views (PNGs) and save the 4x4 camera poses."
    )
    parser.add_argument("--ply", required=True, type=Path, help="Input 3DGS .ply file")
    parser.add_argument(
        "--out",
        required=True,
        type=Path,
        help="Output PNG base path (file or directory)",
    )
    parser.add_argument(
        "--pose-out",
        required=True,
        type=Path,
        help="Output pose base path (file or directory)",
    )
    parser.add_argument("--width", type=int, default=1024, help="Render width (px)")
    parser.add_argument("--height", type=int, default=1024, help="Render height (px)")
    parser.add_argument("--cam-height", type=float, default=None, help="Camera radius from scene center")
    parser.add_argument("--fovy", type=float, default=55.0, help="Vertical FOV in degrees")
    parser.add_argument(
        "--down-axis",
        type=str,
        choices=["y", "z"],
        default="y",
        help="Retained for compatibility; axis selection is now handled automatically.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (cuda or cpu). Falls back to cpu if CUDA unavailable.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.ply.exists():
        raise FileNotFoundError(f"PLY not found: {args.ply}")

    render_topdown(
        ply_path=args.ply,
        out_path=args.out,
        pose_path=args.pose_out,
        width=args.width,
        height=args.height,
        cam_height=args.cam_height,
        fovy_deg=args.fovy,
        down_axis=args.down_axis,
        device=args.device,
    )


if __name__ == "__main__":
    main()

