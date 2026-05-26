"""Geometric reprojection: lidar -> target-view depth -> backward-warp t0/t1 into target.

All format-sensitive bits (distortion model, axis conventions) are isolated and
marked with `# VERIFY on real sample` so they can be checked against one real
meta.json + lidar.npz before trusting numbers.

Conventions (from B_Synthesis.md):
  - Camera axes (OpenCV): x->right, y->down, z->forward.
  - poses_c2w: 4x4 camera-to-world.
  - lidar xyz: world frame (same as poses).
"""
from __future__ import annotations

import cv2
import numpy as np


def intrinsics_to_K(intr: dict) -> np.ndarray:
    return np.array(
        [
            [intr["fx"], 0.0, intr["cx"]],
            [0.0, intr["fy"], intr["cy"]],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def _dist_coeffs(intr: dict) -> np.ndarray:
    coeffs = intr.get("distortion_coeffs") or []
    return np.asarray(coeffs, dtype=np.float64).reshape(-1)


def _is_fisheye(intr: dict) -> bool:
    model = str(intr.get("distortion_model", "")).lower()
    return "fisheye" in model or "kannala" in model or "kb" in model


def world_to_cam(Xw: np.ndarray, c2w: np.ndarray) -> np.ndarray:
    """World points (N,3) -> camera coords (N,3). c2w maps cam->world."""
    R = c2w[:3, :3]
    t = c2w[:3, 3]
    return (Xw - t[None, :]) @ R  # == (R^T @ (Xw-t)^T)^T


def cam_to_pixel(Xc: np.ndarray, intr: dict):
    """Camera points (N,3) -> pixel (N,2), with distortion. Returns uv, z, valid_front."""
    K = intrinsics_to_K(intr)
    dist = _dist_coeffs(intr)
    z = Xc[:, 2]
    front = z > 1e-6

    # cv2.projectPoints expects object points relative to camera with rvec=tvec=0.
    rvec = np.zeros(3)
    tvec = np.zeros(3)
    if _is_fisheye(intr):  # VERIFY on real sample
        uv, _ = cv2.fisheye.projectPoints(
            Xc.reshape(-1, 1, 3), rvec, tvec, K, dist[:4] if dist.size >= 4 else np.zeros(4)
        )
    else:  # radial-tangential (pinhole). VERIFY on real sample
        uv, _ = cv2.projectPoints(Xc, rvec, tvec, K, dist if dist.size else None)
    uv = uv.reshape(-1, 2)
    return uv, z, front


def pixels_to_rays(uv: np.ndarray, intr: dict) -> np.ndarray:
    """Pixel coords (N,2) -> unit ray directions in camera frame (N,3), undistorted."""
    K = intrinsics_to_K(intr)
    dist = _dist_coeffs(intr)
    pts = uv.reshape(-1, 1, 2).astype(np.float64)
    if _is_fisheye(intr):  # VERIFY on real sample
        norm = cv2.fisheye.undistortPoints(pts, K, dist[:4] if dist.size >= 4 else np.zeros(4))
    else:
        norm = cv2.undistortPoints(pts, K, dist if dist.size else None)
    norm = norm.reshape(-1, 2)
    rays = np.concatenate([norm, np.ones((norm.shape[0], 1))], axis=1)
    rays /= np.linalg.norm(rays, axis=1, keepdims=True)
    return rays


def build_target_depth(
    lidar_xyz: np.ndarray,
    c2w_target: np.ndarray,
    intr: dict,
    W: int,
    H: int,
    splat: int = 1,
):
    """Project lidar into the target camera, z-buffer to a dense-ish depth map.

    Returns (depth HxW float32, valid_mask HxW bool) where valid_mask marks pixels
    that received a real lidar return (before hole-filling).
    """
    Xc = world_to_cam(lidar_xyz.astype(np.float64), c2w_target)
    uv, z, front = cam_to_pixel(Xc, intr)
    u = np.round(uv[:, 0]).astype(np.int64)
    v = np.round(uv[:, 1]).astype(np.int64)
    inside = front & (u >= 0) & (u < W) & (v >= 0) & (v < H)
    u, v, zc = u[inside], v[inside], Xc[inside, 2]

    depth = np.full((H, W), np.inf, dtype=np.float32)
    # z-buffer: keep nearest depth per pixel (sort far->near, last write wins)
    order = np.argsort(-zc)
    depth[v[order], u[order]] = zc[order].astype(np.float32)

    if splat > 1:  # tiny splat to reduce holes; nearest depth dominates via min
        kernel = np.ones((splat, splat), np.uint8)
        finite = np.isfinite(depth)
        d_fill = np.where(finite, depth, 1e9).astype(np.float32)
        d_fill = cv2.erode(d_fill, kernel)  # min over neighborhood
        grew = (d_fill < 1e8) & (~finite)
        depth[grew] = d_fill[grew]

    valid = np.isfinite(depth)
    return depth, valid


def fill_depth_holes(depth: np.ndarray, valid: np.ndarray):
    """Nearest-valid fill for pixels without lidar depth (e.g. sky, lidar shadows).

    Returns (depth_filled, hole_mask) where hole_mask==True marks originally-empty
    pixels (so the refiner knows where geometry is unreliable).
    """
    from scipy import ndimage

    hole = ~valid
    if hole.all():
        return np.full_like(depth, 50.0), hole  # no lidar at all -> arbitrary far plane
    idx = ndimage.distance_transform_edt(hole, return_distances=False, return_indices=True)
    filled = depth[tuple(idx)]
    return filled.astype(np.float32), hole


def backward_warp(
    src_img: np.ndarray,
    src_c2w: np.ndarray,
    src_intr: dict,
    target_depth: np.ndarray,
    target_c2w: np.ndarray,
    target_intr: dict,
):
    """Sample src_img at the target-view pixels via target depth.

    For each target pixel: backproject by depth -> world point -> project into src
    camera -> bilinear sample. Returns (warped HxWx3 float[0,1], mask HxW bool).
    """
    H, W = target_depth.shape
    vs, us = np.mgrid[0:H, 0:W]
    uv_t = np.stack([us.ravel(), vs.ravel()], axis=1).astype(np.float64)

    rays = pixels_to_rays(uv_t, target_intr)  # (HW,3) unit, cam frame
    # Scale rays so that their z component equals the stored depth (z is along cam-z).
    z = target_depth.ravel().astype(np.float64)
    scale = z / np.clip(rays[:, 2], 1e-6, None)
    Xc_t = rays * scale[:, None]

    # cam(target) -> world
    Rt = target_c2w[:3, :3]
    tt = target_c2w[:3, 3]
    Xw = Xc_t @ Rt.T + tt[None, :]

    # world -> src cam -> src pixels
    Xc_s = world_to_cam(Xw, src_c2w)
    uv_s, z_s, front = cam_to_pixel(Xc_s, src_intr)

    Hs, Ws = src_img.shape[:2]
    map_x = uv_s[:, 0].reshape(H, W).astype(np.float32)
    map_y = uv_s[:, 1].reshape(H, W).astype(np.float32)

    warped = cv2.remap(
        src_img,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    ).astype(np.float32) / 255.0

    inb = (
        front.reshape(H, W)
        & (map_x >= 0)
        & (map_x < Ws)
        & (map_y >= 0)
        & (map_y < Hs)
    )
    return warped, inb
