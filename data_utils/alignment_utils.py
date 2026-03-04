# -*- coding: utf-8 -*-
"""
Utility functions for point loading, sampling, and two-stage bridge alignment.

Key features:
- Fast point loader: prefer .npy (same stem) + LRU cache; fallback to .txt/.csv parsing.
- Two-stage alignment:
  1) XY-PCA rotation about Z to align the dominant direction to +X.
  2) RANSAC plane fitting on (deck) and rotation to flatten the plane to XY, preserving X as much as possible.
"""

import os
import re
import pickle
from pathlib import Path
from functools import lru_cache

import numpy as np


# =============================
# Basic geometry & sampling
# =============================

def pc_normalize(pc: np.ndarray) -> np.ndarray:
    """Normalize point cloud to zero-mean and unit sphere (returns float32)."""
    centroid = np.mean(pc, axis=0, keepdims=True)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / (float(m) + 1e-9)
    return pc.astype(np.float32)


def farthest_point_sample(point: np.ndarray, npoint: int) -> np.ndarray:
    """Farthest point sampling on XYZ (keeps all columns if present)."""
    N, D = point.shape
    if N <= 0:
        raise ValueError("Empty point cloud.")
    if npoint <= 0:
        raise ValueError("npoint must be > 0.")
    xyz = point[:, :3]
    centroids = np.zeros((npoint,), dtype=np.int32)
    distance = np.ones((N,), dtype=np.float64) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, axis=-1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = int(np.argmax(distance))
    return point[centroids]


# =============================
# Fast point loader
# =============================

@lru_cache(maxsize=8192)
def load_xyz_cached(filepath: str, auto_save_npy_on_miss: bool = False) -> np.ndarray:
    """
    Load points as (N,3) float32.
    Priority: same-stem .npy -> parse text (space or comma).
    Optionally auto-save .npy when parsing text.
    """
    npy_path = None
    low = filepath.lower()
    if low.endswith(".txt") or low.endswith(".csv"):
        npy_path = filepath[:-4] + ".npy"
    elif low.endswith(".npy"):
        npy_path = filepath

    if npy_path and os.path.isfile(npy_path):
        pts = np.load(npy_path).astype(np.float32)
        if pts.ndim == 1:
            pts = pts[None, :]
        return pts[:, :3]

    # Fallback: parse text
    try:
        pts = np.loadtxt(filepath, delimiter=" ").astype(np.float32)
        if pts.ndim == 1:
            pts = pts[None, :]
        if pts.shape[1] < 3:
            raise ValueError("Less than 3 columns with space delimiter.")
    except Exception:
        pts = np.loadtxt(filepath, delimiter=",").astype(np.float32)
        if pts.ndim == 1:
            pts = pts[None, :]
        if pts.shape[1] < 3:
            raise ValueError(f"File {filepath} does not contain at least 3 columns.")

    pts = pts[:, :3]
    if auto_save_npy_on_miss and npy_path:
        try:
            os.makedirs(os.path.dirname(npy_path), exist_ok=True)
            np.save(npy_path, pts.astype(np.float32))
        except Exception:
            pass
    return pts


# =============================
# Naming / bridge id
# =============================

def extract_bridge_id(file_path: str) -> str:
    """
    Extract a stable bridge id from filename.
    Strategy:
    - Split stem by '_' or whitespace, cut before component tokens.
    - Fallback to "Area_x_bridgey" if present.
    - Otherwise use parent folder name.
    """
    orig_stem = Path(file_path).stem
    tokens_orig = re.split(r"[_\s]+", orig_stem)
    tokens_low = [t.lower() for t in tokens_orig]

    comp_keys = ["piercap", "abutment", "girder", "grider", "deck", "handrail", "diaphragm", "pier"]
    cut_idx = None
    for i, t in enumerate(tokens_low):
        if any(t.startswith(k) for k in comp_keys):
            cut_idx = i
            break
    if cut_idx is not None and cut_idx > 0:
        return "_".join(tokens_orig[:cut_idx])

    stem_low = orig_stem.lower()
    area = re.search(r"area[\-_ ]?(\d+)", stem_low)
    bridge = re.search(r"bridge[\-_ ]?(\d+)", stem_low)
    parts = []
    if area:
        parts.append(f"Area_{area.group(1)}")
    if bridge:
        parts.append(f"bridge{bridge.group(1)}")
    if parts:
        return "_".join(parts)

    return Path(file_path).parent.name


# =============================
# Two-stage alignment utilities
# =============================

def rot_z(theta: float) -> np.ndarray:
    """Rotation matrix about Z axis."""
    c, s = float(np.cos(theta)), float(np.sin(theta))
    return np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)


def pca_first_axis_on_plane(points: np.ndarray, plane: str = "xy", eps: float = 1e-9):
    """
    Get the first PCA axis on a given plane ("xy" or "xz"), and the mean.
    Returns:
      u: unit axis (3,)
      mean: mean (3,)
    """
    P = points.copy()
    if plane == "xy":
        P[:, 2] = 0.0
    elif plane == "xz":
        P[:, 1] = 0.0
    else:
        raise ValueError("plane must be 'xy' or 'xz'")

    mean = P.mean(axis=0, keepdims=True)
    X = P - mean
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    u = Vt[0]
    u = u / (np.linalg.norm(u) + eps)

    if plane == "xy":
        u[2] = 0.0
    else:
        u[1] = 0.0

    n = np.linalg.norm(u) + eps
    return (u / n).astype(np.float32), mean.reshape(-1).astype(np.float32)


def world_to_principal(points: np.ndarray, mean: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Apply translation by mean and right-multiply by R (N,3) @ (3,3)."""
    return (points - mean.reshape(1, 3)) @ R


def fit_plane_ransac(points: np.ndarray, n_iters: int = 2000, thresh: float = 0.01, seed: int = 0):
    """
    Fit plane with RANSAC.
    Returns:
      n_refined: plane normal (3,)
      d_refined: plane offset, plane eq: n·x + d = 0
      inliers: boolean mask (N,)
    """
    rng = np.random.default_rng(seed)
    N = points.shape[0]
    best_inliers = None
    best_count = -1

    if N < 3:
        n = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        d = -float(points[:, 2].mean()) if N > 0 else 0.0
        return n, d, np.ones((N,), dtype=bool)

    for _ in range(n_iters):
        idx = rng.choice(N, size=3, replace=False)
        p1, p2, p3 = points[idx]
        v1 = p2 - p1
        v2 = p3 - p1
        n = np.cross(v1, v2)
        n_norm = np.linalg.norm(n)
        if n_norm < 1e-9:
            continue
        n = n / n_norm
        d = -float(np.dot(n, p1))
        dist = np.abs(points @ n + d)
        inliers = dist <= thresh
        cnt = int(np.count_nonzero(inliers))
        if cnt > best_count:
            best_count = cnt
            best_inliers = inliers

    if best_inliers is None:
        n = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        d = -float(points[:, 2].mean())
        return n, d, np.ones((N,), dtype=bool)

    P_in = points[best_inliers]
    c = P_in.mean(axis=0, keepdims=True)
    X = P_in - c
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    n_refined = Vt[-1]
    n_refined = n_refined / (np.linalg.norm(n_refined) + 1e-12)
    d_refined = -float(np.dot(n_refined, c.reshape(-1)))

    return n_refined.astype(np.float32), d_refined, best_inliers


def align_plane_to_XY_preserve_X_about_center(deck_pts: np.ndarray, n_plane: np.ndarray, eps: float = 1e-9):
    """
    Build a rotation matrix R2 such that:
      - The plane normal aligns to +Z.
      - The new X axis is the projection of global X onto the plane (fallback to Y).
    Rotation is applied about the center c (deck centroid).
    Returns:
      R2: (3,3)
      c: (3,)
    """
    n = n_plane / (np.linalg.norm(n_plane) + eps)
    if n[2] < 0:
        n = -n

    c = deck_pts.mean(axis=0).astype(np.float32)

    ex = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    v = ex - float(np.dot(ex, n)) * n
    if np.linalg.norm(v) < 1e-6:
        ey = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        v = ey - float(np.dot(ey, n)) * n
        if np.linalg.norm(v) < 1e-6:
            v = np.array([1.0, 0.0, 0.0], dtype=np.float32)

    x_deck = v / (np.linalg.norm(v) + eps)
    y_deck = np.cross(n, x_deck)
    y_deck /= (np.linalg.norm(y_deck) + eps)

    # Prefer positive Y direction
    if y_deck[1] < 0:
        y_deck = -y_deck
        x_deck = -x_deck

    S = np.stack([x_deck, y_deck, n], axis=1).astype(np.float32)
    R2 = S.copy()

    # Ensure right-handed
    if np.linalg.det(R2) < 0:
        y_deck = -y_deck
        S = np.stack([x_deck, y_deck, n], axis=1).astype(np.float32)
        R2 = S.copy()

    return R2, c


def area_and_zrange_xy(pts: np.ndarray, q_low: float = 0.05, q_high: float = 0.95):
    """
    Compute robust XY area (quantile box) and robust Z range (quantile) for heuristics (deck selection).
    """
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    x_lo, x_hi = np.quantile(x, q_low), np.quantile(x, q_high)
    y_lo, y_hi = np.quantile(y, q_low), np.quantile(y, q_high)
    z_lo, z_hi = np.quantile(z, q_low), np.quantile(z, q_high)
    area_xy = max(0.0, x_hi - x_lo) * max(0.0, y_hi - y_lo)
    z_range = max(0.0, z_hi - z_lo)
    return area_xy, z_range


# =============================
# Simple writers
# =============================

def write_ply_xyz(path: str, pts: np.ndarray):
    """Write xyz to ASCII PLY."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {pts.shape[0]}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("end_header\n")
        for p in pts:
            f.write(f"{float(p[0])} {float(p[1])} {float(p[2])}\n")


# =============================
# Cache IO helpers
# =============================

def load_pickle(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def save_pickle_atomic(path: str, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)
