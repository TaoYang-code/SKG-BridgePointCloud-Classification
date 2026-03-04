# -*- coding: utf-8 -*-
"""
Offline precomputation script to build per-bridge `align_cache.pkl`.

Usage:
  python precompute_align_cache.py --root /path/to/dataset --area 1 --num_category 7 \
      --save_processed_root /path/to/dataset/processed_points_xyplane2 --precompute

Notes:
- This script is intended to run once before training.
- You can optionally save bridge-level aligned point clouds (.ply/.npy).
"""

import os
from tqdm import tqdm
import numpy as np

from data_utils.alignment_utils import (
    load_xyz_cached,
    extract_bridge_id,
    rot_z,
    pca_first_axis_on_plane,
    world_to_principal,
    fit_plane_ransac,
    align_plane_to_XY_preserve_X_about_center,
    area_and_zrange_xy,
    write_ply_xyz,
    save_pickle_atomic,
)


def collect_all_files(root: str, area: int, num_category: int):
    """Collect all instance files for a given area (train+test)."""
    if num_category == 7:
        lst = [line.rstrip() for line in open(os.path.join(root, f"Area{area}/train.txt"))]
        lst += [line.rstrip() for line in open(os.path.join(root, f"Area{area}/test.txt"))]

    files = []
    for sid in lst:
        fl = sid.lower()
        if "cap" in fl:
            cls_name = "piercap"
        elif "abutment" in fl:
            cls_name = "abutment"
        elif "girder" in fl or "grider" in fl:
            cls_name = "girder"
        elif "deck" in fl:
            cls_name = "deck"
        elif "handrail" in fl:
            cls_name = "handrail"
        elif "diaphragm" in fl:
            cls_name = "diaphragm"
        elif "pier" in fl and "cap" not in fl:
            cls_name = "pier"
        else:
            raise ValueError(f"Bad filename token (cannot infer class): {sid}")

        stem = sid if sid.endswith(".txt") else sid + ".txt"
        fp = os.path.join(root, cls_name, stem)
        if os.path.isfile(fp):
            files.append(fp)
        else:
            print("Missing file:", fp)
    return files


def precompute_align_cache(
    root: str,
    area: int,
    num_category: int,
    save_processed_root: str,
    ransac_iter: int,
    ransac_thresh: float,
    save_processed_points: bool,
    save_processed_format: str,
    auto_save_npy_on_miss: bool,
):
    """Compute and save `align_cache.pkl` per bridge (and optional bridge-level step2 points)."""
    files = collect_all_files(root, area, num_category)
    if save_processed_root is None:
        save_processed_root = os.path.join(root, "processed_points_xyplane")

    # Group by bridge id
    by_bridge = {}
    for fp in files:
        bid = extract_bridge_id(fp)
        by_bridge.setdefault(bid, []).append(fp)

    for bridge_id, fps in tqdm(by_bridge.items(), desc="Precomputing align cache (per bridge)"):
        out_dir = os.path.join(save_processed_root, bridge_id)
        os.makedirs(out_dir, exist_ok=True)
        cache_pkl = os.path.join(out_dir, "align_cache.pkl")
        if os.path.isfile(cache_pkl):
            continue

        inst_pts_raw = []
        for filepath in fps:
            pts = load_xyz_cached(filepath, auto_save_npy_on_miss=auto_save_npy_on_miss).astype(np.float32)
            inst_pts_raw.append(pts)

        if len(inst_pts_raw) == 0:
            continue

        P_all = np.concatenate(inst_pts_raw, axis=0)

        # Step-1: XY PCA -> +X
        u_xy_all, mean1 = pca_first_axis_on_plane(P_all, plane="xy")
        theta = float(np.arctan2(u_xy_all[1], u_xy_all[0]))
        R1 = rot_z(-theta)

        P_step1 = world_to_principal(P_all, mean1, R1)
        inst_step1 = [world_to_principal(p, mean1, R1) for p in inst_pts_raw]

        # Deck selection
        stats = [area_and_zrange_xy(p, 0.05, 0.95) for p in inst_step1]
        order = np.argsort([-a for (a, z) in stats])
        if len(order) == 0:
            continue

        top2 = order[:2]
        if len(top2) == 1:
            deck_idx = int(top2[0])
            other_idx = None
        else:
            zr = [stats[top2[0]][1], stats[top2[1]][1]]
            deck_idx = int(top2[int(np.argmin(zr))])
            other_idx = int(top2[1 - int(np.argmin(zr))])

        deck_pts1 = inst_step1[deck_idx]
        other_pts1 = inst_step1[other_idx] if other_idx is not None else None

        # Step-2: flatten deck plane
        n_raw, _, inliers = fit_plane_ransac(deck_pts1, n_iters=ransac_iter, thresh=ransac_thresh, seed=0)
        deck_for_align = deck_pts1[inliers] if (inliers is not None and inliers.sum() > 3) else deck_pts1
        R2, c_deck = align_plane_to_XY_preserve_X_about_center(deck_for_align, n_raw)

        if other_pts1 is not None:
            other_step2 = (other_pts1 - c_deck.reshape(1, 3)) @ R2 + c_deck.reshape(1, 3)
            y_med = float(np.median(other_step2[:, 1]))
            if y_med < 0.0:
                Fz = np.array([[-1.0, 0.0, 0.0],
                               [0.0, -1.0, 0.0],
                               [0.0, 0.0, 1.0]], dtype=np.float32)
                R2 = R2 @ Fz

        # Optional: save bridge-level aligned points
        if save_processed_points:
            P_step2 = (P_step1 - c_deck.reshape(1, 3)) @ R2 + c_deck.reshape(1, 3)
            if save_processed_format in ("npy", "both"):
                np.save(os.path.join(out_dir, f"{bridge_id}__bridge_step2.npy"), P_step2.astype(np.float32))
            if save_processed_format in ("ply", "both"):
                write_ply_xyz(os.path.join(out_dir, f"{bridge_id}__bridge_step2.ply"), P_step2.astype(np.float32))

        # Save minimal cache
        info = dict(
            mean1=mean1.astype(np.float32),
            R1=R1.astype(np.float32),
            R2=R2.astype(np.float32),
            c_deck=c_deck.astype(np.float32),
        )
        save_pickle_atomic(cache_pkl, info)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="/home/tyan632/Desktop/classfication_github/data/mydataset")
    parser.add_argument("--area", type=int, default=1)
    parser.add_argument("--num_category", type=int, default=7)
    parser.add_argument("--save_processed_root", type=str, default="/home/tyan632/Desktop/classfication_github/data/mydataset/processed_points_xyplane")

    parser.add_argument("--ransac_iter", type=int, default=2000)
    parser.add_argument("--ransac_thresh", type=float, default=0.01)

    parser.add_argument("--save_processed_points", action="store_true")
    parser.add_argument("--save_processed_format", type=str, default="ply", choices=["npy", "ply", "both"])

    parser.add_argument("--no_auto_save_npy_on_miss", action="store_true")
    parser.add_argument("--precompute", default=True, action="store_true", help="Run offline precomputation and exit")

    args = parser.parse_args()

    if args.precompute:
        precompute_align_cache(
            root=args.root,
            area=args.area,
            num_category=args.num_category,
            save_processed_root=args.save_processed_root,
            ransac_iter=args.ransac_iter,
            ransac_thresh=args.ransac_thresh,
            save_processed_points=args.save_processed_points,
            save_processed_format=args.save_processed_format,
            auto_save_npy_on_miss=(not args.no_auto_save_npy_on_miss),
        )
        print("[Done] Precomputation complete.")
    else:
        print("Nothing to do. Use --precompute to run offline cache building.")
