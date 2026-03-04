import os
import warnings
from pathlib import Path

import numpy as np
from torch.utils.data import Dataset

from data_utils.alignment_utils import (
    pc_normalize,
    farthest_point_sample,
    load_xyz_cached,
    extract_bridge_id,
    rot_z,
    pca_first_axis_on_plane,
    world_to_principal,
    fit_plane_ransac,
    align_plane_to_XY_preserve_X_about_center,
    area_and_zrange_xy,
    write_ply_xyz,
    load_pickle,
    save_pickle_atomic,
)

warnings.filterwarnings("ignore")


class ModelNetDataLoader(Dataset):
    """
    Returns:
      points_6d: (N,6) = [norm_xyz(3) | aligned_xyz(3)]
      label: int
      bridge_id: str
    """

    def __init__(self, root, args, split="train", process_data=False):
        self.root = root
        self.npoints = int(args.num_point)
        self.process_data = bool(process_data)
        self.uniform = bool(args.use_uniform_sample)
        self.use_normals = bool(args.use_normals)
        self.num_category = int(args.num_category)
        self.area = int(getattr(args, "area", 7))

        # Two-stage alignment switches
        self.apply_bridge_xy_then_plane = bool(getattr(args, "apply_bridge_xy_then_plane", True))
        self.require_align_cache = bool(getattr(args, "require_align_cache", True))
        self.allow_write_align_cache = bool(getattr(args, "allow_write_align_cache", False))
        self.auto_save_npy_on_miss = bool(getattr(args, "auto_save_npy_on_miss", False))

        # RANSAC hyperparameters
        self.ransac_iter = int(getattr(args, "ransac_iter", 2000))
        self.ransac_thresh = float(getattr(args, "ransac_thresh", 0.01))

        # Output root for caches / optional bridge-level dumps
        self.save_processed_points = bool(getattr(args, "save_processed_points", False))
        self.save_processed_format = str(getattr(args, "save_processed_format", "ply"))  # 'npy' | 'ply' | 'both'
        self.save_processed_root = str(getattr(args, "save_processed_root", os.path.join(self.root, "processed_points_xyplane")))

        # Category list file
        if self.num_category == 7:
            self.catfile = os.path.join(self.root, "shape_names.txt")
        self.cat = [line.rstrip() for line in open(self.catfile)]
        self.classes = dict(zip(self.cat, range(len(self.cat))))

        # Split list
        shape_ids = {}
        if self.num_category == 7:
            shape_ids["train"] = [line.rstrip() for line in open(os.path.join(self.root, f"Area{self.area}/train.txt"))]
            shape_ids["test"] = [line.rstrip() for line in open(os.path.join(self.root, f"Area{self.area}/test.txt"))]

        assert split in ("train", "test")

        # Infer class name from filename tokens
        shape_names = []
        for file in shape_ids[split]:
            fl = file.lower()
            if "cap" in fl:
                shape_names.append("piercap")
            elif "abutment" in fl:
                shape_names.append("abutment")
            elif "girder" in fl or "grider" in fl:## keep the typo "grider" to be compatible with legacy/misnamed point-cloud filenames
                shape_names.append("girder")
            elif "deck" in fl:
                shape_names.append("deck")
            elif "handrail" in fl:
                shape_names.append("handrail")
            elif "diaphragm" in fl:
                shape_names.append("diaphragm")
            elif "pier" in fl and "cap" not in fl:
                shape_names.append("pier")
            else:
                raise ValueError(f"Bad filename token (cannot infer class): {file}")

        # Build datapath
        self.datapath = []
        for i, sid in enumerate(shape_ids[split]):
            cls_name = shape_names[i]
            stem = sid if sid.endswith(".txt") else sid + ".txt"
            fp = os.path.join(self.root, cls_name, stem)
            if os.path.isfile(fp):
                self.datapath.append((cls_name, fp))
            else:
                # Skip missing file, but you may choose to raise here
                continue

        print(f"The size of {split} data is {len(self.datapath)}")

        # Bridge ids per instance
        self.bridge_ids = [extract_bridge_id(fp) for _, fp in self.datapath]

        # In-memory bridge cache (alignment parameters only)
        self._bridge_cache = {}

    # =============================
    # Cache paths / IO
    # =============================

    def _bridge_cache_dir(self, bridge_id: str) -> str:
        return os.path.join(self.save_processed_root, bridge_id)

    def _bridge_cache_path(self, bridge_id: str) -> str:
        return os.path.join(self._bridge_cache_dir(bridge_id), "align_cache.pkl")

    def _load_bridge_info_from_disk(self, bridge_id: str):
        p = self._bridge_cache_path(bridge_id)
        if os.path.isfile(p):
            return load_pickle(p)
        return None

    def _save_bridge_info_to_disk(self, bridge_id: str, info: dict):
        p = self._bridge_cache_path(bridge_id)
        save_pickle_atomic(p, info)

    # =============================
    # Bridge-level alignment computation
    # =============================

    def _compute_bridge_info(self, bridge_id: str):
        idxs = [i for i, bid in enumerate(self.bridge_ids) if bid == bridge_id]
        if not idxs:
            return None

        # Load all instances of this bridge
        inst_pts_raw = []
        for i in idxs:
            _, filepath = self.datapath[i]
            pts = load_xyz_cached(filepath, auto_save_npy_on_miss=self.auto_save_npy_on_miss).astype(np.float32)
            inst_pts_raw.append(pts)

        P_all = np.concatenate(inst_pts_raw, axis=0)  # (M,3)

        # Step-1: XY PCA (rotate about Z to align dominant axis to +X)
        u_xy_all, mean1 = pca_first_axis_on_plane(P_all, plane="xy")
        theta = float(np.arctan2(u_xy_all[1], u_xy_all[0]))
        R1 = rot_z(-theta)

        P_step1 = world_to_principal(P_all, mean1, R1)
        inst_step1 = [world_to_principal(p, mean1, R1) for p in inst_pts_raw]

        # Heuristic deck selection: among top-2 XY areas, pick smaller Z-range
        stats = [area_and_zrange_xy(p, 0.05, 0.95) for p in inst_step1]  # (area_xy, z_range)
        order = np.argsort([-a for (a, z) in stats])

        if len(order) == 0:
            return None

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

        # Step-2: RANSAC plane on deck, flatten to XY while preserving X as much as possible
        n_raw, _, inliers = fit_plane_ransac(deck_pts1, n_iters=self.ransac_iter, thresh=self.ransac_thresh, seed=0)
        deck_for_align = deck_pts1[inliers] if (inliers is not None and inliers.sum() > 3) else deck_pts1
        R2, c_deck = align_plane_to_XY_preserve_X_about_center(deck_for_align, n_raw)

        # Optional orientation rule: make "other big cluster" median y positive after step2
        if other_pts1 is not None:
            other_step2 = (other_pts1 - c_deck.reshape(1, 3)) @ R2 + c_deck.reshape(1, 3)
            y_med = float(np.median(other_step2[:, 1]))
            if y_med < 0.0:
                Fz = np.array([[-1.0, 0.0, 0.0],
                               [0.0, -1.0, 0.0],
                               [0.0, 0.0, 1.0]], dtype=np.float32)
                R2 = R2 @ Fz

        # Optionally save bridge-level step2 cloud (only in precompute mode)
        if self.save_processed_points and self.allow_write_align_cache:
            P_step2 = (P_step1 - c_deck.reshape(1, 3)) @ R2 + c_deck.reshape(1, 3)
            out_dir = self._bridge_cache_dir(bridge_id)
            os.makedirs(out_dir, exist_ok=True)
            if self.save_processed_format in ("npy", "both"):
                np.save(os.path.join(out_dir, f"{bridge_id}__bridge_step2.npy"), P_step2.astype(np.float32))
            if self.save_processed_format in ("ply", "both"):
                write_ply_xyz(os.path.join(out_dir, f"{bridge_id}__bridge_step2.ply"), P_step2.astype(np.float32))

        info = dict(
            mean1=mean1.astype(np.float32),
            R1=R1.astype(np.float32),
            R2=R2.astype(np.float32),
            c_deck=c_deck.astype(np.float32),
        )

        self._bridge_cache[bridge_id] = info
        return info

    def _get_bridge_info(self, bridge_id: str):
        # 1) memory
        info = self._bridge_cache.get(bridge_id, None)
        if info is not None:
            return info

        # 2) disk
        info = self._load_bridge_info_from_disk(bridge_id)
        if info is not None:
            self._bridge_cache[bridge_id] = info
            return info

        # 3) compute & write only if explicitly allowed (precompute mode)
        if self.allow_write_align_cache:
            info = self._compute_bridge_info(bridge_id)
            if info is not None:
                self._save_bridge_info_to_disk(bridge_id, info)
            return info

        # 4) strict training
        if self.require_align_cache:
            raise FileNotFoundError(
                f"[align_cache missing] Bridge '{bridge_id}' has no align_cache.pkl under "
                f"'{self._bridge_cache_dir(bridge_id)}'. Please run offline precompute."
            )

        # 5) non-strict: no alignment
        return None

    def _apply_two_stage_align(self, pts: np.ndarray, bridge_id: str) -> np.ndarray:
        """Apply step1 (R1 about mean1) then step2 (R2 about c_deck)."""
        info = self._get_bridge_info(bridge_id)
        if info is None:
            return pts

        mean1 = info["mean1"]
        R1 = info["R1"]
        R2 = info["R2"]
        c = info["c_deck"]

        p1 = world_to_principal(pts, mean1, R1)
        p2 = (p1 - c.reshape(1, 3)) @ R2 + c.reshape(1, 3)
        return p2

    # =============================
    # Load full points and apply alignment
    # =============================

    def _load_full_points_with_align(self, index):
        _, filepath = self.datapath[index]
        pts_full = load_xyz_cached(filepath, auto_save_npy_on_miss=False).astype(np.float32)  # (M,3)
        pts_raw = pts_full  # original coords

        if self.apply_bridge_xy_then_plane:
            bid = self.bridge_ids[index]
            pts_full = self._apply_two_stage_align(pts_full, bid)

        return pts_full, pts_raw

    # =============================
    # Load & sample raw XYZ
    # =============================

    def _load_and_sample_raw_xyz(self, index):
        cls_name, _ = self.datapath[index]
        label = int(self.classes[cls_name])

        pts_full_aligned, pts_full_raw = self._load_full_points_with_align(index)

        if self.uniform:
            pts_aligned = farthest_point_sample(pts_full_aligned, self.npoints)[:, :3]
            # For consistency, sample the same indices from raw coords is non-trivial in FPS.
            # If you need raw coords for normalization, consider implementing FPS index return.
            # Here we normalize using aligned points directly.
            pts_raw_sampled = pts_aligned.copy()
        else:
            M = pts_full_aligned.shape[0]
            idx = np.random.choice(M, self.npoints, replace=(M < self.npoints))
            pts_aligned = pts_full_aligned[idx, :]
            pts_raw_sampled = pts_full_raw[idx, :]

        return pts_aligned.astype(np.float32), pts_raw_sampled.astype(np.float32), label

    # =============================
    # Build (N,6) features
    # =============================

    def _get_item(self, index):
        pts_aligned, pts_raw_sampled, label = self._load_and_sample_raw_xyz(index)
        # Normalize based on ORIGINAL coordinates (your current design choice)
        pts_norm = pc_normalize(pts_raw_sampled.copy())
        points_6d = np.hstack([pts_norm, pts_aligned.astype(np.float32)])
        return points_6d, label

    # =============================
    # PyTorch Dataset Interface
    # =============================

    def __len__(self):
        return len(self.datapath)

    def __getitem__(self, index):
        points, label = self._get_item(index)
        return points, label, self.bridge_ids[index]
