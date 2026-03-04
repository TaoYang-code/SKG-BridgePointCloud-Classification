import math
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Sampler
from tqdm import tqdm


@torch.no_grad()
def evaluate_non_bridgewise(
    classifier: torch.nn.Module,
    test_loader: DataLoader,
    criterion: Any,
    num_class: int,
    args: Optional[Any] = None,
    class_name_to_id: Optional[Dict[str, int]] = None,
    log_fn=print,
    topk: int = 20,
):
    """
    Evaluate the classifier WITHOUT grouping samples by bridge.

    - Treats each instance independently (standard classification evaluation).
    - Tracks:
        * overall instance accuracy (OA)
        * per-class accuracy
        * per-bridge statistics (based on bridge_ids in the batch)
        * most frequent misclassifications (true -> pred)

    Args:
        classifier: Trained classifier model.
        test_loader: DataLoader for evaluation.
        criterion: Loss object (not used here, but kept for interface consistency).
        num_class: Number of semantic classes.
        args: Optional args namespace, used for flags like `use_cpu`, `use_extra3`, etc.
        class_name_to_id: Mapping from class name to class index.
        log_fn: Logging function, default is print.
        topk: Number of most frequent misclassification pairs to log.

    Returns:
        instance_acc (float): Overall instance accuracy.
        mean_class_acc (float): Mean class accuracy over all classes.
        extras (dict): Additional info including per-class, per-bridge, mispairs.
    """
    overall_correct = 0
    overall_total = 0
    # class_acc_arr[c, 0] = correct count of class c
    # class_acc_arr[c, 1] = total samples of class c
    class_acc_arr = np.zeros((num_class, 2), dtype=np.float64)
    per_bridge_stats: Dict[str, Dict[str, int]] = {}  # {bid: {"correct": int, "total": int}}
    mispairs_global: Dict[Tuple[int, int], int] = {}  # {(true_id, pred_id): count}

    # Build id -> name mapping for logging
    if class_name_to_id is not None:
        id2name = {int(v): str(k) for k, v in class_name_to_id.items()}
    else:
        id2name = {i: str(i) for i in range(num_class)}

    classifier.eval()

    for j, batch in enumerate(test_loader):
        # -------- unpack batch --------
        if isinstance(batch, (list, tuple)) and len(batch) >= 3:
            points_all, target, bridge_ids = batch[0], batch[1], batch[2]
            # Normalize bridge_ids to a list of strings (one per sample)
            if isinstance(bridge_ids, (list, tuple)):
                bid_list = [str(b) for b in bridge_ids]
            else:
                try:
                    bid_list = [str(bridge_ids.item()) for _ in range(points_all.size(0))]
                except Exception:
                    bid_list = [str(bridge_ids)] * points_all.size(0)
        else:
            points_all, target = batch[0], batch[1]
            bridge_ids = None
            # Fallback: synthetic bridge id based on batch index
            bid_list = [f"bridge_batch_{j}"] * points_all.size(0)

        # -------- prepare input channels --------
        # points3: normalized or learned features (first 3 dims)
        points3 = points_all[:, :, :3].transpose(2, 1).contiguous()  # (B, 3, N)
        # coords_raw: raw coordinates (for priors / masking)
        coords_raw = points_all[:, :, 3:6].contiguous()              # (B, N, 3)

        # Optional extra features (3-dim or 8-dim)
        if getattr(args, "use_extra3", False):
            extra = points_all[:, :, 6:9].transpose(2, 1).contiguous()
        elif getattr(args, "use_extra8", False):
            extra = points_all[:, :, 6:14].transpose(2, 1).contiguous()
        else:
            extra = None

        # Move to GPU if required
        if not getattr(args, "use_cpu", False):
            points3 = points3.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            coords_raw = coords_raw.cuda(non_blocking=True)
            if extra is not None:
                extra = extra.cuda(non_blocking=True)

        # -------- forward pass --------
        try:
            pred, _ = classifier(points3, extra) if extra is not None else classifier(points3)
        except TypeError:
            # Some models may not accept the extra argument
            pred, _ = classifier(points3)

        prob = F.softmax(pred, dim=1)
        pred_choice = prob.argmax(dim=1)

        # -------- global instance-level statistics --------
        correct_batch = (pred_choice == target).sum().item()
        overall_correct += correct_batch
        overall_total += int(target.shape[0])

        t_cpu = target.detach().cpu().numpy()
        p_cpu = pred_choice.detach().cpu().numpy()

        # Per-class accuracy accumulation
        for c in np.unique(t_cpu):
            mask_c = (t_cpu == c)
            class_acc_arr[c, 0] += (p_cpu[mask_c] == c).sum()
            class_acc_arr[c, 1] += mask_c.sum()

        # Misclassification counting (global)
        mis_mask = (t_cpu != p_cpu)
        if mis_mask.any():
            for t_i, p_i in zip(t_cpu[mis_mask].tolist(), p_cpu[mis_mask].tolist()):
                key = (int(t_i), int(p_i))
                mispairs_global[key] = mispairs_global.get(key, 0) + 1

        # Per-bridge accuracy accumulation
        for idx_b, bid in enumerate(bid_list):
            is_correct = int(p_cpu[idx_b] == t_cpu[idx_b])
            if bid not in per_bridge_stats:
                per_bridge_stats[bid] = {"correct": 0, "total": 0}
            per_bridge_stats[bid]["correct"] += is_correct
            per_bridge_stats[bid]["total"] += 1

    # ====== aggregate metrics ======
    instance_acc = overall_correct / max(overall_total, 1)

    per_class = np.divide(
        class_acc_arr[:, 0],
        np.maximum(class_acc_arr[:, 1], 1),
        out=np.zeros_like(class_acc_arr[:, 0]),
        where=class_acc_arr[:, 1] > 0,
    )
    mean_class_acc = (
        float(np.mean(per_class[class_acc_arr[:, 1] > 0]))
        if class_acc_arr[:, 1].sum() > 0
        else 0.0
    )

    # Per-bridge summary table
    per_bridge_rows: List[Tuple[str, int, int, float]] = []
    log_fn("===== Per-bridge results (no bridge-wise batching) =====")
    for bid in sorted(per_bridge_stats.keys()):
        c = per_bridge_stats[bid]["correct"]
        t = per_bridge_stats[bid]["total"]
        a = c / max(t, 1)
        per_bridge_rows.append((bid, c, t, a))
        log_fn(f"{bid}: {c}/{t} = {a:.4f}")

    # Overall
    log_fn("===== Overall (no bridge grouping) =====")
    log_fn(f"Instance Accuracy (OA): {instance_acc:.4f}")
    log_fn(f"Mean Class Accuracy:    {mean_class_acc:.4f}")

    # Per-class OA log
    log_fn("===== Per-class OA =====")
    for cid in range(num_class):
        total_c = int(class_acc_arr[cid, 1])
        correct_c = int(class_acc_arr[cid, 0])
        name = id2name.get(cid, str(cid))
        if total_c > 0:
            oa_c = correct_c / total_c
            log_fn(f"[{cid}] {name:<12s}: {correct_c}/{total_c} = {oa_c:.4f}")
        else:
            log_fn(f"[{cid}] {name:<12s}: N/A (no samples)")

    # Top-K misclassification log
    sorted_pairs = sorted(mispairs_global.items(), key=lambda kv: kv[1], reverse=True)
    if len(sorted_pairs) > 0:
        k = min(topk, len(sorted_pairs))
        log_fn(f"\nTop-{k} Misclassifications (true → pred : count):")
        for (t_i, p_i), cnt in sorted_pairs[:k]:
            t_name = id2name.get(t_i, str(t_i))
            p_name = id2name.get(p_i, str(p_i))
            log_fn(f"  {t_name} → {p_name} : {cnt}")

    extras = {
        "per_class": per_class,
        "per_bridge": per_bridge_rows,
        "mispairs": sorted_pairs,
    }
    return float(instance_acc), float(mean_class_acc), extras


@torch.no_grad()
def test_by_bridge(
    model: torch.nn.Module,
    loader: DataLoader,
    num_class: int = 7,
    id2name: Optional[Dict[int, str]] = None,
    log_fn=print,
    topk: int = 20,
    use_cpu: bool = False,
    save_csv_path: Optional[str] = None,
    criterion: Any = None,
):
    """
    Evaluate the classifier WITH bridge-wise grouping.

    - Each batch is a set of components from the same bridge (when using GroupByBridgeBatchSampler).
    - The criterion's `apply_mask_at_inference_with_presence` is used, so priors and
      presence-aware masking are applied consistently at test time.

    Args:
        model: Trained classifier model.
        loader: DataLoader with bridge-wise batches.
        num_class: Number of semantic classes.
        id2name: Mapping from class id to class name (used for logging).
        log_fn: Logging function (e.g., logger.info or print).
        topk: Number of most frequent misclassification pairs to log.
        use_cpu: If True, keep tensors on CPU.
        save_csv_path: Optional CSV path (placeholder, not used here).
        criterion: Loss object that implements `apply_mask_at_inference_with_presence`.

    Returns:
        instance_acc (float): Overall instance accuracy.
        mean_class_acc (float): Mean class accuracy.
        per_bridge_rows (list): [(bridge_id, correct, total, acc), ...]
    """
    if id2name is None:
        id2name = {
            0: "pier",
            1: "abutment",
            2: "piercap",
            3: "girder",
            4: "deck",
            5: "handrail",
            6: "diaphragm",
        }

    assert criterion is not None, "test_by_bridge requires `criterion` to apply masking at inference."
    model.eval()

    overall_correct = 0
    overall_total = 0
    class_acc = np.zeros((num_class, 2), dtype=np.float64)
    per_bridge_rows: List[Tuple[str, int, int, float]] = []
    mispairs_global: Dict[Tuple[int, int], int] = {}

    for j, batch in tqdm(enumerate(loader), total=len(loader)):
        if isinstance(batch, (list, tuple)) and len(batch) >= 3:
            points_all, target, bridge_ids = batch[0], batch[1], batch[2]
            bridge_id = str(bridge_ids[0]) if isinstance(bridge_ids, (list, tuple)) else str(bridge_ids)
        else:
            points_all, target = batch[0], batch[1]
            bridge_ids = None
            bridge_id = f"bridge_batch_{j}"

        # Split channels: first 3 = normalized features, last 3 = raw coordinates
        points3 = points_all[:, :, :3].transpose(2, 1).contiguous()  # (B, 3, N)
        coords_raw = points_all[:, :, 3:6].contiguous()              # (B, N, 3)

        if not use_cpu:
            points3 = points3.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            coords_raw = coords_raw.cuda(non_blocking=True)

        pred, _ = model(points3)

        # Apply inference-time masking + presence priors via criterion
        if bridge_ids is None:
            _, _, pred_choice = criterion.apply_mask_at_inference_with_presence(
                logits=pred,
                coords_raw=coords_raw,
                bridge_ids=None,
            )
        else:
            # Normalize to list of bridge ids
            if isinstance(bridge_ids, (list, tuple)):
                bid_list = list(bridge_ids)
            else:
                bid_list = [bridge_id] * points_all.size(0)

            _, _, pred_choice = criterion.apply_mask_at_inference_with_presence(
                logits=pred,
                coords_raw=coords_raw,
                bridge_ids=bid_list,
            )

        # Per-bridge accuracy
        correct = (pred_choice == target).sum().item()
        total = int(target.shape[0])
        per_bridge_rows.append((bridge_id, correct, total, correct / max(total, 1)))
        overall_correct += correct
        overall_total += total

        # Per-class accuracy
        t_cpu = target.detach().cpu().numpy()
        p_cpu = pred_choice.detach().cpu().numpy()
        for c in np.unique(t_cpu):
            mask = (t_cpu == c)
            class_acc[c, 0] += (p_cpu[mask] == c).sum()
            class_acc[c, 1] += mask.sum()

        # Misclassification statistics
        mis_mask = (t_cpu != p_cpu)
        if mis_mask.any():
            t_m = t_cpu[mis_mask].tolist()
            p_m = p_cpu[mis_mask].tolist()
            for t_i, p_i in zip(t_m, p_m):
                key = (int(t_i), int(p_i))
                mispairs_global[key] = mispairs_global.get(key, 0) + 1

    instance_acc = overall_correct / max(overall_total, 1)
    per_class = np.divide(
        class_acc[:, 0],
        np.maximum(class_acc[:, 1], 1),
        out=np.zeros_like(class_acc[:, 0]),
        where=class_acc[:, 1] > 0,
    )
    mean_class_acc = (
        float(np.mean(per_class[class_acc[:, 1] > 0]))
        if class_acc[:, 1].sum() > 0
        else 0.0
    )

    # Per-bridge summary log
    log_fn("===== Per-bridge results =====")
    for bid, c, t, a in per_bridge_rows:
        log_fn(f"{bid}: {c}/{t} = {a:.4f}")

    # Global summary log
    log_fn("===== Overall =====")
    log_fn(f"Instance Accuracy (OA): {instance_acc:.4f}")
    log_fn(f"Mean Class Accuracy:    {mean_class_acc:.4f}")

    # Per-class OA log
    log_fn("===== Per-class OA =====")
    for cid in range(num_class):
        total_c = int(class_acc[cid, 1])
        correct_c = int(class_acc[cid, 0])
        name = id2name.get(cid, str(cid))
        if total_c > 0:
            log_fn(f"[{cid}] {name:<12s}: {correct_c}/{total_c} = {per_class[cid]:.4f}")
        else:
            log_fn(f"[{cid}] {name:<12s}: N/A (no samples)")

    # Misclassification summary
    if len(mispairs_global) > 0:
        sorted_pairs = sorted(mispairs_global.items(), key=lambda kv: kv[1], reverse=True)
        log_fn(f"\nTop-{min(topk, len(sorted_pairs))} Misclassifications (true → pred : count):")
        for (t_i, p_i), c in sorted_pairs[:topk]:
            t_name = id2name.get(t_i, str(t_i))
            p_name = id2name.get(p_i, str(p_i))
            log_fn(f"  {t_name} → {p_name} : {c}")

    # Note: save_csv_path can be used later to dump per_bridge_rows if needed.
    return float(instance_acc), float(mean_class_acc), per_bridge_rows


def extract_bridge_id(file_path: str) -> str:
    """
    Extract a bridge identifier from a component file path.

    Heuristic rules:
        1) Split the stem by '_' or whitespace.
        2) If a component keyword (piercap, abutment, girder, deck, handrail,
           diaphragm, pier) appears, take everything before that as bridge id.
        3) Otherwise, try patterns like 'areaX' and 'bridgeY' from the stem.
        4) Fallback: use the parent directory name.

    Args:
        file_path: Path to a component file.

    Returns:
        bridge_id: String representing the bridge identity.
    """
    path = Path(file_path)
    orig_stem = path.stem
    tokens_orig = re.split(r"[_\s]+", orig_stem)
    tokens_low = [t.lower() for t in tokens_orig]

    comp_keys = [
        "piercap",
        "abutment",
        "girder",
        "grider",
        "deck",
        "handrail",
        "diaphragm",
        "pier",
    ]

    # Rule 1: cut before first component keyword
    cut_idx = None
    for i, t in enumerate(tokens_low):
        if any(t.startswith(k) for k in comp_keys):
            cut_idx = i
            break

    if cut_idx is not None and cut_idx > 0:
        return "_".join(tokens_orig[:cut_idx])

    # Rule 2: fallback to regex search on the whole stem
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

    # Rule 3: final fallback = folder name
    return path.parent.name


class GroupByBridgeBatchSampler(Sampler):
    """
    Batch sampler that groups dataset indices by bridge id.

    - Each yielded batch contains indices belonging to the same bridge.
    - Optionally shuffles bridge order and order within each bridge.
    - Can split a large bridge group into multiple smaller batches via `max_batch_size`.

    This is mainly used to ensure that a batch corresponds to one bridge, so that
    bridge-level priors (e.g., presence constraints, adjacency priors) can be
    applied consistently within each batch.

    Args:
        dataset: Dataset object. It should have either:
                 * attribute `bridge_ids` (list of bridge_id per sample), or
                 * attribute `datapath` where each item is (cls, filepath).
        shuffle_bridges: If True, shuffle the order of bridges.
        shuffle_within: If True, shuffle indices within each bridge group.
        max_batch_size: If not None, split large groups into sub-batches of
                        size at most `max_batch_size`.
        drop_last: If True, drop the last chunk if it is smaller than `max_batch_size`.
    """

    def __init__(
        self,
        dataset: Any,
        shuffle_bridges: bool = True,
        shuffle_within: bool = True,
        max_batch_size: Optional[int] = None,
        drop_last: bool = False,
    ):
        self.dataset = dataset
        self.shuffle_bridges = shuffle_bridges
        self.shuffle_within = shuffle_within
        self.max_batch_size = max_batch_size
        self.drop_last = drop_last

        bridge2idx: Dict[str, List[int]] = defaultdict(list)

        # Prefer explicit bridge_ids if present
        if hasattr(dataset, "bridge_ids") and dataset.bridge_ids is not None:
            for idx, bid in enumerate(dataset.bridge_ids):
                bridge2idx[bid].append(idx)
        else:
            # Fallback: derive bridge id from file path in datapath
            for idx, (_cls, fpath) in enumerate(getattr(self.dataset, "datapath", [])):
                bid = extract_bridge_id(fpath)
                bridge2idx[bid].append(idx)

        # List of groups, each is a list of indices for one bridge
        self.groups: List[List[int]] = list(bridge2idx.values())

    def __len__(self) -> int:
        """
        Approximate number of batches that this sampler will yield.
        """
        if not self.max_batch_size:
            # One batch per bridge group
            return len(self.groups)

        total = 0
        for g in self.groups:
            if self.drop_last and len(g) == 0:
                continue
            if self.drop_last:
                chunks = math.floor(len(g) / self.max_batch_size)
            else:
                chunks = math.ceil(len(g) / self.max_batch_size)
            total += max(chunks, 1)
        return total

    def __iter__(self) -> Iterable[Sequence[int]]:
        """
        Yield index lists, where each list corresponds to a (sub-)group of one bridge.
        """
        groups = self.groups[:]

        # Shuffle bridge order if requested
        if self.shuffle_bridges:
            random.shuffle(groups)

        for g in groups:
            if self.shuffle_within:
                random.shuffle(g)

            # Optionally split large groups into smaller sub-batches
            if self.max_batch_size and len(g) > self.max_batch_size:
                for i in range(0, len(g), self.max_batch_size):
                    chunk = g[i : i + self.max_batch_size]
                    if self.drop_last and len(chunk) < self.max_batch_size:
                        continue
                    yield chunk
            else:
                if self.drop_last and len(g) == 0:
                    continue
                yield g
