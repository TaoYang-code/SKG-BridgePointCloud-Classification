# -*- coding: utf-8 -*-
"""
Test script for bridge component classification.
"""

import os
import sys
import argparse
import logging
from pathlib import Path
import importlib

import torch
import numpy as np
from torch.utils.data import DataLoader

from data_utils.ModelNetDataLoader import ModelNetDataLoader
from Function_bridge import (
    evaluate_non_bridgewise,
    test_by_bridge,
    GroupByBridgeBatchSampler,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, "models"))


# ------------------------------------------------
# 1. Argument parsing
# ------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser("Bridge component classification testing")

    parser.add_argument("--use_cpu", action="store_true", default=False)
    parser.add_argument("--gpu", type=str, default="0")

    # Match training settings
    parser.add_argument("--batch_size", type=int, default=24)
    parser.add_argument("--model", default="pointnet2_cls_ssg")
    parser.add_argument("--num_category", default=7, type=int)
    parser.add_argument("--num_point", type=int, default=4096)
    parser.add_argument("--use_normals", action="store_true", default=False)
    parser.add_argument("--process_data", action="store_true", default=False)
    parser.add_argument("--use_uniform_sample", action="store_true", default=False)
    parser.add_argument("--area", type=int, default=3)

    # Bridge-wise batching (for hybrid evaluation)
    parser.add_argument("--group_by_bridge", action="store_true", default=True)

    # Priors (must be consistent with training)
    parser.add_argument("--Elevation_based_hierarchical_priors", action="store_true", default=False)
    parser.add_argument("--Projection_overlap_priors", action="store_true", default=False)
    parser.add_argument("--Spatial_adjacency_priors", action="store_true", default=False)

    # If True -> baseline mode: no priors + no bridge-wise batching
    parser.add_argument("--PointNet", action="store_true", default=False)

    # Paths
    parser.add_argument("--data_path", type=str, default="data/mydataset/")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint .pth (e.g., best_model.pth)")
    parser.add_argument("--save_csv", type=str, default=None, help="Optional: path to save per-bridge CSV")

    return parser.parse_args()


# ------------------------------------------------
# 2. Logger
# ------------------------------------------------
def setup_logger():
    """Create a simple stdout logger."""
    logger = logging.getLogger("Test")
    logger.setLevel(logging.INFO)

    # Avoid duplicated handlers if re-run in the same interpreter
    if not logger.handlers:
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        ch.setFormatter(fmt)
        logger.addHandler(ch)

    def log_string(s: str):
        logger.info(s)

    return logger, log_string


# ------------------------------------------------
# 3. Data
# ------------------------------------------------
def build_test_loader(args):
    """Build test dataset and dataloader."""
    test_dataset = ModelNetDataLoader(
        root=args.data_path,
        args=args,
        split="test",
        process_data=args.process_data,
    )

    if args.group_by_bridge:
        # One batch per bridge (or per bridge chunk) depending on sampler settings
        test_sampler = GroupByBridgeBatchSampler(
            test_dataset,
            shuffle_bridges=False,
            shuffle_within=False,
            drop_last=False,
        )
        test_loader = DataLoader(test_dataset, batch_sampler=test_sampler, num_workers=10)
    else:
        # Standard fixed batch size loader
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=10,
        )

    return test_dataset, test_loader


# ------------------------------------------------
# 4. Model and criterion loading
# ------------------------------------------------
def compute_global_class_weights(train_dataset, num_class: int) -> torch.Tensor:
    """
    Compute global class weights based on the training split, following the same
    logic as the training script.
    """
    labels = [train_dataset.classes[name] for name, _ in train_dataset.datapath]
    counts = np.bincount(labels, minlength=num_class).astype(np.float64)
    w = 1.0 / np.maximum(counts, 1.0)
    w = w / (w.mean() + 1e-12)
    return torch.tensor(w, dtype=torch.float32)


def load_model_and_criterion(args, class_name_to_id, class_weights):
    """Load model/criterion and restore checkpoint."""
    importlib.invalidate_caches()
    model_lib = importlib.import_module(args.model)

    num_class = args.num_category
    model = model_lib.get_model(num_class, normal_channel=args.use_normals)

    # Criterion is required by test_by_bridge() to apply inference-time masking
    criterion = model_lib.get_loss(
        num_classes=num_class,
        class_weights=(class_weights.cuda() if (class_weights is not None and not args.use_cpu) else class_weights),
        class_name_to_id=class_name_to_id,
        Projection_overlap_priors=args.Projection_overlap_priors,
        Elevation_based_hierarchical_priors=args.Elevation_based_hierarchical_priors,
        Spatial_adjacency_priors=args.Spatial_adjacency_priors,
    )

    if not args.use_cpu:
        model = model.cuda()
        criterion = criterion.cuda()

    # Load checkpoint
    ckpt = torch.load(args.ckpt, map_location=("cpu" if args.use_cpu else "cuda"))
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        # Fallback: checkpoint is directly a state_dict
        model.load_state_dict(ckpt)

    return model.eval(), criterion


# ------------------------------------------------
# 5. Main
# ------------------------------------------------
def main(args):
    # GPU selection
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # Baseline mode: disable priors and bridge-wise batching
    if args.PointNet:
        args.group_by_bridge = False
        args.Elevation_based_hierarchical_priors = False
        args.Projection_overlap_priors = False
        args.Spatial_adjacency_priors = False

    logger, log_string = setup_logger()

    log_string("Loading test dataset ...")
    _, test_loader = build_test_loader(args)

    # For reproducibility of global class weights, also load train split
    train_dataset = ModelNetDataLoader(
        root=args.data_path,
        args=args,
        split="train",
        process_data=args.process_data,
    )

    class_name_to_id = train_dataset.classes
    id2name = {v: k for k, v in class_name_to_id.items()}
    num_class = args.num_category

    class_weights = compute_global_class_weights(train_dataset, num_class=num_class)

    log_string(f"Loading checkpoint: {args.ckpt}")
    model, criterion = load_model_and_criterion(args, class_name_to_id, class_weights)

    log_string("Start testing ...")

    with torch.no_grad():
        if not args.PointNet:
            # Hybrid evaluation: bridge-wise, with inference-time masking via criterion
            save_csv = args.save_csv
            if save_csv is None:
                # Default output location: .../checkpoints/../logs/per_bridge_test.csv
                save_csv = str(Path(args.ckpt).parent.parent / "logs" / "per_bridge_test.csv")

            # IMPORTANT: test_by_bridge signature is (model, loader, ...)
            instance_acc, class_acc, _ = test_by_bridge(
                model,
                test_loader,
                num_class=num_class,
                id2name=id2name,
                log_fn=log_string,
                topk=20,
                use_cpu=bool(args.use_cpu),
                save_csv_path=save_csv,
                criterion=criterion,
            )

            log_string(f"[Hybrid] Test Instance Acc: {instance_acc:.6f}, Class Acc: {class_acc:.6f}")
            log_string(f"Per-bridge CSV path (optional): {save_csv}")

        else:
            # Baseline evaluation: non-bridgewise standard evaluation
            instance_acc, class_acc, _ = evaluate_non_bridgewise(
                classifier=model,
                test_loader=test_loader,
                criterion=criterion,
                num_class=num_class,
                args=args,
                class_name_to_id=class_name_to_id,
                log_fn=log_string,
                topk=20,
            )

            log_string(f"[Baseline] Test Instance Acc: {instance_acc:.6f}, Class Acc: {class_acc:.6f}")


if __name__ == "__main__":
    args = parse_args()
    main(args)