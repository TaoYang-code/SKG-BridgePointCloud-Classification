import os
import sys
import argparse
import datetime
import logging
import shutil
from pathlib import Path
import importlib

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

import provider
from data_utils.ModelNetDataLoader import ModelNetDataLoader

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

from Function_bridge import (evaluate_non_bridgewise,test_by_bridge,GroupByBridgeBatchSampler)

# ------------------------------------------------
# 1. Argument parsing
# ------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser('Bridge component classification training')

    # basic runtime options
    parser.add_argument('--use_cpu', action='store_true', default=False)
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--batch_size', type=int, default=24)
    parser.add_argument('--model', default='pointnet2_cls_ssg')
    parser.add_argument('--num_category', default=7, type=int)
    parser.add_argument('--epoch', default=100, type=int)
    parser.add_argument('--learning_rate', default=0.001, type=float)
    parser.add_argument('--num_point', type=int, default=4096)
    parser.add_argument('--optimizer', type=str, default='Adam')
    parser.add_argument('--log_dir', type=str, default="pointnet2_cls_ssg")
    parser.add_argument('--decay_rate', type=float, default=1e-4)

    # dataset options
    parser.add_argument('--use_normals', action='store_true', default=False)
    parser.add_argument('--process_data', action='store_true', default=False)
    parser.add_argument('--use_uniform_sample', action='store_true', default=False)
    parser.add_argument('--area', type=int, default=3)

    # per-bridge dynamic class weights (clipping range)
    parser.add_argument('--bridge_weight_clip_min', type=float, default=0.25)
    parser.add_argument('--bridge_weight_clip_max', type=float, default=4.0)

    # bridge-wise batching
    parser.add_argument('--group_by_bridge', action='store_true', default=True)

    # prior-knowledge flags
    parser.add_argument('--Elevation_based_hierarchical_priors', action='store_true', default=False)
    parser.add_argument('--Projection_overlap_priors', action='store_true', default=False)
    parser.add_argument('--Spatial_adjacency_priors', action='store_true', default=False)

    # if True, run pure pointNet/PointNet++ baseline (no priors, no bridge-wise batching)
    parser.add_argument('--PointNet', action='store_true', default=False)

    return parser.parse_args()


# ------------------------------------------------
# 2. Small utilities
# ------------------------------------------------
def set_relu_inplace(module, inplace: bool = False):
    """Set inplace flag for all ReLU modules to avoid gradient issues."""
    import torch.nn as nn
    for m in module.modules():
        if isinstance(m, nn.ReLU):
            m.inplace = bool(inplace)


def setup_experiment_dirs(args):
    """Create experiment directory: root / checkpoints / logs."""
    timestr = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')

    exp_root = Path('./log/')
    exp_root.mkdir(exist_ok=True)

    exp_dir = exp_root.joinpath('classification')
    exp_dir.mkdir(exist_ok=True)

    if args.log_dir is None:
        exp_dir = exp_dir.joinpath(timestr)
    else:
        # encode main options into folder name for easier comparison
        exp_name = (f"Pointnet{args.PointNet}"+f"_Area{args.area}"+f"_EHP{args.Elevation_based_hierarchical_priors}"+f"_POP{args.Projection_overlap_priors}"+f"_SAP{args.Spatial_adjacency_priors}")
        exp_dir = exp_dir.joinpath(exp_name)

    exp_dir.mkdir(exist_ok=True)

    checkpoints_dir = exp_dir.joinpath('checkpoints')
    checkpoints_dir.mkdir(exist_ok=True)

    log_dir = exp_dir.joinpath('logs')
    log_dir.mkdir(exist_ok=True)

    return exp_dir, checkpoints_dir, log_dir


def setup_logger(log_dir: Path, model_name: str):
    """Configure logger that writes to file and prints to stdout."""
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    log_file = log_dir / f'{model_name}.txt'
    file_handler = logging.FileHandler(str(log_file))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    # avoid adding multiple handlers if script is re-imported
    if not logger.handlers:
        logger.addHandler(file_handler)

    def log_string(s: str):
        logger.info(s)
        print(s)

    return logger, log_string


def build_dataloaders(args, data_path: str):
    """Create train / test datasets and DataLoaders."""
    train_dataset = ModelNetDataLoader(root=data_path, args=args, split='train', process_data=args.process_data)
    test_dataset = ModelNetDataLoader(root=data_path, args=args, split='test', process_data=args.process_data)

    if args.group_by_bridge:
        # variable batch size grouped by bridge id
        train_sampler = GroupByBridgeBatchSampler(train_dataset,shuffle_bridges=True,shuffle_within=True,drop_last=False)
        train_loader = DataLoader(train_dataset, batch_sampler=train_sampler, num_workers=10)

        test_sampler = GroupByBridgeBatchSampler(test_dataset,shuffle_bridges=False,shuffle_within=False,drop_last=False)
        test_loader = DataLoader(test_dataset, batch_sampler=test_sampler, num_workers=10)
    else:
        # standard fixed batch size mode
        train_loader = DataLoader(train_dataset,batch_size=args.batch_size,shuffle=True,num_workers=10,drop_last=True,)
        test_loader = DataLoader(test_dataset,batch_size=args.batch_size,shuffle=False,num_workers=10)

    return train_dataset, test_dataset, train_loader, test_loader

# ------------------------------------------------
# 3. Main training routine
# ------------------------------------------------
def main(args):
    # GPU selection
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # If running plain PointNet baseline, disable priors and bridge-wise batching
    if args.PointNet:
        args.group_by_bridge = False
        args.Elevation_based_hierarchical_priors = False
        args.Projection_overlap_priors = False
        args.Spatial_adjacency_priors = False

    # directories & logger
    exp_dir, checkpoints_dir, log_dir = setup_experiment_dirs(args)
    logger, log_string = setup_logger(log_dir, args.model)

    log_string('PARAMETERS:')
    log_string(str(args))

    # ---------------- Dataset ----------------
    log_string('Loading dataset ...')
    data_path = ('data/mydataset/')

    train_dataset, test_dataset, train_loader, test_loader = build_dataloaders(args, data_path)

    class_name_to_id = train_dataset.classes
    num_class = args.num_category
    print("ID MAP:", class_name_to_id)

    # ---------------- Model & Loss ----------------
    importlib.invalidate_caches()
    model_lib = importlib.import_module(args.model)

    # Backup key scripts for reproducibility
    # shutil.copy(f'./models/{args.model}.py', str(exp_dir))
    # if os.path.exists('models/pointnet2_utils.py'):
    #     shutil.copy('models/pointnet2_utils.py', str(exp_dir))
    # shutil.copy('data_utils/ModelNetDataLoader.py', str(exp_dir))

    # copy this training script itself
    # this_script = os.path.basename(__file__)
    # if os.path.exists(this_script):
    #     shutil.copy(this_script, str(exp_dir))

    # global class weights based on the training dataset
    labels = [train_dataset.classes[name] for name, _ in train_dataset.datapath]
    counts = np.bincount(labels, minlength=num_class).astype(np.float64)
    w = 1.0 / np.maximum(counts, 1.0)
    w = w / (w.mean() + 1e-12)
    class_weights = torch.tensor(w, dtype=torch.float32)

    # classifier and criterion
    classifier = model_lib.get_model(num_class, normal_channel=args.use_normals)
    criterion = model_lib.get_loss(num_classes=num_class,class_weights=(class_weights.cuda()if (class_weights is not None and not args.use_cpu)else class_weights),
        class_name_to_id=class_name_to_id,
        Projection_overlap_priors=args.Projection_overlap_priors,
        Elevation_based_hierarchical_priors=args.Elevation_based_hierarchical_priors,
        Spatial_adjacency_priors=args.Spatial_adjacency_priors,
    )

    set_relu_inplace(classifier, inplace=False)

    if not args.use_cpu:
        classifier = classifier.cuda()
        criterion = criterion.cuda()

    # ---------------- Checkpoint & Optimizer ----------------
    try:
        ckpt_path = str(checkpoints_dir / 'best_model.pth')
        checkpoint = torch.load(ckpt_path)
        start_epoch = checkpoint['epoch']
        classifier.load_state_dict(checkpoint['model_state_dict'])
        log_string(f'Loaded pretrained model from {ckpt_path}')
    except Exception:
        log_string('No existing model, starting training from scratch...')
        start_epoch = 0

    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            classifier.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate,
        )
    else:
        optimizer = torch.optim.SGD(
            classifier.parameters(), lr=0.01, momentum=0.9
        )

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=20, gamma=0.7
    )

    # ---------------- Training Loop ----------------
    global_epoch = 0
    best_instance_acc = 0.0
    best_class_acc = 0.0
    best_instance_acc2 = 0.0
    best_class_acc2 = 0.0
    best_epoch = start_epoch

    logger.info('Start training...')

    for epoch in range(start_epoch, args.epoch):
        log_string(f'Epoch {global_epoch + 1} ({epoch + 1}/{args.epoch})')

        classifier.train()
        scheduler.step()

        mean_correct = []

        for batch_id, batch in tqdm(
            enumerate(train_loader, 0), total=len(train_loader), smoothing=0.9
        ):
            optimizer.zero_grad()

            # unpack batch
            if isinstance(batch, (list, tuple)) and len(batch) == 3:
                points_all, target, bridge_ids = batch
            elif isinstance(batch, (list, tuple)) and len(batch) >= 2:
                points_all, target = batch[0], batch[1]
                bridge_ids = None
            else:
                raise RuntimeError(
                    'Unexpected batch format: expected (points, target, bridge_ids)'
                )

            # keep raw coords for prior-based masking
            coords_raw = points_all[:, :, 3:6].clone()  # (B,N,3)

            # data augmentation on normalized coords (first 3 dims)
            pts_np = points_all.cpu().numpy()
            pts_np = provider.random_point_dropout(pts_np)
            pts_np[:, :, 0:3] = provider.random_scale_point_cloud(pts_np[:, :, 0:3])
            pts_np[:, :, 0:3] = provider.shift_point_cloud(pts_np[:, :, 0:3])

            points3 = torch.tensor(pts_np[:, :, 0:3]).transpose(2, 1).contiguous()

            if not args.use_cpu:
                points3 = points3.cuda(non_blocking=True)
                target = target.cuda(non_blocking=True)
                coords_raw = coords_raw.cuda(non_blocking=True)

            # forward pass
            try:
                pred, _ = classifier(points3)
            except TypeError:
                pred, _ = classifier(points3)

            # dynamic per-batch class weights (to reduce bridge-level imbalance)
            t_cpu = target.detach().cpu().numpy()
            counts_b = np.bincount(t_cpu, minlength=num_class).astype(np.float64)
            w_b = 1.0 / np.maximum(counts_b, 1.0)
            w_b = w_b / (w_b.mean() + 1e-12)
            w_b = torch.tensor(w_b, dtype=torch.float32, device=pred.device)
            w_override = torch.clamp(w_b,min=float(args.bridge_weight_clip_min),max=float(args.bridge_weight_clip_max))

            # main loss: masked cross-entropy with priors,
            # or plain CE for PointNet baseline
            if args.PointNet:
                loss = F.cross_entropy(pred, target.long())
            else:
                loss = criterion(
                    pred,
                    target.long(),
                    trans_feat=None,
                    coords_raw=coords_raw,
                    bridge_ids=list(bridge_ids) if bridge_ids is not None else None,
                    class_weights_override=w_override,
                )

            # prediction for training accuracy
            if not args.PointNet:
                with torch.no_grad():
                    _, _, pred_choice = criterion.apply_mask_at_inference_with_presence(
                        logits=pred,
                        coords_raw=coords_raw,
                        bridge_ids=list(bridge_ids) if bridge_ids is not None else None,
                    )
            else:
                prob = F.softmax(pred, dim=1)
                pred_choice = prob.argmax(dim=1)

            correct = (pred_choice == target.long()).sum().item()
            mean_correct.append(correct / float(points3.size(0)))

            loss.backward()
            optimizer.step()

        train_instance_acc = float(np.mean(mean_correct)) if mean_correct else 0.0
        log_string('Train Instance Accuracy: %f' % train_instance_acc)

        # ---------------- Validation per epoch ----------------
        with torch.no_grad():
            if not args.PointNet:
                id2name = {v: k for k, v in class_name_to_id.items()}
                log_string('=' * 12 + f' EVAL @ Epoch {epoch + 1} ' + '=' * 12)
                save_csv = str(exp_dir.joinpath('logs/per_bridge_test.csv'))
                instance_acc, class_acc, _ = test_by_bridge(
                    classifier.eval(),
                    test_loader,
                    num_class=num_class,
                    id2name=id2name,
                    log_fn=log_string,
                    save_csv_path=save_csv,
                    criterion=criterion,
                )
            else:
                instance_acc, class_acc, _ = evaluate_non_bridgewise(
                    classifier=classifier.eval(),
                    testDataLoader=test_loader,
                    criterion=criterion,
                    num_class=num_class,
                    args=args,
                    class_name_to_id=class_name_to_id,
                    log_fn=log_string,
                    topk=20,
                )

            # update best metrics
            if instance_acc >= best_instance_acc:
                best_instance_acc = instance_acc
                best_epoch = epoch + 1
            if class_acc >= best_class_acc:
                best_class_acc = class_acc

            log_string(
                'Test Instance Accuracy: %f, Class Accuracy: %f'
                % (instance_acc, class_acc)
            )
            log_string(
                'Best Instance Accuracy: %f, Class Accuracy: %f'
                % (best_instance_acc, best_class_acc)
            )

            # "stable" best after 60 epochs
            if epoch >= 70:
                if instance_acc >= best_instance_acc2:
                    best_instance_acc2 = instance_acc
                    best_epoch = epoch + 1
                if class_acc >= best_class_acc2:
                    best_class_acc2 = class_acc
                log_string(
                    'Best Instance Accuracy_after70epoch: %f, Class Accuracy: %f'
                    % (best_instance_acc2, best_class_acc2)
                )

            # save best_model.pth after 60 epochs when improved
            if (epoch >= 70) and (instance_acc >= best_instance_acc2):
                logger.info('Save best model...')
                savepath = str(checkpoints_dir / 'best_model.pth')
                log_string(f'Saving at {savepath}')
                state = {
                    'epoch': best_epoch,
                    'instance_acc': instance_acc,
                    'class_acc': class_acc,
                    'model_state_dict': classifier.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(state, savepath)

            # save snapshot every 10 epochs after epoch 70
            if (epoch >= 70) and (epoch % 10 == 0):
                logger.info('Save epoch snapshot...')
                savepath = str(checkpoints_dir / f'{epoch}.pth')
                log_string(f'Saving at {savepath}')
                state = {
                    'epoch': epoch,
                    'instance_acc': instance_acc,
                    'class_acc': class_acc,
                    'model_state_dict': classifier.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(state, savepath)

            global_epoch += 1

    logger.info('End of training...')


# ------------------------------------------------
# 4. Entry point
# ------------------------------------------------
if __name__ == '__main__':
    args = parse_args()
    main(args)
