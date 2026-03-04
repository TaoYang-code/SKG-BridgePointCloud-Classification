# Structural Knowledge Guided Point Cloud Classification Model for Reinforced Concrete Bridge Components

This repository provides the implementation for **bridge component classification using bridge-wise batching and structural prior knowledge**.

The framework combines **PointNet / PointNet++** with structural priors to improve classification robustness for bridge components in point cloud data.

The following priors are incorporated:

* **Elevation-based hierarchical priors**
* **Projection-overlap priors**
* **Spatial adjacency priors**

These priors are applied during both **training and inference**.

<img width="1058" height="491" alt="image" src="https://github.com/user-attachments/assets/896f7320-28b9-46a9-88e6-a702bcc93ed2" />

---

# Environment

Experiments were conducted with the following environment:

```
Python: 3.8.20
PyTorch: 2.0.1
CUDA: 11.8
cuDNN: 8.7.0
GPU: NVIDIA Quadro RTX 5000
```

---

# Dataset Structure

Training data should be placed in:

```
data/mydataset/
```

Each bridge component type should be stored in a separate folder, for example:

```
data/mydataset/
│
├── pier/
├── abutment/
├── piercap/
├── girder/
├── deck/
├── handrail/
└── diaphragm/
```

Each **Area folder** contains the train/test split files used for cross-validation:

```
Area_1/
├── train.txt
└── test.txt
```

These files list the component filenames used for training and testing.

Example:

```
pier_001.txt
pier_002.txt
deck_004.txt
```

Example format files are provided in the repository.

---

# Data Preprocessing

Before training, preprocessing parameters must be generated.

Run the following script inside the `data_utils` directory:

```
cd data_utils
python precompute_align_cache.py
```

⚠️ **Important**

You must modify the dataset path inside `precompute_align_cache.py` to match your local dataset directory.

This step computes the alignment and normalization parameters required during data loading.

---

# Model Training

To train the hybrid model with structural priors:

```
python -B train_classification.py \
--model pointnet2_cls_ssg \
--log_dir pointnet2_cls_ssg \
--area 1 \
--Elevation_based_hierarchical_priors \
--Projection_overlap_priors \
--Spatial_adjacency_priors
```

Training logs and model checkpoints will be saved in:

```
log/classification/
```

---

# Model Testing

To evaluate a trained model:

```
python test_classification.py \
--model pointnet2_cls_ssg \
--area 1 \
--ckpt log/classification/PointnetFalse_Area1_EHPTrue_POPTrue_SAPTrue/checkpoints/best_model.pth \
--data_path data/mydataset/ \
--Elevation_based_hierarchical_priors \
--Projection_overlap_priors \
--Spatial_adjacency_priors
```

# Baseline Model (Without Structural Priors)

To train the baseline **PointNet / PointNet++ model without structural priors**, use:

```
--PointNet
```

Example:

```
python train_classification.py --PointNet
```

This disables:

* Bridge-wise batching
* Structural priors

---

# Switching Between PointNet and PointNet++

The model backbone can be manually switched in:

```
models/pointnet2_cls_ssg.py
```

Comment or uncomment the relevant code sections to switch between **PointNet** and **PointNet++** implementations.

---

# Notes

* Bridge-wise batching ensures that each batch contains components from the same bridge, enabling the use of bridge-level structural priors.
* The proposed framework improves classification performance when training data is limited or imbalanced.
* Although this work uses PointNet as the baseline model, the proposed structural knowledge guided strategy can be easily extended to other deep learning architectures.
* The framework can therefore be transferred to more advanced models for more complex bridge component classification tasks.
* Some prior-related thresholds are empirically defined in the current implementation and can be further refined when adapting the framework to more complex scenarios.

---
# Acknowledgements

This codebase is developed based on the PyTorch implementation of PointNet and PointNet++ from the following repository:https://github.com/yanx27/Pointnet_Pointnet2_pytorch
We thank the original authors for releasing their implementation.
