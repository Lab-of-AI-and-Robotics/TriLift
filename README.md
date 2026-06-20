

<div align="center">
<h2>TriLift: Interpolation-Free Tri-Plane Lifting for Efficient 3D Perception on Embedded Systems</h2>

[**Sibaek Lee**](https://minjae-lulu.github.io/) · [**Jiung Yeon**](https://humdrum-balance-b8f.notion.site/Jiung-Yeon-6754922a22814c9a95af88801a96fb4b) · [**Hyeonwoo Yu**](https://bogus2000.github.io/) 

Sungkyunkwan University 

<h3 align="center"> IROS 2026 </h3>

</div>


## Overview

<div align="center">
<img src="figs/overview.png" alt="System Overview" width="100%">
</div>

**System Overview.** (a) A dense 3D ConvNet baseline. (b) Our proposed hybrid architecture, which utilizes interpolation-free broadcast-summation to reconstruct feature volumes while shifting non-linear complexity to 2D convolutions for GPU acceleration. (c) The data-adaptive positional encoding process where 3D context is summarized into 1D tokens to generate dynamic spatial embeddings. These embeddings provide essential spatial cues to compensate for information loss and retain geometric details in our pipeline.

## Code

| Folder | Task(s) | Dataset(s) | Input voxel |
|---|---|---|---|
| `Classification_Completion/` | Shape classification & completion | ModelNet40 | occupancy (C=1) |
| `Semantic_Segmentation/` | Semantic segmentation | ScanNet, Stanford3D | occupancy + RGB (C=4) |
| `Object_Detection/` | 3D object detection (NeRF-RPN) | 3D-FRONT, Hypersim, ScanNet | NeRF density (C=1) |

---

## Installation

```bash
conda env create -f environment.yml      # creates the `trilift` env
conda activate trilift
pip install -r requirements.txt

# (Object Detection only) build the rotated-IoU CUDA op used by --rotated_bbox:
cd Object_Detection/model/rotated_iou/cuda_op
python setup.py install
cd -
```

---

## 1. Classification & Completion  (`Classification_Completion/`)

### Data (ModelNet40)
We host the raw data on Google Drive — download without any account (via `gdown`):

```bash
pip install gdown
cd Classification_Completion/datasets

# Classification raw (HDF5, ~416MB)
gdown 1s_lM8yAaQ8xXEpau6kTbMlH4Inng2sXI
tar xzf modelnet40_ply_hdf5_2048.tar.gz

# Completion raw (.off meshes, ~2GB)
gdown 1sShW_ItA7yX0_8yBvAH70FzOEBKnvOS9
tar xzf ModelNet40.tar.gz
cd ..
```

### Preprocess (point cloud / mesh → voxel grids)
```bash
cd Classification_Completion
python prepare_classification.py      # -> processed_data/classification_{train,test}_res128_binary
python prepare_completion.py          # -> processed_data/completion_chair_{train,test}_res128
```
Preprocessing scripts resolve all paths relative to their own location, so they can be run from any working directory.

### Train — our method
```bash
bash run_classification.sh        # TriLift-F(1/2);  `bash run_classification.sh 0.25` for TriLift-F(1/4)
bash run_completion.sh            # same ratio convention
```

### Visualize a preprocessed voxel sample
```bash
python vis_voxel.py processed_data/completion_chair_train_res128/chair_0001.pt
```

---

## 2. Semantic Segmentation  (`Semantic_Segmentation/`)

### Data (ScanNet / Stanford3D)
ScanNet and Stanford3D require accepting their respective license agreements and **cannot be
redistributed**, so download them from their official sources:
1. Generate the raw per-scene data using
   [SpatioTemporalSegmentation](https://github.com/chrischoy/SpatioTemporalSegmentation.git) (branch `v0.5`).
2. Place the produced `preprocessed/` folder inside `Semantic_Segmentation/`.
3. For Stanford3D, the folder must be named `Stanford3D` (capitalized).

### Preprocess (→ voxel grids)
```bash
cd Semantic_Segmentation
python prepare_segmentation.py --save                       # ScanNet + Stanford3D (if present)
python prepare_segmentation.py --dataset scannet --save
python prepare_segmentation.py --dataset stanford3d --save
```

### Train — our method
```bash
bash run_segmentation.sh                          # ScanNet, TriLift-F(1/2); `bash run_segmentation.sh 0.25` for (1/4)
bash run_segmentation.sh 0.5 --dataset stanford3d # Stanford3D
```

### Visualize
```bash
python vis.py --train-dir processed_data/scannet/train --test-dir processed_data/scannet/test
```

---

## 3. Object Detection  (`Object_Detection/`)

### Data (NeRF-RPN)
Download the `*_rpn_data.zip` files for each dataset from the
[NeRF-RPN dataset](https://huggingface.co/datasets/lyclyc52/NeRF_RPN/tree/main) on HuggingFace,
then extract each archive (some are nested, so keep unzipping until you see the `*_rpn_data/`
folder containing `features/`, `obb/`, and `*_split.npz`). Place them under `Object_Detection/data/`:
```
data/
├── front3d/   front3d_rpn_data/   (features/, obb/, aabb/, 3dfront_split.npz)
├── hypersim/  hypersim_rpn_data/
└── scannet/   scannet_rpn_data/
```

### Preprocess (rgbsigma .npz → cube features)
```bash
cd Object_Detection
python data_modify.py --dataset front3d
python data_modify.py --dataset hypersim
python data_modify.py --dataset scannet
```

### Train / Test
```bash
bash train.sh
bash test.sh
```

---

## Run everything

`run_all.sh` runs all 4 tasks at ratio 0.5 (TriLift-F 1/2) and 0.25 (TriLift-F 1/4):
```bash
conda activate trilift
bash run_all.sh 2>&1 | tee run_all.log
```

---

## Citation

```bibtex
@article{lee2025trilift,
  title={TriLift: Interpolation-Free Tri-Plane Lifting for Efficient 3D Perception on Embedded Systems},
  author={Lee, Sibaek and Yeon, Jiung and Yu, Hyeonwoo},
  journal={arXiv preprint arXiv:2509.14641},
  year={2025}
}
```
