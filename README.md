# PCCAN

This repository contains the implementation of a physics-guided counterfactual zero-shot compound-fault diagnosis pipeline under time-varying speed conditions. The code includes two experimental branches:

- **TTMB/GXU bearing branch**: inner-race fault, outer-race fault, rolling-element fault, and their compound modes.
- **BGMF/MFD bearing-gear branch**: bearing faults plus gear broken-tooth fault and bearing-gear compound modes.

The full workflow is:

```text
raw variable-speed vibration .mat files
        ↓
physics-guided compound fault feature decoupling
        ↓
component/order-spectrum cache: components.npz
        ↓
ordered graph neural backend with counterfactual intervention losses
        ↓
zero-shot / generalized zero-shot compound-fault evaluation
```

## 1. Repository structure

```text
PCCAN/
├── README.md
├── requirements.txt
├── environment.yml
├── common.py / common_bg.py
├── ordered_gat_utils.py
├── train_zs_TTMB.py                     # TTMB/GXU counterfactual GNN backend training
├── infer_TTMB.py                        # TTMB/GXU inference, metrics, visualization
├── dataset_gnn.py / models_gnn.py / backend_utils_gnn.py
├── train_zs_BGMF.py                     # BGMF/MFD counterfactual GNN backend training
├── infer_BGMF.py                        # BGMF/MFD inference, metrics, visualization
├── dataset_gnn_bg.py / models_gnn_bg.py / backend_utils_gnn_bg.py
├── data_TTMB/                           # put TTMB/GXU .mat files here; ignored by Git
├── data_BGMF/                           # put BGMF/MFD .mat files here; ignored by Git
├── checkpoints/                         # optional pretrained checkpoints; ignored by Git
├── scripts/                             # runnable shell/batch examples
└── docs/                                # data and output documentation
```

## 2. Environment and installation

All models in this repository were implemented and tested with the following software stack:

```text
Python 3.12
PyTorch 2.4.0
PyTorch Geometric backend for graph neural networks
```

Using the same Python and PyTorch versions is recommended for reproducing the reported results.

### 2.1 Install with pip

Install PyTorch 2.4.0 according to your CUDA/CPU environment first. For example, for a standard pip environment:

```bash
pip install torch==2.4.0
pip install -r requirements.txt
```

If you use a CUDA-enabled GPU, install the PyTorch 2.4.0 wheel that matches your CUDA runtime before installing the remaining packages.

### 2.2 Install with Conda

```bash
conda env create -f environment.yml
conda activate pccan
```

The provided `environment.yml` uses `python=3.12` and `pytorch=2.4.0`. If your machine requires a specific CUDA build, install the matching PyTorch 2.4.0 CUDA package in the Conda environment before running the experiments.

> Note: `torch-geometric` must be compatible with PyTorch 2.4.0 and the selected CUDA/CPU backend. If installation fails, install PyTorch Geometric using the wheel selector corresponding to PyTorch 2.4.0, then rerun `pip install -r requirements.txt` for the remaining packages.

## 3. Data preparation

### 3.1 TTMB/GXU branch

Place raw `.mat` files in `data_TTMB/`. By default, each file should contain:

- vibration signal variable: `data1`
- speed variable: `speed1`
- sampling frequency: `fs = 20000`

The code obtains weak labels from file names. Recommended names are:

```text
S_Normal.mat
S_IR.mat
S_OR.mat
S_Ball.mat
C_IO.mat
C_IB.mat
C_OB.mat
C_IOB.mat
```

Recognized bearing fault labels are:

| Code | Meaning |
|---|---|
| `IR` | inner-race fault |
| `OR` | outer-race fault |
| `Ball` / `B` | rolling-element fault |
| `C_IO` | `IR + OR` compound fault |
| `C_IB` | `IR + Ball` compound fault |
| `C_OB` | `OR + Ball` compound fault |
| `C_IOB` | `IR + OR + Ball` compound fault |

### 3.2 BGMF/MFD branch

Place raw `.mat` files in `data_BGMF/`. By default, each file should contain:

- vibration signal variable: `data`
- speed variable: `speed`
- sampling frequency: `fs = 16000`

Recommended names are:

```text
S_Normal.mat
S_IR.mat
S_OR.mat
S_Ball.mat
S_GTB.mat
C_IO.mat
C_ITB.mat
C_OTB.mat
```

Recognized BGMF labels are:

| Code | Meaning |
|---|---|
| `IR` | inner-race fault |
| `OR` | outer-race fault |
| `Ball` / `B` | rolling-element fault |
| `GTB` / `TB` | gear broken-tooth fault |
| `C_IO` | `IR + OR` compound fault |
| `C_ITB` | `IR + GTB` compound fault |
| `C_OTB` | `OR + GTB` compound fault |

## 4. Tasks

The backend supports two evaluation protocols:

- **Task A: strict zero-shot compound-fault diagnosis**  
  Training uses only single-fault samples. Real compound-fault samples are used only for testing.

- **Task B: generalized zero-shot compound-fault diagnosis**  
  Training still uses single-fault samples. Testing contains held-out seen classes and unseen compound classes, and reports seen accuracy, unseen accuracy, and harmonic mean when applicable.

## 5. Quick start

### 5.1 TTMB training

```bash
python train_zs_TTMB.py \
  --data_dir ./data_TTMB \
  --data_key data1 \
  --speed_key speed1 \
  --fs 20000 \
  --task A \
  --epochs 100 \
  --seed 42
```

For generalized zero-shot evaluation:

```bash
python train_zs_TTMB.py --data_dir ./data_TTMB --task B --epochs 100 --seed 42
```

### 5.2 TTMB inference

```bash
python infer_TTMB.py \
  --cache_root ./multicomponent_results_TTMB \
  --ckpt ./zs_outputs_TTMB/best_gnn_backend_cf.pt \
  --out_dir ./infer_outputs_TTMB \
  --eval_mode taskA_unseen
```

For Task B:

```bash
python infer_TTMB.py \
  --cache_root ./multicomponent_results_TTMB \
  --ckpt ./zs_outputs_TTMB/best_gnn_backend_cf.pt \
  --out_dir ./infer_outputs_TTMB_taskB \
  --eval_mode taskB_all
```

### 5.3 BGMF training

```bash
python train_zs_BGMF.py \
  --data_dir ./data_BGMF \
  --data_key data \
  --speed_key speed \
  --fs 16000 \
  --task A \
  --epochs 100 \
  --seed 42
```

For Task B:

```bash
python train_zs_BGMF.py --data_dir ./data_BGMF --task B --epochs 100 --seed 42
```

### 5.4 BGMF inference

```bash
python infer_BGMF.py \
  --cache_root ./multicomponent_results_BGMF \
  --ckpt ./zs_outputs_BGMF/best_model.pt \
  --out_dir ./infer_outputs_BGMF \
  --eval_mode taskA_unseen
```

For Task B:

```bash
python infer_BGMF.py \
  --cache_root ./multicomponent_results_BGMF \
  --ckpt ./zs_outputs_BGMF/best_model.pt \
  --out_dir ./infer_outputs_BGMF_taskB \
  --eval_mode taskB_all
```

## 6. Main outputs

### training outputs

`zs_outputs_TTMB/` or `zs_outputs_BGMF/` contains:

- trained checkpoint, for example `best_gnn_backend_cf.pt` or `best_model.pt`.
- `train_history.json`: training curves and validation records.
- `eval_metrics.json`: backend evaluation metrics when generated.
- `ordered_graph_nodes.csv` and `ordered_graph_nodes.json`: ordered pre-GAT graph node definitions.

### Inference outputs

`infer_outputs_*/` contains:

- `predictions.csv`: per-sample prediction probabilities and labels.
- `infer_metrics.json`: exact-match accuracy and split metrics.
- `probability_figures_by_label/`: spectrum and probability figures by label.
- `counterfactual_intervention_by_label/`: counterfactual intervention visualization results.
- `repeated_seed_metrics.csv`: generated when multi-seed inference is enabled.

## 7. Citation

If this code is used in a paper, please cite the corresponding manuscript. Add the final BibTeX entry here after publication.
