## Chameleon: Anisotropic Sharpness-Aware Minimization for Cross-Domain Adaptation

This repository contains the official implementation of the ICML 2026 submission **"Chameleon: Anisotropic Sharpness-Aware Minimization for Cross-Domain Adaptation"**.

The codebase provides:
- **Anisotropic SAM-style optimizers** (`algo/`, including `car`, `asam`, `fsam`, `vasso`, `mu2p`, `ssam`, etc.)
- **Cross-domain and noisy-label image benchmarks** (`datasets/`, including CIFAR-10/100, Tiny-ImageNet, PACS, Office-Caltech10, DomainNet)
- **CNN / ViT / Swin / EfficientNet backbones** (`models/`)
- **Training / evaluation / landscape & sharpness analysis utilities** (`train.py`, `utils/`)

If you use this code, please cite our paper once it becomes publicly available.

---

### 1. Environment setup

- **Python**: recommended 3.9+  
- **CUDA**: recommended CUDA-capable GPU and recent PyTorch build

Install dependencies (no version pins; adjust as needed):

```bash
pip install -r requirements.txt
```

The main libraries used include:
- `torch`, `torchvision`
- `transformers` (for DeiT / ViT / Swin / EfficientNet image models)
- `numpy`, `scipy`, `pandas`, `scikit-learn`
- `tqdm`, `matplotlib`, `seaborn`
- `pillow`, `opencv-python`, `psutil`
- `pyyaml`, `ujson`, `h5py`

---

### 2. Datasets

The dataset loaders are implemented in `datasets/`:
- `cifar10.py`, `cifar100.py`
- `tinyimagenet.py`
- `pacs.py`
- `office_caltech10.py`
- `domainnet.py`

Most datasets will be **automatically downloaded** (or prepared) into a local directory when first used, following the paths and download utilities defined in each file. For large domain adaptation benchmarks (e.g., PACS, DomainNet), you may need to:
- Manually download the datasets to a local folder
- Adjust root paths in the dataset files or provide environment variables / symlinks as appropriate

Please refer to the corresponding dataset scripts in `datasets/` if you need to customize paths or preprocessing.

---

### 3. Basic usage

All experiments are launched via `train.py`. The script exposes a rich set of arguments for:
- **Model architecture**: `--model` (e.g., `resnet18`, `vgg16`, `wide_resnet28_10`, `vit`, `swin`, etc. depending on `models/`)
- **Dataset**: `--dataset` (e.g., `cifar10`, `cifar100`, `tinyimagenet`, `pacs`, `office_caltech10`, `domainnet`)
- **Algorithm**: `--algo` (e.g., `erm`, `car`, `asam`, `fsam`, `vasso`, `mu2p`, `ssam`, `salp`, etc.)
- **Optimization & scheduler**: `--optimizer`, `--lr`, `--scheduler`, `--weight_decay`, etc.
- **SAM-related hyperparameters**: `--rho`, `--alpha`, `--beta`, `--sparsity`, `--rho_lr`, etc.

Key generic arguments (see `get_args()` in `train.py` for the full list):
- `--epochs`: total training epochs
- `--batch_size`: batch size (can accept multiple values)
- `--device`: GPU index (integer)
- `--seed`: random seed(s)
- `--aug`: data augmentation mode (`basic`, `cutout`, `cutmix`)
- `--noise_ratio`, `--noise_mode`: noisy label configuration

#### 3.1 Example: CIFAR-10 with ERM baseline

```bash
python train.py \
  --dataset cifar10 \
  --model resnet18 \
  --algo erm \
  --epochs 200 \
  --batch_size 128 \
  --lr 0.05 \
  --optimizer sgd \
  --scheduler cosine \
  --aug cutout
```

#### 3.2 Example: CIFAR-10 with Chameleon / CAR-style anisotropic SAM

```bash
python train.py \
  --dataset cifar10 \
  --model resnet18 \
  --algo car \
  --epochs 200 \
  --batch_size 128 \
  --lr 0.05 \
  --optimizer sgd \
  --scheduler cosine \
  --rho 0.1 \
  --alpha 0.01 \
  --beta 0.01 \
  --aug cutout
```

#### 3.3 Example: Cross-domain benchmark (e.g., PACS)

For cross-domain adaptation datasets (`pacs`, `office_caltech10`, `domainnet`), data loader keywords internally use `leave_domain` (domain left out for testing):

```bash
python train.py \
  --dataset pacs \
  --model resnet18 \
  --algo car \
  --leave_domain sketch \
  --epochs 200 \
  --batch_size 32 \
  --lr 0.0005 \
  --optimizer adamw \
  --scheduler cosine
```

Check the comments near `--classes` and `--leave_domain` in `train.py` for the valid domain names of each dataset.

---

### 4. Logging and checkpoints

Training and evaluation statistics are handled by `utils/logger.py` and saved as:
- **Excel files** (per experiment, per seed, and per hyperparameter configuration)
- **Model checkpoints** in folders under `records/` (by default)

Main conventions:
- Base directory: `--save_root` (default `records`)
- Folder naming encodes: model, dataset, optimizer, scheduler, algorithm, and key hyperparameters
- Checkpoints: `checkpoint_{epoch}.pt` plus best-checkpoints per metric (e.g., `checkpoint_{epoch}_best_test_acc_top1.pt`)

You can resume from checkpoints using `--reload_algo` and `--pre_epoch` (see `find_last_checkpoint()` and usage in `train.py`).

---

### 5. Reproducing results

To reproduce results in the paper:
- Use the **same dataset splits and domain configurations** as defined in `datasets/`
- Match the **hyperparameters** reported in the paper (optimizer, scheduler, `rho`, `alpha`, `beta`, `sparsity`, etc.)
- Run multiple seeds via `--seed` (e.g., `--seed 0 1 2 3`) to obtain averaged metrics

Performance and sharpness-related statistics:
- `utils/sharpness.py`, `utils/hessian.py`, and `utils/landscape.py` provide tools to compute eigenvalues, visualize loss landscapes, and log anisotropic sharpness metrics.

---

### 6. License and contact

The code is released for **academic research** related to optimization, sharpness-aware minimization, and cross-domain adaptation.  
For questions about the code or paper, please contact the authors of **"Chameleon: Anisotropic Sharpness-Aware Minimization for Cross-Domain Adaptation"** (contact information as listed in the ICML 2026 submission).

