# Automatic Landmark Identification in 3D Cone-Beam Computed Tomography scans

Authors:
Maxime Gillot,
Baptiste Baquero,
Antonio Ruellas,
Marcela Gurgel,
Najla Al Turkestani,
Elizabeth Biggs,
Marilia Yatabe,
Jonas Bianchi,
Lucia Cevidanes,
Juan Carlos Prieto,
Andrey Zvyagin

We propose a novel approach that reformulates landmark detection as a classification problem through a virtual agent placed inside a 3D Cone-Beam Computed Tomography (CBCT) scan. This agent is trained to navigate in a multi-scale volumetric space to reach the estimated landmark position.

> This is a **modernized fork** of [Maxlo24/ALI_CBCT](https://github.com/Maxlo24/ALI_CBCT), updated to work with current versions of PyTorch, MONAI, and Python.

Landmark placed in the CBCT:
![LM_SELECTION_Trans](https://user-images.githubusercontent.com/46842010/159336503-827d70d5-2212-4dea-8ccc-46fc420be2e2.png)

## Prerequisites

Python >= 3.10

**Main libraries:**

| Package      | Version |
| ------------ | ------- |
| torch        | >= 2.0  |
| monai        | >= 1.0  |
| itk          | >= 5.3  |
| SimpleITK    | >= 2.2  |
| numpy        | >= 1.24 |
| scikit-learn | any     |

Install dependencies:

```bash
pip install -r requirements.txt
```

### Optional (for training only)

```bash
pip install tensorboard seaborn matplotlib
```

## Project structure

```
src/
  AGENT/
    predict_landmarks.py   — inference script
    train_ALI_agent.py     — training script
    Init_training_data.py  — data preparation
    Models_class.py        — Brain, DNet, RNet, DN
    Agents_class.py        — Agent navigation logic
    Environment_class.py   — 3D environment wrapper
    TrainingManager_class.py — training loop orchestrator
    GlobalVar.py           — landmark groups, movements, config
    utils.py               — I/O, resampling, metrics
    resnet2p1d.py          — R(2+1)D ResNet backbone
Docker/
    Dockerfile             — Docker image definition
requirements.txt
```

## Changes from the original repository

| Problem (original code)                             | Fix                                                              |
| --------------------------------------------------- | ---------------------------------------------------------------- |
| `AddChannel` removed in MONAI >= 1.4                | Replaced with `lambda data: data[None]`                          |
| `torch.from_numpy()` fails on MONAI `MetaTensor`    | Replaced with `torch.as_tensor()`                                |
| `from monai.transforms import transform`            | Removed (unused internal API)                                    |
| `from scipy.sparse.construct import random`         | Removed (deleted in scipy >= 1.12, was unused)                   |
| `torch.load()` without `weights_only`               | Added `weights_only=False` for PyTorch >= 2.0                    |
| `from torch.utils.tensorboard import SummaryWriter` | Lazy import via `try/except` (not needed for inference)          |
| ReLU on output layer before CrossEntropyLoss        | Removed — CE expects raw logits                                  |
| Boundary check `new_pos.all() > 0`                  | Fixed to `(new_pos > 0).all()`                                   |
| Network reset on float `==` loss match              | Replaced with patience-based mechanism (`stuck_patience=10`)     |
| No regularization in FC head (4 layers)             | Added `Dropout(0.3)` after first two ReLU layers                 |
| Fixed learning rate for all epochs                  | Added `ReduceLROnPlateau` scheduler                              |
| Weak augmentations (only intensity shift)           | Added Gaussian noise, contrast jitter, random 3D flips           |
| DataLoader without shuffle                          | Added `shuffle=True` for training data                           |
| No mixed precision                                  | Added AMP (`autocast` + `GradScaler`) with auto-detection        |
| DataLoader recreates workers each epoch             | Added `persistent_workers=True`, `pin_memory=True`               |
| PyTorch 1.11 in Docker                              | Updated to PyTorch 2.2 + CUDA 12.1, added `torch.compile()`      |
| `type=bool` for CLI flag                            | Fixed `--clear_temp` / `--no_clear_temp` with `store_true/false` |
| Wildcard `from resnet2p1d import *`                 | Explicit `from resnet2p1d import generate_model`                 |
| `f.close` without parentheses                       | Fixed to `f.close()`                                             |

---

# Running the code

## Using Docker

**Build the image:**

```bash
docker build -t alicbct:local -f Docker/Dockerfile .
```

**Download pre-trained models** from the [original releases](https://github.com/Maxlo24/ALI_CBCT/releases/tag/v0.1-models) and unzip them into a single folder.

Available model packs:

| Archive                 | Landmarks                                         |
| ----------------------- | ------------------------------------------------- |
| `Cranial_Base.zip`      | Ba, S, N, RPo, LPo, RFZyg, LFZyg, C2, C3, C4      |
| `Upper_Bones.zip`       | RInfOr, LInfOr, RPF, LPF, PNS, ANS, A, ...        |
| `Lower_Bones_1.zip`     | RCo, RGo, PogL, LGo, LCo, LAF, LAE, RAF, RAE, ... |
| `Lower_Bones_2.zip`     | Me, Gn, Pog, B, RMeF, LMeF                        |
| `Upper_Left_Teeth.zip`  | UL3O, UL6MB, UL6DB, ...                           |
| `Upper_Right_Teeth.zip` | UR3R, UR1R, UR3O, ...                             |
| `Lower_Left_Teeth.zip`  | LL6MB, LL6DB, LL3R, ...                           |
| `Lower_Right_Teeth.zip` | LR7R, LR5R, LR4R, ...                             |

### Input format

Scans must be in **NIfTI** (`.nii`, `.nii.gz`), **NRRD** (`.nrrd`), or **GIPL** (`.gipl`) format.

If you have DICOM files, convert them first:

```python
import SimpleITK as sitk

reader = sitk.ImageSeriesReader()
reader.SetFileNames(reader.GetGDCMSeriesFileNames("/path/to/dicom/folder"))
image = reader.Execute()
sitk.WriteImage(image, "output_scan.nii.gz")
```

### Run prediction

**On CPU:**

```bash
docker run --rm --shm-size=5gb \
  -v /path/to/scans:/app/data/scans \
  -v /path/to/models:/app/data/models \
  alicbct:local python3 /app/ALI_CBCT/predict_landmarks.py
```

**On GPU:**

```bash
docker run --rm --shm-size=5gb --gpus all \
  -v /path/to/scans:/app/data/scans \
  -v /path/to/models:/app/data/models \
  alicbct:local python3 /app/ALI_CBCT/predict_landmarks.py
```

> If GPU is not working, make sure you have the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker) installed.

### Prediction arguments

| Argument                     | Default             | Description                             |
| ---------------------------- | ------------------- | --------------------------------------- |
| `-lm` / `--landmarks`        | `Ba S`              | Landmarks to identify                   |
| `-sp` / `--spacing`          | `1 0.3`             | Spacing for multi-scale environment     |
| `-sps` / `--speed_per_scale` | `1 1`               | Agent speed at each scale               |
| `-sr` / `--spawn_radius`     | `10`                | Spawning radius around landmark         |
| `-fr` / `--focus_radius`     | `4`                 | Focus refinement radius                 |
| `-fov` / `--agent_FOV`       | `64 64 64`          | Agent field of view (crop size)         |
| `--dir_scans`                | `/app/data/scans`   | Directory with input scans              |
| `--dir_models`               | `/app/data/models`  | Directory with model weights            |
| `--dir_temp`                 | `..`                | Temporary directory for resampled scans |
| `--no_clear_temp`            | (clears by default) | Keep temp files after prediction        |

Example — find Ba, S, and N landmarks:

```bash
docker run --rm --shm-size=5gb \
  -v /path/to/scans:/app/data/scans \
  -v /path/to/models:/app/data/models \
  alicbct:local python3 /app/ALI_CBCT/predict_landmarks.py -lm Ba S N
```

### Output

Results are saved as `.mrk.json` files (3D Slicer Markups format) in the same directory as the input scans.

---

## Training

### Data preparation

Use `Init_training_data.py` to resample scans and convert landmarks to the expected format:

```bash
python3 src/AGENT/Init_training_data.py \
  --dir "path/to/patient_data" \
  --spacing 1 0.3
```

### Run training

```bash
python3 src/AGENT/train_ALI_agent.py \
  --dir_scans path/to/prepared_data/Patients \
  --dir_model path/to/output_models
```

### Training arguments

| Argument                    | Default    | Description                                  |
| --------------------------- | ---------- | -------------------------------------------- |
| `--dir_scans`               | (required) | Prepared patient scans directory             |
| `--dir_model`               | auto       | Output directory for trained models          |
| `-sp` / `--scale_spacing`   | `0.3 0.08` | Spacing per scale                            |
| `-ts` / `--training_scales` | `0 1`      | Which scales to train                        |
| `-fov` / `--agent_FOV`      | `64 64 64` | Agent field of view                          |
| `-bs` / `--batch_size`      | `150`      | Batch size                                   |
| `-ds` / `--dataset_size`    | `12000`    | Generated dataset size per epoch             |
| `-mi` / `--max_epoch`       | `1000`     | Maximum training epochs                      |
| `-lr` / `--learning_rate`   | `1e-4`     | Initial learning rate (decays via scheduler) |
| `-vf` / `--val_freq`        | `1`        | Validation frequency                         |
| `-tp` / `--test_percentage` | `15`       | Percentage of data for validation            |
| `-nw` / `--nbr_worker`      | `5`        | Number of DataLoader workers                 |

---

## Architecture

Environment, low resolution and high resolution:
![2environement_label_zoom](https://user-images.githubusercontent.com/46842010/159337231-0e79e134-a027-4987-ab44-edc2ad54d244.png)

Agent architecture:
![agent_label](https://user-images.githubusercontent.com/46842010/159341624-5d17e5a3-c4b7-4b93-bd7d-0b1348c7ad31.png)

Search steps of the agent to find one landmark:
![Search_3Steps_labeled](https://user-images.githubusercontent.com/46842010/159337300-ecb9e70e-7a65-45e1-96b1-490ad7286aa7.png)
