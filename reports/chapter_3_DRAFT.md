# [OBSOLETE — Superseded by reports/chapter_3_report.md] MediTwin AI — Chapter 3: Medical Image Agent (DRAFT — Paused)

> [!CAUTION]
> Training was stopped after epoch 1. No model checkpoint has been saved yet. All numbers below are real; nothing has been invented.

---

## What Has Been Built

### Dataset — `paultimothymooney/chest-xray-pneumonia`

| Split | NORMAL | PNEUMONIA | Total |
|-------|--------|-----------|-------|
| Official train | 1341 | 3875 | 5216 |
| Official val   | 8    | 8    | 16   |
| Official test  | 234  | 390  | **624** |

Official val (16 images) is too small to use as-is → merged with train.

### New Stratified 85/15 Split (seed=42)

| Split | NORMAL | PNEUMONIA | Total |
|-------|--------|-----------|-------|
| Train | 1146 | 3308 | **4454** |
| Val   | 203  | 575  | **778** |
| Test  | 234  | 390  | **624** (untouched) |

**Patient-level grouping for PNEUMONIA**: 1642 unique person IDs extracted from filenames (e.g., `person12_bacteria_34.jpeg`). All images of one person kept in the same split. No PNEUMONIA files had missing IDs.

**NORMAL files**: no person ID in filenames → random 85/15 split used.

---

## Hardware & Model Choice

- **GPU**: not available (CPU only, 4 cores)
- **Model**: ResNet18 with ImageNet pretrained weights (`IMAGENET1K_V1`)
- **Final layer**: replaced FC(512→1000) with FC(512→2)
- **Config**: batch=32, Adam lr=1e-4, max_epochs=8, early_stopping patience=3
- **Class weights** (to handle 3:1 imbalance): NORMAL=1.9433×, PNEUMONIA=0.6732×
- **Augmentation (train only)**: RandomCrop(224 from 240), RandomRotation(±10°), ColorJitter(brightness=0.2, contrast=0.2), Grayscale→3ch, ImageNet normalisation. **No horizontal flip.**

---

## Training Progress (PARTIAL — 1 epoch completed, then killed)

| Ep | TrLoss | TrAcc | TrAUC | VaLoss | VaAcc | VaAUC | Note |
|----|--------|-------|-------|--------|-------|-------|------|
| 1 | 0.1288 | 95.29% | 98.93% | 0.1716 | 93.32% | **99.74%** | ✓ (would-be checkpoint) |
| 2–8 | — | — | — | — | — | — | not reached |

**Time per epoch**: ~1164–1471 s (~20–25 min) on 4-core CPU.

> [!WARNING]
> The epoch-1 model state was **not saved to disk** — the process was killed before `torch.save()` executed. `models/medical_image/` is empty.

---

## Files Existing Right Now

```
agents/medical_image_agent/train.py            ✓ complete training script
agents/medical_image_agent/medical_image_agent.py  ✓ reusable predict_image() module
reports/chapter_2_report.md                   ✓ Chapter 2 XAI report
reports/chapter_3_DRAFT.md                    ✓ this file
models/medical_image/                         ✗ EMPTY (no checkpoint saved)
logs/medical_image_*                          ✗ none yet
```

---

## Remaining Steps (TODO)

- [ ] **Re-run training to completion** — save `models/medical_image/resnet18_xray.pt`
- [ ] **Step 4** — Final eval on untouched test set (624 images): Accuracy, Precision, Recall, Specificity, F1, ROC-AUC, PR-AUC, Balanced Acc, MCC, confusion matrix PNG, ROC curve PNG, threshold at 0.5 and val-tuned
- [ ] **Step 5** — Grad-CAM for 4 test images → `logs/gradcam_*.png`
- [ ] **Step 6** — Confirm `predict_image()` runs on one real test image (needs checkpoint)
- [ ] **Step 7** — Final `reports/chapter_3_report.md` with honest limitations
- [ ] **Tracker** — Update Medical Image Agent row to Done

---

## Resume Instructions

### Re-run full training (from scratch, ~3–4 h on CPU)

```bash
cd /home/sai/Documents/MEDITWIN/meditwin-ai
source venv/bin/activate
nohup python agents/medical_image_agent/train.py > /tmp/train_ch3.log 2>&1 &
# Monitor:
tail -f /tmp/train_ch3.log
```

The script is idempotent: it will retrain from epoch 1, save the best checkpoint (chosen by val loss), then automatically run Steps 4 (test eval), 5 (Grad-CAM), and 6 (model save) in sequence.

### Skip training, evaluate only (once checkpoint exists)

If `models/medical_image/resnet18_xray.pt` already exists:

```python
# Load checkpoint directly in a notebook or script:
import torch
from torchvision import models
import torch.nn as nn

ckpt = torch.load("models/medical_image/resnet18_xray.pt", map_location="cpu")
model = models.resnet18(weights=None)
model.fc = nn.Linear(512, 2)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()
threshold = ckpt["val_threshold"]
# Then run test evaluation and Grad-CAM using the helpers in train.py
```

---

## Known Limitations (to be expanded in final report)

- Dataset from **one source** (Guangzhou Women and Children's Medical Centre, Kaggle version)
- Test set class balance (37.5% NORMAL, 62.5% PNEUMONIA) differs from real-world prevalence
- CPU training limits epochs and therefore potential accuracy ceiling
- Grayscale X-rays converted to 3-channel: some ImageNet pretraining signal may not transfer perfectly
- Model has not been evaluated on external datasets
