# Model Card: TorchXRayVision DenseNet121-res224-all (Adult Chest X-Ray)

## Overview

| Field | Value |
|---|---|
| Model identifier | `densenet121-res224-all` |
| Library | [TorchXRayVision](https://github.com/mlmed/torchxrayvision) |
| License | Apache-2.0 *(from model card and Hugging Face hub entry)* |
| Input resolution | 224 × 224 px (1-channel, grayscale) |
| Output | 18 pathology scores (op-normalized probabilities) |
| Used in this project | Adult chest X-ray screening route (age ≥ 18 years) |

---

## Training Data

Weights named `densenet121-res224-all` were trained on the union of:

- **NIH ChestX-ray14** (Wang et al., 2017)
- **PadChest** (Bustos et al., 2020)
- **CheXpert** (Irvin et al., 2019)
- **MIMIC-CXR** (Johnson et al., 2019)
- **Google Chestx-ray** (Majkowska et al., 2020)
- **OpenI** (Demner-Fushman et al., 2016)
- **RSNA Pneumonia Detection** (Shih et al., 2019)

> [!CAUTION]
> Training data overlaps **every large public adult chest X-ray dataset**. Independent accuracy measurement in this project is therefore not possible. No benchmark numbers from outside publications are reproduced here.

---

## Output Semantics (Verified from Source)

**Source files read:** `torchxrayvision/models.py` (lines 329–361), `torchxrayvision/datasets.py` (lines 2732–2765), `torchxrayvision/utils.py` (lines 62–84).

The model forward pass:
1. Passes through DenseNet → raw logits.
2. Because `op_threshs` is loaded from `model_urls['all']`, applies `torch.sigmoid(out)` then `op_norm(out, self.op_threshs)`.
3. `op_norm` rescales each output so the operating threshold maps to 0.5.
4. **Final outputs are op-normalized probabilities**, NOT raw sigmoid logits.

Pneumonia op-threshold (index 8): **0.056810737**  
Source: `model_urls['all']['op_threshs'][8]` in `torchxrayvision/models.py` (line ~25).

---

## Preprocessing Pipeline (Verified from Source)

```python
import numpy as np
from PIL import Image
import torchxrayvision as xrv
import torch

img_uint8 = np.array(Image.open(path).convert('L'))   # grayscale 2D uint8
img = xrv.datasets.normalize(img_uint8, 255, reshape=True)  # (1,H,W) float32, range ~[-1024,1024]
img = xrv.datasets.XRayCenterCrop()(img)              # (1,N,N) center-square crop
img = xrv.datasets.XRayResizer(224)(img)              # (1,224,224)
tensor = torch.from_numpy(img).unsqueeze(0)           # (1,1,224,224)
```

---

## 18 Output Labels

| Index | Label |
|---|---|
| 0 | Atelectasis |
| 1 | Consolidation |
| 2 | Infiltration |
| 3 | Pneumothorax |
| 4 | Edema |
| 5 | Emphysema |
| 6 | Fibrosis |
| 7 | Effusion |
| 8 | **Pneumonia** |
| 9 | Pleural_Thickening |
| 10 | Cardiomegaly |
| 11 | Nodule |
| 12 | Mass |
| 13 | Hernia |
| 14 | Lung Lesion |
| 15 | Fracture |
| 16 | Lung Opacity |
| 17 | Enlarged Cardiomediastinum |

---

## What Was NOT Validated in This Project

- No test-set accuracy, AUC, sensitivity, or specificity was measured.
- No data independent of the model's training sources was available for evaluation.
- No clinical validation, prospective study, or regulatory clearance has been performed.

---

## Papers

- Cohen JP, Viviano J, Bertin P, et al. (2021). *TorchXRayVision: A library of chest X-ray datasets and models.* arXiv:2111.00595.
- Cohen JP, Morrison P, Dao L, et al. (2020). *COVID-19 Image Data Collection.* arXiv:2003.11597.
- Primary training/architecture paper: arXiv:2002.02497 (Cohen et al.)

---

## Age Routing Logic in This Project

| Patient age | Model used |
|---|---|
| < 18 years | Pediatric Kermany ResNet18 (Guangzhou Women and Children's Medical Center) |
| ≥ 18 years | **This model** (DenseNet121-res224-all) |
| Missing | Error — age required |

---

## Limitations

1. Pretrained public model; no additional fine-tuning was performed in this project.
2. Training data overlaps every large public adult chest X-ray dataset; independent accuracy cannot be measured in this project.
3. Scores are op-normalized model outputs, not calibrated clinical probabilities.
4. Screening aid only — not a diagnosis. A licensed radiologist must interpret all results.
5. Image quality, projection (PA vs AP), and patient positioning affect model outputs.
6. Performance of this model was not independently measured in this project.

