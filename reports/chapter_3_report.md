# MediTwin AI — Chapter 3: Medical Image Agent Final Report

> **Status**: Completed (Pretrained Transfer Learning Model)  
> **Disclaimer**: **NOT FOR CLINICAL USE**. Screening aid prototype only. All outputs must be verified by a licensed medical professional.

---

## 1. Executive Summary

The Medical Image Agent provides automated binary screening of chest X-ray images (AP view) to detect signs of pediatric pneumonia (`NORMAL` vs `PNEUMONIA`). Rather than completing a resource-intensive local training run, the agent adopts a high-quality, ImageNet-pretrained ResNet18 binary classifier published by Luana Carolina Reis and Jakub Błaszczyk under CC BY 4.0 (`luanacarolina/pneumonia-chest-xray-classifier`). An earlier scratch training attempt (documented in `reports/chapter_3_DRAFT.md`) was abandoned after epoch 1.

The pretrained model was rigorously evaluated by us on the official, untouched 624-image test set from `paultimothymooney/chest-xray-pneumonia` on host CPU.

---

## 2. Model & Checkpoint Architecture

- **Source**: [https://huggingface.co/luanacarolina/pneumonia-chest-xray-classifier](https://huggingface.co/luanacarolina/pneumonia-chest-xray-classifier)
- **Base Architecture**: `torchvision.models.resnet18`
- **Classifier Head**: Single logit linear head (`nn.Linear(512, 1)`) with Sigmoid activation for binary classification (`P(Pneumonia)`).
- **Original Checkpoint**: `models/medical_image/resnet18_xray.pt` (129 MB, SHA-256: `c1f4b6de43f5c6f4844b4e7d5215c8e78a468ff1f75edccc3b26d859a8f40767`)
- **Optimized Deployment Weights**: `models/medical_image/resnet18_xray_weights.pt` (42.71 MB, trimmed down to `model_state_dict` for fast startup and low memory footprint).
- **Interpretability**: Single-logit Grad-CAM on `model.layer4[-1]` producing overlay heatmaps saved to `logs/`.

---

## 3. Independent Evaluation on the Official Test Set (624 Images)

The test set comprises 234 `NORMAL` images and 390 `PNEUMONIA` images. All inferences were run with decision threshold 0.5 under `torch.inference_mode()` on CPU.

| Metric | Measured Value | Details |
|---|---|---|
| **Accuracy** | **84.78%** | 529 / 624 correct predictions |
| **Precision** | **81.18%** | $TP / (TP + FP) = 384 / (384 + 89)$ |
| **Recall / Sensitivity** | **98.46%** | $TP / (TP + FN) = 384 / (384 + 6)$ (missed only 6 pneumonia cases) |
| **Specificity** | **61.97%** | $TN / (TN + FP) = 145 / (145 + 89)$ |
| **F1-Score** | **88.99%** | Harmonic mean of precision and recall |
| **ROC-AUC** | **94.73%** | Threshold-independent discriminative ranking quality |

### Confusion Matrix (Threshold = 0.5)

| True \ Pred | Predicted NORMAL | Predicted PNEUMONIA | Total |
|---|---|---|---|
| **True NORMAL** | **145** (TN) | **89** (FP) | 234 |
| **True PNEUMONIA** | **6** (FN) | **384** (TP) | 390 |
| **Total** | 151 | 473 | 624 |

Artifacts generated:
- Confusion Matrix Plot: `logs/medical_image_confusion_matrix.png`
- ROC Curve Plot: `logs/medical_image_roc.png`
- Per-Image Test Predictions: `logs/medical_image_test_predictions.csv`
- Metrics JSON: `logs/medical_image_agent_metrics.json`

---

## 4. Resource & Runtime Profile

- **RAM Footprint**:
  - Baseline RSS: ~495 MB
  - RSS after loading slim weights (42.71 MB): ~735 MB ($\Delta$ +57.28 MB)
  - RSS during active inference: ~761 MB
  - RSS during Grad-CAM generation: ~827 MB
- **Latency**:
  - Pure inference: **~173 ms / image** (CPU)
  - Inference with Grad-CAM heatmap overlay: **~1.09 s / image**

---

## 5. Honest Limitations & Clinical Nuances

1. **High False Positive Rate (Specificity 61.97%)**:
   The model flagged 89 out of 234 normal chest X-rays as pneumonia. While this conservatism gives the model high clinical screening sensitivity (98.46% recall), it means roughly 38% of healthy individuals screened would receive a false alarm requiring follow-up physician review.
2. **Single-Hospital Pediatric Dataset**:
   Trained exclusively on pediatric patients (ages 1–5) at Guangzhou Women and Children's Medical Center. Generalization to adults, other scanner manufacturers, or different clinical settings is completely unvalidated.
3. **Discrepancy with Author's Headline Metrics**:
   The authors' reported headline numbers (Accuracy 85.6%, F1 89.6%) correspond to a 2-model ensemble (ResNet18 + DenseNet121). Standalone ResNet18 matches our measured 84.78% accuracy.
4. **Abandonment of Scratch Training**:
   The initial attempt to train from scratch locally was halted after epoch 1 because of CPU training bottlenecks and the availability of pre-converged, peer-evaluated model checkpoints.

