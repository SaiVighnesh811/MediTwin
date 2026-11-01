# MediTwin AI — Chapter 4: OCR Agent Report

## What Was Built

Three components delivered in a single `train.py` (Steps 0–4) and a reusable `ocr_agent.py` (Step 5):

1. **Handwriting Classifier** — ResNet18 (ImageNet pretrained), 78-class closed-set, 128×128 input
2. **Printed-text OCR** — EasyOCR (English, pretrained, no fine-tuning) on 30 synthetic PIL images
3. **Field Extraction** — Regex rules for prescription fields (medicine, dosage, frequency, duration) and lab-report fields (test, value, unit, reference)

---

## Step 0 — Dataset

**Dataset**: `mamun1113/doctors-handwritten-prescription-bd-dataset`

| Split | Images | Classes |
|-------|--------|---------|
| Training (official) | 3120 | 78 |
| Validation (official) | 780 | 78 |
| Testing (official — untouched) | 780 | 78 |
| **Total** | **4680** | **78** |

- CSV columns: `IMAGE`, `MEDICINE_NAME` (target, 78 classes), `GENERIC_NAME` (15 generics, not used)
- Pre-split by dataset authors — used as-is, no leakage
- All 78 classes appear in all three splits

---

## Step 1 — Hardware + Model

- **Device**: CPU only (4 cores), no GPU
- **Model**: ResNet18 ImageNet pretrained, final FC replaced: 512 → 78
- **Input**: 128×128, grayscale→3-channel, ImageNet normalisation
- **Augmentation (train only)**: RandomCrop(128 from 136), RandomRotation(±10°), ColorJitter(brightness=0.2, contrast=0.2). No flips.
- **Loss**: CrossEntropyLoss (no class weighting — dataset is balanced: 40 images/class)
- **Optimiser**: Adam lr=1e-4
- **Benchmark**: 2 batches → estimated **8.8 min/epoch**, 5 epochs = 44 min (within 45-min cap)

### Per-Epoch Training Results (real, from log)

| Ep | TrLoss | TrAcc | VaLoss | VaAcc | Note |
|----|--------|-------|--------|-------|------|
| 1 | 3.9280 | 13.75% | 3.5890 | 23.08% | ✓ best |
| 2 | 2.5132 | 52.56% | 2.1111 | 59.23% | ✓ best |
| 3 | 1.4853 | 76.83% | 1.3609 | 74.49% | ✓ best |
| 4 | 0.8343 | 89.13% | 0.8911 | 84.74% | ✓ best |
| 5 | 0.4926 | **94.10%** | 0.6563 | **88.46%** | ✓ best |

Best checkpoint: epoch 5 (val_loss=0.6563). Checkpoint saved after every epoch.

---

## Step 2 — Test Evaluation (untouched, 780 images)

| Metric | Value |
|--------|-------|
| Accuracy | **75.77%** |
| Top-3 Accuracy | **87.31%** |
| Macro Precision | 79.29% |
| Macro Recall | 75.77% |
| Macro F1 | **74.99%** |
| Classes with 0 correct | **0 / 78** |

> Note: Every one of the 78 classes has at least 1 correct prediction on the test set.

Confusion matrix of top-20 most confused classes saved to `logs/ocr_handwriting_confusion_matrix.png`. Most confusions occur between visually similar names (e.g. Esoral/Esonix, Montair/Montene/Montex).

---

## Step 3 — EasyOCR Printed Text (30 Synthetic Images)

- **Engine**: EasyOCR `en`, pretrained, no fine-tuning
- **Images**: 30 synthetic PIL images (15 prescription-style, 15 lab-style); 15 clean + 15 with Gaussian blur + pixel noise
- **Metric library**: `jiwer`

| Set | CER | WER |
|-----|-----|-----|
| Overall (30) | **0.0122** | **0.0509** |
| Clean (15) | 0.0078 | 0.0244 |
| Noisy (15) | 0.0166 | 0.0774 |

> ⚠ **These are synthetic PIL images, NOT real scanned documents.** Performance on real prescriptions or lab reports will differ.

---

## Step 4 — Field Extraction (Regex)

Evaluated on the 30 synthetic ground-truth text strings (not OCR output, to isolate extraction quality).

| Field | Precision | Recall | F1 |
|-------|-----------|--------|----|
| medicine | 1.0000 | 1.0000 | 1.0000 |
| dosage | 1.0000 | 1.0000 | 1.0000 |
| frequency | 1.0000 | 1.0000 | 1.0000 |
| duration | 1.0000 | 1.0000 | 1.0000 |
| test_name | 1.0000 | 1.0000 | 1.0000 |
| value | 1.0000 | 1.0000 | 1.0000 |
| unit | 1.0000 | 1.0000 | 1.0000 |
| reference | 1.0000 | 1.0000 | 1.0000 |

**Failure cases: 0.** All 8 fields extracted correctly from all 30 ground-truth lines.

> Note: These rules match the structured synthetic inputs exactly. Real OCR output will have character errors that can break field detection (e.g. `mgldL` instead of `mg/dL` observed in noisy images).

---

## Step 5 — Reusable Module (`ocr_agent.py`)

Self-test results (confirmed):

```
read_printed(syn_00.png)  →  "Tab Metformin 500 mg twice daily for 30 days"  ✓
extract_fields(...)       →  {medicine:Metformin, dosage:500 mg, frequency:twice daily, duration:30 day}  ✓
extract_fields(lab line)  →  {test:HbA1c, value:6.8, unit:%, reference:4.0-5.6}  ✓
classify_handwritten_medicine()  →  loads model correctly (top-3 API working)  ✓
```

---

## Files Created

```
agents/ocr_agent/train.py
agents/ocr_agent/ocr_agent.py
models/ocr/handwriting_cnn.pt           (86 MB, ResNet18 state_dict)
logs/ocr_handwriting_confusion_matrix.png
logs/ocr_synthetic_images/              (30 PNGs)
logs/ocr_agent_handwriting_metrics.json
logs/ocr_agent_printed_metrics.json
logs/ocr_agent_extraction_metrics.json
reports/chapter_4_report.md
```

---

## Limitations

1. **Closed-set handwriting classifier**: Only the 78 Bangladeshi medicine names from this dataset. Any out-of-vocabulary medicine word will be misclassified as one of the 78.
2. **Single-source dataset**: All handwriting images from one collection (BD medical context). Recognition of handwriting styles from other regions or doctors may be significantly worse.
3. **EasyOCR on synthetic only**: No real scanned prescription or lab report was tested. Real documents have variable fonts, poor scan quality, skew, and background noise not reflected in synthetic PIL images.
4. **Field extraction is rule-based**: Works well on structured synthetic text. Robustness to real OCR errors (character substitutions, merged words) is untested.
5. **5-epoch CPU limit**: A longer training run or fine-tuned feature extractor would likely improve handwriting accuracy beyond 75.77%.

---

## Step 4b — E2E Field Extraction (OCR output → extract_fields)

**What this measures**: EasyOCR is run on the 30 synthetic images; `extract_fields()` is then called on the OCR text — not on the ground truth. This reflects real pipeline failure modes.

### Failure root-cause analysis

| Failure type | Images | Fields affected |
|---|---|---|
| EasyOCR drops entire multi-word lab test name on noisy image (e.g. `"Fasting Blood Glucose"`, `"Total Cholesterol"` corrupted beyond recognition) | imgs 15–17, 19, 23, 25–27 (8 noisy lab images) | test, value, unit, reference — all 4 fields fail together |
| Unit OCR substitution (readable image but character swap) | imgs 21, 22, 24, 29 | unit only |
| Duration OCR: `"1 month"` read as `"1 month"` but article `"for"` corrupted → regex missed | img 03 (noisy) | duration |

### BEFORE rule hardening (v1 — original rules on OCR output)

| Field | P | R | F1 |
|-------|---|---|-----|
| medicine | 1.0000 | 1.0000 | 1.0000 |
| dosage | 1.0000 | 1.0000 | 1.0000 |
| frequency | 1.0000 | 1.0000 | 1.0000 |
| duration | 1.0000 | 0.9000 | 0.9474 |
| test | 1.0000 | 0.4667 | 0.6364 |
| value | 1.0000 | 0.4667 | 0.6364 |
| unit | 0.4286 | 0.2000 | 0.2727 |
| reference | 1.0000 | 0.4667 | 0.6364 |
| **Macro** | **0.9286** | **0.6875** | **0.7662** |

Clean macro F1=0.8750 | Noisy macro F1=0.6361 | Total failures=37

### AFTER rule hardening (v2 — unit normaliser added)

**Rules changed** (unit normaliser only — applied to OCR output before comparison):
1. `mgldL` / `mgl` / `mgl/dL` → `mg/dL`
2. `mEqIL` / `MEq/L` / `meqll` → `mEq/L`
3. `cells/uUL` → `cells/uL`

> ⚠ **Labelled "tuned on same 30 lines"** — the normaliser was written after observing the v1 failures on these exact images.

| Field | P | R | F1 | Δ F1 |
|-------|---|---|-----|------|
| medicine | 1.0000 | 1.0000 | 1.0000 | — |
| dosage | 1.0000 | 1.0000 | 1.0000 | — |
| frequency | 1.0000 | 1.0000 | 1.0000 | — |
| duration | 1.0000 | 0.9000 | 0.9474 | — |
| test | 1.0000 | 0.4667 | 0.6364 | — |
| value | 1.0000 | 0.4667 | 0.6364 | — |
| unit | 0.7143 | 0.3333 | **0.4545** | **+0.1818** |
| reference | 1.0000 | 0.4667 | 0.6364 | — |
| **Macro** | **0.9643** | **0.7042** | **0.7889** | **+0.0227** |

Clean macro F1=0.8958 | Noisy macro F1=0.6611 | Total failures=35

### Remaining failures after v2

- **8 lab images (noisy)** where EasyOCR drops the entire test name → all 4 fields (test/value/unit/reference) return `None`. Cannot be fixed by rule tuning alone; would require OCR robustness improvement (e.g. image pre-processing, larger EasyOCR model).
- **`cellsluL`** (img 22, clean) — `l` inserted before `u`; not covered by v2 normaliser.
- **`mIUIL`** (img 21, noisy) — `I→/` substitution in `mIU/L`; not covered.
- **`"1 month"` duration** (img 03, noisy) — OCR reads `"1 month"` but surrounding context corrupted; regex context match fails.

Logged under `"ocr_agent_extraction_e2e"`.

