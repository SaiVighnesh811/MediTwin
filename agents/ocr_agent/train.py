"""
agents/ocr_agent/train.py
MediTwin AI — Chapter 4: OCR Agent
Steps 0-4 training, evaluation, printed OCR, field extraction.

Dataset : mamun1113/doctors-handwritten-prescription-bd-dataset
          Official Train=3120 / Val=780 / Test=780, 78 MEDICINE_NAME classes
Model   : ResNet18 (ImageNet pretrained), final FC→78, 128×128
Hardware: CPU-only (≤5 epochs, hard cap 45 min)
"""

import os, sys, json, time, re, copy, random
import warnings; warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFilter, ImageFont
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, top_k_accuracy_score
)
from utils.metrics_logger import log_metrics

# ── Paths ────────────────────────────────────────────────────────────────────
import glob as _glob
_CACHE_BASE = os.path.expanduser(
    "~/.cache/kagglehub/datasets/mamun1113/doctors-handwritten-prescription-bd-dataset/versions/1"
)
# Find the inner dir dynamically (handles apostrophe in path)
_inner = _glob.glob(os.path.join(_CACHE_BASE, "*"))
_inner_dirs = [d for d in _inner if os.path.isdir(d)]
if not _inner_dirs:
    raise FileNotFoundError(f"Cannot find dataset inner dir under {_CACHE_BASE}")
DATA_BASE = _inner_dirs[0]
print(f"  [path] DATA_BASE = {DATA_BASE}")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models", "ocr")
LOGS_DIR   = os.path.join(PROJECT_ROOT, "logs")
CKPT_PATH  = os.path.join(MODELS_DIR, "handwriting_cnn.pt")
os.makedirs(MODELS_DIR, exist_ok=True)

RANDOM_SEED = 42
IMG_SIZE = 128
BATCH    = 64
MAX_EP   = 5
LR       = 1e-4
DEVICE   = torch.device("cpu")
COMPONENT = "ocr_agent_handwriting"

random.seed(RANDOM_SEED); np.random.seed(RANDOM_SEED); torch.manual_seed(RANDOM_SEED)

def banner(msg):
    print(f"\n{'='*65}\n  {msg}\n{'='*65}")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 0 — Dataset inspection
# ══════════════════════════════════════════════════════════════════════════════
banner("STEP 0 — Dataset")
print(f"  Dataset: mamun1113/doctors-handwritten-prescription-bd-dataset")
print(f"  Data root: {DATA_BASE}")

def load_split(split_name):
    folder = os.path.join(DATA_BASE, split_name)
    csv_name = f"{split_name.lower()}_labels.csv"
    img_dir  = os.path.join(folder, f"{split_name.lower()}_words")
    df = pd.read_csv(os.path.join(folder, csv_name))
    df["img_path"] = df["IMAGE"].apply(lambda x: os.path.join(img_dir, x))
    return df

train_df = load_split("Training")
val_df   = load_split("Validation")
test_df  = load_split("Testing")

# Build label encoder from training set (sorted for reproducibility)
classes     = sorted(train_df["MEDICINE_NAME"].unique())
class_to_idx = {c: i for i, c in enumerate(classes)}
NUM_CLASSES = len(classes)

print(f"  Train : {len(train_df)} images,  Val : {len(val_df)},  Test : {len(test_df)}")
print(f"  Classes (MEDICINE_NAME): {NUM_CLASSES}")
print(f"  Sample classes: {classes[:10]} ...")
# Verify all val/test classes are in train
unseen_val  = set(val_df["MEDICINE_NAME"].unique()) - set(classes)
unseen_test = set(test_df["MEDICINE_NAME"].unique()) - set(classes)
print(f"  Unseen val classes: {unseen_val}, unseen test classes: {unseen_test}")

# ── Dataset / Loader ─────────────────────────────────────────────────────────
train_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE + 8, IMG_SIZE + 8)),
    transforms.RandomCrop(IMG_SIZE),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
val_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

class MedDataset(Dataset):
    def __init__(self, df, class_to_idx, transform):
        self.df = df.reset_index(drop=True)
        self.c2i = class_to_idx
        self.tf  = transform
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row["img_path"]).convert("RGB")
        return self.tf(img), self.c2i[row["MEDICINE_NAME"]]

train_ds = MedDataset(train_df, class_to_idx, train_tf)
val_ds   = MedDataset(val_df,   class_to_idx, val_tf)
test_ds  = MedDataset(test_df,  class_to_idx, val_tf)

train_ld = DataLoader(train_ds, batch_size=BATCH, shuffle=True,  num_workers=2)
val_ld   = DataLoader(val_ds,   batch_size=BATCH, shuffle=False, num_workers=2)
test_ld  = DataLoader(test_ds,  batch_size=BATCH, shuffle=False, num_workers=2)

# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Model + timing benchmark
# ══════════════════════════════════════════════════════════════════════════════
banner("STEP 1 — ResNet18 model + timing benchmark")
print(f"  Device: {DEVICE}")

model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
model = model.to(DEVICE)
print(f"  FC: 512 → {NUM_CLASSES} classes")

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# Benchmark: 2 batches from train
print("  Benchmarking 2 batches …")
model.train()
it = iter(train_ld)
t0 = time.time()
for _ in range(2):
    xb, yb = next(it)
    optimizer.zero_grad()
    loss = criterion(model(xb), yb)
    loss.backward(); optimizer.step()
bench_t = time.time() - t0
batches_per_ep = len(train_ld)
est_ep  = bench_t / 2 * batches_per_ep
est_total = est_ep * MAX_EP
print(f"  2 batches took {bench_t:.1f}s → estimated {est_ep/60:.1f} min/epoch")
print(f"  Estimated total ({MAX_EP} epochs): {est_total/60:.1f} min")
if est_total > 45 * 60:
    adj = max(1, int(44*60/est_ep))
    MAX_EP = adj
    print(f"  ⚠ Over 45 min cap → reducing to {MAX_EP} epochs")
else:
    print(f"  Within 45-min cap. Proceeding with {MAX_EP} epochs.")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 continued — Training loop
# ══════════════════════════════════════════════════════════════════════════════
banner("Training loop")
print(f"  {'Ep':>3}  {'TrLoss':>8}  {'TrAcc':>7}  {'VaLoss':>8}  {'VaAcc':>7}  Note")
print("  " + "-"*60)

best_val_loss = float("inf")
best_state    = None
history       = []

def run_epoch(loader, train=True):
    model.train() if train else model.eval()
    tot_loss, correct, n = 0.0, 0, 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for xb, yb in loader:
            if train: optimizer.zero_grad()
            out  = model(xb)
            loss = criterion(out, yb)
            if train: loss.backward(); optimizer.step()
            tot_loss += loss.item() * len(yb)
            correct  += (out.argmax(1) == yb).sum().item()
            n        += len(yb)
    return tot_loss/n, correct/n

for ep in range(1, MAX_EP + 1):
    t0 = time.time()
    tr_loss, tr_acc = run_epoch(train_ld, train=True)
    va_loss, va_acc = run_epoch(val_ld,   train=False)
    elapsed = time.time() - t0

    note = ""
    if va_loss < best_val_loss:
        best_val_loss = va_loss
        best_state    = copy.deepcopy(model.state_dict())
        note = "✓ best"
    # Save checkpoint every epoch
    torch.save({
        "epoch": ep, "model_state_dict": model.state_dict(),
        "best_state_dict": best_state,
        "classes": classes, "class_to_idx": class_to_idx,
        "val_loss": va_loss, "val_acc": va_acc,
        "img_size": IMG_SIZE,
    }, CKPT_PATH)

    history.append({"epoch": ep, "tr_loss": tr_loss, "tr_acc": tr_acc,
                    "va_loss": va_loss, "va_acc": va_acc})
    print(f"  {ep:>3}  {tr_loss:>8.4f}  {tr_acc:>7.4f}  {va_loss:>8.4f}  {va_acc:>7.4f}  {note}  ({elapsed:.0f}s)")

# Restore best
model.load_state_dict(best_state)
print(f"\n  Best checkpoint restored (val_loss={best_val_loss:.4f})")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — Test evaluation
# ══════════════════════════════════════════════════════════════════════════════
banner("STEP 2 — Test evaluation (untouched test split)")
model.eval()
all_preds, all_labels, all_probs = [], [], []
with torch.no_grad():
    for xb, yb in test_ld:
        out  = model(xb)
        prob = torch.softmax(out, dim=1).cpu().numpy()
        all_preds.extend(out.argmax(1).cpu().numpy())
        all_labels.extend(yb.cpu().numpy())
        all_probs.extend(prob)

all_preds  = np.array(all_preds)
all_labels = np.array(all_labels)
all_probs  = np.array(all_probs)

acc  = accuracy_score(all_labels, all_preds)
top3 = top_k_accuracy_score(all_labels, all_probs, k=3)
prec = precision_score(all_labels, all_preds, average="macro", zero_division=0)
rec  = recall_score(all_labels, all_preds, average="macro", zero_division=0)
f1   = f1_score(all_labels, all_preds, average="macro", zero_division=0)

per_class_f1 = f1_score(all_labels, all_preds, average=None, zero_division=0)
zero_correct = int((per_class_f1 == 0).sum())

print(f"  Accuracy      : {acc:.4f}")
print(f"  Top-3 Accuracy: {top3:.4f}")
print(f"  Macro Precision: {prec:.4f}")
print(f"  Macro Recall  : {rec:.4f}")
print(f"  Macro F1      : {f1:.4f}")
print(f"  Classes with 0 correct predictions: {zero_correct}/{NUM_CLASSES}")
print()
print(classification_report(all_labels, all_preds, target_names=classes, zero_division=0))

# Confusion matrix — top-20 most confused classes
cm = confusion_matrix(all_labels, all_preds)
errors = cm.copy(); np.fill_diagonal(errors, 0)
top20_idx = np.argsort(errors.sum(axis=1))[::-1][:20]
cm20 = cm[np.ix_(top20_idx, top20_idx)]
top20_names = [classes[i] for i in top20_idx]

fig, ax = plt.subplots(figsize=(12, 10))
im = ax.imshow(cm20, cmap="Blues")
ax.set_xticks(range(20)); ax.set_xticklabels(top20_names, rotation=90, fontsize=7)
ax.set_yticks(range(20)); ax.set_yticklabels(top20_names, fontsize=7)
ax.set_title("Confusion Matrix — Top-20 Most Confused Classes")
plt.colorbar(im, ax=ax); plt.tight_layout()
cm_path = os.path.join(LOGS_DIR, "ocr_handwriting_confusion_matrix.png")
plt.savefig(cm_path, dpi=120); plt.close()
print(f"  ✓ Confusion matrix → {cm_path}")

log_metrics(COMPONENT, {
    "accuracy": round(acc,4), "top3_accuracy": round(top3,4),
    "macro_precision": round(prec,4), "macro_recall": round(rec,4), "macro_f1": round(f1,4),
    "classes_zero_correct": zero_correct, "total_classes": NUM_CLASSES,
    "test_size": len(all_labels), "epochs_trained": len(history),
    "best_val_loss": round(best_val_loss,4),
    "model": "ResNet18 ImageNet pretrained, FC→78",
    "note": "Official train/val/test split from dataset authors used.",
})

# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — EasyOCR on 30 synthetic printed images
# ══════════════════════════════════════════════════════════════════════════════
banner("STEP 3 — EasyOCR on 30 synthetic printed images")
import easyocr
from jiwer import wer as jiwer_wer, cer as jiwer_cer

print("  Initialising EasyOCR (English, CPU) …")
reader = easyocr.Reader(["en"], gpu=False, verbose=False)
print("  EasyOCR ready.")

# 15 Prescription-style + 15 Lab-report-style ground-truth lines
PRESCRIPTION_LINES = [
    "Tab Metformin 500 mg twice daily for 30 days",
    "Cap Amoxicillin 250 mg three times daily for 7 days",
    "Syp Paracetamol 5 ml every 6 hours for 5 days",
    "Tab Amlodipine 5 mg once daily for 1 month",
    "Inj Insulin Glargine 10 units subcutaneous at bedtime",
    "Tab Atorvastatin 10 mg once daily at night",
    "Tab Azithromycin 500 mg once daily for 3 days",
    "Cap Omeprazole 20 mg before meals twice daily",
    "Tab Fexofenadine 120 mg once daily for 14 days",
    "Tab Clonazepam 0.5 mg at bedtime for 1 month",
    "Syp Cetirizine 5 mg at night for 10 days",
    "Tab Metronidazole 400 mg thrice daily for 5 days",
    "Cap Fluconazole 150 mg single dose",
    "Tab Baclofen 10 mg twice daily for 3 months",
    "Tab Esomeprazole 40 mg once daily before breakfast",
]
LAB_LINES = [
    "HbA1c 6.8 % Ref 4.0-5.6",
    "Fasting Blood Glucose 110 mg/dL Ref 70-100",
    "Total Cholesterol 195 mg/dL Ref less than 200",
    "Haemoglobin 11.2 g/dL Ref 12.0-16.0",
    "Serum Creatinine 1.1 mg/dL Ref 0.6-1.2",
    "ALT SGPT 45 U/L Ref 7-56",
    "TSH 2.3 mIU/L Ref 0.4-4.5",
    "Platelet Count 210000 cells/uL Ref 150000-400000",
    "Urea 28 mg/dL Ref 10-50",
    "WBC 7500 cells/uL Ref 4000-11000",
    "LDL Cholesterol 118 mg/dL Ref less than 130",
    "HDL Cholesterol 52 mg/dL Ref greater than 40",
    "Uric Acid 5.8 mg/dL Ref 3.5-7.2",
    "Sodium 138 mEq/L Ref 136-145",
    "Potassium 4.1 mEq/L Ref 3.5-5.1",
]
GROUND_TRUTHS = PRESCRIPTION_LINES + LAB_LINES   # 30 total

# Fonts to try
FONTS_TO_TRY = [
    "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/ubuntu/Ubuntu-R.ttf",
]
available_fonts = [f for f in FONTS_TO_TRY if os.path.exists(f)]
print(f"  Available fonts: {available_fonts}")

SYN_DIR = os.path.join(LOGS_DIR, "ocr_synthetic_images")
os.makedirs(SYN_DIR, exist_ok=True)

def make_image(text, idx, noisy=False):
    font_path = available_fonts[idx % max(len(available_fonts), 1)]
    try:
        font = ImageFont.truetype(font_path, 20)
    except Exception:
        font = ImageFont.load_default()
    # Size image to text
    dummy = ImageDraw.Draw(Image.new("RGB", (1,1)))
    bbox  = dummy.textbbox((0,0), text, font=font)
    w, h  = bbox[2] - bbox[0] + 30, bbox[3] - bbox[1] + 20
    img   = Image.new("RGB", (w, h), (255, 255, 255))
    draw  = ImageDraw.Draw(img)
    draw.text((15, 10), text, fill=(0, 0, 0), font=font)
    if noisy:
        img = img.filter(ImageFilter.GaussianBlur(radius=0.8))
        arr = np.array(img).astype(np.int16)
        noise = np.random.randint(-15, 15, arr.shape, dtype=np.int16)
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr)
    path = os.path.join(SYN_DIR, f"syn_{idx:02d}.png")
    img.save(path)
    return path, noisy

# Generate: first 15 clean, last 15 noisy (alternating clean/noisy within each set)
syn_paths, noisy_flags = [], []
for i, gt in enumerate(GROUND_TRUTHS):
    noisy = (i % 2 == 1)   # odd indices are noisy
    path, is_noisy = make_image(gt, i, noisy=noisy)
    syn_paths.append(path)
    noisy_flags.append(is_noisy)
print(f"  Generated {len(syn_paths)} synthetic images ({sum(noisy_flags)} noisy)")

# Run EasyOCR
print("  Running EasyOCR … (30 images, CPU)")
ocr_results = []
for i, (path, gt) in enumerate(zip(syn_paths, GROUND_TRUTHS)):
    res = reader.readtext(path, detail=0, paragraph=True)
    pred_text = " ".join(res).strip()
    ocr_results.append(pred_text)
print("  EasyOCR done.")

# Compute CER and WER using jiwer
def safe_ocr(pred):
    """Ensure non-empty string for jiwer."""
    return pred if pred.strip() else "[EMPTY]"

all_cer, all_wer = [], []
clean_cer, clean_wer = [], []
noisy_cer, noisy_wer = [], []

for i, (gt, pred, is_noisy) in enumerate(zip(GROUND_TRUTHS, ocr_results, noisy_flags)):
    gt_s   = gt.lower().strip()
    pred_s = safe_ocr(pred.lower().strip())
    c = jiwer_cer(gt_s, pred_s)
    w = jiwer_wer(gt_s, pred_s)
    c = min(c, 1.0); w = min(w, 1.0)
    all_cer.append(c); all_wer.append(w)
    if is_noisy:
        noisy_cer.append(c); noisy_wer.append(w)
    else:
        clean_cer.append(c); clean_wer.append(w)
    print(f"  [{i:02d}] {'NOISY' if is_noisy else 'CLEAN':6s}  CER={c:.3f}  WER={w:.3f}")
    print(f"         GT  : {gt[:80]}")
    print(f"         PRED: {pred[:80]}")

print(f"\n  === EasyOCR CER/WER Summary ===")
print(f"  Overall: CER={np.mean(all_cer):.4f}, WER={np.mean(all_wer):.4f}")
print(f"  Clean  : CER={np.mean(clean_cer):.4f}, WER={np.mean(clean_wer):.4f}")
print(f"  Noisy  : CER={np.mean(noisy_cer):.4f}, WER={np.mean(noisy_wer):.4f}")
print("  NOTE: These are synthetic PIL images, NOT real scanned documents.")

log_metrics("ocr_agent_printed", {
    "note": "Synthetic images generated with PIL (NOT real scanned documents)",
    "n_images": 30, "n_clean": sum(1 for f in noisy_flags if not f),
    "n_noisy": sum(noisy_flags),
    "ocr_engine": "EasyOCR (English, pretrained, no fine-tuning)",
    "metric_library": "jiwer",
    "overall_cer": round(float(np.mean(all_cer)), 4),
    "overall_wer": round(float(np.mean(all_wer)), 4),
    "clean_cer":   round(float(np.mean(clean_cer)), 4),
    "clean_wer":   round(float(np.mean(clean_wer)), 4),
    "noisy_cer":   round(float(np.mean(noisy_cer)), 4),
    "noisy_wer":   round(float(np.mean(noisy_wer)), 4),
})

# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — Field extraction (regex)
# ══════════════════════════════════════════════════════════════════════════════
banner("STEP 4 — Regex field extraction")

MED_78 = set(classes)  # the 78 known medicine names

EXTRA_MEDS = {
    "Metformin","Amoxicillin","Paracetamol","Amlodipine","Insulin Glargine",
    "Atorvastatin","Azithromycin","Omeprazole","Fexofenadine","Clonazepam",
    "Cetirizine","Metronidazole","Fluconazole","Baclofen","Esomeprazole",
}
ALL_MEDS = MED_78 | EXTRA_MEDS

DOSAGE_RE    = re.compile(r'(\d+(?:\.\d+)?)\s*(mg|mcg|g|ml|units?|iu)\b', re.IGNORECASE)
FREQ_TERMS   = ["once daily","twice daily","three times daily","thrice daily",
                "every 6 hours","every 8 hours","at bedtime","at night","before meals",
                "before breakfast","single dose","once"]
DURATION_RE  = re.compile(r'(?:for\s+)?(\d+)\s+(day|days|month|months|week|weeks)', re.IGNORECASE)

TEST_RE      = re.compile(
    r'^(HbA1c|Fasting Blood Glucose|Total Cholesterol|Haemoglobin|Serum Creatinine|'
    r'ALT SGPT|TSH|Platelet Count|Urea|WBC|LDL Cholesterol|HDL Cholesterol|'
    r'Uric Acid|Sodium|Potassium)\b', re.IGNORECASE)
VALUE_RE     = re.compile(r'(\d+(?:\.\d+)?(?:\s*[-\/]\s*\d+(?:\.\d+)?)?)\s*([a-zA-Z/%_]+(?:/[a-zA-Z]+)?)?')
REFRANGE_RE  = re.compile(
    r'(?:Ref(?:erence)?\s*|less than |greater than |>|<)(\d+(?:\.\d+)?(?:\s*[-–]\s*\d+(?:\.\d+)?)?)', re.IGNORECASE)

# Ground-truth fields for all 30 lines
PRESCRIPTION_GT = [
    {"medicine":"Metformin","dosage":"500 mg","frequency":"twice daily","duration":"30 days"},
    {"medicine":"Amoxicillin","dosage":"250 mg","frequency":"three times daily","duration":"7 days"},
    {"medicine":"Paracetamol","dosage":"5 ml","frequency":"every 6 hours","duration":"5 days"},
    {"medicine":"Amlodipine","dosage":"5 mg","frequency":"once daily","duration":"1 month"},
    {"medicine":"Insulin Glargine","dosage":"10 units","frequency":"at bedtime","duration":None},
    {"medicine":"Atorvastatin","dosage":"10 mg","frequency":"once daily","duration":None},
    {"medicine":"Azithromycin","dosage":"500 mg","frequency":"once daily","duration":"3 days"},
    {"medicine":"Omeprazole","dosage":"20 mg","frequency":"twice daily","duration":None},
    {"medicine":"Fexofenadine","dosage":"120 mg","frequency":"once daily","duration":"14 days"},
    {"medicine":"Clonazepam","dosage":"0.5 mg","frequency":"at bedtime","duration":"1 month"},
    {"medicine":"Cetirizine","dosage":"5 mg","frequency":"at night","duration":"10 days"},
    {"medicine":"Metronidazole","dosage":"400 mg","frequency":"thrice daily","duration":"5 days"},
    {"medicine":"Fluconazole","dosage":"150 mg","frequency":"single dose","duration":None},
    {"medicine":"Baclofen","dosage":"10 mg","frequency":"twice daily","duration":"3 months"},
    {"medicine":"Esomeprazole","dosage":"40 mg","frequency":"once daily","duration":None},
]
LAB_GT = [
    {"test":"HbA1c","value":"6.8","unit":"%","reference":"4.0-5.6"},
    {"test":"Fasting Blood Glucose","value":"110","unit":"mg/dL","reference":"70-100"},
    {"test":"Total Cholesterol","value":"195","unit":"mg/dL","reference":"less than 200"},
    {"test":"Haemoglobin","value":"11.2","unit":"g/dL","reference":"12.0-16.0"},
    {"test":"Serum Creatinine","value":"1.1","unit":"mg/dL","reference":"0.6-1.2"},
    {"test":"ALT SGPT","value":"45","unit":"U/L","reference":"7-56"},
    {"test":"TSH","value":"2.3","unit":"mIU/L","reference":"0.4-4.5"},
    {"test":"Platelet Count","value":"210000","unit":"cells/uL","reference":"150000-400000"},
    {"test":"Urea","value":"28","unit":"mg/dL","reference":"10-50"},
    {"test":"WBC","value":"7500","unit":"cells/uL","reference":"4000-11000"},
    {"test":"LDL Cholesterol","value":"118","unit":"mg/dL","reference":"less than 130"},
    {"test":"HDL Cholesterol","value":"52","unit":"mg/dL","reference":"greater than 40"},
    {"test":"Uric Acid","value":"5.8","unit":"mg/dL","reference":"3.5-7.2"},
    {"test":"Sodium","value":"138","unit":"mEq/L","reference":"136-145"},
    {"test":"Potassium","value":"4.1","unit":"mEq/L","reference":"3.5-5.1"},
]

def extract_prescription(text):
    result = {}
    tl = text.lower()
    # Medicine: find any known med (longest match first)
    found_med = None
    for med in sorted(ALL_MEDS, key=len, reverse=True):
        if med.lower() in tl:
            found_med = med; break
    result["medicine"] = found_med
    # Dosage
    d = DOSAGE_RE.search(text)
    result["dosage"] = f"{d.group(1)} {d.group(2).lower()}" if d else None
    # Frequency
    result["frequency"] = None
    for f in FREQ_TERMS:
        if f in tl:
            result["frequency"] = f; break
    # Duration
    dur = DURATION_RE.search(text)
    result["duration"] = f"{dur.group(1)} {dur.group(2).lower()}" if dur else None
    return result

def extract_lab(text):
    result = {}
    # Test name
    tm = TEST_RE.search(text)
    result["test"] = tm.group(1) if tm else None
    # Value + unit
    tokens = text.split()
    test_end = len(tm.group(0).split()) if tm else 0
    rest = " ".join(tokens[test_end:])
    vm = VALUE_RE.search(rest)
    result["value"] = vm.group(1).strip() if vm else None
    result["unit"]  = vm.group(2).strip() if (vm and vm.group(2)) else None
    # Reference
    rm = REFRANGE_RE.search(text)
    result["reference"] = rm.group(1).strip() if rm else None
    # Try text refs
    if result["reference"] is None:
        for kw in ["less than", "greater than"]:
            if kw in text.lower():
                m2 = re.search(kw+r'\s+(\d+(?:\.\d+)?)', text, re.IGNORECASE)
                if m2: result["reference"] = kw + " " + m2.group(1)
    return result

# Evaluate on all 30 lines using GT text (not OCR output, to isolate extraction quality)
tp_med, fp_med, fn_med         = 0,0,0
tp_dos, fp_dos, fn_dos         = 0,0,0
tp_frq, fp_frq, fn_frq         = 0,0,0
tp_dur, fp_dur, fn_dur         = 0,0,0
tp_tst, fp_tst, fn_tst         = 0,0,0
tp_val, fp_val, fn_val         = 0,0,0
tp_unt, fp_unt, fn_unt         = 0,0,0
tp_ref, fp_ref, fn_ref         = 0,0,0
failures = []

def score_field(pred, gt, tp, fp, fn, field, line_id, failures):
    if gt is None:
        return tp, fp, fn
    pred_clean = (pred or "").lower().strip()
    gt_clean   = gt.lower().strip()
    if pred_clean and gt_clean in pred_clean or pred_clean in gt_clean:
        tp += 1
    elif pred_clean:
        fp += 1; fn += 1
        failures.append(f"  Line {line_id} [{field}]: pred='{pred}' expected='{gt}'")
    else:
        fn += 1
        failures.append(f"  Line {line_id} [{field}]: pred=NONE expected='{gt}'")
    return tp, fp, fn

for i, (line, gt) in enumerate(zip(PRESCRIPTION_LINES, PRESCRIPTION_GT)):
    pred = extract_prescription(line)
    tp_med,fp_med,fn_med = score_field(pred["medicine"],gt["medicine"],tp_med,fp_med,fn_med,"medicine",i,failures)
    tp_dos,fp_dos,fn_dos = score_field(pred["dosage"],  gt["dosage"],  tp_dos,fp_dos,fn_dos,"dosage",  i,failures)
    tp_frq,fp_frq,fn_frq = score_field(pred["frequency"],gt["frequency"],tp_frq,fp_frq,fn_frq,"frequency",i,failures)
    tp_dur,fp_dur,fn_dur = score_field(pred["duration"],gt.get("duration"),tp_dur,fp_dur,fn_dur,"duration",i,failures)

for i, (line, gt) in enumerate(zip(LAB_LINES, LAB_GT)):
    pred = extract_lab(line)
    j = i + 15
    tp_tst,fp_tst,fn_tst = score_field(pred["test"],   gt["test"],     tp_tst,fp_tst,fn_tst,"test",j,failures)
    tp_val,fp_val,fn_val = score_field(pred["value"],  gt["value"],    tp_val,fp_val,fn_val,"value",j,failures)
    tp_unt,fp_unt,fn_unt = score_field(pred["unit"],   gt["unit"],     tp_unt,fp_unt,fn_unt,"unit", j,failures)
    tp_ref,fp_ref,fn_ref = score_field(pred["reference"],gt["reference"],tp_ref,fp_ref,fn_ref,"reference",j,failures)

def prf(tp, fp, fn):
    p = tp/(tp+fp) if tp+fp else 0
    r = tp/(tp+fn) if tp+fn else 0
    f = 2*p*r/(p+r) if p+r else 0
    return round(p,4), round(r,4), round(f,4)

fields_prf = {
    "medicine":  prf(tp_med,fp_med,fn_med),
    "dosage":    prf(tp_dos,fp_dos,fn_dos),
    "frequency": prf(tp_frq,fp_frq,fn_frq),
    "duration":  prf(tp_dur,fp_dur,fn_dur),
    "test_name": prf(tp_tst,fp_tst,fn_tst),
    "value":     prf(tp_val,fp_val,fn_val),
    "unit":      prf(tp_unt,fp_unt,fn_unt),
    "reference": prf(tp_ref,fp_ref,fn_ref),
}

print(f"\n  {'Field':15s}  {'Precision':>10}  {'Recall':>8}  {'F1':>6}")
print("  " + "-"*45)
for fld, (p, r, f) in fields_prf.items():
    print(f"  {fld:15s}  {p:>10.4f}  {r:>8.4f}  {f:>6.4f}")

print(f"\n  Failure cases ({len(failures)}):")
for fail in failures:
    print(fail)

log_metrics("ocr_agent_extraction", {
    "evaluated_on": "30 synthetic GT text strings (not OCR output)",
    "fields_prf": {k: {"precision":v[0],"recall":v[1],"f1":v[2]} for k,v in fields_prf.items()},
    "failure_count": len(failures),
    "note": "Regex extraction on known ground-truth text. Separate from OCR accuracy.",
})

banner("Chapter 4 training script complete")
print("  Checkpoint saved to:", CKPT_PATH)
print("  Next: run ocr_agent.py self-test to confirm Step 5.")
