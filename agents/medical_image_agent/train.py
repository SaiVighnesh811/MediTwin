"""
agents/medical_image_agent/train.py
MediTwin AI — Chapter 3: Medical Image Agent
Chest X-ray binary classification: NORMAL vs PNEUMONIA

Hardware: CPU-only (no GPU available) → ResNet18, 224×224, max 8 epochs
Dataset : paultimothymooney/chest-xray-pneumonia (5856 images)
Split   : Merge official train+val. Patient-level 85/15 split for PNEUMONIA
          (person IDs from filenames). Random 85/15 for NORMAL (no IDs).
          Official test folder (624 images) is NEVER touched during training.
"""

import os, sys, glob, re, json, time, copy, random
import warnings; warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, balanced_accuracy_score,
    matthews_corrcoef, confusion_matrix, classification_report
)
from utils.metrics_logger import log_metrics

# ── Constants ──────────────────────────────────────────────────────────────────
DATA_ROOT   = "/home/sai/.cache/kagglehub/datasets/paultimothymooney/chest-xray-pneumonia/versions/2/chest_xray"
MODELS_DIR  = os.path.join(PROJECT_ROOT, "models", "medical_image")
LOGS_DIR    = os.path.join(PROJECT_ROOT, "logs")
os.makedirs(MODELS_DIR, exist_ok=True)

RANDOM_SEED  = 42
IMG_SIZE     = 224
BATCH_SIZE   = 32
MAX_EPOCHS   = 8
LR           = 1e-4
PATIENCE     = 3          # early stopping on val loss
COMPONENT    = "medical_image_agent"
CLASS_NAMES  = ["NORMAL", "PNEUMONIA"]   # 0, 1

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def banner(msg):
    print(f"\n{'='*65}\n  {msg}\n{'='*65}")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 0 — Gather image paths
# ══════════════════════════════════════════════════════════════════════════════
banner("STEP 0 — Dataset overview")

def get_images(split, cls):
    pattern = os.path.join(DATA_ROOT, split, cls, "*")
    return sorted([f for f in glob.glob(pattern)
                   if f.lower().endswith((".jpeg", ".jpg", ".png"))])

train_normal    = get_images("train", "NORMAL")
train_pneumonia = get_images("train", "PNEUMONIA")
val_normal      = get_images("val",   "NORMAL")
val_pneumonia   = get_images("val",   "PNEUMONIA")
test_normal     = get_images("test",  "NORMAL")
test_pneumonia  = get_images("test",  "PNEUMONIA")

all_normal    = train_normal    + val_normal
all_pneumonia = train_pneumonia + val_pneumonia

print(f"  Official train   : NORMAL={len(train_normal)}, PNEUMONIA={len(train_pneumonia)}")
print(f"  Official val     : NORMAL={len(val_normal)},   PNEUMONIA={len(val_pneumonia)}")
print(f"  Official test    : NORMAL={len(test_normal)},  PNEUMONIA={len(test_pneumonia)}")
print(f"  Merged pool      : NORMAL={len(all_normal)},  PNEUMONIA={len(all_pneumonia)}")
print(f"  Test set kept untouched: {len(test_normal)+len(test_pneumonia)} images")
print(f"  Device: {DEVICE}")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — Patient-level split
# ══════════════════════════════════════════════════════════════════════════════
banner("STEP 2 — Stratified 85/15 split with patient-level grouping for PNEUMONIA")

def extract_person_id(filepath):
    """Extract person ID from filenames like person12_bacteria_34.jpeg."""
    fname = os.path.basename(filepath)
    match = re.match(r"(person\d+)", fname, re.IGNORECASE)
    return match.group(1) if match else None

# ── PNEUMONIA: group by person ID ────────────────────────────────────────────
person_to_files = defaultdict(list)
no_id_files = []
for f in all_pneumonia:
    pid = extract_person_id(f)
    if pid:
        person_to_files[pid].append(f)
    else:
        no_id_files.append(f)

person_ids = sorted(person_to_files.keys())
random.Random(RANDOM_SEED).shuffle(person_ids)
val_cutoff = int(len(person_ids) * 0.15)
val_persons  = set(person_ids[:val_cutoff])
train_persons = set(person_ids[val_cutoff:])

train_pneumonia_split = []
val_pneumonia_split   = []
for pid in train_persons:
    train_pneumonia_split += person_to_files[pid]
for pid in val_persons:
    val_pneumonia_split += person_to_files[pid]
# no-ID files → randomly assign
random.Random(RANDOM_SEED).shuffle(no_id_files)
noid_cut = int(len(no_id_files) * 0.85)
train_pneumonia_split += no_id_files[:noid_cut]
val_pneumonia_split   += no_id_files[noid_cut:]

# ── NORMAL: random split (no patient IDs available) ──────────────────────────
shuffled_normal = list(all_normal)
random.Random(RANDOM_SEED).shuffle(shuffled_normal)
cut = int(len(shuffled_normal) * 0.85)
train_normal_split = shuffled_normal[:cut]
val_normal_split   = shuffled_normal[cut:]

print(f"  PNEUMONIA — unique person IDs found: {len(person_ids)}")
print(f"    Person IDs in train: {len(train_persons)}, val: {len(val_persons)}")
print(f"    PNEUMONIA files with no person ID: {len(no_id_files)}")
print(f"  Note: NORMAL filenames have no person ID → random 85/15 split used")
print(f"\n  Final split sizes:")
print(f"    Train : NORMAL={len(train_normal_split)}, PNEUMONIA={len(train_pneumonia_split)}, "
      f"total={len(train_normal_split)+len(train_pneumonia_split)}")
print(f"    Val   : NORMAL={len(val_normal_split)}, PNEUMONIA={len(val_pneumonia_split)}, "
      f"total={len(val_normal_split)+len(val_pneumonia_split)}")
print(f"    Test  : NORMAL={len(test_normal)}, PNEUMONIA={len(test_pneumonia)}, "
      f"total={len(test_normal)+len(test_pneumonia)}")

# ══════════════════════════════════════════════════════════════════════════════
# Dataset + DataLoader
# ══════════════════════════════════════════════════════════════════════════════
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

train_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE + 16, IMG_SIZE + 16)),
    transforms.RandomCrop(IMG_SIZE),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])
val_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

class XRayDataset(Dataset):
    def __init__(self, normal_files, pneumonia_files, transform):
        self.files  = [(f, 0) for f in normal_files] + [(f, 1) for f in pneumonia_files]
        self.transform = transform
    def __len__(self): return len(self.files)
    def __getitem__(self, idx):
        path, label = self.files[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label

train_ds = XRayDataset(train_normal_split, train_pneumonia_split, train_tf)
val_ds   = XRayDataset(val_normal_split,   val_pneumonia_split,   val_tf)
test_ds  = XRayDataset(test_normal,        test_pneumonia,        val_tf)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2, pin_memory=False)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=False)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=False)

print(f"\n  DataLoaders ready. Train batches={len(train_loader)}, Val batches={len(val_loader)}, "
      f"Test batches={len(test_loader)}")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 & 3 — Model + class weights
# ══════════════════════════════════════════════════════════════════════════════
banner("STEP 1 — Hardware + Model choice")
print(f"  GPU available : {torch.cuda.is_available()}")
print(f"  Device        : {DEVICE}")
print(f"  CPU cores     : {os.cpu_count()}")
print(f"  Choice        : ResNet18 (CPU-only; ResNet18 is smallest ResNet, "
      f"trains in reasonable time on CPU)")
print(f"  Max epochs    : {MAX_EPOCHS}")

model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
model.fc = nn.Linear(model.fc.in_features, 2)
model = model.to(DEVICE)
print(f"  FC replaced: 512 → 2 outputs (NORMAL/PNEUMONIA)")

# Class weights: inversely proportional to class frequency
n_train_normal    = len(train_normal_split)
n_train_pneumonia = len(train_pneumonia_split)
n_total           = n_train_normal + n_train_pneumonia
w_normal    = n_total / (2 * n_train_normal)
w_pneumonia = n_total / (2 * n_train_pneumonia)
class_weights = torch.tensor([w_normal, w_pneumonia], dtype=torch.float).to(DEVICE)
print(f"\n  Class weights  : NORMAL={w_normal:.4f}, PNEUMONIA={w_pneumonia:.4f}")

criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(model.parameters(), lr=LR)

# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — Training loop
# ══════════════════════════════════════════════════════════════════════════════
banner("STEP 3 — Training")
print(f"  lr={LR}, batch={BATCH_SIZE}, max_epochs={MAX_EPOCHS}, patience={PATIENCE}")

def run_epoch(loader, train=True):
    model.train() if train else model.eval()
    total_loss, correct, n = 0.0, 0, 0
    all_probs, all_labels = [], []
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            if train: optimizer.zero_grad()
            out  = model(imgs)
            loss = criterion(out, labels)
            if train:
                loss.backward()
                optimizer.step()
            probs = torch.softmax(out, dim=1)[:, 1].detach().cpu().numpy()
            preds = out.argmax(dim=1).cpu().numpy()
            total_loss += loss.item() * len(labels)
            correct    += (preds == labels.cpu().numpy()).sum()
            n          += len(labels)
            all_probs.extend(probs)
            all_labels.extend(labels.cpu().numpy())
    avg_loss = total_loss / n
    acc      = correct / n
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except Exception:
        auc = float("nan")
    return avg_loss, acc, auc, np.array(all_probs), np.array(all_labels)

best_val_loss  = float("inf")
best_state     = None
patience_left  = PATIENCE
history        = []

print(f"\n  {'Ep':>3}  {'TrLoss':>8}  {'TrAcc':>7}  {'TrAUC':>7}  "
      f"{'VaLoss':>8}  {'VaAcc':>7}  {'VaAUC':>7}  {'Note'}")
print("  " + "-"*75)

for epoch in range(1, MAX_EPOCHS + 1):
    t0 = time.time()
    tr_loss, tr_acc, tr_auc, _, _ = run_epoch(train_loader, train=True)
    va_loss, va_acc, va_auc, va_probs, va_labels = run_epoch(val_loader, train=False)
    elapsed = time.time() - t0

    note = ""
    if va_loss < best_val_loss:
        best_val_loss = va_loss
        best_state    = copy.deepcopy(model.state_dict())
        patience_left = PATIENCE
        note          = "✓ saved"
    else:
        patience_left -= 1
        note = f"patience={patience_left}"

    history.append({
        "epoch": epoch, "tr_loss": tr_loss, "tr_acc": tr_acc, "tr_auc": tr_auc,
        "va_loss": va_loss, "va_acc": va_acc, "va_auc": va_auc
    })

    print(f"  {epoch:>3}  {tr_loss:>8.4f}  {tr_acc:>7.4f}  {tr_auc:>7.4f}  "
          f"{va_loss:>8.4f}  {va_acc:>7.4f}  {va_auc:>7.4f}  {note}  ({elapsed:.0f}s)")

    if patience_left <= 0:
        print(f"  Early stopping at epoch {epoch}")
        break

# Restore best
model.load_state_dict(best_state)
print(f"\n  Best checkpoint restored (val_loss={best_val_loss:.4f})")

# ══════════════════════════════════════════════════════════════════════════════
# Threshold tuning on VALIDATION SET ONLY
# ══════════════════════════════════════════════════════════════════════════════
banner("Threshold tuning on VALIDATION SET ONLY")
_, _, _, val_probs_final, val_labels_final = run_epoch(val_loader, train=False)

best_thresh, best_f1 = 0.5, -1.0
for thr in np.arange(0.1, 0.91, 0.02):
    preds_t = (val_probs_final >= thr).astype(int)
    f1 = f1_score(val_labels_final, preds_t, zero_division=0)
    if f1 > best_f1:
        best_f1, best_thresh = f1, float(thr)

print(f"  Best threshold (val F1={best_f1:.4f}): {best_thresh:.2f}")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — Final evaluation on UNTOUCHED test set
# ══════════════════════════════════════════════════════════════════════════════
banner("STEP 4 — Final evaluation on test set (untouched)")
_, _, _, test_probs, test_labels = run_epoch(test_loader, train=False)

def full_eval(probs, labels, thresh, tag):
    preds = (probs >= thresh).astype(int)
    acc   = accuracy_score(labels, preds)
    prec  = precision_score(labels, preds, zero_division=0)
    rec   = recall_score(labels, preds, zero_division=0)      # sensitivity
    spec  = recall_score(labels, preds, pos_label=0, zero_division=0)  # specificity
    f1    = f1_score(labels, preds, zero_division=0)
    bacc  = balanced_accuracy_score(labels, preds)
    mcc   = matthews_corrcoef(labels, preds)
    auc   = roc_auc_score(labels, probs)
    ap    = average_precision_score(labels, probs)
    cm    = confusion_matrix(labels, preds)
    print(f"\n  [{tag}] threshold={thresh:.2f}")
    print(f"    Accuracy     : {acc:.4f}")
    print(f"    Precision    : {prec:.4f}")
    print(f"    Recall/Sens  : {rec:.4f}")
    print(f"    Specificity  : {spec:.4f}")
    print(f"    F1           : {f1:.4f}")
    print(f"    Balanced Acc : {bacc:.4f}")
    print(f"    MCC          : {mcc:.4f}")
    print(f"    ROC-AUC      : {auc:.4f}")
    print(f"    PR-AUC       : {ap:.4f}")
    print(f"\n    Confusion matrix (rows=true, cols=pred):")
    print(f"                  NORMAL  PNEUMONIA")
    print(f"       NORMAL  : {cm[0,0]:6d}  {cm[0,1]:6d}")
    print(f"       PNEUMONIA: {cm[1,0]:6d}  {cm[1,1]:6d}")
    print(f"\n    Classification report:")
    print(classification_report(labels, preds, target_names=CLASS_NAMES, zero_division=0))
    return dict(accuracy=round(acc,4), precision=round(prec,4), recall=round(rec,4),
                specificity=round(spec,4), f1=round(f1,4), balanced_accuracy=round(bacc,4),
                mcc=round(mcc,4), roc_auc=round(auc,4), pr_auc=round(ap,4),
                threshold=round(thresh,2), cm=cm.tolist())

m_05    = full_eval(test_probs, test_labels, 0.5, "threshold=0.50")
m_best  = full_eval(test_probs, test_labels, best_thresh, f"threshold={best_thresh:.2f} (val-tuned)")

# Confusion matrix PNG (best threshold)
cm_best = np.array(m_best["cm"])
fig, ax = plt.subplots(figsize=(5, 4))
ax.imshow(cm_best, interpolation="nearest", cmap=plt.cm.Blues)
ax.set_title(f"Chest X-ray — Confusion Matrix (thresh={best_thresh:.2f})")
for i in range(2):
    for j in range(2):
        ax.text(j, i, cm_best[i, j], ha="center", va="center",
                color="white" if cm_best[i, j] > cm_best.max()/2 else "black", fontsize=14)
ax.set_xticks([0, 1]); ax.set_xticklabels(CLASS_NAMES)
ax.set_yticks([0, 1]); ax.set_yticklabels(CLASS_NAMES)
ax.set_xlabel("Predicted"); ax.set_ylabel("True")
plt.tight_layout()
cm_path = os.path.join(LOGS_DIR, "medical_image_confusion_matrix.png")
plt.savefig(cm_path, dpi=120); plt.close()
print(f"\n  ✓ Confusion matrix PNG → {cm_path}")

# ROC curve
from sklearn.metrics import roc_curve
fpr, tpr, _ = roc_curve(test_labels, test_probs)
fig, ax = plt.subplots(figsize=(5, 4))
ax.plot(fpr, tpr, label=f"AUC={m_best['roc_auc']:.4f}", color="#2196F3", lw=2)
ax.plot([0, 1], [0, 1], "k--", lw=1)
ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curve — Chest X-ray (ResNet18)")
ax.legend(); plt.tight_layout()
roc_path = os.path.join(LOGS_DIR, "medical_image_roc_curve.png")
plt.savefig(roc_path, dpi=120); plt.close()
print(f"  ✓ ROC curve PNG → {roc_path}")

# Log metrics
log_metrics(COMPONENT, {**{f"test_{k}": v for k, v in m_best.items() if k != "cm"},
    "test_cm": m_best["cm"],
    "test_acc_at_0.5": m_05["accuracy"],
    "test_f1_at_0.5": m_05["f1"],
    "val_threshold": best_thresh,
    "val_f1_at_threshold": round(best_f1, 4),
    "model": "ResNet18 ImageNet pretrained",
    "device": str(DEVICE),
    "epochs_run": len(history),
    "best_val_loss": round(best_val_loss, 4),
    "train_size_normal": len(train_normal_split),
    "train_size_pneumonia": len(train_pneumonia_split),
    "val_size": len(val_normal_split) + len(val_pneumonia_split),
    "test_size": len(test_normal) + len(test_pneumonia),
})

# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 — Grad-CAM
# ══════════════════════════════════════════════════════════════════════════════
banner("STEP 5 — Grad-CAM for 4 test images")

class GradCAM:
    def __init__(self, model, target_layer):
        self.model  = model
        self.grads  = None
        self.acts   = None
        target_layer.register_forward_hook(self._save_acts)
        target_layer.register_full_backward_hook(self._save_grads)

    def _save_acts(self, _, __, output):
        self.acts = output.detach()

    def _save_grads(self, _, __, grad_output):
        self.grads = grad_output[0].detach()

    def __call__(self, img_tensor, class_idx):
        self.model.eval()
        img_tensor = img_tensor.unsqueeze(0).to(DEVICE)
        img_tensor.requires_grad_(True)
        out = self.model(img_tensor)
        self.model.zero_grad()
        score = out[0, class_idx]
        score.backward()
        weights = self.grads.mean(dim=(2, 3), keepdim=True)   # (1, C, 1, 1)
        cam = (weights * self.acts).sum(dim=1, keepdim=True)  # (1, 1, H, W)
        cam = torch.relu(cam)
        cam = cam.squeeze().cpu().numpy()
        if cam.max() > 0:
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        return cam

grad_cam = GradCAM(model, model.layer4[-1])

# Find 4 interesting test images
preds_05   = (test_probs >= 0.5).astype(int)
correct    = np.where(preds_05 == test_labels)[0]
incorrect  = np.where(preds_05 != test_labels)[0]

correct_normal    = [i for i in correct    if test_labels[i] == 0]
correct_pneumonia = [i for i in correct    if test_labels[i] == 1]
wrong             = list(incorrect)

sel_indices = []
descs       = []
if correct_normal:
    sel_indices.append(correct_normal[0]);    descs.append("correct_normal")
if correct_pneumonia:
    sel_indices.append(correct_pneumonia[0]); descs.append("correct_pneumonia")
if wrong:
    sel_indices.append(wrong[0]);             descs.append("wrong_prediction")
# 4th: another correct if possible
others = [i for i in correct_pneumonia[1:] if i not in sel_indices]
if others:
    sel_indices.append(others[0]); descs.append("correct_pneumonia_2")
elif len(correct_normal) > 1:
    sel_indices.append(correct_normal[1]); descs.append("correct_normal_2")
sel_indices = sel_indices[:4]; descs = descs[:4]

# Map dataset index to file path
test_files = ([(f, 0) for f in test_normal] + [(f, 1) for f in test_pneumonia])

for sel_idx, desc in zip(sel_indices, descs):
    img_path, true_label = test_files[sel_idx]
    img_raw = Image.open(img_path).convert("RGB")
    img_tensor = val_tf(img_raw)
    pred_label = preds_05[sel_idx]
    prob_val   = test_probs[sel_idx]

    cam  = grad_cam(img_tensor, pred_label)
    cam_resized = np.array(Image.fromarray(np.uint8(255 * cam)).resize(
        (IMG_SIZE, IMG_SIZE), Image.BILINEAR)) / 255.0
    heatmap = cm.jet(cam_resized)[:, :, :3]

    img_np = np.array(img_raw.resize((IMG_SIZE, IMG_SIZE))).astype(float) / 255.0
    overlay = 0.55 * img_np + 0.45 * heatmap
    overlay = np.clip(overlay, 0, 1)

    fig, axes = plt.subplots(1, 3, figsize=(10, 3))
    axes[0].imshow(img_np);   axes[0].set_title("Original");    axes[0].axis("off")
    axes[1].imshow(heatmap);  axes[1].set_title("Grad-CAM");    axes[1].axis("off")
    axes[2].imshow(overlay);  axes[2].set_title("Overlay");     axes[2].axis("off")
    fig.suptitle(f"{desc} | true={CLASS_NAMES[true_label]} pred={CLASS_NAMES[pred_label]} "
                 f"P(Pneumonia)={prob_val:.3f}", fontsize=9)
    plt.tight_layout()
    fpath = os.path.join(LOGS_DIR, f"gradcam_{desc}.png")
    plt.savefig(fpath, dpi=100); plt.close()
    print(f"  ✓ Grad-CAM → logs/gradcam_{desc}.png  "
          f"(true={CLASS_NAMES[true_label]}, pred={CLASS_NAMES[pred_label]}, "
          f"P={prob_val:.3f})")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 6 — Save model
# ══════════════════════════════════════════════════════════════════════════════
banner("STEP 6 — Save model artefacts")

ckpt_path = os.path.join(MODELS_DIR, "resnet18_xray.pt")
torch.save({
    "model_state_dict": model.state_dict(),
    "class_names": CLASS_NAMES,
    "val_threshold": best_thresh,
    "val_f1": best_f1,
    "test_roc_auc": m_best["roc_auc"],
    "img_size": IMG_SIZE,
    "imagenet_mean": IMAGENET_MEAN,
    "imagenet_std": IMAGENET_STD,
}, ckpt_path)
print(f"  ✓ Model checkpoint → {ckpt_path}")

meta = {
    "model": "ResNet18 (ImageNet pretrained)",
    "task": "Binary classification: NORMAL vs PNEUMONIA",
    "val_threshold": best_thresh,
    "test_roc_auc": m_best["roc_auc"],
    "test_sensitivity": m_best["recall"],
    "test_specificity": m_best["specificity"],
    "disclaimer": "SCREENING AID ONLY. Not a diagnostic tool. Not validated for clinical use.",
    "chapter": 3, "component": COMPONENT,
}
with open(os.path.join(MODELS_DIR, "model_metadata.json"), "w") as fh:
    json.dump(meta, fh, indent=2)
print(f"  ✓ Metadata         → {MODELS_DIR}/model_metadata.json")

banner("Chapter 3 training complete")
