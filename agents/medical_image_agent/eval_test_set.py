"""
Evaluate pretrained ResNet18 on the official 624 chest X-ray test set.
Computes Accuracy, Precision, Recall, Specificity, F1, ROC-AUC, and Confusion Matrix.
Saves:
  - logs/medical_image_confusion_matrix.png
  - logs/medical_image_roc.png
  - logs/medical_image_test_predictions.csv
Logs metrics to logs/medical_image_agent_metrics.json via utils.metrics_logger.
"""

import os, sys, glob, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.metrics_logger import log_metrics

TEST_DIR = os.path.expanduser(
    "~/.cache/kagglehub/datasets/paultimothymooney/chest-xray-pneumonia"
    "/versions/2/chest_xray/test"
)

CKPT_PATH = os.path.join(PROJECT_ROOT, "models", "medical_image", "resnet18_xray_weights.pt")
LOGS_DIR  = os.path.join(PROJECT_ROOT, "logs")

def main():
    normal_imgs = sorted(glob.glob(os.path.join(TEST_DIR, "NORMAL", "*.jpeg")))
    pneumonia_imgs = sorted(glob.glob(os.path.join(TEST_DIR, "PNEUMONIA", "*.jpeg")))
    print(f"Loaded {len(normal_imgs)} NORMAL and {len(pneumonia_imgs)} PNEUMONIA test images.")

    weights = torch.load(CKPT_PATH, map_location="cpu", weights_only=True)
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(512, 1)
    model.load_state_dict(weights)
    model.eval()

    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    items = [(p, 0) for p in normal_imgs] + [(p, 1) for p in pneumonia_imgs]
    records = []
    latencies = []

    t_start = time.time()
    with torch.inference_mode():
        for i, (path, label) in enumerate(items, 1):
            t0 = time.time()
            img = Image.open(path).convert("RGB")
            tensor = tf(img).unsqueeze(0)
            logit = model(tensor).item()
            prob = float(torch.sigmoid(torch.tensor(logit)).item())
            latencies.append(time.time() - t0)
            records.append({
                "image_name": os.path.basename(path),
                "true_label": label,
                "p_pneumonia": prob,
            })
            if i % 100 == 0 or i == len(items):
                print(f"Evaluated {i}/{len(items)} images...")

    total_time = time.time() - t_start
    df = pd.DataFrame(records)

    # Save predictions CSV
    csv_path = os.path.join(LOGS_DIR, "medical_image_test_predictions.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved test predictions to {csv_path}")

    y_true = df["true_label"].values
    y_probs = df["p_pneumonia"].values
    y_pred = (y_probs >= 0.5).astype(int)

    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred))
    rec = float(recall_score(y_true, y_pred))
    f1 = float(f1_score(y_true, y_pred))
    auc = float(roc_auc_score(y_true, y_probs))
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    spec = float(tn / (tn + fp))
    avg_lat_ms = float(np.mean(latencies) * 1000)

    # 1. Confusion Matrix plot
    cm_path = os.path.join(LOGS_DIR, "medical_image_confusion_matrix.png")
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues", interpolation="nearest")
    fig.colorbar(im)
    classes = ["NORMAL", "PNEUMONIA"]
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(classes)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes)
    for i in range(2):
        for j in range(2):
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color=color, fontsize=14)
    ax.set_ylabel("True Label")
    ax.set_xlabel("Predicted Label (Threshold=0.5)")
    ax.set_title("Medical Image Agent — Confusion Matrix")
    plt.tight_layout()
    plt.savefig(cm_path, dpi=150)
    plt.close(fig)
    print(f"Saved confusion matrix plot to {cm_path}")

    # 2. ROC Curve plot
    roc_path = os.path.join(LOGS_DIR, "medical_image_roc.png")
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {auc:.4f})")
    ax.plot([0, 1], [0, 1], color="navy", lw=1.5, linestyle="--")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate (1 - Specificity)")
    ax.set_ylabel("True Positive Rate (Sensitivity / Recall)")
    ax.set_title("Medical Image Agent — ROC Curve")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(roc_path, dpi=150)
    plt.close(fig)
    print(f"Saved ROC curve plot to {roc_path}")

    # 3. Descriptive sensitivity/specificity table at thresholds 0.3, 0.5, 0.7
    print("\n" + "=" * 60)
    print("  Descriptive Threshold Analysis (Not used for selection)")
    print("=" * 60)
    print(f"{'Threshold':<10} {'Sensitivity (Recall)':<22} {'Specificity':<15} {'F1':<10}")
    print("-" * 60)
    for t in [0.3, 0.5, 0.7]:
        preds_t = (y_probs >= t).astype(int)
        cm_t = confusion_matrix(y_true, preds_t)
        tn_t, fp_t, fn_t, tp_t = cm_t.ravel()
        sens_t = tp_t / (tp_t + fn_t)
        spec_t = tn_t / (tn_t + fp_t)
        f1_t   = f1_score(y_true, preds_t)
        print(f"{t:<10.1f} {sens_t:<22.4f} {spec_t:<15.4f} {f1_t:<10.4f}")
    print("=" * 60)

    print("\n" + "=" * 50)
    print("  Official 624 Test Set Evaluation Results (Threshold=0.5)")
    print("=" * 50)
    print(f"Accuracy   : {acc:.4f} ({acc*100:.2f}%)")
    print(f"Precision  : {prec:.4f}")
    print(f"Recall     : {rec:.4f}")
    print(f"Specificity: {spec:.4f}")
    print(f"F1 Score   : {f1:.4f}")
    print(f"ROC-AUC    : {auc:.4f}")
    print(f"Confusion Matrix (TN, FP, FN, TP): TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    print(f"Avg Latency: {avg_lat_ms:.2f} ms per image")
    print(f"Total Time : {total_time:.2f} s")
    print("=" * 50)

    metrics_dict = {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "specificity": round(spec, 4),
        "f1": round(f1, 4),
        "roc_auc": round(auc, 4),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "avg_latency_ms": round(avg_lat_ms, 2),
        "samples": len(items),
        "threshold": 0.5,
    }
    log_metrics("medical_image_agent", metrics_dict)
    print("Metrics logged successfully to logs/medical_image_agent_metrics.json")

if __name__ == "__main__":
    main()
