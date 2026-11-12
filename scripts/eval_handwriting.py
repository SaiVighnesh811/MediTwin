"""
scripts/eval_handwriting.py
MediTwin AI — Chapter 16e: Handwriting Classifier Evaluation on Official Test Split

Evaluates the 78-class ResNet18 handwriting classifier (models/ocr/handwriting_cnn.pt)
on the official test split (Testing/testing_words + testing_labels.csv)
using the validation-chosen probability threshold.

Usage:
    python scripts/eval_handwriting.py [--threshold 0.3] [--data-dir /path/to/dataset]
"""

import os, sys, glob, argparse, json
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms, models
from sklearn.metrics import accuracy_score, top_k_accuracy_score

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

DEFAULT_CACHE_BASE = os.path.expanduser(
    "~/.cache/kagglehub/datasets/mamun1113/doctors-handwritten-prescription-bd-dataset/versions/1"
)
CHOSEN_THRESHOLD = 0.3


def find_data_base(user_dir=None):
    if user_dir and os.path.exists(user_dir):
        return user_dir
    matches = glob.glob(os.path.join(DEFAULT_CACHE_BASE, "*"))
    for d in matches:
        if os.path.isdir(d):
            return d
    return None


def main():
    parser = argparse.ArgumentParser(description="Evaluate handwriting classifier on official test split.")
    parser.add_argument("--threshold", type=float, default=CHOSEN_THRESHOLD, help="Confidence threshold (default 0.3)")
    parser.add_argument("--data-dir", type=str, default=None, help="Path to dataset root folder")
    args = parser.parse_args()

    th = args.threshold
    data_base = find_data_base(args.data_dir)
    if not data_base:
        print(f"ERROR: Dataset not found under {DEFAULT_CACHE_BASE} or provided path.")
        print("Please provide --data-dir pointing to 'Doctor’s Handwritten Prescription BD dataset'.")
        sys.exit(1)

    test_csv = os.path.join(data_base, "Testing", "testing_labels.csv")
    test_dir = os.path.join(data_base, "Testing", "testing_words")

    if not os.path.exists(test_csv) or not os.path.exists(test_dir):
        print(f"ERROR: Expected Testing/testing_labels.csv and Testing/testing_words inside {data_base}")
        sys.exit(1)

    df = pd.read_csv(test_csv)
    print(f"Loaded official test split: {len(df)} samples from {test_csv}")

    ckpt_path = os.path.join(PROJECT_ROOT, "models", "ocr", "handwriting_cnn.pt")
    if not os.path.exists(ckpt_path):
        print(f"ERROR: Checkpoint not found at {ckpt_path}")
        sys.exit(1)

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    classes = ckpt["classes"]
    c2i = {c: i for i, c in enumerate(classes)}

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(classes))
    model.load_state_dict(ckpt["best_state_dict"])
    model.eval()

    tf = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    images = []
    targets = []
    for _, row in df.iterrows():
        p = os.path.join(test_dir, str(row["IMAGE"]))
        img = Image.open(p).convert("RGB")
        images.append(tf(img))
        targets.append(c2i[row["MEDICINE_NAME"]])

    x = torch.stack(images)
    targets = np.array(targets)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).numpy()

    pred_classes = probs.argmax(axis=1)
    max_probs = probs.max(axis=1)

    top1_acc = accuracy_score(targets, pred_classes)
    top3_acc = top_k_accuracy_score(targets, probs, k=3)

    mask = max_probs >= th
    n_ans = int(mask.sum())
    coverage = (n_ans / len(targets)) * 100.0
    prec_ans = ((pred_classes[mask] == targets[mask]).sum() / n_ans * 100.0) if n_ans > 0 else 0.0

    print("=" * 65)
    print(f"HANDWRITING CLASSIFIER EVALUATION (Official Test Split)")
    print(f"Chosen Confidence Threshold: {th:.2f}")
    print("=" * 65)
    print(f"  Total test images       : {len(targets)}")
    print(f"  Top-1 Accuracy (overall): {top1_acc * 100:.2f}% ({top1_acc:.4f})")
    print(f"  Top-3 Accuracy (overall): {top3_acc * 100:.2f}% ({top3_acc:.4f})")
    print(f"  Answered count          : {n_ans} / {len(targets)}")
    print(f"  Coverage                : {coverage:.2f}%")
    print(f"  Precision among answered: {prec_ans:.2f}%")
    print("=" * 65)

    # Read existing training run numbers from logs/
    metrics_log = os.path.join(PROJECT_ROOT, "logs", "ocr_agent_handwriting_metrics.json")
    if os.path.exists(metrics_log):
        with open(metrics_log, "r") as f:
            data = json.load(f)
        if isinstance(data, list) and data:
            last = data[-1]
            print("\nMetrics from training run (logs/ocr_agent_handwriting_metrics.json):")
            print(f"  Top-1 Accuracy          : {last.get('accuracy', 0.0) * 100:.2f}%")
            print(f"  Top-3 Accuracy          : {last.get('top3_accuracy', 0.0) * 100:.2f}%")
            print(f"  Macro Precision         : {last.get('macro_precision', 0.0):.4f}")
            print(f"  Macro Recall            : {last.get('macro_recall', 0.0):.4f}")
            print(f"  Macro F1                : {last.get('macro_f1', 0.0):.4f}")
            print(f"  Test size               : {last.get('test_size', 780)}")
            print("=" * 65)


if __name__ == "__main__":
    main()

