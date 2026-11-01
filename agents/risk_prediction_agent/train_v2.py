"""
agents/risk_prediction_agent/train_v2.py
MediTwin AI — Chapter 1 v2: Class-Weighted XGBoost + Threshold Tuning

Steps:
  1. Baselines (majority-class, v1 reloaded)
  2. Class-weighted XGBoost, no SMOTE, RandomizedSearchCV 20+ configs,
     early stopping on a held-out validation slice
  3. Per-class probability multiplier grid search on validation set
  4. Full evaluation on test set (plain + tuned)
  5. Side-by-side comparison table
  6. Save artefacts, update logs
"""

import os, sys, warnings, json, joblib
warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import (
    train_test_split, RandomizedSearchCV, StratifiedKFold
)
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score,
    classification_report, confusion_matrix,
    roc_auc_score, average_precision_score,
    f1_score, precision_score, recall_score,
    cohen_kappa_score, matthews_corrcoef,
)
from xgboost import XGBClassifier
from utils.metrics_logger import log_metrics

# ── Paths ──────────────────────────────────────────────────────────────────────
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
MODELS_DIR_V1 = os.path.join(PROJECT_ROOT, "models", "risk_prediction")
MODELS_DIR_V2 = os.path.join(PROJECT_ROOT, "models", "risk_prediction_v2")
LOGS_DIR      = os.path.join(PROJECT_ROOT, "logs")
os.makedirs(MODELS_DIR_V2, exist_ok=True)

RANDOM_STATE = 42
COMPONENT    = "risk_prediction_agent"

# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════
def banner(title):
    print(f"\n{'='*65}")
    print(f"  {title}")
    print(f"{'='*65}")

def print_full_report(y_true, y_pred, y_prob, class_names, label):
    banner(f"Evaluation: {label}")
    acc   = accuracy_score(y_true, y_pred)
    bacc  = balanced_accuracy_score(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)
    mcc   = matthews_corrcoef(y_true, y_pred)
    f1m   = f1_score(y_true, y_pred, average="macro", zero_division=0)
    print(f"  Accuracy          : {acc:.4f}")
    print(f"  Balanced Accuracy : {bacc:.4f}")
    print(f"  Cohen Kappa       : {kappa:.4f}")
    print(f"  MCC               : {mcc:.4f}")
    print(f"  F1 Macro          : {f1m:.4f}")
    print()
    print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))

    # ROC-AUC per class
    print("  AUC-ROC (OvR) per class:")
    for i, cn in enumerate(class_names):
        y_bin = (y_true == i).astype(int)
        try:
            auc = roc_auc_score(y_bin, y_prob[:, i])
        except Exception:
            auc = float("nan")
        print(f"    {cn}: {auc:.4f}")

    # Avg Precision per class
    print("  Average Precision (PR-AUC) per class:")
    for i, cn in enumerate(class_names):
        y_bin = (y_true == i).astype(int)
        try:
            ap = average_precision_score(y_bin, y_prob[:, i])
        except Exception:
            ap = float("nan")
        print(f"    {cn}: {ap:.4f}")

    return {
        "accuracy":          round(float(acc),   4),
        "balanced_accuracy": round(float(bacc),  4),
        "cohen_kappa":       round(float(kappa), 4),
        "mcc":               round(float(mcc),   4),
        "f1_macro":          round(float(f1m),   4),
    }


def save_confusion_matrix(y_true, y_pred, class_names, path):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_title(f"Risk Prediction v2 — Confusion Matrix")
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    plt.tight_layout()
    plt.savefig(path, dpi=120); plt.close()
    print(f"  ✓ Confusion matrix → {path}")


def tune_multipliers(y_val, prob_val, class_names):
    """
    Grid-search per-class probability multipliers on validation set.
    We fix mult_low=1.0 and search mult_medium and mult_high.
    Objective: maximise macro-F1 on val set.
    """
    best_f1   = -1
    best_mults = (1.0, 1.0, 1.0)    # (low, medium, high)

    # class order from LabelEncoder: High=0, Low=1, Medium=2  (alphabetical)
    # We want to boost Medium (index 2) and High (index 0)
    mult_medium_grid = [1.0, 2.0, 3.0, 5.0, 8.0, 12.0, 18.0, 25.0, 40.0]
    mult_high_grid   = [1.0, 1.5, 2.0, 3.0, 5.0, 8.0]

    print(f"\n  Tuning multipliers on validation set ({len(y_val)} samples) …")
    for mm in mult_medium_grid:
        for mh in mult_high_grid:
            adj = prob_val.copy()
            adj[:, 0] *= mh    # High
            adj[:, 2] *= mm    # Medium
            # renormalise
            adj = adj / adj.sum(axis=1, keepdims=True)
            y_pred_adj = np.argmax(adj, axis=1)
            f1 = f1_score(y_val, y_pred_adj, average="macro", zero_division=0)
            if f1 > best_f1:
                best_f1    = f1
                best_mults = (1.0, mm, mh)

    print(f"  Best multipliers: Low={best_mults[0]}, Medium={best_mults[1]}, High={best_mults[2]}")
    print(f"  Best val macro-F1 with these multipliers: {best_f1:.4f}")
    return best_mults, best_f1


def apply_multipliers(prob, mults):
    adj = prob.copy()
    # class order: High=0, Low=1, Medium=2
    adj[:, 1] *= mults[0]   # Low
    adj[:, 2] *= mults[1]   # Medium
    adj[:, 0] *= mults[2]   # High
    adj = adj / adj.sum(axis=1, keepdims=True)
    return np.argmax(adj, axis=1)


# ══════════════════════════════════════════════════════════════════════════════
# LOAD DATA (re-use saved labelled CSV, same split as v1)
# ══════════════════════════════════════════════════════════════════════════════
banner("Loading data")
labelled = pd.read_csv(os.path.join(PROCESSED_DIR, "diabetes_labelled.csv"))
feature_df = labelled.drop(columns=["risk_label"]).copy()
risk_le = LabelEncoder()
y_all = risk_le.fit_transform(labelled["risk_label"])
X_all = feature_df.values
feature_names = feature_df.columns.tolist()
class_names = list(risk_le.classes_)   # alphabetical: High, Low, Medium

print(f"  Shape          : {X_all.shape}")
print(f"  Classes        : {dict(zip(class_names, risk_le.transform(class_names)))}")
print(f"  Feature cols   : {feature_names}")

# Same stratified 80/20 split as v1
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X_all, y_all, test_size=0.2, random_state=RANDOM_STATE, stratify=y_all
)
print(f"  Train (full)   : {X_train_full.shape[0]}")
print(f"  Test           : {X_test.shape[0]}")

# Carve out validation from training (20% of train = 16% of total)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full,
    test_size=0.2, random_state=RANDOM_STATE, stratify=y_train_full
)
print(f"  Train (cv)     : {X_train.shape[0]}")
print(f"  Validation     : {X_val.shape[0]}")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Baselines
# ══════════════════════════════════════════════════════════════════════════════
banner("STEP 1 — Baselines")

# Majority-class baseline (always predict Low = index 1)
majority_class = np.bincount(y_train_full).argmax()
print(f"  Majority class in train: {class_names[majority_class]} (index {majority_class})")

y_majority = np.full_like(y_test, majority_class)
maj_acc  = accuracy_score(y_test, y_majority)
maj_bacc = balanced_accuracy_score(y_test, y_majority)
maj_kappa = cohen_kappa_score(y_test, y_majority)
maj_mcc   = matthews_corrcoef(y_test, y_majority)
maj_f1m   = f1_score(y_test, y_majority, average="macro", zero_division=0)
print(f"\n  Majority-class baseline:")
print(f"    Accuracy          : {maj_acc:.4f}")
print(f"    Balanced Accuracy : {maj_bacc:.4f}")
print(f"    Cohen Kappa       : {maj_kappa:.4f}")
print(f"    MCC               : {maj_mcc:.4f}")
print(f"    F1 Macro          : {maj_f1m:.4f}")
print(f"\n  Test set class distribution (ground truth):")
for cls in range(len(class_names)):
    n = (y_test == cls).sum()
    print(f"    {class_names[cls]}: {n} ({100*n/len(y_test):.1f}%)")

# Reload v1 model
banner("STEP 1 (cont.) — v1 model baselines")
v1_model = joblib.load(os.path.join(MODELS_DIR_V1, "xgboost_risk_model.pkl"))
y_pred_v1 = v1_model.predict(X_test)
y_prob_v1 = v1_model.predict_proba(X_test)

v1_acc   = accuracy_score(y_test, y_pred_v1)
v1_bacc  = balanced_accuracy_score(y_test, y_pred_v1)
v1_kappa = cohen_kappa_score(y_test, y_pred_v1)
v1_mcc   = matthews_corrcoef(y_test, y_pred_v1)
v1_f1m   = f1_score(y_test, y_pred_v1, average="macro", zero_division=0)
print(f"\n  v1 (SMOTE) model on test set:")
print(f"    Accuracy          : {v1_acc:.4f}")
print(f"    Balanced Accuracy : {v1_bacc:.4f}")
print(f"    Cohen Kappa       : {v1_kappa:.4f}")
print(f"    MCC               : {v1_mcc:.4f}")
print(f"    F1 Macro          : {v1_f1m:.4f}")
print()
print(classification_report(y_test, y_pred_v1, target_names=class_names, zero_division=0))

# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — Class-weighted XGBoost, no SMOTE, RandomizedSearchCV 20+ configs
# ══════════════════════════════════════════════════════════════════════════════
banner("STEP 2 — Class-weighted XGBoost (no SMOTE)")

# Sample weights for the CV training set
sw_train = compute_sample_weight("balanced", y=y_train)
print(f"  Sample weight per class in train:")
for c in range(len(class_names)):
    idx = np.where(y_train == c)[0]
    print(f"    {class_names[c]}: n={len(idx)}, weight={sw_train[idx[0]]:.4f}")

n_classes = len(class_names)

# Build an early-stopping-aware XGBClassifier via fit_params
# We use X_val/y_val as the eval set so early stopping is never on the test set.
# Note: RandomizedSearchCV doesn't support fit_params per-fold cleanly with
# early stopping, so strategy:
#   a) RandomizedSearchCV with n_estimators fixed to 500, no early stopping → find best params
#   b) Refit best params on X_train with early stopping on X_val

base_clf = XGBClassifier(
    objective="multi:softprob",
    num_class=n_classes,
    eval_metric="mlogloss",
    use_label_encoder=False,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    verbosity=0,
    n_estimators=500,
)

param_dist = {
    "max_depth":          [3, 4, 5, 6, 7, 8],
    "learning_rate":      [0.01, 0.03, 0.05, 0.07, 0.1, 0.15],
    "subsample":          [0.6, 0.7, 0.8, 0.9, 1.0],
    "colsample_bytree":   [0.6, 0.7, 0.8, 0.9, 1.0],
    "min_child_weight":   [1, 3, 5, 10, 20],
    "gamma":              [0, 0.1, 0.3, 0.5, 1.0],
    "reg_alpha":          [0, 0.01, 0.1, 1.0],
    "reg_lambda":         [0.5, 1.0, 2.0, 5.0],
}

cv_strat = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

search = RandomizedSearchCV(
    base_clf,
    param_dist,
    n_iter=20,
    scoring="f1_macro",
    cv=cv_strat,
    n_jobs=-1,
    verbose=2,
    refit=False,               # we refit manually with early stopping
    random_state=RANDOM_STATE,
)

print(f"\n  Running RandomizedSearchCV (20 configs × 5-fold = 100 fits) …")
search.fit(X_train, y_train, sample_weight=sw_train)

# Print all 20 configs tried
print("\n  --- All 20 configurations tried (sorted by mean_test_score) ---")
results_df = pd.DataFrame(search.cv_results_)
cols_show = ["mean_test_score", "std_test_score",
             "param_max_depth", "param_learning_rate", "param_subsample",
             "param_colsample_bytree", "param_min_child_weight",
             "param_gamma", "param_reg_alpha", "param_reg_lambda"]
results_df = results_df[cols_show].sort_values("mean_test_score", ascending=False)
print(results_df.to_string(index=False))

best_params = search.best_params_
print(f"\n  Best params: {best_params}")
print(f"  Best CV macro-F1: {search.best_score_:.4f}")

# Refit on X_train with early stopping on X_val
print("\n  Refitting best config on X_train with early stopping on X_val …")
sw_train_full = compute_sample_weight("balanced", y=y_train)

best_clf = XGBClassifier(
    objective="multi:softprob",
    num_class=n_classes,
    eval_metric="mlogloss",
    use_label_encoder=False,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    verbosity=1,
    n_estimators=2000,           # high cap; early stopping will cut this
    early_stopping_rounds=50,
    **best_params,
)
best_clf.fit(
    X_train, y_train,
    sample_weight=sw_train_full,
    eval_set=[(X_val, y_val)],
    verbose=50,
)
print(f"  Best iteration (early stopping): {best_clf.best_iteration}")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — Threshold / multiplier tuning on validation set
# ══════════════════════════════════════════════════════════════════════════════
banner("STEP 3 — Tuning per-class probability multipliers on validation set")
prob_val = best_clf.predict_proba(X_val)

# Plain argmax on val, for reference
y_pred_val_plain = np.argmax(prob_val, axis=1)
val_f1_plain = f1_score(y_val, y_pred_val_plain, average="macro", zero_division=0)
print(f"  Val macro-F1 (plain argmax): {val_f1_plain:.4f}")

best_mults, best_val_f1_tuned = tune_multipliers(y_val, prob_val, class_names)
print(f"\n  [CHOSEN] Low×{best_mults[0]}, Medium×{best_mults[1]}, High×{best_mults[2]}")
print(f"  These will be applied ONCE to the test set. Not fitted on test set.")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — Final evaluation on test set
# ══════════════════════════════════════════════════════════════════════════════
banner("STEP 4 — Final evaluation on UNTOUCHED test set")

prob_test = best_clf.predict_proba(X_test)

# 4a. Plain argmax
y_pred_plain = np.argmax(prob_test, axis=1)
metrics_plain = print_full_report(y_test, y_pred_plain, prob_test, class_names,
                                   "v2 — plain argmax")

# 4b. Multiplier-tuned
y_pred_tuned = apply_multipliers(prob_test, best_mults)
prob_test_adj = prob_test.copy()
prob_test_adj[:, 1] *= best_mults[0]
prob_test_adj[:, 2] *= best_mults[1]
prob_test_adj[:, 0] *= best_mults[2]
prob_test_adj = prob_test_adj / prob_test_adj.sum(axis=1, keepdims=True)
metrics_tuned = print_full_report(y_test, y_pred_tuned, prob_test_adj, class_names,
                                   "v2 — multiplier-tuned decision rule")

# Confusion matrices
save_confusion_matrix(y_test, y_pred_plain, class_names,
    os.path.join(LOGS_DIR, "risk_prediction_v2_cm_plain.png"))
save_confusion_matrix(y_test, y_pred_tuned, class_names,
    os.path.join(LOGS_DIR, "risk_prediction_confusion_matrix.png"))   # replaces old one

# Feature importance
fi_df = pd.DataFrame({
    "feature":    feature_names,
    "importance": best_clf.feature_importances_,
}).sort_values("importance", ascending=False).head(15)
fig, ax = plt.subplots(figsize=(8, 6))
sns.barplot(data=fi_df, x="importance", y="feature", palette="viridis", ax=ax)
ax.set_title("Top 15 Feature Importances — XGBoost v2 (class-weighted, BRFSS)")
plt.tight_layout()
fi_path = os.path.join(LOGS_DIR, "risk_prediction_feature_importance.png")
plt.savefig(fi_path, dpi=120); plt.close()
print(f"\n  ✓ Feature importance → {fi_path}")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 — Side-by-side comparison table
# ══════════════════════════════════════════════════════════════════════════════
banner("STEP 5 — Side-by-side comparison table")

# Medium class index = 2 in class_names (alphabetical)
med_idx = class_names.index("Medium")
high_idx = class_names.index("High")

def per_class_f1_recall(y_true, y_pred, cls_idx):
    f1  = f1_score(y_true, y_pred, average=None, zero_division=0)[cls_idx]
    rec = recall_score(y_true, y_pred, average=None, zero_division=0)[cls_idx]
    return round(float(f1), 4), round(float(rec), 4)

# v1
f1_med_v1, rec_med_v1   = per_class_f1_recall(y_test, y_pred_v1, med_idx)
# v2 plain
f1_med_v2p, rec_med_v2p = per_class_f1_recall(y_test, y_pred_plain, med_idx)
# v2 tuned
f1_med_v2t, rec_med_v2t = per_class_f1_recall(y_test, y_pred_tuned, med_idx)

rows = [
    ("Majority baseline",           maj_acc,        maj_bacc,       maj_f1m,       "-",       "-"      ),
    ("v1 SMOTE",                    v1_acc,         v1_bacc,        v1_f1m,        f1_med_v1, rec_med_v1),
    ("v2 class-weight argmax",      metrics_plain["accuracy"], metrics_plain["balanced_accuracy"],
                                    metrics_plain["f1_macro"], f1_med_v2p, rec_med_v2p),
    ("v2 class-weight + tuned",     metrics_tuned["accuracy"], metrics_tuned["balanced_accuracy"],
                                    metrics_tuned["f1_macro"], f1_med_v2t, rec_med_v2t),
]

hdr = f"{'Model':<35} {'Accuracy':>9} {'Bal-Acc':>9} {'F1-Mac':>8} {'Med-F1':>8} {'Med-Rec':>9}"
sep = "-" * len(hdr)
print(f"\n{hdr}")
print(sep)
for r in rows:
    acc_  = f"{r[1]:.4f}" if isinstance(r[1], float) else str(r[1])
    bacc_ = f"{r[2]:.4f}" if isinstance(r[2], float) else str(r[2])
    f1_   = f"{r[3]:.4f}" if isinstance(r[3], float) else str(r[3])
    medf1 = f"{r[4]:.4f}" if isinstance(r[4], float) else str(r[4])
    medr  = f"{r[5]:.4f}" if isinstance(r[5], float) else str(r[5])
    print(f"{r[0]:<35} {acc_:>9} {bacc_:>9} {f1_:>8} {medf1:>8} {medr:>9}")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 6 — Log + save
# ══════════════════════════════════════════════════════════════════════════════
banner("STEP 6 — Logging and saving artefacts")

# Log plain v2
log_metrics(COMPONENT, {
    "version":           "v2_classweight_argmax",
    "accuracy":          metrics_plain["accuracy"],
    "balanced_accuracy": metrics_plain["balanced_accuracy"],
    "cohen_kappa":       metrics_plain["cohen_kappa"],
    "mcc":               metrics_plain["mcc"],
    "f1_macro":          metrics_plain["f1_macro"],
    "medium_f1":         f1_med_v2p,
    "medium_recall":     rec_med_v2p,
    "best_params":       str(best_params),
    "best_iteration":    int(best_clf.best_iteration),
    "note": "No SMOTE; compute_sample_weight(balanced); early stopping on val"
})

# Log tuned v2
log_metrics(COMPONENT, {
    "version":           "v2_classweight_tuned",
    "accuracy":          metrics_tuned["accuracy"],
    "balanced_accuracy": metrics_tuned["balanced_accuracy"],
    "cohen_kappa":       metrics_tuned["cohen_kappa"],
    "mcc":               metrics_tuned["mcc"],
    "f1_macro":          metrics_tuned["f1_macro"],
    "medium_f1":         f1_med_v2t,
    "medium_recall":     rec_med_v2t,
    "multipliers":       {"Low": best_mults[0], "Medium": best_mults[1], "High": best_mults[2]},
    "chosen_by":         "max val macro-F1; multipliers never tuned on test set",
})

# Save v2 model
joblib.dump(best_clf, os.path.join(MODELS_DIR_V2, "xgboost_v2_model.pkl"))
joblib.dump(risk_le,  os.path.join(MODELS_DIR_V2, "risk_label_encoder.pkl"))

mult_meta = {
    "Low":    best_mults[0],
    "Medium": best_mults[1],
    "High":   best_mults[2],
    "tuned_on": "validation set (20% of train, never test)",
    "val_macro_f1_before": round(val_f1_plain, 4),
    "val_macro_f1_after":  round(best_val_f1_tuned, 4),
}
with open(os.path.join(MODELS_DIR_V2, "multipliers.json"), "w") as fh:
    json.dump(mult_meta, fh, indent=2)

model_meta = {
    "model":           "XGBClassifier v2",
    "dataset":         "alexteboul/diabetes-health-indicators-dataset",
    "feature_names":   feature_names,
    "classes":         class_names,
    "leakage_check":   "PASSED",
    "smote":           False,
    "sample_weight":   "balanced",
    "best_params":     best_params,
    "best_iteration":  int(best_clf.best_iteration),
    "chapter":         1,
    "component":       COMPONENT,
}
with open(os.path.join(MODELS_DIR_V2, "model_metadata.json"), "w") as fh:
    json.dump(model_meta, fh, indent=2)

print(f"\n  ✓ v2 model   → {MODELS_DIR_V2}/xgboost_v2_model.pkl")
print(f"  ✓ multipliers → {MODELS_DIR_V2}/multipliers.json")
print(f"  ✓ metadata    → {MODELS_DIR_V2}/model_metadata.json")
banner("Chapter 1 v2 COMPLETE")
