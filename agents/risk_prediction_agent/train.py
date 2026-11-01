"""
agents/risk_prediction_agent/train.py
MediTwin AI — Chapter 1: Risk Prediction Agent (CORRECTED)

Dataset : alexteboul/diabetes-health-indicators-dataset
File    : diabetes_012_health_indicators_BRFSS2015.csv
Target  : Diabetes_012  (0=No Diabetes, 1=Prediabetes, 2=Diabetes)
          → renamed Low / Medium / High

NO manual risk relabelling.  NO glucose or HbA1c leakage columns.
All 21 original BRFSS survey features are used as-is.

Pipeline:
  1. Download correct dataset via kagglehub
  2. Map 0/1/2 → Low/Medium/High (pure rename, no threshold logic)
  3. Inspect + verify no leaking columns
  4. Train/test split (stratified)
  5. SMOTE to handle class imbalance
  6. XGBoost + GridSearchCV
  7. Evaluate + log_metrics()
  8. Save model artefacts
"""

import os
import sys
import warnings

warnings.filterwarnings("ignore")

# ── Path setup ─────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

import joblib
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
)
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import kagglehub

from utils.metrics_logger import log_metrics


# ── Constants ──────────────────────────────────────────────────────────────────
RAW_DIR       = os.path.join(PROJECT_ROOT, "data", "raw")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
MODELS_DIR    = os.path.join(PROJECT_ROOT, "models", "risk_prediction")
LOGS_DIR      = os.path.join(PROJECT_ROOT, "logs")

os.makedirs(RAW_DIR,       exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(MODELS_DIR,    exist_ok=True)
os.makedirs(LOGS_DIR,      exist_ok=True)

COMPONENT_NAME  = "risk_prediction"
RANDOM_STATE    = 42
DATASET_HANDLE  = "alexteboul/diabetes-health-indicators-dataset"
DATASET_FILE    = "diabetes_012_health_indicators_BRFSS2015.csv"
TARGET_COL      = "Diabetes_012"

# Pure label rename — NO threshold logic, NO column creation
LABEL_MAP = {0: "Low", 1: "Medium", 2: "High"}


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Download correct dataset
# ══════════════════════════════════════════════════════════════════════════════
def download_dataset() -> pd.DataFrame:
    print(f"\n[Step 1] Downloading dataset: {DATASET_HANDLE} …")
    try:
        path = kagglehub.dataset_download(DATASET_HANDLE)
        print(f"  ✓ Downloaded to: {path}")
    except Exception as e:
        raise RuntimeError(
            f"kagglehub download failed: {e}\n"
            "Ensure ~/.config/kaggle/kaggle.json or KAGGLE_USERNAME/KEY env vars are set."
        )

    target_file = os.path.join(path, DATASET_FILE)
    if not os.path.exists(target_file):
        available = os.listdir(path)
        raise FileNotFoundError(
            f"Expected file '{DATASET_FILE}' not found in {path}.\n"
            f"Available: {available}"
        )

    df = pd.read_csv(target_file)
    dst = os.path.join(RAW_DIR, "diabetes_raw.csv")
    df.to_csv(dst, index=False)
    print(f"  ✓ Saved raw CSV  → {dst}  (shape: {df.shape})")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — Map native label 0/1/2 → Low/Medium/High (rename only)
# ══════════════════════════════════════════════════════════════════════════════
def map_labels(df: pd.DataFrame) -> pd.DataFrame:
    print("\n[Step 2] Mapping native Diabetes_012 → Low / Medium / High …")

    if TARGET_COL not in df.columns:
        raise ValueError(f"Expected target column '{TARGET_COL}' not found. Columns: {list(df.columns)}")

    print(f"  Original value counts:\n{df[TARGET_COL].value_counts().sort_index()}\n")

    df["risk_label"] = df[TARGET_COL].map(LABEL_MAP)
    missing = df["risk_label"].isna().sum()
    if missing > 0:
        raise ValueError(f"Unmapped values in {TARGET_COL}: {df[TARGET_COL].unique()}")

    print(f"  Risk label distribution:\n{df['risk_label'].value_counts()}\n")

    # Drop the original numeric target — it must NOT be a feature
    df = df.drop(columns=[TARGET_COL])

    # Save labelled snapshot
    dst = os.path.join(PROCESSED_DIR, "diabetes_labelled.csv")
    df.to_csv(dst, index=False)
    print(f"  ✓ Saved labelled CSV → {dst}")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — Inspect + verify no leaking columns
# ══════════════════════════════════════════════════════════════════════════════
def inspect_and_verify(df: pd.DataFrame) -> None:
    print("\n[Step 3] Verifying features — checking for leakage …")
    feature_cols = [c for c in df.columns if c != "risk_label"]
    print(f"  Feature columns ({len(feature_cols)}):\n  {feature_cols}\n")

    leaking = [c for c in feature_cols if c.lower() in {"hba1c_level", "blood_glucose_level"}]
    if leaking:
        raise RuntimeError(
            f"DATA LEAKAGE DETECTED — columns used to construct the label are still in features: {leaking}"
        )
    print("  ✓ No leaking columns detected — all clear.\n")
    print(f"  Null counts:\n{df[feature_cols].isnull().sum()[df[feature_cols].isnull().sum() > 0]}\n")
    print(f"  Shape: {df.shape}")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — Preprocessing + train/test split
# ══════════════════════════════════════════════════════════════════════════════
def preprocess(df: pd.DataFrame):
    print("\n[Step 4] Preprocessing + train/test split …")

    feature_df = df.drop(columns=["risk_label"]).copy()

    # Handle any object columns (none expected in BRFSS but just in case)
    cat_cols = feature_df.select_dtypes(include=["object", "category"]).columns.tolist()
    label_encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        feature_df[col] = le.fit_transform(feature_df[col].astype(str))
        label_encoders[col] = le
        print(f"  LabelEncoded: {col}")

    # Fill any nulls with median
    if feature_df.isnull().any().any():
        feature_df = feature_df.fillna(feature_df.median(numeric_only=True))

    risk_le = LabelEncoder()
    y = risk_le.fit_transform(df["risk_label"])
    X = feature_df.values
    feature_names = feature_df.columns.tolist()

    print(f"  Feature matrix   : {X.shape}")
    print(f"  Class mapping    : {dict(zip(risk_le.classes_, risk_le.transform(risk_le.classes_)))}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    print(f"  Train / Test     : {X_train.shape[0]} / {X_test.shape[0]}")

    # Save processed file
    proc_df = pd.DataFrame(X, columns=feature_names)
    proc_df["risk_encoded"] = y
    proc_df.to_csv(os.path.join(PROCESSED_DIR, "diabetes_processed.csv"), index=False)
    print(f"  ✓ Saved processed CSV → {PROCESSED_DIR}/diabetes_processed.csv")

    return X_train, X_test, y_train, y_test, risk_le, label_encoders, feature_names


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 — SMOTE
# ══════════════════════════════════════════════════════════════════════════════
def apply_smote(X_train, y_train):
    print("\n[Step 5] SMOTE …")
    unique, counts = np.unique(y_train, return_counts=True)
    print(f"  Before SMOTE: {dict(zip(unique, counts))}")
    smote = SMOTE(random_state=RANDOM_STATE)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    unique2, counts2 = np.unique(y_res, return_counts=True)
    print(f"  After  SMOTE: {dict(zip(unique2, counts2))}")
    print(f"  Resampled shape: {X_res.shape}")
    return X_res, y_res


# ══════════════════════════════════════════════════════════════════════════════
# STEP 6 — XGBoost + GridSearchCV
# ══════════════════════════════════════════════════════════════════════════════
def train_xgboost(X_train, y_train, n_classes: int):
    print("\n[Step 6] Training XGBoost with GridSearchCV …")

    base_model = XGBClassifier(
        objective="multi:softprob",
        num_class=n_classes,
        eval_metric="mlogloss",
        use_label_encoder=False,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=0,
    )

    param_grid = {
        "n_estimators":  [100, 200],
        "max_depth":     [4, 6],
        "learning_rate": [0.05, 0.1],
        "subsample":     [0.8, 1.0],
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    grid_search = GridSearchCV(
        base_model, param_grid,
        scoring="f1_macro", cv=cv, n_jobs=-1, verbose=1, refit=True,
    )
    grid_search.fit(X_train, y_train)

    print(f"\n  ✓ Best params : {grid_search.best_params_}")
    print(f"  ✓ Best CV F1  : {grid_search.best_score_:.4f}")
    return grid_search.best_estimator_, grid_search.best_params_, grid_search.best_score_


# ══════════════════════════════════════════════════════════════════════════════
# STEP 7 — Evaluate + log_metrics
# ══════════════════════════════════════════════════════════════════════════════
def evaluate(model, X_test, y_test, risk_le, feature_names, best_params, best_cv_score):
    print("\n[Step 7] Evaluating …")

    y_pred      = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
    rec  = recall_score(y_test, y_pred, average="macro", zero_division=0)
    f1   = f1_score(y_test, y_pred, average="macro", zero_division=0)
    try:
        auc = roc_auc_score(y_test, y_pred_prob, multi_class="ovr", average="macro")
    except Exception:
        auc = "N/A"

    class_names = list(risk_le.classes_)
    report = classification_report(y_test, y_pred, target_names=class_names)
    print(f"\n  Classification Report:\n{report}")

    metrics = {
        "accuracy":         round(float(acc),  4),
        "precision_macro":  round(float(prec), 4),
        "recall_macro":     round(float(rec),  4),
        "f1_macro":         round(float(f1),   4),
        "auc_roc_macro":    round(float(auc),  4) if auc != "N/A" else "N/A",
        "best_cv_f1":       round(float(best_cv_score), 4),
        "best_params":      str(best_params),
        "n_test_samples":   int(len(y_test)),
        "classes":          class_names,
        "dataset":          DATASET_HANDLE,
        "leakage_check":    "PASSED — no HbA1c/glucose columns in features",
    }

    log_metrics(COMPONENT_NAME, metrics)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_title("Risk Prediction — Confusion Matrix (Test Set)")
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    plt.tight_layout()
    cm_path = os.path.join(LOGS_DIR, f"{COMPONENT_NAME}_confusion_matrix.png")
    plt.savefig(cm_path, dpi=120); plt.close()
    print(f"  ✓ Confusion matrix → {cm_path}")

    # Feature importance
    importances = model.feature_importances_
    fi_df = pd.DataFrame({"feature": feature_names, "importance": importances}) \
              .sort_values("importance", ascending=False).head(15)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(data=fi_df, x="importance", y="feature", palette="viridis", ax=ax)
    ax.set_title("Top 15 Feature Importances — XGBoost (BRFSS)")
    plt.tight_layout()
    fi_path = os.path.join(LOGS_DIR, f"{COMPONENT_NAME}_feature_importance.png")
    plt.savefig(fi_path, dpi=120); plt.close()
    print(f"  ✓ Feature importance → {fi_path}")

    return metrics


# ══════════════════════════════════════════════════════════════════════════════
# STEP 8 — Save model artefacts
# ══════════════════════════════════════════════════════════════════════════════
def save_model(model, risk_le, label_encoders, feature_names):
    print("\n[Step 8] Saving model artefacts …")

    joblib.dump(model,          os.path.join(MODELS_DIR, "xgboost_risk_model.pkl"))
    joblib.dump(risk_le,        os.path.join(MODELS_DIR, "risk_label_encoder.pkl"))
    joblib.dump(label_encoders, os.path.join(MODELS_DIR, "cat_label_encoders.pkl"))

    meta = {
        "model":         "XGBClassifier",
        "dataset":       DATASET_HANDLE,
        "file":          DATASET_FILE,
        "feature_names": feature_names,
        "classes":       list(risk_le.classes_),
        "leakage_check": "PASSED",
        "chapter":       1,
        "component":     COMPONENT_NAME,
    }
    with open(os.path.join(MODELS_DIR, "model_metadata.json"), "w") as fh:
        json.dump(meta, fh, indent=2)

    print(f"  ✓ All artefacts saved to {MODELS_DIR}/")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 65)
    print("  MediTwin AI — Chapter 1: Risk Prediction Agent (CORRECTED)")
    print(f"  Dataset : {DATASET_HANDLE}")
    print("=" * 65)

    df = download_dataset()
    df = map_labels(df)
    inspect_and_verify(df)

    X_train, X_test, y_train, y_test, risk_le, label_encoders, feature_names = preprocess(df)
    X_train_res, y_train_res = apply_smote(X_train, y_train)

    n_classes = len(np.unique(y_train_res))
    best_model, best_params, best_cv_score = train_xgboost(X_train_res, y_train_res, n_classes)

    metrics = evaluate(best_model, X_test, y_test, risk_le, feature_names, best_params, best_cv_score)
    save_model(best_model, risk_le, label_encoders, feature_names)

    print("\n" + "=" * 65)
    print("  Chapter 1 complete (corrected run) ✓")
    print("=" * 65)
