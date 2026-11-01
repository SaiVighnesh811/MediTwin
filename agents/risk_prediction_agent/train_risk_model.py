"""
Chapter 1 — Risk Prediction Agent
train_risk_model.py

Steps 1-7: load → relabel → preprocess → SMOTE → XGBoost → evaluate → save
"""

import sys, os, shutil, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score
)
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

from utils.metrics_logger import log_metrics

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW_SRC  = os.path.expanduser(
    "~/.cache/kagglehub/datasets/alexteboul/"
    "diabetes-health-indicators-dataset/versions/1/"
    "diabetes_012_health_indicators_BRFSS2015.csv"
)
RAW_DEST   = os.path.join(BASE, "data", "raw", "diabetes_012_health_indicators_BRFSS2015.csv")
PROC_DEST  = os.path.join(BASE, "data", "processed", "risk_prediction_processed.csv")
LOGS_DIR   = os.path.join(BASE, "logs")
MODELS_DIR = os.path.join(BASE, "models", "risk_prediction")
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# ═══════════════════════════════════════════════════════════════════
# STEP 1 — Load raw dataset
# ═══════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("STEP 1 — Loading raw dataset")
print("="*65)

shutil.copy2(RAW_SRC, RAW_DEST)
print(f"Copied raw CSV → {RAW_DEST}")

df = pd.read_csv(RAW_DEST)
print(f"\nShape : {df.shape}")
print(f"\nColumns ({len(df.columns)}):")
for c in df.columns:
    print(f"  {c}")

print("\nLabel column (Diabetes_012) value_counts:")
print(df['Diabetes_012'].value_counts().sort_index())

# ═══════════════════════════════════════════════════════════════════
# STEP 2 — Relabel 0/1/2 → Low/Medium/High
# ═══════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("STEP 2 — Relabeling to risk levels")
print("="*65)

label_map = {0.0: 'Low', 1.0: 'Medium', 2.0: 'High'}
df['RiskLevel'] = df['Diabetes_012'].map(label_map)
df.to_csv(PROC_DEST, index=False)
print(f"Saved processed CSV → {PROC_DEST}")
print("\nRiskLevel value_counts after relabeling:")
print(df['RiskLevel'].value_counts())

# ═══════════════════════════════════════════════════════════════════
# STEP 3 — Preprocessing
# ═══════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("STEP 3 — Preprocessing")
print("="*65)

print("\nMissing values per column:")
missing = df.isnull().sum()
print(missing[missing > 0] if missing.any() else "  None — dataset is complete.")

# Features: all columns except the original label and new risk label
FEATURE_COLS = [c for c in df.columns if c not in ['Diabetes_012', 'RiskLevel']]
print(f"\nFeature columns used ({len(FEATURE_COLS)}): {FEATURE_COLS}")

X = df[FEATURE_COLS].values
y_raw = df['RiskLevel']

# Encode labels to integers for XGBoost
le = LabelEncoder()
y = le.fit_transform(y_raw)
print(f"\nLabel encoding: {dict(zip(le.classes_, le.transform(le.classes_)))}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTrain shape: X={X_train.shape}, y={y_train.shape}")
print(f"Test  shape: X={X_test.shape},  y={y_test.shape}")

# ═══════════════════════════════════════════════════════════════════
# STEP 4 — Class imbalance → SMOTE
# ═══════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("STEP 4 — Class imbalance check + SMOTE")
print("="*65)

unique, counts = np.unique(y_train, return_counts=True)
print("\nClass distribution BEFORE SMOTE (train set):")
for cls, cnt in zip(le.inverse_transform(unique), counts):
    print(f"  {cls}: {cnt}")

smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

unique2, counts2 = np.unique(y_train_bal, return_counts=True)
print("\nClass distribution AFTER SMOTE (train set):")
for cls, cnt in zip(le.inverse_transform(unique2), counts2):
    print(f"  {cls}: {cnt}")

# ═══════════════════════════════════════════════════════════════════
# STEP 5 — Train XGBoost (RandomizedSearchCV)
# ═══════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("STEP 5 — Training XGBoost with RandomizedSearchCV")
print("="*65)

param_dist = {
    'max_depth':      [3, 4, 5, 6],
    'n_estimators':   [100, 200, 300],
    'learning_rate':  [0.05, 0.1, 0.2],
    'subsample':      [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0],
}

base_xgb = XGBClassifier(
    objective='multi:softprob',
    num_class=3,
    eval_metric='mlogloss',
    use_label_encoder=False,
    random_state=42,
    n_jobs=-1,
    verbosity=0,
)

search = RandomizedSearchCV(
    base_xgb,
    param_distributions=param_dist,
    n_iter=12,
    cv=3,
    scoring='f1_macro',
    random_state=42,
    n_jobs=-1,
    verbose=1,
)

print("\nParam grid being searched:")
for k, v in param_dist.items():
    print(f"  {k}: {v}")
print("\nRunning RandomizedSearchCV (n_iter=12, cv=3) …")
search.fit(X_train_bal, y_train_bal)

print(f"\nBest params selected: {search.best_params_}")
print(f"Best CV f1_macro:     {search.best_score_:.4f}")

model = search.best_estimator_

# ═══════════════════════════════════════════════════════════════════
# STEP 6 — Evaluate on held-out test set
# ═══════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("STEP 6 — Evaluation on test set")
print("="*65)

y_pred       = model.predict(X_test)
y_proba      = model.predict_proba(X_test)

accuracy  = accuracy_score(y_test, y_pred)
macro_prec = precision_score(y_test, y_pred, average='macro', zero_division=0)
macro_rec  = recall_score(y_test, y_pred, average='macro', zero_division=0)
macro_f1   = f1_score(y_test, y_pred, average='macro', zero_division=0)
roc_auc    = roc_auc_score(y_test, y_proba, multi_class='ovr', average='macro')

print(f"\nAccuracy          : {accuracy:.4f}")
print(f"Macro Precision   : {macro_prec:.4f}")
print(f"Macro Recall      : {macro_rec:.4f}")
print(f"Macro F1          : {macro_f1:.4f}")
print(f"ROC-AUC (macro-OvR): {roc_auc:.4f}")

print("\n--- Full classification_report ---")
target_names = le.inverse_transform(sorted(np.unique(y_test)))
print(classification_report(y_test, y_pred,
                             target_names=target_names, zero_division=0))

# Confusion matrix — text
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix (rows=actual, cols=predicted):")
print(f"  Labels: {list(target_names)}")
print(cm)

# Confusion matrix — image
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=target_names, yticklabels=target_names, ax=ax)
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
ax.set_title('Risk Prediction — Confusion Matrix')
plt.tight_layout()
cm_path = os.path.join(LOGS_DIR, 'risk_prediction_confusion_matrix.png')
plt.savefig(cm_path, dpi=150)
plt.close()
print(f"\nConfusion matrix image → {cm_path}")

# Feature importance image
feat_imp = model.feature_importances_
fi_df = pd.DataFrame({'feature': FEATURE_COLS, 'importance': feat_imp})
fi_df = fi_df.sort_values('importance', ascending=True).tail(20)

fig, ax = plt.subplots(figsize=(8, 7))
ax.barh(fi_df['feature'], fi_df['importance'], color='steelblue')
ax.set_xlabel('Importance Score')
ax.set_title('Risk Prediction — Top Feature Importances (XGBoost)')
plt.tight_layout()
fi_path = os.path.join(LOGS_DIR, 'risk_prediction_feature_importance.png')
plt.savefig(fi_path, dpi=150)
plt.close()
print(f"Feature importance image → {fi_path}")

# Per-class metrics from classification_report dict
cr = classification_report(y_test, y_pred,
                            target_names=target_names, output_dict=True,
                            zero_division=0)

metrics_dict = {
    'accuracy':           round(accuracy, 4),
    'macro_precision':    round(macro_prec, 4),
    'macro_recall':       round(macro_rec, 4),
    'macro_f1':           round(macro_f1, 4),
    'roc_auc_macro_ovr':  round(roc_auc, 4),
    'best_cv_f1_macro':   round(search.best_score_, 4),
    'best_params':        str(search.best_params_),
}
for cls in target_names:
    metrics_dict[f'{cls}_precision'] = round(cr[cls]['precision'], 4)
    metrics_dict[f'{cls}_recall']    = round(cr[cls]['recall'], 4)
    metrics_dict[f'{cls}_f1']        = round(cr[cls]['f1-score'], 4)

log_metrics('risk_prediction_agent', metrics_dict)

# ═══════════════════════════════════════════════════════════════════
# STEP 7 — Save model + label encoder
# ═══════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("STEP 7 — Saving model artifacts")
print("="*65)

model_path = os.path.join(MODELS_DIR, 'xgboost_risk_model.joblib')
le_path    = os.path.join(MODELS_DIR, 'label_encoder.joblib')
meta_path  = os.path.join(MODELS_DIR, 'model_metadata.json')

joblib.dump(model, model_path)
joblib.dump(le,    le_path)

metadata = {
    'model':        'XGBClassifier',
    'best_params':  search.best_params_,
    'feature_cols': FEATURE_COLS,
    'classes':      list(le.classes_),
    'train_shape':  list(X_train_bal.shape),
    'test_shape':   list(X_test.shape),
    'metrics': {
        'accuracy':          round(accuracy, 4),
        'macro_f1':          round(macro_f1, 4),
        'roc_auc_macro_ovr': round(roc_auc, 4),
    }
}
with open(meta_path, 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"\nFiles saved to {MODELS_DIR}:")
for fname in os.listdir(MODELS_DIR):
    fpath = os.path.join(MODELS_DIR, fname)
    print(f"  {fname}  ({os.path.getsize(fpath):,} bytes)")

print("\n✅  Chapter 1 pipeline complete.")
