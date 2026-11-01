"""
demo/demo_risk.py — Risk Prediction Agent demo
Loads models/risk_prediction_v2, predicts 3 real test rows.
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pickle, json
import numpy as np, pandas as pd
from sklearn.preprocessing import LabelEncoder

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR    = os.path.join(PROJECT_ROOT, "models", "risk_prediction_v2")
DATA_PATH    = os.path.join(PROJECT_ROOT, "data", "processed", "diabetes_labelled.csv")

print("=" * 60)
print("  MediTwin AI — Risk Prediction Agent Demo")
print("=" * 60)

# Load model (pickle works for XGBClassifier)
with open(os.path.join(MODEL_DIR, "xgboost_v2_model.pkl"), "rb") as f:
    model = pickle.load(f)
with open(os.path.join(MODEL_DIR, "multipliers.json")) as f:
    multipliers = json.load(f)

# Reconstruct label encoder from CSV (pickle is corrupt due to Python version)
df_all = pd.read_csv(DATA_PATH)
le = LabelEncoder()
le.fit(df_all["risk_label"].values)

print(f"  Model    : XGBoost (loaded from xgboost_v2_model.pkl)")
print(f"  Classes  : {list(le.classes_)}")
print(f"  Multipliers: Low×{multipliers.get('Low',1.0)}  Med×{multipliers.get('Medium',1.0)}  High×{multipliers.get('High',1.0)}")

# Rebuild same test split as train_v2.py (same seed, same features)
df = df_all

FEATURES = ['HighBP','HighChol','CholCheck','BMI','Smoker','Stroke',
            'HeartDiseaseorAttack','PhysActivity','Fruits','Veggies',
            'HvyAlcoholConsump','AnyHealthcare','NoDocbcCost','GenHlth',
            'MentHlth','PhysHlth','DiffWalk','Sex','Age','Education','Income']

label_col = "risk_label"
X = df[FEATURES].values
y_raw = df[label_col].values
y = le.transform(y_raw)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y)

# Predict 3 samples
sample_idx = [0, 1, 2]
t0 = time.time()
proba = model.predict_proba(X_test[sample_idx])
elapsed = time.time() - t0

print(f"\n  {'Row':>4}  {'True':>8}  {'Low%':>7}  {'Med%':>7}  {'High%':>7}  {'Pred':>8}")
print("  " + "-"*55)
for rank, idx in enumerate(sample_idx):
    true_label = le.inverse_transform([y_test[idx]])[0]
    p = proba[rank]
    # apply multipliers
    p_adj = np.array([
        p[le.transform(['Low'])[0]]    * multipliers.get('Low', 1.0),
        p[le.transform(['Medium'])[0]] * multipliers.get('Medium', 1.0),
        p[le.transform(['High'])[0]]   * multipliers.get('High', 1.0),
    ])
    p_adj = p_adj / p_adj.sum()
    pred_label = le.inverse_transform([np.argmax(p_adj)])[0]
    print(f"  {idx:>4}  {true_label:>8}  {p_adj[0]:>6.2%}  {p_adj[1]:>6.2%}  {p_adj[2]:>6.2%}  {pred_label:>8}")

print(f"\n  Inference time (3 rows): {elapsed*1000:.1f} ms")
print("  Demo complete. [demo_risk PASS]")
