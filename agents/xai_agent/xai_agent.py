"""
agents/xai_agent/xai_agent.py
MediTwin AI — Chapter 2: XAI Agent

Explains predictions from the Risk Prediction v2 model using SHAP and LIME.

SHAP note: shap 0.49.1 TreeExplainer fails on multiclass XGBoost models
(base_score is stored as a vector string). We use shap.Explainer with
shap.maskers.Independent (Permutation explainer) which returns shape
(n_samples, n_features, n_classes).

LIME: lime 0.2.0.1 LimeTabularExplainer on training data.

Public API (for Step 5 reusable module):
    explain_with_shap(patient_df) -> dict
    explain_with_lime(patient_df) -> dict
"""

import os, sys, json, time, warnings
warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

import shap
from lime.lime_tabular import LimeTabularExplainer

from utils.metrics_logger import log_metrics

# ── Constants ──────────────────────────────────────────────────────────────────
MODELS_DIR    = os.path.join(PROJECT_ROOT, "models", "risk_prediction_v2")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
LOGS_DIR      = os.path.join(PROJECT_ROOT, "logs")
RANDOM_STATE  = 42
COMPONENT     = "xai_agent"

os.makedirs(LOGS_DIR, exist_ok=True)

# class order from LabelEncoder (alphabetical): High=0, Low=1, Medium=2
CLASS_NAMES = ["High", "Low", "Medium"]

# BRFSS binary/ordinal features (all categorical for LIME)
BINARY_FEATURE_NAMES = [
    "HighBP", "HighChol", "CholCheck", "Smoker", "Stroke",
    "HeartDiseaseorAttack", "PhysActivity", "Fruits", "Veggies",
    "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost", "DiffWalk", "Sex",
]

def banner(msg):
    print(f"\n{'='*65}\n  {msg}\n{'='*65}")


# ══════════════════════════════════════════════════════════════════════════════
# Data loading — exact replica of train_v2.py split
# ══════════════════════════════════════════════════════════════════════════════
def _load_data():
    labelled     = pd.read_csv(os.path.join(PROCESSED_DIR, "diabetes_labelled.csv"))
    feature_df   = labelled.drop(columns=["risk_label"]).copy()
    risk_le      = LabelEncoder()
    y_all        = risk_le.fit_transform(labelled["risk_label"])
    X_all        = feature_df.values
    feature_names = feature_df.columns.tolist()

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X_all, y_all, test_size=0.2, random_state=RANDOM_STATE, stratify=y_all)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full,
        test_size=0.2, random_state=RANDOM_STATE, stratify=y_train_full)

    return (X_train, X_val, X_test,
            y_train, y_val, y_test,
            feature_names, feature_df, risk_le)


# ══════════════════════════════════════════════════════════════════════════════
# Build SHAP explainer (module-level, built once)
# ══════════════════════════════════════════════════════════════════════════════
def _build_shap_explainer(model, X_bg, feature_names):
    """
    Uses shap.Explainer with Independent masker (Permutation explainer).
    Returns shape (n_samples, n_features, n_classes) from __call__.
    Background: 200-row sample from training data.
    """
    masker   = shap.maskers.Independent(X_bg, max_samples=len(X_bg))
    explainer = shap.Explainer(
        model.predict_proba,
        masker,
        feature_names=feature_names,
        output_names=CLASS_NAMES,
    )
    return explainer


def _build_lime_explainer(X_train, feature_names):
    feature_idx = list(range(len(feature_names)))
    # Mark binary columns as categorical
    cat_idx = [i for i, f in enumerate(feature_names) if f in BINARY_FEATURE_NAMES]
    explainer = LimeTabularExplainer(
        X_train,
        feature_names=feature_names,
        class_names=CLASS_NAMES,
        categorical_features=cat_idx,
        mode="classification",
        random_state=RANDOM_STATE,
    )
    return explainer


# ══════════════════════════════════════════════════════════════════════════════
# SHAP helpers
# ══════════════════════════════════════════════════════════════════════════════
def shap_top5_for_class(shap_vals_3d, patient_x, feature_names, cls_idx):
    """
    shap_vals_3d: shape (n_features, n_classes) for a single patient.
    Returns list of (feature_name, feature_value, shap_value) sorted by |shap|.
    """
    sv_class = shap_vals_3d[:, cls_idx]      # (n_features,)
    order    = np.argsort(np.abs(sv_class))[::-1][:5]
    return [(feature_names[i], float(patient_x[i]), float(sv_class[i])) for i in order]


def _plain_english(feature_name, feature_val, shap_val, predicted_class):
    direction = "increased" if shap_val > 0 else "decreased"
    feat_display = feature_name.replace("_", " ")
    val_str = f"={int(feature_val)}" if float(feature_val) == int(feature_val) else f"={feature_val:.2f}"
    return (f"{feat_display}{val_str} {direction} the probability of "
            f"{predicted_class} risk (SHAP={shap_val:+.4f})")


# ══════════════════════════════════════════════════════════════════════════════
# Public API: explain_with_shap / explain_with_lime
# ══════════════════════════════════════════════════════════════════════════════
_shap_explainer = None
_lime_explainer = None
_feature_names  = None
_model          = None


def _ensure_loaded():
    global _shap_explainer, _lime_explainer, _feature_names, _model
    if _shap_explainer is not None:
        return
    model  = joblib.load(os.path.join(MODELS_DIR, "xgboost_v2_model.pkl"))
    _model = model
    (X_train, _, _, _, _, _, feature_names, _, _) = _load_data()
    _feature_names = feature_names
    bg = shap.sample(X_train, 200, random_state=RANDOM_STATE)
    _shap_explainer = _build_shap_explainer(model, bg, feature_names)
    _lime_explainer = _build_lime_explainer(X_train, feature_names)


def explain_with_shap(patient_df: pd.DataFrame) -> dict:
    """
    patient_df: single-row DataFrame with the 21 BRFSS feature columns.
    Returns dict with predicted_class, top 5 features, values, contributions,
    plain-English sentences.
    """
    _ensure_loaded()
    x = patient_df[_feature_names].values.astype(float)
    t0 = time.time()
    sv_obj = _shap_explainer(x, max_evals=500, silent=True)    # (1, 21, 3)
    elapsed = time.time() - t0
    sv_3d  = sv_obj.values[0]                                  # (21, 3)
    proba  = _model.predict_proba(x)[0]
    pred_idx      = int(np.argmax(proba))
    predicted_cls = CLASS_NAMES[pred_idx]
    top5 = shap_top5_for_class(sv_3d, x[0], _feature_names, pred_idx)
    return {
        "predicted_class": predicted_cls,
        "probabilities":   {c: round(float(proba[i]), 4) for i, c in enumerate(CLASS_NAMES)},
        "top5_features":   [
            {"feature": f, "value": v, "shap": s,
             "sentence": _plain_english(f, v, s, predicted_cls)}
            for f, v, s in top5
        ],
        "runtime_seconds": round(elapsed, 3),
        "note": "SHAP via shap.Explainer+Independent masker (Permutation); shape=(1,21,3)",
    }


def explain_with_lime(patient_df: pd.DataFrame) -> dict:
    """
    patient_df: single-row DataFrame with the 21 BRFSS feature columns.
    Returns dict with predicted_class, top 5 LIME features, values, contributions,
    plain-English sentences.
    """
    _ensure_loaded()
    x = patient_df[_feature_names].values[0].astype(float)
    t0 = time.time()
    exp = _lime_explainer.explain_instance(
        x,
        _model.predict_proba,
        num_features=10,
        num_samples=1000,
        labels=list(range(len(CLASS_NAMES))),
    )
    elapsed = time.time() - t0
    proba      = _model.predict_proba(x.reshape(1, -1))[0]
    pred_idx   = int(np.argmax(proba))
    pred_cls   = CLASS_NAMES[pred_idx]
    lime_feats = exp.as_list(label=pred_idx)[:5]
    return {
        "predicted_class": pred_cls,
        "probabilities":   {c: round(float(proba[i]), 4) for i, c in enumerate(CLASS_NAMES)},
        "top5_features":   [
            {"feature_condition": cond, "lime_weight": round(float(w), 4),
             "sentence": (
                 f"Condition '{cond}' {'pushes toward' if w>0 else 'pushes away from'} "
                 f"{pred_cls} risk (LIME weight={w:+.4f})"
             )}
            for cond, w in lime_feats
        ],
        "runtime_seconds": round(elapsed, 3),
    }


# ══════════════════════════════════════════════════════════════════════════════
# MAIN — runs all 6 steps
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    banner("MediTwin AI — Chapter 2: XAI Agent")
    print(f"  shap version : {shap.__version__}")
    print(f"  SHAP method  : shap.Explainer(predict_proba, Independent masker)")
    print(f"  Note         : TreeExplainer unavailable — shap 0.49.1 cannot parse")
    print(f"                 multiclass XGBoost base_score vector as float.")

    # ── Load model ─────────────────────────────────────────────────────────────
    banner("STEP 0 — Load model and rebuild split")
    model   = joblib.load(os.path.join(MODELS_DIR, "xgboost_v2_model.pkl"))
    risk_le = joblib.load(os.path.join(MODELS_DIR, "risk_label_encoder.pkl"))
    print(f"  Model loaded : {type(model).__name__}")
    print(f"  Classes      : {list(risk_le.classes_)}")

    (X_train, X_val, X_test,
     y_train, y_val, y_test,
     feature_names, feature_df, _) = _load_data()

    print(f"  Train: {X_train.shape}  Val: {X_val.shape}  Test: {X_test.shape}")

    # Prove model loads — predict 3 real test rows
    probs3 = model.predict_proba(X_test[:3])
    preds3 = np.argmax(probs3, axis=1)
    print("\n  3 real test rows (plain argmax, no multipliers):")
    for i in range(3):
        print(f"    Row {i}: pred={CLASS_NAMES[preds3[i]]}, "
              f"probs={probs3[i].round(4)}, true={CLASS_NAMES[y_test[i]]}")

    # ── Build explainers ───────────────────────────────────────────────────────
    banner("Building SHAP and LIME explainers")
    np.random.seed(RANDOM_STATE)
    bg_idx = np.random.choice(len(X_train), 200, replace=False)
    X_bg   = X_train[bg_idx]
    shap_explainer = _build_shap_explainer(model, X_bg, feature_names)
    lime_explainer = _build_lime_explainer(X_train, feature_names)
    print("  SHAP explainer: shap.Explainer(predict_proba, Independent(200 rows))")
    print("  LIME explainer: LimeTabularExplainer(train data, 14 categorical features)")

    # ── Pick 3 representative patients ─────────────────────────────────────────
    # Find first test patient predicted as each class
    all_probs = model.predict_proba(X_test)
    all_preds = np.argmax(all_probs, axis=1)
    HIGH_IDX, LOW_IDX, MED_IDX = 0, 1, 2

    idx_low    = np.where(all_preds == LOW_IDX)[0][0]
    idx_high   = np.where(all_preds == HIGH_IDX)[0][0]
    idx_medium = np.where(all_preds == MED_IDX)[0][0]

    patients = {
        "Low":    (idx_low,    X_test[idx_low],    y_test[idx_low]),
        "High":   (idx_high,   X_test[idx_high],   y_test[idx_high]),
        "Medium": (idx_medium, X_test[idx_medium],  y_test[idx_medium]),
    }
    print(f"\n  Selected test-set indices:")
    for name, (idx, x, yt) in patients.items():
        p = model.predict_proba(x.reshape(1, -1))[0]
        print(f"    {name}: test_idx={idx}, true={CLASS_NAMES[yt]}, "
              f"pred={CLASS_NAMES[int(np.argmax(p))]}, probs={p.round(4)}")

    # ── STEP 1: SHAP ──────────────────────────────────────────────────────────
    banner("STEP 1 — SHAP (2,000-row sample, fixed seed=42)")

    rng    = np.random.default_rng(RANDOM_STATE)
    samp2k = rng.choice(len(X_test), 2000, replace=False)
    X_samp = X_test[samp2k]

    print("  Computing SHAP values on 2,000-row sample (max_evals=800) …")
    t_shap_start = time.time()
    sv_samp = shap_explainer(X_samp, max_evals=800, silent=True)   # (2000, 21, 3)
    shap_per_patient_time = (time.time() - t_shap_start) / 2000
    print(f"  SHAP output shape: {sv_samp.values.shape}  (n_samples, n_features, n_classes)")
    print(f"  ~{shap_per_patient_time:.3f}s per patient")

    # Summary bar plot (mean |SHAP| per feature, all 3 classes combined)
    mean_abs = np.abs(sv_samp.values).mean(axis=(0, 2))  # (21,)
    fi_df    = pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs})
    fi_df    = fi_df.sort_values("mean_abs_shap", ascending=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(fi_df["feature"], fi_df["mean_abs_shap"], color="#4C87C8")
    ax.set_xlabel("Mean |SHAP value| (averaged over 2k samples and 3 classes)")
    ax.set_title("SHAP Feature Importance — XAI Agent")
    plt.tight_layout()
    plt.savefig(os.path.join(LOGS_DIR, "xai_shap_summary.png"), dpi=120)
    plt.close()
    print(f"  ✓ SHAP summary → logs/xai_shap_summary.png")

    # Per-patient waterfall plots + top-5 print
    shap_top5_results = {}
    for label, (idx, px, yt) in patients.items():
        sv_one = shap_explainer(
            px.reshape(1, -1), max_evals=800, silent=True)   # (1, 21, 3)
        cls_idx = CLASS_NAMES.index(label)
        sv_3d   = sv_one.values[0]                           # (21, 3)
        top5    = shap_top5_for_class(sv_3d, px, feature_names, cls_idx)
        shap_top5_results[label] = [f for f, v, s in top5]

        print(f"\n  Patient '{label}' (test_idx={idx}, true={CLASS_NAMES[yt]}):")
        print(f"    Top 5 SHAP features for class '{label}':")
        for f, v, s in top5:
            print(f"      {f:28s} val={v:8.2f}  SHAP={s:+.4f}")

        # Waterfall plot for the predicted class
        sv_cls_21 = shap.Explanation(
            values      = sv_3d[:, cls_idx],
            base_values = sv_one.base_values[0, cls_idx],
            data        = px,
            feature_names = feature_names,
        )
        fig, ax = plt.subplots(figsize=(8, 5))
        shap.waterfall_plot(sv_cls_21, max_display=10, show=False)
        plt.tight_layout()
        fname = f"xai_shap_patient_{label.lower()}.png"
        plt.savefig(os.path.join(LOGS_DIR, fname), dpi=120, bbox_inches="tight")
        plt.close()
        print(f"    ✓ Waterfall → logs/{fname}")

    # ── STEP 2: LIME ──────────────────────────────────────────────────────────
    banner("STEP 2 — LIME (same 3 patients)")

    lime_top5_results = {}
    for label, (idx, px, yt) in patients.items():
        t0  = time.time()
        cls_idx = CLASS_NAMES.index(label)
        exp = lime_explainer.explain_instance(
            px.astype(float),
            model.predict_proba,
            num_features=10,
            num_samples=2000,
            labels=list(range(len(CLASS_NAMES))),
        )
        lime_elapsed = time.time() - t0
        lime_feats = exp.as_list(label=cls_idx)[:5]
        lime_top5_results[label] = [cond for cond, _ in lime_feats]

        print(f"\n  Patient '{label}'  (test_idx={idx}, true={CLASS_NAMES[yt]}):")
        print(f"    Top 5 LIME features for class '{label}':")
        for cond, w in lime_feats:
            print(f"      {cond:35s} weight={w:+.4f}")

        # Save HTML
        html_path = os.path.join(LOGS_DIR, f"xai_lime_patient_{label.lower()}.html")
        exp.save_to_file(html_path)
        print(f"    ✓ LIME HTML → logs/xai_lime_patient_{label.lower()}.html")

    # ── STEP 3: SHAP vs LIME agreement ────────────────────────────────────────
    banner("STEP 3 — SHAP vs LIME top-3 overlap for 3 patients")

    for label in ["Low", "Medium", "High"]:
        # SHAP top-3 feature names
        shap3 = set(shap_top5_results[label][:3])
        # LIME top-3: extract feature name from condition string (before operator)
        lime3_raw = lime_top5_results[label][:3]
        lime3 = set()
        for cond in lime3_raw:
            # e.g. "GenHlth > 3.00" or "BMI <= 26.88" or "GenHlth=5"
            for sep in [" <= ", " > ", " >= ", " < ", " = ", "="]:
                if sep in cond:
                    lime3.add(cond.split(sep)[0].strip())
                    break
            else:
                lime3.add(cond.strip())
        overlap = len(shap3 & lime3)
        print(f"  {label:8s}: SHAP top-3={sorted(shap3)}")
        print(f"           LIME top-3={sorted(lime3)}")
        print(f"           Overlap = {overlap}/3\n")

    # ── STEP 4: XAI Metrics ───────────────────────────────────────────────────
    banner("STEP 4 — XAI Metrics")

    # 4a: LIME stability — run LIME 5 times on same patient (Low) with different seeds
    print("  4a: LIME stability (5 runs on patient 'Low', different seeds) …")
    _, px_low, _ = patients["Low"]
    cls_low = CLASS_NAMES.index("Low")
    lime_runs = []
    for seed in range(5):
        tmp_les = LimeTabularExplainer(
            X_train,
            feature_names=feature_names,
            class_names=CLASS_NAMES,
            categorical_features=[i for i, f in enumerate(feature_names) if f in BINARY_FEATURE_NAMES],
            mode="classification",
            random_state=seed,
        )
        tmp_exp = tmp_les.explain_instance(
            px_low.astype(float), model.predict_proba,
            num_features=5, num_samples=2000, labels=[cls_low])
        top5_names = set()
        for cond, _ in tmp_exp.as_list(label=cls_low)[:5]:
            for sep in [" <= ", " > ", " >= ", " < ", " = ", "="]:
                if sep in cond:
                    top5_names.add(cond.split(sep)[0].strip())
                    break
            else:
                top5_names.add(cond.strip())
        lime_runs.append(top5_names)

    pairwise_overlaps = []
    for i in range(5):
        for j in range(i+1, 5):
            pairwise_overlaps.append(len(lime_runs[i] & lime_runs[j]))
    lime_stability = np.mean(pairwise_overlaps)
    print(f"  LIME avg pairwise top-5 overlap (5 runs): {lime_stability:.2f}/5")
    for seed, s in enumerate(lime_runs):
        print(f"    Seed {seed}: {sorted(s)}")

    # 4b: SHAP-LIME agreement over 30 random test patients
    print("\n  4b: SHAP-LIME top-3 agreement over 30 random test patients …")
    rng30 = np.random.default_rng(99)
    idxs30 = rng30.choice(len(X_test), 30, replace=False)
    overlaps30 = []
    for idx30 in idxs30:
        x30    = X_test[idx30]
        pred30 = CLASS_NAMES[int(np.argmax(model.predict_proba(x30.reshape(1, -1))))]
        ci     = CLASS_NAMES.index(pred30)

        sv30   = shap_explainer(x30.reshape(1, -1), max_evals=800, silent=True)
        sv3d   = sv30.values[0]
        s_top3 = set([feature_names[i] for i in np.argsort(np.abs(sv3d[:, ci]))[::-1][:3]])

        exp30  = lime_explainer.explain_instance(
            x30.astype(float), model.predict_proba,
            num_features=5, num_samples=1000, labels=[ci])
        l_top3 = set()
        for cond, _ in exp30.as_list(label=ci)[:3]:
            for sep in [" <= ", " > ", " >= ", " < ", " = ", "="]:
                if sep in cond:
                    l_top3.add(cond.split(sep)[0].strip())
                    break
            else:
                l_top3.add(cond.strip())

        overlaps30.append(len(s_top3 & l_top3))

    shap_lime_agreement = np.mean(overlaps30)
    print(f"  SHAP-LIME avg top-3 overlap (30 patients): {shap_lime_agreement:.3f}/3")

    # 4c: Consistency — top SHAP features for 20 High-predicted patients
    print("\n  4c: SHAP consistency for 20 High-predicted patients …")
    high_idxs = np.where(all_preds == HIGH_IDX)[0][:20]
    from collections import Counter
    feat_freq = Counter()
    for hi in high_idxs:
        xh   = X_test[hi]
        svh  = shap_explainer(xh.reshape(1, -1), max_evals=800, silent=True)
        sv3d = svh.values[0]
        top3 = [feature_names[i] for i in np.argsort(np.abs(sv3d[:, HIGH_IDX]))[::-1][:3]]
        feat_freq.update(top3)
    total_patients = len(high_idxs)
    print(f"  Feature frequency in top-3 SHAP for {total_patients} High-predicted patients:")
    for feat, cnt in feat_freq.most_common(10):
        print(f"    {feat:28s}: {cnt}/{total_patients*3} appearances ({100*cnt/(total_patients*3):.1f}%)")

    # 4d: Faithfulness — deletion test on 100 patients
    print("\n  4d: Faithfulness (deletion test on 100 patients) …")
    rng100 = np.random.default_rng(7)
    idxs100 = rng100.choice(len(X_test), 100, replace=False)

    # Pre-compute medians and modes for training set
    X_train_df = pd.DataFrame(X_train, columns=feature_names)
    feature_medians = X_train_df.median().values
    feature_modes   = X_train_df.mode().iloc[0].values

    drops_shap = []
    drops_rand = []

    for idx100 in idxs100:
        x100  = X_test[idx100].copy().astype(float)
        pc    = CLASS_NAMES[int(np.argmax(model.predict_proba(x100.reshape(1, -1))))]
        ci100 = CLASS_NAMES.index(pc)
        orig_prob = float(model.predict_proba(x100.reshape(1, -1))[0, ci100])

        # SHAP top-3 features
        sv100  = shap_explainer(x100.reshape(1, -1), max_evals=800, silent=True)
        sv3d   = sv100.values[0]
        top3_idx = np.argsort(np.abs(sv3d[:, ci100]))[::-1][:3]

        # Deletion with median/mode of training set
        x_del = x100.copy()
        for fi in top3_idx:
            fname = feature_names[fi]
            x_del[fi] = feature_modes[fi] if fname in BINARY_FEATURE_NAMES else feature_medians[fi]
        del_prob = float(model.predict_proba(x_del.reshape(1, -1))[0, ci100])
        drops_shap.append(orig_prob - del_prob)

        # Random 3 features (not same as top3)
        other = [i for i in range(len(feature_names)) if i not in top3_idx]
        rand3 = rng100.choice(other, 3, replace=False)
        x_rnd = x100.copy()
        for fi in rand3:
            fname = feature_names[fi]
            x_rnd[fi] = feature_modes[fi] if fname in BINARY_FEATURE_NAMES else feature_medians[fi]
        rnd_prob = float(model.predict_proba(x_rnd.reshape(1, -1))[0, ci100])
        drops_rand.append(orig_prob - rnd_prob)

    mean_drop_shap = float(np.mean(drops_shap))
    mean_drop_rand = float(np.mean(drops_rand))
    print(f"  Avg prob drop — SHAP top-3 deleted: {mean_drop_shap:+.4f}")
    print(f"  Avg prob drop — Random 3 deleted  : {mean_drop_rand:+.4f}")
    faithful  = mean_drop_shap > mean_drop_rand
    print(f"  Faithfulness supported: {faithful}  "
          f"(SHAP>Random: {mean_drop_shap:.4f} > {mean_drop_rand:.4f})")

    # 4e: Actual runtimes
    print("\n  4e: SHAP per-patient time already measured above.")
    print(f"  SHAP per patient: ~{shap_per_patient_time:.3f}s  (2000-sample batch / 2000)")
    # LIME timing
    t_lime = time.time()
    tmp_exp2 = lime_explainer.explain_instance(
        X_test[0].astype(float), model.predict_proba,
        num_features=5, num_samples=1000, labels=[0])
    lime_per_patient = time.time() - t_lime
    print(f"  LIME per patient: ~{lime_per_patient:.3f}s  (num_samples=1000)")

    # ── STEP 5: Reusable module check ─────────────────────────────────────────
    banner("STEP 5 — Reusable module: explain_with_shap / explain_with_lime")
    # Inject loaded objects so _ensure_loaded() is skipped
    _model          = model
    _feature_names  = feature_names
    _shap_explainer = shap_explainer
    _lime_explainer = lime_explainer

    # Run on one real patient (first test row)
    patient_df_test = pd.DataFrame(
        X_test[:1], columns=feature_names)

    shap_out = explain_with_shap(patient_df_test)
    lime_out = explain_with_lime(patient_df_test)

    print(f"\n  explain_with_shap result:")
    print(f"    Predicted class: {shap_out['predicted_class']}")
    print(f"    Probabilities  : {shap_out['probabilities']}")
    print(f"    Top 5 features:")
    for item in shap_out["top5_features"]:
        print(f"      {item['sentence']}")

    print(f"\n  explain_with_lime result:")
    print(f"    Predicted class: {lime_out['predicted_class']}")
    print(f"    Top 5 features:")
    for item in lime_out["top5_features"]:
        print(f"      {item['sentence']}")

    # ── Log metrics ───────────────────────────────────────────────────────────
    banner("Logging XAI metrics")
    log_metrics(COMPONENT, {
        "shap_version":               shap.__version__,
        "shap_method":                "shap.Explainer(predict_proba, Independent masker, 200 bg rows)",
        "shap_output_shape":          "(n_samples, 21_features, 3_classes)",
        "lime_stability_avg_overlap": round(float(lime_stability), 3),
        "shap_lime_agreement_top3":   round(float(shap_lime_agreement), 3),
        "shap_lime_agreement_note":   "avg top-3 overlap over 30 random patients",
        "shap_faithfulness_drop":     round(float(mean_drop_shap), 4),
        "random_faithfulness_drop":   round(float(mean_drop_rand), 4),
        "faithfulness_supported":     faithful,
        "shap_time_per_patient_s":    round(float(shap_per_patient_time), 3),
        "lime_time_per_patient_s":    round(float(lime_per_patient), 3),
        "consistency_top_features":   dict(feat_freq.most_common(5)),
        "patients_used_for_consistency": int(total_patients),
        "note": (
            "TreeExplainer unavailable: shap 0.49.1 cannot parse multiclass XGBoost "
            "base_score vector as float. Used Permutation explainer instead."
        ),
    })

    banner("Chapter 2 XAI Agent COMPLETE")
    print("  Files created:")
    for f in ["xai_shap_summary.png",
              "xai_shap_patient_low.png", "xai_shap_patient_medium.png", "xai_shap_patient_high.png",
              "xai_lime_patient_low.html", "xai_lime_patient_medium.html", "xai_lime_patient_high.html",
              "xai_agent_metrics.json"]:
        full = os.path.join(LOGS_DIR, f)
        exists = "✓" if os.path.exists(full) else "✗ MISSING"
        print(f"    {exists}  logs/{f}")
