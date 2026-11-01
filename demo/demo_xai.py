"""
demo/demo_xai.py - XAI Agent demo (SHAP + LIME on one real test patient).

Run from the project root:   python demo/demo_xai.py
Loads the SAVED Risk model, takes one real test row, and calls the functions
in agents/xai_agent/xai_agent.py. It adapts to the function signature that
actually exists (it prints it), and it prints PASS only if both explainers
really returned top features. Otherwise it prints the problem and exits 1.
"""
import inspect
import os
import pickle
import sys
import time
import traceback

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "agents", "xai_agent"))
os.chdir(ROOT)  # the XAI module may use relative paths

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

FEATURES = ['HighBP', 'HighChol', 'CholCheck', 'BMI', 'Smoker', 'Stroke',
            'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies',
            'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost', 'GenHlth',
            'MentHlth', 'PhysHlth', 'DiffWalk', 'Sex', 'Age', 'Education', 'Income']
DATA = os.path.join(ROOT, "data", "processed", "diabetes_labelled.csv")
MODEL = os.path.join(ROOT, "models", "risk_prediction_v2", "xgboost_v2_model.pkl")


def fmt(v):
    return f"{v:.4f}" if isinstance(v, (float, np.floating)) else str(v)


def load_model(path):
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception:
        import joblib
        return joblib.load(path)


def build_kwargs(fn, ctx):
    """Map the real parameters of fn to the objects this demo has."""
    kwargs, missing = {}, []
    for name, p in inspect.signature(fn).parameters.items():
        low = name.lower()
        if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
            continue
        if any(k in low for k in ("patient", "row", "sample", "instance")) or low in ("x", "df", "data"):
            kwargs[name] = ctx["row"]
        elif "model" in low:
            kwargs[name] = ctx["model"]
        elif "feature" in low and "name" in low:
            kwargs[name] = FEATURES
        elif any(k in low for k in ("train", "background", "bg")):
            kwargs[name] = ctx["train_df"]
        elif "encoder" in low or low in ("le", "enc"):
            kwargs[name] = ctx["le"]
        elif p.default is p.empty:
            missing.append(name)
    if missing:
        raise TypeError(f"{fn.__name__}{inspect.signature(fn)} needs parameter(s) this demo "
                        f"does not know how to fill: {missing}")
    return kwargs


def show(result, title):
    """Print top features + sentences from whatever dict the module returns."""
    print(f"\n  --- {title} ---")
    if not isinstance(result, dict):
        raise RuntimeError(f"{title}: expected a dict, got {type(result).__name__}: {str(result)[:200]}")
    for key, v in result.items():
        if isinstance(v, (str, int, float, np.floating)) and any(k in key.lower() for k in ("class", "pred")):
            print(f"  {key}: {fmt(v)}")
    shown = 0
    for key, v in result.items():
        k = key.lower()
        if ("top" in k or "feature" in k or "contrib" in k) and isinstance(v, (list, dict)) and v:
            print(f"  {key} (top 5):")
            items = list(v.items()) if isinstance(v, dict) else list(v)
            for it in items[:5]:
                if isinstance(it, tuple) and len(it) == 2 and isinstance(v, dict):
                    print(f"    {it[0]}: {fmt(it[1])}")
                elif isinstance(it, dict):
                    print("    " + "   ".join(f"{a}={fmt(b)}" for a, b in it.items()))
                elif isinstance(it, (list, tuple)):
                    print("    " + "   ".join(fmt(x) for x in it))
                else:
                    print(f"    {it}")
                shown += 1
            break
    for key, v in result.items():
        if any(k in key.lower() for k in ("sentence", "explanation", "plain")):
            print(f"  {key}:")
            for s in ([v] if isinstance(v, str) else list(v))[:5]:
                print(f"    - {s}")
    if shown == 0:
        raise RuntimeError(f"{title}: no top features found in the result. Keys returned: {list(result.keys())}")


def main():
    print("=" * 60)
    print("  MediTwin AI - XAI Agent Demo")
    print("=" * 60)

    model = load_model(MODEL)
    df = pd.read_csv(DATA)
    le = LabelEncoder().fit(df["risk_label"].values)  # classes sorted: High, Low, Medium
    y = le.transform(df["risk_label"].values)
    X = df[FEATURES]
    # Assumes the same 80/20 stratified split, seed 42, as training (not verified here).
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
    X_tr = X_tr.reset_index(drop=True)
    X_te = X_te.reset_index(drop=True)
    row = X_te.iloc[[0]]

    proba = model.predict_proba(row.values)[0]
    print(f"\n  Test patient 0 | true label: {le.classes_[y_te[0]]} | "
          f"model predicts: {le.classes_[int(np.argmax(proba))]}")
    print("  Probabilities: " + ", ".join(f"{c}={p:.3f}" for c, p in zip(le.classes_, proba)))

    import xai_agent
    ctx = {"row": row, "model": model, "train_df": X_tr, "le": le}

    for fn_name, title in (("explain_with_shap", "SHAP explanation"),
                           ("explain_with_lime", "LIME explanation")):
        fn = getattr(xai_agent, fn_name)
        print(f"\n  Calling {fn_name}{inspect.signature(fn)}")
        t0 = time.time()
        result = fn(**build_kwargs(fn, ctx))
        elapsed = time.time() - t0
        show(result, title)
        print(f"  {title.split()[0]} runtime (measured): {elapsed:.2f} s")

    print("\n  Demo complete. [demo_xai PASS]")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        print("\n  [demo_xai FAIL]")
        sys.exit(1)