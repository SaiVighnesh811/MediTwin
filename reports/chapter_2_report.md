# MediTwin AI — Chapter 2: XAI Agent Report

## What Was Built

An explainability layer on top of the Chapter 1 Risk Prediction v2 model (XGBoost, BRFSS dataset, 3-class Low/Medium/High risk). Explanations generated using **SHAP** and **LIME** for 3 representative patients and measured across 30–100 patients for quantitative XAI metrics.

### File: `agents/xai_agent/xai_agent.py`

Exposes two public functions:
- `explain_with_shap(patient_df)` → dict with predicted class, top-5 features, SHAP values, plain-English sentences
- `explain_with_lime(patient_df)` → dict with predicted class, top-5 LIME conditions, weights, plain-English sentences

---

## SHAP Method — Model-Agnostic Permutation SHAP

**`TreeExplainer` was unavailable.** shap 0.49.1 calls `float(base_score)` internally, but multiclass XGBoost models store `base_score` as a vector string (e.g. `[7.5E-5,-3.5E-4,2.7E-4]`), not a scalar. This raises `ValueError` and cannot be patched via JSON model editing because XGBoost rewrites the value on reload.

**Workaround used:** `shap.Explainer(model.predict_proba, shap.maskers.Independent(X_bg, 200 rows))` — this is the Permutation SHAP path (model-agnostic). Output shape: `(n_samples, 21_features, 3_classes)`. Runtime: ~3.0 s/patient.

---

## Real Metrics (from `logs/xai_agent_metrics.json`)

| Metric | Value |
|--------|-------|
| LIME stability (avg pairwise top-5 overlap, 5 seeds) | **4.60 / 5** |
| SHAP–LIME top-3 agreement (30 random patients) | **1.47 / 3** |
| Faithfulness — SHAP top-3 deleted, avg prob drop | **+0.1367** |
| Faithfulness — random 3 deleted, avg prob drop | **+0.0054** |
| Faithfulness supported (SHAP > random) | **Yes** (25× larger drop) |
| SHAP time per patient | ~3.0 s |
| LIME time per patient | ~0.22 s |

### SHAP Consistency (20 High-risk patients, top-3 frequency)
| Feature | Appearances | % |
|---------|-------------|---|
| HighBP | 16 / 60 | 26.7% |
| GenHlth | 14 / 60 | 23.3% |
| BMI | 9 / 60 | 15.0% |
| Age | 9 / 60 | 15.0% |
| HeartDiseaseorAttack | 5 / 60 | 8.3% |

---

## 3 Patient Explanations

| Patient | Pred | True | SHAP top-3 | LIME top-3 | Overlap |
|---------|------|------|-----------|-----------|---------|
| Low (idx=0) | Low | Low | GenHlth, Age, HighChol | CholCheck, GenHlth, HighChol | 2/3 |
| High (idx=2) | High | High | BMI, HighBP, Age | BMI, CholCheck, HighBP | 2/3 |
| Medium (idx=7) | Medium | **Low** | Age, HighChol, DiffWalk | CholCheck, BMI, Age | 1/3 |

Note: the Medium patient's true label is Low — the explanation reflects the model's (incorrect) prediction, not ground truth.

---

## Files Created

```
agents/xai_agent/xai_agent.py
logs/xai_shap_summary.png
logs/xai_shap_patient_low.png
logs/xai_shap_patient_medium.png
logs/xai_shap_patient_high.png
logs/xai_lime_patient_low.html
logs/xai_lime_patient_medium.html
logs/xai_lime_patient_high.html
logs/xai_agent_metrics.json
```

---

## Limitations

1. **Permutation SHAP, not TreeSHAP** — Permutation SHAP is unbiased but ~200× slower than TreeSHAP and has higher variance per-patient. Feature interactions are not captured.
2. **SHAP–LIME moderate agreement (1.47/3)** — expected: SHAP perturbs features globally (permutation), LIME fits a local linear model. Divergence is not a sign of error, but means the two explanations ask different questions.
3. **Medium class is unreliable** — the model only predicts Prediabetes correctly ~27% of the time; explanations for Medium-predicted patients may not reflect true clinical drivers.
4. **Survey features only** — all 21 BRFSS features are self-reported. Clinical interpretability is limited compared to lab-value datasets.
