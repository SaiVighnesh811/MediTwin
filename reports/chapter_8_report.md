# Chapter 8 Report: Multi-Agent Coordination & Orchestrator

**Component**: Multi-Agent Coordination Orchestrator (`agents/coordination/orchestrator.py`)  
**Date**: November 1, 2026  
**Hardware Profile**: Linux x86_64, 4 GB RAM, CPU-only, Batch Size = 1  
**Dataset Reference**: All evaluations and benchmarks performed on **SYNTHETIC** multi-modal patient cases and official kagglehub chest X-ray evaluation test set.

---

## 1. System Architecture & Coordination Workflow

The Multi-Agent Orchestrator executes a clinical intelligence pipeline linking seven specialized agents:

```
[Clinical Inputs: BRFSS Profile, Rx/Lab Text/Image, Clinical Notes, Chest X-ray]
                                   │
       ┌───────────────────────────┼───────────────────────────┐
       ▼                           ▼                           ▼
1. OCR Agent               2. NLP Agent                3. Medical Image Agent
(EasyOCR parsing)          (Biomedical DistilBERT NER) (ResNet18 Screening)
       │                           │                           │
       └───────────────────────────┼───────────────────────────┘
                                   │
                                   ▼
                     4. Patient Memory Agent
               (Longitudinal JSON Timeline & Deduplication)
                                   │
                                   ▼
                     5. Risk & Explainability (XAI)
               (21 BRFSS Features: XGBoost + SHAP + LIME)
                                   │
                                   ▼
                     6. Recommendation Agent
               (Rules-Based Lifestyle & Preventive Guidance)
                                   │
                                   ▼
                   7. Case Synthesis & Output
```

### Critical Architectural Separation & Design Constraints:
1. **Risk Feature Isolation**: The Risk Prediction model consumes **strictly the 21 BRFSS epidemiological survey features**. Lab measurements and extracted medications are directed to **Patient Memory** and the **Recommendation Agent**, preventing feature-space leakage between survey classifications and electronic health record lab values.
2. **Sequential Memory Safety (`INFERENCE_LOCK`)**: Heavy neural networks (`d4data/biomedical-ner-all` DistilBERT and ResNet-18) are guarded against concurrent execution to ensure peak RSS remains well beneath the 4 GB system ceiling.
3. **Graceful Partial Execution**: If individual inputs are absent or agents encounter missing fields, the orchestrator continues pipeline execution, logs individual statuses (e.g. `skipped (no input)`), and emits a structured partial clinical summary without fatal exceptions.

---

## 2. Quantitative Evaluation Results

The orchestrator was evaluated against three automated benchmark suites via `agents/coordination/evaluate_pipeline.py`.

### A. End-to-End Synthetic Benchmark (10 Multi-Modal Cases)
- **Task-Completion Rate**: **100.0%** (10/10 complete pipeline runs)
- **Mean Case Duration**: **12.53 s**
- **Max Case Duration**: **39.92 s** (first case with initial model and font cache initialization)
- **Peak Process RSS**: **1502.55 MB** (~1.5 GB, within 4 GB RAM constraint)

### B. Fault Injection & Recovery (5 Synthetic Edge Cases)
Tested scenarios included missing demographic profiles, non-existent chest X-ray file paths, completely blank inputs, corrupt profile dictionaries, and image-only submissions.
- **Failure Recovery Rate**: **100.0%** (5/5 cases handled gracefully as partial results without unhandled exceptions)

### C. Execution Consistency & Local Explainability Variance
- **Deterministic Component Match Rate**: **100.0%** across Risk prediction, Medical Image screening, and NLP extracted entities.
- **LIME Local Perturbation Divergence**: Average top-5 feature overlap was **1/5** across repeated runs. As documented in Chapter 2, LIME uses random sample perturbations on CPU, yielding expected stochastic variation in boundary feature rankings compared to exact TreeSHAP.

---

## 3. Real Execution Logs

Metrics captured directly in `logs/coordination_metrics.json`:
```json
{
  "dataset_type": "SYNTHETIC end-to-end benchmark & failure injection",
  "cases_tested": 10,
  "task_completion_rate": 1.0,
  "mean_case_duration_seconds": 12.53,
  "max_case_duration_seconds": 39.92,
  "peak_rss_mb": 1502.55,
  "failure_cases_tested": 5,
  "failure_recovery_rate": 1.0,
  "consistency_cases_tested": 3,
  "consistency_deterministic_match_rate": 1.0,
  "lime_perturbation_divergence_noted": true,
  "status": "PASS"
}
```

---

## 4. Missing Features Experiment & Imputation Strategy

The Risk Prediction model requires 21 BRFSS survey features. In clinical practice, only a subset of features is directly documented in EHR or lab extractions. We systematically evaluated feature-hiding experiments across the held-out test split (seed 42, 20 repeats per condition) comparing **Training Median/Mode** defaults against **Healthy Defaults**:

| Scenario | Imputation Strategy | Balanced Acc | Macro-F1 | High-Class Recall |
| :--- | :--- | :--- | :--- | :--- |
| **k=0 (Full 21 features)** | None | **51.45%** | **41.86%** | **62.60%** |
| **k=5 hidden** | Training Median/Mode | 48.46% | 42.14% | 46.38% |
| **k=5 hidden** | Healthy Defaults | 45.94% | 42.61% | 40.35% |
| **k=10 hidden** | Training Median/Mode | 43.40% | 39.49% | 24.86% |
| **k=10 hidden** | Healthy Defaults | 38.75% | 37.69% | 16.00% |
| **k=15 hidden** | Training Median/Mode | 38.08% | 34.30% | 9.94% |
| **k=15 hidden** | Healthy Defaults | 35.46% | 32.94% | 5.71% |
| **Extractable only (9 known, 12 assumed)** | **Training Median/Mode** | **47.41%** | **39.97%** | **30.78%** |
| **Extractable only (9 known, 12 assumed)** | Healthy Defaults | 39.82% | 40.06% | 16.58% |

### Honest Conclusion on Imputation Defaults:
**Training Median/Mode is the safer and more clinically sensitive default strategy.**
When unobserved features are imputed with "Healthy" values (e.g. ideal BMI 22, excellent general health GenHlth=1, zero symptoms), high-risk patients are severely masked: High-class recall drops by nearly half (from 30.78% down to 16.58% in the realistic extractable case), leading to false negatives on individuals with elevated risk. Training median/mode preserves epidemiological baselines, retaining higher balanced accuracy (47.41% vs 39.82%) and higher sensitivity for elevated risk.

---

## 5. Honest Clinical & Operational Limitations

1. **CPU Latency Overhead**: Local XAI computation (SHAP and LIME) accounts for over 80% of total pipeline latency (~8–10 seconds per patient case on CPU).
2. **Synthetic Integration Context**: Pipeline integration has been verified on synthetic test cases and isolated test datasets; deployment in live clinical production environments requires clinical decision support compliance and integration with certified EHR standards (HL7/FHIR).
3. **Non-Diagnostic Educational Scope**: All recommendations and synthetic risk summaries are non-diagnostic educational outputs subject to clinical review.

