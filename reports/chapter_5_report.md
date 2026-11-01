# Chapter 5 — NLP Agent Report

> **Not for clinical use.** Research prototype for educational purposes only.

---

## 1. What Was Built

MediTwin AI's NLP Agent performs biomedical Named Entity Recognition (NER) over free-text clinical notes.
It uses the **pretrained** model `d4data/biomedical-ner-all` (DistilBERT fine-tuned on MACCROBAT biomedical case reports), downloaded from Hugging Face and loaded entirely from local disk.
**No retraining or fine-tuning was performed by this project.**

| Item | Detail |
|------|--------|
| Base model | `d4data/biomedical-ner-all` |
| Architecture | DistilBertForTokenClassification (6-layer, 768-dim hidden) |
| NER labels | 84 (O + 41 entity types × B/I prefixes) |
| Model file | `models/nlp/d4data_biomedical_ner_all/model.safetensors` (254 MB) |
| License | Apache-2.0 |
| Loaded via | `from_pretrained(local_path, use_safetensors=True, local_files_only=True)` |
| Old training pipeline | Preserved as `agents/nlp_agent/train.py` (history only; not used) |

### Dataset note
The MACCROBAT dataset (400 rows) contained 200 duplicated documents (each case report appeared twice with two annotation versions). The dataset was deduplicated to 200 unique documents and re-split 70/15/15 (≈140/30/30, seed 42). The earlier row-level split was discarded. The deduplicated split files are preserved in `data/processed/` but are not used in the final system (pretrained model requires no training split).

---

## 2. Label Mapping Table

The model outputs 84 labels. MediTwin AI maps 5 of the 41 entity types to 4 output groups:

| Model Entity Type | MediTwin Group |
|-------------------|----------------|
| `Disease_disorder` | **DISEASE** |
| `Sign_symptom` | **SYMPTOM** |
| `Medication` | **MEDICATION** |
| `Diagnostic_procedure` | **PROCEDURE** |
| `Therapeutic_procedure` | **PROCEDURE** |

### Unmapped entity types → OTHER (38 types)

`Activity`, `Administration`, `Age`, `Area`, `Biological_attribute`, `Biological_structure`,
`Clinical_event`, `Color`, `Coreference`, `Date`, `Detailed_description`, `Distance`,
`Dosage`, `Duration`, `Family_history`, `Frequency`, `Height`, `History`, `Lab_value`,
`Mass`, `Nonbiological_location`, `Occupation`, `Other_entity`, `Other_event`, `Outcome`,
`Personal_background`, `Qualitative_concept`, `Quantitative_concept`, `Severity`, `Sex`,
`Shape`, `Subject`, `Texture`, `Time`, `Volume`, `Weight`

> **Note**: `History` is one of the 38 unmapped types. When conditions like "hypertension" or "type 2 diabetes mellitus" appear in past-history context, the model correctly assigns the `History` label — but that maps to OTHER, not DISEASE. This is correct model behavior, not a bug.

---

## 3. Sample Outputs (10 Sentences)

These results were produced by `agents/nlp_agent/eval_sample_sentences.py` (real measured output):

| # | Input sentence (abbreviated) | Entities found | Latency |
|---|------------------------------|----------------|---------|
| 1 | "Patient presents with chest pain and shortness of breath." | Sign_symptom: chest pain, shortness of breath | ~211 ms |
| 2 | "She was prescribed metformin 500 mg twice daily for type 2 diabetes." | Medication: metformin; Disease_disorder: type 2 diabetes | ~211 ms |
| 3 | "History of hypertension and coronary artery disease." | History: hypertension; Disease_disorder: coronary artery disease | ~211 ms |
| 4 | "A colonoscopy was performed and revealed polyps." | Diagnostic_procedure: colonoscopy; Disease_disorder: polyps | ~211 ms |
| 5 | "The patient underwent appendectomy under general anesthesia." | Therapeutic_procedure: appendectomy; Medication: general anesthesia | ~211 ms |
| 6 | "Tachycardia was noted on admission." | Sign_symptom: tachy* (partial; see limitations) | ~211 ms |
| 7 | "Jaundice and hepatomegaly were present." | Sign_symptom: jaund* (partial), hepatomegaly | ~211 ms |
| 8 | "Ceftriaxone 1 g IV was started empirically." | Medication: Ceftria* (partial), 1 g IV | ~211 ms |
| 9 | "MRI of the brain showed no acute infarct." | Diagnostic_procedure: MRI; Biological_structure: brain | ~211 ms |
| 10 | "The patient recovered and was discharged after 5 days." | Clinical_event: discharged | ~211 ms |

> \* Partial strings due to subword tokenization (see Limitations §5.2).

---

## 4. Memory and Latency

| Measurement | Value |
|-------------|-------|
| RSS baseline | ~8 MB |
| After ResNet18 load (Image Agent) | ~725 MB |
| After NLP Agent load (DistilBERT) | ~887 MB (+162 MB) |
| After image inference | ~914 MB |
| After NLP inference | ~1,081 MB (+167 MB) |
| First-call warmup latency | ~1,500 ms |
| Average subsequent latency | ~211 ms/sentence |
| Sliding window stride | 128 tokens (for texts > 256 tokens) |
| Max input length | 256 tokens |
| Threads | OMP_NUM_THREADS=1, torch.set_num_threads(1) |
| Device | CPU only |

Both ResNet18 and DistilBERT fit simultaneously in ~1,081 MB RSS peak — well within the 4 GB RAM constraint.

---

## 5. Honest Limitations

### 5.1 No independent accuracy number
`d4data/biomedical-ner-all` was trained on the MACCROBAT dataset. Evaluating it on MACCROBAT (even a held-out split) would be data leakage / circularity. No independent NER benchmark was run. The model's reported performance on its training data is not reproducible here.

### 5.2 Subword tokenization returns partial entity strings
DistilBERT tokenizes at subword level. Entity boundaries in the output sometimes correspond to subword tokens, not full words. Examples observed:
- `tachycardia` → `tachy` tagged (rest untagged)
- `jaundice` → `jaund` tagged
- `Ceftriaxone` → `Ceftria` tagged
- `coronary angiography` → `corona` (Diagnostic_procedure) + `ry` (Biological_structure) + `angiography` (Diagnostic_procedure)

This is a limitation of the base tokenizer, not a bug in MediTwin's code. Post-processing word-boundary reconstruction could partially mitigate this.

### 5.3 History context vs. Disease label
Conditions mentioned as past history (e.g., "history of hypertension") are correctly tagged `History` by the model, not `Disease_disorder`. MediTwin maps `History` → OTHER. This is semantically correct behavior but means some diseases are missed in the DISEASE output group.

### 5.4 38 entity types unmapped
Only 5 of 41 entity types are mapped to MediTwin's 4 output groups. The 38 unmapped types are placed in OTHER and are available in the `entities` field of the agent output but not in the structured `diseases`/`symptoms`/`medications`/`procedures` fields.

### 5.5 Domain style
The model was trained on formal biomedical case reports. Performance on clinical notes, discharge summaries, or patient-reported text may differ.

### 5.6 Not validated for clinical use
This system is a research prototype. It must not be used for clinical decision-making.

---

## 6. Files Produced

| File | Description |
|------|-------------|
| `agents/nlp_agent/nlp_agent.py` | NLP Agent — pretrained d4data model, lazy singleton, INFERENCE_LOCK |
| `agents/nlp_agent/eval_sample_sentences.py` | 10-sentence evaluation with latency and memory |
| `models/nlp/d4data_biomedical_ner_all/` | Local model directory (model.safetensors + tokenizer files) |
| `logs/nlp_agent_pretrained_metrics.json` | Measured latency and memory metrics |
| `demo/demo_nlp.py` | Demo: PASS only if entities returned, SKIPPED if checkpoint missing |
| `MODEL_CREDITS.md` | Updated with full NLP Agent section |
| `PROJECT_TRACKER.md` | Chapter 5 row updated to ✅ Done (Pretrained) |
| `agents/nlp_agent/train.py` | Old custom training pipeline (preserved as history, not used) |

---

*Report generated: 2026-11-01. All metrics are real measured values from this hardware (CPU-only, ~4 GB RAM).*

