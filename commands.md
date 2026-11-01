# MediTwin AI — Commands and Demos

Use this guide to inspect progress, run agent demos, and view evaluation reports without triggering any training.

---

## 0. Open the project & activate environment

```bash
cd /home/sai/Documents/MEDITWIN/meditwin-ai
source venv/bin/activate
```

---

## 1. Show overall progress

```bash
python show_progress.py              # Overview of tracker, files, metrics, and reports
python show_progress.py tracker      # Chapter status table only
python show_progress.py evidence     # Verify which model/log files really exist
python show_progress.py metrics      # Latest saved metrics per component
python show_progress.py reports      # List written chapter reports
```

---

## 2. Run agent demos (Measured Runtimes)

All demos run on saved models / pretrained checkpoints on CPU:

```bash
# Risk Prediction Agent (3 test patients) — Runtime: ~3.68s (inference: 16.0 ms)
python demo/demo_risk.py

# XAI Agent (SHAP + LIME on 1 patient) — Runtime: ~28.79s (SHAP: 20.87s, LIME: 0.16s)
python demo/demo_xai.py

# Medical Image Agent (2 test chest X-rays with Grad-CAM) — Runtime: ~13.44s
python demo/demo_medical_image.py

# NLP Agent (Pretrained DistilBERT NER on 3 sentences) — Runtime: ~12.72s
python demo/demo_nlp.py

# Patient Memory Agent (Event store, timelines, trends) — Runtime: ~0.10s
python demo/demo_memory.py

# Recommendation Agent (Rules-based educational guidance & safety) — Runtime: ~0.06s
python demo/demo_recommendation.py

# Multi-Agent Coordination Pipeline (End-to-End Orchestrator) — Runtime: ~41.03s
python demo/demo_pipeline.py

# Full Demo Runner
bash run_demo.sh
```

---

## 3. Read evaluation reports

```bash
cat reports/chapter_2_report.md      # Chapter 2: XAI Agent
cat reports/chapter_3_report.md      # Chapter 3: Medical Image Agent (Pretrained ResNet18)
cat reports/chapter_4_report.md      # Chapter 4: OCR Agent
cat reports/chapter_5_report.md      # Chapter 5: NLP Agent (Pretrained DistilBERT)
cat reports/chapter_6_report.md      # Chapter 6: Patient Memory Agent
cat reports/chapter_7_report.md      # Chapter 7: Recommendation Agent
cat reports/chapter_8_report.md      # Chapter 8: Multi-Agent Coordination Pipeline
cat recommendation_rules_review.md   # Recommendation rule safety audit
cat PROJECT_TRACKER.md               # Project-wide tracker
cat MODEL_CREDITS.md                 # Pretrained model credits and licenses
```

---

## 4. Open evaluation plots

```bash
xdg-open logs/medical_image_confusion_matrix.png
xdg-open logs/medical_image_roc.png
xdg-open logs/risk_prediction_confusion_matrix.png
xdg-open logs/xai_shap_summary.png
xdg-open logs/ocr_handwriting_confusion_matrix.png
```

---

## 5. Chapter 8 Follow-Up Commands

```bash
# Run missing features experiment (inference only, seed 42)
python3 -c "
import json
with open('logs/risk_missing_features_metrics.json') as f:
    d = json.load(f)
for r in d['results']:
    print(r)
"

# Run 6-of-21 partial features test case
python3 -c "
from agents.coordination.orchestrator import run_case
res = run_case({
    'patient_id': 'TEST_6_FEATURES_PATIENT',
    'patient_profile': {'HighBP': 1.0, 'HighChol': 1.0, 'BMI': 31.2, 'Smoker': 0.0, 'Age': 9.0, 'Sex': 1.0},
    'clinical_notes': 'Patient reports occasional mild headache.'
})
print(res['summary'])
"
```

---

## 6. Patient Memory NLP Fallback Results

```bash
# Patient Memory OCR adapter fallback test
python3 -c "
from agents.patient_memory_agent.patient_memory_agent import adapt_ocr_result
from agents.ocr_agent.ocr_agent import extract_fields

line = 'Tab Losartan 50 mg once daily'
raw = extract_fields(line)
events = adapt_ocr_result('p1', raw, raw_text=line)
print(events)
"
```
