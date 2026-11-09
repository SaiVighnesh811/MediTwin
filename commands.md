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


---

## 7. Chapter 9: Backend Authentication Commands

```bash
# Navigate to backend directory
cd /home/sai/Documents/MEDITWIN/meditwin-ai/backend

# Create .env from example (fill in MONGODB_URI and JWT secrets)
cp .env.example .env

# Run Jest integration tests (uses mongodb-memory-server, real pass: 9/9)
npm test

# Start server in development mode (with nodemon auto-restart)
npm run dev

# Start server in production mode
npm start
```


---

## 8. Chapter 9b: Environment & Database Verification

```bash
# Verify database connection
cd /home/sai/Documents/MEDITWIN/meditwin-ai/backend && node scripts/check-db.js

# Start backend dev server
cd backend && npm run dev
```


---

## 9. Chapter 10: Frontend Landing Page Commands

```bash
# Navigate to frontend directory
cd /home/sai/Documents/MEDITWIN/meditwin-ai/frontend

# Start frontend dev server
npm run dev

# Run production build
npm run build
```

---

## 10. Chapter 11: Auth UI & Dashboard Commands

```bash
# Navigate to frontend directory
cd /home/sai/Documents/MEDITWIN/meditwin-ai/frontend

# Run Vitest integration test suite (5 tests: validation, strength meter, server errors, protected route)
npm test

# Run production build
npm run build

# Start frontend dev server
npm run dev
```

---

## 11. Chapter 12: FastAPI ML Microservice Commands

```bash
# Navigate to meditwin-ai root
cd /home/sai/Documents/MEDITWIN/meditwin-ai

# Activate Python virtual environment
source venv/bin/activate

# Run pytest test suite for ML Service (8 integration tests + security)
pytest -v ml_service/tests/test_ml_service.py

# Start ML Service (1 uvicorn worker, internal loopback only)
cd /home/sai/Documents/MEDITWIN/meditwin-ai && source venv/bin/activate && uvicorn ml_service.main:app --host 127.0.0.1 --port 8000 --workers 1

# Curl health check (public route, no API key needed)
curl -s http://127.0.0.1:8000/health | python3 -m json.tool

# Curl risk prediction reading INTERNAL_API_KEY from .env without echoing it
KEY=$(grep INTERNAL_API_KEY ml_service/.env | cut -d '=' -f2)
curl -s -X POST http://127.0.0.1:8000/v1/risk \
  -H "X-Internal-Key: $KEY" \
  -H "Content-Type: application/json" \
  -d '{"HighBP": 1.0, "BMI": 28.5, "Age": 9.0}' | python3 -m json.tool
unset KEY
```

---

## 12. Chapter 13: Node.js Gateway & Analysis Storage Commands

```bash
# Navigate to backend directory
cd /home/sai/Documents/MEDITWIN/meditwin-ai/backend

# Run full backend test suite (Chapter 9 Auth + Chapter 13 Gateway: 20 tests)
npm test

# Start backend dev server
npm run dev

# Example: Run an analysis through the Node gateway using a login session cookie
# 1. Login with cookie jar
curl -s -c cookies.txt -X POST http://localhost:5000/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email":"your_email@example.com","password":"your_password"}'

# 2. Call Gateway /api/analysis/risk
curl -s -b cookies.txt -X POST http://localhost:5000/api/analysis/risk \
  -H "Content-Type: application/json" \
  -d '{"HighBP": 1.0, "BMI": 31.0, "Age": 9.0}' | python3 -m json.tool

# 3. List saved analyses
curl -s -b cookies.txt http://localhost:5000/api/analysis | python3 -m json.tool
rm -f cookies.txt
```

---

## 13. Chapter 14, 14b & 14c: Frontend Dashboard & Polish Commands

```bash
# Navigate to frontend directory
cd /home/sai/Documents/MEDITWIN/meditwin-ai/frontend

# Run full Vitest frontend suite (21 tests: Auth + BRFSS Presentation/Conversion + Explanation Cards + Honest Result View & Notice Logic)
npx vitest run

# Run production build
npm run build

# Start frontend dev server
npm run dev
```

---

## 14. Chapter 3b: Adult X-Ray Model Commands

```bash
# Activate venv
cd /home/sai/Documents/MEDITWIN/meditwin-ai
source venv/bin/activate

# Smoke-test adult_xray_agent (synthetic grayscale PNG)
python -c "
import numpy as np; from PIL import Image; import tempfile, os
arr = np.random.randint(0, 256, (256, 256), dtype=np.uint8)
with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
    Image.fromarray(arr).save(f.name); path = f.name
from agents.medical_image_agent.adult_xray_agent import predict_adult_xray
r = predict_adult_xray(path); os.unlink(path)
print(list(r.keys())); print('pneumonia_score:', r['pneumonia_score'])
"

# Run all 13 ml_service tests (including 5 new Chapter 3b tests) with peak RSS
/usr/bin/time -v python -m pytest ml_service/tests/test_ml_service.py -v 2>&1 | tee /tmp/ml_test_3b.txt

# Run backend jest tests (21 tests)
cd /home/sai/Documents/MEDITWIN/meditwin-ai/backend && npm test

# Run frontend vitest (21 tests)
cd /home/sai/Documents/MEDITWIN/meditwin-ai/frontend && npx vitest run

# Run frontend production build
cd /home/sai/Documents/MEDITWIN/meditwin-ai/frontend && npm run build

# Curl: test gateway image route with age_years (requires running servers)
# Adult route (age=62):
curl -s -b cookies.txt -X POST http://localhost:5000/api/analysis/image \
  -F "file=@/path/to/xray.jpg" -F "age_years=62" | python3 -m json.tool
# Pediatric route (age=8):
curl -s -b cookies.txt -X POST http://localhost:5000/api/analysis/image \
  -F "file=@/path/to/xray.jpg" -F "age_years=8" | python3 -m json.tool
# Missing age → 400:
curl -s -b cookies.txt -X POST http://localhost:5000/api/analysis/image \
  -F "file=@/path/to/xray.jpg" | python3 -m json.tool
```

---

## 15. Chapter 14e: Dashboard Counters, Spacing, X-Ray Only Mode & RSS Inspection

```bash
# Run backend Jest tests (22 tests including new test 13 for gateway age_years forwarding and rejection)
cd /home/sai/Documents/MEDITWIN/meditwin-ai/backend && npm test

# Run full Vitest frontend suite (28 tests across 4 test suites)
cd /home/sai/Documents/MEDITWIN/meditwin-ai/frontend && npx vitest run

# Run production build
cd /home/sai/Documents/MEDITWIN/meditwin-ai/frontend && npm run build

# Check real RSS of running ML Service process (run in your terminal while service is running)
ps -p $(pgrep -f "uvicorn ml_service.main:app" | head -n 1) -o pid,rss,cmd

# Or inspect RSS across all MediTwin processes (Node backend + ML service)
ps aux | grep -E "node.*server.js|uvicorn.*ml_service" | grep -v grep | awk '{printf "PID: %s | RSS: %.2f MB | Command: %s\n", $2, $6/1024, $11}'
```
