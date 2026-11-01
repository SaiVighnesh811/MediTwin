# MediTwin AI

Multi-agent AI-powered healthcare intelligence platform.

## Project Structure

```
meditwin-ai/
  data/          # raw and processed datasets
  models/        # trained model artefacts (risk, imaging, OCR, NLP)
  agents/        # individual AI agents
  utils/         # shared helpers (metrics_logger, etc.)
  logs/          # JSON metric logs written per component
  tests/         # unit and integration tests
  venv/          # Python virtual environment (not committed)
```

## Setup

```bash
cd meditwin-ai
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Chapters

| # | Component |
|---|-----------|
| 0 | Project Setup |
| 1 | Risk Prediction Agent |
| 2 | XAI Agent |
| 3 | Medical Image Agent |
| 4 | OCR Agent |
| 5 | NLP Agent |
| 6 | Patient Memory Agent |
| 7 | Recommendation Agent |
| 8 | Multi-Agent Coordination |
