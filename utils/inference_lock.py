"""
utils/inference_lock.py
MediTwin AI — Shared process-wide inference lock.

Ensures that memory-heavy neural network models (e.g. Medical Image ResNet18
and NLP DistilBERT) never run inference concurrently, strictly enforcing the
single-model-active memory constraint on low-RAM hardware (4 GB RAM).
"""

import threading

# Global re-entrant lock across agents in the same process
INFERENCE_LOCK = threading.Lock()

