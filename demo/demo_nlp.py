"""
demo/demo_nlp.py
MediTwin AI — NLP Agent Demo
Runs pretrained biomedical NER model on 3 test sentences.
Prints [demo_nlp PASS] only if entities are really returned; exits non-zero otherwise.
"""

import os, sys, time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(PROJECT_ROOT, "models", "nlp", "d4data_biomedical_ner_all")
SAFETENSORS_PATH = os.path.join(MODEL_DIR, "model.safetensors")

print("=" * 60)
print("  MediTwin AI — NLP Agent Demo (Pretrained DistilBERT)")
print("=" * 60)

if not os.path.exists(SAFETENSORS_PATH):
    print(f"\n  not available: model missing ({SAFETENSORS_PATH})")
    print("  [demo_nlp SKIPPED]")
    sys.exit(0)

sys.path.insert(0, PROJECT_ROOT)
from agents.nlp_agent.nlp_agent import extract_entities

SENTENCES = [
    "The patient has hypertension and type 2 diabetes mellitus.",
    "She was prescribed Metformin 500 mg and underwent coronary angiography.",
    "Physical examination revealed fever, tachycardia, and mild jaundice.",
]

print(f"\n  Model Path: {MODEL_DIR}")
print("  Extracting entities from 3 sentences …\n")

total_entities_found = 0

for sent in SENTENCES:
    t0 = time.time()
    result = extract_entities(sent)
    elapsed = time.time() - t0

    num_sent_entities = len(result.get("entities", []))
    total_entities_found += num_sent_entities

    print(f"  Input   : {sent}")
    for grp in ["diseases", "symptoms", "medications", "procedures"]:
        ents = result.get(grp, [])
        if ents:
            print(f"    {grp:12s}: {[e['text'] for e in ents]}")
    all_ents = result.get("entities", [])
    if all_ents:
        summary = [f"{e['text']} ({e['entity_type']})" for e in all_ents]
        print(f"    {'all_entities':12s}: {summary}")
    print(f"    Time        : {elapsed*1000:.1f}ms\n")

if total_entities_found == 0:
    print("  ERROR: No entities were extracted! Model failed.")
    print("  [demo_nlp FAIL]")
    sys.exit(1)

print(f"  Extracted {total_entities_found} total entities successfully.")
print("  Demo complete. [demo_nlp PASS]")
sys.exit(0)
