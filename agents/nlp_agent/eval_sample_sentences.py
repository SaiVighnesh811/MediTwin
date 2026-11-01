"""
agents/nlp_agent/eval_sample_sentences.py
Evaluation of pretrained NLP agent (d4data/biomedical-ner-all) on 10 written clinical example sentences.
NOTE: These are human-written clinical illustrative examples, NOT test data.
Computes entities per group, latency per sentence, and process RSS memory.
"""

import os, sys, time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from agents.nlp_agent.nlp_agent import extract_entities, _load

def get_rss():
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return float(line.split()[1]) / 1024.0
    return 0.0

SAMPLE_SENTENCES = [
    # 1. Acute coronary syndrome / cardiac presentation
    "The patient is a 58-year-old male presenting with acute chest pain, shortness of breath, and diaphoresis.",
    # 2. Acute management / medication administration
    "Diagnosed with acute myocardial infarction; sublingual nitroglycerin and Aspirin 325 mg were administered immediately.",
    # 3. Interventional cardiology procedure
    "The patient underwent urgent coronary angiography followed by percutaneous coronary intervention with stent placement.",
    # 4. Chronic endocrine / vascular comorbidities
    "History of poorly controlled type 2 diabetes mellitus and chronic hypertension.",
    # 5. Outpatient pharmacotherapy regimen
    "Prescribed Metformin 1000 mg twice daily, Lisinopril 20 mg once daily, and Atorvastatin 40 mg at bedtime.",
    # 6. Diagnostic imaging / echocardiography
    "Follow-up echocardiogram revealed an ejection fraction of 45% with anterior wall hypokinesia.",
    # 7. General non-specific symptoms
    "Complains of mild headache, dizziness, nausea, and persistent dry cough for the past three days.",
    # 8. Pulmonary infection and intravenous antibiotics
    "Chest radiograph showed right middle lobe infiltrate consistent with bacterial pneumonia; initiated on intravenous Ceftriaxone.",
    # 9. Interventional pulmonology procedure
    "A diagnostic bronchoscopy and bronchoalveolar lavage were performed without immediate complications.",
    # 10. Discharge prescription and monitoring
    "The patient was discharged with oral Amoxicillin 500 mg every 8 hours and advised to monitor blood glucose."
]

def main():
    rss_before = get_rss()
    _load()
    rss_loaded = get_rss()

    print("=" * 80)
    print("  Pretrained NLP Agent — 10 Sample Clinical Sentences (Written Examples)")
    print("=" * 80)

    total_time = 0.0
    all_latencies = []

    for idx, sentence in enumerate(SAMPLE_SENTENCES, 1):
        t0 = time.time()
        res = extract_entities(sentence)
        lat = time.time() - t0
        all_latencies.append(lat)
        total_time += lat

        print(f"\n[Example {idx}] (Latency: {lat*1000:.1f} ms)")
        print(f"  Sentence   : \"{sentence}\"")
        print(f"  DISEASE    : {[e['text'] for e in res['diseases']]}")
        print(f"  SYMPTOM    : {[e['text'] for e in res['symptoms']]}")
        print(f"  MEDICATION : {[e['text'] for e in res['medications']]}")
        print(f"  PROCEDURE  : {[e['text'] for e in res['procedures']]}")
        other_ents = [f"{e['text']} ({e['entity_type']})" for e in res['entities'] if e['group'] == 'OTHER']
        print(f"  OTHER RAW  : {other_ents}")

    rss_after = get_rss()
    avg_lat = sum(all_latencies) / len(all_latencies) * 1000.0

    print("\n" + "=" * 80)
    print("  Performance & Memory Summary")
    print("=" * 80)
    print(f"RSS before model load : {rss_before:.2f} MB")
    print(f"RSS after model load  : {rss_loaded:.2f} MB (Delta: +{rss_loaded - rss_before:.2f} MB)")
    print(f"RSS after 10 examples : {rss_after:.2f} MB (Delta: +{rss_after - rss_loaded:.2f} MB)")
    print(f"Average Latency       : {avg_lat:.2f} ms / sentence")
    print(f"Total Inference Time  : {total_time:.2f} s")
    print("=" * 80)

if __name__ == "__main__":
    main()

