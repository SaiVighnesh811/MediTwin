"""
agents/nlp_agent/train.py
MediTwin AI — Chapter 5: NLP Agent (Medical Entity Extraction)
Steps 0–6: dataset split, label mapping, baseline, fine-tune, evaluation, demo, module.

Dataset : ktgiahieu/maccrobat2018_2020 (400 documents, BIO-tagged)
Model   : dmis-lab/biobert-base-cased-v1.1 (or distilbert-base-uncased fallback)
Hardware: CPU-only (≤3 epochs, hard cap 45 min)
"""

import os, sys, json, time, re, copy, random
import warnings; warnings.filterwarnings("ignore")

os.environ["TOKENIZERS_PARALLELISM"] = "false"

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
from collections import defaultdict
from datasets import load_dataset
from utils.metrics_logger import log_metrics

RANDOM_SEED = 42
random.seed(RANDOM_SEED); np.random.seed(RANDOM_SEED)

MODELS_DIR = os.path.join(PROJECT_ROOT, "models", "nlp")
LOGS_DIR   = os.path.join(PROJECT_ROOT, "logs")
os.makedirs(MODELS_DIR, exist_ok=True)

def banner(msg):
    print(f"\n{'='*65}\n  {msg}\n{'='*65}")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 0 — Dataset
# ══════════════════════════════════════════════════════════════════════════════
banner("STEP 0 — Dataset")

ds = load_dataset("ktgiahieu/maccrobat2018_2020")
print(f"  Splits in HF dataset: {list(ds.keys())}")
print(f"  Total documents: {len(ds['train'])}")
print(f"  Features: {ds['train'].features}")

all_docs_raw = list(ds["train"])  # 400 rows

# ── Deduplicate: dataset has 200 unique token sequences, each appears twice
# (rows 0-199 and rows 200-399, same text, sometimes differing annotations).
# ASSUMPTION: rows 200-399 are the newer 2020 annotations (later occurrence).
# EVIDENCE: the HF dataset is named maccrobat2018_2020; pairs sit at a fixed
# offset of exactly 200 (verified by inspection: offset Counter {200: 200}).
# Pairs where tags differ: 11/200; pairs where tags are identical: 189/200.
# We keep the row with the HIGHER index (index ≥ 200) to prefer later annotations.

tok_to_best_idx = {}
for i, doc in enumerate(all_docs_raw):
    key = tuple(doc["tokens"])
    if key not in tok_to_best_idx or i > tok_to_best_idx[key]:
        tok_to_best_idx[key] = i

all_docs = [all_docs_raw[idx] for idx in sorted(tok_to_best_idx.values())]
print(f"  Raw rows: {len(all_docs_raw)} → after dedup: {len(all_docs)} unique documents")
print(f"  (Kept rows with index ≥ 200; assumption: newer 2020 annotations)")

# Collect all unique raw tag types
all_tags = set()
for doc in all_docs:
    all_tags.update(doc["tags"])
print(f"\n  Unique raw BIO tags ({len(all_tags)}):")
ent_types = sorted({t[2:] for t in all_tags if t not in ("O",)})
print(f"  Entity types ({len(ent_types)}): {ent_types}")

# ── Split BY DOCUMENT 70/15/15 on the 200 unique docs, seed=42
indices = list(range(len(all_docs)))
random.shuffle(indices)
n = len(all_docs)
n_train = int(0.70 * n)
n_val   = int(0.15 * n)
train_docs = [all_docs[i] for i in indices[:n_train]]
val_docs   = [all_docs[i] for i in indices[n_train:n_train+n_val]]
test_docs  = [all_docs[i] for i in indices[n_train+n_val:]]

# ── Assert no token sequence appears in more than one split
_train_keys = {tuple(d["tokens"]) for d in train_docs}
_val_keys   = {tuple(d["tokens"]) for d in val_docs}
_test_keys  = {tuple(d["tokens"]) for d in test_docs}
_tv = _train_keys & _val_keys
_tt = _train_keys & _test_keys
_vt = _val_keys   & _test_keys
assert len(_tv) == 0, f"Train/Val overlap: {len(_tv)} docs"
assert len(_tt) == 0, f"Train/Test overlap: {len(_tt)} docs"
assert len(_vt) == 0, f"Val/Test overlap: {len(_vt)} docs"
print("  ASSERTION PASSED: no token sequence appears in more than one split.")

def doc_stats(docs, name):
    n_toks  = sum(len(d["tokens"]) for d in docs)
    ent_counts = defaultdict(int)
    for d in docs:
        for t in d["tags"]:
            if t.startswith("B-"):
                ent_counts[t[2:]] += 1
    print(f"  {name}: {len(docs)} docs | {n_toks} tokens | entities: {dict(sorted(ent_counts.items()))}")
    return ent_counts

print()
train_ent = doc_stats(train_docs, "Train")
val_ent   = doc_stats(val_docs,   "Val  ")
test_ent  = doc_stats(test_docs,  "Test ")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Label mapping
# ══════════════════════════════════════════════════════════════════════════════
banner("STEP 1 — Label mapping")

# 4 target groups; everything else → O
GROUP_MAP = {
    "Disease_disorder":       "DISEASE",
    "Sign_symptom":           "SYMPTOM",
    "Medication":             "MEDICATION",
    "Therapeutic_procedure":  "PROCEDURE",
    "Diagnostic_procedure":   "PROCEDURE",
}
GROUPS = ["DISEASE", "SYMPTOM", "MEDICATION", "PROCEDURE"]
print("  Mapping:")
for k, v in GROUP_MAP.items():
    print(f"    {k} → {v}")
print("  All other labels → O")

def map_tags(tags):
    """Convert raw BIO tags to 4-group BIO tags."""
    out = []
    for t in tags:
        if t == "O":
            out.append("O")
        else:
            prefix, etype = t[0], t[2:]  # B or I
            grp = GROUP_MAP.get(etype)
            out.append(f"{prefix}-{grp}" if grp else "O")
    return out

def group_ent_count(docs, name):
    counts = defaultdict(int)
    for d in docs:
        mapped = map_tags(d["tags"])
        for t in mapped:
            if t.startswith("B-"):
                counts[t[2:]] += 1
    print(f"  {name}: {dict(sorted(counts.items()))}")
    return counts

print()
train_grp = group_ent_count(train_docs, "Train")
val_grp   = group_ent_count(val_docs,   "Val  ")
test_grp  = group_ent_count(test_docs,  "Test ")

# Label set for NER (BIO)
LABEL_LIST = ["O"] + [f"{p}-{g}" for g in GROUPS for p in ["B","I"]]
LABEL2ID   = {l: i for i, l in enumerate(LABEL_LIST)}
ID2LABEL   = {i: l for l, i in LABEL2ID.items()}
NUM_LABELS = len(LABEL_LIST)
print(f"\n  Label list ({NUM_LABELS}): {LABEL_LIST}")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — Dictionary-match baseline
# ══════════════════════════════════════════════════════════════════════════════
banner("STEP 2 — Dictionary-match baseline")

from seqeval.metrics import precision_score, recall_score, f1_score, classification_report

def build_entity_dict(docs):
    """Extract unique entity strings per group from documents (train only)."""
    ent_dict = defaultdict(set)
    for d in docs:
        tokens, tags = d["tokens"], d["tags"]
        i = 0
        while i < len(tokens):
            t = tags[i]
            if t.startswith("B-"):
                etype = t[2:]
                grp = GROUP_MAP.get(etype)
                if grp:
                    span = [tokens[i]]
                    j = i + 1
                    while j < len(tokens) and tags[j] == f"I-{etype}":
                        span.append(tokens[j]); j += 1
                    ent_dict[grp].add(" ".join(span).lower())
                    i = j
                else:
                    i += 1
            else:
                i += 1
    return ent_dict

def baseline_predict(doc, ent_dict):
    """Case-insensitive exact match against entity dictionary."""
    tokens = doc["tokens"]
    n = len(tokens)
    pred = ["O"] * n
    lower = [t.lower() for t in tokens]
    # Try longest match first
    for grp, strings in ent_dict.items():
        sorted_strings = sorted(strings, key=len, reverse=True)
        for ent_str in sorted_strings:
            words = ent_str.split()
            wlen  = len(words)
            for start in range(n - wlen + 1):
                if lower[start:start+wlen] == words:
                    if pred[start] == "O":
                        pred[start] = f"B-{grp}"
                        for k in range(1, wlen):
                            pred[start+k] = f"I-{grp}"
    return pred

ent_dict = build_entity_dict(train_docs)
print(f"  Dictionary sizes: { {k: len(v) for k,v in ent_dict.items()} }")

def evaluate_seqeval(docs, predict_fn, name):
    y_true, y_pred = [], []
    for d in docs:
        true_tags = map_tags(d["tags"])
        pred_tags = predict_fn(d)
        y_true.append(true_tags)
        y_pred.append(pred_tags)
    p  = precision_score(y_true, y_pred, average="micro",  zero_division=0)
    r  = recall_score(   y_true, y_pred, average="micro",  zero_division=0)
    f  = f1_score(       y_true, y_pred, average="micro",  zero_division=0)
    mp = precision_score(y_true, y_pred, average="macro",  zero_division=0)
    mr = recall_score(   y_true, y_pred, average="macro",  zero_division=0)
    mf = f1_score(       y_true, y_pred, average="macro",  zero_division=0)
    print(f"\n  {name} results:")
    print(f"    Micro  P={p:.4f}  R={r:.4f}  F1={f:.4f}")
    print(f"    Macro  P={mp:.4f}  R={mr:.4f}  F1={mf:.4f}")
    print(classification_report(y_true, y_pred, zero_division=0))
    return {"micro_p":round(p,4),"micro_r":round(r,4),"micro_f1":round(f,4),
            "macro_p":round(mp,4),"macro_r":round(mr,4),"macro_f1":round(mf,4)}

baseline_fn = lambda d: baseline_predict(d, ent_dict)
baseline_results = evaluate_seqeval(test_docs, baseline_fn, "BASELINE (test)")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — Fine-tune BioBERT
# ══════════════════════════════════════════════════════════════════════════════
banner("STEP 3 — Fine-tune BioBERT")

import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (AutoTokenizer, AutoModelForTokenClassification,
                          get_linear_schedule_with_warmup)

DEVICE    = torch.device("cpu")
MAX_LEN   = 128    # keeps RAM < 5.7 GB on CPU
BATCH     = 8     # lowered for 5.7 GB RAM machine
LR        = 5e-5   # raised from 2e-5: DistilBERT classifier head is random-init
MAX_EP    = 15
PATIENCE  = 4      # early stopping: stop if val F1 doesn't improve for 4 epochs

MODEL_NAME = "dmis-lab/biobert-base-cased-v1.2"
print(f"  Loading tokenizer: {MODEL_NAME}")
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=True)
    print(f"  Tokenizer OK  [model: {MODEL_NAME}  reason: primary choice, in local HF cache]")
except Exception as e:
    print(f"  BioBERT tokenizer failed: {e}")
    MODEL_NAME = "distilbert-base-uncased"
    tokenizer  = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=True)
    print(f"  *** USING FALLBACK: {MODEL_NAME} (DistilBERT) — BioBERT failed above ***")

print(f"  Loading model: {MODEL_NAME}")
try:
    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME, num_labels=NUM_LABELS,
        id2label=ID2LABEL, label2id=LABEL2ID,
        ignore_mismatched_sizes=True,
        local_files_only=True,
    )
    print(f"  Model loaded OK  [loaded: {MODEL_NAME}]")
except Exception as e:
    if MODEL_NAME != "distilbert-base-uncased":
        print(f"  BioBERT model failed: {e}")
        MODEL_NAME = "distilbert-base-uncased"
        print(f"  *** USING FALLBACK: {MODEL_NAME} (DistilBERT) — BioBERT model failed above ***")
        model = AutoModelForTokenClassification.from_pretrained(
            MODEL_NAME, num_labels=NUM_LABELS,
            id2label=ID2LABEL, label2id=LABEL2ID,
            local_files_only=True,
        )
        print(f"  Fallback model loaded OK  [loaded: {MODEL_NAME}]")
    else:
        raise

model = model.to(DEVICE)

# ── Tokenise docs → HF-style word-piece alignment ────────────────────────────
class NERDataset(Dataset):
    def __init__(self, docs):
        self.items = []
        for doc in docs:
            tokens = doc["tokens"]
            tags   = map_tags(doc["tags"])
            enc = tokenizer(
                tokens,
                is_split_into_words=True,
                max_length=MAX_LEN,
                truncation=True,
                padding="max_length",
                return_offsets_mapping=False,
            )
            word_ids = enc.word_ids()
            labels = []
            prev_wid = None
            for wid in word_ids:
                if wid is None:
                    labels.append(-100)
                elif wid != prev_wid:
                    labels.append(LABEL2ID.get(tags[wid] if wid < len(tags) else "O", 0))
                else:
                    # Sub-token: propagate B→I for same entity, else -100
                    raw = tags[wid] if wid < len(tags) else "O"
                    if raw.startswith("B-"):
                        labels.append(LABEL2ID.get("I-" + raw[2:], 0))
                    elif raw.startswith("I-"):
                        labels.append(LABEL2ID.get(raw, 0))
                    else:
                        labels.append(-100)
                prev_wid = wid
            self.items.append({
                "input_ids":      torch.tensor(enc["input_ids"],      dtype=torch.long),
                "attention_mask": torch.tensor(enc["attention_mask"], dtype=torch.long),
                "labels":         torch.tensor(labels,                dtype=torch.long),
            })
            if hasattr(enc, "token_type_ids"):
                self.items[-1]["token_type_ids"] = torch.tensor(enc["token_type_ids"], dtype=torch.long)

    def __len__(self): return len(self.items)
    def __getitem__(self, i): return self.items[i]

print("  Tokenising train …", end=" ", flush=True)
train_ds = NERDataset(train_docs); print(f"{len(train_ds)} items")
print("  Tokenising val …",   end=" ", flush=True)
val_ds   = NERDataset(val_docs);   print(f"{len(val_ds)} items")
print("  Tokenising test …",  end=" ", flush=True)
test_ds  = NERDataset(test_docs);  print(f"{len(test_ds)} items")

train_ld = DataLoader(train_ds, batch_size=BATCH, shuffle=True)
val_ld   = DataLoader(val_ds,   batch_size=BATCH, shuffle=False)
test_ld  = DataLoader(test_ds,  batch_size=BATCH, shuffle=False)

optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
total_steps = MAX_EP * len(train_ld)
scheduler   = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=int(0.1*total_steps), num_training_steps=total_steps)

# ── Detect latest resumable checkpoint ───────────────────────────────────────
START_EP   = 1
best_val_f1 = 0.0
history     = []

existing_ckpts = sorted(
    [f for f in os.listdir(MODELS_DIR) if f.startswith("biobert_ner_ep") and f.endswith(".pt")],
    key=lambda s: int(s.replace("biobert_ner_ep","").replace(".pt",""))
)
if existing_ckpts:
    latest = existing_ckpts[-1]
    latest_path = os.path.join(MODELS_DIR, latest)
    print(f"\n  Resuming from checkpoint: {latest}")
    ckpt_data = torch.load(latest_path, map_location=DEVICE, weights_only=False)
    if isinstance(ckpt_data, dict) and "model_state" in ckpt_data:
        model.load_state_dict(ckpt_data["model_state"])
        optimizer.load_state_dict(ckpt_data["optimizer_state"])
        scheduler.load_state_dict(ckpt_data["scheduler_state"])
        START_EP    = ckpt_data["epoch"] + 1
        best_val_f1 = ckpt_data.get("best_val_f1", 0.0)
        history     = ckpt_data.get("history", [])
    else:
        # Legacy: weights-only saved by old code
        model.load_state_dict(ckpt_data)
        ep_num   = int(latest.replace("biobert_ner_ep","").replace(".pt",""))
        START_EP = ep_num + 1
    print(f"  Will continue from epoch {START_EP} (best_val_f1 so far: {best_val_f1:.4f})")
else:
    # Benchmark only on fresh start
    print("\n  Benchmarking 2 batches …")
    model.train()
    it = iter(train_ld)
    t0 = time.time()
    for _ in range(2):
        batch = next(it)
        ids   = batch["input_ids"].to(DEVICE)
        mask  = batch["attention_mask"].to(DEVICE)
        lbls  = batch["labels"].to(DEVICE)
        kw    = {"input_ids": ids, "attention_mask": mask, "labels": lbls}
        if "token_type_ids" in batch:
            kw["token_type_ids"] = batch["token_type_ids"].to(DEVICE)
        out = model(**kw)
        out.loss.backward()
        optimizer.step(); scheduler.step(); optimizer.zero_grad()
    bench_t = time.time() - t0
    batches_per_ep = len(train_ld)
    est_ep    = bench_t / 2 * batches_per_ep
    est_total = est_ep * MAX_EP
    print(f"  2 batches: {bench_t:.1f}s → estimated {est_ep/60:.1f} min/epoch")
    print(f"  Estimated total ({MAX_EP} epochs): {est_total/60:.1f} min")
    if est_total > 45*60:
        MAX_EP = max(1, int(44*60 / est_ep))
        print(f"  ⚠ Over 45-min cap → reducing to {MAX_EP} epochs")
    else:
        print(f"  Within 45-min cap. Proceeding with {MAX_EP} epochs.")

# ── Training loop ─────────────────────────────────────────────────────────────
def eval_loss(loader):
    model.eval()
    tot, n = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            ids  = batch["input_ids"].to(DEVICE)
            mask = batch["attention_mask"].to(DEVICE)
            lbls = batch["labels"].to(DEVICE)
            kw   = {"input_ids": ids, "attention_mask": mask, "labels": lbls}
            if "token_type_ids" in batch:
                kw["token_type_ids"] = batch["token_type_ids"].to(DEVICE)
            loss = model(**kw).loss
            tot += loss.item(); n += 1
    return tot / n if n else 0


def eval_f1(docs_subset):
    """Compute entity-level micro-F1 using seqeval.
    Returns (f1, n_pred_entities, n_true_entities).
    """
    model.eval()
    y_true, y_pred = [], []
    for doc in docs_subset:
        true_tags = map_tags(doc["tags"])
        tokens    = doc["tokens"]
        enc = tokenizer(
            tokens, is_split_into_words=True,
            max_length=MAX_LEN, truncation=True,
            padding="max_length", return_offsets_mapping=False,
        )
        word_ids = enc.word_ids()
        with torch.no_grad():
            ids_t  = torch.tensor([enc["input_ids"]],      dtype=torch.long)
            mask_t = torch.tensor([enc["attention_mask"]], dtype=torch.long)
            kw     = {"input_ids": ids_t, "attention_mask": mask_t}
            if "token_type_ids" in enc.keys():
                kw["token_type_ids"] = torch.tensor([enc["token_type_ids"]], dtype=torch.long)
            logits = model(**kw).logits[0]
        preds   = logits.argmax(-1).tolist()
        pred_tags, seen = [], set()
        for wid, pid in zip(word_ids, preds):
            if wid is None or wid in seen: continue
            seen.add(wid)
            pred_tags.append(ID2LABEL[pid])
        while len(pred_tags) < len(tokens): pred_tags.append("O")
        y_true.append(true_tags)
        y_pred.append(pred_tags[:len(tokens)])
    f1 = f1_score(y_true, y_pred, average="micro", zero_division=0)
    n_pred = sum(1 for seq in y_pred for t in seq if t.startswith("B-"))
    n_true = sum(1 for seq in y_true for t in seq if t.startswith("B-"))
    return f1, n_pred, n_true


banner("Training loop")
print(f"  {'Ep':>3}  {'TrLoss':>8}  {'VaLoss':>8}  {'VaF1':>7}  {'Pred':>6}  {'True':>6}  Note")
print("  " + "-"*62)

best_ckpt    = os.path.join(MODELS_DIR, "biobert_ner_best.pt")
no_improve   = 0  # early-stopping counter

if START_EP > MAX_EP:
    print(f"  All {MAX_EP} epochs already completed. Skipping training.")
else:
    for ep in range(START_EP, MAX_EP + 1):
        model.train()
        t0 = time.time()
        tr_loss, nb = 0.0, 0
        for batch in train_ld:
            ids  = batch["input_ids"].to(DEVICE)
            mask = batch["attention_mask"].to(DEVICE)
            lbls = batch["labels"].to(DEVICE)
            kw   = {"input_ids": ids, "attention_mask": mask, "labels": lbls}
            if "token_type_ids" in batch:
                kw["token_type_ids"] = batch["token_type_ids"].to(DEVICE)
            optimizer.zero_grad()
            out = model(**kw)
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); scheduler.step()
            tr_loss += out.loss.item(); nb += 1
        tr_loss /= nb

        va_loss           = eval_loss(val_ld)
        va_f1, n_pred, n_true = eval_f1(val_docs)
        elapsed = time.time() - t0
        note    = ""
        if va_f1 > best_val_f1:
            best_val_f1 = va_f1
            no_improve  = 0
            note = "✓ best"
            torch.save(model.state_dict(), best_ckpt)
        else:
            no_improve += 1

        ep_ckpt_path = os.path.join(MODELS_DIR, f"biobert_ner_ep{ep}.pt")
        history.append({"epoch": ep, "tr_loss": round(tr_loss, 4),
                        "va_loss": round(va_loss, 4), "va_f1": round(va_f1, 4)})
        torch.save({
            "epoch":           ep,
            "model_state":     model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_val_f1":     best_val_f1,
            "history":         history,
        }, ep_ckpt_path)
        print(f"  {ep:>3}  {tr_loss:>8.4f}  {va_loss:>8.4f}  {va_f1:>7.4f}  {n_pred:>6}  {n_true:>6}  {note}  ({elapsed:.0f}s)")
        if no_improve >= PATIENCE:
            print(f"  Early stopping: no F1 improvement for {PATIENCE} epochs.")
            break

# Reload best
if os.path.exists(best_ckpt):
    ckpt_data = torch.load(best_ckpt, map_location=DEVICE, weights_only=False)
    if isinstance(ckpt_data, dict) and "model_state" in ckpt_data:
        model.load_state_dict(ckpt_data["model_state"])
    else:
        model.load_state_dict(ckpt_data)
    print(f"\n  Best checkpoint restored (best_val_f1={best_val_f1:.4f})")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — Evaluation (test split, untouched)
# ══════════════════════════════════════════════════════════════════════════════
banner("STEP 4 — Fine-tuned model evaluation (test split)")

def model_predict_doc(doc):
    """Return per-token group predictions for one document."""
    tokens = doc["tokens"]
    enc = tokenizer(
        tokens, is_split_into_words=True,
        max_length=MAX_LEN, truncation=True,
        padding="max_length", return_offsets_mapping=False,
    )
    word_ids = enc.word_ids()
    with torch.no_grad():
        ids  = torch.tensor([enc["input_ids"]],      dtype=torch.long)
        mask = torch.tensor([enc["attention_mask"]], dtype=torch.long)
        kw   = {"input_ids": ids, "attention_mask": mask}
        if "token_type_ids" in enc:
            kw["token_type_ids"] = torch.tensor([enc["token_type_ids"]], dtype=torch.long)
        logits = model(**kw).logits[0]  # (seq_len, num_labels)
    preds = logits.argmax(-1).numpy()

    # Align back to words (take first sub-token prediction per word)
    pred_tags = []
    seen = set()
    for wid, pred_id in zip(word_ids, preds):
        if wid is None or wid in seen:
            continue
        seen.add(wid)
        pred_tags.append(ID2LABEL[pred_id])
    # Pad/truncate to actual token length
    while len(pred_tags) < len(tokens):
        pred_tags.append("O")
    return pred_tags[:len(tokens)]

ft_results = evaluate_seqeval(test_docs, model_predict_doc, "FINE-TUNED (test)")

# ── Baseline vs Fine-tuned table ─────────────────────────────────────────────
print("\n  === BASELINE vs FINE-TUNED ===")
print(f"  {'Metric':>15}  {'Baseline':>10}  {'Fine-tuned':>12}")
print("  " + "-"*42)
for met, bkey, fkey in [
    ("Micro P",  "micro_p",  "micro_p"),
    ("Micro R",  "micro_r",  "micro_r"),
    ("Micro F1", "micro_f1", "micro_f1"),
    ("Macro P",  "macro_p",  "macro_p"),
    ("Macro R",  "macro_r",  "macro_r"),
    ("Macro F1", "macro_f1", "macro_f1"),
]:
    b = baseline_results[bkey]; f = ft_results[fkey]
    delta = f - b
    sign  = "+" if delta>=0 else ""
    print(f"  {met:>15}  {b:>10.4f}  {f:>12.4f}  ({sign}{delta:.4f})")

# ── 5 error examples ─────────────────────────────────────────────────────────
print("\n  === 5 ERROR EXAMPLES ===")
err_count = 0
for doc in test_docs:
    if err_count >= 5: break
    true_tags = map_tags(doc["tags"])
    pred_tags = model_predict_doc(doc)
    tokens    = doc["tokens"]
    for i, (tok, gt, pr) in enumerate(zip(tokens, true_tags, pred_tags)):
        if gt != pr and gt != "O" and err_count < 5:
            ctx = " ".join(tokens[max(0,i-3):i+4])
            print(f"    [{err_count+1}] token='{tok}' true={gt} pred={pr}  context: …{ctx}…")
            err_count += 1

log_metrics("nlp_agent", {
    "model": MODEL_NAME,
    "epochs": len(history),
    "training_history": history,
    "best_val_f1": round(best_val_f1, 4),
    "baseline_test": baseline_results,
    "finetuned_test": ft_results,
    "label_list": LABEL_LIST,
    "train_docs": len(train_docs), "val_docs": len(val_docs), "test_docs": len(test_docs),
    "note": "CPU-only, BIO NER, seqeval strict entity-level scoring."
})

# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 — MTSamples demo
# ══════════════════════════════════════════════════════════════════════════════
banner("STEP 5 — MTSamples qualitative demo")

kaggle_token = os.path.expanduser("~/.kaggle/access_token")
if not os.path.exists(kaggle_token):
    print("  ~/.kaggle/access_token NOT found. Skipping Step 5.")
else:
    print("  ~/.kaggle/access_token found. Downloading MTSamples …")
    try:
        import kagglehub, pandas as pd
        mt_path = kagglehub.dataset_download("tboyle10/medicaltranscriptions")
        csv_files = []
        for root, _, files in os.walk(mt_path):
            for f in files:
                if f.endswith(".csv"): csv_files.append(os.path.join(root,f))
        if not csv_files:
            print("  No CSV found in MTSamples download."); raise ValueError("no csv")
        mt_df = pd.read_csv(csv_files[0])
        print(f"  MTSamples loaded: {mt_df.shape[0]} rows, cols: {list(mt_df.columns)[:6]}")
        text_col = "transcription" if "transcription" in mt_df.columns else mt_df.columns[-1]
        notes = mt_df[text_col].dropna().head(20).tolist()
        print(f"  Running model on {len(notes)} notes (qualitative demo only — NO labels).")
        print("  ⚠ MTSamples has no NER labels. This is a qualitative demo only.\n")
        for j, note in enumerate(notes[:3]):
            # Chunk to 300 chars for display
            preview = note[:300].replace("\n"," ")
            doc_sim = {"tokens": preview.split(), "tags": ["O"]*len(preview.split())}
            preds = model_predict_doc(doc_sim)
            entities = []
            i = 0
            toks = preview.split()
            while i < len(preds):
                if preds[i].startswith("B-"):
                    grp = preds[i][2:]
                    span = [toks[i]]; k = i+1
                    while k < len(preds) and preds[k] == f"I-{grp}":
                        span.append(toks[k]); k += 1
                    entities.append((grp, " ".join(span)))
                    i = k
                else:
                    i += 1
            print(f"  Note {j+1} entities: {entities[:10]}")
    except Exception as ex:
        print(f"  MTSamples step failed: {ex}")

# ══════════════════════════════════════════════════════════════════════════════
# Done — module file will be created separately by ocr_agent.py pattern
# ══════════════════════════════════════════════════════════════════════════════
banner("Training complete")
print(f"  Best checkpoint: {best_ckpt}")
print(f"  Training history: {history}")
# Save label list and model name for the module
meta = {"model_name": MODEL_NAME, "label_list": LABEL_LIST,
        "id2label": ID2LABEL, "label2id": LABEL2ID, "max_len": MAX_LEN}
with open(os.path.join(MODELS_DIR, "nlp_model_meta.json"), "w") as f:
    json.dump(meta, f, indent=2)
print("  Metadata saved to models/nlp/nlp_model_meta.json")
