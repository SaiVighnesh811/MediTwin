#!/usr/bin/env python3
"""MediTwin AI - project progress viewer.

READ-ONLY. It never trains, downloads, or changes anything. It only:
  1. prints the chapter table from PROJECT_TRACKER.md
  2. checks which key output files really exist on disk
  3. prints the latest saved metrics from logs/*metrics*.json
  4. lists the plots / explanation pages / reports
Standard library only, so it works even without the virtual environment.

Put this file in the meditwin-ai folder, then run:
  python show_progress.py                 # everything
  python show_progress.py tracker         # chapter status table
  python show_progress.py evidence        # which key files exist
  python show_progress.py metrics         # latest saved metrics per component
  python show_progress.py figures         # list plots and pages
  python show_progress.py figures --open  # open them one by one (Enter = next)
  python show_progress.py reports         # list written chapter reports
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LINE = "=" * 72

# (label, path relative to project root). Paths come from the chapter reports.
EXPECTED = [
    ("Risk Prediction v2 model", "models/risk_prediction_v2/xgboost_v2_model.pkl"),
    ("Risk Prediction v2 metadata", "models/risk_prediction_v2/model_metadata.json"),
    ("XAI module", "agents/xai_agent/xai_agent.py"),
    ("XAI metrics log", "logs/xai_agent_metrics.json"),
    ("Medical image checkpoint", "models/medical_image/resnet18_xray.pt"),
    ("Medical image module", "agents/medical_image_agent/medical_image_agent.py"),
    ("OCR handwriting model", "models/ocr/handwriting_cnn.pt"),
    ("OCR agent module", "agents/ocr_agent/ocr_agent.py"),
    ("NLP model folder", "models/nlp"),
    ("NLP agent module", "agents/nlp_agent/nlp_agent.py"),
    ("Chapter 2 report", "reports/chapter_2_report.md"),
    ("Chapter 3 report (final)", "reports/chapter_3_report.md"),
    ("Chapter 3 draft", "reports/chapter_3_DRAFT.md"),
    ("Chapter 4 report", "reports/chapter_4_report.md"),
    ("Chapter 5 report", "reports/chapter_5_report.md"),
]


def head(title):
    print("\n" + LINE)
    print(title)
    print(LINE)


def human(n):
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024.0


def show_tracker():
    head("1. CHAPTER STATUS (from PROJECT_TRACKER.md)")
    p = ROOT / "PROJECT_TRACKER.md"
    if not p.exists():
        print("PROJECT_TRACKER.md not found in", ROOT)
        return
    rows = []
    for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
        s = line.strip()
        if not s.startswith("|"):
            continue
        cells = [c.strip() for c in s.strip("|").split("|")]
        if len(cells) < 3 or set("".join(cells)) <= set("-: "):
            continue
        rows.append(cells)
    if not rows:
        print("No table rows found in the tracker.")
        return
    for i, c in enumerate(rows):
        print(f"{c[0]:<8} {c[1]:<30} {c[2]}")
        if i == 0:
            print("-" * 72)
        elif len(c) > 5 and c[5]:
            note = c[5] if len(c[5]) <= 150 else c[5][:147] + "..."
            print(f"{'':<8} note: {note}")


def show_evidence():
    head("2. EVIDENCE CHECK (is the file really on disk?)")
    for label, rel in EXPECTED:
        p = ROOT / rel
        if p.is_dir():
            files = [f for f in p.rglob("*") if f.is_file()]
            status = f"OK ({len(files)} files)" if files else "MISSING (folder is empty)"
        elif p.exists():
            status = f"OK ({human(p.stat().st_size)})"
        else:
            status = "MISSING"
        print(f"{label:<34} {status:<28} {rel}")
    print("\nA chapter counts as finished only if the tracker says Done AND its"
          "\nmodel/log files above show OK.")


def fmt(v):
    return f"{v:.4f}" if isinstance(v, float) else str(v)


def render_list(key, v, lines, pad):
    scalars = (int, float, str, bool, type(None))
    if v and all(isinstance(r, list) and len(r) <= 12 and all(isinstance(x, scalars) for x in r) for r in v) and len(v) <= 12:
        lines.append(f"{pad}{key}:")
        for r in v:
            lines.append(f"{pad}  " + "  ".join(f"{fmt(x):>8}" for x in r))
    elif len(v) <= 8 and all(isinstance(x, scalars) for x in v):
        lines.append(f"{pad}{key}: [" + ", ".join(fmt(x) for x in v) + "]")
    else:
        lines.append(f"{pad}{key}: [{len(v)} items]")


def render(obj, lines, indent=0, depth=0):
    pad = "  " * indent
    if not isinstance(obj, dict):
        lines.append(pad + fmt(obj))
        return
    for k, v in obj.items():
        if isinstance(v, dict):
            if depth >= 4:
                lines.append(f"{pad}{k}: {{...}}")
            else:
                lines.append(f"{pad}{k}:")
                render(v, lines, indent + 1, depth + 1)
        elif isinstance(v, list):
            render_list(k, v, lines, pad)
        elif isinstance(v, str) and "\n" in v:
            lines.append(f"{pad}{k}:")
            for ln in v.splitlines()[:15]:
                lines.append(f"{pad}  {ln}")
        else:
            lines.append(f"{pad}{k}: {fmt(v)}")


def show_metrics():
    head("3. LATEST SAVED METRICS (from logs/*metrics*.json)")
    files = sorted((ROOT / "logs").glob("*metrics*.json"))
    if not files:
        print("No metrics files found in logs/.")
        return
    for f in files:
        print(f"\n--- {f.name} ---")
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except Exception as e:  # unreadable or not JSON
            print(f"(could not read: {e})")
            continue
        note = ""
        if isinstance(data, list) and data:
            note = f"({len(data)} saved entries; showing the latest)"
            data = data[-1]
        if note:
            print(note)
        lines = []
        render(data if isinstance(data, dict) else {"value": data}, lines)
        for ln in lines[:70]:
            print(ln)
        if len(lines) > 70:
            print(f"... ({len(lines) - 70} more lines in the file)")


def open_file(path):
    try:
        if sys.platform.startswith("win"):
            os.startfile(str(path))  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(path)])
        else:
            subprocess.Popen(["xdg-open", str(path)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception as e:
        print(f"   (could not open: {e})")


def show_figures(do_open):
    head("4. PLOTS AND EXPLANATION PAGES (logs/)")
    figs = sorted(list((ROOT / "logs").glob("*.png")) + list((ROOT / "logs").glob("*.html")))
    if not figs:
        print("No .png/.html files found in logs/.")
        return
    for i, f in enumerate(figs, 1):
        print(f"{i:>2}. {f.name}")
    if do_open:
        print("\nOpening one at a time. Press Enter for the next, or type q to stop.")
        for f in figs:
            print("->", f.name)
            open_file(f)
            try:
                if input("   [Enter = next, q = quit] ").strip().lower() == "q":
                    break
            except EOFError:
                break
            time.sleep(0.2)


def show_reports():
    head("5. WRITTEN REPORTS (reports/)")
    files = sorted((ROOT / "reports").glob("*.md"))
    if not files:
        print("No reports found.")
        return
    for f in files:
        print(f"{f.name:<32} {human(f.stat().st_size)}")
    print("\nRead one with:  cat reports/chapter_4_report.md")


def main():
    ap = argparse.ArgumentParser(description="MediTwin AI progress viewer (read-only)")
    ap.add_argument("section", nargs="?", default="all",
                    choices=["all", "tracker", "evidence", "metrics", "figures", "reports"])
    ap.add_argument("--open", action="store_true", help="with 'figures': open files one by one")
    a = ap.parse_args()

    print("MediTwin AI - project progress")
    print("Project folder:", ROOT)
    if a.section in ("all", "tracker"):
        show_tracker()
    if a.section in ("all", "evidence"):
        show_evidence()
    if a.section in ("all", "metrics"):
        show_metrics()
    if a.section in ("all", "figures"):
        show_figures(a.open)
    if a.section in ("all", "reports"):
        show_reports()
    print("\n(Read-only view. Nothing was trained, downloaded or changed.)")


if __name__ == "__main__":
    main()