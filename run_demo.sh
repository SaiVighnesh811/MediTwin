#!/usr/bin/env bash
# run_demo.sh — MediTwin AI full demo runner
# Usage: bash run_demo.sh

set -o pipefail

PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$PROJECT/venv/bin/activate"

if [ ! -f "$VENV" ]; then
    echo "ERROR: venv not found at $VENV"; exit 1
fi
source "$VENV"

cd "$PROJECT"

# Declare demos in order
DEMOS=(
    "show_progress.py"
    "demo/demo_risk.py"
    "demo/demo_xai.py"
    "demo/demo_ocr.py"
    "demo/demo_medical_image.py"
    "demo/demo_nlp.py"
    "demo/demo_memory.py"
    "demo/demo_recommendation.py"
    "demo/demo_pipeline.py"
)

declare -A RESULTS
declare -A TIMES

echo ""
echo "================================================================="
echo "  MediTwin AI — Full Demo Suite"
echo "================================================================="
echo ""

for script in "${DEMOS[@]}"; do
    name=$(basename "$script" .py)
    echo "----------------------------------------------------------------"
    echo "  Running: $script"
    echo "----------------------------------------------------------------"
    t_start=$(date +%s%N)
    python "$script"
    exit_code=$?
    t_end=$(date +%s%N)
    runtime_ms=$(( (t_end - t_start) / 1000000 ))

    if [ $exit_code -eq 0 ]; then
        # Check if SKIPPED was printed
        if python "$script" 2>&1 | grep -q "SKIPPED"; then
            RESULTS[$name]="SKIPPED"
        else
            RESULTS[$name]="PASS"
        fi
    else
        RESULTS[$name]="FAIL (exit $exit_code)"
    fi
    TIMES[$name]="${runtime_ms}ms"
    echo ""
done

echo "================================================================="
echo "  SUMMARY"
echo "================================================================="
printf "  %-30s  %-12s  %s\n" "Demo" "Result" "Runtime"
echo "  -----------------------------------------------------------------"
for script in "${DEMOS[@]}"; do
    name=$(basename "$script" .py)
    printf "  %-30s  %-12s  %s\n" "$name" "${RESULTS[$name]}" "${TIMES[$name]}"
done
echo "================================================================="
