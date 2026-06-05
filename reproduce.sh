#!/bin/bash
# ============================================================
# reproduce.sh — Reproduce results for:
#   "Assessing the impact of dimensionality reduction on clustering performance"
#
# Usage:
#   bash reproduce.sh          # Quick mode: regenerate outputs from cached results
#   bash reproduce.sh --full   # Full mode: re-run all experiments from scratch (~3-4 hours)
#
# The quick mode uses pre-computed JSON result files to regenerate
# all tables (CSV) and boxplots (PDF). The full mode runs the entire
# experiment pipeline including dataset loading, DR, clustering, and evaluation.
# ============================================================
set -e

echo "=== Step 1: Install dependencies ==="
pip install -q numpy pandas scikit-learn scipy matplotlib torch repliclust ucimlrepo 2>/dev/null

if [ "$1" == "--full" ]; then
    echo "=== Step 2: Run FULL experiment pipeline ==="
    echo "  This runs all experiments (real-world + synthetic) and saves JSON results."
    echo "  Expected runtime: ~3-4 hours on a single GPU machine."
    python3 main_pipeline.py
else
    echo "=== Step 2: Using cached results (quick mode) ==="
    echo "  To re-run experiments from scratch, use: bash reproduce.sh --full"
    # Verify cached results exist
    for f in results/results_RealWorld.json results/results_Circles.json results/results_Moons.json results/results_RSG.json results/results_Repliclust.json; do
        if [ ! -f "$f" ]; then
            echo "ERROR: Missing cached result file: $f"
            echo "Run with --full to generate from scratch."
            exit 1
        fi
    done
    echo "  All cached result files found."
fi

echo "=== Step 3: Generate all output tables and boxplots ==="
python3 generate_all_outputs.py

echo "=== Step 4: List key outputs ==="
echo ""
echo "Key result files in results/:"
echo ""
echo "  Combined aggregate tables (Paper Tables 1-4, one per algorithm):"
ls results/table_combined_aggregate_*.csv 2>/dev/null
echo ""
echo "  Wilcoxon signed-rank test results (Paper Table 5):"
ls results/table_wilcoxon_RealWorld.csv 2>/dev/null
echo ""
echo "  Per-dataset ARI tables (real-world, Paper Tables 9-12):"
ls results/table_k-means_RealWorld.csv results/table_AHC_RealWorld.csv results/table_GMM_RealWorld.csv results/table_OPTICS_RealWorld.csv 2>/dev/null
echo ""
echo "  Average ARI tables for synthetic data (Paper Tables 3-6):"
ls results/table_average_ARI_*.csv 2>/dev/null
echo ""
echo "  Boxplots (Paper Figures 1-8):"
echo "  $(ls results/boxplot_*.pdf 2>/dev/null | wc -l) PDF boxplot files total"
echo ""
echo "=== Done! All results are in results/ ==="
