#!/bin/bash
# ============================================================
# reproduce.sh — Reproduce results for:
#   "Assessing the impact of dimensionality reduction on clustering performance"
#
# This script performs the full pipeline:
#   1. Installs dependencies
#   2. Runs the experiment pipeline (main_pipeline.py)
#      - Loads/generates datasets (20 UCI + 4 synthetic types)
#      - Applies 5 DR methods × 3 reduction levels
#      - Runs 4 clustering algorithms with hyperparameter search 
#      - Computes ARI scores
#      - Saves raw results as JSON
#   3. Generates all output tables (CSV) and boxplots (PDF)
#      from the cached JSON results using generate_all_outputs.py
#
# Total runtime: ~60-90 minutes on a single GPU machine.
# To skip re-running experiments and use cached results:
#   python3 generate_all_outputs.py
# ============================================================
set -e

echo "=== Step 1: Install dependencies ==="
pip install -q numpy pandas scikit-learn scipy matplotlib torch repliclust ucimlrepo 2>/dev/null

echo "=== Step 2: Run experiment pipeline ==="
echo "  This runs all experiments (real-world + synthetic) and saves JSON results."
echo "  If results JSON files already exist, the pipeline will overwrite them."
python3 main_pipeline.py

echo "=== Step 3: Generate all output tables and boxplots ==="
python3 generate_all_outputs.py

echo "=== Step 4: List key outputs ==="
echo ""
echo "Key result files in results/:"
echo "  Per-dataset ARI tables (corresponds to paper Tables 9-12):"
ls results/table_k-means_RealWorld.csv results/table_AHC_RealWorld.csv results/table_GMM_RealWorld.csv results/table_OPTICS_RealWorld.csv 2>/dev/null
echo ""
echo "  Average ARI tables for synthetic data (corresponds to paper Tables 3-6):"
ls results/table_average_ARI_*.csv 2>/dev/null
echo ""
echo "  Aggregate win%/loss% tables (corresponds to paper Tables 7-10):"
ls results/table_aggregate_*.csv 2>/dev/null
echo ""
echo "  Combined aggregate tables (synth+real, one per algorithm):"
ls results/table_combined_aggregate_*.csv 2>/dev/null
echo ""
echo "  Wilcoxon signed-rank test results (corresponds to paper Table 11):"
ls results/table_wilcoxon*.csv 2>/dev/null
echo ""
echo "  Boxplots (corresponds to paper Figures 7-11):"
ls results/boxplot_*.pdf 2>/dev/null | wc -l
echo "  PDF boxplot files total"
echo ""
echo "=== Done! All results are in results/ ==="
