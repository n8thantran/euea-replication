#!/bin/bash
# Reproduce key results from:
# "Assessing the impact of dimensionality reduction on clustering performance"
#
# This script runs the complete experiment pipeline:
# 1. Real-world experiments (20 UCI datasets × 5 DR methods × 3 levels × 4 clustering algos)
# 2. Synthetic experiments (Circles, Moons, RSG, Repliclust)
# 3. Generate combined aggregate tables (Tables 1-4)
# 4. Generate Wilcoxon signed-rank test table (Table 5)
# 5. Generate boxplots (Figures 1-8)
#
# Usage:
#   ./reproduce.sh          # Full run (may take several hours)
#   ./reproduce.sh quick    # Only regenerate tables/plots from cached results
#   ./reproduce.sh real     # Only run real-world experiments
#   ./reproduce.sh synth    # Only run synthetic experiments

set -e

echo "=============================================="
echo "Reproducing: DR + Clustering Paper Results"
echo "=============================================="

# Install dependencies
pip install -q numpy pandas scikit-learn scipy matplotlib torch ucimlrepo repliclust 2>/dev/null

cd /workspace

MODE=${1:-full}

if [ "$MODE" = "quick" ]; then
    echo ""
    echo "=== Quick mode: regenerating tables and plots from cached results ==="
    python3 run_final.py --mode tables
    python3 run_final.py --mode wilcoxon
    python3 generate_final_tables.py
    echo ""
    echo "=== Done! Results in /workspace/results/ ==="
    
elif [ "$MODE" = "real" ]; then
    echo ""
    echo "=== Running real-world experiments ==="
    python3 run_final.py --mode real
    python3 run_final.py --mode tables
    python3 run_final.py --mode wilcoxon
    echo ""
    echo "=== Done! Results in /workspace/results/ ==="
    
elif [ "$MODE" = "synth" ]; then
    echo ""
    echo "=== Running synthetic experiments ==="
    python3 run_final.py --mode synthetic
    python3 run_final.py --mode tables
    python3 run_final.py --mode wilcoxon
    echo ""
    echo "=== Done! Results in /workspace/results/ ==="
    
else
    echo ""
    echo "=== Running FULL pipeline ==="
    echo ""
    
    echo "--- Step 1: Real-world experiments ---"
    python3 run_final.py --mode real
    
    echo ""
    echo "--- Step 2: Synthetic experiments ---"
    python3 run_final.py --mode synthetic
    
    echo ""
    echo "--- Step 3: Generate combined aggregate tables ---"
    python3 run_final.py --mode tables
    
    echo ""
    echo "--- Step 4: Generate Wilcoxon tests ---"
    python3 run_final.py --mode wilcoxon
    
    echo ""
    echo "--- Step 5: Generate final analysis tables ---"
    python3 generate_final_tables.py
    
    echo ""
    echo "=============================================="
    echo "All results saved to /workspace/results/"
    echo "=============================================="
fi

echo ""
echo "Key output files:"
echo "  Tables:"
echo "    results/table_k-means_RealWorld.csv    - k-means ARI on 20 UCI datasets"
echo "    results/table_AHC_RealWorld.csv         - AHC ARI on 20 UCI datasets"
echo "    results/table_GMM_RealWorld.csv         - GMM ARI on 20 UCI datasets"
echo "    results/table_OPTICS_RealWorld.csv      - OPTICS ARI on 20 UCI datasets"
echo "    results/table_combined_aggregate_*.csv  - Combined aggregate tables (Tables 1-4)"
echo "    results/table_wilcoxon_combined.csv     - Wilcoxon signed-rank tests (Table 5)"
echo "    results/table_average_ARI_*.csv         - Average ARI per synthetic type (Tables A.1-A.4)"
echo "  Boxplots:"
echo "    results/boxplot_*_RealWorld.pdf         - Real-world boxplots (4 algos)"
echo "    results/boxplot_*_Circles.pdf           - Circles boxplots"
echo "    results/boxplot_*_Moons.pdf             - Moons boxplots"
echo "    results/boxplot_*_RSG.pdf               - RSG boxplots"
echo "    results/boxplot_*_Repliclust.pdf        - Repliclust boxplots"
echo ""
echo "Done!"
