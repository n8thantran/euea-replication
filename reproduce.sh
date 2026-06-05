#!/bin/bash
# reproduce.sh - Reproduce key results from the paper
# "Assessing the impact of dimensionality reduction on clustering performance"
#
# This script has three modes:
#   ./reproduce.sh          - Regenerate all tables and figures from pre-computed results (fast, ~1 min)
#   ./reproduce.sh real     - Run real-world experiments from scratch (~30-60 min)
#   ./reproduce.sh full     - Run ALL experiments from scratch (slow, ~4-8 hours)
#
# Pre-computed results are in results/*.json from experiments that already ran.

set -e

MODE=${1:-tables}

echo "============================================================"
echo "Reproducing: Dimensionality Reduction for Clustering"
echo "Mode: $MODE"
echo "============================================================"

# Install dependencies
echo "Installing dependencies..."
pip install -q numpy scipy scikit-learn matplotlib torch ucimlrepo repliclust 2>/dev/null || true

# Make results directory
mkdir -p /workspace/results

if [ "$MODE" = "full" ]; then
    echo ""
    echo "============================================================"
    echo "FULL EXPERIMENT MODE"
    echo "Running all experiments from scratch..."
    echo "This will take several hours."
    echo "============================================================"
    echo ""
    
    cd /workspace
    python main_pipeline.py all 2 50
    
elif [ "$MODE" = "real" ]; then
    echo ""
    echo "============================================================"
    echo "REAL-WORLD EXPERIMENT MODE"
    echo "Running real-world experiments only (~30-60 min)..."
    echo "============================================================"
    echo ""
    
    cd /workspace
    python main_pipeline.py real

else
    echo ""
    echo "============================================================"
    echo "TABLE/FIGURE GENERATION MODE (from pre-computed results)"
    echo "============================================================"
    echo ""
fi

# Always regenerate tables and figures from whatever results exist
echo "Generating all tables and figures..."
cd /workspace
python generate_all_results.py

echo ""
echo "============================================================"
echo "DONE! All results saved to /workspace/results/"
echo "============================================================"
echo ""
echo "Key output files:"
echo ""
echo "  Tables 1-4 (Aggregate win/loss statistics - paper's main results):"
echo "    results/table_combined_aggregate_k-means.csv  (Table 1)"
echo "    results/table_combined_aggregate_AHC.csv      (Table 2)"
echo "    results/table_combined_aggregate_GMM.csv      (Table 3)"
echo "    results/table_combined_aggregate_OPTICS.csv   (Table 4)"
echo ""
echo "  Table 5 (Wilcoxon signed-rank tests):"
echo "    results/table_wilcoxon_RealWorld.csv"
echo ""
echo "  Tables A.1-A.4 (Synthetic average ARI):"
echo "    results/table_average_ARI_Circles.csv"
echo "    results/table_average_ARI_Moons.csv"
echo "    results/table_average_ARI_RSG.csv"
echo "    results/table_average_ARI_Repliclust.csv"
echo ""
echo "  Tables A.5-A.8 (Real-world per-dataset ARI):"
echo "    results/table_k-means_RealWorld.csv"
echo "    results/table_AHC_RealWorld.csv"
echo "    results/table_GMM_RealWorld.csv"
echo "    results/table_OPTICS_RealWorld.csv"
echo ""
echo "  Figures 1-8 (Boxplots):"
echo "    results/boxplot_*.pdf (20 total: 4 algos × 5 data types)"
echo ""
echo "  Raw JSON results:"
echo "    results/results_RealWorld.json"
echo "    results/results_Circles.json"
echo "    results/results_Moons.json"
echo "    results/results_RSG.json"
echo "    results/results_Repliclust.json"
