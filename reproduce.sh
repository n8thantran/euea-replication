#!/bin/bash
# reproduce.sh - Reproduce all results for the paper replication
#
# Usage:
#   ./reproduce.sh          # Quick mode: generate tables/plots from cached results (~10 sec)
#   ./reproduce.sh full     # Full mode: re-run all experiments from scratch (~3-4 hours)

set -e

echo "============================================"
echo "Paper Replication: Dimensionality Reduction"
echo "  for Clustering Performance Assessment"
echo "============================================"

# Install dependencies
pip install -q numpy pandas scipy scikit-learn matplotlib torch ucimlrepo repliclust 2>/dev/null

MODE=${1:-quick}

if [ "$MODE" = "full" ]; then
    echo ""
    echo "=== FULL MODE: Running all experiments from scratch ==="
    echo "WARNING: This will take 3-4 hours!"
    echo ""
    python main_pipeline.py
else
    echo ""
    echo "=== QUICK MODE: Generating outputs from cached results ==="
    echo ""
    
    # Check that cached results exist
    if [ ! -f results/results_RealWorld.json ]; then
        echo "ERROR: No cached results found in results/"
        echo "Run './reproduce.sh full' to generate from scratch."
        exit 1
    fi
    
    python generate_all_outputs.py
fi

echo ""
echo "============================================"
echo "All results saved to results/"
echo ""
echo "Key output files:"
echo "  Aggregate tables (Tables 1-4):"
echo "    results/table_aggregate_k-means.csv"
echo "    results/table_aggregate_AHC.csv"
echo "    results/table_aggregate_GMM.csv"
echo "    results/table_aggregate_OPTICS.csv"
echo ""
echo "  Per-dataset ARI tables (Tables A.1-A.8):"
echo "    results/table_{algo}_RealWorld.csv"
echo "    results/table_{algo}_{synth_type}.csv"
echo ""
echo "  Wilcoxon test (Table 5):"
echo "    results/table_wilcoxon.csv"
echo ""
echo "  Boxplots (Figures 1-8):"
echo "    results/boxplot_{algo}_RealWorld.pdf"
echo "    results/boxplot_{algo}_Synthetic_{type}.pdf"
echo "============================================"
