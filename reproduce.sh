#!/bin/bash
# Reproduce the Hypothesis Hivemind experiment from the paper
# "Agentic AI Scientists Are Not Built for Autonomous Discovery"
#
# Prerequisites:
# - OPENROUTER_API_KEY environment variable set
# - Python 3 with pip
#
# This script regenerates all key results. Steps 1-4 require API calls
# and are expensive (~$50-100 in API costs, ~6000 calls). If data already
# exists, those steps are skipped.
#
# To regenerate from scratch, delete the data/ directory first.

set -e

echo "============================================"
echo "Hypothesis Hivemind Experiment Reproduction"
echo "============================================"

# Install dependencies
pip install requests numpy matplotlib scipy PyMuPDF 2>/dev/null

# Step 1: Download papers (if not already done)
if [ ! -f "data/papers_metadata.json" ]; then
    echo ""
    echo "Step 1: Downloading 50 papers from OpenReview..."
    python download_papers.py
else
    echo "Step 1: Papers already downloaded, skipping."
fi

# Step 2: Generate experiment summaries (if not already done)
SUMMARY_COUNT=$(ls data/summaries/*.json 2>/dev/null | wc -l)
if [ "$SUMMARY_COUNT" -lt 50 ]; then
    echo ""
    echo "Step 2: Generating experiment summaries..."
    python generate_summaries.py
else
    echo "Step 2: Summaries already generated ($SUMMARY_COUNT/50), skipping."
fi

# Step 3 & 4: Generate hypotheses (if not already done)
TASK1_COUNT=$(ls data/hypotheses/task1/*.json 2>/dev/null | wc -l)
TASK2_COUNT=$(ls data/hypotheses/task2/*.json 2>/dev/null | wc -l)
if [ "$TASK1_COUNT" -lt 3000 ] || [ "$TASK2_COUNT" -lt 2990 ]; then
    echo ""
    echo "Step 3-4: Generating hypotheses..."
    echo "  Task 1: $TASK1_COUNT/3000"
    echo "  Task 2: $TASK2_COUNT/3000"
    python generate_hypotheses.py
else
    echo "Step 3-4: Hypotheses already generated (Task1: $TASK1_COUNT, Task2: $TASK2_COUNT), skipping."
fi

# Step 5: Embed hypotheses (if not already done)
if [ ! -f "data/embeddings/task1_embeddings.npz" ] || [ ! -f "data/embeddings/task2_embeddings.npz" ]; then
    echo ""
    echo "Step 5: Embedding hypotheses with text-embedding-3-small..."
    python embed_hypotheses.py
else
    echo "Step 5: Embeddings already computed, skipping."
fi

# Step 6-7: Generate figures (always regenerate)
echo ""
echo "Step 6-7: Computing similarities and generating figures..."
python generate_figures.py

echo ""
echo "============================================"
echo "DONE! Results saved to results/"
echo "============================================"
echo ""
echo "Generated files:"
ls -la results/
