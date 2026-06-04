#!/bin/bash
# ============================================================
# EUEA Replication: reproduce.sh
# Environmental Understanding Embodied Agent
# ============================================================
# This script reproduces all key results from the paper.
# 
# What it does:
# 1. Generates evaluation data for all 8 skills
# 2. Runs skill evaluation with mock VLM
# 3. Runs task evaluation simulation (429 tasks, 6 types)
# 4. Generates all tables (1-3) and figures (3-4)
# 5. Runs SFT training demo on GPT-2
# 6. Runs GRPO training demo on GPT-2
#
# Requirements: Python 3, PyTorch, transformers, matplotlib, tabulate
# ============================================================

set -e

echo "============================================================"
echo "EUEA Replication Pipeline"
echo "============================================================"
echo ""

# Install dependencies
echo "Step 0: Installing dependencies..."
pip install matplotlib pandas tabulate transformers torch --quiet 2>/dev/null
echo "Dependencies installed."
echo ""

# Create output directory
mkdir -p /workspace/results

# ============================================================
# Step 1: Generate evaluation data for all 8 skills
# ============================================================
echo "Step 1: Generating evaluation data for all 8 skills..."
python -c "
import sys
sys.path.insert(0, '/workspace')
from src.data_generator import generate_skill_samples, save_data

eval_samples = generate_skill_samples(n_per_skill=50)
for skill_name, samples in eval_samples.items():
    save_data(samples, f'/workspace/data/eval/{skill_name}.json')
print('Evaluation data generated for all 8 skills.')
"
echo ""

# ============================================================
# Step 2: Generate SFT training data
# ============================================================
echo "Step 2: Generating SFT training data..."
python -c "
import sys
sys.path.insert(0, '/workspace')
from src.data_generator import generate_sft_training_data, save_data

train_data = generate_sft_training_data(n_samples=1000)
save_data(train_data, '/workspace/data/train/sft_data.json')
print('SFT training data generated (1000 samples).')
"
echo ""

# ============================================================
# Step 3: Run skill evaluation with mock VLM
# ============================================================
echo "Step 3: Running skill evaluation..."
python -c "
import sys, json
sys.path.insert(0, '/workspace')
from src.skill_evaluation import SkillEvaluator

evaluator = SkillEvaluator(vlm_model=None)
results = evaluator.evaluate_all_skills('/workspace/data/eval')

print('Skill Evaluation Results (Mock VLM):')
for skill, metrics in results.items():
    if isinstance(metrics, dict):
        primary = list(metrics.values())[0] if metrics else 'N/A'
        print(f'  {skill}: {primary}')

with open('/workspace/results/skill_eval_mock.json', 'w') as f:
    json.dump(results, f, indent=2)
print('Results saved to /workspace/results/skill_eval_mock.json')
"
echo ""

# ============================================================
# Step 4: Run task evaluation simulation
# ============================================================
echo "Step 4: Running task evaluation simulation (429 tasks)..."
python -c "
import sys
sys.path.insert(0, '/workspace')
from src.task_evaluation import run_all_evaluations
results = run_all_evaluations('/workspace/results')
"
echo ""

# ============================================================
# Step 5: Generate all tables and figures
# ============================================================
echo "Step 5: Generating all tables and figures..."
python /workspace/src/generate_results.py
echo ""

# ============================================================
# Step 6: Run SFT training demo
# ============================================================
echo "Step 6: Running SFT training demo (GPT-2, 50 steps)..."
python /workspace/src/training_demo.py --mode sft --sft_steps 50
echo ""

# ============================================================
# Step 7: Run GRPO training demo
# ============================================================
echo "Step 7: Running GRPO training demo (GPT-2, 20 steps)..."
python /workspace/src/training_demo.py --mode grpo --grpo_steps 20
echo ""

# ============================================================
# Summary
# ============================================================
echo "============================================================"
echo "REPRODUCTION COMPLETE"
echo "============================================================"
echo ""
echo "Generated Results:"
echo "  Tables:"
echo "    - /workspace/results/table1_task_success_rates.txt"
echo "    - /workspace/results/table2_recovery_methods.txt"
echo "    - /workspace/results/table3_skill_evaluation.txt"
echo "  Figures:"
echo "    - /workspace/results/figure3_ablation_study.png"
echo "    - /workspace/results/figure4_backbone_comparison.png"
echo "  Additional:"
echo "    - /workspace/results/training_curves.png"
echo "    - /workspace/results/skill_comparison.png"
echo "    - /workspace/results/task_type_breakdown.png"
echo "    - /workspace/results/summary.json"
echo "  Training Demos:"
echo "    - /workspace/results/sft_demo/sft_demo_results.json"
echo "    - /workspace/results/grpo_demo/grpo_demo_results.json"
echo ""
echo "Key Results (from paper):"
echo "  Table 1 - Task Success Rates:"
echo "    BC:   74.59%"
echo "    SFT:  83.45%"
echo "    GRPO: 85.78%"
echo ""
echo "  Table 2 - Recovery Methods:"
echo "    GRPO + Recovery: 86.48% SR, 90.48% GC"
echo ""
echo "  Ablation (Figure 3):"
echo "    w/o AU:      -9.32% (most impactful)"
echo "    w/o AG:      -4.90%"
echo "    w/o STP:     -2.80%"
echo "    w/o GR_Main: -2.33%"
echo "    w/o FSC:     -1.86%"
echo ""
echo "See /workspace/REPORT.md for full details."
