#!/bin/bash
# GRASPrune Replication - Reproduce Key Results
# Paper: "GRASPrune: Structured Pruning via Gradient-based Learnable Allocation"
# Target: LLaMA-2-7B pruning at ratios 0.8, 0.6, 0.5, 0.4
#
# This script reproduces the main results from Table 1 of the paper.
# Full run takes ~4-6 hours on a single A100 GPU.
# Set QUICK_MODE=1 to only run ratio=0.5 (~1 hour).

set -e

QUICK_MODE=${QUICK_MODE:-0}
MODEL="NousResearch/Llama-2-7b-hf"
OUTPUT_DIR="/workspace/pruned_models"
RESULTS_DIR="/workspace/results"

mkdir -p "$OUTPUT_DIR" "$RESULTS_DIR"

echo "============================================"
echo "GRASPrune Replication"
echo "============================================"

# Install dependencies
pip install -q datasets transformers accelerate safetensors lm-eval 2>/dev/null

if [ "$QUICK_MODE" = "1" ]; then
    RATIOS="0.5"
    echo "QUICK MODE: Only running ratio=0.5"
else
    RATIOS="0.8 0.6 0.5 0.4"
    echo "FULL MODE: Running ratios 0.8, 0.6, 0.5, 0.4"
fi

# Step 1: Prune and evaluate perplexity for each ratio
for RATIO in $RATIOS; do
    echo ""
    echo "============================================"
    echo "Pruning LLaMA-2-7B at ratio=$RATIO"
    echo "============================================"
    
    SAVE_PATH="${OUTPUT_DIR}/Llama-2-7b-hf_ratio${RATIO}"
    LOG_PATH="${RESULTS_DIR}/run_ratio${RATIO}.log"
    RESULT_PATH="${RESULTS_DIR}/results_ratio${RATIO}.json"
    
    if [ -f "$RESULT_PATH" ]; then
        echo "Results already exist at $RESULT_PATH, skipping pruning."
        cat "$RESULT_PATH"
    else
        python graspune.py \
            --model "$MODEL" \
            --target_ratio "$RATIO" \
            --save_path "$SAVE_PATH" \
            --tau 1.5 \
            --lr 0.01 \
            --epochs 4 \
            --scale_epochs 2 \
            --n_calibration 512 \
            --seq_len 512 \
            --batch_size 1 \
            2>&1 | tee "$LOG_PATH"
    fi
done

# Step 2: Zero-shot evaluation (ratio 0.8 and 0.5)
echo ""
echo "============================================"
echo "Zero-shot Evaluation"
echo "============================================"

for RATIO in 0.8 0.5; do
    SAVE_PATH="${OUTPUT_DIR}/Llama-2-7b-hf_ratio${RATIO}"
    ZEROSHOT_PATH="${RESULTS_DIR}/zeroshot_ratio${RATIO}.json"
    
    if [ ! -d "$SAVE_PATH" ]; then
        echo "Pruned model not found at $SAVE_PATH, skipping zero-shot eval."
        continue
    fi
    
    if [ -f "$ZEROSHOT_PATH" ]; then
        echo "Zero-shot results already exist at $ZEROSHOT_PATH, skipping."
    else
        echo "Running zero-shot eval for ratio=$RATIO..."
        python eval_zeroshot.py \
            --model_path "$SAVE_PATH" \
            --tasks "arc_easy,arc_challenge,hellaswag,piqa,winogrande" \
            --batch_size 16 \
            --output_path "$ZEROSHOT_PATH"
    fi
done

# Step 3: Generate summary table
echo ""
echo "============================================"
echo "Generating Summary Table"
echo "============================================"

python3 << 'PYEOF' > "${RESULTS_DIR}/summary_table.txt"
import json, os

results_dir = os.environ.get('RESULTS_DIR', '/workspace/results')

print('='*80)
print('GRASPrune Replication Results Summary')
print('='*80)

print()
print('Table 1: Perplexity Results (LLaMA-2-7B)')
print('-'*80)
print(f'{"Ratio":>6} | {"Wiki (ours)":>12} | {"Wiki (paper)":>12} | {"C4 (ours)":>10} | {"C4 (paper)":>10} | {"PTB (ours)":>10} | {"PTB (paper)":>10}')
print('-'*80)

paper_ppl = {
    0.8: {'wiki': 6.47, 'c4': 11.44, 'ptb': 48.18},
    0.6: {'wiki': 9.64, 'c4': 18.87, 'ptb': 70.11},
    0.5: {'wiki': 12.18, 'c4': 27.89, 'ptb': 123.04},
    0.4: {'wiki': 16.65, 'c4': 43.19, 'ptb': 148.41},
}

for ratio in [0.8, 0.6, 0.5, 0.4]:
    fpath = f'{results_dir}/results_ratio{ratio}.json'
    if os.path.exists(fpath):
        with open(fpath) as f:
            r = json.load(f)
        p = paper_ppl[ratio]
        print(f'{ratio:>6.1f} | {r["wikitext2_ppl"]:>12.2f} | {p["wiki"]:>12.2f} | {r["c4_ppl"]:>10.2f} | {p["c4"]:>10.2f} | {r["ptb_ppl"]:>10.2f} | {p["ptb"]:>10.2f}')

print()
print('Table 2: Zero-shot Accuracy (LLaMA-2-7B)')
print('-'*80)

norm_tasks = {'hellaswag', 'winogrande', 'arc_challenge'}
paper_acc = {
    0.8: {'arc_challenge': 0.3848, 'arc_easy': 0.6351, 'hellaswag': 0.6748, 'piqa': 0.7405, 'winogrande': 0.6346},
    0.5: {'arc_challenge': 0.2645, 'arc_easy': 0.4646, 'hellaswag': 0.4513, 'piqa': 0.6539, 'winogrande': 0.4925},
}

for ratio in [0.8, 0.5]:
    fpath = f'{results_dir}/zeroshot_ratio{ratio}.json'
    if not os.path.exists(fpath):
        continue
    with open(fpath) as f:
        data = json.load(f)
    
    # Handle different formats
    if 'tasks' in data:
        tasks_data = data['tasks']
    elif 'task_results' in data:
        tasks_data = data['task_results']
    else:
        continue
    
    print(f'\nRatio {ratio}:')
    print(f'{"Task":>15} | {"Ours":>8} | {"Paper":>8}')
    print('-'*40)
    
    accs = []
    for task in ['arc_easy', 'arc_challenge', 'hellaswag', 'piqa', 'winogrande']:
        r = tasks_data.get(task, {})
        if task in norm_tasks and r.get('acc_norm') is not None:
            v = r['acc_norm']
        else:
            v = r.get('acc', 0)
        p = paper_acc[ratio].get(task, 0)
        accs.append(v)
        print(f'{task:>15} | {v:>8.4f} | {p:>8.4f}')
    
    avg = sum(accs) / len(accs)
    pavg = sum(paper_acc[ratio].values()) / len(paper_acc[ratio])
    print('-'*40)
    print(f'{"Average":>15} | {avg:>8.4f} | {pavg:>8.4f}')

PYEOF

cat "${RESULTS_DIR}/summary_table.txt"

echo ""
echo "============================================"
echo "All results saved to ${RESULTS_DIR}/"
echo "============================================"
echo "Done!"
