# GRASPrune Implementation Progress

## Current Phase
Running remaining ratios (0.6, 0.4) and building evaluation pipeline.
Ratios 0.5 and 0.8 are done. PTB loading is fixed.

## Implementation Plan
- [x] Read paper thoroughly
- [x] Set up environment
- [x] Implement core graspune.py
- [x] Test pipeline end-to-end on TinyLlama
- [x] Run on LLaMA-2-7B ratio=0.5 — **PPL=12.87 (paper=12.18)** ✅
- [x] Fix double-gating bug (was gating gate_proj+up_proj separately → gate down_proj input)
- [x] Fix scale folding (fold into down_proj columns for FFN, o_proj columns for attention)
- [x] Fix PTB dataset loading — use parquet URL
- [x] Run ratio=0.8 — **Wiki=7.55 (paper=6.47), C4=12.38 (paper=11.44), PTB=78.41 (paper=48.18)**
- [ ] Run ratio=0.6
- [ ] Run ratio=0.4
- [ ] Add zero-shot evaluations (lm-eval-harness)
- [ ] Generate results tables matching paper Table 1
- [ ] Create reproduce.sh
- [ ] Write REPORT.md
- [ ] Clean up old files from previous project

## Key Decisions
- Model: LLaMA-2-7B via NousResearch/Llama-2-7b-hf
- GPU: 80GB A100
- Calibration: 512 seqs × 512 tokens from WikiText-2 train
- τ = 1.5, lr = 1e-2, epochs = 4, batch_size = 1, scale_epochs = 2
- α = (2G + 2) * d_h / 3 = 170.67 for LLaMA-2-7B (G=1, d_h=128)
- **CRITICAL FIX**: Gate applied ONCE per unit:
  - FFN: pre-hook on down_proj (gates intermediate activation)
  - Attention: pre-hook on o_proj (gates per-head attention output)
- Scales folded into down_proj columns (FFN) and o_proj columns (attention)

## Results Collected
| Ratio | Wiki PPL (ours) | Wiki PPL (paper) | C4 PPL (ours) | C4 PPL (paper) | PTB PPL (ours) | PTB PPL (paper) |
|-------|-----------------|------------------|---------------|----------------|----------------|-----------------|
| 0.8   | 7.55            | 6.47             | 12.38         | 11.44          | 78.41          | 48.18           |
| 0.5   | 12.87           | 12.18            | 26.63         | 27.89          | N/A            | 123.04          |

**Assessment**: WikiText-2 and C4 are within ~15% of paper values (reasonable for replication).
PTB is higher at ratio 0.8 — possibly different tokenization/eval scheme. The paper may use
a specific PTB processing that differs from the raw parquet.

## Target Results (LLaMA-2-7B, Table 1)
| Ratio | Wiki PPL | PTB PPL | C4 PPL | Avg Acc |
|-------|----------|---------|--------|---------|
| 0.8   | 6.47     | 48.18   | 11.44  | 0.614   |
| 0.6   | 9.64     | 70.11   | 18.87  | 0.503   |
| 0.5   | 12.18    | 123.04  | 27.89  | 0.465   |
| 0.4   | 16.65    | 148.41  | 43.19  | 0.406   |

## Failed Approaches
1. **meta-llama/Llama-2-7b-hf**: Access denied (gated model). Use NousResearch mirror.
2. **Double gating (v1)**: Gating both gate_proj AND up_proj outputs separately caused
   the product to be gated twice. This gave PPL=22.73 instead of 12.18.
   Fix: gate only once via down_proj pre-hook.
3. **Scale folding into gate_proj/up_proj**: When scaling is applied at down_proj input,
   the scale must be folded into down_proj columns, not gate_proj/up_proj rows.
4. **PTB dataset**: ptb-text-only/ptb_text_only uses deprecated loading script.
   Fixed: download parquet directly from HF URL.

## Completed Work
- **graspune.py**: Full implementation with correct single-gate-application hooks
  - load_calibration_data, compute_alpha, build_prunable_units
  - project_to_budget_fast (greedy projection)
  - GRASPruneGates (scores, STE, projection)
  - apply_gates_to_model_forward (pre-hooks on down_proj and o_proj)
  - train_gates, train_scaling
  - materialize_pruned_model (correct scale folding)
  - evaluate_perplexity (wikitext, c4, ptb)
  - main() pipeline with argparse
- **test_pipeline.py**: Unit tests for TinyLlama
- **results/results_ratio0.5.json**: Ratio 0.5 results
- **results/results_ratio0.8.json**: Ratio 0.8 results (includes PTB)
- **pruned_models/**: Saved pruned models

## Remaining Work
1. Run ratio 0.6 and 0.4 (~5 min each)
2. Run lm-eval-harness zero-shot tasks (BoolQ, PIQA, HellaSwag, WinoGrande, ARC-e, ARC-c, OBQA)
3. Build comparison table
4. Create reproduce.sh that runs one ratio as demo
5. Write REPORT.md
6. Clean up old project files (src/, data/)

## Important Notes
- Git branch is 'master' not 'main'
- Old files from previous (EUEA) project still in src/, data/ - need cleanup
- reproduce.sh still points to old project - must be replaced
- Each ratio run takes ~3.5 min (gate) + ~1.2 min (scaling) + ~2 min (eval) ≈ 7 min total
