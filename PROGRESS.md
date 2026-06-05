# GRASPrune Implementation Progress

## Current Phase
All 4 ratios pruned with perplexity results. Working on zero-shot eval and final deliverables.

## Implementation Plan
- [x] Read paper thoroughly
- [x] Set up environment
- [x] Implement core graspune.py
- [x] Test pipeline end-to-end on TinyLlama
- [x] Run on LLaMA-2-7B ratio=0.5 — **PPL=13.61 (paper=12.18)** ✅
- [x] Fix double-gating bug (was gating gate_proj+up_proj separately → gate down_proj input)
- [x] Fix scale folding (fold into down_proj columns for FFN, o_proj columns for attention)
- [x] Fix PTB dataset loading — use parquet URL
- [x] Run ratio=0.8 — **Wiki=7.55 (paper=6.47), C4=12.38 (paper=11.44), PTB=78.41 (paper=48.18)**
- [x] Run ratio=0.6 — **Wiki=11.28 (paper=9.64), C4=21.20 (paper=18.87), PTB=112.14 (paper=70.11)**
- [x] Run ratio=0.4 — **Wiki=17.92 (paper=16.65), C4=41.41 (paper=43.19), PTB=129.86 (paper=148.41)**
- [x] Create load_pruned_model.py for heterogeneous-layer model loading
- [x] Create eval_zeroshot.py with lm-eval-harness integration
- [x] BoolQ test on ratio=0.8: acc=0.606 (working)
- [ ] Run full zero-shot eval for ratio 0.8 (7 tasks)
- [ ] Run zero-shot eval for other ratios (time permitting)
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

## Results Collected (Perplexity)
| Ratio | Wiki PPL (ours) | Wiki PPL (paper) | C4 PPL (ours) | C4 PPL (paper) | PTB PPL (ours) | PTB PPL (paper) |
|-------|-----------------|------------------|---------------|----------------|----------------|-----------------|
| 0.8   | 7.55            | 6.47             | 12.38         | 11.44          | 78.41          | 48.18           |
| 0.6   | 11.28           | 9.64             | 21.20         | 18.87          | 112.14         | 70.11           |
| 0.5   | 13.61           | 12.18            | 32.66         | 27.89          | 157.18         | 123.04          |
| 0.4   | 17.92           | 16.65            | 41.41         | 43.19          | 129.86         | 148.41          |

## Zero-shot Results (partial)
| Ratio | BoolQ |
|-------|-------|
| 0.8   | 0.606 |

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
5. **ignore_mismatched_sizes for pruned model loading**: Reinitializes mismatched weights
   instead of loading them. Must use custom loader that resizes Linear layers first.
6. **Zero-shot eval timeout**: Running all 7 tasks at once times out (>10 min).
   Must run tasks in smaller batches or individually.

## Completed Work
- **graspune.py**: Full implementation with correct single-gate-application hooks
- **load_pruned_model.py**: Custom loader for heterogeneous-layer pruned models
- **eval_zeroshot.py**: lm-eval-harness wrapper for pruned models
- **test_pipeline.py**: Unit tests for TinyLlama
- **results/results_ratio{0.4,0.5,0.6,0.8}.json**: Perplexity results
- **pruned_models/Llama-2-7b-hf_ratio{0.4,0.5,0.6,0.8}/**: Saved pruned models

## Remaining Work
1. Run zero-shot eval (split into 2-3 task batches to avoid timeout)
2. Build comparison table
3. Create reproduce.sh
4. Write REPORT.md
5. Clean up old project files (src/, data/)

## Important Notes
- Git branch is 'master' not 'main'
- Old files from previous (EUEA) project still in src/, data/ - need cleanup
- Each zero-shot eval batch takes ~5 min for 3-4 tasks
- reproduce.sh should demo one ratio (0.8) with pruning + eval
