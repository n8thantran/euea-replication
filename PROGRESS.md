# GRASPrune Implementation Progress

## Current Phase
Testing and debugging the core GRASPrune implementation. Need to run end-to-end pipeline.

## Implementation Plan
- [x] Read paper thoroughly (GRASPrune: global budgeted structured pruning)
- [x] Set up environment (transformers, datasets, accelerate, lm-eval, sentencepiece)
- [x] Implement core graspune.py (gate learning, projection, STE, scaling, materialization, eval)
- [ ] Test pipeline end-to-end on small model (TinyLlama or similar)
- [ ] Fix any bugs discovered during testing
- [ ] Run on LLaMA-2-7B (NousResearch/Llama-2-7b-hf) at ratio=0.5
- [ ] Run at additional ratios (0.8, 0.6, 0.4) if time permits
- [ ] Evaluate perplexity (WikiText-2, PTB, C4)
- [ ] Run zero-shot evaluations (lm-eval-harness)
- [ ] Generate results tables and comparison
- [ ] Create proper reproduce.sh
- [ ] Write REPORT.md
- [ ] Clean up old files from previous project

## Key Decisions
- Model: LLaMA-2-7B via NousResearch/Llama-2-7b-hf (meta-llama gated)
- Alternative: huggyllama/llama-7b (LLaMA-7B, also in paper Table 2)
- GPU: 80GB A100, sufficient for 7B model in bfloat16
- Calibration: 512 seqs × 512 tokens from WikiText-2 train
- τ = 1.5, lr = 1e-2, epochs = 4, batch_size = 1
- α = (2G + 2) * d_h / 3 where G = H / H_kv
- For LLaMA-2-7B: H=32, H_kv=32, d_h=128, G=1 → α = (2+2)*128/3 = 170.67

## Target Results (LLaMA-2-7B, Table 1)
| Ratio | Wiki PPL | PTB PPL | C4 PPL | Avg Acc |
|-------|----------|---------|--------|---------|
| 0.8   | 6.47     | 48.18   | 11.44  | 0.614   |
| 0.6   | 9.64     | 70.11   | 18.87  | 0.503   |
| 0.5   | 12.18    | 123.04  | 27.89  | 0.465   |
| 0.4   | 16.65    | 148.41  | 43.19  | 0.406   |

## Target Results (LLaMA-7B at 0.4, Table 2)
| Model | Wiki PPL | PTB PPL | C4 PPL | Avg Acc |
|-------|----------|---------|--------|---------|
| LLaMA-7B | 15.89 | 49.72 | 40.57 | 0.407 |

## Completed Work
- graspune.py: Full implementation with all components. NOT YET TESTED.

## Failed Approaches
- meta-llama/Llama-2-7b-hf: Access denied (gated model). Use NousResearch mirror.

## Important Notes
- reproduce.sh and src/ are from a DIFFERENT paper (EUEA). Must be replaced.
- Need to handle dtype carefully (model in bfloat16, gates in float32)
- apply_gates_to_model_forward uses hooks - must cast gate values to model dtype
- The paper reports 6 min runtime for full pruning of LLaMA-2-7B on A100
