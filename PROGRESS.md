# GRASPrune Implementation Progress

## Current Phase
Reading paper and creating implementation plan.

## Implementation Plan
- [ ] Set up environment and dependencies (transformers, lm-eval-harness, datasets)
- [ ] Implement data loading (WikiText-2 calibration: 512 seqs of length 512)
- [ ] Implement cost model (FFN channel cost=1, KV head cost=α=(2G+2)*d_h/3)
- [ ] Implement projected STE gate learning
  - [ ] Gate score initialization (all zeros)
  - [ ] Sigmoid conversion with temperature τ=1.5
  - [ ] Budget-feasible projection (sort by p_i descending, greedy select)
  - [ ] STE surrogate: z_tilde = m + (p - stopgrad(p))
  - [ ] Degenerate layer protection
  - [ ] AdamW optimizer: lr=1e-2, bs=1, 4 epochs, no weight decay
- [ ] Implement scaling calibration (per-unit scalar γ)
- [ ] Implement model materialization (slice weights, fold γ)
- [ ] Implement perplexity evaluation (WikiText-2, PTB, C4)
- [ ] Implement zero-shot evaluation (lm-eval-harness)
- [ ] Run experiments on LLaMA-2-7B at multiple ratios
- [ ] Generate results and report

## Key Decisions
- Model: LLaMA-2-7B (primary), potentially others
- Calibration: 512 seqs × 512 tokens from WikiText-2 train
- τ = 1.5, lr = 1e-2, epochs = 4, batch_size = 1
- α = (2G + 2) * d_h / 3 where G = H / H_kv
- For LLaMA-2-7B: H=32, H_kv=32, d_h=128, G=1 → α = (2+2)*128/3 = 170.67

## Target Results (LLaMA-2-7B)
| Ratio | Wiki PPL | PTB PPL | C4 PPL | Avg Acc |
|-------|----------|---------|--------|---------|
| 0.8   | 6.47     | 48.18   | 11.44  | 0.614   |
| 0.6   | 9.64     | 70.11   | 18.87  | 0.503   |
| 0.5   | 12.18    | 123.04  | 27.89  | 0.465   |
| 0.4   | 16.65    | 148.41  | 43.19  | 0.406   |

## Completed Work
(none yet)

## Failed Approaches
(none yet)
