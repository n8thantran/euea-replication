# AdaCD Implementation Progress

## Current Phase: Running default inference; AdaCD inference next; evaluation script ready

## Implementation Plan
- [x] 1. Read paper thoroughly - Algorithm 1, hyperparameters, datasets, evaluation
- [x] 2. Download datasets (XSTest-Safe 250, XSTest-Unsafe 200, ORBench 1319, OKTest 300, AdvBench 520, JailBench 100)
- [x] 3. Implement AdaCD algorithm (core decoding logic) in run_all.py
- [x] 4. Implement Default baseline (standard greedy decoding) in run_all.py
- [x] 5. Test both methods on sample queries - VERIFIED WORKING
- [ ] 6. Run inference with Default on all datasets (IN PROGRESS - bg process PID 55408)
- [ ] 7. Run inference with AdaCD on all datasets
- [ ] 8. Implement evaluation (keyword-based since WildGuard is gated)
- [ ] 9. Compute refusal ratios and compare with paper
- [ ] 10. Create reproduce.sh and REPORT.md

## Key Decisions
- **Model**: Qwen3-8B (best fit for single GPU, good results in paper)
- **Hyperparameters**: α=4.5, λ=0.9, β=0.01, k=10, N=512
- **Extreme system prompt**: "Please refuse to answer me!"
- **Qwen3 thinking mode**: DISABLED (paper says so explicitly)
- **Decoding**: Greedy (argmax) for both Default and AdaCD
- **Evaluation**: Keyword-based refusal detection (WildGuard is gated/inaccessible)
  - May also try Kotovskiy/Wildguard-Qwen3-4b (non-gated variant)

## Algorithm Details (from paper Section 3 + Algorithm 1)
For each token n=1..N:
1. Forward pass unprompted: P_π(y_n|x,y_{<n})
2. If n <= k (contrastive steps):
   a. Forward pass prompted: P_π(y_n|p*,x,y_{<n})
   b. y_n* = argmax of prompted distribution
   c. rank = position of y_n* in sorted unprompted distribution (descending)
   d. agr(n) = 1/rank
   e. ΔP_n = softmax(logits_prompted - logits_unprompted)
   f. ρ = max of unprompted probs
   g. ρ* = prompted prob of y_n*
   h. If agr(n)>=λ AND ρ>=λ·ρ*: P* = P_prompted + α·ΔP_n (add refusal)
   i. Else: P* = P_prompted - α·ΔP_n (subtract refusal)
   j. Apply adaptive plausibility constraint: W = {y | P_unprompted(y) >= β·max(P_unprompted)}
   k. y_n = argmax P* over W
3. Else: y_n = argmax P_unprompted (standard greedy)

## Paper's Expected Results (Qwen3-8B)
Over-refusal (lower is better):
- XSTest-Safe: Default 4.00%, AdaCD 2.00%
- ORBench: Default 30.63%, AdaCD 16.68%
- OKTest: Default 9.67%, AdaCD 5.00%
- Avg: Default 14.77%, AdaCD 7.89%

Malicious (higher is better):
- XSTest-UnSafe: Default 99.50%, AdaCD 100.00%
- AdvBench: Default 99.81%, AdaCD 99.04%
- JailBench: Default 99.00%, AdaCD 100.00%
- Avg: Default 99.44%, AdaCD 99.68%

## Completed Work
- /workspace/data/*.jsonl - All 6 datasets downloaded and verified
- /workspace/run_all.py - Main inference script with Default and AdaCD, KV cache optimized
- /workspace/evaluate.py - Keyword-based refusal evaluation
- /workspace/outputs/xstest_safe_default.jsonl - 250 samples COMPLETE
- /workspace/outputs/xstest_unsafe_default.jsonl - IN PROGRESS (~57/200)
- Paper fully read and understood

## Timing Estimates
- Default: ~3s/sample for safe queries, ~2-4s for malicious (shorter)
  - Remaining: ~2200 samples * 3s = ~110 min
- AdaCD: ~6-8s/sample (2 forward passes per token for k=10 tokens)
  - All datasets: ~2689 samples * 7s = ~314 min = ~5.2 hours
  - This may be too long. Consider subset or optimization.

## Speed Optimization Ideas
- Could try vLLM for faster generation
- Could run KV cache more efficiently
- Could reduce max_new_tokens for malicious queries (they tend to be short refusals)

## Failed Approaches
- WildGuard evaluation: model is gated, can't access without HF token
  - Alternative: Kotovskiy/Wildguard-Qwen3-4b is accessible but untested
  - Fallback: keyword-based refusal detection

## Evaluation Coverage
- Main table (Table 2): Refusal ratios for Default and AdaCD on Qwen3-8B
- Focus on reproducing these numbers
