# AdaCD Implementation Progress

## Current Phase: Default inference running (orbench 349/1319), then AdaCD next

## Implementation Plan
- [x] 1. Read paper thoroughly
- [x] 2. Download all 6 datasets
- [x] 3. Implement AdaCD algorithm in run_all.py
- [x] 4. Implement Default baseline in run_all.py
- [x] 5. Test both methods - VERIFIED WORKING
- [x] 6a. Default on xstest_safe (250) - DONE (512 tokens)
- [x] 6b. Default on xstest_unsafe (200) - DONE (512 tokens)
- [ ] 6c. Default on orbench (1319) - IN PROGRESS 349/1319 (128 tokens, ~49min left)
- [ ] 6d. Default on oktest (300) - PENDING (~15min)
- [ ] 6e. Default on advbench (520) - PENDING (~26min)
- [ ] 6f. Default on jailbench (100) - PENDING (~5min)
- [ ] 7. Run AdaCD on all datasets (~3.7 hours total)
- [x] 8. Implement evaluation (keyword-based)
- [ ] 9. Compute refusal ratios and compare with paper
- [ ] 10. Create reproduce.sh and REPORT.md

## Key Decisions
- **Model**: Qwen3-8B
- **Hyperparameters**: α=4.5, λ=0.9, β=0.01, k=10, N=512 (reduced to 128 for speed)
- **Extreme system prompt**: "Please refuse to answer me!"
- **Qwen3 thinking mode**: DISABLED
- **Decoding**: Greedy (argmax)
- **Evaluation**: Keyword-based refusal detection
- **Speed optimization**: max_new_tokens=128 for orbench+ (xstest used 512)

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
- /workspace/data/*.jsonl - All 6 datasets
- /workspace/run_all.py - Main inference script with Default and AdaCD
- /workspace/evaluate.py - Keyword-based refusal evaluation
- /workspace/outputs/xstest_safe_default.jsonl - 250 samples COMPLETE
- /workspace/outputs/xstest_unsafe_default.jsonl - 200 samples COMPLETE
- /workspace/outputs/orbench_default.jsonl - 349/1319 IN PROGRESS

## Timing
- Default: ~3s/sample with 128 tokens, ~95 min remaining
- AdaCD: ~5-6s/sample estimated, ~224 min for all datasets
- Background PID: 58346

## Failed Approaches
- WildGuard evaluation: model is gated
- 512 max_new_tokens: too slow (~10s/sample for safe queries)
- Reduced to 128 tokens: sufficient for refusal detection

## Priority if time-constrained
Run AdaCD on: XSTest-Safe, XSTest-Unsafe, ORBench (most important for paper claims)
