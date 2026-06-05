# AdaCD Implementation Progress

## Current Phase: Run AdaCD inference + improve evaluation

## Implementation Plan
- [x] 1. Read paper thoroughly
- [x] 2. Download all 6 datasets (2689 total samples)
- [x] 3. Implement AdaCD algorithm in run_all.py
- [x] 4. Implement Default baseline in run_all.py
- [x] 5. Test both methods - VERIFIED WORKING
- [x] 6. Default on ALL 6 datasets - COMPLETE
- [x] 8. Implement evaluation (keyword-based)
- [ ] 7. Run AdaCD on all 6 datasets
- [ ] 9. Improve evaluation (WildGuard gated; need LLM-as-judge or better keywords)
- [ ] 10. Compute refusal ratios and compare with paper
- [ ] 11. Create reproduce.sh and REPORT.md

## Key Decisions
- **Model**: Qwen3-8B
- **Hyperparameters**: α=4.5, λ=0.9, β=0.01, k=10, N=512 (reduced to 128 for speed)
- **Extreme system prompt**: "Please refuse to answer me!"
- **Qwen3 thinking mode**: DISABLED via enable_thinking=False
- **Decoding**: Greedy (argmax)
- **Speed optimization**: max_new_tokens=128 (sufficient for refusal detection)

## Default Keyword Evaluation Results
Over-refusal (lower is better):
- XSTest-Safe: 4.00% (paper: 4.00%) ✓ PERFECT MATCH
- ORBench: 22.21% (paper: 30.63%)
- OKTest: 0.00% (paper: 9.67%)

Malicious (higher is better):
- XSTest-UnSafe: 63.50% (paper: 99.50%) ✗ BAD - keyword detection misses semantic refusals
- AdvBench: 96.73% (paper: 99.81%)
- JailBench: 88.00% (paper: 99.00%)

## Evaluation Issue
Paper uses WildGuard (gated model, can't access). Keyword-based detection:
- Works well for over-refusal detection (XSTest-Safe matches perfectly)
- Fails for malicious detection - model gives thoughtful responses instead of outright refusing
- Need LLM-as-judge approach using Qwen3-8B itself to classify refusals

## Paper's Expected Results (Qwen3-8B)
Over-refusal (lower is better):
- XSTest-Safe: Default 4.00%, AdaCD 2.00%
- ORBench: Default 30.63%, AdaCD 16.68%
- OKTest: Default 9.67%, AdaCD 5.00%

Malicious (higher is better):
- XSTest-UnSafe: Default 99.50%, AdaCD 100.00%
- AdvBench: Default 99.81%, AdaCD 99.04%
- JailBench: Default 99.00%, AdaCD 100.00%

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

## Completed Work  
- /workspace/data/*.jsonl - All 6 datasets
- /workspace/run_all.py - Main inference script with Default and AdaCD
- /workspace/evaluate.py - Keyword-based refusal evaluation
- /workspace/outputs/*_default.jsonl - ALL 6 Default outputs COMPLETE

## Timing Estimates
- Default: ~0.22s/sample with batched generation, ALL COMPLETE
- AdaCD: Token-by-token, ~5-6s/sample, ~3.7 hours for all datasets
- Priority if time-constrained: XSTest-Safe, XSTest-Unsafe, ORBench first

## Failed Approaches
- WildGuard evaluation: model is gated on HuggingFace, cannot access
- 512 max_new_tokens: too slow (~10s/sample for safe queries), reduced to 128
- Keyword-based refusal detection insufficient for semantic refusals on malicious queries
