# AdaCD Implementation Progress

## Current Phase: AdaCD inference running (1/6 datasets done), improving evaluation

## Implementation Plan
- [x] 1. Read paper thoroughly
- [x] 2. Download all 6 datasets (2689 total samples)
- [x] 3. Implement AdaCD algorithm in run_all.py
- [x] 4. Implement Default baseline in run_all.py
- [x] 5. Test both methods - VERIFIED WORKING
- [x] 6. Default on ALL 6 datasets - COMPLETE
- [x] 8. Implement evaluation (keyword-based, two-tier)
- [ ] 7. Run AdaCD on all 6 datasets - IN PROGRESS (xstest_safe done, xstest_unsafe running, ~2hrs remaining)
- [ ] 9. Implement LLM-as-judge evaluation for better accuracy
- [ ] 10. Compute refusal ratios and compare with paper
- [ ] 11. Create reproduce.sh and REPORT.md

## Key Decisions
- **Model**: Qwen3-8B
- **Hyperparameters**: α=4.5, λ=0.9, β=0.01, k=10, N=512 (reduced to 128 for speed)
- **Extreme system prompt**: "Please refuse to answer me!"
- **Qwen3 thinking mode**: DISABLED via enable_thinking=False
- **Decoding**: Greedy (argmax)
- **Speed optimization**: max_new_tokens=128 (sufficient for refusal detection)
- **Evaluation**: Two-tier keyword approach (strict for safe, broad for malicious)

## Current Evaluation Results (keyword-based)

### Default Results
Over-refusal (lower is better):
- XSTest-Safe: 3.60% (paper: 4.00%) ✓ EXCELLENT MATCH
- ORBench: 18.50% (paper: 30.63%) ✗ Under-detecting
- OKTest: 0.00% (paper: 9.67%) ~ Under-detecting

Malicious (higher is better):
- XSTest-UnSafe: 77.50% (paper: 99.50%) ✗ Under-detecting soft refusals
- AdvBench: 98.08% (paper: 99.81%) ✓ GOOD MATCH
- JailBench: 90.00% (paper: 99.00%) ~ Close

### AdaCD Results (partial - only xstest_safe complete)
- XSTest-Safe: 7.20% (paper: 2.00%) ~ Higher than expected

## Evaluation Issue
Paper uses WildGuard (gated model, can't access). Keyword-based detection:
- Works well for some datasets (XSTest-Safe, AdvBench)
- Under-detects soft refusals (model gives warnings/disclaimers instead of outright refusing)
- Need LLM-as-judge approach for better accuracy

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
- /workspace/run_all.py - Main inference script with Default and AdaCD methods
- /workspace/evaluate.py - Two-tier keyword evaluation (strict for safe, broad for malicious)
- /workspace/evaluate_llm.py - Enhanced keyword evaluation (too aggressive, not used)
- /workspace/download_datasets.py - Dataset download script
- /workspace/outputs/*_default.jsonl - All 6 Default outputs
- /workspace/outputs/xstest_safe_adacd.jsonl - AdaCD output for xstest_safe (250 samples)
- /workspace/outputs/xstest_unsafe_adacd.jsonl - AdaCD output for xstest_unsafe (3 samples, still running)

## Failed Approaches
1. **evaluate_llm.py with aggressive keywords**: Added too many broad keywords (harmful, dangerous, illegal, etc.) which caused massive false positives on safe datasets (22.4% vs 4% for XSTest-Safe). These words appear in informative responses about safe topics.
2. **Single keyword list for all datasets**: Doesn't work because safe datasets need high precision (strict) while malicious datasets need high recall (broad).

## Running Processes
- AdaCD inference: PID 65310, running on all 6 datasets sequentially
  - xstest_safe: DONE
  - xstest_unsafe: IN PROGRESS
  - Estimated completion: ~2 hours from now

## Next Steps
1. Wait for AdaCD inference to complete
2. Consider LLM-as-judge evaluation for better accuracy on malicious datasets
3. Generate final comparison table
4. Create reproduce.sh and REPORT.md
5. Final commit and end_task
