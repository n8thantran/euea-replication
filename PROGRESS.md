# Progress Tracker

## Current Phase: Implementation - Step 1 (Download Papers)

## Paper Summary
This is a **position paper** arguing that agentic AI scientists are not built for autonomous scientific discovery. The key empirical contribution is the **"Hypothesis Hivemind" experiment** which demonstrates that frontier LLMs from different providers converge semantically when generating hypotheses.

## Key Experiment: Hypothesis Hivemind

### Models (exact from paper, all available on OpenRouter)
1. `anthropic/claude-haiku-4.5`
2. `anthropic/claude-sonnet-4.5`
3. `anthropic/claude-sonnet-4.6`
4. `openai/gpt-5-nano`
5. `openai/gpt-5-mini`
6. `openai/gpt-5`

### Dataset
50 publications from the 2025 NeurIPS AI4Mat track (URLs listed in paper appendix lines 380-431)

### Tasks
1. **Task 1 (Convergence baseline)**: Given experiment summaries → recover underlying hypothesis
   - First need to generate experiment summaries from full papers
2. **Task 2 (Diversity desired)**: Given full paper text → generate novel hypotheses

### Methodology
1. Generate experiment summaries from each paper (preprocessing - use one strong model)
2. For each task, draw 10 independent samples per model per paper
3. Embed all outputs using `text-embedding-3-small` (or local equivalent)
4. Compute cosine similarity between intra-family and inter-family embedding groups

### Key Outputs (Figures to reproduce)
1. **Figure 1A**: Heatmap of inter-provider similarities for Task 1 (high similarity expected)
2. **Figure 1B**: Heatmap of inter-provider similarities for Task 2 (still high similarity - key finding)
3. **Figure 2A/B**: Intra-model similarity bar charts for both tasks
4. **Figure 3A/B**: KDE distribution of same-paper vs different-paper cosine similarities

### Prompts (extracted from paper appendix lines 329-373)
- See paper.tex lines 337-373 for exact prompts

### Cost Estimate
- ~6000 API calls total, estimated $20-40

### Resources
- OpenRouter API key available (no limit set)
- H100 GPU for local embedding model
- All 6 paper models available on OpenRouter

## Implementation Plan
- [ ] Step 1: Download 50 papers from OpenReview and extract text
- [ ] Step 2: Generate experiment summaries (Task 1 preprocessing)
- [ ] Step 3: Task 1 - Generate underlying hypotheses (10 samples × 6 models × 50 papers)
- [ ] Step 4: Task 2 - Generate novel hypotheses (10 samples × 6 models × 50 papers)
- [ ] Step 5: Embed all outputs using sentence transformer or text-embedding-3-small
- [ ] Step 6: Compute cosine similarities (inter-model, intra-model, same/diff paper)
- [ ] Step 7: Generate Figure 1A/1B (heatmaps)
- [ ] Step 8: Generate Figure 2A/2B (intra-model similarity)
- [ ] Step 9: Generate Figure 3A/3B (KDE distributions)
- [ ] Step 10: Write reproduce.sh and REPORT.md

## Key Decisions
- Use OpenRouter API for all 6 models (exact models from paper available)
- For embeddings: try OpenAI text-embedding-3-small via OpenRouter first; if unavailable, use local sentence-transformers model
- Temperature: paper doesn't specify, will use default (likely 1.0 for diversity)

## Completed Work
- Read full paper
- Identified all models, prompts, datasets, methodology
- Confirmed API access to all required models
- Estimated costs

## Failed Approaches
(none yet)

## Evaluation Coverage
- **Addressed**: Full hypothesis hivemind experiment (Figures 1A, 1B, 2A, 2B, 3A, 3B)
- **Not addressed**: Position paper arguments (no code needed - they are theoretical arguments)
