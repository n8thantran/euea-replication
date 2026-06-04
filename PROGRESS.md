# Progress Tracker

## Current Phase: Generating Hypotheses (Task 1 & Task 2)

## Paper Summary
Position paper arguing agentic AI scientists aren't built for autonomous discovery. Key empirical contribution: **Hypothesis Hivemind** experiment showing frontier LLMs converge semantically when generating hypotheses.

## Key Experiment: Hypothesis Hivemind

### Models (exact OpenRouter IDs)
1. `anthropic/claude-haiku-4.5`
2. `anthropic/claude-sonnet-4.5`
3. `anthropic/claude-sonnet-4.6`
4. `openai/gpt-5-nano`
5. `openai/gpt-5-mini`
6. `openai/gpt-5`

### Dataset
50 publications from 2025 NeurIPS AI4Mat track

### Tasks
- **Task 1**: Given experiment summaries → recover underlying hypothesis (convergence baseline)
- **Task 2**: Given full paper text → generate novel hypotheses (diversity desired)

### Sampling
- 10 independent samples per model per paper per task
- Total: 6000 API calls (3000 per task)
- Plus 50 summary generation calls (already done)

### Embedding
- Use `text-embedding-3-small` (OpenAI via OpenRouter) as in paper
- Or local sentence-transformers as backup

### Figures to Reproduce
1. **Figure 1A (heatmap_A)**: Inter-provider similarity heatmap, Task 1
2. **Figure 1B (heatmap_B)**: Inter-provider similarity heatmap, Task 2
3. **Figure 2A**: Intra-model similarity bar chart, Task 1
4. **Figure 2B**: Intra-model similarity bar chart, Task 2
5. **Figure 3A**: KDE of same-paper vs different-paper similarities, Task 1
6. **Figure 3B**: KDE of same-paper vs different-paper similarities, Task 2

## Implementation Plan
- [x] Step 0: Clean up unrelated files
- [x] Step 1: Download 50 papers from OpenReview and extract text
- [x] Step 2: Generate experiment summaries (50/50 done)
- [ ] Step 3: Generate hypotheses Task 1 (6 models × 50 papers × 10 samples = 3000 calls)
- [ ] Step 4: Generate hypotheses Task 2 (6 models × 50 papers × 10 samples = 3000 calls)
- [ ] Step 5: Embed all outputs using text-embedding-3-small
- [ ] Step 6: Compute cosine similarities (inter-model, intra-model, same-paper vs diff-paper)
- [ ] Step 7: Generate all 6 figures
- [ ] Step 8: Create reproduce.sh and REPORT.md

## Key Decisions
- Using OpenRouter API for all model calls (API key in env var OPENROUTER_API_KEY)
- Summaries generated with Claude Sonnet 4.5 (low temperature 0.3)
- Hypothesis generation at temperature 0.7 (standard for diverse sampling)
- Paper truncated to 100K chars for context limits

## Completed Work
- `download_papers.py`: Downloads 50 PDFs, extracts text → `data/pdfs/`, `data/texts/`, `data/papers_metadata.json`
- `generate_summaries.py`: Generates experiment summaries → `data/summaries/`
- All 50 papers downloaded and text extracted
- All 50 experiment summaries generated

## Failed Approaches
- None so far

## Evaluation Coverage
- **Addressed**: Hypothesis Hivemind experiment (the main empirical contribution)
- **Not addressed**: The rest is position paper argumentation (no code needed)
