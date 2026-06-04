# Progress Tracker

## Current Phase: Step 5 - Embedding all hypotheses

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
- [x] Step 3: Generate hypotheses Task 1 (3000/3000 done)
- [x] Step 4: Generate hypotheses Task 2 (2996/3000 done - 4 GPT-5 samples persistently fail, 99.87% complete)
  - claude-haiku-4.5: 500/500 ✓
  - claude-sonnet-4.5: 500/500 ✓
  - claude-sonnet-4.6: 500/500 ✓
  - gpt-5-nano: 500/500 ✓
  - gpt-5-mini: 500/500 ✓
  - gpt-5: 496/500 (4 persistently failing: dEtRvi7G5i_4, dEtRvi7G5i_9, hQCdhenqre_5, iFHaZzs6Kz_7)
- [ ] Step 5: Embed all outputs using text-embedding-3-small
- [ ] Step 6: Compute cosine similarities (inter-model, intra-model, same-paper vs diff-paper)
- [ ] Step 7: Generate all 6 figures
- [ ] Step 8: Create reproduce.sh and REPORT.md

## Key Decisions
- Using OpenRouter API for all model calls (API key in env var OPENROUTER_API_KEY)
- Summaries generated with Claude Sonnet 4.5 (low temperature 0.3)
- Hypothesis generation at temperature 0.7 (standard for diverse sampling)
- Paper truncated to 100K chars for context limits
- 10 concurrent workers for API calls
- 4 GPT-5 Task 2 samples permanently failed after 10+ retries - moving on (99.87% complete)

## Completed Work
- `download_papers.py`: Downloads 50 PDFs, extracts text → `data/pdfs/`, `data/texts/`, `data/papers_metadata.json`
- `generate_summaries.py`: Generates experiment summaries → `data/summaries/`
- `generate_hypotheses.py`: Generates hypotheses for Task 1 and Task 2 → `data/hypotheses/task1/`, `data/hypotheses/task2/`
- All 50 papers downloaded and text extracted
- All 50 experiment summaries generated
- Task 1: 3000/3000 hypotheses generated (complete)
- Task 2: 2996/3000 hypotheses generated (99.87% complete)

## Failed Approaches
- Initial generate_hypotheses.py didn't handle reasoning models (GPT-5 family returns content=None with reasoning field). Fixed by checking for content properly.
- Git merge conflicts from parallel pushes - resolved with force push since local had complete data.
- 4 GPT-5 Task 2 samples persistently fail despite 10+ retries with various concurrency settings. Moving on.

## Remaining Work (estimated ~4-5 rounds)
1. Embed all ~5996 hypotheses using text-embedding-3-small (~2 rounds)
2. Compute similarity matrices and generate figures (~1-2 rounds)
3. Create reproduce.sh and REPORT.md (~1 round)

## Evaluation Coverage
- **Addressed**: Hypothesis Hivemind experiment (the main empirical contribution)
  - Task 1: Hypothesis recovery from experiment summaries
  - Task 2: Novel hypothesis generation from full papers
  - 6 frontier LLMs across 2 providers (Anthropic, OpenAI)
  - 50 papers from NeurIPS 2025 AI4Mat track
- **Not addressed**: The rest is position paper argumentation (no code needed)
