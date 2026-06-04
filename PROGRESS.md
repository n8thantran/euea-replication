# Progress Tracker

## Current Phase: Generating Hypotheses (Task 2 in progress)

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
- [x] Step 3: Generate hypotheses Task 1 (3000/3000 done - all 6 models complete)
- [ ] Step 4: Generate hypotheses Task 2 (1791/3000 done - Claude models done, GPT models in progress)
  - claude-haiku-4.5: 500/500 ✓
  - claude-sonnet-4.5: 500/500 ✓
  - claude-sonnet-4.6: 500/500 ✓
  - gpt-5-nano: 291/500 (in progress)
  - gpt-5-mini: 0/500
  - gpt-5: 0/500
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

## Completed Work
- `download_papers.py`: Downloads 50 PDFs, extracts text → `data/pdfs/`, `data/texts/`, `data/papers_metadata.json`
- `generate_summaries.py`: Generates experiment summaries → `data/summaries/`
- `generate_hypotheses.py`: Generates hypotheses for Task 1 and Task 2 → `data/hypotheses/task1/`, `data/hypotheses/task2/`
- All 50 papers downloaded and text extracted
- All 50 experiment summaries generated
- Task 1: 3000/3000 hypotheses generated (complete)
- Task 2: 1791/3000 hypotheses generated (in progress)

## Failed Approaches
- Initial generate_hypotheses.py didn't handle reasoning models (GPT-5 family returns content=None with reasoning field). Fixed by checking for content properly.
- Git merge conflicts from parallel pushes - resolved with force push since local had complete data.

## Remaining Work (estimated ~5-6 more rounds)
1. Finish Task 2 generation (~1209 more API calls, ~3 rounds of 540s timeout)
2. Embed all 6000 hypotheses using text-embedding-3-small (~2 rounds)
3. Compute similarity matrices and generate figures (~1 round)
4. Create reproduce.sh and REPORT.md (~1 round)

## Evaluation Coverage
- **Addressed**: Hypothesis Hivemind experiment (the main empirical contribution)
  - Task 1: Hypothesis recovery from experiment summaries
  - Task 2: Novel hypothesis generation from full papers
  - 6 frontier LLMs across 2 providers (Anthropic, OpenAI)
  - 50 papers from NeurIPS 2025 AI4Mat track
- **Not addressed**: The rest is position paper argumentation (no code needed)
