# Progress Tracker

## Current Phase: Planning

## Paper Summary
This is a **position paper** arguing that agentic AI scientists are not built for autonomous scientific discovery. The key empirical contribution is the **"Hypothesis Hivemind" experiment** which demonstrates that frontier LLMs from different providers converge semantically when generating hypotheses.

## Key Experiment: Hypothesis Hivemind

### Models
1. Claude Haiku 4.5
2. Claude Sonnet 4.5
3. Claude Sonnet 4.6
4. GPT-5 Nano
5. GPT-5 Mini
6. GPT-5

### Dataset
50 publications from the 2025 NeurIPS AI4Mat track (URLs listed in paper appendix)

### Tasks
1. **Task 1 (Convergence baseline)**: Given experiment summaries → recover underlying hypothesis
2. **Task 2 (Diversity desired)**: Given full paper text → generate novel hypotheses

### Methodology
1. Generate experiment summaries from each paper (preprocessing)
2. For each task, draw 10 independent samples per model per paper
3. Embed all outputs using `text-embedding-3-small`
4. Compute cosine similarity between intra-family and inter-family embedding groups

### Key Outputs (Figures)
1. **Figure 1A**: Heatmap of inter-provider similarities for Task 1 (high similarity expected)
2. **Figure 1B**: Heatmap of inter-provider similarities for Task 2 (still high similarity - key finding)
3. **Figure 2A/B**: Intra-model similarity plots for both tasks
4. **Figure 3A/B**: KDE distribution of same-paper vs different-paper cosine similarities

### Prompts (from paper appendix)
- Task 1 summary generation prompt
- Task 1 hypothesis recovery prompt
- Task 2 novel hypothesis generation prompt

## Implementation Plan
- [ ] Download 50 papers from OpenReview
- [ ] Extract text from PDFs
- [ ] Check for API access (OpenAI, Anthropic) or set up local models
- [ ] Implement Task 1: experiment summary generation
- [ ] Implement Task 1: underlying hypothesis generation (10 samples × 6 models × 50 papers)
- [ ] Implement Task 2: novel hypothesis generation (10 samples × 6 models × 50 papers)
- [ ] Embed all outputs
- [ ] Compute cosine similarities (inter-model, intra-model)
- [ ] Generate heatmap plots (Figure 1A, 1B)
- [ ] Generate intra-model similarity plots (Figure 2A, 2B)
- [ ] Generate KDE distribution plots (Figure 3A, 3B)
- [ ] Write reproduce.sh
- [ ] Write REPORT.md

## Key Decisions
- Need to determine which models to use (API or local)
- May use available open-source models as substitutes for future models listed in paper
- For embeddings, use text-embedding-3-small (OpenAI) or local alternative

## Completed Work
(none yet)

## Failed Approaches
(none yet)
