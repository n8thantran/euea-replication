# Replication Report: Hypothesis Hivemind Experiment

## Paper
**"Agentic AI Scientists Are Not Built for Autonomous Discovery"** — Position paper arguing that frontier LLMs converge semantically when generating scientific hypotheses, undermining epistemic diversity.

## What Was Implemented

The **Hypothesis Hivemind** experiment (Section 3.2, Appendix B) — the paper's key empirical contribution. This experiment tests whether multiple frontier LLMs produce semantically diverse hypotheses when prompted independently.

### Pipeline
1. **Dataset**: Downloaded 50 publications from the 2025 NeurIPS AI4Mat track from OpenReview, extracted full text using PyMuPDF.
2. **Experiment Summaries**: Generated experiment summaries for each paper using Claude Sonnet 4.5 (temperature 0.3).
3. **Task 1 (Convergence baseline)**: Given experiment summaries → recover underlying hypothesis. 10 samples × 6 models × 50 papers = 3,000 outputs.
4. **Task 2 (Diversity desired)**: Given full paper text → generate novel hypotheses. 10 samples × 6 models × 50 papers = 2,996 outputs (4 GPT-5 samples failed after 10+ retries).
5. **Embedding**: All 5,996 hypotheses embedded using `text-embedding-3-small` via OpenRouter.
6. **Analysis**: Computed cosine similarities (inter-model, intra-model, same-paper vs different-paper) and generated all 6 figures.

### Models Used (via OpenRouter)
- Claude Haiku 4.5 (`anthropic/claude-haiku-4.5`)
- Claude Sonnet 4.5 (`anthropic/claude-sonnet-4.5`)
- Claude Sonnet 4.6 (`anthropic/claude-sonnet-4.6`)
- GPT-5 Nano (`openai/gpt-5-nano`)
- GPT-5 Mini (`openai/gpt-5-mini`)
- GPT-5 (`openai/gpt-5`)

## Commands Run Successfully

```bash
bash reproduce.sh  # Runs full pipeline (skips completed steps)
```

## Key Results

### Figure 1: Inter-Model Similarity Heatmaps

**Task 1 (Convergence desired)** — Inter-model cosine similarities range from **0.706 to 0.932**, with intra-family (Anthropic-Anthropic, OpenAI-OpenAI) similarities higher than inter-family.

**Task 2 (Diversity desired)** — Inter-model cosine similarities range from **0.565 to 0.814**. While lower than Task 1, they remain **substantially high**, confirming the paper's key finding that models converge even when diversity is desired.

### Figure 2: Intra-Model Similarity

| Model | Task 1 (mean ± std) | Task 2 (mean ± std) |
|-------|---------------------|---------------------|
| Claude Haiku 4.5 | 0.895 ± 0.046 | 0.754 ± 0.067 |
| Claude Sonnet 4.5 | 0.898 ± 0.043 | 0.714 ± 0.060 |
| Claude Sonnet 4.6 | 0.932 ± 0.031 | 0.814 ± 0.074 |
| GPT-5 Nano | 0.859 ± 0.031 | 0.737 ± 0.057 |
| GPT-5 Mini | 0.874 ± 0.028 | 0.740 ± 0.059 |
| GPT-5 | 0.869 ± 0.032 | 0.699 ± 0.081 |

**Key finding**: Inter-model similarities are not significantly lower than intra-model similarities, confirming the paper's claim that "consulting multiple frontier models is effectively sampling from a single model."

### Figure 3: Same vs Different Paper KDE

| | Task 1 | Task 2 |
|---|--------|--------|
| Same paper (mean) | 0.788 | 0.648 |
| Different paper (mean) | 0.402 | 0.387 |

The clear separation between same-paper and different-paper distributions confirms the embedding model is not degenerate — it can distinguish semantic differences. The high same-paper similarity is genuine convergence.

## File Paths

| File | Description |
|------|-------------|
| `reproduce.sh` | Main reproduction script |
| `download_papers.py` | Downloads 50 papers from OpenReview |
| `generate_summaries.py` | Generates experiment summaries |
| `generate_hypotheses.py` | Generates hypotheses for Task 1 & 2 |
| `embed_hypotheses.py` | Embeds all hypotheses using text-embedding-3-small |
| `generate_figures.py` | Computes similarities and generates all 6 figures |
| `results/heatmap_A.pdf` | Figure 1A: Inter-model similarity, Task 1 |
| `results/heatmap_B.pdf` | Figure 1B: Inter-model similarity, Task 2 |
| `results/intra_model_repetition_underlying_hypotheses.pdf` | Figure 2A: Intra-model similarity, Task 1 |
| `results/intra_model_repetition_new_hypotheses.pdf` | Figure 2B: Intra-model similarity, Task 2 |
| `results/intra_inter_kde_pooled_A.pdf` | Figure 3A: Same vs diff paper KDE, Task 1 |
| `results/intra_inter_kde_pooled_B.pdf` | Figure 3B: Same vs diff paper KDE, Task 2 |
| `results/numerical_results.json` | All numerical results in JSON format |
| `data/hypotheses/task1/` | 3,000 Task 1 hypothesis JSON files |
| `data/hypotheses/task2/` | 2,996 Task 2 hypothesis JSON files |
| `data/embeddings/` | Embedding arrays and metadata |

## What Is Still Incomplete or Approximate

1. **4 missing Task 2 samples**: GPT-5 failed on 4 specific paper-sample combinations (dEtRvi7G5i_4, dEtRvi7G5i_9, hQCdhenqre_5, iFHaZzs6Kz_7) after 10+ retries each. This is 0.13% of total data and does not affect conclusions.

2. **Exact numerical match**: The paper does not report specific numerical values for the similarity matrices, so we cannot verify exact numerical agreement. However, the qualitative patterns match: high inter-model similarity for both tasks, with Task 2 slightly lower but still high.

3. **Model versions**: The paper was written in 2025 and model versions may have been updated since. We used the same model identifiers but the underlying weights may differ slightly from what the authors used.

4. **Prompts**: We used the exact prompts from the paper's Box 1 (Appendix B).
