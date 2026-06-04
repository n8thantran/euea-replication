#!/usr/bin/env python3
"""
Generate all 6 figures for the Hypothesis Hivemind experiment.

Figures:
1. Figure 1A (heatmap_A): Inter-provider similarity heatmap, Task 1
2. Figure 1B (heatmap_B): Inter-provider similarity heatmap, Task 2
3. Figure 2A: Intra-model similarity bar chart, Task 1
4. Figure 2B: Intra-model similarity bar chart, Task 2
5. Figure 3A: KDE of same-paper vs different-paper similarities, Task 1
6. Figure 3B: KDE of same-paper vs different-paper similarities, Task 2
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from collections import defaultdict
from scipy.stats import gaussian_kde

# Model display names (short names for plots)
MODEL_DISPLAY = {
    "claude-haiku-4.5": "Claude Haiku 4.5",
    "claude-sonnet-4.5": "Claude Sonnet 4.5",
    "claude-sonnet-4.6": "Claude Sonnet 4.6",
    "gpt-5-nano": "GPT-5 Nano",
    "gpt-5-mini": "GPT-5 Mini",
    "gpt-5": "GPT-5",
}

MODEL_ORDER = [
    "claude-haiku-4.5",
    "claude-sonnet-4.5",
    "claude-sonnet-4.6",
    "gpt-5-nano",
    "gpt-5-mini",
    "gpt-5",
]


def load_data(task):
    """Load embeddings and metadata for a task."""
    emb_file = f"data/embeddings/{task}_embeddings.npz"
    meta_file = f"data/embeddings/{task}_metadata.json"
    
    embeddings = np.load(emb_file)["embeddings"]
    metadata = json.load(open(meta_file))
    
    return embeddings, metadata


def cosine_similarity(a, b):
    """Compute cosine similarity between two vectors."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def cosine_similarity_matrix(embeddings):
    """Compute pairwise cosine similarity matrix."""
    # Normalize
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normed = embeddings / norms
    return normed @ normed.T


def build_index(metadata):
    """Build indices mapping (model, paper_id) -> list of embedding indices."""
    idx = defaultdict(list)
    for i, m in enumerate(metadata):
        idx[(m["model"], m["paper_id"])].append(i)
    return idx


def compute_inter_model_similarity(embeddings, metadata):
    """
    Compute average cosine similarity between all pairs of models.
    For each pair of models (A, B), for each paper, compute average cosine
    similarity between all outputs of A and all outputs of B for that paper.
    Then average across papers.
    """
    idx = build_index(metadata)
    papers = sorted(set(m["paper_id"] for m in metadata))
    
    # Precompute normalized embeddings
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normed = embeddings / norms
    
    n_models = len(MODEL_ORDER)
    sim_matrix = np.zeros((n_models, n_models))
    
    for i, model_a in enumerate(MODEL_ORDER):
        for j, model_b in enumerate(MODEL_ORDER):
            paper_sims = []
            for paper in papers:
                indices_a = idx.get((model_a, paper), [])
                indices_b = idx.get((model_b, paper), [])
                if not indices_a or not indices_b:
                    continue
                
                # Compute all pairwise similarities
                emb_a = normed[indices_a]  # (na, d)
                emb_b = normed[indices_b]  # (nb, d)
                sims = emb_a @ emb_b.T  # (na, nb)
                
                if i == j:
                    # For same model, exclude self-comparisons (diagonal)
                    # Only take upper triangle (excluding diagonal)
                    mask = np.triu(np.ones_like(sims, dtype=bool), k=1)
                    if mask.sum() > 0:
                        paper_sims.append(sims[mask].mean())
                else:
                    paper_sims.append(sims.mean())
            
            if paper_sims:
                sim_matrix[i, j] = np.mean(paper_sims)
    
    return sim_matrix


def compute_intra_model_similarity(embeddings, metadata):
    """
    Compute intra-model similarity: for each model, for each paper,
    compute average pairwise cosine similarity among the 10 samples.
    Return mean and std across papers for each model.
    """
    idx = build_index(metadata)
    papers = sorted(set(m["paper_id"] for m in metadata))
    
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normed = embeddings / norms
    
    results = {}
    for model in MODEL_ORDER:
        paper_sims = []
        for paper in papers:
            indices = idx.get((model, paper), [])
            if len(indices) < 2:
                continue
            emb = normed[indices]
            sim = emb @ emb.T
            # Upper triangle excluding diagonal
            mask = np.triu(np.ones_like(sim, dtype=bool), k=1)
            paper_sims.append(sim[mask].mean())
        
        results[model] = {
            "mean": np.mean(paper_sims),
            "std": np.std(paper_sims),
        }
    
    return results


def compute_same_diff_paper_sims(embeddings, metadata):
    """
    Compute cosine similarities for:
    - Same paper: all pairs of outputs (any model) for the same paper
    - Different paper: all pairs of outputs (any model) for different papers
    
    Returns two arrays of similarity values.
    """
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normed = embeddings / norms
    
    # Group by paper
    paper_indices = defaultdict(list)
    for i, m in enumerate(metadata):
        paper_indices[m["paper_id"]].append(i)
    
    papers = sorted(paper_indices.keys())
    
    same_sims = []
    diff_sims = []
    
    # For same-paper: compute within each paper
    for paper in papers:
        indices = paper_indices[paper]
        if len(indices) < 2:
            continue
        emb = normed[indices]
        sim = emb @ emb.T
        mask = np.triu(np.ones_like(sim, dtype=bool), k=1)
        same_sims.extend(sim[mask].tolist())
    
    # For different-paper: sample pairs to keep computation manageable
    # There are ~3000 embeddings, so full matrix is 3000x3000 = 9M pairs
    # Sample a subset for KDE
    np.random.seed(42)
    n_samples = 500000  # 500K samples should be enough for KDE
    
    all_papers = [m["paper_id"] for m in metadata]
    n = len(normed)
    
    count = 0
    while count < n_samples:
        batch_size = min(n_samples - count, 100000)
        i_idx = np.random.randint(0, n, batch_size)
        j_idx = np.random.randint(0, n, batch_size)
        
        # Filter: different papers and i < j
        valid = []
        for k in range(batch_size):
            if i_idx[k] != j_idx[k] and all_papers[i_idx[k]] != all_papers[j_idx[k]]:
                valid.append(k)
        
        if valid:
            valid = np.array(valid)
            sims = np.sum(normed[i_idx[valid]] * normed[j_idx[valid]], axis=1)
            diff_sims.extend(sims.tolist())
            count += len(valid)
    
    return np.array(same_sims), np.array(diff_sims[:n_samples])


def plot_heatmap(sim_matrix, task_label, filename):
    """Plot inter-model similarity heatmap."""
    fig, ax = plt.subplots(figsize=(8, 7))
    
    labels = [MODEL_DISPLAY[m] for m in MODEL_ORDER]
    
    im = ax.imshow(sim_matrix, cmap='YlOrRd', vmin=0.4, vmax=1.0, aspect='equal')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Average Cosine Similarity', fontsize=12)
    
    # Set ticks
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=10)
    ax.set_yticklabels(labels, fontsize=10)
    
    # Add text annotations
    for i in range(len(labels)):
        for j in range(len(labels)):
            text = f"{sim_matrix[i, j]:.3f}"
            color = "white" if sim_matrix[i, j] > 0.8 else "black"
            ax.text(j, i, text, ha="center", va="center", color=color, fontsize=9)
    
    ax.set_title(f"Inter-Model Similarity — {task_label}", fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {filename}")


def plot_intra_model_bar(results, task_label, filename):
    """Plot intra-model similarity bar chart."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    labels = [MODEL_DISPLAY[m] for m in MODEL_ORDER]
    means = [results[m]["mean"] for m in MODEL_ORDER]
    stds = [results[m]["std"] for m in MODEL_ORDER]
    
    colors = ['#2196F3', '#1976D2', '#0D47A1', '#FF9800', '#F57C00', '#E65100']
    
    bars = ax.bar(range(len(labels)), means, yerr=stds, capsize=5,
                  color=colors, edgecolor='black', linewidth=0.5, alpha=0.85)
    
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=10)
    ax.set_ylabel('Average Intra-Model Cosine Similarity', fontsize=12)
    ax.set_title(f"Intra-Model Similarity — {task_label}", fontsize=14, fontweight='bold')
    
    # Add value labels on bars
    for i, (mean, std) in enumerate(zip(means, stds)):
        ax.text(i, mean + std + 0.01, f"{mean:.3f}", ha='center', va='bottom', fontsize=9)
    
    ax.set_ylim(0, 1.0)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {filename}")


def plot_kde(same_sims, diff_sims, task_label, filename):
    """Plot KDE of same-paper vs different-paper similarities."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Compute KDEs
    x = np.linspace(-0.2, 1.0, 1000)
    
    kde_same = gaussian_kde(same_sims, bw_method=0.05)
    kde_diff = gaussian_kde(diff_sims, bw_method=0.05)
    
    ax.fill_between(x, kde_same(x), alpha=0.4, color='#2196F3', label='Same paper')
    ax.fill_between(x, kde_diff(x), alpha=0.4, color='#FF5722', label='Different paper')
    ax.plot(x, kde_same(x), color='#1565C0', linewidth=2)
    ax.plot(x, kde_diff(x), color='#BF360C', linewidth=2)
    
    ax.set_xlabel('Cosine Similarity', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title(f"Same vs Different Paper Similarity — {task_label}", fontsize=14, fontweight='bold')
    ax.legend(fontsize=12)
    
    # Add vertical lines for means
    ax.axvline(np.mean(same_sims), color='#1565C0', linestyle='--', alpha=0.7, 
               label=f'Same mean: {np.mean(same_sims):.3f}')
    ax.axvline(np.mean(diff_sims), color='#BF360C', linestyle='--', alpha=0.7,
               label=f'Diff mean: {np.mean(diff_sims):.3f}')
    ax.legend(fontsize=11)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {filename}")


def main():
    os.makedirs("results", exist_ok=True)
    
    # Store numerical results
    all_results = {}
    
    for task, task_label_A, task_label_B in [
        ("task1", "Recover Underlying Hypothesis (Task 1)", "(A) Convergence desired"),
        ("task2", "Generate Novel Hypothesis (Task 2)", "(B) Diversity desired"),
    ]:
        print(f"\n{'='*60}")
        print(f"Processing {task}")
        print(f"{'='*60}")
        
        embeddings, metadata = load_data(task)
        print(f"Loaded {len(metadata)} embeddings, shape {embeddings.shape}")
        
        # 1. Inter-model similarity heatmap
        print("\nComputing inter-model similarity...")
        sim_matrix = compute_inter_model_similarity(embeddings, metadata)
        print("Inter-model similarity matrix:")
        for i, m in enumerate(MODEL_ORDER):
            row = " ".join(f"{sim_matrix[i,j]:.3f}" for j in range(len(MODEL_ORDER)))
            print(f"  {MODEL_DISPLAY[m]:25s}: {row}")
        
        suffix = "A" if task == "task1" else "B"
        plot_heatmap(sim_matrix, task_label_B, f"results/heatmap_{suffix}.pdf")
        plot_heatmap(sim_matrix, task_label_B, f"results/heatmap_{suffix}.png")
        
        # 2. Intra-model similarity
        print("\nComputing intra-model similarity...")
        intra_results = compute_intra_model_similarity(embeddings, metadata)
        for m in MODEL_ORDER:
            print(f"  {MODEL_DISPLAY[m]:25s}: {intra_results[m]['mean']:.3f} ± {intra_results[m]['std']:.3f}")
        
        intra_label = "Recover underlying hypothesis" if task == "task1" else "Generate novel hypothesis"
        intra_file = "intra_model_repetition_underlying_hypotheses" if task == "task1" else "intra_model_repetition_new_hypotheses"
        plot_intra_model_bar(intra_results, intra_label, f"results/{intra_file}.pdf")
        plot_intra_model_bar(intra_results, intra_label, f"results/{intra_file}.png")
        
        # 3. Same vs different paper KDE
        print("\nComputing same vs different paper similarities...")
        same_sims, diff_sims = compute_same_diff_paper_sims(embeddings, metadata)
        print(f"  Same paper: {len(same_sims)} pairs, mean={np.mean(same_sims):.3f}")
        print(f"  Diff paper: {len(diff_sims)} pairs, mean={np.mean(diff_sims):.3f}")
        
        kde_file = f"intra_inter_kde_pooled_{suffix}"
        plot_kde(same_sims, diff_sims, intra_label, f"results/{kde_file}.pdf")
        plot_kde(same_sims, diff_sims, intra_label, f"results/{kde_file}.png")
        
        # Store results
        all_results[task] = {
            "inter_model_similarity": {
                MODEL_ORDER[i]: {MODEL_ORDER[j]: float(sim_matrix[i,j]) for j in range(len(MODEL_ORDER))}
                for i in range(len(MODEL_ORDER))
            },
            "intra_model_similarity": {
                m: {"mean": float(intra_results[m]["mean"]), "std": float(intra_results[m]["std"])}
                for m in MODEL_ORDER
            },
            "same_paper_similarity": {
                "mean": float(np.mean(same_sims)),
                "std": float(np.std(same_sims)),
                "n_pairs": len(same_sims),
            },
            "diff_paper_similarity": {
                "mean": float(np.mean(diff_sims)),
                "std": float(np.std(diff_sims)),
                "n_pairs": len(diff_sims),
            },
        }
    
    # Save numerical results
    json.dump(all_results, open("results/numerical_results.json", 'w'), indent=2)
    print(f"\nSaved numerical results to results/numerical_results.json")
    
    print("\n" + "="*60)
    print("ALL FIGURES GENERATED SUCCESSFULLY")
    print("="*60)


if __name__ == "__main__":
    main()
