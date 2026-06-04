#!/usr/bin/env python3
"""Embed all hypotheses using text-embedding-3-small via OpenRouter."""

import os
import json
import glob
import time
import numpy as np
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

API_KEY = os.environ.get("OPENROUTER_API_KEY")
EMBED_URL = "https://openrouter.ai/api/v1/embeddings"

MODELS = [
    "claude-haiku-4.5",
    "claude-sonnet-4.5",
    "claude-sonnet-4.6",
    "gpt-5-nano",
    "gpt-5-mini",
    "gpt-5",
]

def load_hypotheses(task):
    """Load all hypotheses for a task, return list of dicts with metadata."""
    hyps = []
    pattern = f"data/hypotheses/{task}/*.json"
    files = sorted(glob.glob(pattern))
    for f in files:
        data = json.load(open(f))
        # Parse filename: model_paperid_sample.json
        basename = os.path.basename(f).replace('.json', '')
        # Model name can contain hyphens, so we need to be careful
        # Format: {model}_{paperid}_{sample}
        parts = basename.rsplit('_', 1)  # split off sample number
        sample = int(parts[1])
        rest = parts[0]
        # Now split off paper_id (which doesn't contain underscores... actually it might)
        # Let's try matching against known model names
        model_name = None
        paper_id = None
        for m in MODELS:
            if rest.startswith(m + '_'):
                model_name = m
                paper_id = rest[len(m)+1:]
                break
        if model_name is None:
            print(f"WARNING: Could not parse {basename}")
            continue
        
        text = data.get("hypothesis", "")
        if not text:
            continue
            
        hyps.append({
            "task": task,
            "model": model_name,
            "paper_id": paper_id,
            "sample": sample,
            "text": text,
            "file": f,
        })
    return hyps


def embed_batch(texts, max_retries=5):
    """Embed a batch of texts using OpenRouter embedding API."""
    for attempt in range(max_retries):
        try:
            resp = requests.post(
                EMBED_URL,
                headers={
                    "Authorization": f"Bearer {API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "openai/text-embedding-3-small",
                    "input": texts,
                },
                timeout=120,
            )
            if resp.status_code == 200:
                data = resp.json()
                embeddings = [item["embedding"] for item in data["data"]]
                return embeddings
            elif resp.status_code == 429:
                wait = 2 ** attempt
                print(f"  Rate limited, waiting {wait}s...")
                time.sleep(wait)
            else:
                print(f"  Error {resp.status_code}: {resp.text[:200]}")
                time.sleep(2 ** attempt)
        except Exception as e:
            print(f"  Exception: {e}")
            time.sleep(2 ** attempt)
    return None


def main():
    os.makedirs("data/embeddings", exist_ok=True)
    
    for task in ["task1", "task2"]:
        output_file = f"data/embeddings/{task}_embeddings.npz"
        meta_file = f"data/embeddings/{task}_metadata.json"
        
        # Check if already done
        if os.path.exists(output_file) and os.path.exists(meta_file):
            meta = json.load(open(meta_file))
            print(f"{task}: Already embedded {len(meta)} hypotheses, skipping.")
            continue
        
        print(f"\n{'='*60}")
        print(f"Embedding {task} hypotheses")
        print(f"{'='*60}")
        
        hyps = load_hypotheses(task)
        print(f"Loaded {len(hyps)} hypotheses")
        
        # Embed in batches of 100
        batch_size = 100
        all_embeddings = []
        all_meta = []
        
        for i in range(0, len(hyps), batch_size):
            batch = hyps[i:i+batch_size]
            texts = [h["text"] for h in batch]
            
            print(f"  Embedding batch {i//batch_size + 1}/{(len(hyps)-1)//batch_size + 1} ({len(texts)} texts)...")
            embeddings = embed_batch(texts)
            
            if embeddings is None:
                print(f"  FAILED batch {i//batch_size + 1}, skipping")
                continue
            
            all_embeddings.extend(embeddings)
            for j, h in enumerate(batch):
                all_meta.append({
                    "task": h["task"],
                    "model": h["model"],
                    "paper_id": h["paper_id"],
                    "sample": h["sample"],
                    "idx": len(all_meta),
                })
            
            # Small delay to avoid rate limits
            time.sleep(0.5)
        
        # Save
        embeddings_array = np.array(all_embeddings, dtype=np.float32)
        np.savez_compressed(output_file, embeddings=embeddings_array)
        json.dump(all_meta, open(meta_file, 'w'), indent=2)
        
        print(f"\nSaved {len(all_meta)} embeddings to {output_file}")
        print(f"Shape: {embeddings_array.shape}")


if __name__ == "__main__":
    main()
