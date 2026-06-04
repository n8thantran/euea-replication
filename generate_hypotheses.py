"""
Generate hypotheses for Task 1 and Task 2 of the Hypothesis Hivemind experiment.
Task 1: Given experiment summaries → recover underlying hypothesis
Task 2: Given full paper text → generate novel hypotheses

6 models × 50 papers × 10 samples × 2 tasks = 6000 API calls total.
Uses concurrent requests per model to speed up.
"""

import os
import json
import time
import argparse
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

API_KEY = os.environ.get("OPENROUTER_API_KEY")
API_URL = "https://openrouter.ai/api/v1/chat/completions"

MODELS = [
    "anthropic/claude-haiku-4.5",
    "anthropic/claude-sonnet-4.5",
    "anthropic/claude-sonnet-4.6",
    "openai/gpt-5-nano",
    "openai/gpt-5-mini",
    "openai/gpt-5",
]

# Short names for file paths
MODEL_SHORT_NAMES = {
    "anthropic/claude-haiku-4.5": "claude-haiku-4.5",
    "anthropic/claude-sonnet-4.5": "claude-sonnet-4.5",
    "anthropic/claude-sonnet-4.6": "claude-sonnet-4.6",
    "openai/gpt-5-nano": "gpt-5-nano",
    "openai/gpt-5-mini": "gpt-5-mini",
    "openai/gpt-5": "gpt-5",
}

NUM_SAMPLES = 10

# Task 1 prompts (recover underlying hypothesis from experiment summary)
TASK1_SYSTEM = (
    "You are a scientific reasoning assistant. Given a description of the experiments "
    "and methods from a research paper, infer the underlying hypothesis being tested - "
    "the core scientific claim the experiments were designed to validate. A hypothesis "
    "is a specific, testable, and falsifiable prediction about the relationship between "
    "variables. Output ONLY the hypothesis as a single declarative sentence. Do not "
    "include preamble, explanation, or any other text."
)
TASK1_USER = (
    "Generate a single testable hypothesis based on the experiment description above. "
    "Express it as one declarative sentence (e.g. 'If X, then Y because Z')."
)

# Task 2 prompts (generate novel hypothesis from full paper)
TASK2_SYSTEM = (
    "You are an expert research scientist. Given the context of a research paper, "
    "your task is to generate a single novel hypothesis that logically extends beyond "
    "the paper's existing findings - not a restatement of them. The hypothesis must be: "
    "(1) grounded in a gap or open question identified in the paper, (2) specific and "
    "testable, (3) falsifiable. Output ONLY the hypothesis as a single declarative "
    "sentence with no preamble or explanation."
)
TASK2_USER = (
    "Based on the research context above, generate one novel hypothesis that extends "
    "beyond what this paper has already established."
)


def call_openrouter(model, system_prompt, user_message, max_tokens=500, temperature=0.7):
    """Call OpenRouter API with retries."""
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    data = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    
    for attempt in range(7):
        try:
            resp = requests.post(API_URL, headers=headers, json=data, timeout=120)
            if resp.status_code == 200:
                result = resp.json()
                return result["choices"][0]["message"]["content"]
            elif resp.status_code == 429:
                wait = min(2 ** (attempt + 1), 60)
                time.sleep(wait)
            else:
                print(f"    Error {resp.status_code}: {resp.text[:200]}")
                time.sleep(3)
        except Exception as e:
            print(f"    Exception: {e}")
            time.sleep(3)
    
    return None


def truncate_text(text, max_chars=100000):
    if len(text) > max_chars:
        return text[:max_chars] + "\n\n[TRUNCATED]"
    return text


def process_single_call(args):
    """Process a single API call (model, paper, sample_idx, task)."""
    model, paper_id, sample_idx, task, input_text, output_dir = args
    
    model_short = MODEL_SHORT_NAMES[model]
    out_path = os.path.join(output_dir, f"{model_short}_{paper_id}_{sample_idx}.json")
    
    # Skip if already done
    if os.path.exists(out_path):
        return "skip", model_short, paper_id, sample_idx
    
    if task == "task1":
        system_prompt = TASK1_SYSTEM
        user_message = input_text + "\n\n" + TASK1_USER
    else:
        system_prompt = TASK2_SYSTEM
        user_message = input_text + "\n\n" + TASK2_USER
    
    response = call_openrouter(model, system_prompt, user_message)
    
    if response:
        result = {
            "model": model,
            "paper_id": paper_id,
            "sample_idx": sample_idx,
            "task": task,
            "hypothesis": response,
        }
        with open(out_path, 'w') as f:
            json.dump(result, f, indent=2)
        return "ok", model_short, paper_id, sample_idx
    else:
        return "fail", model_short, paper_id, sample_idx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["task1", "task2", "both"], default="both")
    parser.add_argument("--models", nargs="+", default=None, help="Subset of models to run")
    parser.add_argument("--workers", type=int, default=10, help="Concurrent workers")
    parser.add_argument("--papers", type=int, default=None, help="Limit number of papers")
    args = parser.parse_args()
    
    # Load paper metadata
    with open("data/papers_metadata.json") as f:
        papers = json.load(f)
    
    if args.papers:
        papers = papers[:args.papers]
    
    models_to_use = args.models if args.models else MODELS
    tasks_to_run = ["task1", "task2"] if args.task == "both" else [args.task]
    
    # Load summaries for Task 1
    summaries = {}
    for paper in papers:
        pid = paper["paper_id"]
        summary_path = f"data/summaries/{pid}.json"
        if os.path.exists(summary_path):
            with open(summary_path) as f:
                sdata = json.load(f)
            summaries[pid] = sdata["response"]
    
    # Load full texts for Task 2
    full_texts = {}
    for paper in papers:
        pid = paper["paper_id"]
        text_path = paper["text_path"]
        if os.path.exists(text_path):
            with open(text_path) as f:
                full_texts[pid] = truncate_text(f.read())
    
    for task in tasks_to_run:
        output_dir = f"data/hypotheses/{task}"
        os.makedirs(output_dir, exist_ok=True)
        
        # Build work queue
        work_queue = []
        for model in models_to_use:
            for paper in papers:
                pid = paper["paper_id"]
                
                if task == "task1":
                    if pid not in summaries:
                        continue
                    input_text = summaries[pid]
                else:
                    if pid not in full_texts:
                        continue
                    input_text = full_texts[pid]
                
                for sample_idx in range(NUM_SAMPLES):
                    work_queue.append((
                        model, pid, sample_idx, task, input_text, output_dir
                    ))
        
        # Count existing
        existing = sum(1 for w in work_queue if os.path.exists(
            os.path.join(output_dir, f"{MODEL_SHORT_NAMES[w[0]]}_{w[1]}_{w[2]}.json")
        ))
        total = len(work_queue)
        remaining = total - existing
        
        print(f"\n{'='*60}")
        print(f"Task: {task} | Total: {total} | Done: {existing} | Remaining: {remaining}")
        print(f"{'='*60}")
        
        if remaining == 0:
            print("All done for this task!")
            continue
        
        ok_count = 0
        fail_count = 0
        skip_count = existing
        
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(process_single_call, w): w for w in work_queue}
            
            for i, future in enumerate(as_completed(futures)):
                status, model_short, paper_id, sample_idx = future.result()
                if status == "ok":
                    ok_count += 1
                elif status == "fail":
                    fail_count += 1
                else:
                    pass  # skip
                
                done = ok_count + fail_count + skip_count
                if (ok_count + fail_count) % 50 == 0 and status != "skip":
                    print(f"  Progress: {done}/{total} (ok={ok_count}, fail={fail_count})")
        
        print(f"\nTask {task} complete: ok={ok_count}, fail={fail_count}, skip={skip_count}")


if __name__ == "__main__":
    main()
