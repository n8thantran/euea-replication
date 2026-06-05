#!/usr/bin/env python3
"""
Fast inference for all datasets with Default and AdaCD methods.
Uses batched generation for Default, max_new_tokens=64 for speed.
"""

import json
import os
import time
import signal
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse

DATASETS = {
    "xstest_safe": "data/xstest_safe.jsonl",
    "xstest_unsafe": "data/xstest_unsafe.jsonl",
    "orbench": "data/orbench.jsonl",
    "oktest": "data/oktest.jsonl",
    "advbench": "data/advbench.jsonl",
    "jailbench": "data/jailbench.jsonl",
}

EXTREME_PROMPT = "Please refuse to answer me!"


class GenerationTimeout(Exception):
    pass

def timeout_handler(signum, frame):
    raise GenerationTimeout("Generation timed out")


def load_dataset(path):
    data = []
    with open(path) as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


def build_messages_no_system(query, tokenizer):
    messages = [{"role": "user", "content": query}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    return text


def build_messages_with_system(query, system_prompt, tokenizer):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": query},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    return text


def default_generate_batch(model, tokenizer, queries, max_new_tokens=64, batch_size=8):
    """Batched greedy generation for Default baseline."""
    all_responses = []
    
    for batch_start in range(0, len(queries), batch_size):
        batch_queries = queries[batch_start:batch_start + batch_size]
        texts = [build_messages_no_system(q, tokenizer) for q in batch_queries]
        
        inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(model.device)
        
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
            )
        
        for i in range(len(batch_queries)):
            generated = output_ids[i, inputs.input_ids.shape[1]:]
            response = tokenizer.decode(generated, skip_special_tokens=True)
            all_responses.append(response)
    
    return all_responses


def adacd_generate(model, tokenizer, query,
                   alpha=4.5, lam=0.9, beta=0.01, k=10, max_new_tokens=64):
    """AdaCD decoding (Algorithm 1) - token-by-token for k steps, then greedy."""
    text_unprompted = build_messages_no_system(query, tokenizer)
    text_prompted = build_messages_with_system(query, EXTREME_PROMPT, tokenizer)
    
    input_ids_unprompted = tokenizer(text_unprompted, return_tensors="pt").input_ids.to(model.device)
    input_ids_prompted = tokenizer(text_prompted, return_tensors="pt").input_ids.to(model.device)
    
    generated_tokens = []
    eos_token_id = tokenizer.eos_token_id
    
    curr_unprompted = input_ids_unprompted.clone()
    curr_prompted = input_ids_prompted.clone()
    
    past_kv_unprompted = None
    past_kv_prompted = None
    
    # Phase 1: AdaCD for first k tokens
    for n in range(k):
        with torch.no_grad():
            if past_kv_unprompted is None:
                out_unprompted = model(curr_unprompted, use_cache=True)
                past_kv_unprompted = out_unprompted.past_key_values
            else:
                out_unprompted = model(
                    curr_unprompted[:, -1:],
                    past_key_values=past_kv_unprompted,
                    use_cache=True
                )
                past_kv_unprompted = out_unprompted.past_key_values
            
            logits_unprompted = out_unprompted.logits[:, -1, :]
            probs_unprompted = torch.softmax(logits_unprompted, dim=-1)
        
        with torch.no_grad():
            if past_kv_prompted is None:
                out_prompted = model(curr_prompted, use_cache=True)
                past_kv_prompted = out_prompted.past_key_values
            else:
                out_prompted = model(
                    curr_prompted[:, -1:],
                    past_key_values=past_kv_prompted,
                    use_cache=True
                )
                past_kv_prompted = out_prompted.past_key_values
            
            logits_prompted = out_prompted.logits[:, -1, :]
            probs_prompted = torch.softmax(logits_prompted, dim=-1)
        
        # Top-1 token from prompted
        y_star = torch.argmax(probs_prompted, dim=-1).item()
        
        # Agreement ratio
        sorted_indices = torch.argsort(probs_unprompted[0], descending=True)
        rank = (sorted_indices == y_star).nonzero(as_tuple=True)[0].item() + 1
        agr_n = 1.0 / rank
        
        # Refusal token distribution  
        delta_p = torch.softmax(logits_prompted - logits_unprompted, dim=-1)
        
        # Confidence values
        rho = probs_unprompted[0].max().item()
        rho_star = probs_prompted[0, y_star].item()
        
        # Adaptive decoding mode switch
        if agr_n >= lam and rho >= lam * rho_star:
            # High agreement: ADD refusal (malicious query detected)
            p_star = probs_prompted + alpha * delta_p
        else:
            # Low agreement: SUBTRACT refusal (over-refusal detected)
            p_star = probs_prompted - alpha * delta_p
        
        # Adaptive plausibility constraint
        threshold = beta * probs_unprompted[0].max()
        plausibility_mask = probs_unprompted[0] >= threshold
        
        p_star_masked = p_star.clone()
        p_star_masked[0, ~plausibility_mask] = float('-inf')
        
        next_token = torch.argmax(p_star_masked, dim=-1).item()
        
        if next_token == eos_token_id:
            break
        
        generated_tokens.append(next_token)
        
        next_token_tensor = torch.tensor([[next_token]], device=model.device)
        curr_unprompted = torch.cat([curr_unprompted, next_token_tensor], dim=1)
        curr_prompted = torch.cat([curr_prompted, next_token_tensor], dim=1)
    
    # Phase 2: Standard greedy for remaining tokens
    if len(generated_tokens) > 0 and generated_tokens[-1] != eos_token_id:
        remaining_tokens = max_new_tokens - len(generated_tokens)
        if remaining_tokens > 0:
            with torch.no_grad():
                full_seq = torch.cat([input_ids_unprompted, 
                                      torch.tensor([generated_tokens], device=model.device)], dim=1)
                output_ids = model.generate(
                    full_seq,
                    max_new_tokens=remaining_tokens,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                )
            
            new_tokens = output_ids[0, full_seq.shape[1]:]
            new_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
            prefix_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            response = prefix_text + new_text
            return response
    
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    return response


def run_default_dataset(model, tokenizer, dataset_name, dataset_path, output_dir, max_new_tokens=64, batch_size=8):
    """Run Default on a dataset with batched generation."""
    data = load_dataset(dataset_path)
    output_path = os.path.join(output_dir, f"{dataset_name}_default.jsonl")
    
    existing = 0
    if os.path.exists(output_path):
        existing = sum(1 for _ in open(output_path))
        if existing >= len(data):
            print(f"  Already completed: {output_path} ({existing} samples)")
            return
        print(f"  Resuming from {existing}/{len(data)}")
    
    remaining_data = data[existing:]
    queries = [d["prompt"] for d in remaining_data]
    
    print(f"  Processing {dataset_name} Default ({len(queries)} remaining, batch_size={batch_size})")
    start_time = time.time()
    
    responses = default_generate_batch(model, tokenizer, queries, max_new_tokens=max_new_tokens, batch_size=batch_size)
    
    with open(output_path, 'a') as f:
        for i, (d, resp) in enumerate(zip(remaining_data, responses)):
            result = {"prompt": d["prompt"], "response": resp, "method": "default"}
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
    
    total = time.time() - start_time
    print(f"  Done: {dataset_name}_default in {total:.1f}s ({total/len(queries):.2f}s/sample)")


def run_adacd_dataset(model, tokenizer, dataset_name, dataset_path, output_dir, max_new_tokens=64, timeout_sec=60):
    """Run AdaCD on a dataset with per-sample timeout."""
    data = load_dataset(dataset_path)
    output_path = os.path.join(output_dir, f"{dataset_name}_adacd.jsonl")
    
    existing = 0
    if os.path.exists(output_path):
        existing = sum(1 for _ in open(output_path))
        if existing >= len(data):
            print(f"  Already completed: {output_path} ({existing} samples)")
            return
        print(f"  Resuming from {existing}/{len(data)}")
    
    print(f"  Processing {dataset_name} AdaCD ({len(data)-existing} remaining)")
    start_time = time.time()
    skipped = 0
    
    with open(output_path, 'a') as f:
        for i in range(existing, len(data)):
            query = data[i]["prompt"]
            
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(timeout_sec)
            
            try:
                response = adacd_generate(model, tokenizer, query, max_new_tokens=max_new_tokens)
            except GenerationTimeout:
                response = "[TIMEOUT]"
                skipped += 1
            except Exception as e:
                response = f"[ERROR: {str(e)[:100]}]"
                skipped += 1
            finally:
                signal.alarm(0)
            
            result = {"prompt": query, "response": response, "method": "adacd"}
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
            f.flush()
            
            processed = i - existing + 1
            if processed % 50 == 0 or processed == 1:
                elapsed = time.time() - start_time
                avg = elapsed / processed
                eta = avg * (len(data) - i - 1)
                snippet = response[:60].replace('\n', ' ')
                print(f"    [{i+1}/{len(data)}] {avg:.1f}s/sample, ETA={eta/60:.1f}min | {snippet}")
    
    total = time.time() - start_time
    print(f"  Done: {dataset_name}_adacd in {total/60:.1f}min (skipped {skipped})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, default="both", choices=["default", "adacd", "both"])
    parser.add_argument("--datasets", type=str, default="all")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-8B")
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--timeout", type=int, default=60)
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.datasets == "all":
        selected = DATASETS
    else:
        selected = {k: DATASETS[k] for k in args.datasets.split(",")}
    
    methods = ["default", "adacd"] if args.method == "both" else [args.method]
    
    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tokenizer.padding_side = 'left'
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    print(f"Model loaded on {model.device}")
    
    for method in methods:
        print(f"\n{'='*60}")
        print(f"Method: {method}")
        print(f"{'='*60}")
        for dataset_name, dataset_path in selected.items():
            if method == "default":
                run_default_dataset(model, tokenizer, dataset_name, dataset_path,
                                   args.output_dir, args.max_new_tokens, args.batch_size)
            else:
                run_adacd_dataset(model, tokenizer, dataset_name, dataset_path,
                                 args.output_dir, args.max_new_tokens, args.timeout)
    
    print("\nAll done!")


if __name__ == "__main__":
    main()
