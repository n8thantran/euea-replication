#!/usr/bin/env python3
"""
Run inference for all datasets with both Default and AdaCD methods.
Processes everything in one model loading pass for efficiency.
"""

import json
import os
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


DATASETS = {
    "xstest_safe": "data/xstest_safe.jsonl",
    "xstest_unsafe": "data/xstest_unsafe.jsonl",
    "orbench": "data/orbench.jsonl",
    "oktest": "data/oktest.jsonl",
    "advbench": "data/advbench.jsonl",
    "jailbench": "data/jailbench.jsonl",
}

EXTREME_PROMPT = "Please refuse to answer me!"


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


def default_generate(model, tokenizer, query, max_new_tokens=512):
    """Standard greedy decoding."""
    text = build_messages_no_system(query, tokenizer)
    input_ids = tokenizer(text, return_tensors="pt").input_ids.to(model.device)
    
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
        )
    
    generated_ids = output_ids[0, input_ids.shape[1]:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return response


def adacd_generate(model, tokenizer, query,
                   alpha=4.5, lam=0.9, beta=0.01, k=10, max_new_tokens=512):
    """AdaCD decoding (Algorithm 1)."""
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
    
    for n in range(max_new_tokens):
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
        
        if n < k:
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
                p_star = probs_prompted + alpha * delta_p
            else:
                p_star = probs_prompted - alpha * delta_p
            
            # Adaptive plausibility constraint
            threshold = beta * probs_unprompted[0].max()
            plausibility_mask = probs_unprompted[0] >= threshold
            
            p_star_masked = p_star.clone()
            p_star_masked[0, ~plausibility_mask] = float('-inf')
            
            next_token = torch.argmax(p_star_masked, dim=-1).item()
        else:
            next_token = torch.argmax(probs_unprompted, dim=-1).item()
        
        if next_token == eos_token_id:
            break
        
        generated_tokens.append(next_token)
        
        next_token_tensor = torch.tensor([[next_token]], device=model.device)
        curr_unprompted = torch.cat([curr_unprompted, next_token_tensor], dim=1)
        curr_prompted = torch.cat([curr_prompted, next_token_tensor], dim=1)
    
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    return response


def run_dataset(model, tokenizer, dataset_name, dataset_path, method, output_dir, max_new_tokens=512):
    """Run inference on a single dataset."""
    data = load_dataset(dataset_path)
    output_path = os.path.join(output_dir, f"{dataset_name}_{method}.jsonl")
    
    # Check if already done
    if os.path.exists(output_path):
        existing = sum(1 for _ in open(output_path))
        if existing >= len(data):
            print(f"  Already completed: {output_path} ({existing} samples)")
            return
        else:
            print(f"  Resuming from {existing}/{len(data)}")
    else:
        existing = 0
    
    print(f"  Processing {dataset_name} with {method} ({len(data)} samples)")
    
    start_time = time.time()
    
    # Open in append mode for resume
    with open(output_path, 'a') as f:
        for i in range(existing, len(data)):
            query = data[i]["prompt"]
            
            if method == "default":
                response = default_generate(model, tokenizer, query, max_new_tokens=max_new_tokens)
            else:
                response = adacd_generate(model, tokenizer, query, max_new_tokens=max_new_tokens)
            
            result = {"prompt": query, "response": response, "method": method}
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
            f.flush()
            
            elapsed = time.time() - start_time
            processed = i - existing + 1
            avg_time = elapsed / processed
            remaining = avg_time * (len(data) - i - 1)
            
            if processed % 20 == 0 or processed == 1:
                snippet = response[:80].replace('\n', ' ')
                print(f"    [{i+1}/{len(data)}] {avg_time:.1f}s/sample, ETA={remaining/60:.1f}min | {snippet}")
    
    total = time.time() - start_time
    print(f"  Done: {dataset_name}_{method} in {total/60:.1f} minutes")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, default="both", choices=["default", "adacd", "both"])
    parser.add_argument("--datasets", type=str, default="all",
                        help="Comma-separated dataset names, or 'all'")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-8B")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Select datasets
    if args.datasets == "all":
        selected = DATASETS
    else:
        selected = {k: DATASETS[k] for k in args.datasets.split(",")}
    
    # Select methods
    methods = ["default", "adacd"] if args.method == "both" else [args.method]
    
    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
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
            run_dataset(model, tokenizer, dataset_name, dataset_path, method, 
                       args.output_dir, args.max_new_tokens)
    
    print("\nAll done!")


if __name__ == "__main__":
    main()
