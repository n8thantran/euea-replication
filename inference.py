#!/usr/bin/env python3
"""
AdaCD Inference Script
Implements both Default (greedy) and AdaCD decoding for Qwen3-8B
Based on: "Please refuse to answer me! Mitigating Over-Refusal in LLMs via Adaptive Contrastive Decoding"
"""

import argparse
import json
import os
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_dataset(path):
    """Load JSONL dataset, return list of dicts with 'prompt' field."""
    data = []
    with open(path) as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


def build_messages_no_system(query, tokenizer):
    """Build input for Qwen3 WITHOUT system prompt."""
    messages = [{"role": "user", "content": query}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    return text


def build_messages_with_system(query, system_prompt, tokenizer):
    """Build input for Qwen3 WITH system prompt."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": query},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    return text


def default_generate(model, tokenizer, query, max_new_tokens=512):
    """Standard greedy decoding (Default baseline)."""
    text = build_messages_no_system(query, tokenizer)
    input_ids = tokenizer(text, return_tensors="pt").input_ids.to(model.device)
    
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # greedy
            temperature=None,
            top_p=None,
        )
    
    # Decode only the generated part
    generated_ids = output_ids[0, input_ids.shape[1]:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return response


def adacd_generate(model, tokenizer, query, 
                   system_prompt="Please refuse to answer me!",
                   alpha=4.5, lam=0.9, beta=0.01, k=10, max_new_tokens=512):
    """
    AdaCD decoding (Algorithm 1 from the paper).
    
    For first k tokens: adaptive contrastive decoding
    For remaining tokens: standard greedy decoding
    """
    # Prepare inputs for both prompted and unprompted
    text_unprompted = build_messages_no_system(query, tokenizer)
    text_prompted = build_messages_with_system(query, system_prompt, tokenizer)
    
    input_ids_unprompted = tokenizer(text_unprompted, return_tensors="pt").input_ids.to(model.device)
    input_ids_prompted = tokenizer(text_prompted, return_tensors="pt").input_ids.to(model.device)
    
    # We'll do token-by-token generation
    # For the first k tokens, we need both forward passes
    # After k tokens, we just do greedy from unprompted
    
    generated_tokens = []
    eos_token_id = tokenizer.eos_token_id
    
    # Current sequences (will grow as we generate)
    curr_unprompted = input_ids_unprompted.clone()
    curr_prompted = input_ids_prompted.clone()
    
    # Use KV cache for efficiency
    past_kv_unprompted = None
    past_kv_prompted = None
    
    for n in range(max_new_tokens):
        # Forward pass unprompted
        with torch.no_grad():
            if past_kv_unprompted is None:
                out_unprompted = model(curr_unprompted, use_cache=True)
                past_kv_unprompted = out_unprompted.past_key_values
            else:
                # Only pass the last token
                out_unprompted = model(
                    curr_unprompted[:, -1:],
                    past_key_values=past_kv_unprompted,
                    use_cache=True
                )
                past_kv_unprompted = out_unprompted.past_key_values
            
            logits_unprompted = out_unprompted.logits[:, -1, :]  # [1, vocab_size]
            probs_unprompted = torch.softmax(logits_unprompted, dim=-1)
        
        if n < k:
            # AdaCD contrastive decoding
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
                
                logits_prompted = out_prompted.logits[:, -1, :]  # [1, vocab_size]
                probs_prompted = torch.softmax(logits_prompted, dim=-1)
            
            # Step 1: Get top-1 token from prompted distribution
            y_star = torch.argmax(probs_prompted, dim=-1).item()
            
            # Step 2: Compute agreement ratio
            # Sort unprompted probs descending, find rank of y_star
            sorted_indices = torch.argsort(probs_unprompted[0], descending=True)
            rank = (sorted_indices == y_star).nonzero(as_tuple=True)[0].item() + 1  # 1-indexed
            agr_n = 1.0 / rank
            
            # Step 3: Compute refusal token distribution
            # ΔP_n = softmax(logits_prompted - logits_unprompted)
            delta_p = torch.softmax(logits_prompted - logits_unprompted, dim=-1)
            
            # Step 4: Compute ρ and ρ*
            rho = probs_unprompted[0].max().item()
            rho_star = probs_prompted[0, y_star].item()
            
            # Step 5: Adaptive decoding mode switch
            if agr_n >= lam and rho >= lam * rho_star:
                # High agreement + high confidence: ADD refusal (malicious query)
                p_star = probs_prompted + alpha * delta_p
            else:
                # Low agreement: SUBTRACT refusal (over-refusal query)
                p_star = probs_prompted - alpha * delta_p
            
            # Step 6: Adaptive plausibility constraint
            # W = {y | P_unprompted(y) >= beta * max(P_unprompted)}
            threshold = beta * probs_unprompted[0].max()
            plausibility_mask = probs_unprompted[0] >= threshold
            
            # Apply mask: set non-plausible tokens to -inf
            p_star_masked = p_star.clone()
            p_star_masked[0, ~plausibility_mask] = float('-inf')
            
            # Select token
            next_token = torch.argmax(p_star_masked, dim=-1).item()
        else:
            # Standard greedy decoding (fallback after k tokens)
            next_token = torch.argmax(probs_unprompted, dim=-1).item()
        
        # Check for EOS
        if next_token == eos_token_id:
            break
        
        generated_tokens.append(next_token)
        
        # Update sequences for next iteration
        next_token_tensor = torch.tensor([[next_token]], device=model.device)
        curr_unprompted = torch.cat([curr_unprompted, next_token_tensor], dim=1)
        curr_prompted = torch.cat([curr_prompted, next_token_tensor], dim=1)
    
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    return response


def main():
    parser = argparse.ArgumentParser(description="AdaCD Inference")
    parser.add_argument("--method", type=str, required=True, choices=["default", "adacd"],
                        help="Decoding method")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Path to dataset JSONL file")
    parser.add_argument("--output", type=str, required=True,
                        help="Path to output JSONL file")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-8B",
                        help="Model name or path")
    parser.add_argument("--alpha", type=float, default=4.5)
    parser.add_argument("--lam", type=float, default=0.9)
    parser.add_argument("--beta", type=float, default=0.01)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Max samples to process (for testing)")
    args = parser.parse_args()
    
    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    print(f"Model loaded on {model.device}")
    
    # Load dataset
    data = load_dataset(args.dataset)
    if args.max_samples:
        data = data[:args.max_samples]
    
    print(f"Processing {len(data)} samples with method={args.method}")
    
    # Create output directory
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)
    
    results = []
    start_time = time.time()
    
    for i, sample in enumerate(data):
        query = sample["prompt"]
        
        if args.method == "default":
            response = default_generate(model, tokenizer, query, 
                                       max_new_tokens=args.max_new_tokens)
        elif args.method == "adacd":
            response = adacd_generate(model, tokenizer, query,
                                     alpha=args.alpha, lam=args.lam, 
                                     beta=args.beta, k=args.k,
                                     max_new_tokens=args.max_new_tokens)
        
        result = {
            "prompt": query,
            "response": response,
            "method": args.method,
        }
        results.append(result)
        
        elapsed = time.time() - start_time
        avg_time = elapsed / (i + 1)
        remaining = avg_time * (len(data) - i - 1)
        
        if (i + 1) % 10 == 0 or i == 0:
            print(f"[{i+1}/{len(data)}] avg={avg_time:.1f}s/sample, ETA={remaining/60:.1f}min")
            # Print snippet of last response
            snippet = response[:100].replace('\n', ' ')
            print(f"  Response: {snippet}...")
    
    # Save results
    with open(args.output, 'w') as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    
    total_time = time.time() - start_time
    print(f"\nDone! {len(results)} samples in {total_time/60:.1f} minutes")
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
