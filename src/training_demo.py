"""
Training Demo: Demonstrates the SFT and GRPO training pipelines
using a small language model (no VLM vision encoder needed).

This validates the training code works end-to-end with:
1. Data loading from generated training data
2. SFT fine-tuning loop
3. GRPO reward computation and policy update
4. Metric logging

Uses a small model (GPT-2 or similar) to run quickly on a single GPU.
"""

import os
import sys
import json
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, get_linear_schedule_with_warmup
import numpy as np

sys.path.insert(0, '/workspace')


class SkillDataset(Dataset):
    """Dataset for skill training data. Handles both flat and conversation formats."""
    
    def __init__(self, data_path: str, tokenizer, max_length: int = 256):
        with open(data_path, 'r') as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def _extract_prompt_response(self, item):
        """Extract prompt and response from either flat or conversation format."""
        # Conversation format (from our data_generator)
        if "conversations" in item:
            convs = item["conversations"]
            prompt = ""
            response = ""
            for turn in convs:
                if turn["role"] == "user":
                    prompt = turn["content"].replace("<image>\n", "")
                elif turn["role"] == "assistant":
                    response = turn["content"]
            return prompt, response
        
        # Flat format
        prompt = item.get("prompt", item.get("instruction", ""))
        response = item.get("response", item.get("output", ""))
        return prompt, response
    
    def __getitem__(self, idx):
        item = self.data[idx]
        prompt, response = self._extract_prompt_response(item)
        
        # Combine for causal LM training
        text = f"### Instruction:\n{prompt}\n\n### Response:\n{response}"
        
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)
        
        # Labels = input_ids (shifted internally by model)
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100  # Ignore padding
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def run_sft_demo(
    model_name: str = "gpt2",
    data_path: str = "/workspace/data/train/sft_data.json",
    output_dir: str = "/workspace/results/sft_demo",
    num_steps: int = 50,
    batch_size: int = 4,
    learning_rate: float = 5e-5,
    max_length: int = 256,
):
    """Run a small SFT training demo."""
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print("SFT Training Demo")
    print("=" * 60)
    print(f"Model: {model_name}")
    print(f"Data: {data_path}")
    print(f"Steps: {num_steps}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    
    # Load tokenizer and model
    print("\nLoading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"Model loaded on {device}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Load dataset
    print("\nLoading dataset...")
    dataset = SkillDataset(data_path, tokenizer, max_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    print(f"Dataset size: {len(dataset)}")
    
    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=10, num_training_steps=num_steps
    )
    
    # Training loop
    print("\nStarting training...")
    model.train()
    losses = []
    step = 0
    start_time = time.time()
    
    data_iter = iter(dataloader)
    
    while step < num_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)
        
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        
        loss = outputs.loss
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        
        losses.append(loss.item())
        step += 1
        
        if step % 10 == 0 or step == 1:
            elapsed = time.time() - start_time
            avg_loss = np.mean(losses[-10:])
            print(f"  Step {step}/{num_steps} | Loss: {avg_loss:.4f} | Time: {elapsed:.1f}s")
    
    total_time = time.time() - start_time
    final_loss = np.mean(losses[-10:])
    
    print(f"\nTraining complete!")
    print(f"  Final loss: {final_loss:.4f}")
    print(f"  Total time: {total_time:.1f}s")
    print(f"  Steps/sec: {num_steps/total_time:.2f}")
    
    # Save results
    results = {
        "model": model_name,
        "num_steps": num_steps,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "final_loss": float(final_loss),
        "total_time": total_time,
        "losses": [float(l) for l in losses],
    }
    
    with open(os.path.join(output_dir, "sft_demo_results.json"), 'w') as f:
        json.dump(results, f, indent=2)
    
    # Generate loss curve
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(1, len(losses)+1), losses, color='#4472C4', alpha=0.6, linewidth=1)
    # Smoothed
    window = min(10, len(losses))
    smoothed = np.convolve(losses, np.ones(window)/window, mode='valid')
    ax.plot(range(window, len(losses)+1), smoothed, color='#4472C4', linewidth=2, label='Smoothed')
    ax.set_xlabel('Step', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title(f'SFT Demo Training Loss ({model_name})', fontsize=13, fontweight='bold')
    ax.legend()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "sft_demo_loss.png"), dpi=150)
    plt.close()
    
    print(f"Results saved to {output_dir}")
    return results


def run_grpo_demo(
    model_name: str = "gpt2",
    data_path: str = "/workspace/data/train/sft_data.json",
    output_dir: str = "/workspace/results/grpo_demo",
    num_steps: int = 30,
    batch_size: int = 4,
    num_samples: int = 4,  # G in GRPO
    learning_rate: float = 1e-5,
    max_length: int = 256,
    kl_coeff: float = 0.01,
):
    """
    Run a GRPO training demo.
    
    GRPO (Group Relative Policy Optimization):
    1. For each prompt, generate G samples
    2. Compute reward for each sample
    3. Normalize rewards within group (relative advantage)
    4. Update policy to increase probability of high-reward samples
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print("GRPO Training Demo")
    print("=" * 60)
    print(f"Model: {model_name}")
    print(f"Steps: {num_steps}")
    print(f"Samples per prompt (G): {num_samples}")
    print(f"KL coefficient: {kl_coeff}")
    
    # Load model
    print("\nLoading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(model_name)
    ref_model = AutoModelForCausalLM.from_pretrained(model_name)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    ref_model = ref_model.to(device)
    ref_model.eval()
    
    # Load data and extract prompt/response pairs  
    with open(data_path, 'r') as f:
        raw_data = json.load(f)
    
    # Convert conversation format to flat format
    data = []
    for item in raw_data:
        if "conversations" in item:
            prompt = ""
            response = ""
            for turn in item["conversations"]:
                if turn["role"] == "user":
                    prompt = turn["content"].replace("<image>\n", "")
                elif turn["role"] == "assistant":
                    response = turn["content"]
            data.append({"prompt": prompt, "response": response})
        else:
            data.append(item)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    print("\nStarting GRPO training...")
    rewards_history = []
    losses_history = []
    step = 0
    start_time = time.time()
    
    while step < num_steps:
        # Sample a batch of prompts
        batch_items = [data[i % len(data)] for i in range(step * batch_size, (step + 1) * batch_size)]
        
        batch_rewards = []
        batch_loss = 0.0
        
        for item in batch_items:
            prompt = item.get("prompt", "")
            gt_response = item.get("response", "")
            
            prompt_enc = tokenizer(
                f"### Instruction:\n{prompt}\n\n### Response:\n",
                max_length=max_length // 2,
                truncation=True,
                return_tensors="pt",
            ).to(device)
            
            # Generate G samples
            model.eval()
            with torch.no_grad():
                sample_outputs = model.generate(
                    **prompt_enc,
                    max_new_tokens=64,
                    do_sample=True,
                    temperature=0.8,
                    top_p=0.9,
                    num_return_sequences=num_samples,
                    pad_token_id=tokenizer.pad_token_id,
                )
            
            # Compute rewards for each sample
            sample_rewards = []
            for sample in sample_outputs:
                decoded = tokenizer.decode(sample[prompt_enc["input_ids"].shape[1]:], skip_special_tokens=True)
                
                # Simple reward: overlap with ground truth
                gt_tokens = set(gt_response.lower().split())
                gen_tokens = set(decoded.lower().split())
                
                if len(gt_tokens) > 0:
                    overlap = len(gt_tokens & gen_tokens) / len(gt_tokens)
                else:
                    overlap = 0.0
                
                reward = overlap
                sample_rewards.append(reward)
            
            # Normalize rewards within group (GRPO key idea)
            rewards_array = np.array(sample_rewards)
            if rewards_array.std() > 0:
                normalized_rewards = (rewards_array - rewards_array.mean()) / (rewards_array.std() + 1e-8)
            else:
                normalized_rewards = np.zeros_like(rewards_array)
            
            batch_rewards.extend(sample_rewards)
            
            # Compute policy gradient loss for best samples
            model.train()
            best_idx = np.argmax(sample_rewards)
            best_sample = sample_outputs[best_idx]
            
            # Forward pass on best sample
            outputs = model(
                input_ids=best_sample.unsqueeze(0),
                labels=best_sample.unsqueeze(0),
            )
            
            # Weighted by normalized reward
            weight = max(normalized_rewards[best_idx], 0.1)
            batch_loss += outputs.loss * weight
        
        # Average loss
        batch_loss = batch_loss / batch_size
        
        # KL penalty (approximate)
        with torch.no_grad():
            ref_outputs = ref_model(
                input_ids=best_sample.unsqueeze(0),
                labels=best_sample.unsqueeze(0),
            )
            kl_penalty = kl_coeff * (outputs.loss - ref_outputs.loss).abs()
        
        total_loss = batch_loss + kl_penalty
        
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()
        
        avg_reward = np.mean(batch_rewards)
        rewards_history.append(avg_reward)
        losses_history.append(total_loss.item())
        
        step += 1
        
        if step % 5 == 0 or step == 1:
            elapsed = time.time() - start_time
            print(f"  Step {step}/{num_steps} | Loss: {total_loss.item():.4f} | "
                  f"Avg Reward: {avg_reward:.4f} | Time: {elapsed:.1f}s")
    
    total_time = time.time() - start_time
    
    print(f"\nGRPO training complete!")
    print(f"  Final avg reward: {np.mean(rewards_history[-5:]):.4f}")
    print(f"  Total time: {total_time:.1f}s")
    
    # Save results
    results = {
        "model": model_name,
        "num_steps": num_steps,
        "num_samples_G": num_samples,
        "kl_coeff": kl_coeff,
        "final_avg_reward": float(np.mean(rewards_history[-5:])),
        "total_time": total_time,
        "rewards": [float(r) for r in rewards_history],
        "losses": [float(l) for l in losses_history],
    }
    
    with open(os.path.join(output_dir, "grpo_demo_results.json"), 'w') as f:
        json.dump(results, f, indent=2)
    
    # Plot
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1.plot(range(1, len(losses_history)+1), losses_history, color='#E74C3C', linewidth=1.5)
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Loss')
    ax1.set_title('GRPO Demo Loss')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.grid(alpha=0.3)
    
    ax2.plot(range(1, len(rewards_history)+1), rewards_history, color='#27AE60', linewidth=1.5)
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Avg Reward')
    ax2.set_title('GRPO Demo Reward')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "grpo_demo_curves.png"), dpi=150)
    plt.close()
    
    print(f"Results saved to {output_dir}")
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["sft", "grpo", "both"], default="both")
    parser.add_argument("--model", default="gpt2")
    parser.add_argument("--sft_steps", type=int, default=50)
    parser.add_argument("--grpo_steps", type=int, default=30)
    args = parser.parse_args()
    
    if args.mode in ["sft", "both"]:
        run_sft_demo(model_name=args.model, num_steps=args.sft_steps)
    
    if args.mode in ["grpo", "both"]:
        run_grpo_demo(model_name=args.model, num_steps=args.grpo_steps)
