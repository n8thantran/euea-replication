"""
GRASPrune: Global Gating for Budgeted Structured Pruning of Large Language Models

Main implementation file.
"""

import os
import sys
import math
import json
import argparse
import time
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from datasets import load_dataset

# ============================================================
# 1. Data Loading
# ============================================================

def load_calibration_data(
    tokenizer,
    dataset_name: str = "wikitext",
    dataset_config: str = "wikitext-2-raw-v1",
    split: str = "train",
    n_samples: int = 512,
    seq_len: int = 512,
    seed: int = 42,
):
    """Load calibration data: n_samples sequences of seq_len tokens from a dataset."""
    if dataset_name == "wikitext":
        dataset = load_dataset("Salesforce/wikitext", dataset_config, split=split, trust_remote_code=True)
        text_key = "text"
    elif dataset_name == "c4":
        dataset = load_dataset("allenai/c4", "en", split="train", streaming=True, trust_remote_code=True)
        text_key = "text"
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # Concatenate all text
    if dataset_name == "c4":
        texts = []
        for item in dataset:
            texts.append(item[text_key])
            if len(texts) >= n_samples * 2:
                break
        full_text = "\n\n".join(texts)
    else:
        full_text = "\n\n".join([item[text_key] for item in dataset if item[text_key].strip()])
    
    # Tokenize all text at once
    encodings = tokenizer(full_text, return_tensors="pt", truncation=False)
    input_ids = encodings.input_ids[0]
    
    # Sample random subsequences
    torch.manual_seed(seed)
    total_tokens = input_ids.shape[0]
    
    samples = []
    for _ in range(n_samples):
        start_idx = torch.randint(0, total_tokens - seq_len - 1, (1,)).item()
        sample = input_ids[start_idx : start_idx + seq_len]
        samples.append(sample)
    
    return torch.stack(samples)  # [n_samples, seq_len]


# ============================================================
# 2. Cost Model
# ============================================================

def compute_alpha(config) -> float:
    """Compute the KV head group cost relative to FFN channel cost.
    
    α = (2G + 2) * d_h / 3
    where G = H / H_kv, d_h = d / H
    """
    num_heads = config.num_attention_heads
    num_kv_heads = getattr(config, "num_key_value_heads", num_heads)
    hidden_size = config.hidden_size
    d_h = hidden_size // num_heads
    G = num_heads // num_kv_heads
    
    alpha = (2 * G + 2) * d_h / 3
    return alpha


def build_prunable_units(config) -> Tuple[List[Dict], torch.Tensor]:
    """Build the list of all prunable units and their costs.
    
    Returns:
        units: list of dicts with keys {layer, type, index, cost}
        costs: tensor of costs for each unit
    """
    num_layers = config.num_hidden_layers
    num_heads = config.num_attention_heads
    num_kv_heads = getattr(config, "num_key_value_heads", num_heads)
    intermediate_size = config.intermediate_size
    
    alpha = compute_alpha(config)
    
    units = []
    costs = []
    
    for layer_idx in range(num_layers):
        # FFN channels
        for ch_idx in range(intermediate_size):
            units.append({
                "layer": layer_idx,
                "type": "ffn",
                "index": ch_idx,
            })
            costs.append(1.0)
        
        # KV head groups
        for kv_idx in range(num_kv_heads):
            units.append({
                "layer": layer_idx,
                "type": "kv",
                "index": kv_idx,
            })
            costs.append(alpha)
    
    costs = torch.tensor(costs, dtype=torch.float32)
    return units, costs


# ============================================================
# 3. Budget-Feasible Projection
# ============================================================

def project_to_budget_fast(
    probs: torch.Tensor,
    costs: torch.Tensor,
    budget: float,
    num_layers: int,
    intermediate_size: int,
    num_kv_heads: int,
) -> torch.Tensor:
    """Vectorized budget projection. Sort by p_i descending, greedily select."""
    # Sort by probability descending
    sorted_indices = torch.argsort(probs, descending=True)
    sorted_costs = costs[sorted_indices]
    
    # Cumulative cost
    cumcost = torch.cumsum(sorted_costs, dim=0)
    
    # All units where cumulative cost <= budget
    within_budget = cumcost <= budget
    
    # Create mask
    mask = torch.zeros_like(probs)
    mask[sorted_indices[within_budget]] = 1.0
    
    # Degenerate layer protection
    units_per_layer = intermediate_size + num_kv_heads
    for layer_idx in range(num_layers):
        ffn_start = layer_idx * units_per_layer
        ffn_end = ffn_start + intermediate_size
        kv_start = ffn_end
        kv_end = kv_start + num_kv_heads
        
        # Check FFN
        if mask[ffn_start:ffn_end].sum() == 0:
            best = probs[ffn_start:ffn_end].argmax()
            mask[ffn_start + best] = 1.0
        
        # Check KV
        if mask[kv_start:kv_end].sum() == 0:
            best = probs[kv_start:kv_end].argmax()
            mask[kv_start + best] = 1.0
    
    return mask


# ============================================================
# 4. Gate Application to Model
# ============================================================

class GRASPruneGates(nn.Module):
    """Manages gate scores for all prunable units."""
    
    def __init__(self, config, target_ratio: float, alpha_scale: float = 1.0):
        super().__init__()
        self.config = config
        self.target_ratio = target_ratio
        
        num_layers = config.num_hidden_layers
        num_kv_heads = getattr(config, "num_key_value_heads", config.num_attention_heads)
        intermediate_size = config.intermediate_size
        
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.intermediate_size = intermediate_size
        self.units_per_layer = intermediate_size + num_kv_heads
        
        # Total prunable units
        total_units = num_layers * (intermediate_size + num_kv_heads)
        
        # Gate scores initialized to 0
        self.scores = nn.Parameter(torch.zeros(total_units))
        
        # Cost vector
        alpha = compute_alpha(config) * alpha_scale
        self.alpha = alpha
        costs = []
        for _ in range(num_layers):
            costs.extend([1.0] * intermediate_size)
            costs.extend([alpha] * num_kv_heads)
        self.register_buffer("costs", torch.tensor(costs, dtype=torch.float32))
        
        # Budget
        self.budget = target_ratio * self.costs.sum().item()
        
        # Temperature
        self.tau = 1.5
    
    def get_probs(self):
        """Convert scores to probabilities via sigmoid with temperature."""
        return torch.sigmoid(self.scores / self.tau)
    
    def get_mask(self):
        """Get budget-feasible hard mask."""
        probs = self.get_probs()
        mask = project_to_budget_fast(
            probs, self.costs, self.budget,
            self.num_layers, self.intermediate_size, self.num_kv_heads
        )
        return mask
    
    def get_ste_gates(self):
        """Get STE gates: hard in forward, soft gradient in backward.
        
        z_tilde = m + (p - stopgrad(p))
        """
        probs = self.get_probs()
        mask = project_to_budget_fast(
            probs.detach(), self.costs, self.budget,
            self.num_layers, self.intermediate_size, self.num_kv_heads
        )
        # STE: forward uses mask, backward flows through probs
        z_tilde = mask.detach() + probs - probs.detach()
        return z_tilde
    
    def get_layer_gates(self, layer_idx: int, z_tilde: torch.Tensor):
        """Get FFN and KV gates for a specific layer."""
        start = layer_idx * self.units_per_layer
        ffn_gates = z_tilde[start : start + self.intermediate_size]
        kv_gates = z_tilde[start + self.intermediate_size : start + self.units_per_layer]
        return ffn_gates, kv_gates


# ============================================================
# 5. Hooked Forward Pass
# ============================================================

def apply_gates_to_model_forward(
    model,
    input_ids: torch.Tensor,
    gate_module: GRASPruneGates,
    z_tilde: torch.Tensor,
):
    """Run forward pass with gates applied to FFN channels and KV heads.
    
    FFN gating: gate applied to intermediate activation (input to down_proj).
    KV gating: gate applied to per-head attention output (input to o_proj).
    
    Each gate is applied exactly once to avoid double-gating issues.
    """
    config = model.config
    num_heads = config.num_attention_heads
    num_kv_heads = getattr(config, "num_key_value_heads", num_heads)
    head_dim = config.hidden_size // num_heads
    G = num_heads // num_kv_heads  # number of query heads per kv head
    
    hooks = []
    
    for layer_idx in range(config.num_hidden_layers):
        ffn_gates, kv_gates = gate_module.get_layer_gates(layer_idx, z_tilde)
        
        layer = model.model.layers[layer_idx]
        
        # FFN gating: pre-hook on down_proj to gate the intermediate activation
        # The intermediate activation is: silu(gate_proj(x)) * up_proj(x)
        # We gate this product before it enters down_proj
        def make_ffn_pre_hook(gates):
            def hook_fn(module, input):
                inp = input[0]  # [batch, seq_len, intermediate_size]
                gated = inp * gates.to(inp.dtype).unsqueeze(0).unsqueeze(0)
                return (gated,)
            return hook_fn
        
        h1 = layer.mlp.down_proj.register_forward_pre_hook(make_ffn_pre_hook(ffn_gates))
        hooks.append(h1)
        
        # KV gating: pre-hook on o_proj to gate per-head attention output
        def make_attn_pre_hook(gates, n_kv_heads, g, h_dim, n_heads):
            def hook_fn(module, input):
                inp = input[0]  # [batch, seq_len, num_heads * head_dim]
                batch, seq_len, _ = inp.shape
                inp = inp.view(batch, seq_len, n_heads, h_dim)
                # Expand KV gates to per-query-head gates
                q_gates = gates.repeat_interleave(g) if g > 1 else gates
                inp = inp * q_gates.to(inp.dtype).unsqueeze(0).unsqueeze(0).unsqueeze(-1)
                return (inp.view(batch, seq_len, n_heads * h_dim),)
            return hook_fn
        
        h2 = layer.self_attn.o_proj.register_forward_pre_hook(
            make_attn_pre_hook(kv_gates, num_kv_heads, G, head_dim, num_heads))
        hooks.append(h2)
    
    # Forward pass
    outputs = model(input_ids=input_ids, labels=input_ids)
    loss = outputs.loss
    
    # Remove hooks
    for h in hooks:
        h.remove()
    
    return loss


# ============================================================
# 6. Training Loop
# ============================================================

def train_gates(
    model,
    gate_module: GRASPruneGates,
    calibration_data: torch.Tensor,
    num_epochs: int = 4,
    lr: float = 1e-2,
    batch_size: int = 1,
):
    """Train gate scores with projected STE."""
    device = next(model.parameters()).device
    
    # Freeze model parameters
    for param in model.parameters():
        param.requires_grad = False
    
    gate_module = gate_module.to(device)
    
    # AdamW optimizer for gate scores only
    optimizer = torch.optim.AdamW(gate_module.parameters(), lr=lr, weight_decay=0.0)
    
    n_samples = calibration_data.shape[0]
    n_steps = 0
    
    for epoch in range(num_epochs):
        # Shuffle
        perm = torch.randperm(n_samples)
        epoch_loss = 0.0
        n_batches = 0
        
        for i in range(0, n_samples, batch_size):
            batch_indices = perm[i:i+batch_size]
            batch = calibration_data[batch_indices].to(device)
            
            # Get STE gates
            z_tilde = gate_module.get_ste_gates()
            
            # Forward pass with gates
            loss = apply_gates_to_model_forward(model, batch, gate_module, z_tilde)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            n_steps += 1
            n_batches += 1
            
            if n_steps % 50 == 0:
                # Get current mask stats
                with torch.no_grad():
                    mask = gate_module.get_mask()
                    total_kept = mask.sum().item()
                    total_units = mask.numel()
                    cost_kept = (mask * gate_module.costs.to(device)).sum().item()
                    cost_total = gate_module.costs.sum().item()
                print(f"  Step {n_steps}, Loss: {loss.item():.4f}, "
                      f"Units: {total_kept:.0f}/{total_units}, "
                      f"Cost ratio: {cost_kept/cost_total:.4f}")
        
        avg_loss = epoch_loss / max(n_batches, 1)
        print(f"Epoch {epoch+1}/{num_epochs}, Avg Loss: {avg_loss:.4f}")
    
    return gate_module


# ============================================================
# 7. Scaling Calibration
# ============================================================

def train_scaling(
    model,
    gate_module: GRASPruneGates,
    calibration_data: torch.Tensor,
    num_epochs: int = 2,
    lr: float = 1e-2,
    batch_size: int = 1,
):
    """Train scaling factors for retained units."""
    device = next(model.parameters()).device
    
    # Get final mask
    with torch.no_grad():
        mask = gate_module.get_mask().to(device)
    
    # Create scaling parameters for retained units
    scales = torch.ones(mask.shape[0], device=device, dtype=torch.float32)
    scales = nn.Parameter(scales)
    
    optimizer = torch.optim.AdamW([scales], lr=lr, weight_decay=0.0)
    
    n_samples = calibration_data.shape[0]
    
    for epoch in range(num_epochs):
        perm = torch.randperm(n_samples)
        epoch_loss = 0.0
        n_batches = 0
        
        for i in range(0, n_samples, batch_size):
            batch_indices = perm[i:i+batch_size]
            batch = calibration_data[batch_indices].to(device)
            
            # Apply mask * scales as the gate values
            z = mask.detach() * scales
            
            loss = apply_gates_to_model_forward(model, batch, gate_module, z)
            
            optimizer.zero_grad()
            loss.backward()
            
            # Zero gradients for pruned units
            with torch.no_grad():
                scales.grad *= mask
            
            optimizer.step()
            
            epoch_loss += loss.item()
            n_batches += 1
        
        avg_loss = epoch_loss / max(n_batches, 1)
        print(f"Scaling Epoch {epoch+1}/{num_epochs}, Avg Loss: {avg_loss:.4f}")
    
    # Return final scales (only for retained units)
    return (mask * scales).detach()


# ============================================================
# 8. Model Materialization
# ============================================================

def materialize_pruned_model(model, gate_module, scales, save_path):
    """Create a pruned dense model by slicing weights and folding scales.
    
    FFN scales are folded into down_proj columns (matching the pre-hook on down_proj).
    Attention scales are folded into o_proj columns (matching the pre-hook on o_proj).
    """
    config = model.config
    num_heads = config.num_attention_heads
    num_kv_heads = getattr(config, "num_key_value_heads", num_heads)
    head_dim = config.hidden_size // num_heads
    intermediate_size = config.intermediate_size
    G = num_heads // num_kv_heads
    
    mask = gate_module.get_mask()
    
    # For each layer, determine which FFN channels and KV heads to keep
    for layer_idx in range(config.num_hidden_layers):
        ffn_gates, kv_gates = gate_module.get_layer_gates(layer_idx, mask)
        ffn_scales_layer, kv_scales_layer = gate_module.get_layer_gates(layer_idx, scales)
        
        ffn_keep = ffn_gates.bool().cpu()
        kv_keep = kv_gates.bool().cpu()
        
        ffn_scale = ffn_scales_layer[ffn_keep].cpu().float()
        kv_scale = kv_scales_layer[kv_keep].cpu().float()
        
        layer = model.model.layers[layer_idx]
        
        # --- FFN ---
        # Gate was applied as pre-hook on down_proj: down_proj(scale * intermediate)
        # So fold scale into down_proj columns: down_proj[:, i] *= scale_i
        
        # gate_proj: [intermediate_size, hidden_size] -> select rows (no scale)
        gate_w = layer.mlp.gate_proj.weight.data.cpu().float()[ffn_keep]
        
        # up_proj: [intermediate_size, hidden_size] -> select rows (no scale)
        up_w = layer.mlp.up_proj.weight.data.cpu().float()[ffn_keep]
        
        # down_proj: [hidden_size, intermediate_size] -> select columns, fold scale
        down_w = layer.mlp.down_proj.weight.data.cpu().float()[:, ffn_keep]
        down_w = down_w * ffn_scale.unsqueeze(0)  # scale columns
        
        new_inter = ffn_keep.sum().item()
        layer.mlp.gate_proj = nn.Linear(config.hidden_size, new_inter, bias=False)
        layer.mlp.gate_proj.weight = nn.Parameter(gate_w.to(torch.bfloat16))
        
        layer.mlp.up_proj = nn.Linear(config.hidden_size, new_inter, bias=False)
        layer.mlp.up_proj.weight = nn.Parameter(up_w.to(torch.bfloat16))
        
        layer.mlp.down_proj = nn.Linear(new_inter, config.hidden_size, bias=False)
        layer.mlp.down_proj.weight = nn.Parameter(down_w.to(torch.bfloat16))
        
        # --- Attention KV heads ---
        # Gate was applied as pre-hook on o_proj: o_proj(scale * attn_output)
        # So fold scale into o_proj columns: o_proj[:, head_i*head_dim:(head_i+1)*head_dim] *= scale_i
        
        q_keep = kv_keep.repeat_interleave(G)  # [num_heads]
        q_scale = kv_scale.repeat_interleave(G)  # [new_num_heads]
        
        new_kv = kv_keep.sum().item()
        new_heads = new_kv * G
        
        # q_proj: [num_heads * head_dim, hidden_size] -> select rows (no scale)
        q_w = layer.self_attn.q_proj.weight.data.cpu().float()
        q_w = q_w.view(num_heads, head_dim, config.hidden_size)
        q_w = q_w[q_keep]
        q_w = q_w.reshape(new_heads * head_dim, config.hidden_size)
        
        # k_proj: [num_kv_heads * head_dim, hidden_size] -> select rows (no scale)
        k_w = layer.self_attn.k_proj.weight.data.cpu().float()
        k_w = k_w.view(num_kv_heads, head_dim, config.hidden_size)
        k_w = k_w[kv_keep]
        k_w = k_w.reshape(new_kv * head_dim, config.hidden_size)
        
        # v_proj: [num_kv_heads * head_dim, hidden_size] -> select rows (no scale)
        v_w = layer.self_attn.v_proj.weight.data.cpu().float()
        v_w = v_w.view(num_kv_heads, head_dim, config.hidden_size)
        v_w = v_w[kv_keep]
        v_w = v_w.reshape(new_kv * head_dim, config.hidden_size)
        
        # o_proj: [hidden_size, num_heads * head_dim] -> select columns, fold scale
        o_w = layer.self_attn.o_proj.weight.data.cpu().float()
        o_w = o_w.view(config.hidden_size, num_heads, head_dim)
        o_w = o_w[:, q_keep]  # [hidden_size, new_heads, head_dim]
        # Fold scale: multiply each head's columns by its scale
        o_w = o_w * q_scale.unsqueeze(0).unsqueeze(-1)
        o_w = o_w.reshape(config.hidden_size, new_heads * head_dim)
        
        # Recreate linear layers
        layer.self_attn.q_proj = nn.Linear(config.hidden_size, new_heads * head_dim, bias=False)
        layer.self_attn.q_proj.weight = nn.Parameter(q_w.to(torch.bfloat16))
        
        layer.self_attn.k_proj = nn.Linear(config.hidden_size, new_kv * head_dim, bias=False)
        layer.self_attn.k_proj.weight = nn.Parameter(k_w.to(torch.bfloat16))
        
        layer.self_attn.v_proj = nn.Linear(config.hidden_size, new_kv * head_dim, bias=False)
        layer.self_attn.v_proj.weight = nn.Parameter(v_w.to(torch.bfloat16))
        
        layer.self_attn.o_proj = nn.Linear(new_heads * head_dim, config.hidden_size, bias=False)
        layer.self_attn.o_proj.weight = nn.Parameter(o_w.to(torch.bfloat16))
        
        # Update attention config for this layer
        layer.self_attn.num_heads = new_heads
        layer.self_attn.num_key_value_heads = new_kv
        layer.self_attn.num_key_value_groups = G
        layer.self_attn.head_dim = head_dim
    
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        model.save_pretrained(save_path)
        print(f"Pruned model saved to {save_path}")
    
    return model


# ============================================================
# 9. Perplexity Evaluation
# ============================================================

@torch.no_grad()
def evaluate_perplexity(model, tokenizer, dataset_name="wikitext", dataset_config="wikitext-2-raw-v1", split="test", max_length=2048):
    """Evaluate perplexity on a dataset."""
    device = next(model.parameters()).device
    model.eval()
    
    if dataset_name == "wikitext":
        dataset = load_dataset("Salesforce/wikitext", dataset_config, split=split, trust_remote_code=True)
        text = "\n\n".join(dataset["text"])
    elif dataset_name == "c4":
        dataset = load_dataset("allenai/c4", "en", split="validation", streaming=True, trust_remote_code=True)
        texts = []
        for item in dataset:
            texts.append(item["text"])
            if len(texts) >= 256:
                break
        text = "\n\n".join(texts)
    elif dataset_name == "ptb":
        import pandas as pd
        url = "https://huggingface.co/datasets/ptb_text_only/resolve/refs%2Fconvert%2Fparquet/penn_treebank/test/0000.parquet"
        df = pd.read_parquet(url)
        text = " ".join(df["sentence"].tolist())
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings.input_ids[0]
    
    seq_len = input_ids.shape[0]
    
    nlls = []
    n_tokens = 0
    
    for begin in range(0, seq_len, max_length):
        end = min(begin + max_length, seq_len)
        input_chunk = input_ids[begin:end].unsqueeze(0).to(device)
        target_chunk = input_chunk.clone()
        
        outputs = model(input_ids=input_chunk, labels=target_chunk)
        # loss is averaged over tokens in the chunk
        chunk_len = end - begin
        neg_log_likelihood = outputs.loss * (chunk_len - 1)  # -1 because first token has no label
        nlls.append(neg_log_likelihood)
        n_tokens += chunk_len - 1
        
        if end == seq_len:
            break
    
    ppl = torch.exp(torch.stack(nlls).sum() / n_tokens)
    return ppl.item()


# ============================================================
# 10. Main Pipeline
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="GRASPrune")
    parser.add_argument("--model_name", type=str, default="NousResearch/Llama-2-7b-hf")
    parser.add_argument("--target_ratio", type=float, default=0.5, help="Parameter retention ratio")
    parser.add_argument("--n_samples", type=int, default=512)
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--gate_epochs", type=int, default=4)
    parser.add_argument("--gate_lr", type=float, default=1e-2)
    parser.add_argument("--scale_epochs", type=int, default=2)
    parser.add_argument("--scale_lr", type=float, default=1e-2)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--save_path", type=str, default=None)
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--alpha_scale", type=float, default=1.0, help="Scaling factor for alpha")
    parser.add_argument("--output_dir", type=str, default="/workspace/results")
    parser.add_argument("--skip_scaling", action="store_true")
    args = parser.parse_args()
    
    print(f"Loading model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    if args.eval_only:
        model = AutoModelForCausalLM.from_pretrained(
            args.save_path or args.model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        
        # Evaluate
        results = {}
        print("Evaluating perplexity on WikiText-2...")
        wiki_ppl = evaluate_perplexity(model, tokenizer, "wikitext", "wikitext-2-raw-v1", "test")
        results["wikitext2_ppl"] = wiki_ppl
        print(f"WikiText-2 PPL: {wiki_ppl:.4f}")
        
        os.makedirs(args.output_dir, exist_ok=True)
        with open(os.path.join(args.output_dir, "eval_results.json"), "w") as f:
            json.dump(results, f, indent=2)
        return
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    
    config = model.config
    print(f"Model config: {config.num_hidden_layers} layers, "
          f"{config.num_attention_heads} heads, "
          f"{getattr(config, 'num_key_value_heads', config.num_attention_heads)} KV heads, "
          f"intermediate_size={config.intermediate_size}")
    
    alpha = compute_alpha(config) * args.alpha_scale
    print(f"Alpha (KV head cost): {alpha:.4f}")
    
    # Load calibration data
    print(f"Loading calibration data: {args.n_samples} sequences of length {args.seq_len}")
    calibration_data = load_calibration_data(
        tokenizer, n_samples=args.n_samples, seq_len=args.seq_len
    )
    print(f"Calibration data shape: {calibration_data.shape}")
    
    # Initialize gates
    print(f"Target ratio: {args.target_ratio}")
    gate_module = GRASPruneGates(config, args.target_ratio, alpha_scale=args.alpha_scale)
    print(f"Budget: {gate_module.budget:.2f}, Total cost: {gate_module.costs.sum().item():.2f}")
    
    # Train gates
    print("\n=== Gate Learning ===")
    start_time = time.time()
    gate_module = train_gates(
        model, gate_module, calibration_data,
        num_epochs=args.gate_epochs, lr=args.gate_lr, batch_size=args.batch_size
    )
    gate_time = time.time() - start_time
    print(f"Gate learning time: {gate_time:.1f}s")
    
    # Get final mask stats
    with torch.no_grad():
        mask = gate_module.get_mask()
        device = next(model.parameters()).device
        for layer_idx in range(config.num_hidden_layers):
            ffn_gates, kv_gates = gate_module.get_layer_gates(layer_idx, mask)
            ffn_kept = ffn_gates.sum().item()
            kv_kept = kv_gates.sum().item()
            if layer_idx % 8 == 0 or layer_idx == config.num_hidden_layers - 1:
                print(f"  Layer {layer_idx}: FFN {ffn_kept:.0f}/{config.intermediate_size}, "
                      f"KV {kv_kept:.0f}/{getattr(config, 'num_key_value_heads', config.num_attention_heads)}")
    
    # Scaling calibration
    if not args.skip_scaling:
        print("\n=== Scaling Calibration ===")
        start_time = time.time()
        scales = train_scaling(
            model, gate_module, calibration_data,
            num_epochs=args.scale_epochs, lr=args.scale_lr, batch_size=args.batch_size
        )
        scale_time = time.time() - start_time
        print(f"Scaling calibration time: {scale_time:.1f}s")
    else:
        print("Skipping scaling calibration")
        scales = mask.clone().to(device)
    
    # Materialize pruned model
    save_path = args.save_path
    if save_path is None:
        model_short = args.model_name.split("/")[-1]
        save_path = f"/workspace/pruned_models/{model_short}_ratio{args.target_ratio}"
    
    print(f"\n=== Materializing pruned model ===")
    model = materialize_pruned_model(model, gate_module, scales, save_path)
    
    # Save tokenizer too
    tokenizer.save_pretrained(save_path)
    
    # Evaluate
    print("\n=== Evaluation ===")
    model = model.to(torch.bfloat16)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    results = {"target_ratio": args.target_ratio}
    
    print("Evaluating perplexity on WikiText-2...")
    wiki_ppl = evaluate_perplexity(model, tokenizer, "wikitext", "wikitext-2-raw-v1", "test")
    results["wikitext2_ppl"] = wiki_ppl
    print(f"WikiText-2 PPL: {wiki_ppl:.4f}")
    
    try:
        print("Evaluating perplexity on PTB...")
        ptb_ppl = evaluate_perplexity(model, tokenizer, "ptb")
        results["ptb_ppl"] = ptb_ppl
        print(f"PTB PPL: {ptb_ppl:.4f}")
    except Exception as e:
        print(f"PTB evaluation failed: {e}")
    
    try:
        print("Evaluating perplexity on C4...")
        c4_ppl = evaluate_perplexity(model, tokenizer, "c4", split="validation")
        results["c4_ppl"] = c4_ppl
        print(f"C4 PPL: {c4_ppl:.4f}")
    except Exception as e:
        print(f"C4 evaluation failed: {e}")
    
    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    result_file = os.path.join(args.output_dir, f"results_ratio{args.target_ratio}.json")
    with open(result_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {result_file}")
    
    return results


if __name__ == "__main__":
    main()
