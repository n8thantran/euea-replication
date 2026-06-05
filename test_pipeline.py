"""Quick test of GRASPrune pipeline on TinyLlama with minimal data."""

import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer

from graspune import (
    load_calibration_data, compute_alpha, GRASPruneGates,
    apply_gates_to_model_forward, train_gates, train_scaling,
    materialize_pruned_model, evaluate_perplexity
)

def test_pipeline():
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    
    print("=== Loading model ===")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    config = model.config
    print(f"Config: {config.num_hidden_layers} layers, {config.num_attention_heads} heads, "
          f"{config.num_key_value_heads} KV heads, intermediate={config.intermediate_size}")
    
    alpha = compute_alpha(config)
    print(f"Alpha: {alpha:.4f}")
    
    # Test 1: Load calibration data (small)
    print("\n=== Test 1: Load calibration data ===")
    cal_data = load_calibration_data(tokenizer, n_samples=8, seq_len=128)
    print(f"Calibration data shape: {cal_data.shape}")
    
    # Test 2: Initialize gates
    print("\n=== Test 2: Initialize gates ===")
    target_ratio = 0.5
    gate_module = GRASPruneGates(config, target_ratio)
    print(f"Total units: {gate_module.scores.shape[0]}")
    print(f"Budget: {gate_module.budget:.2f}, Total cost: {gate_module.costs.sum().item():.2f}")
    
    # Test 3: Forward pass with gates
    print("\n=== Test 3: Forward pass with gates ===")
    device = next(model.parameters()).device
    gate_module = gate_module.to(device)
    
    z_tilde = gate_module.get_ste_gates()
    print(f"z_tilde shape: {z_tilde.shape}, requires_grad: {z_tilde.requires_grad}")
    
    batch = cal_data[:1].to(device)
    loss = apply_gates_to_model_forward(model, batch, gate_module, z_tilde)
    print(f"Loss: {loss.item():.4f}")
    
    # Test backward
    for p in model.parameters():
        p.requires_grad = False
    loss.backward()
    print(f"Gate grad norm: {gate_module.scores.grad.norm().item():.6f}")
    print(f"Gate grad nonzero: {(gate_module.scores.grad != 0).sum().item()}/{gate_module.scores.shape[0]}")
    
    # Test 4: Train gates (1 epoch, few samples)
    print("\n=== Test 4: Train gates (1 epoch) ===")
    gate_module2 = GRASPruneGates(config, target_ratio)
    start = time.time()
    gate_module2 = train_gates(model, gate_module2, cal_data, num_epochs=1, lr=1e-2, batch_size=1)
    elapsed = time.time() - start
    print(f"Training time: {elapsed:.1f}s for {cal_data.shape[0]} samples")
    
    # Test 5: Get mask stats
    print("\n=== Test 5: Mask stats ===")
    with torch.no_grad():
        mask = gate_module2.get_mask()
        for layer_idx in range(config.num_hidden_layers):
            ffn_gates, kv_gates = gate_module2.get_layer_gates(layer_idx, mask)
            ffn_kept = ffn_gates.sum().item()
            kv_kept = kv_gates.sum().item()
            print(f"  Layer {layer_idx}: FFN {ffn_kept:.0f}/{config.intermediate_size}, "
                  f"KV {kv_kept:.0f}/{config.num_key_value_heads}")
    
    # Test 6: Scaling
    print("\n=== Test 6: Scaling calibration ===")
    scales = train_scaling(model, gate_module2, cal_data[:4], num_epochs=1, lr=1e-2, batch_size=1)
    print(f"Scales shape: {scales.shape}, nonzero: {(scales != 0).sum().item()}")
    
    # Test 7: Materialize
    print("\n=== Test 7: Materialize pruned model ===")
    model = materialize_pruned_model(model, gate_module2, scales, save_path=None)
    
    # Quick check: can we do a forward pass?
    model = model.to(torch.bfloat16).to(device)
    with torch.no_grad():
        out = model(input_ids=batch)
        print(f"Output logits shape: {out.logits.shape}")
    
    # Test 8: Evaluate perplexity
    print("\n=== Test 8: Evaluate perplexity ===")
    ppl = evaluate_perplexity(model, tokenizer, "wikitext", "wikitext-2-raw-v1", "test", max_length=512)
    print(f"WikiText-2 PPL: {ppl:.4f}")
    
    print("\n=== ALL TESTS PASSED ===")

if __name__ == "__main__":
    test_pipeline()
