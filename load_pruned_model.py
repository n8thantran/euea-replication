"""Load a pruned (heterogeneous-layer) model correctly."""
import torch
import json
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from safetensors.torch import load_file


def load_pruned_model(model_path, dtype=torch.bfloat16, device="cuda"):
    """Load a pruned model with heterogeneous layer sizes."""
    model_path = Path(model_path)
    
    # Load state dict
    if (model_path / "model.safetensors").exists():
        state_dict = load_file(str(model_path / "model.safetensors"))
    else:
        state_dict = torch.load(str(model_path / "pytorch_model.bin"), map_location="cpu")
    
    # Load config
    config = AutoConfig.from_pretrained(model_path)
    
    # Build model with original config
    model = AutoModelForCausalLM.from_config(config, torch_dtype=dtype)
    
    # For each layer, find actual sizes from state dict and resize Linear layers
    n_layers = config.num_hidden_layers
    hidden_size = config.hidden_size
    
    for i in range(n_layers):
        prefix = f"model.layers.{i}"
        
        # Attention: q_proj, k_proj, v_proj, o_proj
        q_w = state_dict[f"{prefix}.self_attn.q_proj.weight"]
        k_w = state_dict[f"{prefix}.self_attn.k_proj.weight"]
        v_w = state_dict[f"{prefix}.self_attn.v_proj.weight"]
        o_w = state_dict[f"{prefix}.self_attn.o_proj.weight"]
        
        attn_out = q_w.shape[0]  # Number of attention output units kept
        
        attn = model.model.layers[i].self_attn
        
        # Replace with correctly-sized Linear layers
        attn.q_proj = torch.nn.Linear(hidden_size, attn_out, bias=False, dtype=dtype)
        attn.k_proj = torch.nn.Linear(hidden_size, k_w.shape[0], bias=False, dtype=dtype)
        attn.v_proj = torch.nn.Linear(hidden_size, v_w.shape[0], bias=False, dtype=dtype)
        attn.o_proj = torch.nn.Linear(attn_out, hidden_size, bias=False, dtype=dtype)
        
        # Update attention head counts
        head_dim = config.hidden_size // config.num_attention_heads
        attn.num_heads = attn_out // head_dim
        attn.num_key_value_heads = k_w.shape[0] // head_dim
        attn.num_key_value_groups = attn.num_heads // attn.num_key_value_heads if attn.num_key_value_heads > 0 else 1
        # Also update head_dim  
        attn.head_dim = head_dim
        
        # MLP: gate_proj, up_proj, down_proj
        gate_w = state_dict[f"{prefix}.mlp.gate_proj.weight"]
        up_w = state_dict[f"{prefix}.mlp.up_proj.weight"]
        down_w = state_dict[f"{prefix}.mlp.down_proj.weight"]
        
        intermediate = gate_w.shape[0]
        
        mlp = model.model.layers[i].mlp
        mlp.gate_proj = torch.nn.Linear(hidden_size, intermediate, bias=False, dtype=dtype)
        mlp.up_proj = torch.nn.Linear(hidden_size, intermediate, bias=False, dtype=dtype)
        mlp.down_proj = torch.nn.Linear(intermediate, hidden_size, bias=False, dtype=dtype)
    
    # Now load the state dict
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device)
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    return model, tokenizer


if __name__ == "__main__":
    import sys
    model_path = sys.argv[1] if len(sys.argv) > 1 else "/workspace/pruned_models/Llama-2-7b-hf_ratio0.8"
    
    print(f"Loading pruned model from {model_path}")
    model, tokenizer = load_pruned_model(model_path)
    
    # Quick test
    inputs = tokenizer("The quick brown fox jumps over the", return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=10)
    print("Generated:", tokenizer.decode(out[0], skip_special_tokens=True))
    
    # Check per-layer sizes
    for i in [0, 8, 16, 24, 31]:
        layer = model.model.layers[i]
        print(f"  Layer {i}: FFN={layer.mlp.gate_proj.weight.shape[0]}, "
              f"heads={layer.self_attn.num_heads}, "
              f"kv_heads={layer.self_attn.num_key_value_heads}")
