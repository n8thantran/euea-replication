"""
SFT Training pipeline for EUEA.
Fine-tunes InternVL3-8B on the 8 core skills dataset.
Based on the paper's training details:
- Full fine-tune for 1 epoch
- Vision encoder frozen, MLP + LLM fine-tuned
- Batch size: 128, Max seq length: 8192
- 8×A100 80GB GPUs (we adapt for single GPU)
"""

import os
import sys
import json
import torch
import argparse
from typing import Dict, List, Any
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, '/workspace')


class SkillSFTDataset(Dataset):
    """Dataset for multi-skill SFT training."""
    
    def __init__(self, data_path: str, tokenizer=None, max_length: int = 2048):
        with open(data_path) as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        conversation = sample.get("conversations", [])
        skill_type = sample.get("skill_type", "unknown")
        
        # Build conversation text
        text_parts = []
        for turn in conversation:
            role = turn.get("role", turn.get("from", ""))
            content = turn.get("content", turn.get("value", ""))
            if role in ["user", "human"]:
                text_parts.append(f"<|user|>\n{content}")
            elif role in ["assistant", "gpt"]:
                text_parts.append(f"<|assistant|>\n{content}")
        
        full_text = "\n".join(text_parts) + "<|end|>"
        
        if self.tokenizer:
            encoding = self.tokenizer(
                full_text,
                max_length=self.max_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt"
            )
            return {
                "input_ids": encoding["input_ids"].squeeze(0),
                "attention_mask": encoding["attention_mask"].squeeze(0),
                "labels": encoding["input_ids"].squeeze(0).clone(),
                "skill_type": skill_type,
            }
        else:
            return {
                "text": full_text,
                "skill_type": skill_type,
            }


def prepare_sft_data(data_dir: str = "/workspace/data/train") -> str:
    """Prepare SFT training data from skill datasets."""
    sft_data_path = os.path.join(data_dir, "sft_data.json")
    
    if os.path.exists(sft_data_path):
        with open(sft_data_path) as f:
            data = json.load(f)
        print(f"Loaded existing SFT data with {len(data)} samples")
        return sft_data_path
    
    print("No SFT data found. Generate via data_generator.py first.")
    return sft_data_path


def train_sft(
    model_name: str = "OpenGVLab/InternVL3-8B",
    data_path: str = "/workspace/data/train/sft_data.json",
    output_dir: str = "/workspace/models/sft",
    batch_size: int = 1,
    grad_accum: int = 8,
    learning_rate: float = 2e-5,
    num_epochs: int = 1,
    max_length: int = 2048,
    use_lora: bool = True,
    lora_r: int = 64,
    lora_alpha: int = 128,
):
    """Run SFT training on skill dataset."""
    
    from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
    from peft import get_peft_model, LoraConfig, TaskType
    
    print(f"Loading tokenizer from {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"Loading model from {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
    )
    
    # Freeze vision encoder (for InternVL models)
    if hasattr(model, 'vision_model'):
        for param in model.vision_model.parameters():
            param.requires_grad = False
        print("Vision encoder frozen")
    
    if use_lora:
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=0.05,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    
    # Create dataset
    dataset = SkillSFTDataset(data_path, tokenizer, max_length)
    print(f"Training dataset: {len(dataset)} samples")
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=learning_rate,
        warmup_ratio=0.1,
        weight_decay=0.01,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        remove_unused_columns=False,
        dataloader_num_workers=2,
        report_to="none",
    )
    
    from transformers import Trainer
    
    # Custom collator
    def collate_fn(batch):
        input_ids = torch.stack([b["input_ids"] for b in batch])
        attention_mask = torch.stack([b["attention_mask"] for b in batch])
        labels = torch.stack([b["labels"] for b in batch])
        # Mask padding tokens in labels
        labels[attention_mask == 0] = -100
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collate_fn=collate_fn,
    )
    
    print("Starting SFT training...")
    trainer.train()
    
    # Save model
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model saved to {output_dir}")
    
    return output_dir


def train_sft_simple(
    data_path: str = "/workspace/data/train/sft_data.json",
    output_dir: str = "/workspace/models/sft",
    num_epochs: int = 1,
    batch_size: int = 4,
    learning_rate: float = 2e-5,
    max_length: int = 512,
):
    """Simplified SFT using a small model for demonstration."""
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import get_peft_model, LoraConfig, TaskType
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Use a small model for demonstration
    model_name = "Qwen/Qwen2.5-0.5B"
    
    print(f"Loading small model for demo: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to("cuda" if torch.cuda.is_available() else "cpu")
    
    # LoRA
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Dataset
    dataset = SkillSFTDataset(data_path, tokenizer, max_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, 
                           collate_fn=lambda batch: {
                               "input_ids": torch.stack([b["input_ids"] for b in batch]),
                               "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
                               "labels": torch.stack([b["labels"] for b in batch]),
                           })
    
    # Training
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    device = next(model.parameters()).device
    
    model.train()
    total_loss = 0
    step = 0
    
    for epoch in range(num_epochs):
        for batch_idx, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            labels[attention_mask == 0] = -100
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            
            optimizer.step()
            optimizer.zero_grad()
            
            total_loss += loss.item()
            step += 1
            
            if step % 10 == 0:
                avg_loss = total_loss / step
                print(f"Epoch {epoch+1}, Step {step}, Loss: {loss.item():.4f}, Avg Loss: {avg_loss:.4f}")
            
            if step >= 100:  # Limit for demo
                break
        
        if step >= 100:
            break
    
    # Save
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Save training log
    log = {
        "model": model_name,
        "epochs": num_epochs,
        "steps": step,
        "final_avg_loss": total_loss / max(step, 1),
        "data_path": data_path,
    }
    with open(os.path.join(output_dir, "training_log.json"), 'w') as f:
        json.dump(log, f, indent=2)
    
    print(f"SFT demo training complete. Model saved to {output_dir}")
    return output_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="demo", choices=["demo", "full"])
    parser.add_argument("--data_path", type=str, default="/workspace/data/train/sft_data.json")
    parser.add_argument("--output_dir", type=str, default="/workspace/models/sft")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-5)
    args = parser.parse_args()
    
    if args.mode == "demo":
        train_sft_simple(
            data_path=args.data_path,
            output_dir=args.output_dir,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
        )
    else:
        train_sft(
            data_path=args.data_path,
            output_dir=args.output_dir,
            num_epochs=args.epochs,
            batch_size=1,
            learning_rate=args.lr,
        )
