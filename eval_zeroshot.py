"""Zero-shot evaluation using lm-eval-harness with custom model loading."""
import argparse
import json
import torch
from pathlib import Path
from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM
from load_pruned_model import load_pruned_model


class PrunedHFLM(HFLM):
    """Custom wrapper for pruned models with heterogeneous layer sizes."""
    
    def __init__(self, model_path, batch_size=16, **kwargs):
        # Don't call super().__init__() - we'll set things up manually
        self._model_path = model_path
        self._batch_size = batch_size
        
        # Load model and tokenizer
        model, tokenizer = load_pruned_model(model_path, dtype=torch.bfloat16, device="cuda")
        
        # Now initialize parent with the pre-loaded model
        super().__init__(
            pretrained=model,
            tokenizer=tokenizer,
            batch_size=batch_size,
            dtype="bfloat16",
            **kwargs
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--tasks", type=str, default="boolq,piqa,hellaswag,winogrande,arc_easy,arc_challenge,openbookqa")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--output_path", type=str, default=None)
    args = parser.parse_args()
    
    tasks = args.tasks.split(",")
    
    print(f"Loading pruned model from {args.model_path}")
    lm = PrunedHFLM(args.model_path, batch_size=args.batch_size)
    
    print(f"Running evaluation on tasks: {tasks}")
    results = evaluator.simple_evaluate(
        model=lm,
        tasks=tasks,
        num_fewshot=0,
        batch_size=args.batch_size,
    )
    
    # Extract accuracies
    task_results = {}
    for task_name in tasks:
        if task_name in results["results"]:
            r = results["results"][task_name]
            # Try different metric names
            acc = r.get("acc,none", r.get("acc_norm,none", r.get("acc", None)))
            acc_norm = r.get("acc_norm,none", None)
            task_results[task_name] = {
                "acc": acc,
                "acc_norm": acc_norm,
            }
            print(f"  {task_name}: acc={acc}, acc_norm={acc_norm}")
    
    # Compute average accuracy (using acc_norm where available, else acc)
    accs = []
    for task_name, r in task_results.items():
        # For hellaswag, winogrande, arc_challenge, openbookqa use acc_norm
        if r["acc_norm"] is not None and task_name in ["hellaswag", "winogrande", "arc_challenge", "openbookqa"]:
            accs.append(r["acc_norm"])
        elif r["acc"] is not None:
            accs.append(r["acc"])
    
    avg_acc = sum(accs) / len(accs) if accs else 0
    print(f"\nAverage accuracy: {avg_acc:.4f}")
    
    output = {
        "model_path": args.model_path,
        "task_results": task_results,
        "avg_acc": avg_acc,
    }
    
    if args.output_path:
        with open(args.output_path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"Results saved to {args.output_path}")
    
    return output


if __name__ == "__main__":
    main()
