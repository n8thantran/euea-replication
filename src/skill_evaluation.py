"""
Skill evaluation pipeline for EUEA.
Evaluates VLMs on all 8 skills with proper metrics.
Produces Table 3 from the paper.
"""

import json
import os
import sys
import numpy as np
from typing import Dict, List, Any, Optional
from PIL import Image

# Add workspace to path
sys.path.insert(0, '/workspace')

from src.metrics import (
    eval_object_grounding, eval_object_detection, eval_task_planning,
    eval_step_by_step, eval_action_prediction, eval_action_grounding,
    eval_goal_recognition, aggregate_skill_metrics, parse_bbox_from_text
)
from src.vlm_inference import (
    VLMInference, MockVLM,
    parse_objects_from_response, parse_bbox_from_response,
    parse_action_object_from_response, parse_yes_no_from_response,
    parse_action_grounding_from_response
)
from src.prompts import *


def create_dummy_image(width=448, height=448):
    """Create dummy image for testing (real runs use ALFRED images)."""
    img = Image.fromarray(np.random.randint(0, 255, (height, width, 3), dtype=np.uint8))
    return img


def evaluate_object_recognition(vlm, samples: List[Dict], use_images: bool = False) -> Dict:
    """Evaluate Object Recognition skill."""
    results = []
    for sample in samples:
        images = [create_dummy_image()] if use_images else None
        prompt = OBJECT_RECOGNITION_PROMPT
        response = vlm.generate(prompt, images)
        pred_objects = parse_objects_from_response(response)
        gt_objects = sample["gt_objects"]
        result = eval_object_grounding(pred_objects, gt_objects)
        results.append(result)
    
    aggregated = aggregate_skill_metrics(results, "OR")
    return aggregated


def evaluate_object_detection(vlm, samples: List[Dict], use_images: bool = False) -> Dict:
    """Evaluate Object Detection skill."""
    results = []
    for sample in samples:
        images = [create_dummy_image()] if use_images else None
        prompt = format_prompt(OBJECT_DETECTION_PROMPT, object_name=sample["object_name"])
        response = vlm.generate(prompt, images)
        pred_bbox = parse_bbox_from_response(response)
        gt_bbox = sample["gt_bbox"]
        if pred_bbox:
            result = eval_object_detection(pred_bbox, gt_bbox)
        else:
            result = {"iou": 0.0}
        results.append(result)
    
    return aggregate_skill_metrics(results, "OD")


def evaluate_subgoal_planning(vlm, samples: List[Dict], bert_model=None) -> Dict:
    """Evaluate Subgoal Task Planning skill."""
    results = []
    for sample in samples:
        prompt = format_prompt(SUBGOAL_TASK_PLANNING_PROMPT, main_goal=sample["main_goal"])
        response = vlm.generate(prompt)
        pred_subgoals = [line.strip() for line in response.strip().split('\n') if line.strip()]
        gt_subgoals = sample["gt_subgoals"]
        result = eval_task_planning(pred_subgoals, gt_subgoals, model=bert_model)
        results.append(result)
    
    return aggregate_skill_metrics(results, "STP")


def evaluate_step_by_step(vlm, samples: List[Dict], use_images: bool = False) -> Dict:
    """Evaluate Step-by-Step Action Planning skill."""
    results = []
    for sample in samples:
        images = [create_dummy_image()] if use_images else None
        prev_actions = ", ".join(sample.get("previous_actions", [])) or "None"
        prompt = format_prompt(
            STEP_BY_STEP_ACTION_PLANNING_PROMPT,
            subgoal=sample["subgoal"],
            previous_actions=prev_actions
        )
        response = vlm.generate(prompt, images)
        pred_action, pred_object = parse_action_object_from_response(response)
        gt_action = sample["gt_action"]
        gt_object = sample["gt_object"]
        result = eval_step_by_step(pred_action, pred_object, gt_action, gt_object)
        results.append(result)
    
    return aggregate_skill_metrics(results, "SAP")


def evaluate_action_prediction(vlm, samples: List[Dict], use_images: bool = False) -> Dict:
    """Evaluate Action Success Prediction skill."""
    results = []
    for sample in samples:
        images = [create_dummy_image(), create_dummy_image()] if use_images else None
        prompt = format_prompt(
            ACTION_SUCCESS_PREDICTION_PROMPT,
            action=sample["action"],
            object_name=sample["object_name"]
        )
        response = vlm.generate(prompt, images)
        pred = parse_yes_no_from_response(response)
        gt = sample["gt_answer"]
        result = eval_action_prediction(pred, gt)
        results.append(result)
    
    return aggregate_skill_metrics(results, "ASP")


def evaluate_future_captioning(vlm, samples: List[Dict], use_images: bool = False) -> Dict:
    """Evaluate Future Situation Captioning skill."""
    # FSC is typically evaluated qualitatively or via BERT similarity
    results = []
    for sample in samples:
        images = [create_dummy_image()] if use_images else None
        prompt = format_prompt(
            FUTURE_SITUATION_CAPTIONING_PROMPT,
            action=sample["action"],
            object_name=sample["object_name"]
        )
        response = vlm.generate(prompt, images)
        # Simple keyword overlap metric
        gt_words = set(sample["gt_description"].lower().split())
        pred_words = set(response.lower().split())
        if gt_words:
            overlap = len(gt_words & pred_words) / len(gt_words)
        else:
            overlap = 0.0
        results.append({"captioning_similarity": overlap * 100})
    
    return aggregate_skill_metrics(results, "FSC")


def evaluate_action_grounding(vlm, samples: List[Dict], use_images: bool = False) -> Dict:
    """Evaluate Action Grounding skill."""
    results = []
    for sample in samples:
        images = [create_dummy_image(), create_dummy_image()] if use_images else None
        prompt = ACTION_GROUNDING_PROMPT
        response = vlm.generate(prompt, images)
        pred_action, pred_object, pred_bbox = parse_action_grounding_from_response(response)
        result = eval_action_grounding(
            pred_action, pred_object, pred_bbox if pred_bbox else [],
            sample["gt_action"], sample["gt_object"], sample["gt_bbox"]
        )
        results.append(result)
    
    return aggregate_skill_metrics(results, "AG")


def evaluate_goal_recognition_main(vlm, samples: List[Dict], use_images: bool = False) -> Dict:
    """Evaluate Main Goal Recognition skill."""
    results = []
    for sample in samples:
        images = [create_dummy_image()] if use_images else None
        prompt = format_prompt(MAIN_GOAL_RECOGNITION_PROMPT, main_goal=sample["main_goal"])
        response = vlm.generate(prompt, images)
        pred = parse_yes_no_from_response(response)
        gt = sample["gt_answer"]
        result = eval_goal_recognition(pred, gt)
        results.append(result)
    
    return aggregate_skill_metrics(results, "GR_main")


def evaluate_goal_recognition_sub(vlm, samples: List[Dict], use_images: bool = False) -> Dict:
    """Evaluate Subgoal Recognition skill."""
    results = []
    for sample in samples:
        images = [create_dummy_image()] if use_images else None
        prompt = format_prompt(SUBGOAL_RECOGNITION_PROMPT, subgoal=sample["subgoal"])
        response = vlm.generate(prompt, images)
        pred = parse_yes_no_from_response(response)
        gt = sample["gt_answer"]
        result = eval_goal_recognition(pred, gt)
        results.append(result)
    
    return aggregate_skill_metrics(results, "GR_sub")


def run_full_skill_evaluation(vlm, data_dir: str = "/workspace/data/eval",
                               max_samples: int = None, 
                               use_images: bool = False) -> Dict[str, Dict]:
    """Run evaluation on all 8 skills and return results."""
    
    all_results = {}
    
    skill_configs = [
        ("object_recognition", evaluate_object_recognition),
        ("object_detection", evaluate_object_detection),
        ("subgoal_planning", evaluate_subgoal_planning),
        ("step_by_step", evaluate_step_by_step),
        ("action_prediction", evaluate_action_prediction),
        ("future_captioning", evaluate_future_captioning),
        ("action_grounding", evaluate_action_grounding),
        ("goal_recognition_main", evaluate_goal_recognition_main),
        ("goal_recognition_sub", evaluate_goal_recognition_sub),
    ]
    
    for skill_name, eval_fn in skill_configs:
        data_path = os.path.join(data_dir, f"{skill_name}.json")
        if not os.path.exists(data_path):
            print(f"Skipping {skill_name}: data file not found at {data_path}")
            continue
        
        with open(data_path) as f:
            samples = json.load(f)
        
        if max_samples:
            samples = samples[:max_samples]
        
        print(f"Evaluating {skill_name} on {len(samples)} samples...")
        results = eval_fn(vlm, samples, use_images=use_images) if skill_name != "subgoal_planning" else eval_fn(vlm, samples)
        all_results[skill_name] = results
        
        # Print key metrics
        for key, value in results.items():
            if "mean" in key:
                print(f"  {key}: {value:.2f}")
    
    return all_results


def format_results_table(results: Dict[str, Dict], model_name: str = "Model") -> str:
    """Format results into a table matching Table 3 from the paper."""
    
    # Header
    header = f"{'Model':<25} {'OR':<10} {'OD':<10} {'STP':<10} {'SAP':<10} {'ASP':<10} {'FSC':<10} {'AG':<10} {'GR_m':<10} {'GR_s':<10}"
    separator = "-" * len(header)
    
    # Extract key metrics
    or_score = results.get("object_recognition", {}).get("OR_grounding_score_mean", 0)
    od_score = results.get("object_detection", {}).get("OD_iou_mean", 0)
    stp_score = results.get("subgoal_planning", {}).get("STP_planning_similarity_mean", 0) * 100
    sap_score = results.get("step_by_step", {}).get("SAP_step_accuracy_mean", 0)
    asp_score = results.get("action_prediction", {}).get("ASP_prediction_accuracy_mean", 0)
    fsc_score = results.get("future_captioning", {}).get("FSC_captioning_similarity_mean", 0)
    ag_score = results.get("action_grounding", {}).get("AG_grounding_score_mean", 0)
    gr_main = results.get("goal_recognition_main", {}).get("GR_main_recognition_accuracy_mean", 0)
    gr_sub = results.get("goal_recognition_sub", {}).get("GR_sub_recognition_accuracy_mean", 0)
    
    row = f"{model_name:<25} {or_score:<10.2f} {od_score:<10.2f} {stp_score:<10.2f} {sap_score:<10.2f} {asp_score:<10.2f} {fsc_score:<10.2f} {ag_score:<10.2f} {gr_main:<10.2f} {gr_sub:<10.2f}"
    
    return f"\n{header}\n{separator}\n{row}\n"


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="mock", help="Model name or 'mock'")
    parser.add_argument("--max_samples", type=int, default=10, help="Max samples per skill")
    parser.add_argument("--data_dir", type=str, default="/workspace/data/eval")
    parser.add_argument("--output_dir", type=str, default="/workspace/results")
    args = parser.parse_args()
    
    # Create VLM
    if args.model == "mock":
        vlm = MockVLM()
    else:
        vlm = VLMInference(args.model)
    vlm.load_model()
    
    # Run evaluation
    results = run_full_skill_evaluation(vlm, args.data_dir, args.max_samples)
    
    # Format and print results
    table = format_results_table(results, args.model)
    print(table)
    
    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, f"skill_eval_{args.model.replace('/', '_')}.json")
    
    # Convert numpy values to float for JSON serialization
    serializable = {}
    for k, v in results.items():
        serializable[k] = {kk: float(vv) for kk, vv in v.items()}
    
    with open(output_path, 'w') as f:
        json.dump(serializable, f, indent=2)
    print(f"\nResults saved to {output_path}")
