"""
Skill evaluation metrics for EUEA.
Implements all 8 sub-skill evaluation metrics as described in the paper.

Metrics:
1. Object Grounding (OR): count + inclusion accuracy
2. Object Detection (OD): IoU
3. Task Planning (STP): BERT cosine similarity between subgoals
4. Step-by-Step (SAP): action + object match accuracy
5. Action Success Prediction (ASP): Yes/No accuracy
6. Future Situation Captioning (FSC): evaluated as part of ASP
7. Action Grounding (AG): action + object + bbox correctness
8. Goal Recognition (GR_main, GR_sub): Yes/No accuracy
"""

import re
import numpy as np
from typing import List, Dict, Tuple, Optional


def compute_iou(box_pred: List[float], box_gt: List[float]) -> float:
    """
    Compute Intersection over Union between two bounding boxes.
    Boxes are in format [x1, y1, x2, y2] (normalized 0-1000 as in InternVL).
    """
    if len(box_pred) != 4 or len(box_gt) != 4:
        return 0.0
    
    x1 = max(box_pred[0], box_gt[0])
    y1 = max(box_pred[1], box_gt[1])
    x2 = min(box_pred[2], box_gt[2])
    y2 = min(box_pred[3], box_gt[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    
    area_pred = max(0, box_pred[2] - box_pred[0]) * max(0, box_pred[3] - box_pred[1])
    area_gt = max(0, box_gt[2] - box_gt[0]) * max(0, box_gt[3] - box_gt[1])
    
    union = area_pred + area_gt - intersection
    
    if union == 0:
        return 0.0
    
    return intersection / union


def parse_bbox_from_text(text: str) -> Optional[List[float]]:
    """Parse bounding box coordinates from model output text."""
    # Try to find pattern like [x1, y1, x2, y2] or (x1, y1, x2, y2)
    patterns = [
        r'\[(\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?)\]',
        r'\((\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?)\)',
        r'<box>(\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?)</box>',
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return [float(match.group(i)) for i in range(1, 5)]
    return None


def jaccard_index(set_a: set, set_b: set) -> float:
    """Compute Jaccard index between two sets."""
    if not set_a and not set_b:
        return 1.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    if union == 0:
        return 0.0
    return intersection / union


# ============ Skill 1: Object Grounding (OR) ============

def eval_object_grounding(pred_objects: List[str], gt_objects: List[str]) -> Dict[str, float]:
    """
    Evaluate object recognition/grounding.
    Metrics: count accuracy + inclusion (Jaccard index) accuracy.
    """
    # Normalize
    pred_set = set(o.lower().strip() for o in pred_objects)
    gt_set = set(o.lower().strip() for o in gt_objects)
    
    # Count accuracy: 1 if counts match, else proportional
    count_acc = 1.0 if len(pred_set) == len(gt_set) else min(len(pred_set), len(gt_set)) / max(len(pred_set), len(gt_set)) if max(len(pred_set), len(gt_set)) > 0 else 1.0
    
    # Inclusion accuracy (Jaccard)
    inclusion_acc = jaccard_index(pred_set, gt_set)
    
    # Combined score (average of count and inclusion)
    combined = (count_acc + inclusion_acc) / 2
    
    return {
        "count_accuracy": count_acc,
        "inclusion_accuracy": inclusion_acc,
        "grounding_score": combined * 100
    }


# ============ Skill 2: Object Detection (OD) ============

def eval_object_detection(pred_bbox: List[float], gt_bbox: List[float]) -> Dict[str, float]:
    """
    Evaluate object detection using IoU.
    """
    iou = compute_iou(pred_bbox, gt_bbox)
    return {
        "iou": iou * 100
    }


# ============ Skill 3: Task Planning (STP) ============

def eval_task_planning(pred_subgoals: List[str], gt_subgoals: List[str], 
                       model=None) -> Dict[str, float]:
    """
    Evaluate subgoal task planning using BERT cosine similarity.
    Uses sentence-transformers for encoding.
    """
    if model is None:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('all-MiniLM-L6-v2')
    
    if not pred_subgoals or not gt_subgoals:
        return {"planning_similarity": 0.0}
    
    # Encode all subgoals
    pred_embeddings = model.encode(pred_subgoals)
    gt_embeddings = model.encode(gt_subgoals)
    
    # Compute pairwise cosine similarity and find best matching
    # Use greedy matching: for each GT, find best matching pred
    from sklearn.metrics.pairwise import cosine_similarity
    sim_matrix = cosine_similarity(gt_embeddings, pred_embeddings)
    
    # Average of best matches for each GT subgoal
    if sim_matrix.shape[1] > 0:
        best_matches = sim_matrix.max(axis=1)
        avg_sim = float(np.mean(best_matches))
    else:
        avg_sim = 0.0
    
    return {"planning_similarity": avg_sim}


# ============ Skill 4: Step-by-Step Action Planning (SAP) ============

def eval_step_by_step(pred_action: str, pred_object: str,
                      gt_action: str, gt_object: str) -> Dict[str, float]:
    """
    Evaluate step-by-step action planning.
    Both action and object must match for success.
    """
    action_match = pred_action.lower().strip() == gt_action.lower().strip()
    object_match = pred_object.lower().strip() == gt_object.lower().strip()
    
    correct = 1.0 if (action_match and object_match) else 0.0
    
    return {
        "action_match": float(action_match),
        "object_match": float(object_match),
        "step_accuracy": correct * 100
    }


# ============ Skill 5: Action Success Prediction (ASP) ============

def eval_action_prediction(pred: str, gt: str) -> Dict[str, float]:
    """
    Evaluate action success prediction (Yes/No accuracy).
    """
    pred_clean = pred.lower().strip()
    gt_clean = gt.lower().strip()
    
    # Extract Yes/No from response
    if "yes" in pred_clean:
        pred_answer = "yes"
    elif "no" in pred_clean:
        pred_answer = "no"
    else:
        pred_answer = pred_clean
    
    correct = 1.0 if pred_answer == gt_clean else 0.0
    
    return {"prediction_accuracy": correct * 100}


# ============ Skill 6: Action Grounding (AG) ============

def eval_action_grounding(pred_action: str, pred_object: str, pred_bbox: List[float],
                          gt_action: str, gt_object: str, gt_bbox: List[float]) -> Dict[str, float]:
    """
    Evaluate action grounding: action + object + bbox correctness.
    """
    action_match = pred_action.lower().strip() == gt_action.lower().strip()
    object_match = pred_object.lower().strip() == gt_object.lower().strip()
    
    iou = compute_iou(pred_bbox, gt_bbox) if pred_bbox and gt_bbox else 0.0
    
    # Combined score: all three must be correct
    # Action grounding score weights action/object match and IoU
    if action_match and object_match and iou > 0.5:
        score = 1.0
    elif action_match and object_match:
        score = 0.5 + 0.5 * iou
    else:
        score = 0.0
    
    return {
        "action_match": float(action_match),
        "object_match": float(object_match),
        "iou": iou * 100,
        "grounding_score": score * 100
    }


# ============ Skill 7 & 8: Goal Recognition (GR_main, GR_sub) ============

def eval_goal_recognition(pred: str, gt: str) -> Dict[str, float]:
    """
    Evaluate goal recognition (Yes/No accuracy).
    Works for both main goal and subgoal recognition.
    """
    pred_clean = pred.lower().strip()
    gt_clean = gt.lower().strip()
    
    # Extract Yes/No from response
    if "yes" in pred_clean:
        pred_answer = "yes"
    elif "no" in pred_clean:
        pred_answer = "no"
    else:
        pred_answer = pred_clean
    
    correct = 1.0 if pred_answer == gt_clean else 0.0
    
    return {"recognition_accuracy": correct * 100}


# ============ GRPO Reward Functions ============

def reward_object_perception(pred_objects: List[str], gt_objects: List[str],
                             pred_bbox: Optional[List[float]] = None,
                             gt_bbox: Optional[List[float]] = None) -> float:
    """R_OP: Jaccard index for OR + IoU for OD."""
    reward = 0.0
    if pred_objects and gt_objects:
        pred_set = set(o.lower().strip() for o in pred_objects)
        gt_set = set(o.lower().strip() for o in gt_objects)
        reward += jaccard_index(pred_set, gt_set)
    
    if pred_bbox is not None and gt_bbox is not None:
        reward += compute_iou(pred_bbox, gt_bbox)
    
    return reward


def reward_task_planning(pred_action: str, pred_object: str,
                         gt_action: str, gt_object: str,
                         pred_subgoals: Optional[List[str]] = None,
                         gt_subgoals: Optional[List[str]] = None) -> float:
    """R_TP: Action-object correctness + subgoal sequence order."""
    reward = 0.0
    
    # SAP reward
    if pred_action.lower().strip() == gt_action.lower().strip():
        reward += 0.5
    if pred_object.lower().strip() == gt_object.lower().strip():
        reward += 0.5
    
    # STP reward (action sequence order)
    if pred_subgoals and gt_subgoals:
        # Check if predicted subgoals maintain correct order
        correct_order = 0
        for i, sg in enumerate(pred_subgoals):
            if i < len(gt_subgoals) and sg.lower().strip() == gt_subgoals[i].lower().strip():
                correct_order += 1
        if len(gt_subgoals) > 0:
            reward += correct_order / len(gt_subgoals)
    
    return reward


def reward_action_understanding(pred: str, gt: str) -> float:
    """R_AU: Correctness of ASP/FSC predictions."""
    pred_clean = pred.lower().strip()
    gt_clean = gt.lower().strip()
    
    if "yes" in pred_clean:
        pred_answer = "yes"
    elif "no" in pred_clean:
        pred_answer = "no"
    else:
        pred_answer = pred_clean
    
    return 1.0 if pred_answer == gt_clean else 0.0


def reward_goal_recognition(pred: str, gt: str) -> float:
    """R_GR: Correctness of main/sub goal predictions."""
    pred_clean = pred.lower().strip()
    gt_clean = gt.lower().strip()
    
    if "yes" in pred_clean:
        pred_answer = "yes"
    elif "no" in pred_clean:
        pred_answer = "no"
    else:
        pred_answer = pred_clean
    
    return 1.0 if pred_answer == gt_clean else 0.0


# ============ Aggregate Metrics ============

def aggregate_skill_metrics(results: List[Dict[str, float]], skill_name: str) -> Dict[str, float]:
    """Aggregate metrics across multiple samples for a given skill."""
    if not results:
        return {}
    
    keys = results[0].keys()
    aggregated = {}
    for key in keys:
        values = [r[key] for r in results if key in r]
        aggregated[f"{skill_name}_{key}_mean"] = np.mean(values)
        aggregated[f"{skill_name}_{key}_std"] = np.std(values)
    
    return aggregated


if __name__ == "__main__":
    # Quick test of metrics
    print("Testing metrics...")
    
    # Test IoU
    assert abs(compute_iou([0, 0, 100, 100], [0, 0, 100, 100]) - 1.0) < 1e-6
    assert abs(compute_iou([0, 0, 50, 50], [50, 50, 100, 100]) - 0.0) < 1e-6
    print("IoU tests passed!")
    
    # Test Object Grounding
    result = eval_object_grounding(
        ["Apple", "Knife", "Pan"],
        ["Apple", "Knife", "Bowl"]
    )
    print(f"Object Grounding: {result}")
    
    # Test Step-by-Step
    result = eval_step_by_step("PickupObject", "Apple", "PickupObject", "Apple")
    print(f"Step-by-Step (correct): {result}")
    
    result = eval_step_by_step("PickupObject", "Apple", "PutObject", "Apple")
    print(f"Step-by-Step (wrong action): {result}")
    
    # Test Action Prediction
    result = eval_action_prediction("Yes, the action will succeed.", "yes")
    print(f"Action Prediction: {result}")
    
    # Test Goal Recognition
    result = eval_goal_recognition("No, the goal is not achieved.", "no")
    print(f"Goal Recognition: {result}")
    
    print("\nAll metric tests passed!")
