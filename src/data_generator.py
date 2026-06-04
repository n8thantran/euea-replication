"""
Generate synthetic skill evaluation data matching ALFRED format.
This creates evaluation samples for all 8 skills to test the pipeline.
In the real paper, data comes from ALFRED trajectories.
"""

import json
import random
import os
from typing import List, Dict, Any

# ALFRED-like task templates
TASK_TEMPLATES = {
    "Look": {
        "goals": [
            "Examine the {obj} under the {light}.",
            "Look at the {obj} using the {light}.",
        ],
        "subgoals": [
            "Pick up the {obj}.",
            "Go to the {light}.",
            "Turn on the {light}.",
        ],
        "objects": ["book", "pen", "cd", "cell phone", "watch", "credit card"],
        "lights": ["desk lamp", "floor lamp"],
    },
    "Pick": {
        "goals": [
            "Put the {obj} on the {recep}.",
            "Place the {obj} in the {recep}.",
        ],
        "subgoals": [
            "Pick up the {obj}.",
            "Go to the {recep}.",
            "Put the {obj} on the {recep}.",
        ],
        "objects": ["apple", "mug", "plate", "knife", "fork", "spoon", "egg", "potato", "tomato", "bread"],
        "receptacles": ["counter top", "dining table", "shelf", "side table", "desk"],
    },
    "Pick Two": {
        "goals": [
            "Put two {obj}s on the {recep}.",
        ],
        "subgoals": [
            "Pick up the first {obj}.",
            "Go to the {recep}.",
            "Put the {obj} on the {recep}.",
            "Pick up the second {obj}.",
            "Go to the {recep}.",
            "Put the {obj} on the {recep}.",
        ],
        "objects": ["apple", "pen", "cd", "candle", "credit card"],
        "receptacles": ["shelf", "desk", "counter top"],
    },
    "Clean": {
        "goals": [
            "Clean the {obj} and put it on the {recep}.",
        ],
        "subgoals": [
            "Pick up the {obj}.",
            "Go to the sink.",
            "Clean the {obj}.",
            "Go to the {recep}.",
            "Put the {obj} on the {recep}.",
        ],
        "objects": ["apple", "mug", "plate", "knife", "fork", "spoon", "potato", "tomato", "cup"],
        "receptacles": ["counter top", "dining table", "shelf"],
    },
    "Cool": {
        "goals": [
            "Cool the {obj} and put it on the {recep}.",
        ],
        "subgoals": [
            "Pick up the {obj}.",
            "Go to the fridge.",
            "Cool the {obj}.",
            "Go to the {recep}.",
            "Put the {obj} on the {recep}.",
        ],
        "objects": ["apple", "mug", "cup", "potato", "tomato", "bread", "egg", "lettuce"],
        "receptacles": ["counter top", "dining table", "shelf"],
    },
    "Heat": {
        "goals": [
            "Heat the {obj} and put it on the {recep}.",
        ],
        "subgoals": [
            "Pick up the {obj}.",
            "Go to the microwave.",
            "Heat the {obj}.",
            "Go to the {recep}.",
            "Put the {obj} on the {recep}.",
        ],
        "objects": ["apple", "mug", "cup", "potato", "bread", "egg"],
        "receptacles": ["counter top", "dining table", "shelf"],
    },
}


def generate_bbox():
    """Generate a random bounding box [x1, y1, x2, y2] in [0, 1000]."""
    x1 = random.randint(50, 700)
    y1 = random.randint(50, 700)
    x2 = x1 + random.randint(50, 250)
    y2 = y1 + random.randint(50, 250)
    return [min(x1, 999), min(y1, 999), min(x2, 999), min(y2, 999)]


def generate_object_list(n_objects=None):
    """Generate a random list of visible objects."""
    all_objects = [
        "apple", "mug", "plate", "knife", "fork", "spoon", "egg",
        "potato", "tomato", "bread", "cup", "bowl", "pan", "pot",
        "sink", "fridge", "microwave", "counter top", "stove burner",
        "cabinet", "drawer", "shelf", "dining table", "desk lamp",
    ]
    if n_objects is None:
        n_objects = random.randint(3, 10)
    return random.sample(all_objects, min(n_objects, len(all_objects)))


def generate_skill_samples(n_per_skill=100, seed=42):
    """Generate evaluation samples for all 8 skills."""
    random.seed(seed)
    samples = {
        "object_recognition": [],
        "object_detection": [],
        "subgoal_planning": [],
        "step_by_step": [],
        "action_prediction": [],
        "future_captioning": [],
        "action_grounding": [],
        "goal_recognition_main": [],
        "goal_recognition_sub": [],
    }
    
    actions = ["PickupObject", "PutObject", "OpenObject", "CloseObject",
               "ToggleObjectOn", "ToggleObjectOff", "SliceObject"]
    
    for _ in range(n_per_skill):
        task_type = random.choice(list(TASK_TEMPLATES.keys()))
        template = TASK_TEMPLATES[task_type]
        obj = random.choice(template["objects"])
        
        # Object Recognition
        visible_objects = generate_object_list()
        if obj not in visible_objects:
            visible_objects.append(obj)
        samples["object_recognition"].append({
            "skill": "OR",
            "gt_objects": visible_objects,
            "image_id": f"scene_{random.randint(0, 999):04d}",
        })
        
        # Object Detection
        bbox = generate_bbox()
        samples["object_detection"].append({
            "skill": "OD",
            "object_name": obj,
            "gt_bbox": bbox,
            "image_id": f"scene_{random.randint(0, 999):04d}",
        })
        
        # Subgoal Task Planning
        if "receptacles" in template:
            recep = random.choice(template["receptacles"])
            goal = random.choice(template["goals"]).format(obj=obj, recep=recep)
            subgoals = [s.format(obj=obj, recep=recep) for s in template["subgoals"]]
        else:
            light = random.choice(template["lights"])
            goal = random.choice(template["goals"]).format(obj=obj, light=light)
            subgoals = [s.format(obj=obj, light=light) for s in template["subgoals"]]
        
        samples["subgoal_planning"].append({
            "skill": "STP",
            "main_goal": goal,
            "gt_subgoals": subgoals,
        })
        
        # Step-by-Step Action Planning
        action = random.choice(actions)
        samples["step_by_step"].append({
            "skill": "SAP",
            "subgoal": subgoals[0] if subgoals else f"Pick up the {obj}.",
            "previous_actions": [],
            "gt_action": action,
            "gt_object": obj,
            "image_id": f"scene_{random.randint(0, 999):04d}",
        })
        
        # Action Success Prediction
        success = random.choice(["yes", "no"])
        samples["action_prediction"].append({
            "skill": "ASP",
            "action": action,
            "object_name": obj,
            "gt_answer": success,
            "image_id_before": f"scene_{random.randint(0, 999):04d}",
            "image_id_after": f"scene_{random.randint(0, 999):04d}",
        })
        
        # Future Situation Captioning
        samples["future_captioning"].append({
            "skill": "FSC",
            "action": action,
            "object_name": obj,
            "gt_description": f"After {action} on {obj}, the {obj} is now in a different state.",
            "image_id": f"scene_{random.randint(0, 999):04d}",
        })
        
        # Action Grounding
        bbox2 = generate_bbox()
        samples["action_grounding"].append({
            "skill": "AG",
            "gt_action": action,
            "gt_object": obj,
            "gt_bbox": bbox2,
            "image_id_before": f"scene_{random.randint(0, 999):04d}",
            "image_id_after": f"scene_{random.randint(0, 999):04d}",
        })
        
        # Goal Recognition - Main
        achieved = random.choice(["yes", "no"])
        samples["goal_recognition_main"].append({
            "skill": "GR_main",
            "main_goal": goal,
            "gt_answer": achieved,
            "image_id": f"scene_{random.randint(0, 999):04d}",
        })
        
        # Goal Recognition - Sub
        sub_achieved = random.choice(["yes", "no"])
        samples["goal_recognition_sub"].append({
            "skill": "GR_sub",
            "subgoal": subgoals[0] if subgoals else f"Pick up the {obj}.",
            "gt_answer": sub_achieved,
            "image_id": f"scene_{random.randint(0, 999):04d}",
        })
    
    return samples


def generate_sft_training_data(n_samples=1000, seed=42):
    """
    Generate SFT training data in conversation format.
    Each sample is a multi-turn conversation with image + instruction + response.
    """
    random.seed(seed)
    training_data = []
    
    actions = ["PickupObject", "PutObject", "OpenObject", "CloseObject",
               "ToggleObjectOn", "ToggleObjectOff", "SliceObject"]
    
    for i in range(n_samples):
        task_type = random.choice(list(TASK_TEMPLATES.keys()))
        template = TASK_TEMPLATES[task_type]
        obj = random.choice(template["objects"])
        
        if "receptacles" in template:
            recep = random.choice(template["receptacles"])
            goal = random.choice(template["goals"]).format(obj=obj, recep=recep)
            subgoals = [s.format(obj=obj, recep=recep) for s in template["subgoals"]]
        else:
            light = random.choice(template["lights"])
            goal = random.choice(template["goals"]).format(obj=obj, light=light)
            subgoals = [s.format(obj=obj, light=light) for s in template["subgoals"]]
        
        # Randomly select a skill for this sample
        skill = random.choice(["OR", "OD", "STP", "SAP", "ASP", "FSC", "AG", "GR_main", "GR_sub"])
        
        if skill == "OR":
            visible_objects = generate_object_list()
            if obj not in visible_objects:
                visible_objects.append(obj)
            conversation = {
                "id": f"train_{i:06d}",
                "skill": skill,
                "conversations": [
                    {"role": "user", "content": "<image>\nList all visible objects in this scene."},
                    {"role": "assistant", "content": "\n".join(visible_objects)},
                ],
            }
        elif skill == "OD":
            bbox = generate_bbox()
            conversation = {
                "id": f"train_{i:06d}",
                "skill": skill,
                "conversations": [
                    {"role": "user", "content": f"<image>\nDetect the bounding box of {obj} in the image. Output as [x1, y1, x2, y2]."},
                    {"role": "assistant", "content": f"[{bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]}]"},
                ],
            }
        elif skill == "STP":
            conversation = {
                "id": f"train_{i:06d}",
                "skill": skill,
                "conversations": [
                    {"role": "user", "content": f"Given the main goal: {goal}\nGenerate a sequence of subgoals."},
                    {"role": "assistant", "content": "\n".join(subgoals)},
                ],
            }
        elif skill == "SAP":
            action = random.choice(actions)
            conversation = {
                "id": f"train_{i:06d}",
                "skill": skill,
                "conversations": [
                    {"role": "user", "content": f"<image>\nCurrent subgoal: {subgoals[0]}\nPredict the next action and target object."},
                    {"role": "assistant", "content": f"Action: {action}\nObject: {obj}"},
                ],
            }
        elif skill == "ASP":
            action = random.choice(actions)
            answer = random.choice(["Yes", "No"])
            conversation = {
                "id": f"train_{i:06d}",
                "skill": skill,
                "conversations": [
                    {"role": "user", "content": f"<image>\nAction performed: {action} on {obj}.\nWas the action successful? Answer Yes or No."},
                    {"role": "assistant", "content": answer},
                ],
            }
        elif skill == "FSC":
            action = random.choice(actions)
            conversation = {
                "id": f"train_{i:06d}",
                "skill": skill,
                "conversations": [
                    {"role": "user", "content": f"<image>\nAction to perform: {action} on {obj}.\nDescribe the expected scene change."},
                    {"role": "assistant", "content": f"After performing {action} on {obj}, the {obj} will be in a different state."},
                ],
            }
        elif skill == "AG":
            action = random.choice(actions)
            bbox = generate_bbox()
            conversation = {
                "id": f"train_{i:06d}",
                "skill": skill,
                "conversations": [
                    {"role": "user", "content": "<image>\nGiven two consecutive observations, identify the action, object, and bounding box."},
                    {"role": "assistant", "content": f"Action: {action}\nObject: {obj}\nBounding box: [{bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]}]"},
                ],
            }
        elif skill == "GR_main":
            answer = random.choice(["Yes", "No"])
            conversation = {
                "id": f"train_{i:06d}",
                "skill": skill,
                "conversations": [
                    {"role": "user", "content": f"<image>\nMain goal: {goal}\nHas the main goal been achieved? Answer Yes or No."},
                    {"role": "assistant", "content": answer},
                ],
            }
        elif skill == "GR_sub":
            answer = random.choice(["Yes", "No"])
            conversation = {
                "id": f"train_{i:06d}",
                "skill": skill,
                "conversations": [
                    {"role": "user", "content": f"<image>\nSubgoal: {subgoals[0]}\nHas the subgoal been completed? Answer Yes or No."},
                    {"role": "assistant", "content": answer},
                ],
            }
        
        training_data.append(conversation)
    
    return training_data


def save_data(data: Any, filepath: str):
    """Save data to JSON file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Saved {len(data) if isinstance(data, (list, dict)) else 'data'} to {filepath}")


if __name__ == "__main__":
    # Generate evaluation data
    eval_samples = generate_skill_samples(n_per_skill=100)
    for skill_name, samples in eval_samples.items():
        save_data(samples, f"/workspace/data/eval/{skill_name}.json")
    
    # Generate training data
    train_data = generate_sft_training_data(n_samples=1000)
    save_data(train_data, "/workspace/data/train/sft_data.json")
    
    print(f"\nGenerated evaluation data:")
    for skill_name, samples in eval_samples.items():
        print(f"  {skill_name}: {len(samples)} samples")
    print(f"\nGenerated training data: {len(train_data)} samples")
