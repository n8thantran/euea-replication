"""
Prompt templates for all 8 EUEA skills.
Based on the paper's description of instruction templates I_SKILL.
"""

# ============ Object Perception ============

OBJECT_RECOGNITION_PROMPT = """You are an embodied agent observing a scene. List all visible objects in the image.

Output format: List each object name, one per line.
Objects:"""

OBJECT_DETECTION_PROMPT = """You are an embodied agent. Detect the bounding box of the specified object in the image.

Target object: {object_name}

Output the bounding box as [x1, y1, x2, y2] where coordinates are in the range [0, 1000].
Bounding box:"""

# ============ Task Planning ============

SUBGOAL_TASK_PLANNING_PROMPT = """You are an embodied agent. Given the main goal, generate a sequence of subgoals needed to complete the task.

Main goal: {main_goal}

Output format: List each subgoal in order, one per line.
Subgoals:"""

STEP_BY_STEP_ACTION_PLANNING_PROMPT = """You are an embodied agent performing a task. Given the current observation and the current subgoal, predict the next action and target object.

Current subgoal: {subgoal}
Previous actions: {previous_actions}

What is the next action and target object?
Output format:
Action: <action_name>
Object: <object_name>"""

# ============ Action Understanding ============

ACTION_SUCCESS_PREDICTION_PROMPT = """You are an embodied agent. Given two consecutive observations (before and after an action), predict whether the action was successful.

Action performed: {action}
Target object: {object_name}

Was the action successful? Answer with Yes or No.
Answer:"""

FUTURE_SITUATION_CAPTIONING_PROMPT = """You are an embodied agent. Given the current observation and an action to be performed, describe what the scene will look like after the action is executed.

Action to perform: {action}
Target object: {object_name}

Describe the expected change in the scene:"""

ACTION_GROUNDING_PROMPT = """You are an embodied agent. Given two consecutive observations (before and after), identify what action was performed, on which object, and provide the bounding box of the target object.

Output format:
Action: <action_name>
Object: <object_name>
Bounding box: [x1, y1, x2, y2]"""

# ============ Goal Recognition ============

MAIN_GOAL_RECOGNITION_PROMPT = """You are an embodied agent. Given the current observation and the main goal, determine whether the main goal has been achieved.

Main goal: {main_goal}

Has the main goal been achieved? Answer with Yes or No.
Answer:"""

SUBGOAL_RECOGNITION_PROMPT = """You are an embodied agent. Given the current observation and the current subgoal, determine whether the subgoal has been completed.

Current subgoal: {subgoal}

Has the subgoal been completed? Answer with Yes or No.
Answer:"""

# ============ Task Evaluation Pipeline Prompts ============

NAVIGATION_OR_PROMPT = """You are an embodied agent navigating through an environment. Observe the current scene and list all visible objects.

Objects:"""

INTERACTION_SAP_PROMPT = """You are an embodied agent. Given the current subgoal and memory of previous actions, predict the next action and target object.

Current subgoal: {subgoal}
Memory (last {k} steps): {memory}

Action: 
Object:"""

INTERACTION_OD_PROMPT = """You are an embodied agent. Detect the bounding box of the target object for the current action.

Action: {action}
Target object: {object_name}

Bounding box: [x1, y1, x2, y2]"""

INTERACTION_GR_SUB_PROMPT = """You are an embodied agent. Has the current subgoal been completed based on the observation?

Current subgoal: {subgoal}

Answer (Yes/No):"""

# ============ ALFRED Action Space ============

ALFRED_ACTIONS = [
    "PickupObject",
    "PutObject", 
    "OpenObject",
    "CloseObject",
    "ToggleObjectOn",
    "ToggleObjectOff",
    "SliceObject",
    "CleanObject",
    "HeatObject",
    "CoolObject",
]

ALFRED_OBJECTS = [
    "Apple", "Bread", "Butter Knife", "CD", "Candle", "Cell Phone",
    "Cloth", "Credit Card", "Cup", "Dish Sponge", "Egg", "Fork",
    "Kettle", "Knife", "Ladle", "Lettuce", "Mug", "Newspaper",
    "Pan", "Pen", "Pencil", "Pepper Shaker", "Plate", "Pot",
    "Potato", "Remote Control", "Salt Shaker", "Soap Bar",
    "Soap Bottle", "Spatula", "Spoon", "Spray Bottle",
    "Statue", "Tissue Box", "Toilet Paper", "Tomato", "Vase",
    "Watch", "Watering Can", "Wine Bottle",
    # Receptacles
    "Bathtub", "Bowl", "Cabinet", "Coffee Machine", "Counter Top",
    "Desk", "Dining Table", "Drawer", "Dresser", "Fridge",
    "Garbage Can", "Microwave", "Ottoman", "Safe", "Shelf",
    "Side Table", "Sink", "Sofa", "Stove Burner", "Toilet",
    "Toaster",
]

ALFRED_TASK_TYPES = {
    "Look": "Examine an object under a light source",
    "Pick": "Pick up an object and place it somewhere",
    "Pick Two": "Pick up two objects and place them somewhere",
    "Clean": "Clean an object and place it somewhere",
    "Cool": "Cool an object and place it somewhere",
    "Heat": "Heat an object and place it somewhere",
}


def format_prompt(template: str, **kwargs) -> str:
    """Format a prompt template with given arguments."""
    return template.format(**kwargs)


if __name__ == "__main__":
    # Test prompt formatting
    prompt = format_prompt(
        OBJECT_DETECTION_PROMPT,
        object_name="Apple"
    )
    print("Object Detection Prompt:")
    print(prompt)
    print()
    
    prompt = format_prompt(
        STEP_BY_STEP_ACTION_PLANNING_PROMPT,
        subgoal="Pick up the apple",
        previous_actions="PickupObject Apple"
    )
    print("Step-by-Step Prompt:")
    print(prompt)
