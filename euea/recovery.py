"""
EUEA Recovery Step via Sampling.

Implements the recovery step algorithm from Section 3.2 of the paper.

When an action fails, the VLM agent samples alternative actions using SAP skill.
The sampled output differs from the failed one but remains aligned with the given goal.

Algorithm:
1. When action fails, sample n=10 alternative actions using SAP
2. Score each sample using negative log-likelihood: s_{t,i} = -log pi(a_{t,i}, o_{t,i} | I_SAP, m_{t-k:t})
3. If all samples match the failed action, instead sample n=10 OD bounding boxes
4. Select the action-object pair or bounding box with the lowest score (highest confidence)
5. If action-object pair selected, perform OD to get bounding box
"""

import math
import random
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass


@dataclass
class ActionSample:
    """A sampled action-object pair with its score."""
    action: str
    obj: str
    bbox: Optional[List[float]]
    score: float  # negative log-likelihood (lower = more confident)
    is_bbox_only: bool = False  # True if this is a bbox-only recovery


@dataclass
class MemoryState:
    """Memory state at time step t."""
    frame: Any  # Image frame
    visible_objects: List[str]
    position: Any
    action: str
    obj: str
    bbox: List[float]
    result: str  # "succeeded" or "failed"
    subgoal: str


class RecoveryStep:
    """
    Recovery Step via Sampling.
    
    When an interaction fails, samples alternative actions to recover.
    Uses SAP skill for action sampling and OD for bounding box sampling.
    
    Args:
        n_samples: Number of samples to generate (default 10)
        k_memory: Number of past memory steps to use (default 4)
        temperature: Sampling temperature for diversity
    """
    
    def __init__(self, n_samples: int = 10, k_memory: int = 4, temperature: float = 1.0):
        self.n_samples = n_samples
        self.k_memory = k_memory
        self.temperature = temperature
    
    def compute_score(self, log_prob: float) -> float:
        """
        Compute score as negative log-likelihood.
        s_{t,i} = -log pi(a_{t,i}, o_{t,i} | I_SAP, m_{t-k:t})
        
        Lower score = higher confidence = better candidate.
        """
        return -log_prob
    
    def sample_actions(self, vlm_agent, instruction_sap: str, 
                       memory: List[MemoryState], current_frame: Any,
                       failed_action: str, failed_object: str) -> List[ActionSample]:
        """
        Sample n alternative actions using SAP skill.
        
        Args:
            vlm_agent: VLM agent with generate() and log_prob() methods
            instruction_sap: SAP instruction template
            memory: Past k memory states
            current_frame: Current observation frame
            failed_action: The action that failed
            failed_object: The object that failed
        
        Returns:
            List of ActionSample candidates
        """
        samples = []
        
        for i in range(self.n_samples):
            # Generate alternative action-object pair
            response = vlm_agent.generate(
                instruction=instruction_sap,
                frame=current_frame,
                memory=memory[-self.k_memory:],
                temperature=self.temperature
            )
            
            action = response.get("action", "")
            obj = response.get("object", "")
            log_prob = response.get("log_prob", -10.0)
            
            score = self.compute_score(log_prob)
            
            samples.append(ActionSample(
                action=action,
                obj=obj,
                bbox=None,
                score=score,
                is_bbox_only=False
            ))
        
        return samples
    
    def sample_bboxes(self, vlm_agent, instruction_od: str,
                      current_frame: Any, subgoal: str,
                      action: str, obj: str) -> List[ActionSample]:
        """
        Sample n alternative bounding boxes using OD skill.
        Used when all SAP samples match the failed action.
        
        Args:
            vlm_agent: VLM agent
            instruction_od: OD instruction template
            current_frame: Current observation frame
            subgoal: Current subgoal
            action: The action to perform
            obj: The target object
        
        Returns:
            List of ActionSample candidates with bounding boxes
        """
        samples = []
        
        for i in range(self.n_samples):
            response = vlm_agent.generate(
                instruction=instruction_od,
                frame=current_frame,
                subgoal=subgoal,
                action=action,
                obj=obj,
                temperature=self.temperature
            )
            
            bbox = response.get("bbox", [0, 0, 0, 0])
            log_prob = response.get("log_prob", -10.0)
            
            score = self.compute_score(log_prob)
            
            samples.append(ActionSample(
                action=action,
                obj=obj,
                bbox=bbox,
                score=score,
                is_bbox_only=True
            ))
        
        return samples
    
    def recover(self, vlm_agent, instruction_sap: str, instruction_od: str,
                memory: List[MemoryState], current_frame: Any,
                failed_action: str, failed_object: str,
                current_subgoal: str) -> ActionSample:
        """
        Execute the full recovery step.
        
        1. Sample n actions using SAP
        2. If all match failed action, sample n bounding boxes using OD
        3. Select candidate with lowest score
        4. If action-object selected, get bbox via OD
        
        Args:
            vlm_agent: VLM agent
            instruction_sap: SAP instruction
            instruction_od: OD instruction
            memory: Past memory states
            current_frame: Current observation
            failed_action: Failed action
            failed_object: Failed object
            current_subgoal: Current subgoal
        
        Returns:
            Best ActionSample for recovery
        """
        # Step 1: Sample alternative actions
        action_samples = self.sample_actions(
            vlm_agent, instruction_sap, memory, current_frame,
            failed_action, failed_object
        )
        
        # Step 2: Check if all samples match the failed action
        all_same = all(
            s.action.lower() == failed_action.lower() and 
            s.obj.lower() == failed_object.lower()
            for s in action_samples
        )
        
        if all_same:
            # All samples are the same failed action -> sample bounding boxes instead
            bbox_samples = self.sample_bboxes(
                vlm_agent, instruction_od, current_frame,
                current_subgoal, failed_action, failed_object
            )
            
            # Select bbox with lowest score
            best_sample = min(bbox_samples, key=lambda s: s.score)
            return best_sample
        else:
            # Filter out samples that match the failed action
            different_samples = [
                s for s in action_samples
                if not (s.action.lower() == failed_action.lower() and 
                       s.obj.lower() == failed_object.lower())
            ]
            
            if not different_samples:
                different_samples = action_samples
            
            # Select action-object with lowest score
            best_sample = min(different_samples, key=lambda s: s.score)
            
            # Get bounding box for the selected action-object pair via OD
            if best_sample.bbox is None:
                od_response = vlm_agent.generate(
                    instruction=instruction_od,
                    frame=current_frame,
                    subgoal=current_subgoal,
                    action=best_sample.action,
                    obj=best_sample.obj,
                    temperature=0.0  # Greedy for bbox
                )
                best_sample.bbox = od_response.get("bbox", [0, 0, 0, 0])
            
            return best_sample


class MockVLMAgent:
    """
    Mock VLM agent for testing the recovery step.
    Simulates VLM responses with configurable behavior.
    """
    
    def __init__(self, action_space: List[str] = None, object_space: List[str] = None):
        self.action_space = action_space or [
            "PickupObject", "PutObject", "OpenObject", "CloseObject",
            "ToggleObjectOn", "ToggleObjectOff", "SliceObject"
        ]
        self.object_space = object_space or [
            "Apple", "Mug", "Plate", "Knife", "Fridge", "Microwave",
            "SinkBasin", "CounterTop", "DiningTable"
        ]
    
    def generate(self, instruction: str = "", frame: Any = None,
                 memory: List = None, temperature: float = 1.0,
                 subgoal: str = "", action: str = "", obj: str = "",
                 **kwargs) -> Dict[str, Any]:
        """Generate a mock response."""
        if temperature > 0:
            # Random sampling
            sampled_action = random.choice(self.action_space)
            sampled_obj = random.choice(self.object_space)
            log_prob = random.uniform(-5.0, -0.1)
        else:
            # Greedy
            sampled_action = action if action else self.action_space[0]
            sampled_obj = obj if obj else self.object_space[0]
            log_prob = -0.1
        
        bbox = [
            random.uniform(0, 500),
            random.uniform(0, 500),
            random.uniform(500, 1000),
            random.uniform(500, 1000)
        ]
        
        return {
            "action": sampled_action,
            "object": sampled_obj,
            "bbox": bbox,
            "log_prob": log_prob
        }


def test_recovery_step():
    """Test the recovery step with mock VLM agent."""
    agent = MockVLMAgent()
    recovery = RecoveryStep(n_samples=10, k_memory=4)
    
    # Create mock memory
    memory = [
        MemoryState(
            frame=None, visible_objects=["Apple", "Mug"],
            position=(0, 0), action="PickupObject", obj="Apple",
            bbox=[100, 100, 200, 200], result="succeeded", subgoal="Pick up the apple"
        )
    ]
    
    # Test recovery
    result = recovery.recover(
        vlm_agent=agent,
        instruction_sap="Generate the next action...",
        instruction_od="Detect the bounding box...",
        memory=memory,
        current_frame=None,
        failed_action="PutObject",
        failed_object="Apple",
        current_subgoal="Put the apple on the table"
    )
    
    print(f"Recovery result: action={result.action}, object={result.obj}, "
          f"bbox={result.bbox}, score={result.score:.3f}, "
          f"is_bbox_only={result.is_bbox_only}")
    
    assert result.action is not None
    assert result.obj is not None
    assert result.bbox is not None
    print("Recovery step test passed!")


if __name__ == "__main__":
    test_recovery_step()
