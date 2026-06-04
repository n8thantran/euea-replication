"""
VLM inference module for EUEA skill evaluation.
Supports InternVL3-8B and other VLMs via transformers.
"""

import torch
import re
import json
from typing import List, Dict, Optional, Tuple
from PIL import Image
import numpy as np


class VLMInference:
    """Wrapper for VLM inference using InternVL3-8B or similar models."""
    
    def __init__(self, model_name: str = "OpenGVLab/InternVL3-8B", 
                 device: str = "cuda", dtype=torch.bfloat16):
        self.model_name = model_name
        self.device = device
        self.dtype = dtype
        self.model = None
        self.tokenizer = None
        self.processor = None
        
    def load_model(self):
        """Load the VLM model and tokenizer."""
        print(f"Loading model: {self.model_name}")
        
        if "InternVL" in self.model_name:
            self._load_internvl()
        else:
            self._load_generic()
        
        print(f"Model loaded on {self.device}")
    
    def _load_internvl(self):
        """Load InternVL3-8B model."""
        from transformers import AutoModel, AutoTokenizer, AutoProcessor
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=True
        )
        self.model = AutoModel.from_pretrained(
            self.model_name,
            torch_dtype=self.dtype,
            trust_remote_code=True,
            device_map="auto",
        )
        self.model.eval()
    
    def _load_generic(self):
        """Load generic VLM via transformers."""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=self.dtype,
            trust_remote_code=True,
            device_map="auto",
        )
        self.model.eval()
    
    def generate(self, prompt: str, images: Optional[List[Image.Image]] = None,
                 max_new_tokens: int = 512, temperature: float = 0.0) -> str:
        """Generate response from the VLM."""
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        if "InternVL" in self.model_name:
            return self._generate_internvl(prompt, images, max_new_tokens, temperature)
        else:
            return self._generate_generic(prompt, images, max_new_tokens, temperature)
    
    def _generate_internvl(self, prompt: str, images: Optional[List[Image.Image]] = None,
                           max_new_tokens: int = 512, temperature: float = 0.0) -> str:
        """Generate using InternVL3-8B."""
        generation_config = dict(
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=max(temperature, 0.01),
        )
        
        if images and len(images) > 0:
            # Multi-image or single image
            pixel_values_list = []
            for img in images:
                pixel_values = self._preprocess_image_internvl(img)
                pixel_values_list.append(pixel_values)
            pixel_values = torch.cat(pixel_values_list, dim=0).to(self.device).to(self.dtype)
            
            response = self.model.chat(
                self.tokenizer, pixel_values, prompt, generation_config
            )
        else:
            response = self.model.chat(
                self.tokenizer, None, prompt, generation_config
            )
        
        return response
    
    def _preprocess_image_internvl(self, image: Image.Image) -> torch.Tensor:
        """Preprocess image for InternVL."""
        # InternVL uses its own image processing
        from torchvision import transforms
        
        # Standard InternVL preprocessing
        IMAGENET_MEAN = (0.485, 0.456, 0.406)
        IMAGENET_STD = (0.229, 0.224, 0.225)
        
        transform = transforms.Compose([
            transforms.Resize((448, 448)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        pixel_values = transform(image).unsqueeze(0)
        return pixel_values
    
    def _generate_generic(self, prompt: str, images: Optional[List[Image.Image]] = None,
                          max_new_tokens: int = 512, temperature: float = 0.0) -> str:
        """Generic generation fallback."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0,
                temperature=max(temperature, 0.01),
            )
        
        response = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        return response
    
    def batch_generate(self, prompts: List[str], 
                       images_list: Optional[List[List[Image.Image]]] = None,
                       max_new_tokens: int = 512) -> List[str]:
        """Generate responses for multiple prompts."""
        responses = []
        for i, prompt in enumerate(prompts):
            imgs = images_list[i] if images_list else None
            resp = self.generate(prompt, imgs, max_new_tokens)
            responses.append(resp)
        return responses


class MockVLM:
    """
    Mock VLM for testing the pipeline without GPU.
    Returns plausible mock responses for each skill type.
    """
    
    def __init__(self):
        self.model_name = "MockVLM"
    
    def load_model(self):
        print("Mock VLM loaded (no GPU required)")
    
    def generate(self, prompt: str, images=None, max_new_tokens=512, temperature=0.0) -> str:
        """Generate mock responses based on prompt content."""
        prompt_lower = prompt.lower()
        
        if "list all visible objects" in prompt_lower or "list all objects" in prompt_lower:
            return "apple\nmug\nknife\ncounter top\nsink\nfridge\nbowl"
        
        elif "bounding box" in prompt_lower and "detect" in prompt_lower:
            return "[245, 312, 498, 567]"
        
        elif "subgoal" in prompt_lower and "generate" in prompt_lower:
            return "Pick up the object.\nGo to the destination.\nPut the object down."
        
        elif "next action" in prompt_lower or "predict the next" in prompt_lower:
            return "Action: PickupObject\nObject: apple"
        
        elif "successful" in prompt_lower or "was the action" in prompt_lower:
            return "Yes" if np.random.random() > 0.3 else "No"
        
        elif "describe the expected" in prompt_lower or "scene change" in prompt_lower:
            return "After performing the action, the object has been moved to a new location."
        
        elif "identify the action" in prompt_lower or "what action was performed" in prompt_lower:
            return "Action: PickupObject\nObject: apple\nBounding box: [200, 300, 450, 550]"
        
        elif "main goal" in prompt_lower and ("achieved" in prompt_lower or "completed" in prompt_lower):
            return "No"
        
        elif "subgoal" in prompt_lower and ("completed" in prompt_lower or "achieved" in prompt_lower):
            return "Yes" if np.random.random() > 0.5 else "No"
        
        else:
            return "I observe a kitchen scene with various objects."


# ============ Response Parsers ============

def parse_objects_from_response(response: str) -> List[str]:
    """Parse object list from VLM response."""
    objects = []
    for line in response.strip().split('\n'):
        line = line.strip()
        if line and not line.startswith('#'):
            # Remove numbering like "1. " or "- "
            line = re.sub(r'^[\d]+[\.\)]\s*', '', line)
            line = re.sub(r'^[-*]\s*', '', line)
            if line:
                objects.append(line.lower().strip())
    return objects


def parse_bbox_from_response(response: str) -> Optional[List[float]]:
    """Parse bounding box from VLM response."""
    from src.metrics import parse_bbox_from_text
    return parse_bbox_from_text(response)


def parse_action_object_from_response(response: str) -> Tuple[str, str]:
    """Parse action and object from VLM response."""
    action = ""
    obj = ""
    
    for line in response.strip().split('\n'):
        line = line.strip()
        if line.lower().startswith('action:'):
            action = line.split(':', 1)[1].strip()
        elif line.lower().startswith('object:'):
            obj = line.split(':', 1)[1].strip()
    
    if not action:
        # Try to extract from single-line response
        match = re.search(r'(\w+Object)\s+(\w+)', response)
        if match:
            action = match.group(1)
            obj = match.group(2)
    
    return action, obj


def parse_yes_no_from_response(response: str) -> str:
    """Parse Yes/No from VLM response."""
    response_lower = response.lower().strip()
    if response_lower.startswith('yes') or 'yes' in response_lower[:20]:
        return "yes"
    elif response_lower.startswith('no') or 'no' in response_lower[:20]:
        return "no"
    return response_lower


def parse_action_grounding_from_response(response: str) -> Tuple[str, str, Optional[List[float]]]:
    """Parse action, object, and bbox from action grounding response."""
    action, obj = parse_action_object_from_response(response)
    bbox = parse_bbox_from_response(response)
    return action, obj, bbox


if __name__ == "__main__":
    # Test mock VLM
    vlm = MockVLM()
    vlm.load_model()
    
    # Test various prompts
    test_prompts = [
        "List all visible objects in the scene.",
        "Detect the bounding box of apple in the image. Output as [x1, y1, x2, y2].",
        "Given the main goal: Put the apple on the counter.\nGenerate a sequence of subgoals.",
        "Current subgoal: Pick up the apple.\nPredict the next action and target object.",
        "Action performed: PickupObject on apple.\nWas the action successful? Answer Yes or No.",
        "Main goal: Put the apple on the counter.\nHas the main goal been achieved? Answer Yes or No.",
    ]
    
    for prompt in test_prompts:
        response = vlm.generate(prompt)
        print(f"Prompt: {prompt[:60]}...")
        print(f"Response: {response}")
        print()
    
    # Test parsers
    print("Testing parsers:")
    objects = parse_objects_from_response("apple\nmug\nknife\ncounter top")
    print(f"Objects: {objects}")
    
    bbox = parse_bbox_from_response("[245, 312, 498, 567]")
    print(f"BBox: {bbox}")
    
    action, obj = parse_action_object_from_response("Action: PickupObject\nObject: apple")
    print(f"Action: {action}, Object: {obj}")
    
    yn = parse_yes_no_from_response("Yes, the action was successful.")
    print(f"Yes/No: {yn}")
