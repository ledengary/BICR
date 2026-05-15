#!/usr/bin/env python3
"""
P(True) Evaluation Script for Vision-Language Models.

This script implements the P(True) confidence estimation method for VLMs:
1. Loads generated responses from npz files (from generate_and_extract.py)
2. Reconstructs the original conversation
3. Appends the model's response
4. Adds the uncertainty query ("Is the proposed answer correct? A) no B) yes")
5. Extracts logits for tokens A and B at the first generation step
6. Computes P(True) = softmax([logit_A, logit_B])[1] (probability of "yes")
7. Evaluates and saves results similar to PIK_eval.py
"""

# CRITICAL: Import only os and argparse first to set CUDA_VISIBLE_DEVICES before importing torch
import os
import argparse

# Parse GPU IDs argument first (before importing any CUDA libraries)
_parser = argparse.ArgumentParser(add_help=False)
_parser.add_argument('--gpu_ids', type=str, default='0',
                     help='GPU IDs to use (comma-separated)')
_known_args, _ = _parser.parse_known_args()
os.environ["CUDA_VISIBLE_DEVICES"] = _known_args.gpu_ids

# Now import all other libraries
import gc
import json
import sys
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from tqdm import tqdm
from datetime import datetime
import logging
from PIL import Image
import requests
import io
import base64

# Transformers imports
from transformers import AutoProcessor, AutoModelForCausalLM

# Model-specific imports
try:
    from transformers import Qwen3VLForConditionalGeneration, LlavaNextProcessor, LlavaNextForConditionalGeneration
    from qwen_vl_utils import process_vision_info
except ImportError:
    print("Warning: Some VLM dependencies not found.")

try:
    from transformers import Gemma3ForConditionalGeneration, Gemma3Processor
except ImportError:
    pass

try:
    from transformers import AutoModelForImageTextToText
except ImportError:
    pass

# DeepSeek VL2 imports (optional)
try:
    from deepseek_vl2.models import DeepseekVLV2Processor, DeepseekVLV2ForCausalLM
    from deepseek_vl2.utils.io import load_pil_images
    DEEPSEEK_AVAILABLE = True
except ImportError:
    DEEPSEEK_AVAILABLE = False

from datasets import load_from_disk

# Add utils directory to path for eval functions
sys.path.append(str(Path(__file__).parent / "../utils"))
from eval import calculate_all_metrics, save_evaluation_results

# Create logs directory if it doesn't exist
os.makedirs("logs", exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/ptrue_eval.log", mode='a')
    ]
)
logger = logging.getLogger(__name__)


# ============================================================================
# Constants
# ============================================================================

SYSTEM_PROMPT = "You are a vision language assistant. Provide brief, complete answers."
APPENDED_SYSTEM_PROMPT = "Provide a brief, complete answer."

def get_uncertainty_query():
    """Get the uncertainty query for P(True) evaluation."""
    return (
        "Is the proposed answer correct?\n"
        "A) no\nB) yes\n"
        "Reply with A or B only.\n"
        "Answer: "
    )


# ============================================================================
# Argument Parser
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description='P(True) Evaluation for Vision-Language Models',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model configuration
    parser.add_argument('--model_id', type=str, required=True,
                        help='HuggingFace model identifier (e.g., Qwen/Qwen2.5-VL-3B-Instruct)')
    parser.add_argument('--gpu_ids', type=str, default='0',
                        help='GPU IDs to use (comma-separated)')
    parser.add_argument('--dtype', type=str, default='float32',
                        choices=['float32', 'float16', 'bfloat16'],
                        help='Model dtype')
    
    # Data configuration
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directory containing extracted representations (npz files)')
    parser.add_argument('--dataset_path', type=str, required=True,
                        help='Path to the original dataset (for images)')
    parser.add_argument('--test_dataset_name', type=str, required=True,
                        help='Test dataset name (e.g., test)')
    parser.add_argument('--image_column', type=str, default='image',
                        help='Column name for images in dataset')
    parser.add_argument('--question_column', type=str, default='question',
                        help='Column name for questions in dataset')
    parser.add_argument('--answer_column', type=str, default='answer',
                        help='Column name for answers in dataset')
    
    # Output configuration
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save evaluation results')
    
    # Processing configuration
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Maximum number of samples to process (None for all)')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug mode with verbose output')
    parser.add_argument('--debug_samples', type=int, default=5,
                        help='Number of samples to show detailed debug info')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from existing results. Loads existing test_labels.json and only processes missing samples.')
    
    return parser.parse_args()


# ============================================================================
# Model Loading
# ============================================================================

def load_model_and_processor(model_id, dtype_str):
    """Load VLM model and processor/tokenizer"""
    logger.info(f"Loading model: {model_id}")
    logger.info(f"Dtype: {dtype_str}")
    
    # Map dtype string to torch dtype
    dtype_map = {
        'float32': torch.float32,
        'float16': torch.float16,
        'bfloat16': torch.bfloat16,
    }
    dtype = dtype_map[dtype_str]
    
    # Determine model type based on model_id
    model_id_lower = model_id.lower()
    
    if 'deepseek' in model_id_lower and 'vl' in model_id_lower:
        # DeepSeek VL2 model
        if not DEEPSEEK_AVAILABLE:
            raise ImportError(
                "DeepSeek VL2 dependencies not found. Please install deepseek_vl2: "
                "pip install deepseek_vl2 or clone from https://github.com/deepseek-ai/DeepSeek-VL2"
            )
        model_type = 'deepseek'
        logger.info("Detected DeepSeek VL2 model")
        
        # Use the requested dtype (default float32)
        logger.info(f"Using dtype: {dtype_str} for DeepSeek VL2")
        
        # Load DeepSeek processor
        processor = DeepseekVLV2Processor.from_pretrained(model_id)
        
        # Load DeepSeek model - load first, then convert dtype (matches working diagnostics script)
        # Don't pass torch_dtype to from_pretrained as it may not convert all components (e.g., timm vision encoder)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
        )
        # Convert entire model to requested dtype, then move to CUDA
        model = model.to(dtype).cuda().eval()
        
        # Explicitly convert vision encoder and all its submodules to match model dtype
        # This fixes dtype mismatch issues where vision encoder biases stay in float32
        if hasattr(model, 'vision') and model.vision is not None:
            # Recursively convert all parameters and buffers to the target dtype
            for param in model.vision.parameters():
                param.data = param.data.to(dtype=dtype)
            for buffer in model.vision.buffers():
                buffer.data = buffer.data.to(dtype=dtype)
            logger.info(f"Vision encoder converted to {dtype}")
        
        # Log the actual dtype
        actual_dtype = next(model.parameters()).dtype
        if actual_dtype != dtype:
            logger.warning(f"Model loaded in {actual_dtype} (expected {dtype}). Inputs will be converted to match model dtype.")
        else:
            logger.info(f"Model loaded in {actual_dtype} as requested.")
        
        logger.info("Model and processor/tokenizer loaded successfully")
        return model, processor, model_type
        
    elif 'llava' in model_id_lower:
        model_type = 'llava'
        model_class = LlavaNextForConditionalGeneration
        processor_class = LlavaNextProcessor
        processor_kwargs = {}
        logger.info("Detected LLaVA model")
    elif 'qwen' in model_id_lower:
        model_type = 'qwen'
        model_class = Qwen3VLForConditionalGeneration
        processor_class = AutoProcessor
        processor_kwargs = {'trust_remote_code': True}
        logger.info("Detected Qwen model")
    elif 'gemma' in model_id_lower:
        model_type = 'gemma'
        try:
            from transformers import Gemma3ForConditionalGeneration, Gemma3Processor
            model_class = Gemma3ForConditionalGeneration
            processor_class = Gemma3Processor
        except ImportError:
            raise ImportError("Gemma3 dependencies not found. Please install transformers with Gemma support.")
        processor_kwargs = {}
        logger.info("Detected Gemma model")
    elif 'internvl' in model_id_lower:
        model_type = 'internvl'
        model_class = AutoModelForImageTextToText
        processor_class = AutoProcessor
        processor_kwargs = {'trust_remote_code': True}
        logger.info("Detected InternVL model")
    else:
        raise ValueError(f"Unknown model type. Model ID must contain 'llava', 'qwen', 'deepseek', 'gemma', or 'internvl': {model_id}")
    
    # Load model (same parameters for both)
    model = model_class.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map="auto",  # Auto-distribute across available devices
        attn_implementation='eager',  # Required for attention extraction
        trust_remote_code=True,
    )
    model.eval()
    
    # Log the actual dtype the model loaded in (may differ from requested)
    actual_dtype = next(model.parameters()).dtype
    if actual_dtype != dtype:
        logger.warning(f"Model loaded in {actual_dtype} (requested {dtype}). Inputs will be converted to match model dtype.")
    else:
        logger.info(f"Model loaded in {actual_dtype} as requested.")
    
    # Load processor/tokenizer
    processor = processor_class.from_pretrained(model_id, **processor_kwargs)
    
    logger.info("Model and processor/tokenizer loaded successfully")
    return model, processor, model_type


# ============================================================================
# Dataset Loading
# ============================================================================

def load_dataset(dataset_path, split_name=None):
    """Load dataset from disk."""
    logger.info(f"Loading dataset from: {dataset_path}")
    
    # If split_name is provided, try loading from subdirectory first
    if split_name:
        split_path = os.path.join(dataset_path, split_name)
        if os.path.isdir(split_path):
            logger.info(f"Loading split '{split_name}' from subdirectory: {split_path}")
            try:
                dataset = load_from_disk(split_path)
                logger.info(f"Loaded {len(dataset)} samples from split '{split_name}'")
                return dataset
            except Exception as e:
                logger.warning(f"Failed to load from subdirectory {split_path}: {e}. Trying parent directory...")
    
    # Try loading from the main path (could be DatasetDict or single Dataset)
    if os.path.isdir(dataset_path):
        try:
            dataset = load_from_disk(dataset_path)
            # If it's a DatasetDict and split_name is provided, extract the split
            if split_name and hasattr(dataset, 'keys') and split_name in dataset:
                logger.info(f"Extracting split '{split_name}' from DatasetDict")
                dataset = dataset[split_name]
            logger.info(f"Loaded {len(dataset)} samples")
            return dataset
        except Exception as e:
            logger.error(f"Failed to load dataset from {dataset_path}: {e}")
            raise
    else:
        from datasets import load_dataset as hf_load_dataset
        dataset = hf_load_dataset(dataset_path)
        if split_name and hasattr(dataset, 'keys') and split_name in dataset:
            dataset = dataset[split_name]
        logger.info(f"Loaded {len(dataset)} samples")
        return dataset


def load_npz_samples(samples_dir: Path, model_name: str = None, dataset_name: str = None) -> Dict[str, Dict[str, Any]]:
    """Load all npz sample files and index by hash_id.
    
    Args:
        samples_dir: Base directory containing npz files
        model_name: Model name subdirectory (e.g., 'deepseek-vl2')
        dataset_name: Dataset name subdirectory (e.g., 'test')
    
    Returns:
        Dict mapping hash_id to sample data
    """
    samples = {}
    
    # Try different path structures
    search_paths = []
    
    # First try: model_name/dataset_name/samples/ (most common structure)
    if model_name and dataset_name:
        search_paths.append(samples_dir / model_name / dataset_name / "samples")
    
    # Second try: model_name/dataset_name/ (npz files directly in dataset folder)
    if model_name and dataset_name:
        search_paths.append(samples_dir / model_name / dataset_name)
    
    # Third try: samples_dir directly (legacy structure)
    search_paths.append(samples_dir)
    
    # Find the first path that exists and has npz files
    npz_files = []
    actual_path = None
    for search_path in search_paths:
        if search_path.exists() and search_path.is_dir():
            found_files = list(search_path.glob("*.npz"))
            if found_files:
                npz_files = found_files
                actual_path = search_path
                break
    
    logger.info(f"Loading {len(npz_files)} npz files from {actual_path or samples_dir}")
    
    for npz_file in tqdm(npz_files, desc="Loading npz samples"):
        try:
            data = np.load(npz_file, allow_pickle=True)
            hash_id = str(data['hash_id'])
            
            samples[hash_id] = {
                'hash_id': hash_id,
                'sample_id': str(data['sample_id']),
                'question': str(data['question']),
                'answer': str(data['answer']),
                'generated_response': str(data['generated_response']),
                'is_correct': data['is_correct'].item() if isinstance(data['is_correct'], np.ndarray) else data['is_correct'],
            }
        except Exception as e:
            logger.warning(f"Error loading {npz_file.name}: {e}")
            continue
    
    logger.info(f"Successfully loaded {len(samples)} samples")
    return samples


# ============================================================================
# Hash ID Generation (matching generate_and_extract.py)
# ============================================================================

def generate_hash_id(target_dataset, question, answer):
    """Generate hash ID for a sample."""
    import hashlib
    hash_string = f"{target_dataset}[SEP]{question}[SEP]{answer}"
    hash_obj = hashlib.sha256(hash_string.encode('utf-8'))
    return hash_obj.hexdigest()


# ============================================================================
# P(True) Extraction
# ============================================================================

class PTrueExtractor:
    """Extracts P(True) confidence scores from VLMs."""
    
    def __init__(self, model, processor, model_type, debug=False):
        self.model = model
        self.processor = processor
        self.model_type = model_type
        self.debug = debug
        self.device = next(model.parameters()).device
        
        # Get token IDs for A and B
        self._setup_token_ids()
        
        # Patch DeepSeek vision encoder forward method once during initialization
        # to handle dtype mismatches (only patch if not already patched)
        if model_type == 'deepseek' and hasattr(model, 'vision') and model.vision is not None:
            if not hasattr(model.vision, '_ptrue_patched'):
                vision_dtype = next(model.vision.parameters()).dtype
                original_vision_forward = model.vision.forward
                
                def patched_vision_forward(x):
                    # Convert input to match vision encoder dtype
                    if x.dtype != vision_dtype:
                        x = x.to(dtype=vision_dtype)
                    return original_vision_forward(x)
                
                model.vision.forward = patched_vision_forward
                model.vision._ptrue_patched = True  # Mark as patched
    
    def _setup_token_ids(self):
        """Setup token IDs for choice extraction."""
        if self.model_type == 'deepseek':
            tokenizer = self.processor.tokenizer
        else:
            tokenizer = self.processor.tokenizer if hasattr(self.processor, 'tokenizer') else self.processor
        
        # Get various forms of A and B tokens
        self.choice_tokens = {}
        
        for choice in ['A', 'B', 'a', 'b']:
            try:
                # Try different tokenization patterns
                token_ids = []
                
                # Direct encoding
                direct = tokenizer.encode(choice, add_special_tokens=False)
                token_ids.extend(direct)
                
                # With space prefix
                with_space = tokenizer.encode(f' {choice}', add_special_tokens=False)
                token_ids.extend(with_space)
                
                # With parentheses
                with_paren = tokenizer.encode(f'{choice})', add_special_tokens=False)
                token_ids.extend(with_paren)
                
                self.choice_tokens[choice] = list(set(token_ids))
                
            except Exception as e:
                logger.warning(f"Error getting token IDs for '{choice}': {e}")
                self.choice_tokens[choice] = []
        
        if self.debug:
            print(f"\n🔤 CHOICE TOKEN IDS:")
            for choice, tokens in self.choice_tokens.items():
                print(f"   '{choice}': {tokens}")
                for tid in tokens:
                    token_str = tokenizer.decode([tid])
                    print(f"      {tid} -> {repr(token_str)}")
    
    def _get_best_logit(self, logits: torch.Tensor, choice: str) -> float:
        """Get the best logit value for a choice (A or B)."""
        if self.model_type == 'deepseek':
            tokenizer = self.processor.tokenizer
        else:
            tokenizer = self.processor.tokenizer if hasattr(self.processor, 'tokenizer') else self.processor
        
        # Collect all possible token IDs for this choice
        all_token_ids = set()
        
        # Add uppercase and lowercase variants
        for variant in [choice.upper(), choice.lower()]:
            if variant in self.choice_tokens:
                all_token_ids.update(self.choice_tokens[variant])
        
        if not all_token_ids:
            logger.warning(f"No token IDs found for choice '{choice}'")
            return float('-inf')
        
        # Get the maximum logit among all variants
        max_logit = float('-inf')
        best_token_id = None
        
        for token_id in all_token_ids:
            if token_id < logits.shape[0]:
                logit_val = logits[token_id].item()
                if logit_val > max_logit:
                    max_logit = logit_val
                    best_token_id = token_id
        
        if self.debug and best_token_id is not None:
            token_str = tokenizer.decode([best_token_id])
            print(f"      Best '{choice}' token: {best_token_id} -> {repr(token_str)} (logit: {max_logit:.4f})")
        
        return max_logit
    
    def extract_ptrue(self, image, question: str, generated_response: str) -> Dict[str, Any]:
        """
        Extract P(True) for a single sample.
        
        Args:
            image: PIL Image or image path
            question: Original question
            generated_response: Model's generated response
        
        Returns:
            Dict with 'ptrue', 'logit_a', 'logit_b', 'argmax_choice', 'success'
        """
        result = {
            'ptrue': None,
            'logit_a': None,
            'logit_b': None,
            'argmax_choice': None,
            'success': False
        }
        
        uncertainty_query = get_uncertainty_query()
        
        try:
            if self.model_type == 'qwen':
                result = self._extract_ptrue_qwen(image, question, generated_response, uncertainty_query)
            elif self.model_type == 'llava':
                result = self._extract_ptrue_llava(image, question, generated_response, uncertainty_query)
            elif self.model_type == 'gemma':
                result = self._extract_ptrue_gemma(image, question, generated_response, uncertainty_query)
            elif self.model_type == 'internvl':
                result = self._extract_ptrue_internvl(image, question, generated_response, uncertainty_query)
            elif self.model_type == 'deepseek':
                result = self._extract_ptrue_deepseek(image, question, generated_response, uncertainty_query)
            else:
                logger.error(f"Unknown model type: {self.model_type}")
                
        except Exception as e:
            logger.error(f"Error extracting P(True): {e}")
            if self.debug:
                import traceback
                traceback.print_exc()
        
        return result
    
    def _extract_ptrue_qwen(self, image, question, generated_response, uncertainty_query):
        """Extract P(True) for Qwen model."""
        result = {'ptrue': None, 'logit_a': None, 'logit_b': None, 'argmax_choice': None, 'success': False}
        
        # Build conversation with original Q&A
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question},
            ]},
            {"role": "assistant", "content": generated_response},
            {"role": "user", "content": uncertainty_query},
        ]
        
        # Process inputs
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)
        
        if self.debug:
            print(f"\n🔧 QWEN PTRUE EXTRACTION:")
            print(f"   Input shape: {inputs.input_ids.shape}")
        
        # Use generation mode to get logits for the first generated token
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=1,
                output_scores=True,
                return_dict_in_generate=True,
                do_sample=False,
            )
            # Extract logits from the first (and only) generated token
            logits = outputs.scores[0][0]  # scores[0] is first token, [0] is batch dimension
        
        result = self._compute_ptrue_from_logits(logits)
        return result
    
    def _extract_ptrue_llava(self, image, question, generated_response, uncertainty_query):
        """Extract P(True) for LLaVA model."""
        result = {'ptrue': None, 'logit_a': None, 'logit_b': None, 'argmax_choice': None, 'success': False}
        
        # Build conversation - LLaVA format
        # First turn: user question with image, assistant response
        # Second turn: user uncertainty query
        conversation = [
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": question + "\n\n" + APPENDED_SYSTEM_PROMPT},
            ]},
            {"role": "assistant", "content": [
                {"type": "text", "text": generated_response},
            ]},
            {"role": "user", "content": [
                {"type": "text", "text": uncertainty_query},
            ]},
        ]
        
        prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        
        # Process image
        if isinstance(image, Image.Image):
            image_to_process = image
        elif isinstance(image, str):
            if image.startswith('http://') or image.startswith('https://'):
                image_to_process = Image.open(requests.get(image, stream=True).raw)
            else:
                image_to_process = Image.open(image)
        else:
            image_to_process = image
        
        inputs = self.processor(images=image_to_process, text=prompt, return_tensors="pt")
        
        # Move to device
        device = next(self.model.parameters()).device
        model_dtype = next(self.model.parameters()).dtype
        
        inputs = {k: v.to(device=device) if isinstance(v, torch.Tensor) else v 
                  for k, v in inputs.items()}
        
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(dtype=model_dtype)
        
        if self.debug:
            print(f"\n🔧 LLAVA PTRUE EXTRACTION:")
            print(f"   Input shape: {inputs['input_ids'].shape}")
        
        # Use generation mode to get logits for the first generated token
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=1,
                output_scores=True,
                return_dict_in_generate=True,
                do_sample=False,
            )
            # Extract logits from the first (and only) generated token
            logits = outputs.scores[0][0]  # scores[0] is first token, [0] is batch dimension
        
        result = self._compute_ptrue_from_logits(logits)
        return result
    
    def _extract_ptrue_gemma(self, image, question, generated_response, uncertainty_query):
        """Extract P(True) for Gemma model."""
        result = {'ptrue': None, 'logit_a': None, 'logit_b': None, 'argmax_choice': None, 'success': False}
        
        conversation = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": question},
            ]},
            {"role": "assistant", "content": [
                {"type": "text", "text": generated_response},
            ]},
            {"role": "user", "content": [
                {"type": "text", "text": uncertainty_query},
            ]},
        ]
        
        prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        
        # Process image
        if isinstance(image, Image.Image):
            image_to_process = image
        elif isinstance(image, str):
            if image.startswith('http://') or image.startswith('https://'):
                image_to_process = Image.open(requests.get(image, stream=True).raw)
            else:
                image_to_process = Image.open(image)
        else:
            image_to_process = image
        
        inputs = self.processor(images=image_to_process, text=prompt, return_tensors="pt")
        
        device = next(self.model.parameters()).device
        model_dtype = next(self.model.parameters()).dtype
        
        inputs = {k: v.to(device=device) if isinstance(v, torch.Tensor) else v 
                  for k, v in inputs.items()}
        
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(dtype=model_dtype)
        
        if self.debug:
            print(f"\n🔧 GEMMA PTRUE EXTRACTION:")
            print(f"   Input shape: {inputs['input_ids'].shape}")
        
        # Use generation mode to get logits for the first generated token
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=1,
                output_scores=True,
                return_dict_in_generate=True,
                do_sample=False,
            )
            # Extract logits from the first (and only) generated token
            logits = outputs.scores[0][0]  # scores[0] is first token, [0] is batch dimension
        
        result = self._compute_ptrue_from_logits(logits)
        return result
    
    def _extract_ptrue_internvl(self, image, question, generated_response, uncertainty_query):
        """Extract P(True) for InternVL model."""
        result = {'ptrue': None, 'logit_a': None, 'logit_b': None, 'argmax_choice': None, 'success': False}
        
        conversation = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": question},
            ]},
            {"role": "assistant", "content": [
                {"type": "text", "text": generated_response},
            ]},
            {"role": "user", "content": [
                {"type": "text", "text": uncertainty_query},
            ]},
        ]
        
        prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        
        # Process image
        if isinstance(image, Image.Image):
            image_to_process = image
        elif isinstance(image, str):
            if image.startswith('http://') or image.startswith('https://'):
                image_to_process = Image.open(requests.get(image, stream=True).raw)
            else:
                image_to_process = Image.open(image)
        else:
            image_to_process = image
        
        inputs = self.processor(images=image_to_process, text=prompt, return_tensors="pt")
        
        device = next(self.model.parameters()).device
        model_dtype = next(self.model.parameters()).dtype
        
        inputs = {k: v.to(device=device) if isinstance(v, torch.Tensor) else v 
                  for k, v in inputs.items()}
        
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(dtype=model_dtype)
        
        if self.debug:
            print(f"\n🔧 INTERNVL PTRUE EXTRACTION:")
            print(f"   Input shape: {inputs['input_ids'].shape}")
        
        # Use generation mode to get logits for the first generated token
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=1,
                output_scores=True,
                return_dict_in_generate=True,
                do_sample=False,
            )
            # Extract logits from the first (and only) generated token
            logits = outputs.scores[0][0]  # scores[0] is first token, [0] is batch dimension
        
        result = self._compute_ptrue_from_logits(logits)
        return result
    
    def _extract_ptrue_deepseek(self, image, question, generated_response, uncertainty_query):
        """Extract P(True) for DeepSeek model."""
        result = {'ptrue': None, 'logit_a': None, 'logit_b': None, 'argmax_choice': None, 'success': False}
        
        import tempfile
        temp_image_created = False
        
        # Convert image to path if needed
        if isinstance(image, Image.Image):
            temp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
            image.save(temp_file.name)
            image_path = temp_file.name
            temp_image_created = True
        else:
            image_path = image
        
        try:
            # Build conversation - note DeepSeek uses special role markers
            # First: user question with image and assistant response
            # Second: user uncertainty query
            conversation = [
                {
                    "role": "<|User|>",
                    "content": f"<image>\n{question}",
                    "images": [image_path],
                },
                {"role": "<|Assistant|>", "content": generated_response},
                {
                    "role": "<|User|>",
                    "content": uncertainty_query,
                },
                {"role": "<|Assistant|>", "content": ""},
            ]
            
            pil_images = load_pil_images(conversation)
            prepare_inputs = self.processor(
                conversations=conversation,
                images=pil_images,
                force_batchify=True,
                system_prompt=SYSTEM_PROMPT
            ).to(self.model.device)
            
            model_dtype = next(self.model.parameters()).dtype
            
            # Convert all tensor inputs to match model dtype
            if hasattr(prepare_inputs, 'pixel_values') and prepare_inputs.pixel_values is not None:
                prepare_inputs.pixel_values = prepare_inputs.pixel_values.to(dtype=model_dtype)
            
            # Vision encoder forward method is already patched during initialization
            # No need to patch it again here (which would cause recursion)
            
            # Get input embeddings
            inputs_embeds = self.model.prepare_inputs_embeds(**prepare_inputs)
            
            if self.debug:
                print(f"\n🔧 DEEPSEEK PTRUE EXTRACTION:")
                print(f"   Input embeds shape: {inputs_embeds.shape}")
            
            # Use generation mode to get logits for the first generated token
            with torch.no_grad():
                outputs = self.model.language.generate(
                    inputs_embeds=inputs_embeds,
                    attention_mask=prepare_inputs.attention_mask,
                    pad_token_id=self.processor.tokenizer.eos_token_id,
                    bos_token_id=self.processor.tokenizer.bos_token_id,
                    eos_token_id=self.processor.tokenizer.eos_token_id,
                    max_new_tokens=1,
                    output_scores=True,
                    return_dict_in_generate=True,
                    do_sample=False,
                )
                # Extract logits from the first (and only) generated token
                logits = outputs.scores[0][0]  # scores[0] is first token, [0] is batch dimension
            
            result = self._compute_ptrue_from_logits(logits)
            
        finally:
            if temp_image_created:
                try:
                    os.unlink(image_path)
                except:
                    pass
        
        return result
    
    def _compute_ptrue_from_logits(self, logits: torch.Tensor) -> Dict[str, Any]:
        """Compute P(True) from logits."""
        result = {
            'ptrue': None,
            'logit_a': None,
            'logit_b': None,
            'argmax_choice': None,
            'success': False
        }
        
        if self.debug:
            print(f"\n🎲 COMPUTING P(TRUE) FROM LOGITS:")
            print(f"   Logits shape: {logits.shape}")
            
            # Show top 5 predictions
            top_5_logits, top_5_indices = torch.topk(logits, 5)
            if self.model_type == 'deepseek':
                tokenizer = self.processor.tokenizer
            else:
                tokenizer = self.processor.tokenizer if hasattr(self.processor, 'tokenizer') else self.processor
            
            print(f"\n📊 TOP 5 PREDICTIONS:")
            for i, (logit_val, token_id) in enumerate(zip(top_5_logits, top_5_indices)):
                token_text = tokenizer.decode([token_id.item()])
                marker = " ⭐ ARGMAX" if i == 0 else ""
                print(f"   {i+1}. ID: {token_id.item():6d}, Logit: {logit_val.item():8.4f}, Token: {repr(token_text)}{marker}")
        
        # Get logits for A (no) and B (yes)
        logit_a = self._get_best_logit(logits, 'A')
        logit_b = self._get_best_logit(logits, 'B')
        
        if logit_a == float('-inf') or logit_b == float('-inf'):
            logger.warning("Could not find token IDs for A or B")
            return result
        
        result['logit_a'] = logit_a
        result['logit_b'] = logit_b
        
        # Compute P(True) using softmax
        # A = no (False), B = yes (True)
        # P(True) = P(B) = softmax([logit_A, logit_B])[1]
        logits_tensor = torch.tensor([logit_a, logit_b], dtype=torch.float32)
        probs = F.softmax(logits_tensor, dim=0)
        
        ptrue = float(probs[1])  # P(B) = P(True)
        result['ptrue'] = ptrue
        result['success'] = True
        
        # Determine argmax choice
        result['argmax_choice'] = 'B' if logit_b > logit_a else 'A'
        
        if self.debug:
            print(f"\n🎯 P(TRUE) COMPUTATION:")
            print(f"   Logit A (no):  {logit_a:.4f}")
            print(f"   Logit B (yes): {logit_b:.4f}")
            print(f"   P(A) = P(False): {probs[0]:.4f}")
            print(f"   P(B) = P(True):  {probs[1]:.4f}")
            print(f"   Argmax choice: {result['argmax_choice']}")
        
        return result


# ============================================================================
# Evaluation Metrics (using utils/eval.py)
# ============================================================================
# Note: Evaluation functions are imported from utils/eval.py


# ============================================================================
# Main Evaluation
# ============================================================================

class PTrueEvaluator:
    """Main evaluator class for P(True) evaluation."""
    
    def __init__(self, model, processor, model_type, args):
        self.model = model
        self.processor = processor
        self.model_type = model_type
        self.args = args
        self.extractor = PTrueExtractor(model, processor, model_type, debug=args.debug)
    
    def evaluate(self, dataset, npz_samples: Dict[str, Dict[str, Any]], output_dir: Path) -> Dict[str, Any]:
        """Run P(True) evaluation."""
        logger.info("Starting P(True) evaluation...")
        
        # Setup output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        labels_path = output_dir / "test_labels.json"
        
        # Load existing records if resuming
        existing_records = []
        processed_hash_ids = set()
        if self.args.resume:
            if labels_path.exists():
                logger.info(f"Resume mode: Loading existing results from {labels_path}")
                try:
                    with open(labels_path, 'r', encoding='utf-8') as f:
                        existing_records = json.load(f)
                    processed_hash_ids = {record['hash_id'] for record in existing_records}
                    logger.info(f"Loaded {len(existing_records)} existing records. {len(processed_hash_ids)} hash_ids already processed.")
                except Exception as e:
                    logger.warning(f"Failed to load existing results: {e}. Starting fresh.")
                    existing_records = []
                    processed_hash_ids = set()
            else:
                logger.info("Resume mode enabled but no existing results found. Starting fresh.")
        
        evaluation_records = existing_records.copy()  # Start with existing records
        new_records_count = 0
        skipped_already_processed = 0
        processed_count = len(existing_records)  # Count includes existing
        # Setup output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        labels_path = output_dir / "test_labels.json"
        
        # Statistics (include existing records)
        ptrue_values = [r['confidence_score'] for r in existing_records]
        logit_a_values = [r['logit_a'] for r in existing_records]
        logit_b_values = [r['logit_b'] for r in existing_records]
        
        # Main evaluation loop
        num_samples = len(dataset)
        skipped_no_npz = 0
        skipped_no_correctness = 0
        failed_extractions = 0
        
        # Debug output for first few new samples
        show_debug = self.args.debug and new_records_count < self.args.debug_samples
        
        for idx in tqdm(range(num_samples), desc="Evaluating P(True)"):
            sample = dataset[idx]
            
            # Extract sample data
            hash_id = sample.get('hash_id')
            if not hash_id:
                # Generate hash_id if not present
                hash_id = generate_hash_id(
                    self.args.test_dataset_name if hasattr(self.args, 'test_dataset_name') else 'unknown',
                    sample.get('question', ''),
                    sample.get('answer', '')
                )
            
            # Skip if already processed (resume mode)
            if hash_id in processed_hash_ids:
                skipped_already_processed += 1
                continue
            
            # Get npz data (required for generated_response)
            if hash_id not in npz_samples:
                skipped_no_npz += 1
                if skipped_no_npz <= 5:
                    logger.warning(f"Sample {hash_id} not found in npz samples. Skipping.")
                continue
            
            npz_data = npz_samples[hash_id]
            question = sample.get('question', npz_data.get('question', ''))
            answer = sample.get('answer', npz_data.get('answer', ''))
            generated_response = npz_data.get('generated_response', '')
            
            # Get is_correct from npz files if available, otherwise from dataset
            is_correct = npz_data.get('is_correct')
            if is_correct is None:
                is_correct = sample.get('is_correct')
            
            # Skip if no correctness assessment
            if is_correct is None:
                skipped_no_correctness += 1
                continue
            
            # Get image from dataset
            image = sample.get('image')
            if image is None:
                logger.warning(f"Sample {hash_id} has no image. Skipping.")
                continue
            
            # Convert image to PIL if needed
            if isinstance(image, Image.Image):
                image_pil = image
            elif isinstance(image, dict) and 'bytes' in image:
                image_pil = Image.open(io.BytesIO(image['bytes']))
            elif isinstance(image, str):
                if image.startswith('http://') or image.startswith('https://'):
                    image_pil = Image.open(requests.get(image, stream=True).raw)
                else:
                    image_pil = Image.open(image)
            else:
                try:
                    image_pil = Image.open(io.BytesIO(image))
                except:
                    logger.warning(f"Could not process image for sample {hash_id}. Skipping.")
                    continue
            
            # Extract P(True)
            if show_debug:
                print(f"\n🔬 NEW SAMPLE {new_records_count + 1} (Total: {processed_count + 1})")
                print(f"   Hash ID: {hash_id}")
                print(f"   Question: {question[:100]}...")
                print(f"   Answer: {answer[:100]}...")
                print(f"   Generated Response: {generated_response[:100]}...")
            
            try:
                ptrue_result = self.extractor.extract_ptrue(image_pil, question, generated_response)
                
                if not ptrue_result.get('success', False):
                    failed_extractions += 1
                    logger.warning(f"Failed to extract P(True) for sample {hash_id}")
                    continue
                
                confidence_score = ptrue_result.get('ptrue', 0.0)
                logit_a = ptrue_result.get('logit_a', 0.0)
                logit_b = ptrue_result.get('logit_b', 0.0)
                
                # Create evaluation record
                evaluation_records.append({
                    'hash_id': hash_id,
                    'sample_id': npz_data.get('sample_id', hash_id),
                    'ground_truth_correctness': int(bool(is_correct)),
                    'confidence_score': float(confidence_score),
                    'logit_a': float(logit_a),
                    'logit_b': float(logit_b),
                    'argmax_choice': ptrue_result.get('argmax_choice', 'unknown'),
                    'dataset': sample.get('dataset', npz_data.get('dataset', 'unknown')),
                })
                
                new_records_count += 1
                processed_count += 1
                
                # Save incrementally every 100 new records (safety measure)
                if new_records_count > 0 and new_records_count % 100 == 0:
                    try:
                        # Save to temporary file first, then rename (atomic operation)
                        temp_labels_path = labels_path.with_suffix('.json.tmp')
                        with open(temp_labels_path, 'w', encoding='utf-8') as f:
                            json.dump(evaluation_records, f, indent=2, ensure_ascii=False)
                        temp_labels_path.replace(labels_path)
                        logger.info(f"Incremental save: {len(evaluation_records)} records saved ({new_records_count} new)")
                    except Exception as e:
                        logger.warning(f"Failed to save incrementally: {e}")
                
                # Clear CUDA cache periodically to prevent OOM
                if new_records_count % 50 == 0:
                    gc.collect()
                    torch.cuda.empty_cache()
                    
            except Exception as e:
                import traceback
                logger.error(f"Error processing sample {hash_id}: {e}")
                if self.args.debug:
                    traceback.print_exc()
                failed_extractions += 1
                continue
        
        logger.info(f"Completed evaluation of {len(evaluation_records)} samples")
        if skipped_no_npz > 0:
            logger.warning(f"Skipped {skipped_no_npz} samples due to missing npz data")
        if skipped_no_correctness > 0:
            logger.warning(f"Skipped {skipped_no_correctness} samples due to missing 'is_correct' field")
        if failed_extractions > 0:
            logger.warning(f"Failed to extract P(True) from {failed_extractions} samples")
        
        # Save evaluation records (atomic write: temp file then rename)
        try:
            temp_labels_path = labels_path.with_suffix('.json.tmp')
            with open(temp_labels_path, 'w', encoding='utf-8') as f:
                json.dump(evaluation_records, f, indent=2, ensure_ascii=False)
            temp_labels_path.replace(labels_path)
            logger.info(f"Evaluation records saved to {labels_path} ({len(evaluation_records)} total records)")
        except Exception as e:
            logger.error(f"Failed to save evaluation records: {e}")
            # If atomic save fails, try direct save as fallback
            try:
                with open(labels_path, 'w', encoding='utf-8') as f:
                    json.dump(evaluation_records, f, indent=2, ensure_ascii=False)
                logger.info(f"Evaluation records saved (fallback method) to {labels_path}")
            except Exception as e2:
                logger.error(f"Fallback save also failed: {e2}")
                return None
        
        # Calculate statistics from evaluation records
        ptrue_values = [r['confidence_score'] for r in evaluation_records]
        logit_a_values = [r['logit_a'] for r in evaluation_records]
        logit_b_values = [r['logit_b'] for r in evaluation_records]
        
        # Log summary
        logger.info(f"  Total successfully processed: {processed_count}")
        if self.args.resume:
            logger.info(f"    - Existing records: {len(existing_records)}")
            logger.info(f"    - New records: {new_records_count}")
            logger.info(f"  Skipped (already processed): {skipped_already_processed}")
        
        # Calculate metrics from evaluation records
        labels = np.array([r['ground_truth_correctness'] for r in evaluation_records], dtype=float)
        scores = np.array([r['confidence_score'] for r in evaluation_records], dtype=float)

        metrics = calculate_all_metrics(labels, scores)

        results = {
            'overall': {
                'n_samples': len(evaluation_records),
                'n_total_samples': num_samples,
                **metrics
            },
            'metadata': {
                'model_name': self.args.model_id,
                'model_name_part': self.args.model_id.split("/")[-1],
                'test_dataset_name': self.args.test_dataset_name,
                'total_records': len(evaluation_records),
                'evaluation_timestamp': datetime.now().isoformat(),
                'ptrue_statistics': {
                    'avg_confidence': float(np.mean(scores)),
                    'confidence_std': float(np.std(scores)),
                    'min_confidence': float(np.min(scores)),
                    'max_confidence': float(np.max(scores)),
                },
            }
        }

        results_path = output_dir / "test_results.json"
        save_evaluation_results(results, str(results_path))
        logger.info(f"Evaluation results saved to {results_path}")

        return results


# ============================================================================
# Main Function
# ============================================================================

def main():
    args = parse_args()
    
    logger.info(f"Dataset path: {args.dataset_path}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"GPU IDs: {args.gpu_ids}")
    logger.info(f"Debug mode: {args.debug}")
    
    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")
    
    # Load model and processor
    logger.info(f"Loading model: {args.model_id}")
    model, processor, model_type = load_model_and_processor(args.model_id, args.dtype)
    
    # Load dataset
    logger.info(f"Loading dataset: {args.dataset_path}")
    dataset = load_dataset(args.dataset_path, split_name=args.test_dataset_name)
    
    # Setup output directory
    model_name_part = args.model_id.split('/')[-1]
    output_dir = Path(args.output_dir) / model_name_part / args.test_dataset_name
    
    # Load npz samples (look in model_name/dataset_name/samples/ subdirectory)
    logger.info(f"Loading npz samples from: {args.data_dir}")
    npz_samples = load_npz_samples(Path(args.data_dir), model_name=model_name_part, dataset_name=args.test_dataset_name)
    
    # Run evaluation
    evaluator = PTrueEvaluator(model, processor, model_type, args)
    results = evaluator.evaluate(dataset, npz_samples, output_dir)
    
    if results:
        print(f"\n✅ Evaluation complete!")
        print(f"📁 Results saved to: {output_dir}")
        print("\nGenerated files:")
        print("  - test_labels.json: Records with ground truth and P(True) scores")


if __name__ == "__main__":
    main()

# Example usage:
# CUDA_VISIBLE_DEVICES=5,6 python PTRUE_eval.py \
#     --model_id "deepseek-ai/deepseek-vl2" \
#     --data_dir "../data/extraction/raw/" \
#     --dataset_path "../data/VLCB/raw" \
#     --test_dataset_name "test" \
#     --output_dir "../results/PTRUE" \
#     --gpu_ids "5,6" \
#     --debug \
#     --debug_samples 5

# For Gemma:
# python PTRUE_eval.py \
#     --model_id "google/gemma-3-27b-it" \
#     --data_dir "../data/extraction/raw/" \
#     --dataset_path "../data/VLCB/raw" \
#     --test_dataset_name "test" \
#     --output_dir "../results/PTRUE" \
#     --gpu_ids "0" \
#     --debug \
#     --debug_samples 5

# For DeepSeek VL2:
# CUDA_VISIBLE_DEVICES=0,1 python PTRUE_eval.py \
#     --model_id "deepseek-ai/deepseek-vl2" \
#     --data_dir "../data/extraction/raw/" \
#     --dataset_path "../data/VLCB/raw" \
#     --test_dataset_name "test" \
#     --output_dir "../results/PTRUE" \
#     --gpu_ids "0,1" \
#     --dtype "bfloat16"

# For Qwen:
# CUDA_VISIBLE_DEVICES=0 python PTRUE_eval.py \
#     --model_id "Qwen/Qwen2.5-VL-3B-Instruct" \
#     --data_dir "../data/extraction/raw/" \
#     --dataset_path "../data/VLCB/raw" \
#     --test_dataset_name "test" \
#     --output_dir "../results/PTRUE" \
#     --gpu_ids "0" \
#     --debug \
#     --debug_samples 5

# For InternVL:
# CUDA_VISIBLE_DEVICES=0,1 python PTRUE_eval.py \
#     --model_id "OpenGVLab/InternVL3_5-14B-HF" \
#     --data_dir "../data/extraction/raw/" \
#     --dataset_path "../data/VLCB/raw" \
#     --test_dataset_name "test" \
#     --output_dir "../results/PTRUE" \
#     --gpu_ids "0,1"