#!/usr/bin/env python3
"""
Self-Probing Evaluation Script for Vision-Language Models.

This script implements the self-probing confidence estimation method for VLMs:
1. Loads generated responses from npz files (from generate_and_extract.py)
2. Reconstructs the original conversation
3. Appends a self-evaluation query asking the model to assess its own answer
4. Extracts the confidence score from the model's response
5. Evaluates and saves results similar to PTRUE_eval.py

Supports both HuggingFace and vLLM backends.
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

# Set vLLM worker multiprocessing method (must be set before importing vLLM)
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

# Now import all other libraries
import gc
import json
import sys
import re
import numpy as np
import torch
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from tqdm import tqdm
from datetime import datetime
import logging
from PIL import Image
import requests
import io
import tempfile
import traceback as tb_module

# OpenAI import for GPT fallback judge
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

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

# vLLM imports (optional)
try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False

from datasets import load_from_disk

# Add utils directory to path for eval functions
sys.path.append(str(Path(__file__).parent / "../utils"))
try:
    from eval import calculate_all_metrics, save_evaluation_results
except ImportError:
    # Fallback if utils not found - define minimal versions
    def calculate_all_metrics(labels, scores):
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, brier_score_loss
        predictions = (scores >= 0.5).astype(int)
        return {
            'accuracy': float(accuracy_score(labels, predictions)),
            'precision': float(precision_score(labels, predictions, zero_division=0)),
            'recall': float(recall_score(labels, predictions, zero_division=0)),
            'f1': float(f1_score(labels, predictions, zero_division=0)),
            'sensitivity': float(recall_score(labels, predictions, zero_division=0)),
            'specificity': float(recall_score(1-labels, 1-predictions, zero_division=0)),
            'ece': 0.0,  # Placeholder
            'brier': float(brier_score_loss(labels, scores)),
            'auroc': float(roc_auc_score(labels, scores)) if len(np.unique(labels)) > 1 else 0.0,
            'aucpr': float(average_precision_score(labels, scores)) if len(np.unique(labels)) > 1 else 0.0,
        }
    
    def save_evaluation_results(results, path):
        with open(path, 'w') as f:
            json.dump(results, f, indent=2)

# Create logs directory if it doesn't exist
os.makedirs("logs", exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/self_probing_eval.log", mode='a')
    ]
)
logger = logging.getLogger(__name__)


# ============================================================================
# Constants
# ============================================================================

MAX_IMAGE_DIMENSION = 2048  # Default maximum allowed dimension
SYSTEM_PROMPT = "You are a vision language assistant. Provide brief, complete answers."
APPENDED_SYSTEM_PROMPT = "Provide a brief, complete answer."

# vLLM Model configurations - only need to specify if model needs special handling
VLLM_MODEL_CONFIGS = {
    "deepseek-ai/deepseek-vl2": {
        "hf_overrides": {"architectures": ["DeepseekVLV2ForCausalLM"]},
    }
}


# ============================================================================
# Image Preprocessing (matching generate_and_extract.py)
# ============================================================================

def resize_image_if_needed(img, max_dim=MAX_IMAGE_DIMENSION):
    """
    Resize image if either dimension exceeds max_dim, preserving aspect ratio.
    No upsampling - only downsample large images.
    
    Args:
        img: PIL Image, bytes, or dict with 'bytes'
        max_dim: Maximum allowed dimension (default 2048)
    
    Returns:
        PIL Image (resized if needed, original otherwise)
    """
    # Convert to PIL Image
    if isinstance(img, Image.Image):
        pil_img = img
    elif isinstance(img, bytes):
        pil_img = Image.open(io.BytesIO(img))
    elif isinstance(img, dict) and 'bytes' in img:
        pil_img = Image.open(io.BytesIO(img['bytes']))
    else:
        # Unknown format, return as-is
        return img
    
    # Ensure RGB mode
    if pil_img.mode != 'RGB':
        pil_img = pil_img.convert('RGB')
    
    w, h = pil_img.size
    max_current = max(w, h)
    
    # Only resize if LARGER than limit (no upsampling)
    if max_current <= max_dim:
        return pil_img  # Return unchanged
    
    # Print resizing notice
    logger.debug(f"Resizing image from {w}x{h}")
    
    # Calculate new dimensions preserving aspect ratio
    scale = max_dim / max_current
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    # Use LANCZOS for high-quality downsampling
    resized = pil_img.resize((new_w, new_h), Image.LANCZOS)
    
    logger.debug(f"Resized image from {w}x{h} to {new_w}x{new_h}")
    
    return resized


def load_and_preprocess_image(image, max_dim=MAX_IMAGE_DIMENSION):
    """
    Load and preprocess image from various formats.
    
    Args:
        image: PIL Image, path string, URL, bytes, or dict with 'bytes'
        max_dim: Maximum dimension for resizing
    
    Returns:
        PIL Image (preprocessed)
    """
    # Handle different input types
    if isinstance(image, Image.Image):
        pil_img = image
    elif isinstance(image, str):
        if image.startswith('http://') or image.startswith('https://'):
            pil_img = Image.open(requests.get(image, stream=True).raw)
        else:
            pil_img = Image.open(image)
    elif isinstance(image, bytes):
        pil_img = Image.open(io.BytesIO(image))
    elif isinstance(image, dict) and 'bytes' in image:
        pil_img = Image.open(io.BytesIO(image['bytes']))
    else:
        raise ValueError(f"Unknown image type: {type(image)}")
    
    # Convert to RGB and resize
    if pil_img.mode != 'RGB':
        pil_img = pil_img.convert('RGB')
    
    return resize_image_if_needed(pil_img, max_dim)


# ============================================================================
# Self-Probing Query Generation
# ============================================================================

def get_self_probing_query(question: str, generated_answer: str) -> str:
    """Get the self-probing query for confidence evaluation."""
    return (
        f"Question: {question}\n"
        f"Possible Answer: {generated_answer}\n\n"
        f"Q: How likely is the above answer to be correct? give your confidence in the following format:\n"
        f"Confidence: <number from 0 to 100>%\n"
        f"Note: The confidence indicates how likely you think the answer is true."
    )


# ============================================================================
# Confidence Extraction from Response
# ============================================================================

def extract_confidence_from_response(response: str) -> Optional[float]:
    """
    Extract confidence score from model's self-probing response.
    
    Looks for patterns like:
    - "Confidence: 85%"
    - "Confidence: 85"
    - "confidence: 85%"
    - "85%"
    
    Returns:
        Confidence as a float between 0 and 1, or None if not found
    """
    # Pattern 1: "Confidence: XX%" or "Confidence: XX"
    pattern1 = r'[Cc]onfidence:\s*(\d+(?:\.\d+)?)\s*%?'
    match = re.search(pattern1, response)
    if match:
        confidence = float(match.group(1))
        # Normalize to 0-1 range
        if confidence > 1.0:
            confidence = confidence / 100.0
        return min(max(confidence, 0.0), 1.0)
    
    # Pattern 2: Just a percentage "XX%"
    pattern2 = r'(\d+(?:\.\d+)?)\s*%'
    matches = re.findall(pattern2, response)
    if matches:
        # Take the last percentage mentioned
        confidence = float(matches[-1])
        if confidence > 1.0:
            confidence = confidence / 100.0
        return min(max(confidence, 0.0), 1.0)
    
    # Pattern 3: Look for numbers between 0-100 near end of response
    pattern3 = r'(\d+(?:\.\d+)?)'
    matches = re.findall(pattern3, response[-200:])  # Look in last 200 chars
    if matches:
        for match_str in reversed(matches):  # Start from end
            num = float(match_str)
            if 0 <= num <= 100:
                confidence = num / 100.0 if num > 1.0 else num
                return min(max(confidence, 0.0), 1.0)
    
    return None


# ============================================================================
# GPT Fallback Judge for Failed Confidence Extraction
# ============================================================================

def create_openai_client(api_key=None):
    """Create OpenAI client for GPT fallback."""
    if not OPENAI_AVAILABLE:
        logger.error("OpenAI package not installed. Cannot use GPT fallback.")
        return None
    if api_key is None:
        api_key = os.getenv("OPENAI_API_KEY")
    if api_key is None:
        logger.error("No OpenAI API key provided for GPT fallback.")
        return None
    return OpenAI(api_key=api_key)


def gpt_judge_confidence(raw_response: str, question: str, generated_answer: str,
                         client, gpt_model: str = 'gpt-5-mini',
                         reasoning_effort: str = 'medium') -> Optional[float]:
    """
    Use GPT as a fallback judge to extract a confidence score from a VLM's
    self-probing response that couldn't be parsed by regex.

    Args:
        raw_response: The VLM's raw self-probing output we couldn't parse
        question: The original question
        generated_answer: The VLM's original answer
        client: OpenAI client instance
        gpt_model: GPT model to use
        reasoning_effort: Reasoning effort level

    Returns:
        Confidence as float 0-1, or None on failure
    """
    if client is None:
        return None

    system_prompt = (
        "You are an expert judge. A vision-language model was asked to rate its own confidence "
        "in its answer on a scale of 0 to 100%. The model produced a response, but we could not "
        "automatically extract a numeric confidence value from it.\n\n"
        "Your task: Read the model's self-assessment response and determine the confidence score "
        "(0-100) that the response is trying to convey. If the response expresses high certainty, "
        "give a high score. If it expresses doubt or uncertainty, give a low score. If it refuses "
        "to answer or is completely irrelevant, give 50 (maximum uncertainty).\n\n"
        "Return ONLY a single integer between 0 and 100, nothing else."
    )

    user_content = (
        f"Original Question: {question}\n\n"
        f"Model's Answer: {generated_answer}\n\n"
        f"Model's Self-Assessment Response (from which we need a confidence score):\n"
        f"{raw_response}\n\n"
        f"What confidence score (0-100) does this response convey?"
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]

    try:
        response = client.responses.create(
            model=gpt_model,
            reasoning={"effort": reasoning_effort},
            input=messages,
            max_output_tokens=10000,
        )
        output_text = response.output_text.strip()

        # Extract number from GPT response
        numbers = re.findall(r'(\d+(?:\.\d+)?)', output_text)
        if numbers:
            score = float(numbers[0])
            if score > 1.0:
                score = score / 100.0
            return min(max(score, 0.0), 1.0)

        logger.warning(f"GPT fallback returned unparseable response: {output_text}")
        return None

    except Exception as e:
        logger.error(f"GPT fallback failed: {e}")
        return None


# ============================================================================
# Argument Parser
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description='Self-Probing Evaluation for Vision-Language Models',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model configuration
    parser.add_argument('--model_id', type=str, required=True,
                        help='HuggingFace model identifier (e.g., Qwen/Qwen2.5-VL-3B-Instruct)')
    parser.add_argument('--gpu_ids', type=str, default='0',
                        help='GPU IDs to use (comma-separated)')
    parser.add_argument('--dtype', type=str, default='float32',
                        choices=['float32', 'float16', 'bfloat16'],
                        help='Model dtype (applies to both HF and vLLM backends)')
    
    # Backend configuration
    parser.add_argument('--backend', type=str, default='hf',
                        choices=['hf', 'vllm'],
                        help='Inference backend: hf (HuggingFace) or vllm')
    parser.add_argument('--tensor_parallel_size', type=int, default=1,
                        help='Tensor parallel size for vLLM (number of GPUs)')
    parser.add_argument('--gpu_memory_utilization', type=float, default=0.9,
                        help='GPU memory utilization for vLLM (0.0-1.0)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for HuggingFace inference (vLLM always processes entire dataset as one batch)')
    
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
    parser.add_argument('--max_new_tokens', type=int, default=128,
                        help='Maximum tokens to generate for self-probing response')
    parser.add_argument('--max_image_dim', type=int, default=MAX_IMAGE_DIMENSION,
                        help='Maximum image dimension (images larger than this will be resized)')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug mode with verbose output')
    parser.add_argument('--debug_samples', type=int, default=5,
                        help='Number of samples to show detailed debug info')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from existing results')

    # GPT fallback configuration
    parser.add_argument('--gpt_fallback', action='store_true',
                        help='Use GPT as fallback judge when confidence extraction fails')
    parser.add_argument('--openai_api_key', type=str, default=None,
                        help='OpenAI API key for GPT fallback (defaults to OPENAI_API_KEY env var)')
    parser.add_argument('--gpt_model', type=str, default='gpt-5-mini',
                        help='GPT model for fallback confidence judging')
    parser.add_argument('--gpt_reasoning_effort', type=str, default='medium',
                        choices=['low', 'medium', 'high'],
                        help='Reasoning effort for GPT fallback')
    parser.add_argument('--gpt_n_parallel', type=int, default=20,
                        help='Number of parallel GPT fallback requests')

    return parser.parse_args()


# ============================================================================
# Dataset Loading
# ============================================================================

def load_dataset(dataset_path):
    """Load dataset from disk."""
    logger.info(f"Loading dataset from: {dataset_path}")
    
    if os.path.isdir(dataset_path):
        dataset = load_from_disk(dataset_path)
    else:
        from datasets import load_dataset as hf_load_dataset
        dataset = hf_load_dataset(dataset_path)
    
    logger.info(f"Loaded {len(dataset)} samples")
    return dataset


def load_npz_samples(samples_dir: Path) -> Dict[str, Dict[str, Any]]:
    """Load all npz sample files and index by hash_id."""
    samples = {}
    npz_files = list(samples_dir.glob("*.npz"))
    
    logger.info(f"Loading {len(npz_files)} npz files from {samples_dir}")
    
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
# HuggingFace Model Loading (matching generate_and_extract.py)
# ============================================================================

def load_hf_model_and_processor(model_id, dtype_str, device):
    """Load VLM model and processor/tokenizer using HuggingFace."""
    logger.info(f"Loading HF model: {model_id}")
    logger.info(f"Dtype: {dtype_str}")
    logger.info(f"Device: {device}")
    
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

        # DeepSeek-VL2 is designed to run in bfloat16
        if dtype_str != 'bfloat16':
            logger.warning(
                f"DeepSeek VL2 requires bfloat16 for stable vision encoder behaviour. "
                f"Overriding requested dtype '{dtype_str}' to 'bfloat16'."
            )
        dtype_str = 'bfloat16'
        dtype = torch.bfloat16
        logger.info(f"Using dtype: {dtype_str} for DeepSeek VL2")
        
        # Load DeepSeek processor
        processor = DeepseekVLV2Processor.from_pretrained(model_id)
        
        # Load DeepSeek model - load first, then convert dtype (matches working diagnostics script)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
        )
        # Convert entire model to bfloat16, then move to CUDA
        model = model.to(dtype).cuda().eval()
        
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
        model_class = Gemma3ForConditionalGeneration
        processor_class = Gemma3Processor
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
    
    # Set max_memory per device
    max_memory = {
        0: "100GiB",
        1: "130GiB",
        2: "130GiB",
        3: "130GiB",
        4: "130GiB",
        5: "130GiB",
        6: "130GiB",
    }
    
    # Load model
    model = model_class.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map="auto",
        max_memory=max_memory,
        attn_implementation='eager',
        trust_remote_code=True,
    )
    model.eval()
    
    # Log the actual dtype
    actual_dtype = next(model.parameters()).dtype
    if actual_dtype != dtype:
        logger.warning(f"Model loaded in {actual_dtype} (requested {dtype}). Inputs will be converted to match model dtype.")
    else:
        logger.info(f"Model loaded in {actual_dtype} as requested.")
    
    # Load processor
    processor = processor_class.from_pretrained(model_id, **processor_kwargs)
    
    logger.info("Model and processor loaded successfully")
    return model, processor, model_type


# ============================================================================
# vLLM Model Loading
# ============================================================================

class VLLMInference:
    """Wrapper for vLLM VLM inference with automatic prompt formatting."""
    
    def __init__(
        self,
        model_name: str,
        dtype: str = 'float32',
        gpu_memory_utilization: float = 0.9,
        tensor_parallel_size: int = 1,
        max_model_len: Optional[int] = None,
        seed: int = 42,
    ):
        if not VLLM_AVAILABLE:
            raise ImportError("vLLM not available. Please install with: pip install vllm")
        
        self.model_name = model_name
        
        # Get model-specific config
        model_config = VLLM_MODEL_CONFIGS.get(model_name, {})
        
        # Load processor for chat template
        logger.info(f"Loading processor for {model_name}...")
        self.processor = AutoProcessor.from_pretrained(
            model_name, 
            trust_remote_code=True
        )
        
        # Determine model type
        model_id_lower = model_name.lower()
        if 'deepseek' in model_id_lower and 'vl' in model_id_lower:
            self.model_type = 'deepseek'
        elif 'llava' in model_id_lower:
            self.model_type = 'llava'
        elif 'qwen' in model_id_lower:
            self.model_type = 'qwen'
        elif 'gemma' in model_id_lower:
            self.model_type = 'gemma'
        elif 'internvl' in model_id_lower:
            self.model_type = 'internvl'
        else:
            self.model_type = 'generic'
        
        # Map dtype string to vLLM dtype
        dtype_map = {
            'float32': 'float32',
            'float16': 'float16',
            'bfloat16': 'bfloat16',
        }
        vllm_dtype = dtype_map.get(dtype, 'float32')
        
        # Initialize vLLM
        logger.info(f"Loading vLLM model {model_name}...")
        logger.info(f"Dtype: {dtype} (vLLM dtype: {vllm_dtype})")
        llm_kwargs = {
            "model": model_name,
            "trust_remote_code": True,
            "disable_log_stats": True,
            "dtype": vllm_dtype,
            "gpu_memory_utilization": gpu_memory_utilization,
            "tensor_parallel_size": tensor_parallel_size,
            "seed": seed,
            "limit_mm_per_prompt": {"image": 1},
        }
        
        # Add model-specific overrides (e.g., DeepSeek-VL2 architecture)
        if "hf_overrides" in model_config:
            llm_kwargs["hf_overrides"] = model_config["hf_overrides"]
        
        if max_model_len is not None:
            llm_kwargs["max_model_len"] = max_model_len
            
        self.llm = LLM(**llm_kwargs)
        logger.info("vLLM model loaded successfully!")
    
    def _build_messages(self, image: Image.Image, question: str) -> list:
        """Build OpenAI-style messages with image."""
        return [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": question},
                ],
            }
        ]
    
    def _apply_chat_template(self, messages: list) -> str:
        """Apply model's chat template to get formatted prompt."""
        return self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    
    def single_inference(
        self,
        image: Image.Image,
        question: str,
        max_tokens: int = 128,
        temperature: float = 0.0,
    ) -> str:
        """Run inference on a single image-question pair."""
        # Build and format prompt
        messages = self._build_messages(image, question)
        prompt = self._apply_chat_template(messages)
        
        # Generate
        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
        )
        
        outputs = self.llm.generate(
            {"prompt": prompt, "multi_modal_data": {"image": image}},
            sampling_params=sampling_params,
        )
        
        return outputs[0].outputs[0].text
    
    def batch_inference(
        self,
        images: List[Image.Image],
        questions: List[str],
        max_tokens: int = 128,
        temperature: float = 0.0,
        use_tqdm: bool = True,
    ) -> List[str]:
        """
        Run batch inference on multiple image-question pairs.
        
        Args:
            images: List of PIL Images
            questions: List of questions (same length as images)
            max_tokens: Max tokens to generate
            temperature: Sampling temperature
            use_tqdm: Show progress bar
            
        Returns:
            List of generated answers
        """
        assert len(images) == len(questions), \
            f"Images and questions must have same length: {len(images)} != {len(questions)}"
        
        # Prepare all inputs
        inputs = []
        for img, question in zip(images, questions):
            # Build and format prompt
            messages = self._build_messages(img, question)
            prompt = self._apply_chat_template(messages)
            
            inputs.append({
                "prompt": prompt,
                "multi_modal_data": {"image": img},
            })
        
        # Sampling params
        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
        )
        
        # Batch generate
        outputs = self.llm.generate(
            inputs,
            sampling_params=sampling_params,
            use_tqdm=use_tqdm,
        )
        
        return [output.outputs[0].text for output in outputs]


# ============================================================================
# HuggingFace Self-Probing Extractor
# ============================================================================

class HFSelfProbingExtractor:
    """Extracts self-probing confidence scores from VLMs using HuggingFace."""
    
    def __init__(self, model, processor, model_type, max_new_tokens=128, debug=False, max_image_dim=MAX_IMAGE_DIMENSION):
        self.model = model
        self.processor = processor
        self.model_type = model_type
        self.max_new_tokens = max_new_tokens
        self.debug = debug
        self.max_image_dim = max_image_dim
        self.device = next(model.parameters()).device
    
    def extract_confidence(self, image, question: str, generated_response: str, sample_id: str = None) -> Dict[str, Any]:
        """
        Extract self-probing confidence for a single sample.
        """
        result = {
            'confidence': None,
            'raw_response': None,
            'success': False
        }
        
        # Preprocess image
        image = load_and_preprocess_image(image, max_dim=self.max_image_dim)
        
        # Build self-probing query
        probing_query = get_self_probing_query(question, generated_response)
        
        try:
            if self.model_type == 'qwen':
                result = self._extract_confidence_qwen(image, probing_query)
            elif self.model_type == 'llava':
                result = self._extract_confidence_llava(image, probing_query)
            elif self.model_type == 'gemma':
                result = self._extract_confidence_gemma(image, probing_query)
            elif self.model_type == 'internvl':
                result = self._extract_confidence_internvl(image, probing_query)
            elif self.model_type == 'deepseek':
                result = self._extract_confidence_deepseek(image, probing_query)
            else:
                logger.error(f"Unknown model type: {self.model_type}")
                
        except torch.cuda.OutOfMemoryError as e:
            import traceback
            sample_context = f" (sample_id: {sample_id})" if sample_id else ""
            logger.error(f"CUDA OOM error{sample_context}: {str(e)}")
            logger.error(f"Full traceback:\n{traceback.format_exc()}")
            torch.cuda.empty_cache()
            gc.collect()
        except Exception as e:
            import traceback
            sample_context = f" (sample_id: {sample_id})" if sample_id else ""
            logger.error(f"Error extracting confidence{sample_context}: {str(e)}")
            logger.error(f"Full traceback:\n{traceback.format_exc()}")
        
        return result
    
    def _extract_confidence_qwen(self, image, probing_query):
        """Extract confidence for Qwen model."""
        result = {'confidence': None, 'raw_response': None, 'success': False}
        
        # Build conversation with system prompt
        messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": probing_query},
                ],
            }
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
            print(f"\n🔧 QWEN SELF-PROBING EXTRACTION:")
            print(f"   Input shape: {inputs.input_ids.shape}")
        
        # Generate response
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )
        
        # Decode response
        input_len = inputs.input_ids.shape[1]
        generated_ids = output[0][input_len:]
        response_text = self.processor.decode(generated_ids, skip_special_tokens=True)
        
        result['raw_response'] = response_text
        
        # Extract confidence
        confidence = extract_confidence_from_response(response_text)
        if confidence is not None:
            result['confidence'] = confidence
            result['success'] = True
        
        if self.debug:
            print(f"   Raw response: {response_text[:200]}...")
            print(f"   Extracted confidence: {confidence}")
        
        return result
    
    def _extract_confidence_llava(self, image, probing_query):
        """Extract confidence for LLaVA model."""
        result = {'confidence': None, 'raw_response': None, 'success': False}
        
        # Build conversation (LLaVA doesn't support system prompt in same way)
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": probing_query + "\n\n" + APPENDED_SYSTEM_PROMPT},
                ],
            },
        ]
        
        prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        
        inputs = self.processor(images=image, text=prompt, return_tensors="pt")
        
        # Move to device
        device = next(self.model.parameters()).device
        model_dtype = next(self.model.parameters()).dtype
        
        inputs = {k: v.to(device=device) if isinstance(v, torch.Tensor) else v 
                  for k, v in inputs.items()}
        
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(dtype=model_dtype)
        
        if self.debug:
            print(f"\n🔧 LLAVA SELF-PROBING EXTRACTION:")
            print(f"   Input shape: {inputs['input_ids'].shape}")
        
        # Generate
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )
        
        # Decode
        input_len = inputs["input_ids"].shape[1]
        generated_ids = output[0][input_len:]
        response_text = self.processor.decode(generated_ids, skip_special_tokens=True)
        
        result['raw_response'] = response_text
        
        # Extract confidence
        confidence = extract_confidence_from_response(response_text)
        if confidence is not None:
            result['confidence'] = confidence
            result['success'] = True
        
        if self.debug:
            print(f"   Raw response: {response_text[:200]}...")
            print(f"   Extracted confidence: {confidence}")
        
        return result
    
    def _extract_confidence_gemma(self, image, probing_query):
        """Extract confidence for Gemma model."""
        result = {'confidence': None, 'raw_response': None, 'success': False}
        
        # Build conversation with system prompt
        conversation = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": probing_query},
                ],
            },
        ]
        
        prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        
        inputs = self.processor(images=image, text=prompt, return_tensors="pt")
        
        device = next(self.model.parameters()).device
        model_dtype = next(self.model.parameters()).dtype
        
        inputs = {k: v.to(device=device) if isinstance(v, torch.Tensor) else v 
                  for k, v in inputs.items()}
        
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(dtype=model_dtype)
        
        if self.debug:
            print(f"\n🔧 GEMMA SELF-PROBING EXTRACTION:")
            print(f"   Input shape: {inputs['input_ids'].shape}")
        
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )
        
        input_len = inputs["input_ids"].shape[1]
        generated_ids = output[0][input_len:]
        response_text = self.processor.decode(generated_ids, skip_special_tokens=True)
        
        result['raw_response'] = response_text
        
        confidence = extract_confidence_from_response(response_text)
        if confidence is not None:
            result['confidence'] = confidence
            result['success'] = True
        
        if self.debug:
            print(f"   Raw response: {response_text[:200]}...")
            print(f"   Extracted confidence: {confidence}")
        
        return result
    
    def _extract_confidence_internvl(self, image, probing_query):
        """Extract confidence for InternVL model."""
        result = {'confidence': None, 'raw_response': None, 'success': False}
        
        # Build conversation with system prompt
        messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": probing_query},
                ],
            },
        ]
        
        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        
        inputs = self.processor(images=image, text=prompt, return_tensors="pt")
        
        device = next(self.model.parameters()).device
        model_dtype = next(self.model.parameters()).dtype
        
        inputs = {k: v.to(device=device) if isinstance(v, torch.Tensor) else v 
                  for k, v in inputs.items()}
        
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(dtype=model_dtype)
        
        if self.debug:
            print(f"\n🔧 INTERNVL SELF-PROBING EXTRACTION:")
            print(f"   Input shape: {inputs['input_ids'].shape}")
        
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )
        
        input_len = inputs["input_ids"].shape[1]
        generated_ids = output[0][input_len:]
        response_text = self.processor.decode(generated_ids, skip_special_tokens=True)
        
        result['raw_response'] = response_text
        
        confidence = extract_confidence_from_response(response_text)
        if confidence is not None:
            result['confidence'] = confidence
            result['success'] = True
        
        if self.debug:
            print(f"   Raw response: {response_text[:200]}...")
            print(f"   Extracted confidence: {confidence}")
        
        return result
    
    def _extract_confidence_deepseek(self, image, probing_query):
        """Extract confidence for DeepSeek model (matching generate_and_extract.py)."""
        result = {'confidence': None, 'raw_response': None, 'success': False}
        
        temp_image_created = False
        image_path = None
        
        try:
            # Save PIL image to a temporary file for DeepSeek
            temp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
            image.save(temp_file.name)
            image_path = temp_file.name
            temp_image_created = True
            
            # Create DeepSeek-style conversation (matching generate_and_extract.py)
            conversation = [
                {
                    "role": "<|User|>",
                    "content": f"<image>\n{probing_query}",
                    "images": [image_path],
                },
                {"role": "<|Assistant|>", "content": ""},
            ]
            
            # Load images and prepare inputs
            pil_images = load_pil_images(conversation)
            prepare_inputs = self.processor(
                conversations=conversation,
                images=pil_images,
                force_batchify=True,
                system_prompt=SYSTEM_PROMPT
            ).to(self.model.device)
            
            # Convert pixel values to match model dtype
            model_dtype = next(self.model.parameters()).dtype
            if hasattr(prepare_inputs, 'pixel_values') and prepare_inputs.pixel_values is not None:
                prepare_inputs.pixel_values = prepare_inputs.pixel_values.to(dtype=model_dtype)
            
            # Get input embeddings
            inputs_embeds = self.model.prepare_inputs_embeds(**prepare_inputs)
            
            if self.debug:
                print(f"\n🔧 DEEPSEEK SELF-PROBING EXTRACTION:")
                print(f"   Input embeds shape: {inputs_embeds.shape}")
            
            # Generate with DeepSeek's language model
            with torch.no_grad():
                output = self.model.language.generate(
                    inputs_embeds=inputs_embeds,
                    attention_mask=prepare_inputs.attention_mask,
                    pad_token_id=self.processor.tokenizer.eos_token_id,
                    bos_token_id=self.processor.tokenizer.bos_token_id,
                    eos_token_id=self.processor.tokenizer.eos_token_id,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                )
            
            # Decode - DeepSeek returns only generated tokens
            generated_ids = output[0]
            response_text = self.processor.tokenizer.decode(generated_ids, skip_special_tokens=True)
            
            result['raw_response'] = response_text
            
            confidence = extract_confidence_from_response(response_text)
            if confidence is not None:
                result['confidence'] = confidence
                result['success'] = True
            
            if self.debug:
                print(f"   Raw response: {response_text[:200]}...")
                print(f"   Extracted confidence: {confidence}")
            
        finally:
            # Clean up temporary image file
            if temp_image_created and image_path:
                try:
                    os.unlink(image_path)
                except Exception as cleanup_error:
                    logger.debug(f"Failed to cleanup temp file {image_path}: {cleanup_error}")
        
        return result


# ============================================================================
# vLLM Self-Probing Extractor
# ============================================================================

class VLLMSelfProbingExtractor:
    """Extracts self-probing confidence scores from VLMs using vLLM with batching."""
    
    def __init__(self, vlm_inference: VLLMInference, max_new_tokens=128, debug=False, max_image_dim=MAX_IMAGE_DIMENSION):
        self.vlm = vlm_inference
        self.max_new_tokens = max_new_tokens
        self.debug = debug
        self.max_image_dim = max_image_dim
    
    def extract_confidence_batch(
        self, 
        images: List, 
        questions: List[str], 
        generated_responses: List[str],
        sample_ids: List[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Extract self-probing confidence for a batch of samples.
        
        Args:
            images: List of images (PIL, path, bytes, etc.)
            questions: List of original questions
            generated_responses: List of model's generated responses
            sample_ids: Optional list of sample IDs for logging
        
        Returns:
            List of dicts with 'confidence', 'raw_response', 'success'
        """
        # Preprocess all images
        processed_images = []
        for img in images:
            try:
                processed_images.append(load_and_preprocess_image(img, max_dim=self.max_image_dim))
            except Exception as e:
                logger.warning(f"Failed to preprocess image: {e}")
                processed_images.append(None)
        
        # Build probing queries
        probing_queries = [
            get_self_probing_query(q, r) 
            for q, r in zip(questions, generated_responses)
        ]
        
        # Filter out failed images
        valid_indices = [i for i, img in enumerate(processed_images) if img is not None]
        valid_images = [processed_images[i] for i in valid_indices]
        valid_queries = [probing_queries[i] for i in valid_indices]
        
        if not valid_images:
            return [{'confidence': None, 'raw_response': None, 'success': False} for _ in images]
        
        # Batch inference
        try:
            responses = self.vlm.batch_inference(
                valid_images,
                valid_queries,
                max_tokens=self.max_new_tokens,
                temperature=0.0,
                use_tqdm=False,
            )
        except Exception as e:
            logger.error(f"Batch inference failed: {e}")
            return [{'confidence': None, 'raw_response': None, 'success': False} for _ in images]
        
        # Map responses back and extract confidence
        results = [{'confidence': None, 'raw_response': None, 'success': False} for _ in images]
        
        for idx, response in zip(valid_indices, responses):
            results[idx]['raw_response'] = response
            
            confidence = extract_confidence_from_response(response)
            if confidence is not None:
                results[idx]['confidence'] = confidence
                results[idx]['success'] = True
            
            if self.debug and idx < 5:
                print(f"\n🔧 vLLM SELF-PROBING (sample {idx}):")
                print(f"   Raw response: {response[:200]}...")
                print(f"   Extracted confidence: {confidence}")
        
        return results
    
    def extract_confidence(self, image, question: str, generated_response: str, sample_id: str = None) -> Dict[str, Any]:
        """Single sample extraction (calls batch with size 1)."""
        results = self.extract_confidence_batch([image], [question], [generated_response], [sample_id])
        return results[0]


# ============================================================================
# Main Evaluator
# ============================================================================

class SelfProbingEvaluator:
    """Main evaluator class for self-probing evaluation."""
    
    def __init__(self, extractor, args):
        self.extractor = extractor
        self.args = args
        self.use_vllm = args.backend == 'vllm'

        # Setup GPT fallback client if enabled
        self.gpt_client = None
        if getattr(args, 'gpt_fallback', False):
            self.gpt_client = create_openai_client(getattr(args, 'openai_api_key', None))
            if self.gpt_client:
                logger.info(f"GPT fallback enabled: model={args.gpt_model}, effort={args.gpt_reasoning_effort}")
            else:
                logger.warning("GPT fallback requested but client creation failed. Falling back to skip.")
    
    def evaluate(self, dataset, npz_samples: Dict[str, Dict[str, Any]], output_dir: Path) -> Dict[str, Any]:
        """Run self-probing evaluation."""
        logger.info("Starting self-probing evaluation...")
        
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
                    logger.info(f"Loaded {len(existing_records)} existing records.")
                except Exception as e:
                    logger.warning(f"Failed to load existing results: {e}. Starting fresh.")
                    existing_records = []
                    processed_hash_ids = set()
            else:
                logger.info("Resume mode enabled but no existing results found. Starting fresh.")
        
        evaluation_records = existing_records.copy()
        new_records_count = 0
        skipped_already_processed = 0
        processed_count = len(existing_records)
        skipped_no_npz = 0
        skipped_no_correctness = 0
        failed_extraction = 0
        
        # Statistics
        confidence_values = [r['confidence_score'] for r in existing_records]
        
        # Limit samples if specified
        total_samples = len(dataset)
        if self.args.max_samples:
            total_samples = min(total_samples, self.args.max_samples)
        
        logger.info(f"Processing {total_samples} samples...")
        if self.args.resume:
            logger.info(f"Resume mode: Will skip {len(processed_hash_ids)} already processed samples")
        
        # Pre-extract hash_ids efficiently
        logger.info("Pre-extracting hash_ids from dataset...")
        all_hash_ids = dataset['hash_id'][:total_samples]
        
        # Build list of indices that need processing
        indices_to_process = []
        for idx, hash_id in enumerate(all_hash_ids):
            if hash_id in processed_hash_ids:
                skipped_already_processed += 1
                continue
            if hash_id not in npz_samples:
                skipped_no_npz += 1
                continue
            indices_to_process.append(idx)
        
        logger.info(f"Found {len(indices_to_process)} samples to process")
        
        # Process based on backend
        if self.use_vllm and len(indices_to_process) > 0:
            # vLLM batch processing
            evaluation_records, new_records_count, failed_extraction, skipped_no_correctness = \
                self._process_vllm_batched(
                    dataset, npz_samples, indices_to_process, 
                    evaluation_records, processed_hash_ids, labels_path
                )
        else:
            # HuggingFace sequential processing
            for idx in tqdm(indices_to_process, desc="Extracting self-probing confidence"):
                sample = dataset[idx]
                hash_id = sample['hash_id']
                
                question = sample[self.args.question_column]
                image = sample[self.args.image_column]
                
                npz_data = npz_samples[hash_id]
                
                is_correct = npz_data.get('is_correct')
                if is_correct is None:
                    skipped_no_correctness += 1
                    continue
                
                generated_response = npz_data['generated_response']
                
                # Debug output
                show_debug = self.args.debug and new_records_count < self.args.debug_samples
                if show_debug:
                    print(f"\n{'='*100}")
                    print(f"🔬 SAMPLE {new_records_count + 1}")
                    print(f"📝 Question: {question[:200]}...")
                    print(f"🤖 Generated: {generated_response[:200]}...")
                    print(f"📊 Is correct: {is_correct}")
                
                # Extract confidence
                confidence_result = self.extractor.extract_confidence(
                    image, question, generated_response,
                    sample_id=npz_data.get('sample_id', hash_id)
                )
                
                if not confidence_result['success']:
                    # GPT fallback: try to judge the confidence from the raw response
                    if self.gpt_client and confidence_result.get('raw_response'):
                        gpt_confidence = gpt_judge_confidence(
                            raw_response=confidence_result['raw_response'],
                            question=question,
                            generated_answer=generated_response,
                            client=self.gpt_client,
                            gpt_model=self.args.gpt_model,
                            reasoning_effort=self.args.gpt_reasoning_effort,
                        )
                        if gpt_confidence is not None:
                            confidence_result['confidence'] = gpt_confidence
                            confidence_result['success'] = True
                            if show_debug:
                                print(f"🔄 GPT fallback confidence: {gpt_confidence:.4f}")
                        else:
                            failed_extraction += 1
                            if show_debug:
                                print(f"❌ Extraction failed (GPT fallback also failed)")
                            continue
                    else:
                        failed_extraction += 1
                        if show_debug:
                            print(f"❌ Extraction failed")
                        continue

                # Store results
                record = {
                    'sample_id': npz_data['sample_id'],
                    'hash_id': hash_id,
                    'ground_truth_correctness': int(is_correct),
                    'confidence_score': confidence_result['confidence'],
                    'raw_response': confidence_result['raw_response'],
                    'dataset': sample.get('dataset', 'unknown'),
                }
                evaluation_records.append(record)
                processed_hash_ids.add(hash_id)
                
                confidence_values.append(confidence_result['confidence'])
                processed_count += 1
                new_records_count += 1
                
                if show_debug:
                    print(f"✅ Confidence: {confidence_result['confidence']:.4f}")
                
                # Incremental save
                if new_records_count > 0 and new_records_count % 100 == 0:
                    self._save_records(evaluation_records, labels_path)
                    logger.info(f"Incremental save: {len(evaluation_records)} records")
                
                # Clear CUDA cache periodically
                if new_records_count % 50 == 0:
                    gc.collect()
                    torch.cuda.empty_cache()
        
        # Log summary
        logger.info(f"\n{'='*60}")
        logger.info(f"PROCESSING SUMMARY:")
        logger.info(f"  Total successfully processed: {len(evaluation_records)}")
        logger.info(f"  New records: {new_records_count}")
        logger.info(f"  Skipped (already processed): {skipped_already_processed}")
        logger.info(f"  Skipped (no npz): {skipped_no_npz}")
        logger.info(f"  Skipped (no correctness): {skipped_no_correctness}")
        logger.info(f"  Failed extraction: {failed_extraction}")
        if getattr(self.args, 'gpt_fallback', False):
            logger.info(f"  GPT fallback: enabled (model={self.args.gpt_model}, effort={self.args.gpt_reasoning_effort})")
        logger.info(f"{'='*60}")
        
        if not evaluation_records:
            logger.error("No valid records to evaluate!")
            return None
        
        # Save final records
        self._save_records(evaluation_records, labels_path)
        
        # Calculate metrics
        labels = np.array([r['ground_truth_correctness'] for r in evaluation_records], dtype=float)
        scores = np.array([r['confidence_score'] for r in evaluation_records], dtype=float)
        
        metrics = calculate_all_metrics(labels, scores)
        
        # Build results structure
        confidence_values = [r['confidence_score'] for r in evaluation_records]
        results = {
            'overall': {
                'n_samples': len(evaluation_records),
                'n_total_samples': len(dataset),
                **metrics
            },
            'metadata': {
                'model_id': self.args.model_id,
                'model_name_part': self.args.model_id.split("/")[-1],
                'test_dataset_name': self.args.test_dataset_name,
                'backend': self.args.backend,
                'total_records': len(evaluation_records),
                'evaluation_timestamp': datetime.now().isoformat(),
                'confidence_statistics': {
                    'avg_confidence': float(np.mean(confidence_values)),
                    'confidence_std': float(np.std(confidence_values)),
                    'min_confidence': float(np.min(confidence_values)),
                    'max_confidence': float(np.max(confidence_values)),
                },
            }
        }
        
        # Save results
        results_path = output_dir / "test_results.json"
        save_evaluation_results(results, str(results_path))
        
        # Print summary
        self._print_results_summary(results)
        
        return results
    
    def _process_vllm_batched(self, dataset, npz_samples, indices_to_process, 
                               evaluation_records, processed_hash_ids, labels_path):
        """Process all samples in one batch using vLLM (entire dataset at once)."""
        new_records_count = 0
        failed_extraction = 0
        skipped_no_correctness = 0
        
        logger.info(f"Processing {len(indices_to_process)} samples in one batch with vLLM...")
        
        # Collect all data for the entire dataset
        batch_images = []
        batch_questions = []
        batch_responses = []
        batch_metadata = []
        
        for idx in tqdm(indices_to_process, desc="Preparing batch data"):
            sample = dataset[idx]
            hash_id = sample['hash_id']
            npz_data = npz_samples[hash_id]
            
            is_correct = npz_data.get('is_correct')
            if is_correct is None:
                skipped_no_correctness += 1
                continue
            
            batch_images.append(sample[self.args.image_column])
            batch_questions.append(sample[self.args.question_column])
            batch_responses.append(npz_data['generated_response'])
            batch_metadata.append({
                'hash_id': hash_id,
                'sample_id': npz_data['sample_id'],
                'is_correct': is_correct,
                'dataset': sample.get('dataset', 'unknown'),
            })
        
        if not batch_images:
            logger.warning("No valid samples to process after filtering")
            return evaluation_records, new_records_count, failed_extraction, skipped_no_correctness
        
        logger.info(f"Running inference on {len(batch_images)} samples in one batch...")
        
        # Process entire dataset as one batch
        results = self.extractor.extract_confidence_batch(
            batch_images, batch_questions, batch_responses
        )
        
        # Process results �� first pass: collect regex successes and GPT fallback candidates
        gpt_fallback_items = []
        regex_success_items = []

        for i, (meta, result) in enumerate(zip(batch_metadata, results)):
            if result['success']:
                regex_success_items.append((meta, result))
            elif self.gpt_client and result.get('raw_response'):
                gpt_fallback_items.append((i, meta, result))
            else:
                failed_extraction += 1

        # Add regex successes
        for meta, result in regex_success_items:
            record = {
                'sample_id': meta['sample_id'],
                'hash_id': meta['hash_id'],
                'ground_truth_correctness': int(meta['is_correct']),
                'confidence_score': result['confidence'],
                'raw_response': result['raw_response'],
                'dataset': meta['dataset'],
            }
            evaluation_records.append(record)
            processed_hash_ids.add(meta['hash_id'])
            new_records_count += 1

        # Run GPT fallback in parallel for failed extractions
        if gpt_fallback_items:
            from concurrent.futures import ThreadPoolExecutor, as_completed
            n_workers = getattr(self.args, 'gpt_n_parallel', 20)
            logger.info(f"Running GPT fallback on {len(gpt_fallback_items)} failed extractions ({n_workers} parallel workers)...")

            def _gpt_fallback_one(item):
                idx, meta, result = item
                conf = gpt_judge_confidence(
                    raw_response=result['raw_response'],
                    question=batch_questions[idx],
                    generated_answer=batch_responses[idx],
                    client=self.gpt_client,
                    gpt_model=self.args.gpt_model,
                    reasoning_effort=self.args.gpt_reasoning_effort,
                )
                return meta, result, conf

            with ThreadPoolExecutor(max_workers=n_workers) as executor:
                futures = [executor.submit(_gpt_fallback_one, item) for item in gpt_fallback_items]
                for future in tqdm(as_completed(futures), total=len(futures), desc="GPT fallback"):
                    meta, result, gpt_confidence = future.result()
                    if gpt_confidence is not None:
                        record = {
                            'sample_id': meta['sample_id'],
                            'hash_id': meta['hash_id'],
                            'ground_truth_correctness': int(meta['is_correct']),
                            'confidence_score': gpt_confidence,
                            'raw_response': result['raw_response'],
                            'dataset': meta['dataset'],
                        }
                        evaluation_records.append(record)
                        processed_hash_ids.add(meta['hash_id'])
                        new_records_count += 1
                    else:
                        failed_extraction += 1

        logger.info(f"Processed {new_records_count} samples successfully (regex: {len(regex_success_items)}, gpt_fallback: {new_records_count - len(regex_success_items)})")
        
        return evaluation_records, new_records_count, failed_extraction, skipped_no_correctness
    
    def _save_records(self, records, path):
        """Atomic save of records."""
        try:
            temp_path = path.with_suffix('.json.tmp')
            with open(temp_path, 'w', encoding='utf-8') as f:
                json.dump(records, f, indent=2, ensure_ascii=False)
            temp_path.replace(path)
        except Exception as e:
            logger.error(f"Failed to save records: {e}")
    
    def _print_results_summary(self, results: Dict[str, Any]) -> None:
        """Print evaluation results summary."""
        print(f"\n{'='*60}")
        print(f"SELF-PROBING EVALUATION SUMMARY")
        print(f"{'='*60}")
        
        if 'overall' in results:
            overall = results['overall']
            print(f"\nOverall Performance:")
            print(f"  Samples: {overall['n_samples']} / {overall['n_total_samples']}")
            print(f"  Accuracy: {overall['accuracy']:.4f}")
            print(f"  AUROC: {overall['auroc']:.4f}")
            print(f"  AUCPR: {overall['aucpr']:.4f}")
            print(f"  ECE: {overall['ece']:.4f}")
            print(f"  Brier Score: {overall['brier']:.4f}")
        
        if 'metadata' in results and 'confidence_statistics' in results['metadata']:
            stats = results['metadata']['confidence_statistics']
            print(f"\nConfidence Statistics:")
            print(f"  Average: {stats['avg_confidence']:.4f}")
            print(f"  Std: {stats['confidence_std']:.4f}")
            print(f"  Range: [{stats['min_confidence']:.4f}, {stats['max_confidence']:.4f}]")
        
        print(f"{'='*60}")


# ============================================================================
# Main Function
# ============================================================================

def main():
    args = parse_args()
    
    model_name_part = args.model_id.split("/")[-1]
    
    logger.info("="*80)
    logger.info("SELF-PROBING EVALUATION FOR VISION-LANGUAGE MODELS")
    logger.info("="*80)
    logger.info(f"Model: {args.model_id}")
    logger.info(f"Backend: {args.backend}")
    logger.info(f"Dtype: {args.dtype}")
    logger.info(f"Test dataset: {args.test_dataset_name}")
    logger.info(f"GPU IDs: {args.gpu_ids}")
    logger.info(f"Debug mode: {args.debug}")
    logger.info(f"GPT fallback: {args.gpt_fallback}")
    if args.gpt_fallback:
        logger.info(f"GPT model: {args.gpt_model}, reasoning effort: {args.gpt_reasoning_effort}")

    if args.backend == 'vllm':
        logger.info(f"vLLM: Processing entire dataset as one batch")
        logger.info(f"Tensor parallel size: {args.tensor_parallel_size}")
    elif args.backend == 'hf':
        logger.info(f"HuggingFace batch size: {args.batch_size}")
    
    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")
    
    # Load model based on backend
    if args.backend == 'vllm':
        if not VLLM_AVAILABLE:
            logger.error("vLLM not available. Please install with: pip install vllm")
            sys.exit(1)
        
        vlm_inference = VLLMInference(
            args.model_id,
            dtype=args.dtype,
            gpu_memory_utilization=args.gpu_memory_utilization,
            tensor_parallel_size=args.tensor_parallel_size,
        )
        extractor = VLLMSelfProbingExtractor(
            vlm_inference,
            max_new_tokens=args.max_new_tokens,
            debug=args.debug,
            max_image_dim=args.max_image_dim,
        )
    else:
        # HuggingFace backend
        model, processor, model_type = load_hf_model_and_processor(args.model_id, args.dtype, device)
        extractor = HFSelfProbingExtractor(
            model, processor, model_type,
            max_new_tokens=args.max_new_tokens,
            debug=args.debug,
            max_image_dim=args.max_image_dim,
        )
    
    # Load original dataset (for images)
    full_dataset_path = os.path.join(args.dataset_path, args.test_dataset_name)
    dataset = load_dataset(full_dataset_path)
    
    # Load npz samples
    samples_dir = Path(args.data_dir) / model_name_part / args.test_dataset_name / "samples"
    npz_samples = load_npz_samples(samples_dir)
    
    if not npz_samples:
        logger.error(f"No npz samples found in {samples_dir}")
        return
    
    # Setup output directory
    output_dir = Path(args.output_dir) / model_name_part / args.test_dataset_name
    
    # Run evaluation
    evaluator = SelfProbingEvaluator(extractor, args)
    results = evaluator.evaluate(dataset, npz_samples, output_dir)
    
    if results:
        print(f"\n✅ Evaluation complete!")
        print(f"📁 Results saved to: {output_dir}")
    else:
        print("\n❌ Evaluation failed!")
        sys.exit(1)
    
    # Cleanup
    if args.backend == 'hf':
        del model, processor
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

# ============================================================================
# Example Usage
# ============================================================================

# Qwen/Qwen3-VL-8B-Instruct
# llava-hf/llava-v1.6-vicuna-13b-hf - 13B
# OpenGVLab/InternVL3_5-14B-HF - 14B
# google/gemma-3-27b-it - 27B
# deepseek-ai/deepseek-vl2 - 27B

# vLLM backend (processes entire dataset as one batch):
# python self_probing_eval.py \
#     --model_id "google/gemma-3-27b-it" \
#     --backend vllm \
#     --data_dir "../data/extraction/raw/" \
#     --dataset_path "../data/VLCB/raw" \
#     --test_dataset_name "test" \
#     --output_dir "../results/SELF_PROBING" \
#     --gpu_ids "5" \
#     --tensor_parallel_size 1 \
#     --max_new_tokens 64 \
#     --dtype "float32"

# HuggingFace backend (default):
# python self_probing_eval.py \
#     --model_id "deepseek-ai/deepseek-vl2" \
#     --backend hf \
#     --data_dir "../data/extraction/raw/" \
#     --dataset_path "../data/VLCB/raw" \
#     --test_dataset_name "test" \
#     --output_dir "../results/SELF_PROBING" \
#     --gpu_ids "6" \
#     --dtype "float32" \
#     --max_new_tokens 64

# DeepSeek with HuggingFace:
# python self_probing_eval.py \
#     --model_id "deepseek-ai/deepseek-vl2" \
#     --backend hf \
#     --data_dir "../data/extraction/raw/" \
#     --dataset_path "../data/VLCB/raw" \
#     --test_dataset_name "test" \
#     --output_dir "../results/SELF_PROBING" \
#     --gpu_ids "0" \
#     --dtype "bfloat16" \
#     --max_new_tokens 64 \
#     --resume

# DeepSeek with vLLM:
# python self_probing_eval.py \
#     --model_id "deepseek-ai/deepseek-vl2" \
#     --backend hf \
#     --data_dir "../data/extraction/raw/" \
#     --dataset_path "../data/VLCB/raw" \
#     --test_dataset_name "test" \
#     --output_dir "../results/SELF_PROBING" \
#     --gpu_ids "1,2" \
#     --tensor_parallel_size 2 \
#     --max_new_tokens 64