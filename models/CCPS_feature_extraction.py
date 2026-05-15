#!/usr/bin/env python3
"""
CCPS Feature Extraction for VLMs
================================
Extracts CCPS (Calibrating LLM Confidence by Probing Perturbed Representation Stability) 
features from VLM outputs.

This script:
1. Loads existing .npz samples from generate_and_extract.py (uses token_ids, metadata)
2. Reconstructs generation context with full response appended
3. Does SINGLE forward pass to get all hidden states (KEY OPTIMIZATION: ~T times faster)
4. Computes Jacobians analytically in batch (no backprop needed)
5. Generates perturbation trajectories (S steps along gradient direction)
6. Extracts comprehensive CCPS features per token in batch
7. Saves features for downstream training

Two modes:
- Default (fast): Single forward pass + analytical gradient computation (no backprop)
- --manual (slow): Sequential forward passes + traditional backpropagation

Usage:
    # Fast mode (analytical gradient - recommended)
    python CCPS_feature_extraction.py \
        --data-dir ../data/extraction/raw/ \
        --model-name Qwen/Qwen3-VL-8B-Instruct \
        --dataset-name train \
        --vlcb-dataset-path ../data/VLCB/raw \
        --output-dir ../data/CCPS_features/ \
        --gpu-ids 0 \
        --pei-radius 20.0 \
        --pei-steps 5
    
    # Manual mode (backprop-based gradient)
    python CCPS_feature_extraction.py \
        --data-dir ../data/extraction/raw/ \
        --model-name Qwen/Qwen3-VL-8B-Instruct \
        --dataset-name train \
        --vlcb-dataset-path ../data/VLCB/raw \
        --output-dir ../data/CCPS_features/ \
        --gpu-ids 0 \
        --pei-radius 20.0 \
        --pei-steps 5 \
        --manual
"""

import os
import argparse
import gc
import json
import logging
import pickle
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime

# Constants (defined before parse_args so they can be used in argument defaults)
MAX_IMAGE_DIMENSION = 2048  # Default maximum allowed dimension (width or height)

# Parse arguments BEFORE importing torch to set CUDA_VISIBLE_DEVICES
def parse_args():
    parser = argparse.ArgumentParser(
        description='Extract CCPS features from VLM outputs',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Input configuration
    parser.add_argument('--data-dir', type=str, required=True,
                       help='Directory containing extracted representations from generate_and_extract.py')
    parser.add_argument('--model-name', type=str, required=True,
                       help='VLM model name (e.g., Qwen/Qwen3-VL-8B-Instruct)')
    parser.add_argument('--dataset-name', type=str, required=True,
                       help='Dataset name (e.g., train)')
    parser.add_argument('--vlcb-dataset-path', type=str, required=True,
                       help='Path to VLCB dataset (for loading images)')
    
    # Output configuration
    parser.add_argument('--output-dir', type=str, required=True,
                       help='Directory to save extracted CCPS features')
    
    # GPU configuration
    parser.add_argument('--gpu-ids', type=str, default='0',
                       help='GPU IDs to use (comma-separated)')
    parser.add_argument('--dtype', type=str, default='bfloat16',
                       choices=['float32', 'float16', 'bfloat16'],
                       help='Model dtype')
    
    # CCPS perturbation configuration
    parser.add_argument('--pei-radius', type=float, default=20.0,
                       help='Maximum perturbation radius (epsilon_max)')
    parser.add_argument('--pei-steps', type=int, default=5,
                       help='Number of perturbation steps (S)')
    
    # Processing configuration
    parser.add_argument('--max-samples', type=int, default=None,
                       help='Maximum number of samples to process (None for all)')
    parser.add_argument('--skip-existing', action='store_true',
                       help='Skip samples that already have features extracted')
    parser.add_argument('--batch-size', type=int, default=1,
                       help='Batch size for processing (1 recommended for memory)')
    parser.add_argument('--max-image-dim', type=int, default=MAX_IMAGE_DIMENSION,
                       help='Maximum image dimension (images larger than this will be resized, default: 2048)')
    
    # Gradient computation mode
    parser.add_argument('--manual', action='store_true',
                       help='Use manual backpropagation for gradient computation (slower but exact). '
                            'Default uses analytical gradient (faster, mathematically equivalent).')
    
    # Debug configuration
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode with verbose output')
    parser.add_argument('--debug-samples', type=int, default=3,
                       help='Number of samples to process in debug mode')
    
    return parser.parse_args()

args = parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids

# Now import torch and other CUDA-dependent libraries
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.amp import autocast
from tqdm import tqdm
from datasets import load_from_disk
from PIL import Image
import io

# VLM imports
from transformers import AutoProcessor, AutoModelForCausalLM

# Qwen-specific imports
try:
    from qwen_vl_utils import process_vision_info
    QWEN_AVAILABLE = True
except ImportError:
    QWEN_AVAILABLE = False

# DeepSeek VL2 imports (optional - only needed if using DeepSeek models)
try:
    from deepseek_vl2.models import DeepseekVLV2Processor, DeepseekVLV2ForCausalLM
    from deepseek_vl2.utils.io import load_pil_images
    DEEPSEEK_AVAILABLE = True
except ImportError:
    DEEPSEEK_AVAILABLE = False

# Setup logging
logging.basicConfig(
    level=logging.DEBUG if args.debug else logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"CCPS_feature_extraction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    ]
)
logger = logging.getLogger(__name__)


# ============================================================================
# Constants
# ============================================================================

SYSTEM_PROMPT = "You are a vision language assistant. Provide brief, complete answers."
APPENDED_SYSTEM_PROMPT = "Provide a brief, complete answer."

# Feature column names (for consistent ordering)
FEATURE_COLUMNS = [
    # Original state features (12)
    'original_log_prob_actual',
    'original_prob_actual', 
    'original_logit_actual',
    'original_prob_argmax',
    'original_logit_argmax',
    'original_entropy',
    'original_margin_logit_top1_top2',
    'original_margin_prob_top1_top2',
    'original_norm_logits_L2',
    'original_std_logits',
    'original_norm_hidden_state_L2',
    'is_actual_token_original_argmax',
    
    # Overall perturbation features (3)
    'jacobian_norm_token',
    'epsilon_to_flip_token',
    'pei_value_token',
    
    # Perturbed state features - statistics (mean, std, min, max for each)
    'perturbed_log_prob_actual_mean', 'perturbed_log_prob_actual_std', 
    'perturbed_log_prob_actual_min', 'perturbed_log_prob_actual_max',
    'perturbed_prob_actual_mean', 'perturbed_prob_actual_std',
    'perturbed_prob_actual_min', 'perturbed_prob_actual_max',
    'perturbed_logit_actual_mean', 'perturbed_logit_actual_std',
    'perturbed_logit_actual_min', 'perturbed_logit_actual_max',
    'delta_log_prob_actual_from_original_mean', 'delta_log_prob_actual_from_original_std',
    'delta_log_prob_actual_from_original_min', 'delta_log_prob_actual_from_original_max',
    'perturbed_prob_argmax_mean', 'perturbed_prob_argmax_std',
    'perturbed_prob_argmax_min', 'perturbed_prob_argmax_max',
    'perturbed_logit_argmax_mean', 'perturbed_logit_argmax_std',
    'perturbed_logit_argmax_min', 'perturbed_logit_argmax_max',
    'did_argmax_change_from_original_mean', 'did_argmax_change_from_original_std',
    'did_argmax_change_from_original_min', 'did_argmax_change_from_original_max',
    'perturbed_entropy_mean', 'perturbed_entropy_std',
    'perturbed_entropy_min', 'perturbed_entropy_max',
    'perturbed_margin_logit_top1_top2_mean', 'perturbed_margin_logit_top1_top2_std',
    'perturbed_margin_logit_top1_top2_min', 'perturbed_margin_logit_top1_top2_max',
    'perturbed_norm_logits_L2_mean', 'perturbed_norm_logits_L2_std',
    'perturbed_norm_logits_L2_min', 'perturbed_norm_logits_L2_max',
    
    # Comparison features - statistics
    'kl_div_perturbed_from_original_mean', 'kl_div_perturbed_from_original_std',
    'kl_div_perturbed_from_original_min', 'kl_div_perturbed_from_original_max',
    'js_div_perturbed_from_original_mean', 'js_div_perturbed_from_original_std',
    'js_div_perturbed_from_original_min', 'js_div_perturbed_from_original_max',
    'cosine_sim_logits_perturbed_to_original_mean', 'cosine_sim_logits_perturbed_to_original_std',
    'cosine_sim_logits_perturbed_to_original_min', 'cosine_sim_logits_perturbed_to_original_max',
    'cosine_sim_hidden_perturbed_to_original_mean', 'cosine_sim_hidden_perturbed_to_original_std',
    'cosine_sim_hidden_perturbed_to_original_min', 'cosine_sim_hidden_perturbed_to_original_max',
    'l2_dist_hidden_perturbed_from_original_mean', 'l2_dist_hidden_perturbed_from_original_std',
    'l2_dist_hidden_perturbed_from_original_min', 'l2_dist_hidden_perturbed_from_original_max',
]


# ============================================================================
# Image Preprocessing
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
    
    w, h = pil_img.size
    max_current = max(w, h)
    
    # Only resize if LARGER than limit (no upsampling)
    if max_current <= max_dim:
        return pil_img  # Return unchanged
    
    # Calculate new dimensions preserving aspect ratio
    scale = max_dim / max_current
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    # Use LANCZOS for high-quality downsampling
    resized = pil_img.resize((new_w, new_h), Image.LANCZOS)
    
    if args.debug:
        logger.debug(f"Resized image from {w}x{h} to {new_w}x{new_h}")
    
    return resized


# ============================================================================
# Model Loading
# ============================================================================

def load_vlm_model(model_name: str, dtype_str: str, device: str = 'cuda'):
    """Load VLM model and processor"""
    logger.info(f"Loading VLM model: {model_name}")
    
    dtype_map = {
        'float32': torch.float32,
        'float16': torch.float16,
        'bfloat16': torch.bfloat16,
    }
    dtype = dtype_map[dtype_str]
    
    model_name_lower = model_name.lower()
    
    # Determine model type and classes based on model_id (matching generate_and_extract.py)
    if 'deepseek' in model_name_lower and 'vl' in model_name_lower:
        # DeepSeek VL2 model
        if not DEEPSEEK_AVAILABLE:
            raise ImportError(
                "DeepSeek VL2 dependencies not found. Please install deepseek_vl2: "
                "pip install deepseek_vl2 or clone from https://github.com/deepseek-ai/DeepSeek-VL2"
            )
        model_type = 'deepseek'
        logger.info("Detected DeepSeek VL2 model")

        # DeepSeek-VL2 is designed to run in bfloat16; force dtype to bfloat16
        if dtype_str != 'bfloat16':
            logger.warning(
                f"DeepSeek VL2 requires bfloat16 for stable vision encoder behaviour. "
                f"Overriding requested dtype '{dtype_str}' to 'bfloat16'."
            )
        dtype_str = 'bfloat16'
        dtype = torch.bfloat16
        logger.info(f"Using dtype: {dtype_str} for DeepSeek VL2")
        
        # Load DeepSeek processor
        processor = DeepseekVLV2Processor.from_pretrained(model_name)
        
        # Load DeepSeek model
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
        )
        model = model.to(dtype).cuda().eval()
        
        actual_dtype = next(model.parameters()).dtype
        if actual_dtype != dtype:
            logger.warning(f"Model loaded in {actual_dtype} (expected {dtype}).")
        else:
            logger.info(f"Model loaded in {actual_dtype} as requested.")
        
        logger.info("Model and processor/tokenizer loaded successfully")
        return model, processor, model_type, dtype
        
    elif 'llava' in model_name_lower:
        model_type = 'llava'
        from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor
        model_class = LlavaNextForConditionalGeneration
        processor_class = LlavaNextProcessor
        processor_kwargs = {}
        logger.info("Detected LLaVA model")
    elif 'qwen' in model_name_lower:
        model_type = 'qwen'
        from transformers import Qwen3VLForConditionalGeneration
        model_class = Qwen3VLForConditionalGeneration
        processor_class = AutoProcessor
        processor_kwargs = {'trust_remote_code': True}
        logger.info("Detected Qwen model")
    elif 'gemma' in model_name_lower:
        model_type = 'gemma'
        from transformers import Gemma3ForConditionalGeneration, Gemma3Processor
        model_class = Gemma3ForConditionalGeneration
        processor_class = Gemma3Processor
        processor_kwargs = {}
        logger.info("Detected Gemma model")
    elif 'internvl' in model_name_lower:
        model_type = 'internvl'
        from transformers import AutoModelForImageTextToText
        model_class = AutoModelForImageTextToText
        processor_class = AutoProcessor
        processor_kwargs = {'trust_remote_code': True}
        logger.info("Detected InternVL model")
    else:
        raise ValueError(f"Unknown model type. Model ID must contain 'llava', 'qwen', 'deepseek', 'gemma', or 'internvl': {model_name}")
    
    # Set max_memory per device
    max_memory = {
        0: "130GiB",
        # 1: "130GiB",
        # 2: "130GiB",
        # 3: "130GiB",
        # 4: "130GiB",
        # 5: "130GiB",
        # 6: "130GiB",
        # 7: "130GiB",
    }
    
    # Load model
    model = model_class.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map="auto",
        max_memory=max_memory,
        attn_implementation='eager',  # Required for gradient computation in manual mode
        trust_remote_code=True,
    )
    model.eval()
    
    actual_dtype = next(model.parameters()).dtype
    if actual_dtype != dtype:
        logger.warning(f"Model loaded in {actual_dtype} (requested {dtype}).")
    else:
        logger.info(f"Model loaded in {actual_dtype} as requested.")
    
    # Load processor/tokenizer
    processor = processor_class.from_pretrained(model_name, **processor_kwargs)
    
    logger.info("Model and processor/tokenizer loaded successfully")
    return model, processor, model_type, dtype


# ============================================================================
# Helper Functions for LM Head Access
# ============================================================================

def get_lm_head(model, model_type: str):
    """Get the lm_head module from the model."""
    if model_type in ['qwen', 'llava', 'internvl', 'gemma']:
        return model.lm_head
    elif model_type == 'deepseek':
        return model.language.lm_head  # DeepSeek has lm_head inside language model
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


def get_lm_head_weight(model, model_type: str) -> torch.Tensor:
    """Get the lm_head weight matrix [vocab_size, hidden_dim]."""
    return get_lm_head(model, model_type).weight


# ============================================================================
# Dataset Loading
# ============================================================================

def load_vlcb_dataset(dataset_path: str, dataset_name: str):
    """Load VLCB dataset for image access"""
    full_path = os.path.join(dataset_path, dataset_name)
    logger.info(f"Loading VLCB dataset from: {full_path}")
    dataset = load_from_disk(full_path)
    
    # Create hash_id to index mapping
    hash_id_to_idx = {}
    for idx in range(len(dataset)):
        hash_id = dataset[idx]['hash_id']
        hash_id_to_idx[hash_id] = idx
    
    logger.info(f"Loaded {len(dataset)} samples, created hash_id mapping")
    return dataset, hash_id_to_idx


def load_extracted_samples(data_dir: str, model_name: str, dataset_name: str) -> List[Path]:
    """Load list of extracted sample files"""
    model_name_part = model_name.split("/")[-1]
    samples_dir = Path(data_dir) / model_name_part / dataset_name / "samples"
    
    if not samples_dir.exists():
        raise FileNotFoundError(f"Samples directory not found: {samples_dir}")
    
    npz_files = sorted(samples_dir.glob("*.npz"))
    logger.info(f"Found {len(npz_files)} sample files in {samples_dir}")
    return npz_files


# ============================================================================
# VLM Input Preparation
# ============================================================================

def prepare_vlm_input(question: str, image: Image.Image, processor, model_type: str, device: str, model=None):
    """Prepare VLM input for a given question and image (matching generate_and_extract.py exactly)"""
    
    if model_type == 'qwen':
        messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": question},
                ],
            }
        ]
        
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(device)
        
    elif model_type == 'llava':
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": question + "\n\n" + APPENDED_SYSTEM_PROMPT},
                ],
            },
        ]
        
        prompt = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        
        if isinstance(image, Image.Image):
            image_to_process = image
        elif isinstance(image, str):
            if image.startswith('http://') or image.startswith('https://'):
                import requests
                image_to_process = Image.open(requests.get(image, stream=True).raw)
            else:
                image_to_process = Image.open(image)
        else:
            image_to_process = image
        
        inputs = processor(images=image_to_process, text=prompt, return_tensors="pt")
        
        if model is not None:
            model_device = next(model.parameters()).device
            model_dtype = next(model.parameters()).dtype
        else:
            model_device = device
            model_dtype = torch.float32
        
        inputs = {k: v.to(device=model_device) if isinstance(v, torch.Tensor) else v 
                for k, v in inputs.items()}
        
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(dtype=model_dtype)
        
    elif model_type == 'gemma':
        conversation = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": question},
                ],
            },
        ]
        
        prompt = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        
        if isinstance(image, Image.Image):
            image_to_process = image
        elif isinstance(image, str):
            import requests
            if image.startswith('http://') or image.startswith('https://'):
                image_to_process = Image.open(requests.get(image, stream=True).raw)
            else:
                image_to_process = Image.open(image)
        else:
            image_to_process = image
        
        inputs = processor(images=image_to_process, text=prompt, return_tensors="pt")
        
        if model is not None:
            model_device = next(model.parameters()).device
            model_dtype = next(model.parameters()).dtype
        else:
            model_device = device
            model_dtype = torch.float32
        
        inputs = {k: v.to(device=model_device) if isinstance(v, torch.Tensor) else v 
                for k, v in inputs.items()}
        
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(dtype=model_dtype)
        
    elif model_type == 'internvl':
        messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": question},
                ],
            },
        ]
        
        prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        
        if isinstance(image, Image.Image):
            image_to_process = image
        elif isinstance(image, str):
            import requests
            if image.startswith('http://') or image.startswith('https://'):
                image_to_process = Image.open(requests.get(image, stream=True).raw)
            else:
                image_to_process = Image.open(image)
        else:
            image_to_process = image
        
        inputs = processor(images=image_to_process, text=prompt, return_tensors="pt")
        
        if model is not None:
            model_device = next(model.parameters()).device
            model_dtype = next(model.parameters()).dtype
        else:
            model_device = device
            model_dtype = torch.float32
        
        inputs = {k: v.to(device=model_device) if isinstance(v, torch.Tensor) else v 
                for k, v in inputs.items()}
        
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(dtype=model_dtype)
    
    elif model_type == 'deepseek':
        import tempfile
        if isinstance(image, Image.Image):
            temp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
            image.save(temp_file.name)
            image_path = temp_file.name
            temp_image_created = True
        elif isinstance(image, str):
            image_path = image
            temp_image_created = False
        else:
            image_path = image
            temp_image_created = False
        
        conversation = [
            {
                "role": "<|User|>",
                "content": f"<image>\n{question}",
                "images": [image_path],
            },
            {"role": "<|Assistant|>", "content": ""},
        ]
        
        pil_images = load_pil_images(conversation)
        prepare_inputs = processor(
            conversations=conversation,
            images=pil_images,
            force_batchify=True,
            system_prompt=SYSTEM_PROMPT
        ).to(device)
        
        model_dtype = torch.bfloat16
        if hasattr(prepare_inputs, 'pixel_values') and prepare_inputs.pixel_values is not None:
            prepare_inputs.pixel_values = prepare_inputs.pixel_values.to(dtype=model_dtype)
        
        inputs = prepare_inputs
        inputs._temp_image_created = temp_image_created
        inputs._temp_image_path = image_path if temp_image_created else None
    
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    return inputs


# ============================================================================
# Input Preparation - WITH Response Appended (for Fast Mode)
# ============================================================================

def prepare_vlm_input_with_response(
    question: str, 
    image: Image.Image, 
    generated_response: str,
    processor, 
    model_type: str, 
    device: str, 
    model=None
) -> Tuple[Any, int]:
    """
    Prepare VLM input with the generated response already appended.
    Returns (inputs, prompt_length) where prompt_length is the number of tokens
    before the generated response starts.
    
    This allows us to do a SINGLE forward pass and extract hidden states for all tokens.
    """
    
    if model_type == 'qwen':
        # Build messages WITH assistant response
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question},
            ]},
            {"role": "assistant", "content": generated_response}
        ]
        
        # Get full text with response
        text_with_response = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        
        # Get prompt-only text to find where response starts
        messages_prompt_only = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question},
            ]},
        ]
        text_prompt_only = processor.apply_chat_template(
            messages_prompt_only, tokenize=False, add_generation_prompt=True
        )
        
        image_inputs, video_inputs = process_vision_info(messages)
        
        inputs = processor(
            text=[text_with_response],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(device)
        
        # Calculate prompt length by tokenizing prompt-only
        prompt_inputs = processor(
            text=[text_prompt_only],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        prompt_length = prompt_inputs['input_ids'].shape[1]
        
        return inputs, prompt_length
        
    elif model_type == 'llava':
        # Build conversation with response
        conversation_with_response = [
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": question + "\n\n" + APPENDED_SYSTEM_PROMPT},
            ]},
            {"role": "assistant", "content": generated_response}
        ]
        
        conversation_prompt_only = [
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": question + "\n\n" + APPENDED_SYSTEM_PROMPT},
            ]},
        ]
        
        text_with_response = processor.apply_chat_template(
            conversation_with_response, add_generation_prompt=False, tokenize=False
        )
        text_prompt_only = processor.apply_chat_template(
            conversation_prompt_only, add_generation_prompt=True, tokenize=False
        )
        
        if isinstance(image, Image.Image):
            image_to_process = image
        else:
            image_to_process = Image.open(image) if isinstance(image, str) else image
        
        inputs = processor(images=image_to_process, text=text_with_response, return_tensors="pt")
        prompt_inputs = processor(images=image_to_process, text=text_prompt_only, return_tensors="pt")
        
        if model is not None:
            model_device = next(model.parameters()).device
            model_dtype = next(model.parameters()).dtype
            inputs = {k: v.to(device=model_device, dtype=model_dtype if k == "pixel_values" else None) 
                     if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        
        prompt_length = prompt_inputs['input_ids'].shape[1]
        return inputs, prompt_length
        
    elif model_type == 'gemma':
        conversation_with_response = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": question},
            ]},
            {"role": "assistant", "content": generated_response}
        ]
        
        conversation_prompt_only = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": question},
            ]},
        ]
        
        text_with_response = processor.apply_chat_template(
            conversation_with_response, add_generation_prompt=False, tokenize=False
        )
        text_prompt_only = processor.apply_chat_template(
            conversation_prompt_only, add_generation_prompt=True, tokenize=False
        )
        
        if isinstance(image, Image.Image):
            image_to_process = image
        else:
            image_to_process = Image.open(image) if isinstance(image, str) else image
        
        inputs = processor(images=image_to_process, text=text_with_response, return_tensors="pt")
        prompt_inputs = processor(images=image_to_process, text=text_prompt_only, return_tensors="pt")
        
        if model is not None:
            model_device = next(model.parameters()).device
            model_dtype = next(model.parameters()).dtype
            inputs = {k: v.to(device=model_device, dtype=model_dtype if k == "pixel_values" else None) 
                     if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        
        prompt_length = prompt_inputs['input_ids'].shape[1]
        return inputs, prompt_length
        
    elif model_type == 'internvl':
        messages_with_response = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": question},
            ]},
            {"role": "assistant", "content": generated_response}
        ]
        
        messages_prompt_only = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": question},
            ]},
        ]
        
        text_with_response = processor.apply_chat_template(
            messages_with_response, add_generation_prompt=False, tokenize=False
        )
        text_prompt_only = processor.apply_chat_template(
            messages_prompt_only, add_generation_prompt=True, tokenize=False
        )
        
        if isinstance(image, Image.Image):
            image_to_process = image
        else:
            image_to_process = Image.open(image) if isinstance(image, str) else image
        
        inputs = processor(images=image_to_process, text=text_with_response, return_tensors="pt")
        prompt_inputs = processor(images=image_to_process, text=text_prompt_only, return_tensors="pt")
        
        if model is not None:
            model_device = next(model.parameters()).device
            model_dtype = next(model.parameters()).dtype
            inputs = {k: v.to(device=model_device, dtype=model_dtype if k == "pixel_values" else None) 
                     if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        
        prompt_length = prompt_inputs['input_ids'].shape[1]
        return inputs, prompt_length
        
    elif model_type == 'deepseek':
        import tempfile
        if isinstance(image, Image.Image):
            temp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
            image.save(temp_file.name)
            image_path = temp_file.name
            temp_image_created = True
        else:
            image_path = image
            temp_image_created = False
        
        # With response
        conversation_with_response = [
            {"role": "<|User|>", "content": f"<image>\n{question}", "images": [image_path]},
            {"role": "<|Assistant|>", "content": generated_response},
        ]
        
        # Prompt only
        conversation_prompt_only = [
            {"role": "<|User|>", "content": f"<image>\n{question}", "images": [image_path]},
            {"role": "<|Assistant|>", "content": ""},
        ]
        
        pil_images = load_pil_images(conversation_with_response)
        inputs = processor(
            conversations=conversation_with_response,
            images=pil_images,
            force_batchify=True,
            system_prompt=SYSTEM_PROMPT
        ).to(device)
        
        pil_images_prompt = load_pil_images(conversation_prompt_only)
        prompt_inputs = processor(
            conversations=conversation_prompt_only,
            images=pil_images_prompt,
            force_batchify=True,
            system_prompt=SYSTEM_PROMPT
        )
        
        if hasattr(inputs, 'pixel_values') and inputs.pixel_values is not None:
            inputs.pixel_values = inputs.pixel_values.to(dtype=torch.bfloat16)
        
        inputs._temp_image_created = temp_image_created
        inputs._temp_image_path = image_path if temp_image_created else None
        
        prompt_length = prompt_inputs.input_ids.shape[1]
        return inputs, prompt_length
    
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


# ============================================================================
# Jacobian Computation - MANUAL MODE (Backpropagation)
# ============================================================================

def compute_jacobian_manual(model, hidden_state: torch.Tensor, token_id: int, 
                            model_type: str) -> torch.Tensor:
    """
    Compute Jacobian using backpropagation (original CCPS method).
    
    Args:
        model: The VLM model
        hidden_state: [1, hidden_dim] tensor - the hidden state that produced the token
        token_id: The token ID that was generated
        model_type: Type of model ('qwen', 'llava', etc.)
    
    Returns:
        jacobian_vector: [hidden_dim] tensor - gradient direction
    """
    model.zero_grad()
    
    # Clone and enable gradients
    hidden_state_grad = hidden_state.clone().detach().requires_grad_(True)
    
    # Get the lm_head
    lm_head = get_lm_head(model, model_type)
    
    # Pass through lm_head
    logits = lm_head(hidden_state_grad)  # [1, vocab_size]
    
    # Compute loss (negative log probability of the actual token)
    log_probs = F.log_softmax(logits, dim=-1)
    loss = -log_probs[0, token_id]
    
    # Backward pass
    loss.backward()
    
    if hidden_state_grad.grad is not None:
        jacobian_vector = hidden_state_grad.grad.clone().detach()
    else:
        jacobian_vector = torch.zeros_like(hidden_state_grad)
    
    model.zero_grad()
    return jacobian_vector.squeeze(0)  # [hidden_dim]


# ============================================================================
# Jacobian Computation - ANALYTICAL MODE (Fast, no backprop)
# ============================================================================

def compute_jacobian_analytical(hidden_state: torch.Tensor, logits: torch.Tensor, 
                                 token_id: int, lm_head_weight: torch.Tensor) -> torch.Tensor:
    """
    Compute Jacobian analytically using closed-form gradient of cross-entropy loss.
    
    The gradient of CE loss w.r.t. hidden state is:
        g = W^T (p - y)
    Where:
        W = lm_head weights [V, D]
        p = softmax probabilities [V]
        y = one-hot target vector [V] (1 at generated token, 0 elsewhere)
    
    Simplified:
        g = sum(p_i * w_i) - w_target
    
    Args:
        hidden_state: [hidden_dim] tensor
        logits: [vocab_size] tensor
        token_id: The token ID that was actually generated
        lm_head_weight: [vocab_size, hidden_dim] tensor (W matrix)
    
    Returns:
        jacobian_vector: [hidden_dim] tensor - gradient direction
    """
    # Compute softmax probabilities
    probs = F.softmax(logits, dim=-1)  # [V]
    
    # Compute weighted sum: sum(p_i * w_i)
    # probs: [V], lm_head_weight: [V, D]
    # Result: [D]
    weighted_sum = torch.matmul(probs, lm_head_weight)  # [D]
    
    # Get embedding of generated token
    w_target = lm_head_weight[token_id]  # [D]
    
    # Gradient: g = weighted_sum - w_target
    gradient = weighted_sum - w_target  # [D]
    
    return gradient


def compute_jacobians_batch_analytical(hidden_states: torch.Tensor, logits: torch.Tensor,
                                        token_ids: torch.Tensor, lm_head_weight: torch.Tensor) -> torch.Tensor:
    """
    Compute Jacobians for all tokens in batch using analytical gradient.
    
    Args:
        hidden_states: [T, D] tensor - hidden states for T tokens
        logits: [T, V] tensor - logits for T tokens
        token_ids: [T] tensor - generated token IDs
        lm_head_weight: [V, D] tensor - lm_head weights
    
    Returns:
        jacobians: [T, D] tensor - gradient vectors for all tokens
    """
    T, D = hidden_states.shape
    V = logits.shape[1]
    
    # Ensure lm_head_weight matches the dtype and device of logits (since we multiply probs with it)
    device = logits.device
    dtype = logits.dtype
    lm_head_weight = lm_head_weight.to(device=device, dtype=dtype)
    
    # Compute softmax probabilities for all tokens
    probs = F.softmax(logits, dim=-1)  # [T, V]
    
    # Ensure probs and lm_head_weight have matching dtype (softmax might change dtype)
    if probs.dtype != lm_head_weight.dtype:
        # Convert lm_head_weight to match probs dtype
        lm_head_weight = lm_head_weight.to(dtype=probs.dtype)
    
    # Compute weighted sums: [T, V] @ [V, D] = [T, D]
    weighted_sums = torch.matmul(probs, lm_head_weight)  # [T, D]
    
    # Get embeddings of generated tokens
    w_targets = lm_head_weight[token_ids]  # [T, D]
    
    # Gradients: g = weighted_sum - w_target
    gradients = weighted_sums - w_targets  # [T, D]
    
    return gradients


# ============================================================================
# Perturbation and Feature Extraction
# ============================================================================

def generate_perturbation_trajectory(hidden_state: torch.Tensor, jacobian_vector: torch.Tensor,
                                     pei_radius: float, pei_steps: int) -> List[torch.Tensor]:
    """
    Generate perturbation trajectory along Jacobian direction.
    
    Args:
        hidden_state: [hidden_dim] tensor
        jacobian_vector: [hidden_dim] tensor
        pei_radius: Maximum perturbation radius (epsilon_max)
        pei_steps: Number of perturbation steps (S)
    
    Returns:
        List of perturbed hidden states [hidden_dim] each
    """
    perturbed_states = []
    
    jacobian_norm = torch.norm(jacobian_vector)
    
    if jacobian_norm.item() > 1e-9:
        jacobian_direction = jacobian_vector / jacobian_norm
        delta_r = pei_radius / pei_steps
        
        for k in range(1, pei_steps + 1):
            r_k = k * delta_r
            perturbed_hidden = hidden_state + r_k * jacobian_direction
            perturbed_states.append(perturbed_hidden)
    else:
        for k in range(1, pei_steps + 1):
            perturbed_states.append(hidden_state.clone())
    
    return perturbed_states


def generate_perturbation_trajectories_batch(hidden_states: torch.Tensor, jacobians: torch.Tensor,
                                              pei_radius: float, pei_steps: int) -> torch.Tensor:
    """
    Generate perturbation trajectories for all tokens in batch.
    
    Args:
        hidden_states: [T, D] tensor
        jacobians: [T, D] tensor
        pei_radius: Maximum perturbation radius
        pei_steps: Number of perturbation steps
    
    Returns:
        perturbed_states: [T, S, D] tensor - perturbed hidden states
    """
    T, D = hidden_states.shape
    
    # Compute Jacobian norms
    jacobian_norms = torch.norm(jacobians, dim=1, keepdim=True)  # [T, 1]
    
    # Normalize Jacobians (handle zero norms)
    safe_norms = torch.where(jacobian_norms > 1e-9, jacobian_norms, torch.ones_like(jacobian_norms))
    jacobian_directions = jacobians / safe_norms  # [T, D]
    
    # Zero out directions for zero-norm Jacobians
    zero_mask = (jacobian_norms <= 1e-9).squeeze(-1)  # [T]
    jacobian_directions[zero_mask] = 0
    
    # Generate perturbation radii
    delta_r = pei_radius / pei_steps
    radii = torch.arange(1, pei_steps + 1, device=hidden_states.device, dtype=hidden_states.dtype) * delta_r  # [S]
    
    # Expand for batch computation
    # hidden_states: [T, D] -> [T, 1, D]
    # jacobian_directions: [T, D] -> [T, 1, D]
    # radii: [S] -> [1, S, 1]
    hidden_states_expanded = hidden_states.unsqueeze(1)  # [T, 1, D]
    directions_expanded = jacobian_directions.unsqueeze(1)  # [T, 1, D]
    radii_expanded = radii.view(1, -1, 1)  # [1, S, 1]
    
    # Compute all perturbed states: [T, S, D]
    perturbed_states = hidden_states_expanded + radii_expanded * directions_expanded
    
    return perturbed_states


def get_perturbed_logits_batch(model, perturbed_hidden: torch.Tensor, model_type: str) -> torch.Tensor:
    """
    Get logits for perturbed hidden states in batch.
    
    Args:
        model: The VLM model
        perturbed_hidden: [T, S, D] tensor - perturbed hidden states
        model_type: Model type string
    
    Returns:
        perturbed_logits: [T, S, V] tensor - logits for all perturbed states
    """
    T, S, D = perturbed_hidden.shape
    lm_head = get_lm_head(model, model_type)
    
    with torch.no_grad():
        # Flatten: [T*S, D]
        flat = perturbed_hidden.view(-1, D)
        logits_flat = lm_head(flat)  # [T*S, V]
        logits = logits_flat.view(T, S, -1)  # [T, S, V]
    
    return logits


def get_logits_from_hidden_state(model, hidden_state: torch.Tensor, model_type: str) -> torch.Tensor:
    """Get logits from a hidden state using the model's lm_head"""
    with torch.no_grad():
        lm_head = get_lm_head(model, model_type)
        if hidden_state.dim() == 1:
            logits = lm_head(hidden_state.unsqueeze(0)).squeeze(0)
        else:
            logits = lm_head(hidden_state)
            if logits.dim() == 3:
                logits = logits.squeeze(1)
    return logits


def get_logits_from_hidden_states_batch(model, hidden_states: torch.Tensor, model_type: str) -> torch.Tensor:
    """
    Get logits from multiple hidden states in batch.
    
    Args:
        model: The VLM model
        hidden_states: [T, D] or [T, S, D] tensor
        model_type: Type of model
    
    Returns:
        logits: Same leading dimensions as input, with vocab_size at end
    """
    with torch.no_grad():
        lm_head = get_lm_head(model, model_type)
        original_shape = hidden_states.shape
        
        if hidden_states.dim() == 3:
            # [T, S, D] -> [T*S, D]
            T, S, D = hidden_states.shape
            hidden_states_flat = hidden_states.view(-1, D)
            logits_flat = lm_head(hidden_states_flat)  # [T*S, V]
            logits = logits_flat.view(T, S, -1)  # [T, S, V]
        else:
            # [T, D] -> [T, V]
            logits = lm_head(hidden_states)
    
    return logits


# ============================================================================
# Feature Extraction
# ============================================================================

def extract_ccps_features_for_token(
    original_hidden_state: torch.Tensor,  # [hidden_dim]
    original_logits: torch.Tensor,         # [vocab_size]
    jacobian_vector: torch.Tensor,         # [hidden_dim]
    perturbed_hidden_states: List[torch.Tensor],  # List of [hidden_dim]
    perturbed_logits_list: List[torch.Tensor],    # List of [vocab_size]
    actual_token_id: int,
    pei_radius: float,
    pei_steps: int,
) -> Dict[str, float]:
    """
    Extract all CCPS features for a single token.
    
    Returns:
        Dictionary of feature_name -> feature_value
    """
    features = {}
    
    # =========================================================================
    # I. Original State Features
    # =========================================================================
    
    probs_0 = F.softmax(original_logits, dim=-1)
    log_probs_0 = F.log_softmax(original_logits, dim=-1)
    
    top2_values, top2_indices = torch.topk(original_logits, 2)
    argmax_0 = top2_indices[0].item()
    second_best_0 = top2_indices[1].item() if len(top2_indices) > 1 else argmax_0
    
    features['original_log_prob_actual'] = log_probs_0[actual_token_id].item()
    features['original_prob_actual'] = probs_0[actual_token_id].item()
    features['original_logit_actual'] = original_logits[actual_token_id].item()
    features['original_prob_argmax'] = probs_0[argmax_0].item()
    features['original_logit_argmax'] = original_logits[argmax_0].item()
    
    features['original_entropy'] = -torch.sum(probs_0 * torch.log(probs_0 + 1e-9)).item()
    
    features['original_margin_logit_top1_top2'] = original_logits[argmax_0].item() - original_logits[second_best_0].item()
    features['original_margin_prob_top1_top2'] = probs_0[argmax_0].item() - probs_0[second_best_0].item()
    
    features['original_norm_logits_L2'] = torch.norm(original_logits).item()
    features['original_std_logits'] = torch.std(original_logits).item()
    features['original_norm_hidden_state_L2'] = torch.norm(original_hidden_state).item()
    
    features['is_actual_token_original_argmax'] = int(actual_token_id == argmax_0)
    
    # =========================================================================
    # II. Overall Perturbation Features
    # =========================================================================
    
    jacobian_norm = torch.norm(jacobian_vector).item()
    features['jacobian_norm_token'] = jacobian_norm
    
    epsilon_to_flip = float('inf')
    delta_r = pei_radius / pei_steps
    for k, perturbed_logits in enumerate(perturbed_logits_list):
        perturbed_argmax = torch.argmax(perturbed_logits).item()
        if perturbed_argmax != argmax_0:
            epsilon_to_flip = (k + 1) * delta_r
            break
    features['epsilon_to_flip_token'] = epsilon_to_flip
    
    log_p_original = log_probs_0[actual_token_id].item()
    f_values = [0.0]
    
    for perturbed_logits in perturbed_logits_list:
        perturbed_log_probs = F.log_softmax(perturbed_logits, dim=-1)
        log_p_perturbed = perturbed_log_probs[actual_token_id].item()
        f_k = log_p_original - log_p_perturbed
        f_values.append(max(0.0, f_k))
    
    pei_value = 0.0
    for k in range(pei_steps):
        pei_value += (f_values[k] + f_values[k + 1]) / 2.0
    pei_value = pei_value / pei_steps if pei_steps > 0 else 0.0
    features['pei_value_token'] = pei_value
    
    # =========================================================================
    # III. Perturbed State Features (statistics across S steps)
    # =========================================================================
    
    perturbed_metrics = {
        'perturbed_log_prob_actual': [],
        'perturbed_prob_actual': [],
        'perturbed_logit_actual': [],
        'delta_log_prob_actual_from_original': [],
        'perturbed_prob_argmax': [],
        'perturbed_logit_argmax': [],
        'did_argmax_change_from_original': [],
        'perturbed_entropy': [],
        'perturbed_margin_logit_top1_top2': [],
        'perturbed_norm_logits_L2': [],
        'kl_div_perturbed_from_original': [],
        'js_div_perturbed_from_original': [],
        'cosine_sim_logits_perturbed_to_original': [],
        'cosine_sim_hidden_perturbed_to_original': [],
        'l2_dist_hidden_perturbed_from_original': [],
    }
    
    for i, (perturbed_hidden, perturbed_logits) in enumerate(zip(perturbed_hidden_states, perturbed_logits_list)):
        probs_p = F.softmax(perturbed_logits, dim=-1)
        log_probs_p = F.log_softmax(perturbed_logits, dim=-1)
        
        perturbed_metrics['perturbed_log_prob_actual'].append(log_probs_p[actual_token_id].item())
        perturbed_metrics['perturbed_prob_actual'].append(probs_p[actual_token_id].item())
        perturbed_metrics['perturbed_logit_actual'].append(perturbed_logits[actual_token_id].item())
        
        delta_log_prob = log_probs_0[actual_token_id].item() - log_probs_p[actual_token_id].item()
        perturbed_metrics['delta_log_prob_actual_from_original'].append(delta_log_prob)
        
        argmax_p = torch.argmax(perturbed_logits).item()
        perturbed_metrics['perturbed_prob_argmax'].append(probs_p[argmax_p].item())
        perturbed_metrics['perturbed_logit_argmax'].append(perturbed_logits[argmax_p].item())
        perturbed_metrics['did_argmax_change_from_original'].append(int(argmax_p != argmax_0))
        
        entropy_p = -torch.sum(probs_p * torch.log(probs_p + 1e-9)).item()
        perturbed_metrics['perturbed_entropy'].append(entropy_p)
        
        top2_p_values, top2_p_indices = torch.topk(perturbed_logits, 2)
        if len(top2_p_indices) >= 2:
            margin = perturbed_logits[top2_p_indices[0]].item() - perturbed_logits[top2_p_indices[1]].item()
        else:
            margin = 0.0
        perturbed_metrics['perturbed_margin_logit_top1_top2'].append(margin)
        
        perturbed_metrics['perturbed_norm_logits_L2'].append(torch.norm(perturbed_logits).item())
        
        kl_div = F.kl_div(log_probs_p, probs_0, reduction='sum').item()
        perturbed_metrics['kl_div_perturbed_from_original'].append(kl_div)
        
        m_probs = 0.5 * (probs_0 + probs_p)
        js_div = 0.5 * F.kl_div(log_probs_0, m_probs, reduction='sum') + \
                 0.5 * F.kl_div(log_probs_p, m_probs, reduction='sum')
        perturbed_metrics['js_div_perturbed_from_original'].append(js_div.item())
        
        cos_sim_logits = F.cosine_similarity(original_logits.unsqueeze(0), 
                                              perturbed_logits.unsqueeze(0)).item()
        perturbed_metrics['cosine_sim_logits_perturbed_to_original'].append(cos_sim_logits)
        
        cos_sim_hidden = F.cosine_similarity(original_hidden_state.unsqueeze(0),
                                              perturbed_hidden.unsqueeze(0)).item()
        perturbed_metrics['cosine_sim_hidden_perturbed_to_original'].append(cos_sim_hidden)
        
        l2_dist = torch.norm(perturbed_hidden - original_hidden_state).item()
        perturbed_metrics['l2_dist_hidden_perturbed_from_original'].append(l2_dist)
    
    for metric_name, values in perturbed_metrics.items():
        if values:
            features[f'{metric_name}_mean'] = np.mean(values)
            features[f'{metric_name}_std'] = np.std(values) if len(values) > 1 else 0.0
            features[f'{metric_name}_min'] = np.min(values)
            features[f'{metric_name}_max'] = np.max(values)
        else:
            features[f'{metric_name}_mean'] = 0.0
            features[f'{metric_name}_std'] = 0.0
            features[f'{metric_name}_min'] = 0.0
            features[f'{metric_name}_max'] = 0.0
    
    return features


def extract_ccps_features_batch(
    hidden_states: torch.Tensor,      # [T, D]
    original_logits: torch.Tensor,     # [T, V]
    jacobians: torch.Tensor,           # [T, D]
    perturbed_hidden_states: torch.Tensor,  # [T, S, D]
    perturbed_logits: torch.Tensor,    # [T, S, V]
    token_ids: torch.Tensor,           # [T]
    pei_radius: float,
    pei_steps: int,
) -> List[Dict[str, float]]:
    """
    Extract CCPS features for all tokens in batch (vectorized where possible).
    
    Returns:
        List of feature dictionaries, one per token
    """
    T = hidden_states.shape[0]
    all_features = []
    
    # Pre-compute common values
    probs_0 = F.softmax(original_logits, dim=-1)  # [T, V]
    log_probs_0 = F.log_softmax(original_logits, dim=-1)  # [T, V]
    
    top2_values, top2_indices = torch.topk(original_logits, 2, dim=-1)  # [T, 2]
    argmax_0 = top2_indices[:, 0]  # [T]
    second_best_0 = top2_indices[:, 1]  # [T]
    
    # Perturbed probs and log_probs
    perturbed_probs = F.softmax(perturbed_logits, dim=-1)  # [T, S, V]
    perturbed_log_probs = F.log_softmax(perturbed_logits, dim=-1)  # [T, S, V]
    
    delta_r = pei_radius / pei_steps
    
    for t in range(T):
        features = {}
        token_id = token_ids[t].item()
        
        # Original state features
        features['original_log_prob_actual'] = log_probs_0[t, token_id].item()
        features['original_prob_actual'] = probs_0[t, token_id].item()
        features['original_logit_actual'] = original_logits[t, token_id].item()
        features['original_prob_argmax'] = probs_0[t, argmax_0[t]].item()
        features['original_logit_argmax'] = original_logits[t, argmax_0[t]].item()
        features['original_entropy'] = -torch.sum(probs_0[t] * torch.log(probs_0[t] + 1e-9)).item()
        features['original_margin_logit_top1_top2'] = (original_logits[t, argmax_0[t]] - original_logits[t, second_best_0[t]]).item()
        features['original_margin_prob_top1_top2'] = (probs_0[t, argmax_0[t]] - probs_0[t, second_best_0[t]]).item()
        features['original_norm_logits_L2'] = torch.norm(original_logits[t]).item()
        features['original_std_logits'] = torch.std(original_logits[t]).item()
        features['original_norm_hidden_state_L2'] = torch.norm(hidden_states[t]).item()
        features['is_actual_token_original_argmax'] = int(token_id == argmax_0[t].item())
        
        # Jacobian norm
        features['jacobian_norm_token'] = torch.norm(jacobians[t]).item()
        
        # Epsilon to flip
        epsilon_to_flip = float('inf')
        perturbed_argmaxes = torch.argmax(perturbed_logits[t], dim=-1)  # [S]
        for k in range(pei_steps):
            if perturbed_argmaxes[k].item() != argmax_0[t].item():
                epsilon_to_flip = (k + 1) * delta_r
                break
        features['epsilon_to_flip_token'] = epsilon_to_flip
        
        # PEI
        log_p_original = log_probs_0[t, token_id].item()
        f_values = [0.0]
        for k in range(pei_steps):
            log_p_perturbed = perturbed_log_probs[t, k, token_id].item()
            f_k = max(0.0, log_p_original - log_p_perturbed)
            f_values.append(f_k)
        pei_value = sum((f_values[k] + f_values[k + 1]) / 2.0 for k in range(pei_steps)) / pei_steps if pei_steps > 0 else 0.0
        features['pei_value_token'] = pei_value
        
        # Perturbed features
        perturbed_metrics = {
            'perturbed_log_prob_actual': [],
            'perturbed_prob_actual': [],
            'perturbed_logit_actual': [],
            'delta_log_prob_actual_from_original': [],
            'perturbed_prob_argmax': [],
            'perturbed_logit_argmax': [],
            'did_argmax_change_from_original': [],
            'perturbed_entropy': [],
            'perturbed_margin_logit_top1_top2': [],
            'perturbed_norm_logits_L2': [],
            'kl_div_perturbed_from_original': [],
            'js_div_perturbed_from_original': [],
            'cosine_sim_logits_perturbed_to_original': [],
            'cosine_sim_hidden_perturbed_to_original': [],
            'l2_dist_hidden_perturbed_from_original': [],
        }
        
        for s in range(pei_steps):
            perturbed_metrics['perturbed_log_prob_actual'].append(perturbed_log_probs[t, s, token_id].item())
            perturbed_metrics['perturbed_prob_actual'].append(perturbed_probs[t, s, token_id].item())
            perturbed_metrics['perturbed_logit_actual'].append(perturbed_logits[t, s, token_id].item())
            
            delta_log_prob = log_probs_0[t, token_id].item() - perturbed_log_probs[t, s, token_id].item()
            perturbed_metrics['delta_log_prob_actual_from_original'].append(delta_log_prob)
            
            argmax_p = perturbed_argmaxes[s].item()
            perturbed_metrics['perturbed_prob_argmax'].append(perturbed_probs[t, s, argmax_p].item())
            perturbed_metrics['perturbed_logit_argmax'].append(perturbed_logits[t, s, argmax_p].item())
            perturbed_metrics['did_argmax_change_from_original'].append(int(argmax_p != argmax_0[t].item()))
            
            entropy_p = -torch.sum(perturbed_probs[t, s] * torch.log(perturbed_probs[t, s] + 1e-9)).item()
            perturbed_metrics['perturbed_entropy'].append(entropy_p)
            
            top2_p = torch.topk(perturbed_logits[t, s], 2)
            margin = (top2_p.values[0] - top2_p.values[1]).item() if len(top2_p.values) >= 2 else 0.0
            perturbed_metrics['perturbed_margin_logit_top1_top2'].append(margin)
            
            perturbed_metrics['perturbed_norm_logits_L2'].append(torch.norm(perturbed_logits[t, s]).item())
            
            kl_div = F.kl_div(perturbed_log_probs[t, s], probs_0[t], reduction='sum').item()
            perturbed_metrics['kl_div_perturbed_from_original'].append(kl_div)
            
            m_probs = 0.5 * (probs_0[t] + perturbed_probs[t, s])
            js_div = (0.5 * F.kl_div(log_probs_0[t], m_probs, reduction='sum') + 
                     0.5 * F.kl_div(perturbed_log_probs[t, s], m_probs, reduction='sum')).item()
            perturbed_metrics['js_div_perturbed_from_original'].append(js_div)
            
            cos_sim_logits = F.cosine_similarity(original_logits[t].unsqueeze(0), 
                                                  perturbed_logits[t, s].unsqueeze(0)).item()
            perturbed_metrics['cosine_sim_logits_perturbed_to_original'].append(cos_sim_logits)
            
            cos_sim_hidden = F.cosine_similarity(hidden_states[t].unsqueeze(0),
                                                  perturbed_hidden_states[t, s].unsqueeze(0)).item()
            perturbed_metrics['cosine_sim_hidden_perturbed_to_original'].append(cos_sim_hidden)
            
            l2_dist = torch.norm(perturbed_hidden_states[t, s] - hidden_states[t]).item()
            perturbed_metrics['l2_dist_hidden_perturbed_from_original'].append(l2_dist)
        
        # Compute statistics
        for metric_name, values in perturbed_metrics.items():
            if values:
                features[f'{metric_name}_mean'] = np.mean(values)
                features[f'{metric_name}_std'] = np.std(values) if len(values) > 1 else 0.0
                features[f'{metric_name}_min'] = np.min(values)
                features[f'{metric_name}_max'] = np.max(values)
            else:
                features[f'{metric_name}_mean'] = 0.0
                features[f'{metric_name}_std'] = 0.0
                features[f'{metric_name}_min'] = 0.0
                features[f'{metric_name}_max'] = 0.0
        
        all_features.append(features)
    
    return all_features


# ============================================================================
# Main Processing Functions
# ============================================================================

def process_sample_manual(
    npz_path: Path,
    vlcb_dataset,
    hash_id_to_idx: Dict[str, int],
    model,
    processor,
    model_type: str,
    dtype: torch.dtype,
    device: str,
    pei_radius: float,
    pei_steps: int,
) -> Optional[Dict[str, Any]]:
    """
    Process a single sample using MANUAL (backpropagation) mode.
    This is the original CCPS method - slower but exact.
    """
    # Load the existing npz file
    try:
        data = np.load(npz_path, allow_pickle=True)
    except Exception as e:
        logger.error(f"Failed to load {npz_path}: {e}")
        return None
    
    hash_id = str(data['hash_id'])
    question = str(data['question'])
    answer = str(data['answer'])
    generated_response = str(data.get('generated_response', ''))
    is_correct = data.get('is_correct')
    token_ids = data['token_ids']
    token_strs = data['token_strs']
    
    if is_correct is None or (isinstance(is_correct, np.ndarray) and is_correct.item() is None):
        logger.warning(f"Sample {hash_id} has no correctness label, skipping")
        return None
    
    if hash_id not in hash_id_to_idx:
        logger.warning(f"Sample {hash_id} not found in VLCB dataset")
        return None
    
    vlcb_idx = hash_id_to_idx[hash_id]
    vlcb_sample = vlcb_dataset[vlcb_idx]
    image_raw = vlcb_sample['image']
    
    # Resize image if needed (prevents OOM on large images)
    image = resize_image_if_needed(image_raw, max_dim=args.max_image_dim)
    
    try:
        inputs = prepare_vlm_input(question, image, processor, model_type, device, model=model)
    except Exception as e:
        logger.error(f"Failed to prepare input for {hash_id}: {e}")
        return None
    
    all_token_features = []
    
    for token_idx, token_id in enumerate(token_ids):
        token_id = int(token_id)
        
        try:
            # Forward pass
            with torch.no_grad():
                if model_type == 'deepseek':
                    inputs_dict = {
                        'input_ids': inputs.input_ids,
                        'attention_mask': inputs.attention_mask if hasattr(inputs, 'attention_mask') else None,
                    }
                    if hasattr(inputs, 'pixel_values'):
                        inputs_dict['pixel_values'] = inputs.pixel_values
                    
                    current_inputs_embeds = model.prepare_inputs_embeds(**inputs_dict)
                    outputs = model.language(
                        inputs_embeds=current_inputs_embeds,
                        attention_mask=inputs_dict.get('attention_mask'),
                        output_hidden_states=True,
                        return_dict=True,
                    )
                else:
                    if isinstance(inputs, dict):
                        outputs = model(**inputs, output_hidden_states=True, return_dict=True)
                    else:
                        outputs = model(
                            input_ids=inputs.input_ids,
                            attention_mask=inputs.attention_mask if hasattr(inputs, 'attention_mask') else None,
                            pixel_values=inputs.pixel_values if hasattr(inputs, 'pixel_values') else None,
                            output_hidden_states=True,
                            return_dict=True,
                        )
                
                hidden_state = outputs.hidden_states[-1][0, -1, :]
            
            # Get original logits
            original_logits = get_logits_from_hidden_state(model, hidden_state, model_type)
            
            # Compute Jacobian using backpropagation
            jacobian_vector = compute_jacobian_manual(model, hidden_state.unsqueeze(0), token_id, model_type)
            
            # Generate perturbation trajectory
            perturbed_hidden_states = generate_perturbation_trajectory(
                hidden_state, jacobian_vector, pei_radius, pei_steps
            )
            
            # Get logits for perturbed states
            perturbed_logits_list = []
            for perturbed_hidden in perturbed_hidden_states:
                perturbed_logits = get_logits_from_hidden_state(model, perturbed_hidden, model_type)
                perturbed_logits_list.append(perturbed_logits)
            
            # Extract features
            token_features = extract_ccps_features_for_token(
                original_hidden_state=hidden_state,
                original_logits=original_logits,
                jacobian_vector=jacobian_vector,
                perturbed_hidden_states=perturbed_hidden_states,
                perturbed_logits_list=perturbed_logits_list,
                actual_token_id=token_id,
                pei_radius=pei_radius,
                pei_steps=pei_steps,
            )
            
            token_features['token_idx'] = token_idx
            token_features['token_id'] = token_id
            token_features['token_str'] = str(token_strs[token_idx]) if token_idx < len(token_strs) else ''
            
            all_token_features.append(token_features)
            
            # Update input for next token
            next_token_tensor = torch.tensor([[token_id]], device=device)
            
            if model_type == 'deepseek':
                inputs.input_ids = torch.cat([inputs.input_ids, next_token_tensor], dim=1)
                if hasattr(inputs, 'attention_mask'):
                    inputs.attention_mask = torch.cat([
                        inputs.attention_mask,
                        torch.ones((1, 1), device=device, dtype=inputs.attention_mask.dtype)
                    ], dim=1)
            elif isinstance(inputs, dict):
                if 'input_ids' in inputs:
                    inputs['input_ids'] = torch.cat([inputs['input_ids'], next_token_tensor], dim=1)
                    if 'attention_mask' in inputs:
                        inputs['attention_mask'] = torch.cat([
                            inputs['attention_mask'], 
                            torch.ones((1, 1), device=device, dtype=inputs['attention_mask'].dtype)
                        ], dim=1)
            else:
                if hasattr(inputs, 'input_ids'):
                    inputs.input_ids = torch.cat([inputs.input_ids, next_token_tensor], dim=1)
                if hasattr(inputs, 'attention_mask'):
                    inputs.attention_mask = torch.cat([
                        inputs.attention_mask,
                        torch.ones((1, 1), device=device, dtype=inputs.attention_mask.dtype)
                    ], dim=1)
            
        except Exception as e:
            logger.error(f"Error processing token {token_idx} for {hash_id}: {e}")
            continue
    
    # Cleanup temp file for DeepSeek
    if model_type == 'deepseek' and hasattr(inputs, '_temp_image_created') and inputs._temp_image_created:
        try:
            if hasattr(inputs, '_temp_image_path') and inputs._temp_image_path:
                os.unlink(inputs._temp_image_path)
        except:
            pass
    
    if not all_token_features:
        logger.warning(f"No features extracted for {hash_id}")
        return None
    
    return {
        'hash_id': hash_id,
        'question': question,
        'answer': answer,
        'generated_response': generated_response,
        'is_correct': bool(is_correct),
        'num_tokens': len(all_token_features),
        'token_features': all_token_features,
    }


def process_sample_analytical(
    npz_path: Path,
    vlcb_dataset,
    hash_id_to_idx: Dict[str, int],
    model,
    processor,
    model_type: str,
    dtype: torch.dtype,
    device: str,
    pei_radius: float,
    pei_steps: int,
    lm_head_weight: torch.Tensor,
) -> Optional[Dict[str, Any]]:
    """
    Process a single sample using ANALYTICAL (fast, no backprop) mode.
    
    KEY OPTIMIZATION: Uses a SINGLE forward pass instead of T sequential passes.
    1. Prepares input with full generated response appended
    2. Does ONE forward pass to get all hidden states
    3. Computes analytical gradients in batch
    4. Extracts all features in batch
    
    This is ~T times faster than sequential forward passes.
    """
    # Load the existing npz file
    try:
        data = np.load(npz_path, allow_pickle=True)
    except Exception as e:
        logger.error(f"Failed to load {npz_path}: {e}")
        return None
    
    hash_id = str(data['hash_id'])
    question = str(data['question'])
    answer = str(data['answer'])
    generated_response = str(data.get('generated_response', ''))
    is_correct = data.get('is_correct')
    token_ids_np = data['token_ids']
    token_strs = data['token_strs']
    
    if is_correct is None or (isinstance(is_correct, np.ndarray) and is_correct.item() is None):
        logger.warning(f"Sample {hash_id} has no correctness label, skipping")
        return None
    
    if hash_id not in hash_id_to_idx:
        logger.warning(f"Sample {hash_id} not found in VLCB dataset")
        return None
    
    vlcb_idx = hash_id_to_idx[hash_id]
    vlcb_sample = vlcb_dataset[vlcb_idx]
    image_raw = vlcb_sample['image']
    
    # Resize image if needed (prevents OOM on large images)
    image = resize_image_if_needed(image_raw, max_dim=args.max_image_dim)
    
    T = len(token_ids_np)
    if T == 0:
        logger.warning(f"Sample {hash_id} has no tokens")
        return None
    
    try:
        # === KEY OPTIMIZATION: Prepare input with FULL response ===
        inputs, prompt_length = prepare_vlm_input_with_response(
            question, image, generated_response, processor, model_type, device, model
        )
        
        if args.debug:
            logger.debug(f"Sample {hash_id}: prompt_length={prompt_length}, expected_tokens={T}")
        
        # === SINGLE forward pass ===
        with torch.no_grad():
            if model_type == 'deepseek':
                inputs_dict = {
                    'input_ids': inputs.input_ids,
                    'attention_mask': inputs.attention_mask if hasattr(inputs, 'attention_mask') else None,
                }
                if hasattr(inputs, 'pixel_values'):
                    inputs_dict['pixel_values'] = inputs.pixel_values
                inputs_embeds = model.prepare_inputs_embeds(**inputs_dict)
                outputs = model.language(
                    inputs_embeds=inputs_embeds,
                    attention_mask=inputs_dict.get('attention_mask'),
                    output_hidden_states=True,
                    return_dict=True,
                )
            else:
                # Convert BatchEncoding to dict to pass ALL keys (including image_grid_thw for Qwen3-VL)
                if isinstance(inputs, dict):
                    inputs_dict = dict(inputs)
                elif hasattr(inputs, 'keys'):
                    # BatchEncoding has a keys() method
                    inputs_dict = {k: inputs[k] for k in inputs.keys()}
                else:
                    # Fallback for other input types
                    inputs_dict = {
                        'input_ids': inputs.input_ids,
                        'attention_mask': getattr(inputs, 'attention_mask', None),
                        'pixel_values': getattr(inputs, 'pixel_values', None),
                    }
                
                # Add output_hidden_states and return_dict
                inputs_dict['output_hidden_states'] = True
                inputs_dict['return_dict'] = True
                
                outputs = model(**inputs_dict)
        
        # === Extract hidden states for generated tokens ===
        # For autoregressive models, the hidden state at position i predicts token i+1
        # We need to find where the generated tokens actually start in the full sequence
        
        # Get the full input_ids to find where generated tokens start
        try:
            if isinstance(inputs, dict):
                full_input_ids = inputs['input_ids'][0].cpu().tolist()
            else:
                full_input_ids = inputs.input_ids[0].cpu().tolist()
        except (AttributeError, KeyError, TypeError) as e:
            logger.error(f"Sample {hash_id}: Failed to get input_ids: {e}")
            return None
        
        # Get the last layer hidden states
        if outputs is None:
            logger.error(f"Sample {hash_id}: Model returned None outputs")
            return None
        
        if not hasattr(outputs, 'hidden_states') or outputs.hidden_states is None:
            logger.error(f"Sample {hash_id}: No hidden_states attribute or it's None")
            return None
        
        if len(outputs.hidden_states) == 0:
            logger.error(f"Sample {hash_id}: hidden_states is empty")
            return None
        
        if outputs.hidden_states[-1] is None:
            logger.error(f"Sample {hash_id}: Last hidden state layer is None")
            return None
        
        if not hasattr(outputs, 'logits') or outputs.logits is None:
            logger.error(f"Sample {hash_id}: No logits attribute or it's None")
            return None
        
        try:
            all_hidden = outputs.hidden_states[-1][0]  # [seq_len, D]
            all_logits = outputs.logits[0]  # [seq_len, V]
        except (TypeError, IndexError) as e:
            logger.error(f"Sample {hash_id}: Failed to extract hidden states/logits: {e}")
            logger.error(f"  hidden_states type: {type(outputs.hidden_states)}, length: {len(outputs.hidden_states) if hasattr(outputs.hidden_states, '__len__') else 'N/A'}")
            logger.error(f"  logits type: {type(outputs.logits)}")
            return None
        
        seq_len = all_hidden.shape[0]
        
        # Find where generated tokens start by searching for the first generated token ID
        # Convert token_ids_np to list of ints for comparison
        gen_token_ids = [int(tid) for tid in token_ids_np]
        first_gen_token_id = gen_token_ids[0]
        
        # Start searching from prompt_length (should be close, but tokenization might differ)
        search_start = max(0, prompt_length - 20)  # Start a bit before prompt_length
        search_end = min(len(full_input_ids), prompt_length + T + 20)  # Search a bit after expected end
        
        # Find first occurrence of first generated token that matches the sequence
        gen_start_pos = None
        for pos in range(search_start, search_end):
            if full_input_ids[pos] == first_gen_token_id:
                # Verify this is the start by checking multiple consecutive tokens match
                matches = True
                check_len = min(5, T, len(full_input_ids) - pos)  # Check up to 5 tokens
                for i in range(check_len):
                    if pos + i >= len(full_input_ids) or full_input_ids[pos + i] != gen_token_ids[i]:
                        matches = False
                        break
                if matches:
                    gen_start_pos = pos
                    break
        
        if gen_start_pos is None:
            # Fallback: use prompt_length estimate, but ensure it's valid
            gen_start_pos = min(prompt_length, seq_len - 1)
            if args.debug:
                logger.debug(f"Sample {hash_id}: Could not find generated tokens, using prompt_length={gen_start_pos}")
        
        # Hidden state at position i predicts token at position i+1
        # So for generated tokens at positions [gen_start_pos, gen_start_pos+T-1],
        # we need hidden states at positions [gen_start_pos-1, gen_start_pos+T-2]
        start_idx = gen_start_pos - 1
        end_idx = gen_start_pos - 1 + T
        
        # Ensure indices are valid
        if start_idx < 0:
            start_idx = 0
            logger.warning(f"Sample {hash_id}: start_idx < 0, adjusting to 0")
        
        if end_idx > seq_len:
            # Adjust T to fit available sequence
            available = seq_len - start_idx
            if available <= 0:
                logger.warning(f"Sample {hash_id}: No valid tokens available (start_idx={start_idx}, seq_len={seq_len})")
                return None
            T = min(T, available)
            end_idx = start_idx + T
            token_ids_np = token_ids_np[:T]
            token_strs = token_strs[:T]
            if args.debug:
                logger.debug(f"Sample {hash_id}: Adjusted T to {T} to fit sequence")
        
        hidden_states = all_hidden[start_idx:end_idx]  # [T, D]
        original_logits = all_logits[start_idx:end_idx]  # [T, V]
        
        # Ensure logits and hidden_states have matching dtype (important for DeepSeek bfloat16)
        if original_logits.dtype != hidden_states.dtype:
            original_logits = original_logits.to(dtype=hidden_states.dtype)
        
        if args.debug:
            logger.debug(f"Extracted hidden_states shape: {hidden_states.shape}, dtype: {hidden_states.dtype}")
            logger.debug(f"Original logits dtype: {original_logits.dtype}")
        
        # Cleanup temp file for DeepSeek
        if model_type == 'deepseek' and hasattr(inputs, '_temp_image_created') and inputs._temp_image_created:
            try:
                if hasattr(inputs, '_temp_image_path') and inputs._temp_image_path:
                    os.unlink(inputs._temp_image_path)
            except:
                pass
        
    except Exception as e:
        logger.error(f"Forward pass failed for {hash_id}: {e}")
        import traceback
        if args.debug:
            logger.error(traceback.format_exc())
        return None
    
    token_ids = torch.tensor(token_ids_np[:T], dtype=torch.long, device=device)
    
    # Compute Jacobians analytically (FAST - no backprop!)
    # Note: compute_jacobians_batch_analytical handles dtype conversion internally
    jacobians = compute_jacobians_batch_analytical(hidden_states, original_logits, token_ids, lm_head_weight)  # [T, D]
    
    # Generate perturbation trajectories
    perturbed_hidden_states = generate_perturbation_trajectories_batch(hidden_states, jacobians, pei_radius, pei_steps)  # [T, S, D]
    
    # Get logits for all perturbed states
    perturbed_logits = get_logits_from_hidden_states_batch(model, perturbed_hidden_states, model_type)  # [T, S, V]
    
    # Extract features
    all_features = extract_ccps_features_batch(
        hidden_states=hidden_states,
        original_logits=original_logits,
        jacobians=jacobians,
        perturbed_hidden_states=perturbed_hidden_states,
        perturbed_logits=perturbed_logits,
        token_ids=token_ids,
        pei_radius=pei_radius,
        pei_steps=pei_steps,
    )
    
    # Add token metadata
    for token_idx, features in enumerate(all_features):
        features['token_idx'] = token_idx
        features['token_id'] = int(token_ids_np[token_idx])
        features['token_str'] = str(token_strs[token_idx]) if token_idx < len(token_strs) else ''
    
    return {
        'hash_id': hash_id,
        'question': question,
        'answer': answer,
        'generated_response': generated_response,
        'is_correct': bool(is_correct),
        'num_tokens': len(all_features),
        'token_features': all_features,
    }


def save_features(results: List[Dict], output_path: Path, feature_columns: List[str]):
    """Save extracted features to disk"""
    
    all_rows = []
    for sample in results:
        hash_id = sample['hash_id']
        is_correct = sample['is_correct']
        
        for token_feat in sample['token_features']:
            row = {
                'hash_id': hash_id,
                'is_correct': int(is_correct),
                'token_idx': token_feat['token_idx'],
                'token_id': token_feat['token_id'],
                'token_str': token_feat['token_str'],
            }
            
            for col in feature_columns:
                row[col] = token_feat.get(col, np.nan)
            
            all_rows.append(row)
    
    df = pd.DataFrame(all_rows)
    
    df.to_pickle(output_path.with_suffix('.pkl'))
    logger.info(f"Saved features to {output_path.with_suffix('.pkl')}")
    
    df.to_csv(output_path.with_suffix('.csv'), index=False)
    logger.info(f"Saved features to {output_path.with_suffix('.csv')}")
    
    return df


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    logger.info("=" * 80)
    logger.info("CCPS Feature Extraction for VLMs")
    logger.info("=" * 80)
    logger.info(f"Model: {args.model_name}")
    logger.info(f"Dataset: {args.dataset_name}")
    logger.info(f"PEI radius: {args.pei_radius}, steps: {args.pei_steps}")
    logger.info(f"GPU: {args.gpu_ids}")
    logger.info(f"Mode: {'MANUAL (backprop, sequential passes)' if args.manual else 'ANALYTICAL (single forward pass, batch processing)'}")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")
    
    # Load VLM model
    model, processor, model_type, dtype = load_vlm_model(
        args.model_name, args.dtype, device
    )

    logger.info(f"Model type: {model_type}")
    logger.info(f"Dtype: {dtype}")
    
    # Get lm_head weight for analytical mode
    lm_head_weight = None
    if not args.manual:
        lm_head_weight = get_lm_head_weight(model, model_type)
        logger.info(f"LM head weight shape: {lm_head_weight.shape}")
    
    # Load VLCB dataset (for images)
    vlcb_dataset, hash_id_to_idx = load_vlcb_dataset(
        args.vlcb_dataset_path, args.dataset_name
    )
    
    # Load extracted sample files
    npz_files = load_extracted_samples(args.data_dir, args.model_name, args.dataset_name)
    
    # Limit samples if specified
    if args.debug:
        npz_files = npz_files[:args.debug_samples]
        logger.info(f"Debug mode: processing {len(npz_files)} samples")
    elif args.max_samples:
        npz_files = npz_files[:args.max_samples]
        logger.info(f"Limited to {len(npz_files)} samples")
    
    # Create output directory
    model_name_part = args.model_name.split("/")[-1]
    output_dir = Path(args.output_dir) / model_name_part / args.dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check for existing features if skip_existing is set
    existing_hash_ids = set()
    if args.skip_existing:
        existing_features_path = output_dir / "features.pkl"
        if existing_features_path.exists():
            existing_df = pd.read_pickle(existing_features_path)
            existing_hash_ids = set(existing_df['hash_id'].unique())
            logger.info(f"Found {len(existing_hash_ids)} existing samples, will skip")
    
    # Process samples
    all_results = []
    failed_count = 0
    skipped_count = 0
    
    # Select processing function based on mode
    if args.manual:
        process_func = lambda npz_path: process_sample_manual(
            npz_path=npz_path,
            vlcb_dataset=vlcb_dataset,
            hash_id_to_idx=hash_id_to_idx,
            model=model,
            processor=processor,
            model_type=model_type,
            dtype=dtype,
            device=device,
            pei_radius=args.pei_radius,
            pei_steps=args.pei_steps,
        )
    else:
        process_func = lambda npz_path: process_sample_analytical(
            npz_path=npz_path,
            vlcb_dataset=vlcb_dataset,
            hash_id_to_idx=hash_id_to_idx,
            model=model,
            processor=processor,
            model_type=model_type,
            dtype=dtype,
            device=device,
            pei_radius=args.pei_radius,
            pei_steps=args.pei_steps,
            lm_head_weight=lm_head_weight,
        )
    
    for npz_path in tqdm(npz_files, desc="Extracting CCPS features"):
        hash_id = npz_path.stem
        
        if hash_id in existing_hash_ids:
            skipped_count += 1
            continue
        
        try:
            result = process_func(npz_path)
            
            if result is not None:
                all_results.append(result)
                
                if args.debug:
                    logger.debug(f"Processed {hash_id}: {result['num_tokens']} tokens")
            else:
                failed_count += 1
                
        except Exception as e:
            logger.error(f"Failed to process {npz_path.name}: {e}")
            failed_count += 1
            continue
        
        # Clear cache periodically
        if len(all_results) % 10 == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    logger.info(f"\nProcessing complete:")
    logger.info(f"  Successful: {len(all_results)}")
    logger.info(f"  Failed: {failed_count}")
    logger.info(f"  Skipped: {skipped_count}")
    
    # Save features
    if all_results:
        output_path = output_dir / "features"
        df = save_features(all_results, output_path, FEATURE_COLUMNS)

        # When --skip-existing is used, save_features overwrites features.pkl/csv
        # with only the *new* rows, which would delete the previously-cached
        # rows we just skipped. Merge them back here: existing rows whose
        # hash_id is NOT in the new batch are preserved; rows whose hash_id
        # IS in the new batch are replaced by the new ones.
        if args.skip_existing and existing_hash_ids:
            try:
                existing_df = pd.read_pickle(output_dir / "features.pkl.backup") \
                    if (output_dir / "features.pkl.backup").exists() \
                    else None
            except Exception:
                existing_df = None
            # backup was not pre-created; reconstruct by reading the pre-save
            # cache via metadata.json hash_ids trick — instead, rely on the
            # caller having taken a backup. If not, just skip the merge and
            # log a loud warning so we never silently lose data.
            if existing_df is None:
                logger.warning(
                    "--skip-existing is set but no features.pkl.backup was found; "
                    "the previously cached rows for skipped hash_ids have been OVERWRITTEN. "
                    "Always take a backup (cp features.pkl features.pkl.backup) before rerunning "
                    "with --skip-existing to surgically refresh a subset."
                )
            else:
                new_ids = set(df['hash_id'].unique())
                keep = existing_df[~existing_df['hash_id'].isin(new_ids)]
                merged = pd.concat([keep, df], ignore_index=True)
                merged.to_pickle(output_path.with_suffix('.pkl'))
                merged.to_csv(output_path.with_suffix('.csv'), index=False)
                logger.info(
                    f"Merged {len(keep)} preserved + {len(df)} fresh rows "
                    f"= {len(merged)} total ({merged['hash_id'].nunique()} unique hash_ids)"
                )
                df = merged
        
        # Save metadata
        metadata = {
            'model_name': args.model_name,
            'dataset_name': args.dataset_name,
            'pei_radius': args.pei_radius,
            'pei_steps': args.pei_steps,
            'num_samples': len(all_results),
            'num_tokens': len(df),
            'feature_columns': FEATURE_COLUMNS,
            'mode': 'manual' if args.manual else 'analytical',
            'created_at': datetime.now().isoformat(),
        }
        
        with open(output_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"\nFeatures saved to: {output_dir}")
        logger.info(f"Total tokens: {len(df)}")
        logger.info(f"Unique samples: {df['hash_id'].nunique()}")
    else:
        logger.warning("No results to save!")
    
    return output_dir


if __name__ == "__main__":
    output_path = main()
    print(f"\nFeature extraction complete. Results: {output_path}")

# Qwen/Qwen3-VL-8B-Instruct
# llava-hf/llava-v1.6-vicuna-13b-hf - 13B
# OpenGVLab/InternVL3_5-14B-HF - 14B
# google/gemma-3-27b-it - 27B
# deepseek-ai/deepseek-vl2 - 27B

# Example usage (FAST - analytical gradient, DEFAULT):
# python CCPS_feature_extraction.py \
#     --gpu-ids 6,7 \
#     --data-dir ../data/extraction/raw/ \
#     --model-name deepseek-ai/deepseek-vl2 \
#     --dataset-name validation \
#     --vlcb-dataset-path ../data/VLCB/raw \
#     --output-dir ../data/CCPS_features/ \
#     --pei-radius 20.0 \
#     --pei-steps 5 \
#     --skip-existing

# Example usage (SLOW - backprop gradient, for verification):
# python CCPS_feature_extraction.py \
#     --gpu-ids 0 \
#     --data-dir ../data/extraction/raw/ \
#     --model-name Qwen/Qwen2-VL-7B-Instruct \
#     --dataset-name train \
#     --vlcb-dataset-path ../data/VLCB/raw \
#     --output-dir ../data/CCPS_features/ \
#     --pei-radius 20.0 \
#     --pei-steps 5 \
#     --manual

# Debug mode:
# python CCPS_feature_extraction.py \
#     --gpu-ids 0 \
#     --data-dir ../data/extraction/raw/ \
#     --model-name Qwen/Qwen2-VL-7B-Instruct \
#     --dataset-name train \
#     --vlcb-dataset-path ../data/VLCB/raw \
#     --output-dir ../data/CCPS_features/ \
#     --debug \
#     --debug-samples 3