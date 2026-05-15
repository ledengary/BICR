"""
VLM Hidden State, Logit, and Attention Extraction Script
Extracts representations from vision-language models for confidence estimation.
Supports full attention vectors with metadata for flexible post-processing.
"""

# CRITICAL: Import only os and argparse first to set CUDA_VISIBLE_DEVICES before importing torch
import os
import argparse

# Parse GPU IDs argument first (before importing any CUDA libraries)
# This must happen before torch/transformers imports
_parser = argparse.ArgumentParser(add_help=False)
_parser.add_argument('--gpu_ids', type=str, default='0',
                     help='GPU IDs to use (comma-separated)')
_known_args, _ = _parser.parse_known_args()
os.environ["CUDA_VISIBLE_DEVICES"] = _known_args.gpu_ids

# Now import all other libraries (torch will respect CUDA_VISIBLE_DEVICES)
import gc
import pickle
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime
try:
    from transformers import Qwen3VLForConditionalGeneration, LlavaNextProcessor, LlavaNextForConditionalGeneration
    from qwen_vl_utils import process_vision_info
    from transformers import AutoProcessor, AutoModelForCausalLM, AutoModelForImageTextToText
except ImportError:
    print("transformers package not found.")
from transformers import AutoProcessor, AutoModelForCausalLM
# from transformers import Mistral3ForConditionalGeneration, MistralCommonBackend
from datasets import load_from_disk
import logging
import json
import hashlib
from pathlib import Path
import traceback
from openai import OpenAI
import io
import base64
from PIL import Image
import requests

# DeepSeek VL2 imports (optional - only needed if using DeepSeek models)
try:
    from deepseek_vl2.models import DeepseekVLV2Processor, DeepseekVLV2ForCausalLM
    from deepseek_vl2.utils.io import load_pil_images
    DEEPSEEK_AVAILABLE = True
except ImportError:
    DEEPSEEK_AVAILABLE = False

# ============================================================================
# Image Preprocessing
# ============================================================================

MAX_IMAGE_DIMENSION = 2048  # Default maximum allowed dimension (width or height), can be overridden by --max_image_dim

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
    
    # Print resizing notice
    print("\n<RESIZING>\n")
    
    # Calculate new dimensions preserving aspect ratio
    scale = max_dim / max_current
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    # Use LANCZOS for high-quality downsampling
    resized = pil_img.resize((new_w, new_h), Image.LANCZOS)
    
    logger.debug(f"Resized image from {w}x{h} to {new_w}x{new_h}")
    
    return resized

# Setup logging (initial setup - console only, file handlers added per model)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
    ]
)
logger = logging.getLogger(__name__)

# Store model-specific file handlers for cleanup
_model_file_handlers = {}

# ============================================================================
# Constants
# ============================================================================

SYSTEM_PROMPT = "You are a vision language assistant. Provide brief, complete answers."
APPENDED_SYSTEM_PROMPT = "Provide a brief, complete answer."

# ============================================================================
# Logging Management
# ============================================================================

def setup_model_logging(model_id):
    """
    Add a model-specific file handler to the logger.
    
    Args:
        model_id: str, model identifier (e.g., 'Qwen/Qwen2.5-VL-3B-Instruct')
    
    Returns:
        logging.FileHandler: The file handler that was added
    """
    # Extract model name for filename
    model_name = model_id.split("/")[-1].replace("/", "_")
    
    # Ensure logs directory exists
    os.makedirs("logs", exist_ok=True)
    
    # Create log filename with model name postfix
    log_filename = f"logs/generate_and_extract_{model_name}.log"
    
    # Create file handler
    file_handler = logging.FileHandler(log_filename)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(
        logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    )
    
    # Add handler to logger
    logger.addHandler(file_handler)
    
    # Store handler for later cleanup
    _model_file_handlers[model_id] = file_handler
    
    logger.info(f"Added model-specific log file: {log_filename}")
    return file_handler


def remove_model_logging(model_id):
    """
    Remove the model-specific file handler from the logger.
    
    Args:
        model_id: str, model identifier
    """
    if model_id in _model_file_handlers:
        file_handler = _model_file_handlers[model_id]
        logger.removeHandler(file_handler)
        file_handler.close()
        del _model_file_handlers[model_id]
        logger.info(f"Removed model-specific log handler for {model_id}")


# ============================================================================
# Argument Parser
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description='Extract hidden states, logits, and attention from VLMs'
    )
    
    # Model configuration
    parser.add_argument('--model_id', type=str, nargs='+',
                        default=['Qwen/Qwen2.5-VL-3B-Instruct'],
                        help='HuggingFace model identifier(s) - can specify multiple models')
    parser.add_argument('--gpu_ids', type=str, default='0',
                        help='GPU IDs to use (comma-separated)')
    parser.add_argument('--dtype', type=str, default='float32',
                        choices=['float32', 'float16', 'bfloat16'],
                        help='Model dtype')
    
    # Data configuration
    parser.add_argument('--dataset_path', type=str, required=True,
                        help='Base path to datasets directory (e.g., ../../data/GQA/)')
    parser.add_argument('--target_datasets', type=str, nargs='+', required=True,
                        help='List of target dataset names to process (e.g., train_subset_hf_5000 val_subset_hf_500)')
    parser.add_argument('--image_column', type=str, default='image',
                        help='Column name for images in dataset')
    parser.add_argument('--question_column', type=str, default='question',
                        help='Column name for questions in dataset')
    parser.add_argument('--answer_column', type=str, default='answer',
                        help='Column name for answers in dataset')
    parser.add_argument('--id_column', type=str, default='id',
                        help='Column name for sample IDs')
    
    # Extraction configuration
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Maximum number of samples to process (None for all)')
    parser.add_argument('--start_at_idx', type=int, default=None,
                        help='Start processing from this index (slices dataset from this index to the end)')
    parser.add_argument('--max_new_tokens', type=int, default=64,
                        help='Maximum tokens to generate per sample')
    parser.add_argument('--attention_aggregation', type=str, default='mean',
                        choices=['mean', 'last_layer', 'max'],
                        help='How to aggregate attention across layers/heads')
    
    # Output configuration
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save extracted representations')
    parser.add_argument('--save_logits', action='store_true',
                        help='Save all logits (large files!)')
    parser.add_argument('--save_top_logits', action='store_true',
                        help='Save only top-k logits instead of all logits')
    parser.add_argument('--top_logits', type=int, default=100,
                        help='Number of top logits to save when --save_top_logits is set')
    parser.add_argument('--compression', type=str, default='compressed',
                        choices=['compressed', 'uncompressed'],
                        help='Use compression for npz files')
    
    # Debug configuration
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug mode with detailed per-sample printing')
    parser.add_argument('--debug_samples', type=int, default=5,
                        help='Number of samples to debug (if --debug is set)')
    
    # Processing configuration
    parser.add_argument('--skip-if-processed', action='store_true',
                        help='Skip samples that have already been processed (check by hash_id)')
    
    # GPT Judge configuration
    parser.add_argument('--openai-api-key', type=str, default=None,
                        help='OpenAI API key for GPT correctness judge (if not provided, uses OPENAI_API_KEY environment variable)')
    parser.add_argument('--skip-correctness-assessment', action='store_true',
                        help='Skip GPT correctness assessment API calls (useful for separating extraction from assessment)')
    
    # Cache configuration
    parser.add_argument('--disable_cache', action='store_true',
                        help='Disable KV cache during generation (use_cache=False)')
    
    # Image preprocessing
    parser.add_argument('--max_image_dim', type=int, default=MAX_IMAGE_DIMENSION,
                        help='Maximum image dimension (images larger than this will be resized, default: 2048)')
    
    return parser.parse_args()


# ============================================================================
# Model Loading
# ============================================================================

def load_model_and_processor(model_id, dtype_str, device):
    """Load VLM model and processor/tokenizer"""
    logger.info(f"Loading model: {model_id}")
    logger.info(f"Dtype: {dtype_str}")
    logger.info(f"Device: {device}")
    
    # Map dtype string to torch dtype
    dtype_map = {
        'float32': torch.float32,
        'float16': torch.float16,
        'bfloat16': torch.bfloat16,
    }
    dtype = dtype_map[dtype_str]
    
    # Determine model type and classes based on model_id
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
        processor = DeepseekVLV2Processor.from_pretrained(model_id)
        
        # Load DeepSeek model - load first, then convert dtype (matches working diagnostics script)
        # Don't pass torch_dtype to from_pretrained as it may not convert all components (e.g., timm vision encoder)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
        )
        # Convert entire model to bfloat16, then move to CUDA
        model = model.to(dtype).cuda().eval()
        
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
        from transformers import Gemma3ForConditionalGeneration, Gemma3Processor
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
    elif 'mistral' in model_id_lower:
        model_type = 'mistral'
        # model_class = Mistral3ForConditionalGeneration
        # processor_class = MistralCommonBackend
        # processor_kwargs = {}
        logger.info("Detected Mistral model (not fully supported yet)")
        raise ValueError(f"Mistral models are not fully supported yet: {model_id}")
    else:
        raise ValueError(f"Unknown model type. Model ID must contain 'llava', 'qwen', 'deepseek', or 'gemma': {model_id}")
    
    # Set max_memory per device (as integers for max_memory parameter)
    # Only include indices for devices actually visible to torch, otherwise HF
    # tries to call cudaMemGetInfo on nonexistent ordinals.
    _full_max_memory = {
        0: "100GiB",
        1: "130GiB",
        2: "130GiB",
        3: "130GiB",
        4: "130GiB",
        5: "130GiB",
        6: "130GiB",
    }
    _visible = torch.cuda.device_count()
    max_memory = {i: _full_max_memory.get(i, "130GiB") for i in range(_visible)}
    # Load model (same parameters for both)
    model = model_class.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map="auto",  # Auto-distribute across available devices
        max_memory=max_memory,  # Memory limits per device
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

def load_dataset(dataset_path):
    """Load dataset from HuggingFace or local path"""
    logger.info(f"Loading dataset from: {dataset_path}")
    
    if os.path.isdir(dataset_path):
        # Load from disk
        dataset = load_from_disk(dataset_path)
    else:
        # Load from HuggingFace hub
        from datasets import load_dataset as hf_load_dataset
        dataset = hf_load_dataset(dataset_path)
    
    logger.info(f"Loaded {len(dataset)} samples")
    return dataset


# ============================================================================
# GPT Correctness Assessment
# ============================================================================

def assess_correctness_with_gpt(question, ground_truth_answer, generated_response, openai_api_key=None):
    """
    Queries GPT to assess whether the generated response is correct compared to ground truth.
    
    Args:
        question: The question string
        ground_truth_answer: The ground truth answer string
        generated_response: The student's generated response string
        openai_api_key: OpenAI API key (if None, uses environment variable)
    
    Returns:
        bool: True if the generated response is correct, False otherwise
    """
    if openai_api_key is None:
        openai_api_key = os.getenv("OPENAI_API_KEY")
    
    if openai_api_key is None:
        logger.warning("No OpenAI API key provided. Skipping correctness assessment.")
        return None
    
    client = OpenAI(api_key=openai_api_key)
    
    # System prompt for correctness assessment
    system_prompt = """You are an expert answer evaluator. Your task is to determine if a student's answer to a question is correct by comparing it to the ground truth answer.

1. Read the question carefully.
2. Compare the student's answer to the ground truth answer.
3. Consider semantic equivalence - answers that mean the same thing should be considered correct even if worded differently.
4. Return ONLY "yes" if the answer is correct, or "no" if it is incorrect.
5. Be lenient with minor variations in wording, capitalization, or punctuation.

Examples:
Question: What color is the sky?
Ground Truth Answer: Blue
Student Answer: The sky is blue
Output: yes

Question: What color is the sky?
Ground Truth Answer: Blue
Student Answer: Red
Output: no

Question: How many legs does a cat have?
Ground Truth Answer: Four
Student Answer: 4
Output: yes"""
    
    # Prepare messages
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Question: {question}\n\nGround Truth Answer: {ground_truth_answer}\n\nStudent Answer: {generated_response}\n\nIs the student's answer correct? (yes/no):"}
    ]
    
    try:
        # Make OpenAI API call
        response = client.responses.create(
            model="gpt-5-mini",
            reasoning={"effort": "low"},
            input=messages,
            max_output_tokens=10000,
        )
        
        # Extract output text and convert to boolean
        output_text = response.output_text.strip().lower()
        
        # Parse yes/no response
        if "yes" in output_text and "no" not in output_text:
            return True
        elif "no" in output_text and "yes" not in output_text:
            return False
        else:
            # Fallback: try to infer from response
            logger.warning(f"Unexpected GPT response format: {output_text}. Attempting to parse...")
            # Default to False if unclear
            return False
            
    except Exception as e:
        logger.error(f"Failed to assess correctness with GPT: {e}")
        return None


# ============================================================================
# Token Boundary Detection
# ============================================================================

def detect_token_boundaries(input_ids, processor, model_type='qwen'):
    """
    Detect boundaries between special/vision/text tokens.
    Supports Qwen, LLaVA, Gemma-3, InternVL, and DeepSeek models.
    """
    decoded_tokens = [processor.tokenizer.decode([tok]) for tok in input_ids]
    
    vision_start_idx = None
    vision_end_idx = None
    
    if model_type == 'qwen':
        # Qwen-specific detection
        for i, tok in enumerate(decoded_tokens):
            if '<|vision_start|>' in tok or '<|image_pad|>' in tok:
                if vision_start_idx is None:
                    vision_start_idx = i
            elif vision_start_idx is not None and '<|image_pad|>' not in tok:
                vision_end_idx = i
                break
                
    elif model_type == 'llava':
        # LLaVA-specific detection
        # LLaVA uses consecutive tokens with ID 32000 for vision
        image_token_id = processor.tokenizer.image_token_id
        
        # Find first and last occurrence of image token
        for i, tok_id in enumerate(input_ids):
            if tok_id == image_token_id:
                if vision_start_idx is None:
                    vision_start_idx = i
                vision_end_idx = i + 1  # Keep updating until last occurrence
    
    elif model_type == 'gemma':
        # Gemma-3-specific detection
        # Gemma uses explicit start/end markers + soft tokens
        # <start_of_image> (255999) -> N × <image_soft_token> (262144) -> <end_of_image> (256000)
        
        start_of_image_id = 255999
        end_of_image_id = 256000
        
        # Find <start_of_image> token
        for i, tok_id in enumerate(input_ids):
            if tok_id == start_of_image_id:
                vision_start_idx = i
                break
        
        # Find <end_of_image> token
        if vision_start_idx is not None:
            for i in range(vision_start_idx + 1, len(input_ids)):
                if input_ids[i] == end_of_image_id:
                    vision_end_idx = i + 1  # Include the end marker
                    break
    
    elif model_type == 'internvl':
        # InternVL-specific detection
        # InternVL uses <img> (151669) -> N × <IMG_CONTEXT> (151671) -> </img> (151670)
        
        img_start_id = 151669  # <img>
        img_end_id = 151670    # </img>
        img_context_id = 151671  # <IMG_CONTEXT>
        
        # Find <img> token
        for i, tok_id in enumerate(input_ids):
            if tok_id == img_start_id:
                vision_start_idx = i
                break
        
        # Find </img> token
        if vision_start_idx is not None:
            for i in range(vision_start_idx + 1, len(input_ids)):
                if input_ids[i] == img_end_id:
                    vision_end_idx = i + 1  # Include the end marker
                    break
    
    elif model_type == 'deepseek':
        # DeepSeek VL2-specific detection
        # DeepSeek uses <image> tokens for vision
        # The processor adds many consecutive <image> tokens for the image representation
        
        # Try to get image token ID from tokenizer, fallback to known ID
        try:
            image_token_id = processor.tokenizer.convert_tokens_to_ids('<image>')
        except:
            image_token_id = 128815  # Fallback: <image> token ID from tokenizer output
        
        # Find first and last occurrence of image token
        for i, tok_id in enumerate(input_ids):
            if tok_id == image_token_id:
                if vision_start_idx is None:
                    vision_start_idx = i
                vision_end_idx = i + 1  # Keep updating until last occurrence
    
    # Fallback if no vision tokens detected
    if vision_start_idx is None:
        vision_start_idx = 0
        vision_end_idx = 0
    
    boundaries = {
        'vision_start': vision_start_idx,
        'vision_end': vision_end_idx,
        'text_start': vision_end_idx,  # Text starts AFTER vision ends
        'text_end': len(input_ids),
        'num_vision': vision_end_idx - vision_start_idx,
        'num_text': len(input_ids) - vision_end_idx,
        'num_other': vision_start_idx,
    }
    
    return boundaries


# ============================================================================
# Attention Extraction
# ============================================================================

def extract_attention_with_stats(attentions_tuple, boundaries, aggregation='mean'):
    """
    Extract full attention vector and compute statistics.
    
    Args:
        attentions_tuple: Tuple of attention tensors from model
        boundaries: Dict with token boundary information
        aggregation: 'mean', 'last_layer', or 'max'
    
    Returns:
        dict with:
            - 'attention_full': [input_len + generated_len] full attention vector
            - 'stats': dict with statistics including attention to generated tokens
    """
    valid_attns = [a for a in attentions_tuple if a is not None]
    
    if len(valid_attns) == 0:
        # Fallback when no attention is available
        input_len = boundaries['text_end']
        return {
            'attention_full': torch.zeros(input_len),
            'stats': {
                'total_to_vision': 0.0,
                'total_to_text': 0.0,
                'total_to_other': 0.0,
                'total_to_generated': 0.0,
                'pct_to_vision': 0.0,
                'pct_to_text': 0.0,
                'pct_to_other': 0.0,
                'pct_to_generated': 0.0,
                'entropy': 0.0,
                'max_position': 0,
                'max_value': 0.0,
            }
        }
    
    # Aggregate attention across layers and heads
    if aggregation == 'last_layer':
        attn_full = valid_attns[-1][0, :, -1, :].mean(dim=0)  # [seq_len]
    elif aggregation == 'mean':
        all_layer_attns = []
        for layer_attn in valid_attns:
            attn = layer_attn[0, :, -1, :]  # [num_heads, seq_len]
            all_layer_attns.append(attn.mean(dim=0))  # [seq_len]
        attn_full = torch.stack(all_layer_attns).mean(dim=0)  # [seq_len]
    elif aggregation == 'max':
        all_layer_attns = []
        for layer_attn in valid_attns:
            attn = layer_attn[0, :, -1, :]
            all_layer_attns.append(attn.max(dim=0)[0])
        attn_full = torch.stack(all_layer_attns).max(dim=0)[0]
    
    # Normalize attention vector (should sum to 1.0)
    attn_sum = attn_full.sum()
    if attn_sum > 0:
        attn_full = attn_full / attn_sum
    
    # Compute statistics - USE ACTUAL LENGTH OF ATTENTION VECTOR
    actual_len = len(attn_full)
    vision_start = boundaries['vision_start']
    vision_end = min(boundaries['vision_end'], actual_len)
    text_start = vision_end
    text_end = min(boundaries['text_end'], actual_len)
    
    # Split attention into regions
    # Region 1: OTHER (special tokens before vision)
    other_attn = attn_full[:vision_start]
    
    # Region 2: VISION (image tokens)
    vision_attn = attn_full[vision_start:vision_end]
    
    # Region 3: TEXT (input text prompt)
    text_attn = attn_full[text_start:text_end]
    
    # Region 4: GENERATED (attention to previously generated tokens)
    # This is the KEY FIX - attention grows as generation progresses!
    generated_attn = attn_full[text_end:]
    
    total = attn_full.sum()  # Should be 1.0 after normalization
    
    stats = {
        # Absolute attention amounts
        'total_to_vision': vision_attn.sum().item(),
        'total_to_text': text_attn.sum().item(),
        'total_to_other': other_attn.sum().item(),
        'total_to_generated': generated_attn.sum().item(),
        
        # Percentages (should sum to 100%)
        'pct_to_vision': (vision_attn.sum() / total).item() if total > 0 else 0.0,
        'pct_to_text': (text_attn.sum() / total).item() if total > 0 else 0.0,
        'pct_to_other': (other_attn.sum() / total).item() if total > 0 else 0.0,
        'pct_to_generated': (generated_attn.sum() / total).item() if total > 0 else 0.0,
        
        # Additional statistics
        'entropy': -(attn_full * torch.log(attn_full + 1e-9)).sum().item(),
        'max_position': attn_full.argmax().item(),
        'max_value': attn_full.max().item(),
        
        # Metadata about regions
        'actual_len': actual_len,
        'vision_range': [vision_start, vision_end],
        'text_range': [text_start, text_end],
        'generated_range': [text_end, actual_len],
    }
    
    return {
        'attention_full': attn_full,
        'stats': stats,
    }


# ============================================================================
# CUDA Memory Management
# ============================================================================

def clear_cuda_cache(aggressive=False):
    """
    Clear CUDA cache and perform garbage collection.
    
    Args:
        aggressive: If True, performs more aggressive cleanup including synchronization
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        if aggressive:
            torch.cuda.synchronize()
            try:
                torch.cuda.reset_peak_memory_stats()
            except:
                pass
    torch.cuda.empty_cache(); torch.cuda.empty_cache(); torch.cuda.empty_cache()  # Triple cleanup


# ============================================================================
# Main Extraction Function
# ============================================================================

def extract_sample_representations(sample, model, processor, model_type, args):
    """
    Extract representations for a single sample.
    
    Returns:
        dict with:
            - 'hidden_states': [num_tokens, hidden_size]
            - 'logits': [num_tokens, vocab_size] (if args.save_logits)
            - 'attention_full': [num_tokens, input_len]
            - 'token_ids': [num_tokens]
            - 'token_strs': [num_tokens]
            - 'attention_stats': list of dicts (per token)
            - 'boundaries': dict with token boundaries
            - 'metadata': dict with sample info
        None if required fields are missing
    """
    # Validate required fields - MUST be present in record
    if args.image_column not in sample:
        logger.warning(f"Sample missing required field '{args.image_column}'. Skipping.")
        return None
    if args.question_column not in sample:
        logger.warning(f"Sample missing required field '{args.question_column}'. Skipping.")
        return None
    if args.answer_column not in sample:
        logger.warning(f"Sample missing required field '{args.answer_column}'. Skipping.")
        return None
    
    # Prepare input - direct access since we validated above
    image_raw = sample[args.image_column]
    question = sample[args.question_column]
    answer = sample[args.answer_column]
    sample_id = sample.get(args.id_column, 'unknown')
    
    # Resize image if needed (prevents OOM on large images)
    image = resize_image_if_needed(image_raw, max_dim=args.max_image_dim)
    
    # Process inputs based on model type
    if model_type == 'qwen':
        # Qwen processing
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
        
        # Process inputs
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
        ).to(model.device)
        
        # Detect token boundaries
        input_ids = inputs.input_ids[0].cpu().tolist()
        boundaries = detect_token_boundaries(input_ids, processor, model_type)
        
        # Generate with full extraction
        model.eval()
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                output_hidden_states=True,
                output_scores=True,
                output_attentions=True,
                return_dict_in_generate=True,
                do_sample=False,
                # use_cache=not args.disable_cache,
            )
        
        # Extract generated tokens
        generated_sequence = output.sequences[0]
        input_len = inputs.input_ids.shape[1]
        generated_token_ids = generated_sequence[input_len:].cpu().tolist()
        
    elif model_type == 'llava':
        # LLaVA processing
        # Prepare conversation format for LLaVA
        conversation = [
            # {
            #     "role": "system",
            #     "content": SYSTEM_PROMPT
            # },
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": question + "\n\n" + APPENDED_SYSTEM_PROMPT},
                ],
            },
        ]
        
        # Apply chat template and process inputs
        prompt = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        
        # Process image and text
        # Handle PIL Image or path/URL
        if isinstance(image, Image.Image):
            image_to_process = image
        elif isinstance(image, str):
            # If it's a path or URL, load it
            if image.startswith('http://') or image.startswith('https://'):
                image_to_process = Image.open(requests.get(image, stream=True).raw)
            else:
                image_to_process = Image.open(image)
        else:
            image_to_process = image
        
        inputs = processor(images=image_to_process, text=prompt, return_tensors="pt")
        
        # Move to device
        device = next(model.parameters()).device
        model_dtype = next(model.parameters()).dtype
        
        # Convert inputs to device and correct dtype
        inputs = {k: v.to(device=device) if isinstance(v, torch.Tensor) else v 
                for k, v in inputs.items()}
        
        # Convert pixel_values to model dtype if present
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(dtype=model_dtype)
        
        # Detect token boundaries using the correct function
        input_ids = inputs["input_ids"][0].cpu().tolist()
        boundaries = detect_token_boundaries(input_ids, processor, model_type)  # ← USE THE FUNCTION!
        
        # Generate with full extraction
        model.eval()
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                output_hidden_states=True,
                output_scores=True,
                output_attentions=True,
                return_dict_in_generate=True,
                do_sample=False,
                # use_cache=not args.disable_cache,
            )
        
        # Extract generated tokens
        generated_sequence = output.sequences[0]
        input_len = inputs["input_ids"].shape[1]
        generated_token_ids = generated_sequence[input_len:].cpu().tolist()
        
    elif model_type == 'gemma':
        # Gemma-3 processing
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
        
        # Apply chat template and process inputs
        prompt = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        
        # Process image and text
        if isinstance(image, Image.Image):
            image_to_process = image
        elif isinstance(image, str):
            if image.startswith('http://') or image.startswith('https://'):
                image_to_process = Image.open(requests.get(image, stream=True).raw)
            else:
                image_to_process = Image.open(image)
        else:
            image_to_process = image
        
        inputs = processor(images=image_to_process, text=prompt, return_tensors="pt")
        
        # Move to device
        device = next(model.parameters()).device
        model_dtype = next(model.parameters()).dtype
        
        # Convert inputs to device and correct dtype
        inputs = {k: v.to(device=device) if isinstance(v, torch.Tensor) else v 
                for k, v in inputs.items()}
        
        # Convert pixel_values to model dtype if present
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(dtype=model_dtype)
        
        # Detect token boundaries
        input_ids = inputs["input_ids"][0].cpu().tolist()
        boundaries = detect_token_boundaries(input_ids, processor, model_type)
        
        # Generate with full extraction
        model.eval()
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                output_hidden_states=True,
                output_scores=True,
                output_attentions=True,
                return_dict_in_generate=True,
                do_sample=False,
                # use_cache=not args.disable_cache,
            )
        
        # Extract generated tokens
        generated_sequence = output.sequences[0]
        input_len = inputs["input_ids"].shape[1]
        generated_token_ids = generated_sequence[input_len:].cpu().tolist()

    elif model_type == 'internvl':
        # InternVL processing
        # Uses similar structure to Gemma but with different chat format
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
        
        # Apply chat template
        prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        
        # Process image and text
        if isinstance(image, Image.Image):
            image_to_process = image
        elif isinstance(image, str):
            if image.startswith('http://') or image.startswith('https://'):
                image_to_process = Image.open(requests.get(image, stream=True).raw)
            else:
                image_to_process = Image.open(image)
        else:
            image_to_process = image
        
        inputs = processor(images=image_to_process, text=prompt, return_tensors="pt")
        
        # Move to device and match model dtype
        device = next(model.parameters()).device
        model_dtype = next(model.parameters()).dtype
        
        inputs = {k: v.to(device=device) if isinstance(v, torch.Tensor) else v 
                for k, v in inputs.items()}
        
        # Convert pixel_values to model dtype if present
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(dtype=model_dtype)
        
        # Detect token boundaries
        input_ids = inputs["input_ids"][0].cpu().tolist()
        boundaries = detect_token_boundaries(input_ids, processor, model_type)
        
        # Generate with full extraction
        model.eval()
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                output_hidden_states=True,
                output_scores=True,
                output_attentions=True,
                return_dict_in_generate=True,
                do_sample=False,
                # use_cache=not args.disable_cache,
            )
        
        # Extract generated tokens
        generated_sequence = output.sequences[0]
        input_len = inputs["input_ids"].shape[1]
        generated_token_ids = generated_sequence[input_len:].cpu().tolist()

    elif model_type == 'deepseek':
        # DeepSeek VL2 processing
        # Uses specific conversation format with <|User|> and <|Assistant|> roles
        
        # Convert image to path if it's a PIL Image (DeepSeek expects paths in conversation)
        # We'll save to a temporary location or handle PIL directly
        if isinstance(image, Image.Image):
            # Save PIL image to a temporary file for DeepSeek
            import tempfile
            temp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
            image.save(temp_file.name)
            image_path = temp_file.name
            temp_image_created = True
        elif isinstance(image, str):
            image_path = image
            temp_image_created = False
        else:
            # Try to handle other types by converting to PIL first
            image_path = image
            temp_image_created = False
        
        # Create DeepSeek-style conversation
        conversation = [
            {
                "role": "<|User|>",
                "content": f"<image>\n{question}",
                "images": [image_path],
            },
            {"role": "<|Assistant|>", "content": ""},
        ]
        
        # Load images and prepare inputs
        pil_images = load_pil_images(conversation)
        prepare_inputs = processor(
            conversations=conversation,
            images=pil_images,
            force_batchify=True,
            system_prompt=SYSTEM_PROMPT
        ).to(model.device)
        
        # Convert pixel values to match model dtype (fixes bfloat16/float32 mismatch)
        model_dtype = next(model.parameters()).dtype
        if hasattr(prepare_inputs, 'pixel_values') and prepare_inputs.pixel_values is not None:
            prepare_inputs.pixel_values = prepare_inputs.pixel_values.to(dtype=model_dtype)
        
        # Detect token boundaries
        input_ids = prepare_inputs.input_ids[0].cpu().tolist()
        boundaries = detect_token_boundaries(input_ids, processor, model_type)
        
        # Get input embeddings
        inputs_embeds = model.prepare_inputs_embeds(**prepare_inputs)
        
        input_len = prepare_inputs.input_ids.shape[1]
        
        # Generate with DeepSeek's language model
        # Note: DeepSeek uses model.language.generate() instead of model.generate()
        model.eval()
        with torch.no_grad():
            output = model.language.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=prepare_inputs.attention_mask,
                pad_token_id=processor.tokenizer.eos_token_id,
                bos_token_id=processor.tokenizer.bos_token_id,
                eos_token_id=processor.tokenizer.eos_token_id,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                # use_cache=not args.disable_cache,
                output_hidden_states=True,
                output_scores=True,
                output_attentions=True,
                return_dict_in_generate=True,
            )
        
        # Clean up temporary image file if created
        if temp_image_created:
            try:
                os.unlink(image_path)
            except:
                pass
        
        # Extract generated tokens - DeepSeek returns only generated tokens (not including input)
        generated_sequence = output.sequences[0]
        generated_token_ids = generated_sequence.cpu().tolist()
        
    elif model_type == 'mistral':
        # Mistral processing
        # Convert PIL Image to a format Mistral understands
        # Mistral expects "type": "image_url" with a URL or base64 data, not "type": "image" with a PIL Image object
        
        # Convert PIL Image to base64 data URL
        if isinstance(image, Image.Image):
            buffered = io.BytesIO()
            image.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            image_data_url = f"data:image/png;base64,{img_str}"
        else:
            # If it's already a path/URL, use it directly
            image_data_url = image
        
        messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                    {"type": "image_url", "image_url": {"url": image_data_url}},
                ],
            }
        ]
        
        # Process inputs using Mistral tokenizer
        tokenized = processor.apply_chat_template(
            messages, 
            return_tensors="pt", 
            return_dict=True
        )
        
        # Move to device and handle pixel values
        device = next(model.parameters()).device
        # Get the actual dtype of the model (may differ from requested dtype)
        model_dtype = next(model.parameters()).dtype
        
        # Convert all inputs to match the model's dtype
        tokenized["input_ids"] = tokenized["input_ids"].to(device=device)
        
        # Convert attention_mask if present
        if "attention_mask" in tokenized:
            tokenized["attention_mask"] = tokenized["attention_mask"].to(device=device)
        
        if "pixel_values" in tokenized:
            # Use the model's actual dtype, not the requested dtype
            tokenized["pixel_values"] = tokenized["pixel_values"].to(
                dtype=model_dtype,
                device=device
            )
            image_sizes = [tokenized["pixel_values"].shape[-2:]]
        else:
            image_sizes = None
        
        inputs = tokenized
        
        # Detect token boundaries (simplified for Mistral - may need adjustment)
        input_ids = inputs["input_ids"][0].cpu().tolist()
        # For Mistral, we'll use a simplified boundary detection
        boundaries = {
            'vision_start': 0,
            'vision_end': len(input_ids),  # Simplified - Mistral may handle this differently
            'text_start': 0,
            'text_end': len(input_ids),
            'num_vision': 0,
            'num_text': len(input_ids),
            'num_other': 0,
        }
        
        # Generate with full extraction
        model.eval()
        with torch.no_grad():
            generate_kwargs = {
                **inputs,
                "max_new_tokens": args.max_new_tokens,
                "output_hidden_states": True,
                "output_scores": True,
                "output_attentions": True,
                "return_dict_in_generate": True,
                "do_sample": False,
                "use_cache": not args.disable_cache,
            }
            if image_sizes is not None:
                generate_kwargs["image_sizes"] = image_sizes
                
            output = model.generate(**generate_kwargs)
        
        # Extract generated tokens
        generated_sequence = output.sequences[0]
        input_len = inputs["input_ids"].shape[1]
        generated_token_ids = generated_sequence[input_len:].cpu().tolist()
        
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    
    if len(generated_token_ids) == 0:
        logger.warning(f"No tokens generated for sample {sample_id}")
        return None
    
    # Decode the full generated response
    # DeepSeek uses processor.tokenizer, others use processor directly
    if model_type == 'deepseek':
        generated_response = processor.tokenizer.decode(generated_token_ids, skip_special_tokens=True)
    else:
        generated_response = processor.decode(generated_token_ids, skip_special_tokens=True)
    
    # Assess correctness with GPT judge
    # answer is guaranteed to exist due to validation above
    is_correct = None
    if not args.skip_correctness_assessment and (args.openai_api_key is not None or os.getenv("OPENAI_API_KEY") is not None):
        is_correct = assess_correctness_with_gpt(
            question, 
            answer, 
            generated_response,
            args.openai_api_key
        )
    
    # Extract per-token representations
    hidden_states_list = []
    logits_list = [] if args.save_logits else None
    top_logits_list = [] if args.save_top_logits else None
    top_logit_indices_list = [] if args.save_top_logits else None
    attention_full_list = []
    token_strs_list = []
    attention_stats_list = []
    
    for i in range(len(generated_token_ids)):
        token_id = generated_token_ids[i]
        # Use correct decoder based on model type
        if model_type == 'deepseek':
            token_str = processor.tokenizer.decode(token_id, skip_special_tokens=True)
        else:
            token_str = processor.decode(token_id, skip_special_tokens=True)
        
        # Extract hidden state
        last_layer = output.hidden_states[i][-1].squeeze(0)  # Remove batch
        if i == 0:
            hidden_state = last_layer[-1, :]  # Last position of input
        else:
            hidden_state = last_layer.squeeze(0)  # Single vector
        
        # Extract logits
        if args.save_logits:
            logits = output.scores[i].squeeze()
            # Convert bfloat16 to float32 before numpy conversion (numpy doesn't support bfloat16)
            logits_cpu = logits.cpu()
            if logits_cpu.dtype == torch.bfloat16:
                logits_cpu = logits_cpu.float()
            logits_list.append(logits_cpu.numpy())
        
        # Extract top-k logits
        if args.save_top_logits:
            logits = output.scores[i].squeeze()  # [vocab_size]
            logits_cpu = logits.cpu()
            # Convert bfloat16 to float32 before numpy conversion
            if logits_cpu.dtype == torch.bfloat16:
                logits_cpu = logits_cpu.float()
            
            # Get top-k values and indices
            top_k = min(args.top_logits, len(logits_cpu))
            top_values, top_indices = torch.topk(logits_cpu, k=top_k, dim=-1)
            
            # Sort by value (descending) to ensure consistent ordering
            sorted_indices = torch.argsort(top_values, descending=True)
            top_logits_list.append(top_values[sorted_indices].numpy())
            top_logit_indices_list.append(top_indices[sorted_indices].numpy())
        
        # Extract attention
        attn_result = extract_attention_with_stats(
            output.attentions[i],
            boundaries,
            aggregation=args.attention_aggregation
        )
        
        # Convert bfloat16 to float32 before numpy conversion
        hidden_state_cpu = hidden_state.cpu()
        if hidden_state_cpu.dtype == torch.bfloat16:
            hidden_state_cpu = hidden_state_cpu.float()
        hidden_states_list.append(hidden_state_cpu.numpy())
        
        attn_full_cpu = attn_result['attention_full'].cpu()
        if attn_full_cpu.dtype == torch.bfloat16:
            attn_full_cpu = attn_full_cpu.float()
        attention_full_list.append(attn_full_cpu.numpy())
        token_strs_list.append(token_str)
        attention_stats_list.append(attn_result['stats'])
    
    # Package results
    result = {
        'hidden_states': np.array(hidden_states_list),  # [num_tokens, D]
        'attention_full': attention_full_list,  # List of arrays (variable length!)
        'token_ids': generated_token_ids,
        'token_strs': token_strs_list,
        'attention_stats': attention_stats_list,
        'boundaries': boundaries,
        'metadata': {
            'sample_id': sample_id,
            'question': question,
            'answer': answer,
            'generated_response': generated_response,  # Full decoded response string
            'is_correct': is_correct,  # Boolean correctness assessment from GPT judge
            'num_generated_tokens': len(generated_token_ids),
            'input_length': input_len,
        }
    }
    
    if args.save_logits:
        result['logits'] = np.array(logits_list)  # [num_tokens, vocab_size]
    
    if args.save_top_logits:
        result['top_logits'] = np.array(top_logits_list)  # [num_tokens, top_k]
        result['top_logit_indices'] = np.array(top_logit_indices_list)  # [num_tokens, top_k]
        result['metadata']['top_k'] = args.top_logits
    
    # Clear intermediate tensors to free memory
    del output
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return result


# ============================================================================
# Debug Printing
# ============================================================================

def print_debug_info(result, sample_idx):
    """Print detailed debug information for a sample"""
    print("\n" + "="*80)
    print(f"SAMPLE {sample_idx}: {result['metadata']['sample_id']}")
    print("="*80)
    
    print(f"\nQuestion: {result['metadata']['question']}")
    print(f"Answer (ground truth): {result['metadata']['answer']}")  # Required field, guaranteed to exist
    print(f"Generated response: {result['metadata'].get('generated_response', 'N/A')}")
    is_correct = result['metadata'].get('is_correct')
    if is_correct is not None:
        print(f"Correctness (GPT judge): {'✓ CORRECT' if is_correct else '✗ INCORRECT'}")
    else:
        print(f"Correctness (GPT judge): Not assessed (no API key)")
    print(f"Input length: {result['metadata']['input_length']}")
    print(f"Generated tokens: {result['metadata']['num_generated_tokens']}")
    
    boundaries = result['boundaries']
    print(f"\nToken Boundaries:")
    print(f"  OTHER (special): [0:{boundaries['vision_start']}] = {boundaries['num_other']} tokens")
    print(f"  VISION (image):  [{boundaries['vision_start']}:{boundaries['vision_end']}] = {boundaries['num_vision']} tokens")
    print(f"  TEXT (prompt):   [{boundaries['text_start']}:{boundaries['text_end']}] = {boundaries['num_text']} tokens")
    
    print(f"\n{'Idx':<4} | {'Token':<15} | {'Hidden':<10} | {'Attn':<10} | {'%Vis':<6} | {'%Txt':<6} | {'%Oth':<6} | {'%Gen':<6} | {'Entropy':<8}")
    print("-" * 105)
    
    for i in range(len(result['token_ids'])):
        token_str = result['token_strs'][i][:15].ljust(15)
        h_shape = f"({result['hidden_states'].shape[1]},)"
        a_shape = f"({len(result['attention_full'][i])},)"
        
        stats = result['attention_stats'][i]
        pct_vis = f"{stats['pct_to_vision']*100:.1f}%"
        pct_txt = f"{stats['pct_to_text']*100:.1f}%"
        pct_oth = f"{stats['pct_to_other']*100:.1f}%"
        pct_gen = f"{stats['pct_to_generated']*100:.1f}%"
        entropy = f"{stats['entropy']:.3f}"
        
        print(f"{i:<4} | {token_str} | {h_shape:<10} | {a_shape:<10} | {pct_vis:<6} | {pct_txt:<6} | {pct_oth:<6} | {pct_gen:<6} | {entropy:<8}")
    
    print("\nAttention Statistics Summary:")
    avg_vis = np.mean([s['pct_to_vision'] for s in result['attention_stats']]) * 100
    avg_txt = np.mean([s['pct_to_text'] for s in result['attention_stats']]) * 100
    avg_oth = np.mean([s['pct_to_other'] for s in result['attention_stats']]) * 100
    avg_gen = np.mean([s['pct_to_generated'] for s in result['attention_stats']]) * 100
    
    print(f"  Average attention to VISION:    {avg_vis:.1f}%")
    print(f"  Average attention to TEXT:      {avg_txt:.1f}%")
    print(f"  Average attention to OTHER:     {avg_oth:.1f}%")
    print(f"  Average attention to GENERATED: {avg_gen:.1f}%")
    print(f"  TOTAL:                          {avg_vis + avg_txt + avg_oth + avg_gen:.1f}%")



# ============================================================================
# Per-Sample Saving (SAFER FOR LONG RUNS)
# ============================================================================

def save_sample(sample_result, hash_id, output_dir, args):
    """
    Save a single sample's data to its own .npz file.
    
    Args:
        sample_result: dict with all sample data
        hash_id: str, hash ID for this sample (used as filename)
        output_dir: str, directory to save to
        args: argparse args object
    
    Returns:
        str: path to saved file
    """
    samples_dir = os.path.join(output_dir, 'samples')
    os.makedirs(samples_dir, exist_ok=True)
    
    # Create filename using hash_id
    filename = f"{hash_id}.npz"
    filepath = os.path.join(samples_dir, filename)
    
    # Prepare data for saving
    save_dict = {
        'hash_id': hash_id,  # Store hash_id inside the file
        'sample_id': str(sample_result['metadata']['sample_id']),
        'question': str(sample_result['metadata']['question']),
        'answer': str(sample_result['metadata']['answer']),  # Required field, guaranteed to exist
        'generated_response': str(sample_result['metadata'].get('generated_response', '')),  # Generated response string
        'is_correct': sample_result['metadata'].get('is_correct'),  # Boolean correctness (can be None if not assessed)
        'num_generated_tokens': sample_result['metadata']['num_generated_tokens'],
        'input_length': sample_result['metadata']['input_length'],
        
        # Arrays
        'hidden_states': sample_result['hidden_states'].astype(np.float16),
        'attention_full': np.array(sample_result['attention_full'], dtype=object),
        
        # Token info
        'token_ids': np.array(sample_result['token_ids'], dtype=np.int32),
        'token_strs': np.array(sample_result['token_strs'], dtype=object),
        
        # Metadata as JSON strings (for variable-length dicts)
        'boundaries': json.dumps(sample_result['boundaries']),
        'attention_stats': json.dumps(sample_result['attention_stats']),
    }
    
    # Add logits if requested
    if args.save_logits:
        save_dict['logits'] = sample_result['logits'].astype(np.float16)
    
    # Add top logits if requested
    if args.save_top_logits:
        save_dict['top_logits'] = sample_result['top_logits'].astype(np.float16)
        save_dict['top_logit_indices'] = sample_result['top_logit_indices'].astype(np.int32)
        save_dict['top_k'] = args.top_logits
    
    # Save with compression
    save_fn = np.savez_compressed if args.compression == 'compressed' else np.savez
    save_fn(filepath, **save_dict)
    
    return filepath


def create_manifest(output_dir, num_samples, config):
    """
    Create a manifest file listing all processed samples.
    
    Args:
        output_dir: str, directory where samples are saved
        num_samples: int, number of samples processed
        config: dict, configuration used for extraction
    """
    manifest_path = os.path.join(output_dir, 'manifest.json')
    
    manifest_data = {
        'num_samples': num_samples,
        'samples_dir': 'samples/',
        'sample_filename_pattern': '{hash_id}.npz',
        'config': config,
        'created_at': datetime.now().isoformat(),
    }
    
    with open(manifest_path, 'w') as f:
        json.dump(manifest_data, f, indent=2)
    
    logger.info(f"Created manifest: {manifest_path}")
    return manifest_path


def get_completed_samples(output_dir):
    """
    Get set of already-completed sample hash_ids.
    Useful for resuming interrupted runs.
    
    Args:
        output_dir: str, directory to check for samples
    
    Returns:
        set: set of completed hash_ids (filenames without .npz extension)
    """
    samples_dir = os.path.join(output_dir, 'samples')
    if not os.path.exists(samples_dir):
        return set()
    
    completed = set()
    for filename in os.listdir(samples_dir):
        if filename.endswith('.npz'):
            # Extract hash_id from filename: {hash_id}.npz -> {hash_id}
            hash_id = filename.replace('.npz', '')
            completed.add(hash_id)
    
    return completed


# ============================================================================
# Main Function
# ============================================================================

def main():
    args = parse_args()
    
    # Validate arguments
    if args.save_logits and args.save_top_logits:
        logger.warning("Both --save_logits and --save_top_logits are set. "
                      "Will save both full logits and top-k logits.")
    
    # Note: CUDA_VISIBLE_DEVICES was already set at import time (before torch import)
    # This ensures PyTorch only sees the specified GPUs
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Ensure model_id is a list
    if isinstance(args.model_id, str):
        model_ids = [args.model_id]
    else:
        model_ids = args.model_id
    
    logger.info("="*80)
    logger.info("VLM Representation Extraction (PER-SAMPLE SAVING)")
    logger.info("="*80)
    logger.info(f"Models: {model_ids}")
    logger.info(f"Base dataset path: {args.dataset_path}")
    logger.info(f"Cuda Visible Devices: {args.gpu_ids}")
    logger.info(f"Target datasets: {args.target_datasets}")
    logger.info(f"Output base: {args.output_dir}")
    logger.info(f"Debug mode: {args.debug}")
    if args.save_logits:
        logger.info("Saving full logits")
    if args.save_top_logits:
        logger.info(f"Saving top-{args.top_logits} logits")
    if args.skip_correctness_assessment:
        logger.info("GPT correctness assessment skipped (--skip-correctness-assessment flag set)")
    elif args.openai_api_key or os.getenv("OPENAI_API_KEY"):
        logger.info("GPT correctness assessment enabled")
    else:
        logger.info("GPT correctness assessment disabled (no API key provided)")
    if args.disable_cache:
        logger.info("KV cache disabled (use_cache=False)")
    if args.start_at_idx is not None:
        logger.info(f"Starting from index: {args.start_at_idx}")
    
    # Process each model
    for model_id in model_ids:
        # Setup model-specific logging
        setup_model_logging(model_id)
        
        logger.info(f"\n{'#'*80}")
        logger.info(f"Processing model: {model_id}")
        logger.info(f"{'#'*80}")
        
        # Extract model name for directory structure
        model_dir_name = model_id.split("/")[-1]
        
        # Load model and processor for this model
        try:
            model, processor, model_type = load_model_and_processor(model_id, args.dtype, device)
        except Exception as e:
            logger.error(f"Failed to load model {model_id}: {e}")
            logger.error(traceback.format_exc())
            continue
        
        # Process each target dataset with this model
        for target_dataset in args.target_datasets:
            logger.info(f"\n{'='*80}")
            logger.info(f"Processing dataset: {target_dataset}")
            logger.info(f"{'='*80}")
            
            # Construct full dataset path
            full_dataset_path = os.path.join(args.dataset_path, target_dataset)
            
            # Construct output directory: output_dir/model_dir_name/target_dataset/
            output_dir_for_dataset = os.path.join(args.output_dir, model_dir_name, target_dataset)
            os.makedirs(output_dir_for_dataset, exist_ok=True)
            
            logger.info(f"Dataset path: {full_dataset_path}")
            logger.info(f"Output directory: {output_dir_for_dataset}")
            
            # Check for already-completed samples (only if --skip-if-processed is set)
            completed_hash_ids = set()
            if args.skip_if_processed:
                completed_hash_ids = get_completed_samples(output_dir_for_dataset)
                if completed_hash_ids:
                    logger.info(f"Found {len(completed_hash_ids)} already-completed samples (will skip)")
            
            # Load dataset
            try:
                dataset = load_dataset(full_dataset_path)
            except Exception as e:
                logger.error(f"Failed to load dataset from {full_dataset_path}: {e}")
                continue
            
            original_len = len(dataset)
            
            # Start from specific index if specified
            if args.start_at_idx is not None:
                if args.start_at_idx < 0:
                    logger.warning(f"start_at_idx ({args.start_at_idx}) is negative, using 0 instead")
                    args.start_at_idx = 0
                if args.start_at_idx >= len(dataset):
                    logger.warning(f"start_at_idx ({args.start_at_idx}) >= dataset length ({len(dataset)}), skipping dataset")
                    continue
                dataset = dataset.select(range(args.start_at_idx, len(dataset)))
                logger.info(f"Starting from index {args.start_at_idx}: {original_len} -> {len(dataset)} samples")
            
            # Limit samples if specified (applies to each dataset individually)
            if args.max_samples is not None:
                dataset = dataset.select(range(min(args.max_samples, len(dataset))))
                logger.info(f"Limited to {len(dataset)} samples")
            
            # Save config
            config_path = os.path.join(output_dir_for_dataset, 'config.json')
            with open(config_path, 'w') as f:
                json.dump(vars(args), f, indent=2)
            
            # Process samples
            num_saved = 0
            num_skipped = 0
            num_failed = 0
            
            if args.debug:
                # Debug mode: detailed printing for first N samples
                logger.info(f"\n{'='*80}")
                logger.info(f"DEBUG MODE: Processing first {args.debug_samples} samples")
                logger.info(f"{'='*80}\n")
                
                iterator = range(min(args.debug_samples, len(dataset)))
            else:
                # Normal mode: tqdm progress bar
                logger.info("\nProcessing samples...")
                iterator = tqdm(range(len(dataset)), desc=f"Extracting {target_dataset}")
            
            for sample_idx in iterator:
                sample = dataset[sample_idx]
                
                # Validate required fields before processing
                if args.question_column not in sample:
                    logger.warning(f"Sample {sample_idx} missing required field '{args.question_column}'. Skipping.")
                    num_skipped += 1
                    continue
                if args.answer_column not in sample:
                    logger.warning(f"Sample {sample_idx} missing required field '{args.answer_column}'. Skipping.")
                    num_skipped += 1
                    continue
                if args.image_column not in sample:
                    logger.warning(f"Sample {sample_idx} missing required field '{args.image_column}'. Skipping.")
                    num_skipped += 1
                    continue
                
                # Get required fields - direct access since validated above
                question = sample[args.question_column]
                answer = sample[args.answer_column]
                
                # Generate hash_id for this sample
                hash_id = sample["hash_id"]
                
                # Get dataset field from sample (not the target_dataset being processed)
                sample_dataset = sample.get("dataset", "unknown")
                
                # Log start of processing for this record
                logger.info(f"Processing record: hash_id={hash_id}, dataset={sample_dataset}, sample_idx={sample_idx}")
                
                # Skip if already completed (only if --skip-if-processed is set)
                if args.skip_if_processed and hash_id in completed_hash_ids:
                    logger.info(f"Skipping already processed record: hash_id={hash_id}")
                    num_skipped += 1
                    continue
                
                try:
                    # Extract representations
                    result = extract_sample_representations(sample, model, processor, model_type, args)
                    
                    if result is not None:
                        # Save immediately with hash_id as filename
                        save_sample(result, hash_id, output_dir_for_dataset, args)
                        num_saved += 1
                        
                        if args.debug:
                            logger.info(f"\nProcessed sample {sample_idx + 1}/{len(dataset)}...")
                            logger.info(f"Hash ID: {hash_id}")
                            print_debug_info(result, sample_idx)
                    else:
                        num_failed += 1
                        logger.warning(f"Sample {sample_idx} returned None")
                    
                except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
                    # ANY CUDA-related error gets aggressive cleanup
                    num_failed += 1
                    logger.error(f"CUDA error processing sample {sample_idx}: {e}")
                    logger.warning("Performing aggressive memory cleanup...")
                    
                    # MULTIPLE rounds of aggressive cleanup
                    for _ in range(5):
                        clear_cuda_cache(aggressive=True)
                    
                    # Additional cleanup
                    gc.collect()
                    if torch.cuda.is_available():
                        try:
                            torch.cuda.ipc_collect()  # Clean up inter-process memory
                        except:
                            pass
                    
                    if args.debug:
                        logger.error(traceback.format_exc())
                    continue
                    
                except Exception as e:
                    num_failed += 1
                    logger.error(f"Failed to process sample {sample_idx}: {e}")
                    if args.debug:
                        logger.error(traceback.format_exc())
                    # Clear cache after any error
                    clear_cuda_cache()
                    continue
                
                # AGGRESSIVE cleanup after every sample to prevent memory accumulation
                clear_cuda_cache(aggressive=True)

                # Extra cleanup every 5 samples
                if num_saved % 5 == 0:
                    for _ in range(3):  # Multiple rounds
                        clear_cuda_cache(aggressive=True)
                    logger.info(f"Performed extra cleanup after {num_saved} samples")
                
                # More aggressive clearing every 100 samples
                if num_saved % 100 == 0:
                    clear_cuda_cache(aggressive=True)
                    logger.info(f"Processed {num_saved} samples, performed aggressive CUDA cache clear")
            
            # Create manifest
            create_manifest(output_dir_for_dataset, num_saved, vars(args))
            
            # Summary
            logger.info(f"\n{'='*80}")
            logger.info(f"Extraction complete for {target_dataset}!")
            logger.info(f"  Saved: {num_saved} samples")
            logger.info(f"  Skipped (already done): {num_skipped} samples")
            logger.info(f"  Failed: {num_failed} samples")
            logger.info(f"  Output directory: {output_dir_for_dataset}")
            logger.info(f"{'='*80}")
            
            # Clear cache between datasets
            clear_cuda_cache(aggressive=True)
        
        # Remove model-specific logging handler
        remove_model_logging(model_id)
        
        # Clear model from memory before loading next model
        del model, processor
        clear_cuda_cache(aggressive=True)
        logger.info(f"Cleared model {model_id} from memory")
    
    logger.info(f"\n{'='*80}")
    logger.info("All models and datasets processed!")
    logger.info(f"{'='*80}")
    
    return args.output_dir


if __name__ == "__main__":
    output_path = main()
    print(f"\nExtraction completed. Results: {output_path}")

# Qwen/Qwen3-VL-8B-Instruct
# llava-hf/llava-v1.6-vicuna-13b-hf - 13B
# google/gemma-3-27b-it - 27B
# deepseek-ai/deepseek-vl2 - 27B
# OpenGVLab/InternVL3_5-14B-HF

## Example Usage:
# python generate_and_extract.py \
#     --model_id "Qwen/Qwen3-VL-8B-Instruct" \
#     --dataset_path "../../data/VLCB/raw" \
#     --target_datasets train validation test \
#     --image_column "image" \
#     --question_column "question" \
#     --answer_column "answer" \
#     --output_dir "../../data/extraction/raw/" \
#     --save_top_logits \
#     --top_logits 100 \
#     --max_new_tokens 64 \
#     --attention_aggregation "mean" \
#     --compression "compressed" \
#     --dtype "float32" \
#     --gpu_ids "0,5,6,7" \
#     --max_image_dim 2048 \
#     --skip-if-processed \
#     --skip-correctness-assessment

#     --max_samples 3 \
#     --debug \
#     --debug_samples 5 \
#     --skip-if-processed