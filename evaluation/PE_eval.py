#!/usr/bin/env python3
"""
Prompt Ensemble (PE) Evaluation Script for Vision-Language Models.

This script implements the Prompt Ensemble confidence estimation method for VLMs:
1. Loads samples with pre-generated paraphrases (from PE_paraphrase_generation.py)
2. Runs inference on original question + all paraphrases
3. Extracts logprobs for generated tokens to compute confidence
4. Averages confidence scores across all prompt variations
5. Uses original question's answer as the final prediction

The key insight: We don't change the prediction (unlike Self-Consistency).
Instead, we measure confidence stability across semantically equivalent prompts.

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
import tempfile

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
        logging.FileHandler("logs/PE_eval.log", mode='a')
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
    """
    # Convert to PIL Image
    if isinstance(img, Image.Image):
        pil_img = img
    elif isinstance(img, bytes):
        pil_img = Image.open(io.BytesIO(img))
    elif isinstance(img, dict) and 'bytes' in img:
        pil_img = Image.open(io.BytesIO(img['bytes']))
    else:
        return img
    
    # Ensure RGB mode
    if pil_img.mode != 'RGB':
        pil_img = pil_img.convert('RGB')
    
    w, h = pil_img.size
    max_current = max(w, h)
    
    if max_current <= max_dim:
        return pil_img
    
    scale = max_dim / max_current
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    resized = pil_img.resize((new_w, new_h), Image.LANCZOS)
    logger.debug(f"Resized image from {w}x{h} to {new_w}x{new_h}")
    
    return resized


def load_and_preprocess_image(image, max_dim=MAX_IMAGE_DIMENSION):
    """Load and preprocess image from various formats."""
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
    
    if pil_img.mode != 'RGB':
        pil_img = pil_img.convert('RGB')
    
    return resize_image_if_needed(pil_img, max_dim)


# ============================================================================
# Argument Parser
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description='Prompt Ensemble Evaluation for Vision-Language Models',
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
    
    # Data configuration
    parser.add_argument('--pe_output_path', type=str, required=True,
                        help='Output path where PE paraphrase data was created (e.g., ../data/PE/)')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directory containing extracted representations (npz files) from generate_and_extract.py (e.g., ../data/extraction/raw/)')
    parser.add_argument('--dataset_path', type=str, required=True,
                        help='Path to the original dataset (for images)')
    parser.add_argument('--test_dataset_name', type=str, required=True,
                        help='PE test dataset name (e.g., test_PE)')
    parser.add_argument('--source_dataset_name', type=str, default=None,
                        help='Source dataset name for loading images (e.g., test). If not provided, will try to infer from test_dataset_name.')
    parser.add_argument('--image_column', type=str, default='image',
                        help='Column name for images in dataset')
    parser.add_argument('--question_column', type=str, default='question',
                        help='Column name for questions in dataset')
    parser.add_argument('--answer_column', type=str, default='answer',
                        help='Column name for answers in dataset')
    
    # Output configuration
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save evaluation results')
    
    # PE-specific configuration
    parser.add_argument('--num_paraphrases_to_use', type=int, default=None,
                        help='Number of paraphrases to use (None = use all available)')
    parser.add_argument('--confidence_aggregation', type=str, default='mean',
                        choices=['mean', 'median', 'min', 'max'],
                        help='How to aggregate confidence across paraphrases')
    
    # Processing configuration
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Maximum number of samples to process (None for all)')
    parser.add_argument('--max_new_tokens', type=int, default=64,
                        help='Maximum tokens to generate for each response')
    parser.add_argument('--max_image_dim', type=int, default=MAX_IMAGE_DIMENSION,
                        help='Maximum image dimension (images larger than this will be resized)')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug mode with verbose output')
    parser.add_argument('--debug_samples', type=int, default=3,
                        help='Number of samples to show detailed debug info')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from existing results')
    
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


def load_npz_samples_with_paraphrases(samples_dir: Path) -> Dict[str, Dict[str, Any]]:
    """Load all npz sample files with paraphrases and index by hash_id."""
    samples = {}
    
    # Check if directory exists
    if not samples_dir.exists():
        logger.error(f"Directory does not exist: {samples_dir}")
        logger.error(f"Please ensure PE_paraphrase_generation.py has been run and created the dataset.")
        return samples
    
    if not samples_dir.is_dir():
        logger.error(f"Path is not a directory: {samples_dir}")
        return samples
    
    npz_files = list(samples_dir.glob("*.npz"))
    
    logger.info(f"Loading {len(npz_files)} npz files from {samples_dir}")
    
    for npz_file in tqdm(npz_files, desc="Loading npz samples"):
        try:
            # Use 'with' statement to ensure file handle is closed
            with np.load(npz_file, allow_pickle=True) as data:
                # Get hash_id (required)
                if 'hash_id' not in data.files:
                    logger.warning(f"Skipping {npz_file.name}: missing 'hash_id' field")
                    continue
                
                hash_id = str(data['hash_id'])
                
                # Load paraphrases (required)
                if 'paraphrases' not in data.files:
                    logger.warning(f"Skipping {npz_file.name}: missing 'paraphrases' field")
                    continue
                
                paraphrases = data['paraphrases']
                if isinstance(paraphrases, np.ndarray):
                    paraphrases = list(paraphrases)
                
                # Get optional fields with defaults
                # Try 'sample_id' first, then 'id', then use hash_id as fallback
                sample_id = None
                if 'sample_id' in data.files:
                    sample_id = str(data['sample_id'])
                elif 'id' in data.files:
                    sample_id = str(data['id'])
                else:
                    sample_id = hash_id  # Use hash_id as fallback
                
                # Get question (required for evaluation)
                if 'question' not in data.files:
                    logger.warning(f"Skipping {npz_file.name}: missing 'question' field")
                    continue
                question = str(data['question'])
                
                # Get answer (required for evaluation)
                if 'answer' not in data.files:
                    logger.warning(f"Skipping {npz_file.name}: missing 'answer' field")
                    continue
                answer = str(data['answer'])
                
                # Get generated_response (optional)
                generated_response = ""
                if 'generated_response' in data.files:
                    generated_response = str(data['generated_response'])
                
                # Get is_correct (optional - will be merged from generation extraction data if missing)
                is_correct = None
                if 'is_correct' in data.files:
                    is_correct = data['is_correct']
                    if isinstance(is_correct, np.ndarray):
                        is_correct = is_correct.item()
                
                samples[hash_id] = {
                    'hash_id': hash_id,
                    'sample_id': sample_id,
                    'question': question,
                    'answer': answer,
                    'generated_response': generated_response,
                    'is_correct': is_correct,  # May be None, will be filled from generation extraction
                    'paraphrases': paraphrases,
                    'num_paraphrases': len(paraphrases),
                }
        except KeyError as e:
            logger.warning(f"Error loading {npz_file.name}: missing field {e}")
            continue
        except Exception as e:
            logger.warning(f"Error loading {npz_file.name}: {e}")
            continue
    
    # Log statistics about paraphrases
    paraphrase_counts = [s['num_paraphrases'] for s in samples.values()]
    if paraphrase_counts:
        logger.info(f"Successfully loaded {len(samples)} samples")
        logger.info(f"Paraphrase stats: min={min(paraphrase_counts)}, max={max(paraphrase_counts)}, "
                   f"avg={np.mean(paraphrase_counts):.1f}")
    
    return samples


def load_generation_extraction_npz(samples_dir: Path) -> Dict[str, Dict[str, Any]]:
    """Load npz files from generation extraction directory to get is_correct values."""
    samples = {}
    
    # Check if directory exists
    if not samples_dir.exists():
        logger.warning(f"Generation extraction directory does not exist: {samples_dir}")
        return samples
    
    if not samples_dir.is_dir():
        logger.warning(f"Path is not a directory: {samples_dir}")
        return samples
    
    npz_files = list(samples_dir.glob("*.npz"))
    
    logger.info(f"Loading {len(npz_files)} generation extraction npz files from {samples_dir}")
    
    for npz_file in tqdm(npz_files, desc="Loading generation extraction npz"):
        try:
            # Use 'with' statement to ensure file handle is closed
            with np.load(npz_file, allow_pickle=True) as data:
                # Get hash_id (required)
                if 'hash_id' not in data.files:
                    continue
                
                hash_id = str(data['hash_id'])
                
                # Get is_correct (required)
                if 'is_correct' not in data.files:
                    continue
                
                is_correct = data['is_correct']
                if isinstance(is_correct, np.ndarray):
                    is_correct = is_correct.item()
                
                sample_data = {
                    'hash_id': hash_id,
                    'is_correct': is_correct,
                    'original_response': None,
                }
                
                # Get generated_response if available (for reference/debugging)
                if 'generated_response' in data.files:
                    sample_data['original_response'] = str(data['generated_response'])
                
                samples[hash_id] = sample_data
        except KeyError as e:
            logger.debug(f"Error loading {npz_file.name}: missing field {e}")
            continue
        except Exception as e:
            logger.debug(f"Error loading {npz_file.name}: {e}")
            continue
    
    logger.info(f"Successfully loaded {len(samples)} generation extraction samples with is_correct")
    return samples


# ============================================================================
# HuggingFace Model Loading (matching generate_and_extract.py)
# ============================================================================

def load_hf_model_and_processor(model_id, dtype_str, device):
    """Load VLM model and processor/tokenizer using HuggingFace."""
    logger.info(f"Loading HF model: {model_id}")
    logger.info(f"Dtype: {dtype_str}")
    logger.info(f"Device: {device}")
    
    dtype_map = {
        'float32': torch.float32,
        'float16': torch.float16,
        'bfloat16': torch.bfloat16,
    }
    dtype = dtype_map[dtype_str]
    
    model_id_lower = model_id.lower()
    
    if 'deepseek' in model_id_lower and 'vl' in model_id_lower:
        if not DEEPSEEK_AVAILABLE:
            raise ImportError(
                "DeepSeek VL2 dependencies not found. Please install deepseek_vl2."
            )
        model_type = 'deepseek'
        logger.info("Detected DeepSeek VL2 model")

        if dtype_str != 'bfloat16':
            logger.warning(f"DeepSeek VL2 requires bfloat16. Overriding '{dtype_str}' to 'bfloat16'.")
        dtype_str = 'bfloat16'
        dtype = torch.bfloat16
        
        processor = DeepseekVLV2Processor.from_pretrained(model_id)
        
        # DeepSeek VL2 doesn't work well with device_map="auto" due to internal architecture
        # Load to single GPU like generate_and_extract.py does
        model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)
        model = model.to(dtype).cuda().eval()
        
        logger.info(f"Model loaded on single GPU (DeepSeek VL2 requires single-GPU mode)")
        
        return model, processor, model_type
        
    elif 'llava' in model_id_lower:
        model_type = 'llava'
        model_class = LlavaNextForConditionalGeneration
        processor_class = LlavaNextProcessor
        processor_kwargs = {}
        
    elif 'qwen' in model_id_lower:
        model_type = 'qwen'
        model_class = Qwen3VLForConditionalGeneration
        processor_class = AutoProcessor
        processor_kwargs = {'trust_remote_code': True}
        
    elif 'gemma' in model_id_lower:
        model_type = 'gemma'
        model_class = Gemma3ForConditionalGeneration
        processor_class = Gemma3Processor
        processor_kwargs = {}
        
    elif 'internvl' in model_id_lower:
        model_type = 'internvl'
        model_class = AutoModelForImageTextToText
        processor_class = AutoProcessor
        processor_kwargs = {'trust_remote_code': True}
        
    else:
        raise ValueError(f"Unknown model type: {model_id}")
    
    logger.info(f"Detected {model_type} model")
    
    # When CUDA_VISIBLE_DEVICES is set, device indices are remapped
    # (e.g., "5,6,7" -> visible devices 0,1,2)
    num_visible_gpus = torch.cuda.device_count()
    max_memory = {i: "130GiB" for i in range(num_visible_gpus)}
    
    model = model_class.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map="auto",
        max_memory=max_memory,
        attn_implementation='eager',
        trust_remote_code=True,
    )
    model.eval()
    
    processor = processor_class.from_pretrained(model_id, **processor_kwargs)
    
    logger.info("Model and processor loaded successfully")
    return model, processor, model_type


# ============================================================================
# vLLM Inference with Logprobs
# ============================================================================

class VLLMPEInference:
    """vLLM inference wrapper for Prompt Ensemble with logprob extraction."""
    
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
        
        model_config = VLLM_MODEL_CONFIGS.get(model_name, {})
        
        logger.info(f"Loading processor for {model_name}...")
        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        
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
        
        dtype_map = {'float32': 'float32', 'float16': 'float16', 'bfloat16': 'bfloat16'}
        vllm_dtype = dtype_map.get(dtype, 'float32')
        
        logger.info(f"Loading vLLM model {model_name}...")
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
    
    def batch_inference_with_logprobs(
        self,
        images: List[Image.Image],
        questions: List[str],
        max_tokens: int = 64,
        temperature: float = 0.0,
        use_tqdm: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Run batch inference with logprob extraction.
        
        Returns:
            List of dicts with 'text', 'token_logprobs', 'token_ids', 'confidence'
        """
        assert len(images) == len(questions)
        
        inputs = []
        for img, question in zip(images, questions):
            messages = self._build_messages(img, question)
            prompt = self._apply_chat_template(messages)
            inputs.append({
                "prompt": prompt,
                "multi_modal_data": {"image": img},
            })
        
        # logprobs=1 returns logprob for the top-1 (greedy) token
        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            logprobs=1,  # Get logprob of the generated token
        )
        
        outputs = self.llm.generate(inputs, sampling_params=sampling_params, use_tqdm=use_tqdm)
        
        results = []
        for output in outputs:
            gen_output = output.outputs[0]
            text = gen_output.text
            token_ids = gen_output.token_ids
            logprobs_list = gen_output.logprobs  # List of dicts, one per token
            
            # Extract logprobs for each generated token
            token_logprobs = []
            if logprobs_list:
                for i, logprob_dict in enumerate(logprobs_list):
                    if logprob_dict and token_ids[i] in logprob_dict:
                        token_logprobs.append(logprob_dict[token_ids[i]].logprob)
                    else:
                        # Fallback if token not in logprobs (shouldn't happen with logprobs=1)
                        token_logprobs.append(-100.0)
            
            # Per-response confidence: GEOMETRIC MEAN of token probabilities
            # (computed in log-space for numerical stability).
            if token_logprobs:
                token_probs = [np.exp(lp) for lp in token_logprobs]
                clipped = np.clip(np.asarray(token_probs, dtype=np.float64),
                                  1e-12, 1.0)
                confidence = float(np.exp(np.mean(np.log(clipped))))
            else:
                confidence = 0.0
            
            results.append({
                'text': text,
                'token_ids': list(token_ids),
                'token_logprobs': token_logprobs,
                'token_probs': [np.exp(lp) for lp in token_logprobs] if token_logprobs else [],
                'confidence': confidence,
            })
        
        return results


# ============================================================================
# HuggingFace Prompt Ensemble Extractor
# ============================================================================

class HFPEExtractor:
    """Extracts Prompt Ensemble confidence scores from VLMs using HuggingFace."""
    
    def __init__(self, model, processor, model_type, max_new_tokens=64, debug=False, max_image_dim=MAX_IMAGE_DIMENSION):
        self.model = model
        self.processor = processor
        self.model_type = model_type
        self.max_new_tokens = max_new_tokens
        self.debug = debug
        self.max_image_dim = max_image_dim
        self.device = next(model.parameters()).device
    
    def extract_with_logprobs(self, image, question: str, sample_id: str = None) -> Dict[str, Any]:
        """
        Generate response and extract logprobs for confidence estimation.
        
        Returns:
            Dict with 'text', 'token_probs', 'confidence', 'success'
        """
        result = {
            'text': None,
            'token_probs': [],
            'confidence': None,
            'success': False
        }
        
        image = load_and_preprocess_image(image, max_dim=self.max_image_dim)
        
        try:
            if self.model_type == 'qwen':
                result = self._extract_qwen(image, question)
            elif self.model_type == 'llava':
                result = self._extract_llava(image, question)
            elif self.model_type == 'gemma':
                result = self._extract_gemma(image, question)
            elif self.model_type == 'internvl':
                result = self._extract_internvl(image, question)
            elif self.model_type == 'deepseek':
                result = self._extract_deepseek(image, question)
            else:
                logger.error(f"Unknown model type: {self.model_type}")
                
        except torch.cuda.OutOfMemoryError as e:
            logger.error(f"CUDA OOM error: {e}")
            torch.cuda.empty_cache()
            gc.collect()
        except Exception as e:
            import traceback
            logger.error(f"Error extracting: {e}")
            logger.error(traceback.format_exc())
        
        return result
    
    def _compute_confidence_from_scores(self, scores, generated_ids):
        """
        Compute confidence from generation scores.
        
        Args:
            scores: Tuple of score tensors, one per generation step
            generated_ids: Generated token IDs
        
        Returns:
            Tuple of (confidence, token_probs)
        """
        token_probs = []
        
        for i, score in enumerate(scores):
            # Apply softmax to get probabilities
            probs = F.softmax(score, dim=-1)
            
            # Get the probability of the generated token
            if i < len(generated_ids):
                token_id = generated_ids[i].item()
                token_prob = probs[0, token_id].item()
                token_probs.append(token_prob)
        
        # Per-response confidence: GEOMETRIC MEAN of token probabilities
        # (computed in log-space for numerical stability).
        if token_probs:
            clipped = np.clip(np.asarray(token_probs, dtype=np.float64),
                              1e-12, 1.0)
            confidence = float(np.exp(np.mean(np.log(clipped))))
        else:
            confidence = 0.0

        return confidence, token_probs
    
    def _extract_qwen(self, image, question):
        """Extract for Qwen model."""
        result = {'text': None, 'token_probs': [], 'confidence': None, 'success': False}
        
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": question},
                ],
            }
        ]
        
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)
        
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                return_dict_in_generate=True,
                output_scores=True,
            )
        
        input_len = inputs.input_ids.shape[1]
        generated_ids = output.sequences[0][input_len:]
        response_text = self.processor.decode(generated_ids, skip_special_tokens=True)
        
        confidence, token_probs = self._compute_confidence_from_scores(output.scores, generated_ids)
        
        result['text'] = response_text
        result['token_probs'] = token_probs
        result['confidence'] = confidence
        result['success'] = True
        
        return result
    
    def _extract_llava(self, image, question):
        """Extract for LLaVA model."""
        result = {'text': None, 'token_probs': [], 'confidence': None, 'success': False}
        
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": question + "\n\n" + APPENDED_SYSTEM_PROMPT},
                ],
            },
        ]
        
        prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        inputs = self.processor(images=image, text=prompt, return_tensors="pt")
        
        device = next(self.model.parameters()).device
        model_dtype = next(self.model.parameters()).dtype
        
        inputs = {k: v.to(device=device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(dtype=model_dtype)
        
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                return_dict_in_generate=True,
                output_scores=True,
            )
        
        input_len = inputs["input_ids"].shape[1]
        generated_ids = output.sequences[0][input_len:]
        response_text = self.processor.decode(generated_ids, skip_special_tokens=True)
        
        confidence, token_probs = self._compute_confidence_from_scores(output.scores, generated_ids)
        
        result['text'] = response_text
        result['token_probs'] = token_probs
        result['confidence'] = confidence
        result['success'] = True
        
        return result
    
    def _extract_gemma(self, image, question):
        """Extract for Gemma model."""
        result = {'text': None, 'token_probs': [], 'confidence': None, 'success': False}
        
        conversation = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": question},
                ],
            },
        ]
        
        prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        inputs = self.processor(images=image, text=prompt, return_tensors="pt")
        
        device = next(self.model.parameters()).device
        model_dtype = next(self.model.parameters()).dtype
        
        inputs = {k: v.to(device=device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(dtype=model_dtype)
        
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                return_dict_in_generate=True,
                output_scores=True,
            )
        
        input_len = inputs["input_ids"].shape[1]
        generated_ids = output.sequences[0][input_len:]
        response_text = self.processor.decode(generated_ids, skip_special_tokens=True)
        
        confidence, token_probs = self._compute_confidence_from_scores(output.scores, generated_ids)
        
        result['text'] = response_text
        result['token_probs'] = token_probs
        result['confidence'] = confidence
        result['success'] = True
        
        return result
    
    def _extract_internvl(self, image, question):
        """Extract for InternVL model."""
        result = {'text': None, 'token_probs': [], 'confidence': None, 'success': False}
        
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": question},
                ],
            },
        ]
        
        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        inputs = self.processor(images=image, text=prompt, return_tensors="pt")
        
        device = next(self.model.parameters()).device
        model_dtype = next(self.model.parameters()).dtype
        
        inputs = {k: v.to(device=device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(dtype=model_dtype)
        
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                return_dict_in_generate=True,
                output_scores=True,
            )
        
        input_len = inputs["input_ids"].shape[1]
        generated_ids = output.sequences[0][input_len:]
        response_text = self.processor.decode(generated_ids, skip_special_tokens=True)
        
        confidence, token_probs = self._compute_confidence_from_scores(output.scores, generated_ids)
        
        result['text'] = response_text
        result['token_probs'] = token_probs
        result['confidence'] = confidence
        result['success'] = True
        
        return result
    
    def _extract_deepseek(self, image, question):
        """Extract for DeepSeek model."""
        result = {'text': None, 'token_probs': [], 'confidence': None, 'success': False}
        
        temp_image_created = False
        image_path = None
        
        try:
            temp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
            image.save(temp_file.name)
            image_path = temp_file.name
            temp_image_created = True
            
            conversation = [
                {
                    "role": "<|User|>",
                    "content": f"<image>\n{question}",
                    "images": [image_path],
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
            if hasattr(prepare_inputs, 'pixel_values') and prepare_inputs.pixel_values is not None:
                prepare_inputs.pixel_values = prepare_inputs.pixel_values.to(dtype=model_dtype)
            
            inputs_embeds = self.model.prepare_inputs_embeds(**prepare_inputs)
            
            with torch.no_grad():
                output = self.model.language.generate(
                    inputs_embeds=inputs_embeds,
                    attention_mask=prepare_inputs.attention_mask,
                    pad_token_id=self.processor.tokenizer.eos_token_id,
                    bos_token_id=self.processor.tokenizer.bos_token_id,
                    eos_token_id=self.processor.tokenizer.eos_token_id,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    return_dict_in_generate=True,
                    output_scores=True,
                )
            
            generated_ids = output.sequences[0]
            response_text = self.processor.tokenizer.decode(generated_ids, skip_special_tokens=True)
            
            confidence, token_probs = self._compute_confidence_from_scores(output.scores, generated_ids)
            
            result['text'] = response_text
            result['token_probs'] = token_probs
            result['confidence'] = confidence
            result['success'] = True
            
        finally:
            if temp_image_created and image_path:
                try:
                    os.unlink(image_path)
                except:
                    pass
        
        return result


# ============================================================================
# vLLM Prompt Ensemble Extractor
# ============================================================================

class VLLMPEExtractor:
    """Extracts Prompt Ensemble confidence scores from VLMs using vLLM."""
    
    def __init__(self, vlm_inference: VLLMPEInference, max_new_tokens=64, debug=False, max_image_dim=MAX_IMAGE_DIMENSION):
        self.vlm = vlm_inference
        self.max_new_tokens = max_new_tokens
        self.debug = debug
        self.max_image_dim = max_image_dim
    
    def extract_batch_with_logprobs(
        self, 
        images: List, 
        questions: List[str],
    ) -> List[Dict[str, Any]]:
        """
        Extract responses and logprobs for a batch of image-question pairs.
        """
        # Preprocess images
        processed_images = []
        valid_indices = []
        
        for i, img in enumerate(images):
            try:
                processed = load_and_preprocess_image(img, max_dim=self.max_image_dim)
                processed_images.append(processed)
                valid_indices.append(i)
            except Exception as e:
                logger.warning(f"Failed to preprocess image {i}: {e}")
        
        if not processed_images:
            return [{'text': None, 'confidence': None, 'success': False} for _ in images]
        
        # Run batch inference
        valid_questions = [questions[i] for i in valid_indices]
        
        try:
            batch_results = self.vlm.batch_inference_with_logprobs(
                processed_images,
                valid_questions,
                max_tokens=self.max_new_tokens,
                temperature=0.0,
                use_tqdm=False,
            )
        except Exception as e:
            logger.error(f"Batch inference failed: {e}")
            return [{'text': None, 'confidence': None, 'success': False} for _ in images]
        
        # Map results back
        results = [{'text': None, 'token_probs': [], 'confidence': None, 'success': False} for _ in images]
        
        for idx, br in zip(valid_indices, batch_results):
            results[idx] = {
                'text': br['text'],
                'token_probs': br['token_probs'],
                'confidence': br['confidence'],
                'success': True,
            }
        
        return results


# ============================================================================
# Main Evaluator
# ============================================================================

class PEEvaluator:
    """Main evaluator class for Prompt Ensemble evaluation."""
    
    def __init__(self, extractor, args, use_vllm=False):
        self.extractor = extractor
        self.args = args
        self.use_vllm = use_vllm
    
    def _aggregate_confidences(self, confidences: List[float], method: str = 'mean') -> float:
        """Aggregate confidence scores across prompt variations."""
        if not confidences:
            return 0.0
        
        if method == 'mean':
            return float(np.mean(confidences))
        elif method == 'median':
            return float(np.median(confidences))
        elif method == 'min':
            return float(np.min(confidences))
        elif method == 'max':
            return float(np.max(confidences))
        else:
            return float(np.mean(confidences))
    
    def evaluate(self, dataset, npz_samples: Dict[str, Dict[str, Any]], output_dir: Path) -> Dict[str, Any]:
        """Run Prompt Ensemble evaluation."""
        logger.info("Starting Prompt Ensemble evaluation...")
        
        output_dir.mkdir(parents=True, exist_ok=True)
        labels_path = output_dir / "test_labels.json"
        
        # Load existing records if resuming
        existing_records = []
        processed_hash_ids = set()
        if self.args.resume and labels_path.exists():
            logger.info(f"Resume mode: Loading existing results from {labels_path}")
            try:
                with open(labels_path, 'r', encoding='utf-8') as f:
                    existing_records = json.load(f)
                processed_hash_ids = {record['hash_id'] for record in existing_records}
                logger.info(f"Loaded {len(existing_records)} existing records.")
            except Exception as e:
                logger.warning(f"Failed to load existing results: {e}")
        
        evaluation_records = existing_records.copy()
        new_records_count = 0
        skipped_already_processed = 0
        skipped_no_npz = 0
        skipped_no_correctness = 0
        skipped_no_paraphrases = 0
        failed_extraction = 0
        
        # Limit samples if specified
        total_samples = len(dataset)
        if self.args.max_samples:
            total_samples = min(total_samples, self.args.max_samples)
        
        logger.info(f"Processing {total_samples} samples...")
        
        # Pre-extract hash_ids
        all_hash_ids = dataset['hash_id'][:total_samples]
        
        # Build list of indices to process
        indices_to_process = []
        for idx, hash_id in enumerate(all_hash_ids):
            if hash_id in processed_hash_ids:
                skipped_already_processed += 1
                continue
            if hash_id not in npz_samples:
                skipped_no_npz += 1
                continue
            if npz_samples[hash_id].get('is_correct') is None:
                skipped_no_correctness += 1
                continue
            if not npz_samples[hash_id].get('paraphrases'):
                skipped_no_paraphrases += 1
                continue
            indices_to_process.append(idx)
        
        logger.info(f"Found {len(indices_to_process)} samples to process")
        logger.info(f"Skipped: already_processed={skipped_already_processed}, no_npz={skipped_no_npz}, "
                   f"no_correctness={skipped_no_correctness}, no_paraphrases={skipped_no_paraphrases}")
        
        # Process samples
        if self.use_vllm:
            evaluation_records, new_records_count, failed_extraction = \
                self._process_vllm_batched(dataset, npz_samples, indices_to_process, 
                                           evaluation_records, processed_hash_ids, labels_path)
        else:
            evaluation_records, new_records_count, failed_extraction = \
                self._process_hf_sequential(dataset, npz_samples, indices_to_process,
                                            evaluation_records, processed_hash_ids, labels_path)
        
        # Log summary
        logger.info(f"\n{'='*60}")
        logger.info(f"PROCESSING SUMMARY:")
        logger.info(f"  Total successfully processed: {len(evaluation_records)}")
        logger.info(f"  New records: {new_records_count}")
        logger.info(f"  Failed extraction: {failed_extraction}")
        logger.info(f"{'='*60}")
        
        if not evaluation_records:
            logger.error("No valid records to evaluate!")
            return None
        
        # Save final records
        self._save_records(evaluation_records, labels_path)
        
        # Calculate metrics
        labels = np.array([r['ground_truth_correctness'] for r in evaluation_records], dtype=float)
        scores = np.array([r['ensemble_confidence'] for r in evaluation_records], dtype=float)
        
        metrics = calculate_all_metrics(labels, scores)
        
        # Build results structure
        confidence_values = [r['ensemble_confidence'] for r in evaluation_records]
        original_confidence_values = [r['original_confidence'] for r in evaluation_records]
        
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
                'num_paraphrases_used': self.args.num_paraphrases_to_use,
                'confidence_aggregation': self.args.confidence_aggregation,
                'total_records': len(evaluation_records),
                'evaluation_timestamp': datetime.now().isoformat(),
                'ensemble_confidence_statistics': {
                    'avg_confidence': float(np.mean(confidence_values)),
                    'confidence_std': float(np.std(confidence_values)),
                    'min_confidence': float(np.min(confidence_values)),
                    'max_confidence': float(np.max(confidence_values)),
                },
                'original_confidence_statistics': {
                    'avg_confidence': float(np.mean(original_confidence_values)),
                    'confidence_std': float(np.std(original_confidence_values)),
                    'min_confidence': float(np.min(original_confidence_values)),
                    'max_confidence': float(np.max(original_confidence_values)),
                },
            }
        }
        
        # Save results
        results_path = output_dir / "test_results.json"
        save_evaluation_results(results, str(results_path))
        
        self._print_results_summary(results)
        
        return results
    
    def _process_hf_sequential(self, dataset, npz_samples, indices_to_process,
                                evaluation_records, processed_hash_ids, labels_path):
        """Process samples sequentially using HuggingFace."""
        new_records_count = 0
        failed_extraction = 0
        
        for idx in tqdm(indices_to_process, desc="Processing samples (HF)"):
            sample = dataset[idx]
            hash_id = sample['hash_id']
            npz_data = npz_samples[hash_id]
            
            image = sample[self.args.image_column]
            original_question = sample[self.args.question_column]
            paraphrases = npz_data['paraphrases']
            is_correct = npz_data['is_correct']
            
            # Limit paraphrases if specified
            if self.args.num_paraphrases_to_use:
                paraphrases = paraphrases[:self.args.num_paraphrases_to_use]
            
            # Build all questions: original + paraphrases
            all_questions = [original_question] + list(paraphrases)
            
            # Debug output
            show_debug = self.args.debug and new_records_count < self.args.debug_samples
            if show_debug:
                print(f"\n{'='*100}")
                print(f"🔬 SAMPLE {new_records_count + 1}")
                print(f"📝 Original Question: {original_question[:100]}...")
                print(f"📊 Paraphrases: {len(paraphrases)}")
                print(f"✅ Is Correct: {is_correct}")
            
            # Extract confidence for each question (original + paraphrases)
            all_confidences = []
            original_confidence = None
            original_response = None
            
            for i, question in enumerate(all_questions):
                result = self.extractor.extract_with_logprobs(image, question, sample_id=hash_id)
                
                if result['success']:
                    all_confidences.append(result['confidence'])
                    
                    if i == 0:  # Original question
                        original_confidence = result['confidence']
                        original_response = result['text']
                    
                    if show_debug and i < 3:
                        q_type = "Original" if i == 0 else f"Paraphrase {i}"
                        print(f"   {q_type}: conf={result['confidence']:.4f}, text='{result['text'][:50]}...'")
                else:
                    if show_debug:
                        print(f"   Question {i}: FAILED")
            
            if not all_confidences or original_confidence is None:
                failed_extraction += 1
                continue
            
            # Compute ensemble confidence
            ensemble_confidence = self._aggregate_confidences(
                all_confidences, self.args.confidence_aggregation
            )
            
            if show_debug:
                print(f"   📊 Ensemble Confidence ({self.args.confidence_aggregation}): {ensemble_confidence:.4f}")
                print(f"   📊 Original Confidence: {original_confidence:.4f}")
            
            # Store record
            record = {
                'sample_id': npz_data['sample_id'],
                'hash_id': hash_id,
                'ground_truth_correctness': int(is_correct),
                'ensemble_confidence': ensemble_confidence,
                'original_confidence': original_confidence,
                'all_confidences': all_confidences,
                'original_response': original_response,
                'num_prompts_used': len(all_confidences),
                'dataset': sample.get('dataset', 'unknown'),
            }
            evaluation_records.append(record)
            processed_hash_ids.add(hash_id)
            new_records_count += 1
            
            # Incremental save
            if new_records_count > 0 and new_records_count % 50 == 0:
                self._save_records(evaluation_records, labels_path)
                logger.info(f"Incremental save: {len(evaluation_records)} records")
            
            # Clear CUDA cache periodically
            if new_records_count % 25 == 0:
                gc.collect()
                torch.cuda.empty_cache()
        
        return evaluation_records, new_records_count, failed_extraction
    
    def _process_vllm_batched(self, dataset, npz_samples, indices_to_process,
                               evaluation_records, processed_hash_ids, labels_path):
        """Process all samples using vLLM batched inference."""
        new_records_count = 0
        failed_extraction = 0
        
        logger.info(f"Preparing batch data for {len(indices_to_process)} samples...")
        
        # Collect all data - batch original + all paraphrases together
        batch_data = []  # List of (sample_idx, question_idx, image, question)
        sample_metadata = {}  # sample_idx -> metadata
        
        for sample_idx, idx in enumerate(tqdm(indices_to_process, desc="Preparing batch")):
            sample = dataset[idx]
            hash_id = sample['hash_id']
            npz_data = npz_samples[hash_id]
            
            image = sample[self.args.image_column]
            original_question = sample[self.args.question_column]
            paraphrases = npz_data['paraphrases']
            
            # Limit paraphrases if specified
            if self.args.num_paraphrases_to_use:
                paraphrases = paraphrases[:self.args.num_paraphrases_to_use]
            
            # Store metadata
            sample_metadata[sample_idx] = {
                'hash_id': hash_id,
                'sample_id': npz_data['sample_id'],
                'is_correct': npz_data['is_correct'],
                'dataset': sample.get('dataset', 'unknown'),
                'num_questions': 1 + len(paraphrases),
            }
            
            # Add original question
            batch_data.append((sample_idx, 0, image, original_question))
            
            # Add paraphrases
            for p_idx, paraphrase in enumerate(paraphrases):
                batch_data.append((sample_idx, p_idx + 1, image, paraphrase))
        
        if not batch_data:
            logger.warning("No valid samples to process")
            return evaluation_records, new_records_count, failed_extraction
        
        logger.info(f"Running inference on {len(batch_data)} total prompts...")
        
        # Extract all at once
        all_images = [bd[2] for bd in batch_data]
        all_questions = [bd[3] for bd in batch_data]
        
        all_results = self.extractor.extract_batch_with_logprobs(all_images, all_questions)
        
        # Group results by sample
        sample_results = {}  # sample_idx -> list of (question_idx, result)
        
        for bd, result in zip(batch_data, all_results):
            sample_idx, question_idx, _, _ = bd
            if sample_idx not in sample_results:
                sample_results[sample_idx] = []
            sample_results[sample_idx].append((question_idx, result))
        
        # Process each sample's results
        for sample_idx, results in sample_results.items():
            meta = sample_metadata[sample_idx]
            
            # Sort by question_idx to ensure original is first
            results.sort(key=lambda x: x[0])
            
            all_confidences = []
            original_confidence = None
            original_response = None
            
            for question_idx, result in results:
                if result['success']:
                    all_confidences.append(result['confidence'])
                    
                    if question_idx == 0:  # Original question
                        original_confidence = result['confidence']
                        original_response = result['text']
            
            if not all_confidences or original_confidence is None:
                failed_extraction += 1
                continue
            
            # Compute ensemble confidence
            ensemble_confidence = self._aggregate_confidences(
                all_confidences, self.args.confidence_aggregation
            )
            
            # Store record
            record = {
                'sample_id': meta['sample_id'],
                'hash_id': meta['hash_id'],
                'ground_truth_correctness': int(meta['is_correct']),
                'ensemble_confidence': ensemble_confidence,
                'original_confidence': original_confidence,
                'all_confidences': all_confidences,
                'original_response': original_response,
                'num_prompts_used': len(all_confidences),
                'dataset': meta['dataset'],
            }
            evaluation_records.append(record)
            processed_hash_ids.add(meta['hash_id'])
            new_records_count += 1
        
        logger.info(f"Processed {new_records_count} samples successfully")
        
        return evaluation_records, new_records_count, failed_extraction
    
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
        print(f"PROMPT ENSEMBLE EVALUATION SUMMARY")
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
        
        if 'metadata' in results:
            meta = results['metadata']
            print(f"\nConfiguration:")
            print(f"  Aggregation: {meta['confidence_aggregation']}")
            print(f"  Num paraphrases: {meta['num_paraphrases_used'] or 'all'}")
            
            if 'ensemble_confidence_statistics' in meta:
                stats = meta['ensemble_confidence_statistics']
                print(f"\nEnsemble Confidence Statistics:")
                print(f"  Average: {stats['avg_confidence']:.4f}")
                print(f"  Std: {stats['confidence_std']:.4f}")
                print(f"  Range: [{stats['min_confidence']:.4f}, {stats['max_confidence']:.4f}]")
            
            if 'original_confidence_statistics' in meta:
                stats = meta['original_confidence_statistics']
                print(f"\nOriginal Confidence Statistics:")
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
    logger.info("PROMPT ENSEMBLE (PE) EVALUATION FOR VISION-LANGUAGE MODELS")
    logger.info("="*80)
    logger.info(f"Model: {args.model_id}")
    logger.info(f"Backend: {args.backend}")
    logger.info(f"Dtype: {args.dtype}")
    logger.info(f"Test dataset: {args.test_dataset_name}")
    logger.info(f"GPU IDs: {args.gpu_ids}")
    logger.info(f"Confidence aggregation: {args.confidence_aggregation}")
    logger.info(f"Num paraphrases to use: {args.num_paraphrases_to_use or 'all'}")
    logger.info(f"Debug mode: {args.debug}")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")
    
    # Load model based on backend
    if args.backend == 'vllm':
        if not VLLM_AVAILABLE:
            logger.error("vLLM not available. Please install with: pip install vllm")
            sys.exit(1)
        
        vlm_inference = VLLMPEInference(
            args.model_id,
            dtype=args.dtype,
            gpu_memory_utilization=args.gpu_memory_utilization,
            tensor_parallel_size=args.tensor_parallel_size,
        )
        extractor = VLLMPEExtractor(
            vlm_inference,
            max_new_tokens=args.max_new_tokens,
            debug=args.debug,
            max_image_dim=args.max_image_dim,
        )
        use_vllm = True
    else:
        model, processor, model_type = load_hf_model_and_processor(args.model_id, args.dtype, device)
        extractor = HFPEExtractor(
            model, processor, model_type,
            max_new_tokens=args.max_new_tokens,
            debug=args.debug,
            max_image_dim=args.max_image_dim,
        )
        use_vllm = False
    
    # Load original dataset (for images)
    if args.source_dataset_name:
        source_dataset_name = args.source_dataset_name
    else:
        # Try to infer source dataset name from PE dataset name
        # Remove _PE suffix if present
        base_name = args.test_dataset_name.replace('_PE', '')
        
        # Check if base name exists
        base_path = os.path.join(args.dataset_path, base_name)
        if os.path.exists(base_path):
            source_dataset_name = base_name
        else:
            # Try adding _raw suffix (common pattern)
            raw_name = base_name + '_raw'
            raw_path = os.path.join(args.dataset_path, raw_name)
            if os.path.exists(raw_path):
                source_dataset_name = raw_name
            else:
                # Fall back to base name and let load_dataset handle the error
                logger.warning(f"Could not infer source dataset name. Tried: {base_name}, {raw_name}")
                logger.warning(f"Please provide --source_dataset_name explicitly.")
                source_dataset_name = base_name
    
    logger.info(f"Loading source dataset: {source_dataset_name}")
    full_dataset_path = os.path.join(args.dataset_path, source_dataset_name)
    dataset = load_dataset(full_dataset_path)
    
    # Load npz samples with paraphrases
    # Note: PE output structure is {pe_output_path}/{test_dataset_name}/samples/
    # (no model-specific directories)
    samples_dir = Path(args.pe_output_path) / args.test_dataset_name / "samples"
    logger.info(f"Loading npz samples from: {samples_dir}")
    npz_samples = load_npz_samples_with_paraphrases(samples_dir)
    
    if not npz_samples:
        logger.error(f"No npz samples found in {samples_dir}")
        logger.error(f"Expected directory structure: {args.pe_output_path}/{args.test_dataset_name}/samples/")
        logger.error(f"Please ensure:")
        logger.error(f"  1. PE_paraphrase_generation.py has been run successfully")
        logger.error(f"  2. The output dataset name matches --test_dataset_name: {args.test_dataset_name}")
        logger.error(f"  3. The --pe_output_path is correct: {args.pe_output_path}")
        return
    
    # Load generation extraction npz files to get is_correct values
    # Structure: {data_dir}/{model_name_part}/{source_dataset_name}/samples/
    gen_extraction_samples_dir = Path(args.data_dir) / model_name_part / source_dataset_name / "samples"
    logger.info(f"Loading generation extraction npz samples from: {gen_extraction_samples_dir}")
    gen_extraction_samples = load_generation_extraction_npz(gen_extraction_samples_dir)
    
    if not gen_extraction_samples:
        logger.warning(f"No generation extraction samples found in {gen_extraction_samples_dir}")
        logger.warning(f"Expected directory structure: {args.data_dir}/{model_name_part}/{source_dataset_name}/samples/")
        logger.warning(f"Samples without is_correct will be skipped during evaluation")
    else:
        logger.info(f"Found {len(gen_extraction_samples)} generation extraction samples with is_correct")
    
    # Merge is_correct and original_response from generation extraction into PE samples
    merged_count = 0
    already_had_count = 0
    missing_count = 0
    
    for hash_id in npz_samples:
        if hash_id in gen_extraction_samples:
            gen_ext_data = gen_extraction_samples[hash_id]
            # Override is_correct from generation extraction data (even if it already existed)
            npz_samples[hash_id]['is_correct'] = gen_ext_data['is_correct']
            
            # Merge original_response if available (for reference/debugging)
            if gen_ext_data.get('original_response') is not None:
                npz_samples[hash_id]['original_response'] = gen_ext_data['original_response']
            
            merged_count += 1
        elif npz_samples[hash_id].get('is_correct') is not None:
            # Sample already had is_correct from PE data
            already_had_count += 1
        else:
            # Sample doesn't have is_correct in either place
            missing_count += 1
    
    logger.info(f"Merged is_correct from generation extraction: {merged_count} samples")
    if already_had_count > 0:
        logger.info(f"Samples that already had is_correct in PE data: {already_had_count}")
    if missing_count > 0:
        logger.warning(f"{missing_count} samples still missing is_correct after merge (will be skipped during evaluation)")
    
    # Check that samples have paraphrases
    samples_with_paraphrases = sum(1 for s in npz_samples.values() if s['num_paraphrases'] > 0)
    logger.info(f"Samples with paraphrases: {samples_with_paraphrases}/{len(npz_samples)}")
    
    if samples_with_paraphrases == 0:
        logger.error("No samples have paraphrases! Run PE_paraphrase_generation.py first.")
        return
    
    # Setup output directory
    output_dir = Path(args.output_dir) / model_name_part / args.test_dataset_name
    
    # Run evaluation
    evaluator = PEEvaluator(extractor, args, use_vllm=use_vllm)
    results = evaluator.evaluate(dataset, npz_samples, output_dir)
    
    if results:
        print(f"\n✅ Evaluation complete!")
        print(f"📁 Results saved to: {output_dir}")
    else:
        print("\n❌ Evaluation failed!")
        sys.exit(1)
    
    # Cleanup
    if not use_vllm:
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
# google/gemma-3-27b-it - 27B
# deepseek-ai/deepseek-vl2 - 27B
# OpenGVLab/InternVL3_5-14B-HF

# vLLM backend (faster batched inference):
# python PE_eval.py \
#     --model_id "deepseek-ai/deepseek-vl2" \
#     --backend hf \
#     --pe_output_path "../data/PE/" \
#     --data_dir "../data/extraction/raw/" \
#     --dataset_path "../data/VLCB/raw" \
#     --test_dataset_name "test_PE" \
#     --source_dataset_name "test" \
#     --output_dir "../results/PE" \
#     --gpu_ids "6,7" \
#     --tensor_parallel_size 2 \
#     --dtype "float32" \
#     --num_paraphrases_to_use 10 \
#     --confidence_aggregation mean \
#     --max_samples 10 \
#     --resume

# With all paraphrases and resume (source_dataset_name will be inferred if not provided):
# python PE_eval.py \
#     --model_id "google/gemma-3-27b-it" \
#     --backend vllm \
#     --pe_output_path "../data/PE/" \
#     --data_dir "../data/extraction/raw/" \
#     --dataset_path "../data/VLCB/raw" \
#     --test_dataset_name "test_PE" \
#     --output_dir "../results/PE" \
#     --gpu_ids "0,1" \
#     --tensor_parallel_size 2 \
#     --confidence_aggregation mean \
#     --resume