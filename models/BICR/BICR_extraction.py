#!/usr/bin/env python3
"""
extraction/BICR.py — Interventional Contrastive Confidence Extraction (v3)

Changes vs v2 (all rooted in Phase-1 appendix ablations,
`manuscript/phase1_noise_view_ablation.md`):

  V3-CHANGE 1 — Paraphrase view
    Paraphrases now exist for train / val / test (Phase 0). No code change required —
    `load_paraphrase_from_pe()` now hits for every split because
    `data/PE/VLCB_{train,val,test}_raw_PE/paraphrases_json/{hash_id}.json` are populated.

  V3-CHANGE 2 — Noise view
    Gaussian σ lowered from 0.25 → 0.10 (Phase-1 §2 ablation, 11 transforms on 5k test).
    σ=0.25 gave near-zero cor/inc separation (dp_gap ≈ 0, flip_gap ≈ 0); σ=0.10 gives
    dp_gap = −0.090 and a 2.76× cor/inc flip ratio (7% vs 19%).

  V3-CHANGE 3 — Swap view
    No change — uniform-random from train pool. Label-opposition sampler tested in
    Phase-1 §3 and statistically indistinguishable from uniform (flip_gap differs by 0.006).

  V3-CHANGE 4 — Blank view
    Blank color changed from gray (128,128,128) → black (0,0,0) (Phase-1 §4). Lower
    residual VLM confidence (0.729 vs 0.732) and higher mode% (28.1% vs 26.7%).

  V3-CHANGE 5 — Per-sample diagnostics embedded in every npz
    Each sample saves: cos_base_{para,noise,swap,blank}, dp_{para,noise,swap,blank},
    flip_{para,noise,swap,blank}, p_base_top1, base_top1_token, p_{view}_of_base_tok,
    top1_{view}. These are cheap by-products of the forward passes we already do, and
    they feed directly into the paper's view-sanity panel without any re-extraction.

All v2 fixes (dtype auto, PIK attention-mask validation, layer_offsets) are preserved.

Usage:
  python models/extraction/BICR.py \\
      --model_id Qwen/Qwen3-VL-8B-Instruct \\
      --gpu_ids 0 \\
      --dtype float32 \\
      --dataset_path data/VLCB/raw \\
      --target_datasets train validation test \\
      --train_dataset train \\
      --generation_extraction_dir data/extraction/raw \\
      --pe_dir data/PE \\
      --output_dir data/extraction/BICR \\
      --skip-if-processed
  (Defaults: noise_std=0.10, blank_color=(0,0,0), layer_offsets=0.)
"""

# CRITICAL: Set CUDA_VISIBLE_DEVICES before importing torch
import os
import argparse

_parser = argparse.ArgumentParser(add_help=False)
_parser.add_argument('--gpu_ids', type=str, default='0')
_known_args, _ = _parser.parse_known_args()
os.environ["CUDA_VISIBLE_DEVICES"] = _known_args.gpu_ids

import gc
import json
import logging
import traceback
import io
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from datasets import load_from_disk

from transformers import AutoProcessor, AutoModelForCausalLM

try:
    from transformers import (
        Qwen3VLForConditionalGeneration,
        LlavaNextForConditionalGeneration,
        LlavaNextProcessor,
        AutoModelForImageTextToText,
    )
    from qwen_vl_utils import process_vision_info
except ImportError:
    pass

try:
    from transformers import Gemma3ForConditionalGeneration, Gemma3Processor
except ImportError:
    pass

try:
    from deepseek_vl2.models import DeepseekVLV2Processor, DeepseekVLV2ForCausalLM
    from deepseek_vl2.utils.io import load_pil_images
    DEEPSEEK_AVAILABLE = True
except ImportError:
    DEEPSEEK_AVAILABLE = False

import requests

# ============================================================================
# Logging
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# ============================================================================
# Constants
# ============================================================================

MAX_IMAGE_DIMENSION = 2048
SYSTEM_PROMPT = "You are a vision language assistant. Provide brief, complete answers."
APPENDED_SYSTEM_PROMPT = "Provide a brief, complete answer."

# FIX 2: threshold above which float32 is used instead of float16
FLOAT16_OVERFLOW_THRESHOLD = 60000.0

# ============================================================================
# FIX 2: Safe dtype conversion
# ============================================================================

_overflow_log_count = 0  # limit log spam

def safe_to_storage(h: np.ndarray) -> Tuple[np.ndarray, str]:
    """
    Convert float32 hidden state to storage dtype.

    FIX 2: v1 always used float16, causing inf values for Gemma
    (2 dims reach 84k-102k, exceeding float16 max of 65504).
    v2 auto-detects overflow and falls back to float32.

    Returns:
        (array_in_storage_dtype, dtype_name_str)
    """
    global _overflow_log_count
    h32 = h.astype(np.float32)

    # Check for overflow dims
    n_overflow = int((np.abs(h32) > FLOAT16_OVERFLOW_THRESHOLD).sum())

    if n_overflow > 0:
        if _overflow_log_count < 10:
            logger.warning(
                f"[ICCv3] {n_overflow} dims exceed float16 threshold "
                f"({FLOAT16_OVERFLOW_THRESHOLD}) — saving as float32 "
                f"(max_abs={np.abs(h32).max():.1f})"
            )
            _overflow_log_count += 1
        elif _overflow_log_count == 10:
            logger.warning("[ICCv3] (suppressing further float32 fallback warnings)")
            _overflow_log_count += 1
        return h32, "float32"
    else:
        return h32.astype(np.float16), "float16"


# ============================================================================
# Stable Hash Utility  (unchanged from v1)
# ============================================================================

def stable_hash_to_uint32(s: str) -> int:
    h = hashlib.sha256(s.encode('utf-8')).digest()
    return int.from_bytes(h[:4], byteorder='big')


def get_deterministic_seed(hash_id: str, global_seed: int, salt: str = "") -> int:
    combined = f"{hash_id}_{salt}" if salt else hash_id
    sample_hash = stable_hash_to_uint32(combined)
    return sample_hash ^ global_seed


# ============================================================================
# Image Preprocessing  (unchanged from v1)
# ============================================================================

def resize_image_if_needed(img, max_dim=MAX_IMAGE_DIMENSION) -> Image.Image:
    if isinstance(img, Image.Image):
        pil_img = img
    elif isinstance(img, bytes):
        pil_img = Image.open(io.BytesIO(img))
    elif isinstance(img, dict) and 'bytes' in img:
        pil_img = Image.open(io.BytesIO(img['bytes']))
    else:
        return img
    if pil_img.mode != 'RGB':
        pil_img = pil_img.convert('RGB')
    w, h = pil_img.size
    if max(w, h) <= max_dim:
        return pil_img
    scale = max_dim / max(w, h)
    return pil_img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)


def add_gaussian_noise_to_image(pil_img: Image.Image, noise_std: float, seed: int) -> Image.Image:
    img_array = np.array(pil_img).astype(np.float32) / 255.0
    rng = np.random.default_rng(seed=seed)
    noise = rng.normal(0, noise_std, img_array.shape).astype(np.float32)
    img_noised = np.clip(img_array + noise, 0.0, 1.0)
    return Image.fromarray((img_noised * 255).astype(np.uint8))


def create_blank_image(reference_img: Image.Image, color=(0, 0, 0)) -> Image.Image:
    """V3: default color changed from gray (128,) to black (0,) per Phase-1 §4 ablation."""
    w, h = reference_img.size
    return Image.new('RGB', (w, h), color=color)


# ============================================================================
# Train Image Pool  (unchanged from v1)
# ============================================================================

class TrainImagePool:
    def __init__(self, train_dataset, image_column: str):
        self.image_column = image_column
        self.images = []
        self.hash_ids = []
        logger.info("Building train image pool...")
        for idx, sample in enumerate(tqdm(train_dataset, desc="Indexing train images")):
            image = sample.get(image_column)
            hash_id = sample.get('hash_id', f'idx_{idx}')
            if image is not None:
                self.images.append(image)
                self.hash_ids.append(str(hash_id))
        logger.info(f"Built pool with {len(self.images)} train images")

    def get_deterministic_image(self, seed: int, exclude_hash_id: Optional[str] = None):
        if not self.images:
            return None
        rng = np.random.default_rng(seed=seed)
        if exclude_hash_id is not None:
            valid_indices = [i for i, hid in enumerate(self.hash_ids) if hid != exclude_hash_id]
            if not valid_indices:
                valid_indices = list(range(len(self.images)))
            idx = rng.choice(valid_indices)
        else:
            idx = rng.integers(0, len(self.images))
        return self.images[idx]


# ============================================================================
# CUDA Memory Management  (unchanged from v1)
# ============================================================================

def clear_cuda_cache(aggressive: bool = False):
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        if aggressive:
            torch.cuda.synchronize()
            try:
                torch.cuda.reset_peak_memory_stats()
            except Exception:
                pass


# ============================================================================
# Generation Extraction Loader  (unchanged from v1)
# ============================================================================

def load_generation_extraction_sample(gen_extraction_dir: str, model_dir_name: str,
                                       dataset_name: str, hash_id: str) -> Optional[Dict]:
    sample_path = os.path.join(gen_extraction_dir, model_dir_name, dataset_name,
                               'samples', f'{hash_id}.npz')
    if not os.path.exists(sample_path):
        return None
    try:
        data = np.load(sample_path, allow_pickle=True)
        gen_resp = data.get('generated_response', np.array(''))
        if isinstance(gen_resp, np.ndarray):
            generated_response = str(gen_resp.item() if gen_resp.shape == () else gen_resp)
        else:
            generated_response = str(gen_resp)
        is_corr = data.get('is_correct', None)
        if is_corr is None:
            is_correct = None
        elif isinstance(is_corr, np.ndarray):
            val = is_corr.item() if is_corr.shape == () else is_corr
            is_correct = bool(val) if val is not None else None
        else:
            is_correct = bool(is_corr) if is_corr is not None else None
        return {'generated_response': generated_response, 'is_correct': is_correct}
    except Exception as e:
        logger.warning(f"Error loading generation extraction for {hash_id}: {e}")
        return None


# ============================================================================
# PE Paraphrase Loader  (unchanged from v1)
# ============================================================================

_paraphrase_cache = {}

def load_paraphrase_from_pe(pe_dir: str, dataset_name: str, hash_id: str) -> Optional[str]:
    possible_paths = [
        os.path.join(pe_dir, f"{dataset_name}_PE", "paraphrases_json", f"{hash_id}.json"),
        os.path.join(pe_dir, dataset_name, "paraphrases_json", f"{hash_id}.json"),
    ]
    for json_path in possible_paths:
        if os.path.exists(json_path):
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                paraphrases = data.get('paraphrases', [])
                if paraphrases:
                    return paraphrases[0]
            except Exception as e:
                logger.warning(f"Error loading PE paraphrase for {hash_id}: {e}")

    cache_key = (pe_dir, dataset_name)
    if cache_key not in _paraphrase_cache:
        mapping_paths = [
            os.path.join(pe_dir, f"{dataset_name}.json"),
            os.path.join(pe_dir, f"{dataset_name}_paraphrases.json"),
            os.path.join(pe_dir, "paraphrases.json"),
        ]
        mapping = None
        for map_path in mapping_paths:
            if os.path.exists(map_path):
                try:
                    with open(map_path, 'r', encoding='utf-8') as f:
                        mapping = json.load(f)
                    logger.info(f"Loaded paraphrase mapping from {map_path}")
                    break
                except Exception as e:
                    logger.warning(f"Error loading paraphrase mapping from {map_path}: {e}")
        _paraphrase_cache[cache_key] = mapping

    mapping = _paraphrase_cache[cache_key]
    if mapping is not None and isinstance(mapping, dict) and hash_id in mapping:
        paraphrase_data = mapping[hash_id]
        if isinstance(paraphrase_data, str):
            return paraphrase_data
        elif isinstance(paraphrase_data, dict):
            paraphrases = paraphrase_data.get('paraphrases', [])
            if paraphrases:
                return paraphrases[0]
        elif isinstance(paraphrase_data, list) and paraphrase_data:
            return paraphrase_data[0]
    return None


# ============================================================================
# Model Loading  (unchanged from v1)
# ============================================================================

def load_model_and_processor(model_id: str, dtype_str: str, device: str):
    logger.info(f"Loading model: {model_id}")
    dtype_map = {'float32': torch.float32, 'float16': torch.float16, 'bfloat16': torch.bfloat16}
    dtype = dtype_map[dtype_str]
    model_id_lower = model_id.lower()

    if 'deepseek' in model_id_lower and 'vl' in model_id_lower:
        if not DEEPSEEK_AVAILABLE:
            raise ImportError("deepseek_vl2 not installed.")
        model_type = 'deepseek'
        if dtype_str != 'bfloat16':
            logger.warning(f"DeepSeek requires bfloat16, overriding {dtype_str}")
        dtype = torch.bfloat16
        processor = DeepseekVLV2Processor.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)
        model = model.to(dtype).cuda().eval()
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

    max_memory = {i: "130GiB" for i in range(torch.cuda.device_count())}
    max_memory[0] = "100GiB"

    model = model_class.from_pretrained(
        model_id, torch_dtype=dtype, device_map="auto",
        max_memory=max_memory, attn_implementation='eager', trust_remote_code=True,
    )
    model.eval()
    processor = processor_class.from_pretrained(model_id, **processor_kwargs)
    logger.info(f"Model loaded in {next(model.parameters()).dtype}")
    return model, processor, model_type


# ============================================================================
# Layer Index Computation  (unchanged from v1)
# ============================================================================

def get_layer_indices(model, model_type: str, layer_offsets: List[int]) -> Tuple[int, List[int]]:
    num_layers = None
    if model_type == 'deepseek':
        if hasattr(model, 'language') and hasattr(model.language, 'model') and hasattr(model.language.model, 'layers'):
            num_layers = len(model.language.model.layers)
        elif hasattr(model, 'config') and hasattr(model.config, 'num_hidden_layers'):
            num_layers = model.config.num_hidden_layers
    elif model_type == 'qwen':
        if hasattr(model, 'model') and hasattr(model.model, 'layers'):
            num_layers = len(model.model.layers)
        elif hasattr(model, 'config') and hasattr(model.config, 'num_hidden_layers'):
            num_layers = model.config.num_hidden_layers
        elif hasattr(model, 'language_model'):
            if hasattr(model.language_model, 'model') and hasattr(model.language_model.model, 'layers'):
                num_layers = len(model.language_model.model.layers)
            elif hasattr(model.language_model, 'config'):
                num_layers = model.language_model.config.num_hidden_layers
    elif model_type in ['llava', 'gemma', 'internvl']:
        if hasattr(model, 'language_model'):
            if hasattr(model.language_model, 'model') and hasattr(model.language_model.model, 'layers'):
                num_layers = len(model.language_model.model.layers)
            elif hasattr(model.language_model, 'config'):
                num_layers = model.language_model.config.num_hidden_layers

    if num_layers is None and hasattr(model, 'config'):
        for attr in ['text_config', 'llm_config', 'language_config']:
            if hasattr(model.config, attr):
                sub_config = getattr(model.config, attr)
                if hasattr(sub_config, 'num_hidden_layers'):
                    num_layers = sub_config.num_hidden_layers
                    break
    if num_layers is None and hasattr(model, 'config') and hasattr(model.config, 'num_hidden_layers'):
        num_layers = model.config.num_hidden_layers
    if num_layers is None:
        raise ValueError(f"Cannot determine number of layers for model type {model_type}")

    # FIX: do NOT normalize offset=0 to 1.
    # offset=0 → actual_index = num_layers → hidden_states[num_layers] = FINAL layer output
    # offset=1 → actual_index = num_layers - 1 → hidden_states[num_layers-1] = second-to-last
    # hidden_states has num_layers+1 elements: [embedding, layer0, ..., layer_{N-1}]
    # so hidden_states[num_layers] is always the last transformer layer output.
    max_offset = max(layer_offsets)
    if num_layers >= max_offset:
        actual_indices = [num_layers - offset for offset in layer_offsets]
    elif num_layers >= 2:
        logger.warning(f"Model has {num_layers} layers, falling back to last two layers")
        actual_indices = [num_layers, num_layers - 1]
    else:
        actual_indices = [num_layers]

    logger.info(f"Model has {num_layers} layers, using indices {actual_indices}")
    return num_layers, actual_indices


# ============================================================================
# Dataset Loading  (unchanged from v1)
# ============================================================================

def load_dataset(dataset_path: str):
    logger.info(f"Loading dataset from: {dataset_path}")
    if os.path.isdir(dataset_path):
        dataset = load_from_disk(dataset_path)
    else:
        from datasets import load_dataset as hf_load
        dataset = hf_load(dataset_path)
    logger.info(f"Loaded {len(dataset)} samples")
    return dataset


# ============================================================================
# Input Preparation  (unchanged from v1)
# ============================================================================

def prepare_prompt_inputs(question: str, image: Image.Image, processor, model_type: str, model):
    device = next(model.parameters()).device
    model_dtype = next(model.parameters()).dtype

    if model_type == 'qwen':
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question},
            ]},
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text], images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt",
        ).to(device)
        input_length = inputs.input_ids.shape[1]
        return inputs, input_length

    elif model_type == 'llava':
        conversation = [
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": question + "\n\n" + APPENDED_SYSTEM_PROMPT},
            ]},
        ]
        prompt = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        inputs = processor(images=image, text=prompt, return_tensors="pt")
        inputs = {k: v.to(device=device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(dtype=model_dtype)
        input_length = inputs["input_ids"].shape[1]
        return inputs, input_length

    elif model_type in ('gemma', 'internvl'):
        conversation = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]},
        ]
        prompt = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        inputs = processor(images=image, text=prompt, return_tensors="pt")
        inputs = {k: v.to(device=device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(dtype=model_dtype)
        input_length = inputs["input_ids"].shape[1]
        return inputs, input_length

    elif model_type == 'deepseek':
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            image.save(tmp.name)
            image_path = tmp.name
        try:
            conversation = [
                {"role": "<|User|>", "content": f"<image>\n{question}", "images": [image_path]},
                {"role": "<|Assistant|>", "content": ""},
            ]
            pil_images = load_pil_images(conversation)
            prepare_inputs = processor(
                conversations=conversation, images=pil_images,
                force_batchify=True, system_prompt=SYSTEM_PROMPT,
            ).to(model.device)
            if hasattr(prepare_inputs, 'pixel_values') and prepare_inputs.pixel_values is not None:
                prepare_inputs.pixel_values = prepare_inputs.pixel_values.to(dtype=model_dtype)
            input_length = prepare_inputs.input_ids.shape[1]
            return prepare_inputs, input_length
        finally:
            try:
                os.unlink(image_path)
            except Exception:
                pass
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


def run_forward_pass(model, inputs, model_type: str) -> Optional[torch.Tensor]:
    """Return hidden_states only (kept for backward compat with any non-v3 callers)."""
    out = run_forward_pass_full(model, inputs, model_type)
    return out.hidden_states if out is not None else None


def run_forward_pass_full(model, inputs, model_type: str):
    """V3: return the full model outputs so callers can grab both hidden_states and logits."""
    model.eval()
    with torch.no_grad():
        if model_type == 'deepseek':
            inputs_embeds = model.prepare_inputs_embeds(**inputs)
            outputs = model.language(
                inputs_embeds=inputs_embeds,
                attention_mask=inputs.attention_mask,
                output_hidden_states=True, return_dict=True,
            )
        else:
            outputs = model(**inputs, output_hidden_states=True, return_dict=True)
    return outputs


def logit_and_top1_at(outputs, position: int, base_token: Optional[int] = None):
    """Return (top1_token, p_top1, p_of_base_token) from outputs.logits[0, position, :].
       If outputs has no .logits (rare), returns (None, None, None)."""
    if not hasattr(outputs, 'logits') or outputs.logits is None:
        return None, None, None
    logits = outputs.logits[0, position]
    probs = torch.softmax(logits.float(), dim=-1)
    top1 = int(logits.argmax().item())
    p_top1 = float(probs[top1].item())
    p_base = float(probs[base_token].item()) if base_token is not None else p_top1
    return top1, p_top1, p_base


def extract_hidden_at_position(hidden_states, position: int, layer_indices: List[int]) -> np.ndarray:
    vectors = []
    for layer_idx in layer_indices:
        h = hidden_states[layer_idx][0, position, :].float().cpu().numpy()
        vectors.append(h)
    return np.concatenate(vectors, axis=0)


# ============================================================================
# FIX 3: Safe PIK position resolution with attention mask check
# ============================================================================

def get_pik_position(inputs, input_length: int, model_type: str) -> int:
    """
    FIX 3: Resolve PIK position with attention mask validation.

    v1: blindly used input_length - 1
    v2: verifies attention_mask[input_length-1] == 1 (not padding).
        Falls back to last non-masked token if PIK position is masked.

    Returns verified PIK position (int).
    """
    pik_pos = input_length - 1

    # Get attention mask (handle both dict and object inputs)
    attn_mask = None
    if isinstance(inputs, dict):
        attn_mask = inputs.get('attention_mask')
    else:
        attn_mask = getattr(inputs, 'attention_mask', None)

    if attn_mask is not None:
        if attn_mask[0, pik_pos].item() == 0:
            # PIK position is padding — find last unmasked token
            nonzero = attn_mask[0].nonzero(as_tuple=False)
            if len(nonzero) > 0:
                fallback_pos = int(nonzero[-1].item())
                logger.warning(
                    f"[ICCv3] PIK pos {pik_pos} is MASKED (attention_mask=0). "
                    f"Falling back to last unmasked pos {fallback_pos}."
                )
                pik_pos = fallback_pos
            else:
                logger.error(f"[ICCv3] All positions masked! Using input_length-1={pik_pos}")

    return pik_pos


# ============================================================================
# Main Extraction for Single Sample  (updated with fixes)
# ============================================================================

def extract_sample_views(
    sample: Dict,
    model,
    processor,
    model_type: str,
    layer_indices: List[int],
    layer_offsets: List[int],
    num_layers: int,
    train_image_pool: TrainImagePool,
    gen_extraction_dir: str,
    pe_dir: str,
    model_dir_name: str,
    dataset_name: str,
    args,
) -> Optional[Dict]:
    hash_id  = str(sample.get('hash_id', 'unknown'))
    question = str(sample.get(args.question_column, ''))
    answer   = str(sample.get(args.answer_column, ''))
    sample_id = str(sample.get(args.id_column, hash_id))
    image_raw = sample.get(args.image_column)

    if not question or image_raw is None:
        logger.warning(f"Sample {hash_id} missing question or image")
        return None

    image = resize_image_if_needed(image_raw, max_dim=args.max_image_dim)
    if image.mode != 'RGB':
        image = image.convert('RGB')

    gen_data = load_generation_extraction_sample(
        gen_extraction_dir, model_dir_name, dataset_name, hash_id)
    if gen_data is not None:
        is_correct = gen_data.get('is_correct')
        generated_response = gen_data.get('generated_response', '')
    else:
        is_correct = None
        generated_response = ''

    paraphrased_question = load_paraphrase_from_pe(pe_dir, dataset_name, hash_id)

    hidden_dim = None
    concat_dim = None

    results = {
        'hash_id': hash_id, 'sample_id': sample_id,
        'question': question, 'answer': answer,
        'generated_response': generated_response,
        'is_correct': is_correct,
        'paraphrased_question': paraphrased_question if paraphrased_question else '',
        'h_base': None, 'h_paraphrase': None, 'h_noise': None,
        'h_swap': None, 'h_blank': None,
        'mask_base': 0, 'mask_paraphrase': 0, 'mask_noise': 0,
        'mask_swap': 0, 'mask_blank': 0,
        'input_length': 0,
        'layer_indices': layer_indices, 'layer_offsets': layer_offsets, 'num_layers': num_layers,
        'save_dtype_base': 'float16', 'save_dtype_noise': 'float16',
        'save_dtype_swap': 'float16', 'save_dtype_blank': 'float16',
        'save_dtype_para': 'float16',
    }

    # V3: diagnostics scaffold — filled per-view; NaN sentinel for missing
    diag = {
        'p_base_top1': float('nan'), 'base_top1_token': -1,
        'p_para_of_base':  float('nan'), 'top1_para':  -1,
        'p_noise_of_base': float('nan'), 'top1_noise': -1,
        'p_swap_of_base':  float('nan'), 'top1_swap':  -1,
        'p_blank_of_base': float('nan'), 'top1_blank': -1,
    }
    base_tok = None

    # ===== VIEW 1: BASE =====
    try:
        inputs, input_length = prepare_prompt_inputs(question, image, processor, model_type, model)
        outputs = run_forward_pass_full(model, inputs, model_type)
        hidden_states = outputs.hidden_states

        # FIX 3: validated PIK position
        pik_position = get_pik_position(inputs, input_length, model_type)
        h_base = extract_hidden_at_position(hidden_states, pik_position, layer_indices)

        # V3: capture logit at first-answer position (input_length - 1)
        top1_b, p_top1_b, _ = logit_and_top1_at(outputs, input_length - 1, base_token=None)
        if top1_b is not None:
            base_tok = top1_b
            diag['base_top1_token'] = top1_b
            diag['p_base_top1'] = p_top1_b

        results['h_base'] = h_base
        results['mask_base'] = 1
        results['input_length'] = input_length
        hidden_dim = h_base.shape[0] // len(layer_indices) if len(layer_indices) > 1 else h_base.shape[0]
        concat_dim = h_base.shape[0]

        if args.debug:
            try:
                if isinstance(inputs, dict):
                    token_id = int(inputs['input_ids'][0, pik_position].item())
                else:
                    token_id = int(inputs.input_ids[0, pik_position].item())
                tokenizer = processor.tokenizer if hasattr(processor, 'tokenizer') else processor
                pik_tok = tokenizer.decode([token_id], skip_special_tokens=False).strip()
                logger.info(f"[ICCv3] {hash_id} PIK pos={pik_position} token='{pik_tok}'")
            except Exception:
                pass

        del inputs, hidden_states, outputs
        clear_cuda_cache()
    except Exception as e:
        logger.error(f"Error extracting base view for {hash_id}: {e}")
        return None  # base is required

    # ===== VIEW 2: PARAPHRASE =====
    if paraphrased_question:
        try:
            inputs, input_length = prepare_prompt_inputs(
                paraphrased_question, image, processor, model_type, model)
            outputs = run_forward_pass_full(model, inputs, model_type)
            pik_position = get_pik_position(inputs, input_length, model_type)
            h_para = extract_hidden_at_position(outputs.hidden_states, pik_position, layer_indices)
            top1_p, _, p_base_p = logit_and_top1_at(outputs, input_length - 1, base_token=base_tok)
            if top1_p is not None:
                diag['top1_para'] = top1_p
                diag['p_para_of_base'] = p_base_p
            results['h_paraphrase'] = h_para
            results['mask_paraphrase'] = 1
            del inputs, outputs
            clear_cuda_cache()
        except Exception as e:
            logger.warning(f"Error extracting paraphrase view for {hash_id}: {e}")
            results['h_paraphrase'] = np.zeros(concat_dim, dtype=np.float32)
            results['mask_paraphrase'] = 0
    else:
        results['h_paraphrase'] = np.zeros(concat_dim, dtype=np.float32)
        results['mask_paraphrase'] = 0

    # ===== VIEW 3: NOISE =====  V3: σ default 0.10
    try:
        noise_seed = get_deterministic_seed(hash_id, args.noise_seed, salt="noise")
        noised_image = add_gaussian_noise_to_image(image, args.noise_std, noise_seed)
        inputs, input_length = prepare_prompt_inputs(question, noised_image, processor, model_type, model)
        outputs = run_forward_pass_full(model, inputs, model_type)
        pik_position = get_pik_position(inputs, input_length, model_type)
        h_noise = extract_hidden_at_position(outputs.hidden_states, pik_position, layer_indices)
        top1_n, _, p_base_n = logit_and_top1_at(outputs, input_length - 1, base_token=base_tok)
        if top1_n is not None:
            diag['top1_noise'] = top1_n
            diag['p_noise_of_base'] = p_base_n
        results['h_noise'] = h_noise
        results['mask_noise'] = 1
        results['noise_seed_used'] = noise_seed
        del inputs, outputs
        clear_cuda_cache()
    except Exception as e:
        logger.warning(f"Error extracting noise view for {hash_id}: {e}")
        results['h_noise'] = np.zeros(concat_dim, dtype=np.float32)
        results['mask_noise'] = 0
        results['noise_seed_used'] = 0

    # ===== VIEW 4: SWAP =====  V3: uniform random kept (label-opposition rejected)
    try:
        swap_seed = get_deterministic_seed(hash_id, args.noise_seed, salt="swap")
        swap_image_raw = train_image_pool.get_deterministic_image(swap_seed, exclude_hash_id=hash_id)
        if swap_image_raw is not None:
            swap_image = resize_image_if_needed(swap_image_raw, max_dim=args.max_image_dim)
            if swap_image.mode != 'RGB':
                swap_image = swap_image.convert('RGB')
            inputs, input_length = prepare_prompt_inputs(question, swap_image, processor, model_type, model)
            outputs = run_forward_pass_full(model, inputs, model_type)
            pik_position = get_pik_position(inputs, input_length, model_type)
            h_swap = extract_hidden_at_position(outputs.hidden_states, pik_position, layer_indices)
            top1_s, _, p_base_s = logit_and_top1_at(outputs, input_length - 1, base_token=base_tok)
            if top1_s is not None:
                diag['top1_swap'] = top1_s
                diag['p_swap_of_base'] = p_base_s
            results['h_swap'] = h_swap
            results['mask_swap'] = 1
            del inputs, outputs
            clear_cuda_cache()
        else:
            results['h_swap'] = np.zeros(concat_dim, dtype=np.float32)
            results['mask_swap'] = 0
    except Exception as e:
        logger.warning(f"Error extracting swap view for {hash_id}: {e}")
        results['h_swap'] = np.zeros(concat_dim, dtype=np.float32)
        results['mask_swap'] = 0

    # ===== VIEW 5: BLANK =====  V3: color (0,0,0)
    try:
        blank_image = create_blank_image(image, color=tuple(args.blank_color))
        inputs, input_length = prepare_prompt_inputs(question, blank_image, processor, model_type, model)
        outputs = run_forward_pass_full(model, inputs, model_type)
        pik_position = get_pik_position(inputs, input_length, model_type)
        h_blank = extract_hidden_at_position(outputs.hidden_states, pik_position, layer_indices)
        top1_k, _, p_base_k = logit_and_top1_at(outputs, input_length - 1, base_token=base_tok)
        if top1_k is not None:
            diag['top1_blank'] = top1_k
            diag['p_blank_of_base'] = p_base_k
        results['h_blank'] = h_blank
        results['mask_blank'] = 1
        del inputs, outputs
        clear_cuda_cache()
    except Exception as e:
        logger.warning(f"Error extracting blank view for {hash_id}: {e}")
        results['h_blank'] = np.zeros(concat_dim, dtype=np.float32)
        results['mask_blank'] = 0

    # V3: compute per-view diagnostics from stored hidden states + logit info
    def _cos(a, b):
        a = a.astype(np.float32); b = b.astype(np.float32)
        n = (np.linalg.norm(a) * np.linalg.norm(b))
        return float(np.dot(a, b) / (n + 1e-12)) if n > 0 else float('nan')

    hb = results['h_base']
    pb = diag['p_base_top1']
    for view in ('paraphrase', 'noise', 'swap', 'blank'):
        short = {'paraphrase': 'para', 'noise': 'noise', 'swap': 'swap', 'blank': 'blank'}[view]
        hv = results[f'h_{view}'] if view != 'paraphrase' else results['h_paraphrase']
        mask = results[f'mask_{view}']
        if mask == 1:
            results[f'cos_base_{short}'] = _cos(hb, hv)
            p_under = diag[f'p_{short}_of_base']
            top1_v = diag[f'top1_{short}']
            results[f'dp_{short}'] = float(abs(pb - p_under)) if not (np.isnan(pb) or np.isnan(p_under)) else float('nan')
            results[f'flip_{short}'] = int(top1_v != diag['base_top1_token']) if top1_v >= 0 and diag['base_top1_token'] >= 0 else -1
        else:
            results[f'cos_base_{short}'] = float('nan')
            results[f'dp_{short}'] = float('nan')
            results[f'flip_{short}'] = -1

    # raw logit/top1 diagnostics (for appendix reconstruction)
    results.update(diag)

    results['hidden_dim'] = hidden_dim
    results['concat_dim'] = concat_dim
    results['noise_std_used'] = args.noise_std
    results['blank_color_used'] = str(tuple(args.blank_color))
    return results


# ============================================================================
# Saving  (FIX 2 applied here)
# ============================================================================

def save_sample(result: Dict, output_dir: str, args) -> str:
    samples_dir = os.path.join(output_dir, 'samples')
    os.makedirs(samples_dir, exist_ok=True)
    hash_id = result['hash_id']
    filepath = os.path.join(samples_dir, f"{hash_id}.npz")

    # FIX 2: auto dtype per view (float16 unless overflow → float32)
    h_base_s,  dtype_base  = safe_to_storage(result['h_base'])
    h_para_s,  dtype_para  = safe_to_storage(result['h_paraphrase'])
    h_noise_s, dtype_noise = safe_to_storage(result['h_noise'])
    h_swap_s,  dtype_swap  = safe_to_storage(result['h_swap'])
    h_blank_s, dtype_blank = safe_to_storage(result['h_blank'])

    save_dict = {
        'hash_id':    hash_id,
        'sample_id':  result['sample_id'],
        'question':   result['question'],
        'answer':     result['answer'],
        'generated_response':   result['generated_response'],
        'is_correct':           result['is_correct'],
        'paraphrased_question': result['paraphrased_question'],

        # View hidden states (dtype auto-selected per view)
        'h_base':       h_base_s,
        'h_paraphrase': h_para_s,
        'h_noise':      h_noise_s,
        'h_swap':       h_swap_s,
        'h_blank':      h_blank_s,

        # View masks
        'mask_base':       result['mask_base'],
        'mask_paraphrase': result['mask_paraphrase'],
        'mask_noise':      result['mask_noise'],
        'mask_swap':       result['mask_swap'],
        'mask_blank':      result['mask_blank'],

        # FIX 5: dtype metadata so loader knows what was used
        'save_dtype_base':  dtype_base,
        'save_dtype_para':  dtype_para,
        'save_dtype_noise': dtype_noise,
        'save_dtype_swap':  dtype_swap,
        'save_dtype_blank': dtype_blank,

        # Metadata
        'input_length':    result['input_length'],
        'hidden_dim':      result['hidden_dim'],
        'concat_dim':      result['concat_dim'],
        'layer_indices':   json.dumps(result['layer_indices']),
        'layer_offsets':   json.dumps(result['layer_offsets']),
        'num_layers':      result['num_layers'],
        'noise_seed_used': result['noise_seed_used'],
        'noise_std_used':  result['noise_std_used'],
        'blank_color_used': result['blank_color_used'],
        'extraction_version': 'v3',

        # V3 diagnostics — strict key access: any missing field is a bug, not data to mask
        'cos_base_para':   result['cos_base_para'],
        'cos_base_noise':  result['cos_base_noise'],
        'cos_base_swap':   result['cos_base_swap'],
        'cos_base_blank':  result['cos_base_blank'],
        'dp_para':         result['dp_para'],
        'dp_noise':        result['dp_noise'],
        'dp_swap':         result['dp_swap'],
        'dp_blank':        result['dp_blank'],
        'flip_para':       result['flip_para'],
        'flip_noise':      result['flip_noise'],
        'flip_swap':       result['flip_swap'],
        'flip_blank':      result['flip_blank'],
        'p_base_top1':     result['p_base_top1'],
        'base_top1_token': result['base_top1_token'],
        'p_para_of_base':  result['p_para_of_base'],
        'p_noise_of_base': result['p_noise_of_base'],
        'p_swap_of_base':  result['p_swap_of_base'],
        'p_blank_of_base': result['p_blank_of_base'],
        'top1_para':       result['top1_para'],
        'top1_noise':      result['top1_noise'],
        'top1_swap':       result['top1_swap'],
        'top1_blank':      result['top1_blank'],
    }

    save_fn = np.savez_compressed if args.compression == 'compressed' else np.savez
    save_fn(filepath, **save_dict)
    return filepath


# ============================================================================
# Manifest / Completed helpers  (unchanged from v1)
# ============================================================================

def create_manifest(output_dir: str, num_samples: int, config: Dict) -> str:
    manifest_data = {
        'extraction_method': 'ICC v3 (Interventional Contrastive Confidence — Phase-1-tuned)',
        'version': 'v3',
        'fixes_applied': [
            'V3: noise Gaussian σ lowered to 0.10 (was 0.25, near-zero calibration gap)',
            'V3: blank color black (0,0,0) (was gray 128)',
            'V3: paraphrase view now active for train/val/test (data backfilled in Phase 0)',
            'V3: per-sample diagnostics embedded (cos/dp/flip for every view)',
            'v2: auto float16/float32 dtype per view (prevents Gemma overflow)',
            'v2: PIK position validated against attention_mask',
            'v2: layer_offsets default = 0 (explicit last layer)',
        ],
        'views': {
            'base':       'Original image + original question',
            'paraphrase': 'Original image + paraphrased question (PE files for all splits)',
            'noise':      'Gaussian σ=0.10 pixel noise + original question',
            'swap':       'Uniform-random train image (deterministic seed) + original question',
            'blank':      'Black (0,0,0) blank image + original question',
        },
        'num_samples': num_samples,
        'samples_dir': 'samples/',
        'config': config,
        'layer_info': {
            'layer_offsets': config.get('layer_offsets', 'unknown'),
            'actual_layer_indices': config.get('actual_layer_indices', 'unknown'),
            'num_layers': config.get('num_layers', 'unknown'),
        },
        'created_at': datetime.now().isoformat(),
    }
    path = os.path.join(output_dir, 'manifest.json')
    with open(path, 'w') as f:
        json.dump(manifest_data, f, indent=2)
    logger.info(f"Created manifest: {path}")
    return path


def get_completed_samples(output_dir: str) -> set:
    samples_dir = os.path.join(output_dir, 'samples')
    if not os.path.exists(samples_dir):
        return set()
    completed = set()
    try:
        for fname in os.listdir(samples_dir):
            if fname.endswith('.npz') and os.path.isfile(os.path.join(samples_dir, fname)):
                completed.add(fname[:-4])
    except Exception as e:
        logger.warning(f"Error reading completed samples: {e}")
    return completed


# ============================================================================
# Argument Parser  (FIX 1 + FIX 4 defaults changed)
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description='ICCv3: Interventional Contrastive Confidence Extraction (v2)'
    )
    # Model
    parser.add_argument('--model_id', type=str, nargs='+',
                        default=['Qwen/Qwen3-VL-8B-Instruct'])
    parser.add_argument('--gpu_ids', type=str, default='0')
    parser.add_argument('--dtype', type=str, default='float32',
                        choices=['float32', 'float16', 'bfloat16'])

    # Data
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--target_datasets', type=str, nargs='+', required=True)
    parser.add_argument('--train_dataset', type=str, required=True)
    parser.add_argument('--image_column',    type=str, default='image')
    parser.add_argument('--question_column', type=str, default='question')
    parser.add_argument('--answer_column',   type=str, default='answer')
    parser.add_argument('--id_column',       type=str, default='id')

    # External data sources
    parser.add_argument('--generation_extraction_dir', type=str, required=True)
    parser.add_argument('--pe_dir', type=str, required=True,
                        help='Path to PE paraphrase data. test_raw_PE must exist for paraphrase view.')

    # ICC-specific  (V3: noise_std 0.25→0.10; blank color default black)
    parser.add_argument('--noise_seed', type=int, default=42)
    parser.add_argument('--noise_std',  type=float, default=0.10,
                        help='V3: Gaussian σ in PIL pixel space. Default 0.10 chosen via '
                             'Phase-1 §2 ablation over 11 transforms / 5k test samples.')
    parser.add_argument('--blank_color', type=int, nargs=3, default=[0, 0, 0],
                        help='V3: blank-image RGB color. Default (0,0,0) per Phase-1 §4. '
                             'Use "--blank_color 128 128 128" for gray.')
    parser.add_argument('--layer_offsets', type=str, default='0',
                        help='Layer offsets from last layer. "0"=FINAL layer (default). '
                             '"1"=second-to-last. "0,3" to concatenate final and L-3.')

    # Processing
    parser.add_argument('--max_samples',   type=int, default=None)
    parser.add_argument('--start_at_idx',  type=int, default=None)
    parser.add_argument('--end_at_idx',    type=int, default=None)
    parser.add_argument('--max_image_dim', type=int, default=MAX_IMAGE_DIMENSION)

    # Output
    parser.add_argument('--output_dir',  type=str, required=True)
    parser.add_argument('--compression', type=str, default='compressed',
                        choices=['compressed', 'uncompressed'])

    # Resume / Debug
    parser.add_argument('--skip-if-processed', action='store_true')
    parser.add_argument('--debug',         action='store_true')
    parser.add_argument('--debug_samples', type=int, default=5)

    return parser.parse_args()


# ============================================================================
# Main  (unchanged logic from v1, just using v2 functions)
# ============================================================================

def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_ids = [args.model_id] if isinstance(args.model_id, str) else args.model_id
    layer_offsets = [int(x.strip()) for x in args.layer_offsets.split(',')]

    logger.info("=" * 80)
    logger.info("ICCv3: Interventional Contrastive Confidence — Extraction (v2)")
    logger.info("=" * 80)
    logger.info(f"Models:        {model_ids}")
    logger.info(f"Layer offsets: {layer_offsets}")
    logger.info(f"Noise std:     {args.noise_std}  [FIX1: was 0.05]")
    logger.info(f"Auto dtype:    float16 unless max_abs > {FLOAT16_OVERFLOW_THRESHOLD} → float32  [FIX2]")
    logger.info(f"PIK position:  attention-mask validated  [FIX3]")
    logger.info(f"PE dir:        {args.pe_dir}")

    for model_id in model_ids:
        model_dir_name = model_id.split("/")[-1]
        logger.info(f"\n{'#'*80}\nProcessing model: {model_id}\n{'#'*80}")

        try:
            model, processor, model_type = load_model_and_processor(model_id, args.dtype, device)
            num_layers, actual_layer_indices = get_layer_indices(model, model_type, layer_offsets)
        except Exception as e:
            logger.error(f"Failed to load model {model_id}: {e}")
            logger.error(traceback.format_exc())
            continue

        logger.info(f"Loading train dataset for swap pool: {args.train_dataset}")
        train_dataset_path = os.path.join(args.dataset_path, args.train_dataset)
        try:
            train_dataset = load_dataset(train_dataset_path)
            train_image_pool = TrainImagePool(train_dataset, args.image_column)
        except Exception as e:
            logger.error(f"Failed to load train dataset: {e}")
            continue

        for target_dataset in args.target_datasets:
            logger.info(f"\n{'='*80}\nDataset: {target_dataset}\n{'='*80}")

            full_dataset_path = os.path.join(args.dataset_path, target_dataset)
            output_dir = os.path.join(args.output_dir, model_dir_name, target_dataset)
            os.makedirs(output_dir, exist_ok=True)

            # Check PE coverage for this split
            pe_split_dir = os.path.join(args.pe_dir, f"{target_dataset}_PE", "paraphrases_json")
            if os.path.isdir(pe_split_dir):
                n_pe = len(list(Path(pe_split_dir).glob("*.json")))
                logger.info(f"PE paraphrases available for {target_dataset}: {n_pe} files")
            else:
                logger.info(f"No PE data for {target_dataset} — paraphrase view will be mask=0")

            completed_hash_ids = set()
            if args.skip_if_processed:
                completed_hash_ids = get_completed_samples(output_dir)
                logger.info(f"Found {len(completed_hash_ids)} already-completed samples")

            try:
                dataset = load_dataset(full_dataset_path)
            except Exception as e:
                logger.error(f"Failed to load dataset {full_dataset_path}: {e}")
                continue

            start_idx = max(0, args.start_at_idx) if args.start_at_idx is not None else 0
            end_idx   = min(args.end_at_idx, len(dataset)) if args.end_at_idx is not None else len(dataset)
            if start_idx > 0 or end_idx < len(dataset):
                dataset = dataset.select(range(start_idx, end_idx))
                logger.info(f"Sliced dataset to indices [{start_idx}, {end_idx})")
            if args.max_samples is not None:
                dataset = dataset.select(range(min(args.max_samples, len(dataset))))
                logger.info(f"Limited to {len(dataset)} samples")

            with open(os.path.join(output_dir, 'config.json'), 'w') as f:
                config = vars(args).copy()
                config['actual_layer_indices'] = actual_layer_indices
                config['num_layers'] = num_layers
                config['extraction_version'] = 'v3'
                json.dump(config, f, indent=2)

            num_saved = num_skipped = num_failed = 0
            iterator = (range(min(args.debug_samples, len(dataset))) if args.debug
                        else tqdm(range(len(dataset)), desc=f"ICCv3 [{target_dataset}]"))

            for sample_idx in iterator:
                sample = dataset[sample_idx]
                hash_id = str(sample.get('hash_id', f'sample_{sample_idx}')).strip()

                if args.skip_if_processed and hash_id in completed_hash_ids:
                    num_skipped += 1
                    continue

                try:
                    result = extract_sample_views(
                        sample=sample, model=model, processor=processor,
                        model_type=model_type, layer_indices=actual_layer_indices,
                        layer_offsets=layer_offsets, num_layers=num_layers,
                        train_image_pool=train_image_pool,
                        gen_extraction_dir=args.generation_extraction_dir,
                        pe_dir=args.pe_dir, model_dir_name=model_dir_name,
                        dataset_name=target_dataset, args=args,
                    )
                    if result is not None:
                        save_sample(result, output_dir, args)
                        num_saved += 1
                        if args.debug:
                            logger.info(
                                f"  [{sample_idx}] {hash_id}: "
                                f"base={result['mask_base']}, para={result['mask_paraphrase']}, "
                                f"noise={result['mask_noise']}, swap={result['mask_swap']}, "
                                f"blank={result['mask_blank']}, is_correct={result['is_correct']}"
                            )
                    else:
                        num_failed += 1

                except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
                    num_failed += 1
                    logger.error(f"CUDA error on sample {sample_idx}: {e}")
                    for _ in range(5):
                        clear_cuda_cache(aggressive=True)
                    gc.collect()
                except Exception as e:
                    num_failed += 1
                    logger.error(f"Error on sample {sample_idx}: {e}")
                    if args.debug:
                        logger.error(traceback.format_exc())
                    clear_cuda_cache()

                if num_saved % 50 == 0 and num_saved > 0:
                    clear_cuda_cache(aggressive=True)
                    logger.info(f"Processed {num_saved} samples so far")
                else:
                    clear_cuda_cache(aggressive=False)

            create_manifest(output_dir, num_saved, vars(args))
            logger.info(
                f"\nICCv3 extraction complete for {target_dataset}: "
                f"saved={num_saved}, skipped={num_skipped}, failed={num_failed}"
            )
            clear_cuda_cache(aggressive=True)

        del model, processor
        clear_cuda_cache(aggressive=True)
        logger.info(f"Cleared model {model_id} from memory")

    logger.info("\nAll models and datasets processed!")


if __name__ == "__main__":
    main()

# ============================================================================
# Example commands
# ============================================================================
#
# Full BICR extraction — all splits, all models:
#
# conda run -n vlmce_vllm python models/BICR/BICR_extraction.py \
#     --model_id Qwen/Qwen3-VL-8B-Instruct \
#     --gpu_ids 0 --dtype float32 \
#     --dataset_path data/vlcb \
#     --target_datasets train validation test \
#     --generation_extraction_dir data/extraction/raw \
#     --output_dir data/extraction/BICR
#
# DeepSeek (use dsvl env):
# conda run -n dsvl python models/BICR/BICR_extraction.py \
#     --model_id deepseek-ai/deepseek-vl2 --gpu_ids 4 ...
