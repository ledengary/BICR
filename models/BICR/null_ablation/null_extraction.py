#!/usr/bin/env python3
"""
null_extraction.py — BICR null-image ablation extractor.

Clone of models/extraction/BICR.py with one — and only one — change: the
BLANK view dispatches on `--null_type` instead of always producing a solid
black image. base / paraphrase / image-noise / swap views are extracted
identically to v3 so the resulting npz is a drop-in for BICR_train with no
loader changes.

Output goes under data/null_ablation_extraction/{null_type}/{model_short}/{split}/samples/
parallel to (and never touching) data/extraction/BICR/.

Null types
----------
  white            — solid (255,255,255)
  gaussian_noise   — uniform-random pixels in [0,255]   (per blank_color_study.py pilot)
  blurred          — original image with GaussianBlur(radius=50)
  pixel_shuffled   — pixel-level permutation of the original image (preserves color
                     histogram, destroys all spatial structure)

Forward-pass config is locked to v3's:
  dtype=float32, attn_implementation='eager', max_image_dim=2048, layer_offsets=0
to ensure h_base/h_paraphrase/h_noise/h_swap are bit-identical to what v3
produced.

Usage
-----
  python models/BICR/null_ablation/null_extraction.py \\
      --null_type blurred \\
      --model_id Qwen/Qwen3-VL-8B-Instruct \\
      --gpu_ids 0 \\
      --dtype float32 \\
      --dataset_path data/VLCB/raw \\
      --target_datasets train validation test \\
      --train_dataset train \\
      --generation_extraction_dir data/extraction/raw \\
      --pe_dir data/PE \\
      --output_dir data/null_ablation_extraction \\
      --skip-if-processed
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
from PIL import Image, ImageFilter
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
FLOAT16_OVERFLOW_THRESHOLD = 60000.0

NULL_TYPES = ['white', 'gaussian_noise', 'blurred', 'pixel_shuffled']
DEFAULT_BLUR_RADIUS = 50  # matches blank_color_study.py pilot


# ============================================================================
# Safe dtype conversion (verbatim from v3)
# ============================================================================

_overflow_log_count = 0

def safe_to_storage(h: np.ndarray) -> Tuple[np.ndarray, str]:
    global _overflow_log_count
    h32 = h.astype(np.float32)
    n_overflow = int((np.abs(h32) > FLOAT16_OVERFLOW_THRESHOLD).sum())
    if n_overflow > 0:
        if _overflow_log_count < 10:
            logger.warning(
                f"[null_extract] {n_overflow} dims exceed float16 threshold "
                f"({FLOAT16_OVERFLOW_THRESHOLD}) -- saving as float32 "
                f"(max_abs={np.abs(h32).max():.1f})"
            )
            _overflow_log_count += 1
        elif _overflow_log_count == 10:
            logger.warning("[null_extract] (suppressing further float32 fallback warnings)")
            _overflow_log_count += 1
        return h32, "float32"
    return h32.astype(np.float16), "float16"


# ============================================================================
# Stable Hash Utility (verbatim from v3)
# ============================================================================

def stable_hash_to_uint32(s: str) -> int:
    h = hashlib.sha256(s.encode('utf-8')).digest()
    return int.from_bytes(h[:4], byteorder='big')


def get_deterministic_seed(hash_id: str, global_seed: int, salt: str = "") -> int:
    combined = f"{hash_id}_{salt}" if salt else hash_id
    sample_hash = stable_hash_to_uint32(combined)
    return sample_hash ^ global_seed


# ============================================================================
# Image preprocessing (verbatim from v3)
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
    """For the v3-equivalent NOISE view (additive on base) — unchanged."""
    img_array = np.array(pil_img).astype(np.float32) / 255.0
    rng = np.random.default_rng(seed=seed)
    noise = rng.normal(0, noise_std, img_array.shape).astype(np.float32)
    img_noised = np.clip(img_array + noise, 0.0, 1.0)
    return Image.fromarray((img_noised * 255).astype(np.uint8))


# ============================================================================
# NULL-VIEW IMAGE GENERATORS (the only new logic vs v3)
# ============================================================================

def make_null_image(
    null_type: str,
    reference_img: Image.Image,
    det_seed: int,
    blur_radius: int = DEFAULT_BLUR_RADIUS,
) -> Image.Image:
    """
    Generate the null-view image for the BICR blank slot.

    All null types are functions of the post-resize reference image (so blur
    radius and shuffle pixel count have predictable units across samples).

    Args:
      null_type: one of NULL_TYPES.
      reference_img: post-resize PIL RGB image (used for sizing for white/noise,
          used as the source for blurred/shuffled).
      det_seed: deterministic seed (from get_deterministic_seed) for the
          stochastic null types (gaussian_noise, pixel_shuffled).
      blur_radius: PIL GaussianBlur radius (px). Default matches the pilot.

    Returns:
      PIL Image (RGB) of the same size as reference_img.
    """
    w, h = reference_img.size

    if null_type == 'white':
        return Image.new('RGB', (w, h), color=(255, 255, 255))

    if null_type == 'gaussian_noise':
        rng = np.random.default_rng(seed=det_seed)
        arr = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
        return Image.fromarray(arr)

    if null_type == 'blurred':
        return reference_img.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    if null_type == 'pixel_shuffled':
        # Flatten H*W*3 → (H*W, 3); permute pixel rows; reshape back.
        arr = np.array(reference_img)
        rng = np.random.default_rng(seed=det_seed)
        flat = arr.reshape(-1, 3)
        perm = rng.permutation(flat.shape[0])
        shuffled = flat[perm].reshape(arr.shape)
        return Image.fromarray(shuffled.astype(np.uint8))

    raise ValueError(f"Unknown null_type: {null_type}. Must be one of {NULL_TYPES}.")


# ============================================================================
# Train Image Pool (verbatim from v3)
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
# CUDA / dataset / generation-extraction / paraphrase loaders (verbatim from v3)
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
# Model Loading (verbatim from v3)
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
# Layer index resolution (verbatim from v3)
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
# Dataset / input prep / forward / hidden extraction (verbatim from v3)
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


def run_forward_pass_full(model, inputs, model_type: str):
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


def get_pik_position(inputs, input_length: int, model_type: str) -> int:
    pik_pos = input_length - 1
    attn_mask = None
    if isinstance(inputs, dict):
        attn_mask = inputs.get('attention_mask')
    else:
        attn_mask = getattr(inputs, 'attention_mask', None)
    if attn_mask is not None:
        if attn_mask[0, pik_pos].item() == 0:
            nonzero = attn_mask[0].nonzero(as_tuple=False)
            if len(nonzero) > 0:
                fallback_pos = int(nonzero[-1].item())
                logger.warning(
                    f"[null_extract] PIK pos {pik_pos} is MASKED. "
                    f"Falling back to last unmasked pos {fallback_pos}."
                )
                pik_pos = fallback_pos
            else:
                logger.error(f"[null_extract] All positions masked! Using input_length-1={pik_pos}")
    return pik_pos


# ============================================================================
# Main extraction for a single sample (modified blank view, all else == v3)
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
        pik_position = get_pik_position(inputs, input_length, model_type)
        h_base = extract_hidden_at_position(hidden_states, pik_position, layer_indices)
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
        del inputs, hidden_states, outputs
        clear_cuda_cache()
    except Exception as e:
        logger.error(f"Error extracting base view for {hash_id}: {e}")
        return None

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

    # ===== VIEW 3: NOISE (additive on base — unchanged from v3) =====
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

    # ===== VIEW 4: SWAP (unchanged from v3) =====
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

    # ===== VIEW 5: BLANK (the only divergence from v3 — null_type dispatch) =====
    try:
        null_seed = get_deterministic_seed(hash_id, args.null_seed, salt=f"null_{args.null_type}")
        null_image = make_null_image(
            null_type=args.null_type,
            reference_img=image,
            det_seed=null_seed,
            blur_radius=args.blur_radius,
        )
        inputs, input_length = prepare_prompt_inputs(question, null_image, processor, model_type, model)
        outputs = run_forward_pass_full(model, inputs, model_type)
        pik_position = get_pik_position(inputs, input_length, model_type)
        h_blank = extract_hidden_at_position(outputs.hidden_states, pik_position, layer_indices)
        top1_k, _, p_base_k = logit_and_top1_at(outputs, input_length - 1, base_token=base_tok)
        if top1_k is not None:
            diag['top1_blank'] = top1_k
            diag['p_blank_of_base'] = p_base_k
        results['h_blank'] = h_blank
        results['mask_blank'] = 1
        results['null_seed_used'] = null_seed
        del inputs, outputs
        clear_cuda_cache()
    except Exception as e:
        logger.warning(f"Error extracting blank view for {hash_id} (null_type={args.null_type}): {e}")
        results['h_blank'] = np.zeros(concat_dim, dtype=np.float32)
        results['mask_blank'] = 0
        results['null_seed_used'] = 0

    # Per-view diagnostics
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

    results.update(diag)
    results['hidden_dim'] = hidden_dim
    results['concat_dim'] = concat_dim
    results['noise_std_used'] = args.noise_std
    results['null_type'] = args.null_type
    results['null_blur_radius'] = args.blur_radius if args.null_type == 'blurred' else None
    return results


# ============================================================================
# Saving
# ============================================================================

def save_sample(result: Dict, output_dir: str, args) -> str:
    samples_dir = os.path.join(output_dir, 'samples')
    os.makedirs(samples_dir, exist_ok=True)
    hash_id = result['hash_id']
    filepath = os.path.join(samples_dir, f"{hash_id}.npz")

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

        'h_base':       h_base_s,
        'h_paraphrase': h_para_s,
        'h_noise':      h_noise_s,
        'h_swap':       h_swap_s,
        'h_blank':      h_blank_s,

        'mask_base':       result['mask_base'],
        'mask_paraphrase': result['mask_paraphrase'],
        'mask_noise':      result['mask_noise'],
        'mask_swap':       result['mask_swap'],
        'mask_blank':      result['mask_blank'],

        'save_dtype_base':  dtype_base,
        'save_dtype_para':  dtype_para,
        'save_dtype_noise': dtype_noise,
        'save_dtype_swap':  dtype_swap,
        'save_dtype_blank': dtype_blank,

        'input_length':    result['input_length'],
        'hidden_dim':      result['hidden_dim'],
        'concat_dim':      result['concat_dim'],
        'layer_indices':   json.dumps(result['layer_indices']),
        'layer_offsets':   json.dumps(result['layer_offsets']),
        'num_layers':      result['num_layers'],
        'noise_seed_used': result['noise_seed_used'],
        'noise_std_used':  result['noise_std_used'],
        'null_type':       result['null_type'],
        'null_blur_radius': result.get('null_blur_radius') if result.get('null_blur_radius') is not None else -1,
        'null_seed_used':  result.get('null_seed_used', 0),
        'extraction_version': 'null_ablation_v1',

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
# Manifest / completed helpers
# ============================================================================

def create_manifest(output_dir: str, num_samples: int, config: Dict) -> str:
    manifest_data = {
        'extraction_method': 'BICR null-ablation extraction (v3-derived)',
        'version': 'null_ablation_v1',
        'parent_method': 'ICC v3',
        'null_type': config.get('null_type'),
        'null_params': {
            'blur_radius': config.get('blur_radius') if config.get('null_type') == 'blurred' else None,
            'null_seed': config.get('null_seed'),
        },
        'note': (
            'Identical to extraction/BICR.py except the BLANK view is generated '
            'by null_type-specific image transform. base/paraphrase/noise/swap '
            'views are bit-identical to what v3 would produce (same dtype, same '
            'attn_implementation, same processor).'
        ),
        'views': {
            'base':       'Original image + original question',
            'paraphrase': 'Original image + paraphrased question',
            'noise':      f'Gaussian sigma={config.get("noise_std", 0.10)} pixel noise + original question',
            'swap':       'Uniform-random train image + original question',
            'blank':      f'null_type={config.get("null_type")} (the ablation variable)',
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
# Argument parser
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description='BICR null-ablation extraction (v3-derived; only blank-view changes per --null_type)'
    )

    parser.add_argument('--null_type', type=str, required=True, choices=NULL_TYPES,
                        help='Which null-image strategy to use for the BLANK view.')
    parser.add_argument('--null_seed', type=int, default=42,
                        help='Global seed combined with hash_id for the null view (gaussian_noise + pixel_shuffled).')
    parser.add_argument('--blur_radius', type=int, default=DEFAULT_BLUR_RADIUS,
                        help='PIL GaussianBlur radius for null_type=blurred.')

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

    parser.add_argument('--generation_extraction_dir', type=str, required=True)
    parser.add_argument('--pe_dir', type=str, required=True)

    parser.add_argument('--noise_seed', type=int, default=42)
    parser.add_argument('--noise_std',  type=float, default=0.10)
    parser.add_argument('--layer_offsets', type=str, default='0')

    parser.add_argument('--max_samples',   type=int, default=None)
    parser.add_argument('--start_at_idx',  type=int, default=None)
    parser.add_argument('--end_at_idx',    type=int, default=None)
    parser.add_argument('--max_image_dim', type=int, default=MAX_IMAGE_DIMENSION)

    # Output: caller can pass either the parent dir (we'll add /{null_type} for them)
    # or a literal leaf via --output_dir_literal.
    parser.add_argument('--output_dir',  type=str, required=True,
                        help='Parent dir; null_type subdir is appended automatically.')
    parser.add_argument('--output_dir_literal', action='store_true',
                        help='Treat --output_dir as the literal leaf (do NOT append null_type).')
    parser.add_argument('--compression', type=str, default='compressed',
                        choices=['compressed', 'uncompressed'])

    parser.add_argument('--skip-if-processed', action='store_true')
    parser.add_argument('--debug',         action='store_true')
    parser.add_argument('--debug_samples', type=int, default=5)

    return parser.parse_args()


# ============================================================================
# Main
# ============================================================================

def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_ids = [args.model_id] if isinstance(args.model_id, str) else args.model_id
    layer_offsets = [int(x.strip()) for x in args.layer_offsets.split(',')]

    logger.info("=" * 80)
    logger.info("BICR null-ablation extraction")
    logger.info("=" * 80)
    logger.info(f"Null type:     {args.null_type}")
    if args.null_type == 'blurred':
        logger.info(f"  blur_radius: {args.blur_radius}")
    logger.info(f"Models:        {model_ids}")
    logger.info(f"Layer offsets: {layer_offsets}")
    logger.info(f"PE dir:        {args.pe_dir}")
    logger.info(f"Output parent: {args.output_dir} (literal={args.output_dir_literal})")

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
            if args.output_dir_literal:
                output_dir = os.path.join(args.output_dir, model_dir_name, target_dataset)
            else:
                output_dir = os.path.join(args.output_dir, args.null_type, model_dir_name, target_dataset)
            os.makedirs(output_dir, exist_ok=True)

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
                config['extraction_version'] = 'null_ablation_v1'
                json.dump(config, f, indent=2)

            num_saved = num_skipped = num_failed = 0
            iterator = (range(min(args.debug_samples, len(dataset))) if args.debug
                        else tqdm(range(len(dataset)),
                                  desc=f"{args.null_type[:5]} {target_dataset[:18]} [{start_idx}:{end_idx}]"))

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
                else:
                    clear_cuda_cache(aggressive=False)

            create_manifest(output_dir, num_saved, vars(args))
            logger.info(
                f"\nNull-ablation extraction complete for {target_dataset}: "
                f"saved={num_saved}, skipped={num_skipped}, failed={num_failed}"
            )
            clear_cuda_cache(aggressive=True)

        del model, processor
        clear_cuda_cache(aggressive=True)
        logger.info(f"Cleared model {model_id} from memory")

    logger.info("\nAll datasets processed!")


if __name__ == "__main__":
    main()
