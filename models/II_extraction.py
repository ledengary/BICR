"""
InternalInspector (I²) Extraction Script v2
=============================================
Extracts per-layer internal states for the InternalInspector confidence estimation method.

KEY DESIGN DECISION (vs v1):
  Instead of re-running generation and fighting KV-cache/hook-timing issues,
  this script loads the already-generated response from prior generate_and_extract.py
  .npz files and performs a single TEACHER-FORCED forward pass:

      input = [prompt tokens] + [generated response tokens]
      model(**inputs, output_hidden_states=True)   ← single forward pass, no generate()

  This gives us a clean, unambiguous single forward pass where:
    - Hooks fire exactly ONCE per layer
    - The full sequence (prompt + response) is visible at every layer
    - We extract states at last_input_token_pos = input_len - 1
    - No KV-cache complications whatsoever

  This is maximally faithful to what the paper describes: internal states at the
  final input token across all layers, during the forward pass that produces the
  prediction.

For each sample, extracts at the LAST INPUT TOKEN position across ALL transformer layers:

  - Activation states  (h): post-residual hidden state at each layer         [L, d]
  - Attention states   (a): MHSA sublayer output BEFORE residual addition     [L, d]
  - Feed-forward states(m): FFN  sublayer output BEFORE residual addition     [L, d]

These are stored SEPARATELY so any combination can be used at training time
without re-running extraction.

Output per sample: {hash_id}.npz
  Keys:
    hash_id, sample_id, question, answer, generated_response, is_correct
    activation_states   : float16  [L, d]
    attention_states    : float16  [L, d]
    ff_states           : float16  [L, d]
    num_layers          : int
    hidden_dim          : int
    boundaries          : JSON string
    input_length        : int
    num_generated_tokens: int
    token_ids           : int32 [T]
    token_strs          : object [T]
"""

# ── early CUDA_VISIBLE_DEVICES ──────────────────────────────────────────────
import os
import argparse

_pre_parser = argparse.ArgumentParser(add_help=False)
_pre_parser.add_argument("--gpu_ids", type=str, default="0")
_pre_known, _ = _pre_parser.parse_known_args()
os.environ["CUDA_VISIBLE_DEVICES"] = _pre_known.gpu_ids

# ── standard imports ─────────────────────────────────────────────────────────
import gc
import json
import logging
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

try:
    from PIL import Image
except ImportError:
    Image = None

try:
    from transformers import (
        AutoModelForCausalLM,
        AutoModelForImageTextToText,
        AutoProcessor,
        LlavaNextForConditionalGeneration,
        LlavaNextProcessor,
        Qwen3VLForConditionalGeneration,
    )
    from qwen_vl_utils import process_vision_info
except ImportError:
    print("[WARNING] Some transformers imports failed – check your environment.")

try:
    from deepseek_vl2.models import DeepseekVLV2ForCausalLM, DeepseekVLV2Processor
    from deepseek_vl2.utils.io import load_pil_images
    DEEPSEEK_AVAILABLE = True
except ImportError:
    DEEPSEEK_AVAILABLE = False

import io
from datasets import load_from_disk

# ── logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)
_file_handlers: Dict[str, logging.FileHandler] = {}

# ── constants (must mirror generate_and_extract.py exactly) ──────────────────
SYSTEM_PROMPT = "You are a vision language assistant. Provide brief, complete answers."
APPENDED_SYSTEM_PROMPT = "Provide a brief, complete answer."
MAX_IMAGE_DIMENSION = 2048


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  Logging helpers                                                          ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def _setup_model_logging(model_id: str) -> None:
    model_name = model_id.split("/")[-1]
    os.makedirs("logs", exist_ok=True)
    fh = logging.FileHandler(f"logs/II_extraction_v2_{model_name}.log")
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    logger.addHandler(fh)
    _file_handlers[model_id] = fh


def _remove_model_logging(model_id: str) -> None:
    if model_id in _file_handlers:
        logger.removeHandler(_file_handlers[model_id])
        _file_handlers[model_id].close()
        del _file_handlers[model_id]


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  Argument parser                                                          ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Extract per-layer internal states for InternalInspector (I²) v2 "
                    "– teacher-forced, no generation, no KV-cache issues."
    )

    # ── model ────────────────────────────────────────────────────────────────
    p.add_argument("--model_id", type=str, nargs="+",
                   default=["Qwen/Qwen3-VL-8B-Instruct"])
    p.add_argument("--gpu_ids", type=str, default="0")
    p.add_argument("--dtype", type=str, default="float32",
                   choices=["float32", "float16", "bfloat16"])

    # ── data ─────────────────────────────────────────────────────────────────
    p.add_argument("--dataset_path", type=str, required=True,
                   help="Base path to the raw VLCB datasets directory")
    p.add_argument("--target_datasets", type=str, nargs="+", required=True)
    p.add_argument("--image_column",    type=str, default="image")
    p.add_argument("--question_column", type=str, default="question")
    p.add_argument("--answer_column",   type=str, default="answer")
    p.add_argument("--id_column",       type=str, default="id")

    # ── prior extraction ─────────────────────────────────────────────────────
    p.add_argument("--prior_extraction_dir", type=str, required=True,
                   help="Root of the generate_and_extract output "
                        "({model_name}/{dataset}/samples/*.npz). "
                        "Provides generated_response and is_correct for each sample.")

    # ── extraction ───────────────────────────────────────────────────────────
    p.add_argument("--max_samples",    type=int,   default=None)
    p.add_argument("--start_at_idx",   type=int,   default=None)
    p.add_argument("--end_at_idx",     type=int,   default=None)
    p.add_argument("--max_image_dim",  type=int,   default=MAX_IMAGE_DIMENSION)

    # ── output ───────────────────────────────────────────────────────────────
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--compression", type=str, default="compressed",
                   choices=["compressed", "uncompressed"])

    # ── misc ─────────────────────────────────────────────────────────────────
    p.add_argument("--skip_if_processed", action="store_true")
    p.add_argument("--debug",             action="store_true")
    p.add_argument("--debug_samples",     type=int, default=5)

    return p.parse_args()


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  Image utility                                                            ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def resize_image_if_needed(img, max_dim: int = MAX_IMAGE_DIMENSION):
    if isinstance(img, Image.Image):
        pil_img = img
    elif isinstance(img, bytes):
        pil_img = Image.open(io.BytesIO(img))
    elif isinstance(img, dict) and "bytes" in img:
        pil_img = Image.open(io.BytesIO(img["bytes"]))
    else:
        return img

    w, h = pil_img.size
    if max(w, h) <= max_dim:
        return pil_img

    scale = max_dim / max(w, h)
    pil_img = pil_img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    return pil_img


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  Token boundary detection  (copied verbatim from generate_and_extract)   ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def detect_token_boundaries(input_ids: List[int], processor, model_type: str) -> Dict:
    decoded_tokens = [processor.tokenizer.decode([t]) for t in input_ids]

    vision_start_idx = vision_end_idx = None

    if model_type == "qwen":
        for i, tok in enumerate(decoded_tokens):
            if "<|vision_start|>" in tok or "<|image_pad|>" in tok:
                if vision_start_idx is None:
                    vision_start_idx = i
            elif vision_start_idx is not None and "<|image_pad|>" not in tok:
                vision_end_idx = i
                break

    elif model_type == "llava":
        image_token_id = processor.tokenizer.image_token_id
        for i, tid in enumerate(input_ids):
            if tid == image_token_id:
                if vision_start_idx is None:
                    vision_start_idx = i
                vision_end_idx = i + 1

    elif model_type == "gemma":
        for i, tid in enumerate(input_ids):
            if tid == 255999:
                vision_start_idx = i
                break
        if vision_start_idx is not None:
            for i in range(vision_start_idx + 1, len(input_ids)):
                if input_ids[i] == 256000:
                    vision_end_idx = i + 1
                    break

    elif model_type == "internvl":
        for i, tid in enumerate(input_ids):
            if tid == 151669:
                vision_start_idx = i
                break
        if vision_start_idx is not None:
            for i in range(vision_start_idx + 1, len(input_ids)):
                if input_ids[i] == 151670:
                    vision_end_idx = i + 1
                    break

    elif model_type == "deepseek":
        try:
            image_token_id = processor.tokenizer.convert_tokens_to_ids("<image>")
        except Exception:
            image_token_id = 128815
        for i, tid in enumerate(input_ids):
            if tid == image_token_id:
                if vision_start_idx is None:
                    vision_start_idx = i
                vision_end_idx = i + 1

    if vision_start_idx is None:
        vision_start_idx = vision_end_idx = 0

    return {
        "vision_start": vision_start_idx,
        "vision_end":   vision_end_idx,
        "text_start":   vision_end_idx,
        "text_end":     len(input_ids),
        "num_vision":   vision_end_idx - vision_start_idx,
        "num_text":     len(input_ids) - vision_end_idx,
        "num_other":    vision_start_idx,
    }


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  Model loading                                                            ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def load_model_and_processor(model_id: str, dtype_str: str, device: str):
    logger.info(f"Loading model: {model_id}  dtype={dtype_str}  device={device}")

    dtype_map = {"float32": torch.float32, "float16": torch.float16,
                 "bfloat16": torch.bfloat16}
    dtype = dtype_map[dtype_str]
    model_id_lower = model_id.lower()

    if "deepseek" in model_id_lower and "vl" in model_id_lower:
        if not DEEPSEEK_AVAILABLE:
            raise ImportError("deepseek_vl2 not installed.")
        model_type = "deepseek"
        if dtype_str != "bfloat16":
            logger.warning("Overriding dtype to bfloat16 for DeepSeek VL2.")
            dtype = torch.bfloat16
        processor = DeepseekVLV2Processor.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)
        model = model.to(dtype).cuda().eval()

    elif "llava" in model_id_lower:
        model_type = "llava"
        processor = LlavaNextProcessor.from_pretrained(model_id)
        model = LlavaNextForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=dtype, device_map="auto",
            attn_implementation="eager", trust_remote_code=True)
        model.eval()

    elif "qwen" in model_id_lower:
        model_type = "qwen"
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=dtype, device_map="auto",
            attn_implementation="eager", trust_remote_code=True)
        model.eval()

    elif "gemma" in model_id_lower:
        from transformers import Gemma3ForConditionalGeneration, Gemma3Processor
        model_type = "gemma"
        processor = Gemma3Processor.from_pretrained(model_id)
        model = Gemma3ForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=dtype, device_map="auto",
            attn_implementation="eager", trust_remote_code=True)
        model.eval()

    elif "internvl" in model_id_lower:
        model_type = "internvl"
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModelForImageTextToText.from_pretrained(
            model_id, torch_dtype=dtype, device_map="auto",
            attn_implementation="eager", trust_remote_code=True)
        model.eval()

    else:
        raise ValueError(f"Unknown model type for: {model_id}")

    logger.info(f"Model type: {model_type}  |  actual dtype: {next(model.parameters()).dtype}")
    return model, processor, model_type


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  Forward-hook registration                                               ║
# ║                                                                          ║
# ║  Registers hooks to capture, for each transformer layer:                ║
# ║    - attention output  (MHSA out, BEFORE residual addition)             ║
# ║    - feed-forward out  (FFN out,  BEFORE residual addition)             ║
# ║                                                                          ║
# ║  Because we do a single teacher-forced forward pass (no generation),    ║
# ║  each hook fires EXACTLY ONCE. No guard logic needed.                   ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def _get_transformer_layers(model, model_type: str):
    mt = model_type.lower()
    try:
        if mt == "qwen":
            return list(model.model.language_model.layers)
        elif mt == "llava":
            return list(model.model.language_model.layers)
        elif mt == "gemma":
            return list(model.model.language_model.layers)
        elif mt == "internvl":
            return list(model.model.language_model.layers)
        elif mt == "deepseek":
            return list(model.language.model.layers)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
    except AttributeError as e:
        raise RuntimeError(
            f"Could not locate transformer layers for model_type='{model_type}'. "
            f"AttributeError: {e}."
        )


def _get_attn_and_mlp_modules(layer, model_type: str):
    mt = model_type.lower()
    try:
        if mt in ("qwen", "llava", "gemma", "deepseek"):
            return layer.self_attn, layer.mlp
        elif mt == "internvl":
            attn = getattr(layer, "self_attn", None) or getattr(layer, "attention", None)
            if attn is None:
                raise AttributeError("Cannot find attention sub-module.")
            return attn, layer.mlp
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
    except AttributeError as e:
        raise RuntimeError(
            f"Could not locate attn/mlp sub-modules for model_type='{model_type}'. "
            f"AttributeError: {e}."
        )


class _InternalStateHooks:
    """
    Registers forward hooks on all transformer layers to capture:
      - attention output  (first tensor returned by self_attn)
      - FFN output        (first tensor returned by mlp)

    Because we run a SINGLE teacher-forced forward pass, each hook fires
    exactly once per layer — no guard logic, no KV-cache complications.

    Usage:
        hooks = _InternalStateHooks(model, model_type)
        hooks.register()
        # ... run single forward pass ...
        attn_states, ff_states = hooks.collect(last_input_token_pos)
        hooks.reset()      # clear for next sample
        hooks.remove()     # remove all hooks at end of model lifetime
    """

    def __init__(self, model, model_type: str):
        self.model_type = model_type
        self.layers     = _get_transformer_layers(model, model_type)
        self.num_layers = len(self.layers)

        self._attn_out: List[Optional[torch.Tensor]] = [None] * self.num_layers
        self._ff_out:   List[Optional[torch.Tensor]] = [None] * self.num_layers
        self._handles = []

    # ── hook factories ────────────────────────────────────────────────────

    def _make_attn_hook(self, layer_idx: int):
        def hook(module, input, output):
            if isinstance(output, tuple):
                out = output[0]
            else:
                out = output
            # detach, move to CPU, cast to float32 (bfloat16 → float32 for numpy compat)
            self._attn_out[layer_idx] = out.detach().float().cpu()
        return hook

    def _make_ff_hook(self, layer_idx: int):
        def hook(module, input, output):
            if isinstance(output, tuple):
                out = output[0]
            else:
                out = output
            self._ff_out[layer_idx] = out.detach().float().cpu()
        return hook

    # ── register / remove ────────────────────────────────────────────────

    def register(self):
        for i, layer in enumerate(self.layers):
            attn_mod, mlp_mod = _get_attn_and_mlp_modules(layer, self.model_type)
            h_a = attn_mod.register_forward_hook(self._make_attn_hook(i))
            h_m = mlp_mod.register_forward_hook(self._make_ff_hook(i))
            self._handles.extend([h_a, h_m])
        logger.debug(f"Registered {2 * self.num_layers} forward hooks.")

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def reset(self):
        self._attn_out = [None] * self.num_layers
        self._ff_out   = [None] * self.num_layers

    def collect(self, last_input_token_pos: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract the representation at `last_input_token_pos` from each layer.

        Because this is a teacher-forced pass over [prompt + response], the
        sequence length S = input_len + response_len.  We want position
        (input_len - 1), i.e. the last prompt token, which is what
        last_input_token_pos is set to by the caller.

        Returns:
            attn_states : float32 numpy  [L, d]
            ff_states   : float32 numpy  [L, d]
        """
        attn_list = []
        ff_list   = []

        for i in range(self.num_layers):
            a = self._attn_out[i]
            m = self._ff_out[i]

            if a is None or m is None:
                raise RuntimeError(
                    f"Hook did not fire for layer {i}. "
                    "Ensure attn_implementation='eager' when loading the model."
                )

            # Shape: [B, S, d] — squeeze batch dim
            a = a.squeeze(0)   # [S, d]
            m = m.squeeze(0)   # [S, d]

            pos = min(last_input_token_pos, a.shape[0] - 1)

            attn_list.append(a[pos].numpy())   # [d]
            ff_list.append(m[pos].numpy())     # [d]

        attn_arr = np.stack(attn_list, axis=0).astype(np.float32)   # [L, d]
        ff_arr   = np.stack(ff_list,   axis=0).astype(np.float32)   # [L, d]

        return attn_arr, ff_arr


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  Activation states from output.hidden_states                             ║
# ║                                                                          ║
# ║  model(**inputs, output_hidden_states=True) returns:                    ║
# ║    output.hidden_states : tuple[L+1] of [B, S, d]                      ║
# ║      index 0   = embedding layer                                        ║
# ║      index 1..L = transformer layers                                    ║
# ║                                                                          ║
# ║  S = input_len + response_len (full teacher-forced sequence).           ║
# ║  We extract at last_input_token_pos = input_len - 1.                    ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def collect_activation_states(output, last_input_token_pos: int) -> np.ndarray:
    """
    Extract post-residual hidden states (h^l_N) at last_input_token_pos
    from all transformer layers.

    Skips index 0 (embedding) to align with hook layer indexing.

    Returns:
        activation_states : float32 numpy  [L, d]
    """
    act_list = []

    for layer_hs in output.hidden_states[1:]:   # each: [B, S, d]
        hs  = layer_hs.squeeze(0)               # [S, d]
        pos = min(last_input_token_pos, hs.shape[0] - 1)
        vec = hs[pos].detach().float().cpu().numpy()   # [d]
        act_list.append(vec)

    return np.stack(act_list, axis=0).astype(np.float32)   # [L, d]


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  Prior extraction loader                                                 ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def load_prior_extraction_index(prior_dir: Path) -> Dict[str, Dict]:
    """
    Walk all .npz files under prior_dir and build a dict:
        hash_id  ->  {
            'generated_response': str,
            'is_correct':         bool or None,
            'token_ids':          list[int],
        }
    """
    index: Dict[str, Dict] = {}
    npz_files = list(prior_dir.rglob("*.npz"))
    logger.info(f"Building prior extraction index from {len(npz_files)} files …")

    for f in tqdm(npz_files, desc="Indexing prior extractions", leave=False):
        try:
            data  = np.load(f, allow_pickle=True)
            hid   = str(data["hash_id"])
            resp  = str(data.get("generated_response", ""))
            ic    = data.get("is_correct", None)
            tids  = data.get("token_ids", np.array([], dtype=np.int32))

            is_correct = None
            if ic is not None:
                try:
                    val = ic.item() if isinstance(ic, np.ndarray) else ic
                    is_correct = bool(val) if val is not None else None
                except Exception:
                    is_correct = None

            index[hid] = {
                "generated_response": resp,
                "is_correct":         is_correct,
                "token_ids":          tids.tolist() if isinstance(tids, np.ndarray) else list(tids),
            }
        except Exception:
            pass

    logger.info(f"Prior extraction index built: {len(index)} entries.")
    return index


def get_prior_samples_dir(prior_extraction_dir: str, model_name_part: str,
                          target_dataset: str) -> Optional[Path]:
    p = Path(prior_extraction_dir) / model_name_part / target_dataset / "samples"
    if p.exists():
        return p
    alt = Path(prior_extraction_dir) / model_name_part / target_dataset
    if alt.exists():
        return alt
    logger.warning(f"Prior extraction samples dir not found: {p}")
    return None


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  Build teacher-forced inputs                                              ║
# ║                                                                          ║
# ║  Appends the tokenised generated_response to the prompt tokens so        ║
# ║  that a single model(**inputs) call sees the full sequence.              ║
# ║                                                                          ║
# ║  We do NOT call model.generate() here at all.                           ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def build_teacher_forced_inputs(
    sample: Dict,
    model,
    processor,
    model_type: str,
    generated_response: str,
    args: argparse.Namespace,
) -> Optional[Dict]:
    """
    Build the full [prompt + response] input dict for a teacher-forced forward pass.

    Returns a dict with keys:
        inputs        : dict ready for model(**inputs, output_hidden_states=True)
        input_len     : int   – number of prompt tokens (states extracted here - 1)
        boundaries    : dict  – token boundary metadata for the PROMPT portion
        token_ids     : list[int] – generated response token ids (for saving)
        token_strs    : list[str] – decoded response tokens (for saving)

    Or None on failure.
    """
    for col in (args.image_column, args.question_column, args.answer_column):
        if col not in sample:
            return None

    image_raw = sample[args.image_column]
    question  = sample[args.question_column]
    image     = resize_image_if_needed(image_raw, max_dim=args.max_image_dim)

    # ── step 1: build prompt-only inputs to get prompt token ids ──────────
    # We need the exact same tokenisation as generate_and_extract.py used,
    # then append the response token ids on top.

    if model_type == "qwen":
        from qwen_vl_utils import process_vision_info
        
        # First, get prompt-only inputs for input_len calculation
        messages_prompt_only = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text",  "text":  question},
            ]},
        ]
        text_prompt = processor.apply_chat_template(
            messages_prompt_only, tokenize=False, add_generation_prompt=True)
        image_inputs_prompt, video_inputs_prompt = process_vision_info(messages_prompt_only)
        
        prompt_inputs = processor(
            text=[text_prompt],
            images=image_inputs_prompt,
            videos=video_inputs_prompt,
            padding=True,
            return_tensors="pt",
        ).to(model.device)

        input_len = prompt_inputs.input_ids.shape[1]
        prompt_ids = prompt_inputs.input_ids[0].cpu().tolist()
        boundaries = detect_token_boundaries(prompt_ids, processor, model_type)

        # Now create full conversation with assistant response
        messages_with_response = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text",  "text":  question},
            ]},
            {"role": "assistant", "content": generated_response},
        ]
        text_full = processor.apply_chat_template(
            messages_with_response, tokenize=False, add_generation_prompt=False)
        image_inputs_full, video_inputs_full = process_vision_info(messages_with_response)
        
        full_inputs = processor(
            text=[text_full],
            images=image_inputs_full,
            videos=video_inputs_full,
            padding=True,
            return_tensors="pt",
        ).to(model.device)

        # Extract response token IDs from the difference
        full_ids = full_inputs.input_ids[0].cpu().tolist()
        response_ids = full_ids[input_len:]
        
        dec_fn = processor.decode
        token_strs = [dec_fn([tid], skip_special_tokens=True) for tid in response_ids]

    elif model_type in ("llava", "gemma", "internvl"):
        # ── shared path for models using processor(images, text) ──────────
        img = image if isinstance(image, Image.Image) else Image.open(image)
        device      = next(model.parameters()).device
        model_dtype = next(model.parameters()).dtype

        if model_type == "llava":
            # For LLaVA: append response as string after template (not in conversation)
            conversation = [{"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": question + "\n\n" + APPENDED_SYSTEM_PROMPT},
            ]}]
            
            # Get prompt WITH generation prompt for input_len (matches generation time)
            prompt_with_gen = processor.apply_chat_template(
                conversation, add_generation_prompt=True, tokenize=False)
            prompt_inputs = processor(images=img, text=prompt_with_gen, return_tensors="pt")
            prompt_inputs = {
                k: v.to(device=device) if isinstance(v, torch.Tensor) else v
                for k, v in prompt_inputs.items()
            }
            if "pixel_values" in prompt_inputs:
                prompt_inputs["pixel_values"] = (
                    prompt_inputs["pixel_values"].to(dtype=model_dtype))

            input_len  = prompt_inputs["input_ids"].shape[1]
            prompt_ids = prompt_inputs["input_ids"][0].cpu().tolist()
            boundaries = detect_token_boundaries(prompt_ids, processor, model_type)

            # Get prompt WITHOUT generation prompt
            prompt_base = processor.apply_chat_template(
                conversation, add_generation_prompt=False, tokenize=False)
            prompt_base_inputs = processor(images=img, text=prompt_base, return_tensors="pt")
            prompt_base_ids = prompt_base_inputs["input_ids"][0].cpu().tolist()
            prompt_base_len = len(prompt_base_ids)

            # Try string concatenation first (matches MAP2_extraction.py)
            prompt_with_answer_str = prompt_base + generated_response
            full_inputs_str = processor(images=img, text=prompt_with_answer_str, return_tensors="pt")
            full_ids_from_str = full_inputs_str["input_ids"][0].cpu().tolist()
            
            # Check if tokens merged (same length means response was absorbed into prompt)
            if len(full_ids_from_str) == prompt_base_len:
                # Tokens merged - tokenize response separately and manually append
                response_ids_raw = processor.tokenizer.encode(
                    generated_response, add_special_tokens=False
                )
                
                if len(response_ids_raw) == 0:
                    logger.error(
                        f"LLaVA: Response tokenizes to empty. prompt_base_len={prompt_base_len}, "
                        f"full_len={len(full_ids_from_str)}, generated_response='{generated_response[:50]}...'"
                    )
                    response_ids = []
                else:
                    # Manually concatenate: prompt_base_ids + response_ids
                    full_ids = prompt_base_ids + response_ids_raw
                    response_ids = response_ids_raw
                    
                    # Reconstruct full_inputs with manually concatenated input_ids
                    # First ensure prompt_base_inputs tensors are on the correct device
                    prompt_base_mask = prompt_base_inputs["attention_mask"].to(device=device)
                    response_len = len(response_ids_raw)
                    full_mask = torch.cat([
                        prompt_base_mask,
                        torch.ones(1, response_len, dtype=torch.long, device=device),
                    ], dim=1)
                    
                    full_inputs = dict(prompt_base_inputs)
                    full_inputs["input_ids"] = torch.tensor([full_ids], dtype=torch.long, device=device)
                    full_inputs["attention_mask"] = full_mask
                    
                    # Move all remaining tensors to device and set dtype
                    full_inputs = {
                        k: v.to(device=device) if isinstance(v, torch.Tensor) else v
                        for k, v in full_inputs.items()
                    }
                    if "pixel_values" in full_inputs:
                        full_inputs["pixel_values"] = (
                            full_inputs["pixel_values"].to(dtype=model_dtype))
            else:
                # String concatenation worked - extract response tokens from difference
                response_ids = full_ids_from_str[prompt_base_len:]
                full_inputs = {
                    k: v.to(device=device) if isinstance(v, torch.Tensor) else v
                    for k, v in full_inputs_str.items()
                }
                if "pixel_values" in full_inputs:
                    full_inputs["pixel_values"] = (
                        full_inputs["pixel_values"].to(dtype=model_dtype))
            
            # Diagnostic check
            if len(response_ids) == 0:
                logger.error(
                    f"LLaVA tokenization issue: prompt_base_len={prompt_base_len}, "
                    f"full_len={len(full_ids_from_str)}, input_len={input_len}, "
                    f"generated_response length={len(generated_response)}, "
                    f"generated_response='{generated_response[:50]}...'. "
                    f"No response tokens extracted."
                )
            
            dec_fn = processor.decode
            token_strs = [dec_fn([tid], skip_special_tokens=True) for tid in response_ids]

        elif model_type == "gemma":
            # For Gemma, include assistant response in chat template to avoid tokenization issues
            # First, get prompt-only inputs for input_len calculation
            conversation_prompt_only = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": [
                    {"type": "image"},
                    {"type": "text", "text": question},
                ]},
            ]
            prompt_text = processor.apply_chat_template(
                conversation_prompt_only, add_generation_prompt=True, tokenize=False)
            prompt_inputs = processor(images=img, text=prompt_text, return_tensors="pt")
            prompt_inputs = {
                k: v.to(device=device) if isinstance(v, torch.Tensor) else v
                for k, v in prompt_inputs.items()
            }
            if "pixel_values" in prompt_inputs:
                prompt_inputs["pixel_values"] = (
                    prompt_inputs["pixel_values"].to(dtype=model_dtype))
            
            input_len  = prompt_inputs["input_ids"].shape[1]
            prompt_ids = prompt_inputs["input_ids"][0].cpu().tolist()
            boundaries = detect_token_boundaries(prompt_ids, processor, model_type)

            # Now create full conversation with assistant response
            conversation_with_response = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": [
                    {"type": "image"},
                    {"type": "text", "text": question},
                ]},
                {"role": "assistant", "content": generated_response},
            ]
            full_text = processor.apply_chat_template(
                conversation_with_response, add_generation_prompt=False, tokenize=False)
            full_inputs = processor(images=img, text=full_text, return_tensors="pt")
            full_inputs = {
                k: v.to(device=device) if isinstance(v, torch.Tensor) else v
                for k, v in full_inputs.items()
            }
            if "pixel_values" in full_inputs:
                full_inputs["pixel_values"] = (
                    full_inputs["pixel_values"].to(dtype=model_dtype))

            # Extract response token IDs from the difference
            full_ids = full_inputs["input_ids"][0].cpu().tolist()
            response_ids = full_ids[input_len:]
            
            dec_fn = processor.decode
            token_strs = [dec_fn([tid], skip_special_tokens=True) for tid in response_ids]

        else:  # internvl
            # First, get prompt-only inputs for input_len calculation
            messages_prompt_only = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": [
                    {"type": "image"},
                    {"type": "text", "text": question},
                ]},
            ]
            prompt_text = processor.apply_chat_template(
                messages_prompt_only, add_generation_prompt=True, tokenize=False)
            prompt_inputs = processor(images=img, text=prompt_text, return_tensors="pt")
            prompt_inputs = {
                k: v.to(device=device) if isinstance(v, torch.Tensor) else v
                for k, v in prompt_inputs.items()
            }
            if "pixel_values" in prompt_inputs:
                prompt_inputs["pixel_values"] = (
                    prompt_inputs["pixel_values"].to(dtype=model_dtype))

            input_len  = prompt_inputs["input_ids"].shape[1]
            prompt_ids = prompt_inputs["input_ids"][0].cpu().tolist()
            boundaries = detect_token_boundaries(prompt_ids, processor, model_type)

            # Now create full conversation with assistant response
            messages_with_response = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": [
                    {"type": "image"},
                    {"type": "text", "text": question},
                ]},
                {"role": "assistant", "content": generated_response},
            ]
            full_text = processor.apply_chat_template(
                messages_with_response, add_generation_prompt=False, tokenize=False)
            full_inputs = processor(images=img, text=full_text, return_tensors="pt")
            full_inputs = {
                k: v.to(device=device) if isinstance(v, torch.Tensor) else v
                for k, v in full_inputs.items()
            }
            if "pixel_values" in full_inputs:
                full_inputs["pixel_values"] = (
                    full_inputs["pixel_values"].to(dtype=model_dtype))

            # Extract response token IDs from the difference
            full_ids = full_inputs["input_ids"][0].cpu().tolist()
            response_ids = full_ids[input_len:]
            
            dec_fn = processor.decode
            token_strs = [dec_fn([tid], skip_special_tokens=True) for tid in response_ids]

    elif model_type == "deepseek":
        import tempfile

        if isinstance(image, Image.Image):
            tf = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            image.save(tf.name)
            image_path  = tf.name
            tmp_created = True
        else:
            image_path  = image
            tmp_created = False

        # First, get prompt-only inputs for input_len calculation.
        # NOTE: DeepSeek's processor requires an Assistant turn (even empty) so EOS
        # gets appended; its process_one asserts input_ids[-1] == eos_id.
        conversation_prompt_only = [
            {"role": "<|User|>",      "content": f"<image>\n{question}", "images": [image_path]},
            {"role": "<|Assistant|>", "content": ""},
        ]
        pil_images_prompt = load_pil_images(conversation_prompt_only)
        prompt_inputs = processor(
            conversations=conversation_prompt_only, images=pil_images_prompt,
            force_batchify=True, system_prompt=SYSTEM_PROMPT,
        ).to(model.device)

        model_dtype = next(model.parameters()).dtype
        if (hasattr(prompt_inputs, "pixel_values")
                and prompt_inputs.pixel_values is not None):
            prompt_inputs.pixel_values = (
                prompt_inputs.pixel_values.to(dtype=model_dtype))

        input_len  = prompt_inputs.input_ids.shape[1]
        prompt_ids = prompt_inputs.input_ids[0].cpu().tolist()
        boundaries = detect_token_boundaries(prompt_ids, processor, model_type)

        # Now create full conversation with assistant response
        conversation_with_response = [
            {"role": "<|User|>",      "content": f"<image>\n{question}", "images": [image_path]},
            {"role": "<|Assistant|>", "content": generated_response},
        ]
        pil_images_full = load_pil_images(conversation_with_response)
        full_inputs = processor(
            conversations=conversation_with_response, images=pil_images_full,
            force_batchify=True, system_prompt=SYSTEM_PROMPT,
        ).to(model.device)

        if (hasattr(full_inputs, "pixel_values")
                and full_inputs.pixel_values is not None):
            full_inputs.pixel_values = (
                full_inputs.pixel_values.to(dtype=model_dtype))

        if tmp_created:
            try:
                os.unlink(image_path)
            except Exception:
                pass

        # Extract response token IDs from the difference
        full_ids = full_inputs.input_ids[0].cpu().tolist()
        response_ids = full_ids[input_len:]
        
        dec_fn = processor.tokenizer.decode
        token_strs = [dec_fn([tid], skip_special_tokens=True) for tid in response_ids]

        # For DeepSeek we use inputs_embeds rather than input_ids for the forward pass
        # Get embeddings via DeepSeek's prepare_inputs_embeds
        full_embeds = model.prepare_inputs_embeds(**full_inputs)
        # [1, full_len, d]

        full_inputs = {
            "inputs_embeds":  full_embeds,
            "attention_mask": full_inputs.attention_mask,
        }

    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    return {
        "inputs":    full_inputs,
        "input_len": input_len,
        "boundaries": boundaries,
        "token_ids":  response_ids,
        "token_strs": token_strs,
    }


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  Core extraction function                                                 ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def extract_internal_states(
    sample: Dict,
    model,
    processor,
    model_type: str,
    hooks: _InternalStateHooks,
    args: argparse.Namespace,
    prior_info: Dict,
) -> Optional[Dict]:
    """
    For a single sample:
      1. Load generated_response and is_correct from prior extraction.
      2. Build teacher-forced inputs: [prompt] + [response].
      3. Run a SINGLE model(**inputs, output_hidden_states=True) forward pass.
      4. Collect activation states from output.hidden_states at last_input_token_pos.
      5. Collect attention + FFN states from forward hooks at last_input_token_pos.

    No model.generate() call — no KV-cache ambiguity.
    """
    generated_response = prior_info["generated_response"]
    is_correct         = prior_info["is_correct"]

    if not generated_response:
        logger.warning("Empty generated_response from prior extraction. Skipping.")
        return None

    # Reset hooks before this sample's forward pass
    hooks.reset()

    # Build teacher-forced inputs
    tf = build_teacher_forced_inputs(
        sample, model, processor, model_type, generated_response, args)
    if tf is None:
        return None

    full_inputs = tf["inputs"]
    input_len   = tf["input_len"]
    boundaries  = tf["boundaries"]
    token_ids   = tf["token_ids"]
    token_strs  = tf["token_strs"]

    if len(token_ids) == 0:
        hash_id = sample.get("hash_id", sample.get(args.id_column, "unknown"))
        generated_response_preview = generated_response[:100] if generated_response else "(empty)"
        logger.warning(
            f"No response tokens extracted for hash_id={hash_id}. "
            f"input_len={input_len}, full_input_len={full_inputs['input_ids'].shape[1]}, "
            f"generated_response length={len(generated_response)}, "
            f"generated_response preview='{generated_response_preview}...'. "
            f"This may indicate tokenization mismatch or empty/whitespace-only response. Skipping."
        )
        return None

    # ── single teacher-forced forward pass ──────────────────────────────
    model.eval()
    with torch.no_grad():
        if model_type == "deepseek":
            # DeepSeek uses inputs_embeds path through model.language
            output = model.language(
                **full_inputs,
                output_hidden_states=True,
                return_dict=True,
            )
        else:
            output = model(
                **full_inputs,
                output_hidden_states=True,
                return_dict=True,
            )

    # ── extract states at last input token position ───────────────────────
    # last_input_token_pos is the index of the final prompt token in the
    # full [prompt + response] sequence — this is exactly what the paper
    # refers to as position N.
    last_input_pos = input_len - 1

    activation_states = collect_activation_states(output, last_input_pos)  # [L, d]
    attn_states, ff_states = hooks.collect(last_input_pos)                 # each [L, d]

    # ── sanity check ─────────────────────────────────────────────────────
    assert (activation_states.shape[0]
            == attn_states.shape[0]
            == ff_states.shape[0]), (
        f"Layer count mismatch: act={activation_states.shape[0]}, "
        f"attn={attn_states.shape[0]}, ff={ff_states.shape[0]}"
    )

    num_layers = activation_states.shape[0]
    hidden_dim = activation_states.shape[1]

    sample_id = sample.get(args.id_column, "unknown")
    hash_id   = sample.get("hash_id", sample_id)
    question  = sample.get(args.question_column, "")
    answer    = sample.get(args.answer_column, "")

    return {
        "hash_id":            str(hash_id),
        "sample_id":          str(sample_id),
        "question":           str(question),
        "answer":             str(answer),
        "generated_response": str(generated_response),
        "is_correct":         is_correct,

        # Core internal states – STORED SEPARATELY for flexible training
        "activation_states":  activation_states.astype(np.float32),  # [L, d]
        "attention_states":   attn_states.astype(np.float32),        # [L, d]
        "ff_states":          ff_states.astype(np.float32),          # [L, d]

        # Metadata
        "num_layers":         num_layers,
        "hidden_dim":         hidden_dim,
        "boundaries":         json.dumps(boundaries),
        "input_length":       input_len,
        "num_generated_tokens": len(token_ids),
        "token_ids":          np.array(token_ids, dtype=np.int32),
        "token_strs":         np.array(token_strs, dtype=object),
    }


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  Saving helpers                                                           ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def save_sample(result: Dict, output_dir: str, compression: str) -> str:
    samples_dir = os.path.join(output_dir, "samples")
    os.makedirs(samples_dir, exist_ok=True)
    filepath = os.path.join(samples_dir, f"{result['hash_id']}.npz")

    save_dict = {
        "hash_id":              result["hash_id"],
        "sample_id":            result["sample_id"],
        "question":             result["question"],
        "answer":               result["answer"],
        "generated_response":   result["generated_response"],
        "is_correct":           result["is_correct"],

        # float16 on disk, loaded as float32 at train time
        "activation_states":    result["activation_states"].astype(np.float16),
        "attention_states":     result["attention_states"].astype(np.float16),
        "ff_states":            result["ff_states"].astype(np.float16),

        "num_layers":           np.int32(result["num_layers"]),
        "hidden_dim":           np.int32(result["hidden_dim"]),
        "boundaries":           result["boundaries"],
        "input_length":         np.int32(result["input_length"]),
        "num_generated_tokens": np.int32(result["num_generated_tokens"]),
        "token_ids":            result["token_ids"],
        "token_strs":           result["token_strs"],
    }

    save_fn = np.savez_compressed if compression == "compressed" else np.savez
    save_fn(filepath, **save_dict)
    return filepath


def get_completed_samples(output_dir: str) -> set:
    samples_dir = os.path.join(output_dir, "samples")
    if not os.path.exists(samples_dir):
        return set()
    return {f.replace(".npz", "")
            for f in os.listdir(samples_dir) if f.endswith(".npz")}


def create_manifest(output_dir: str, num_samples: int, config: Dict) -> str:
    manifest_path = os.path.join(output_dir, "manifest.json")
    manifest = {
        "num_samples":              num_samples,
        "samples_dir":              "samples/",
        "sample_filename_pattern":  "{hash_id}.npz",
        "extraction_method":        "teacher_forced_single_forward_pass",
        "state_keys": {
            "activation_states": "Post-residual hidden state at last prompt token, all layers [L,d], float16",
            "attention_states":  "MHSA output BEFORE residual add, last prompt token, all layers [L,d], float16",
            "ff_states":         "FFN output BEFORE residual add, last prompt token, all layers [L,d], float16",
        },
        "config":     config,
        "created_at": datetime.now().isoformat(),
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info(f"Manifest saved: {manifest_path}")
    return manifest_path


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  CUDA helpers                                                             ║
# ╚══════════════════════════════════════════════════════════════════════════╝

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


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  Main                                                                     ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def main():
    args    = parse_args()
    device  = "cuda" if torch.cuda.is_available() else "cpu"
    model_ids = [args.model_id] if isinstance(args.model_id, str) else args.model_id

    logger.info("=" * 80)
    logger.info("InternalInspector (I²) v2  –  Teacher-Forced Internal State Extraction")
    logger.info("=" * 80)
    logger.info(f"Models          : {model_ids}")
    logger.info(f"Dataset path    : {args.dataset_path}")
    logger.info(f"Target datasets : {args.target_datasets}")
    logger.info(f"Prior extr. dir : {args.prior_extraction_dir}")
    logger.info(f"Output dir      : {args.output_dir}")
    logger.info(f"GPU IDs         : {args.gpu_ids}")
    logger.info(f"Extraction      : single teacher-forced forward pass (no generation)")

    for model_id in model_ids:
        _setup_model_logging(model_id)
        model_name_part = model_id.split("/")[-1]

        logger.info(f"\n{'#'*80}")
        logger.info(f"  Model: {model_id}")
        logger.info(f"{'#'*80}")

        # ── load model ────────────────────────────────────────────────────
        try:
            model, processor, model_type = load_model_and_processor(
                model_id, args.dtype, device)
        except Exception as e:
            logger.error(f"Failed to load model {model_id}: {e}\n{traceback.format_exc()}")
            continue

        # ── register hooks (once per model) ───────────────────────────────
        hooks = _InternalStateHooks(model, model_type)
        hooks.register()
        logger.info(f"Registered hooks on {hooks.num_layers} transformer layers.")

        for target_dataset in args.target_datasets:
            logger.info(f"\n{'='*80}")
            logger.info(f"  Dataset: {target_dataset}")
            logger.info(f"{'='*80}")

            full_dataset_path = os.path.join(args.dataset_path, target_dataset)
            output_dir = os.path.join(args.output_dir, model_name_part, target_dataset)
            os.makedirs(output_dir, exist_ok=True)

            # ── load prior extraction index ───────────────────────────────
            prior_samples_dir = get_prior_samples_dir(
                args.prior_extraction_dir, model_name_part, target_dataset)

            if prior_samples_dir is None:
                logger.error(
                    f"No prior extraction found for {model_name_part}/{target_dataset}. "
                    "Cannot run teacher-forced extraction without prior responses. Skipping.")
                continue

            prior_index = load_prior_extraction_index(prior_samples_dir)

            if len(prior_index) == 0:
                logger.error("Prior extraction index is empty. Skipping.")
                continue

            # ── already completed ─────────────────────────────────────────
            completed = set()
            if args.skip_if_processed:
                completed = get_completed_samples(output_dir)
                logger.info(f"Already completed: {len(completed)} samples (will skip).")

            # ── load dataset ──────────────────────────────────────────────
            try:
                dataset = load_from_disk(full_dataset_path)
            except Exception as e:
                logger.error(f"Failed to load dataset {full_dataset_path}: {e}")
                continue
            logger.info(f"Loaded dataset: {len(dataset)} samples.")

            # ── slice ─────────────────────────────────────────────────────
            start_idx = args.start_at_idx if args.start_at_idx is not None else 0
            end_idx   = args.end_at_idx   if args.end_at_idx   is not None else len(dataset)
            if args.start_at_idx is not None or args.end_at_idx is not None:
                dataset = dataset.select(range(start_idx, end_idx))
                logger.info(f"Sliced to {len(dataset)} samples "
                            f"(start={start_idx}, end={end_idx})")

            if args.max_samples is not None:
                dataset = dataset.select(range(min(args.max_samples, len(dataset))))
                logger.info(f"Capped to {len(dataset)} samples.")

            # ── save config ───────────────────────────────────────────────
            with open(os.path.join(output_dir, "config.json"), "w") as f:
                json.dump(vars(args), f, indent=2)

            # ── main loop ─────────────────────────────────────────────────
            n_saved = n_skipped = n_failed = n_no_prior = 0

            n_iter   = min(args.debug_samples, len(dataset)) if args.debug else len(dataset)
            iterator = (range(n_iter) if args.debug
                        else tqdm(range(n_iter), desc=f"Extracting {target_dataset}"))

            for sample_idx in iterator:
                sample  = dataset[sample_idx]
                hash_id = str(sample.get("hash_id",
                              sample.get(args.id_column, f"idx_{sample_idx}")))

                logger.info(f"Processing hash_id={hash_id}  idx={sample_idx}")

                # ── skip if done ──────────────────────────────────────────
                if args.skip_if_processed and hash_id in completed:
                    n_skipped += 1
                    continue

                # ── look up prior info ────────────────────────────────────
                prior_info = prior_index.get(hash_id, None)
                if prior_info is None:
                    logger.warning(
                        f"  hash_id={hash_id} not found in prior extraction index. "
                        "Skipping (cannot do teacher-forced pass without prior response).")
                    n_no_prior += 1
                    continue

                try:
                    result = extract_internal_states(
                        sample=sample,
                        model=model,
                        processor=processor,
                        model_type=model_type,
                        hooks=hooks,
                        args=args,
                        prior_info=prior_info,
                    )

                    if result is None:
                        n_failed += 1
                        continue

                    if args.debug:
                        print(f"\n{'─'*60}")
                        print(f"  hash_id          : {hash_id}")
                        print(f"  question         : {result['question'][:80]}")
                        print(f"  generated resp.  : {result['generated_response'][:80]}")
                        print(f"  is_correct       : {result['is_correct']}")
                        print(f"  num_layers       : {result['num_layers']}")
                        print(f"  hidden_dim       : {result['hidden_dim']}")
                        print(f"  activation_states: {result['activation_states'].shape}")
                        print(f"  attention_states : {result['attention_states'].shape}")
                        print(f"  ff_states        : {result['ff_states'].shape}")
                        print(f"{'─'*60}")

                    save_sample(result, output_dir, args.compression)
                    n_saved += 1

                except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
                    n_failed += 1
                    logger.error(f"CUDA error on hash_id={hash_id}: {e}")
                    for _ in range(5):
                        clear_cuda_cache(aggressive=True)
                    gc.collect()
                    continue

                except Exception as e:
                    n_failed += 1
                    logger.error(
                        f"Error on hash_id={hash_id}: {e}\n{traceback.format_exc()}")
                    clear_cuda_cache()
                    continue

                finally:
                    hooks.reset()

                # periodic cleanup
                clear_cuda_cache(aggressive=True)
                if n_saved % 5 == 0:
                    for _ in range(3):
                        clear_cuda_cache(aggressive=True)
                if n_saved % 100 == 0:
                    clear_cuda_cache(aggressive=True)
                    logger.info(f"  {n_saved} samples saved so far.")

            # ── manifest ──────────────────────────────────────────────────
            create_manifest(output_dir, n_saved, vars(args))

            logger.info(f"\n{'='*80}")
            logger.info(f"  Done: {target_dataset}")
            logger.info(f"  Saved      : {n_saved}")
            logger.info(f"  Skipped    : {n_skipped}")
            logger.info(f"  Failed     : {n_failed}")
            logger.info(f"  No prior   : {n_no_prior}")
            logger.info(f"  Output     : {output_dir}")
            logger.info(f"{'='*80}")

            clear_cuda_cache(aggressive=True)

        # ── teardown ──────────────────────────────────────────────────────
        hooks.remove()
        logger.info("Forward hooks removed.")
        _remove_model_logging(model_id)
        del model, processor
        clear_cuda_cache(aggressive=True)
        logger.info(f"Model {model_id} unloaded.")

    logger.info("\nAll models and datasets processed.")
    return args.output_dir


if __name__ == "__main__":
    out = main()
    print(f"\nExtraction complete. Results: {out}")


# ── Supported models ──────────────────────────────────────────────────────────
# Qwen/Qwen3-VL-8B-Instruct
# llava-hf/llava-v1.6-vicuna-13b-hf
# OpenGVLab/InternVL3_5-14B-HF
# google/gemma-3-27b-it
# deepseek-ai/deepseek-vl2

# ── Example usage ─────────────────────────────────────────────────────────────
# python II_extraction_v2.py \
#     --gpu_ids "0" \
#     --model_id "llava-hf/llava-v1.6-vicuna-13b-hf" \
#     --dataset_path "../data/VLCB/raw" \
#     --target_datasets test \
#     --prior_extraction_dir "../data/extraction/raw/" \
#     --output_dir "../data/II_extraction_v2/" \
#     --dtype "float32" \
#     --max_image_dim 2048 \
#     --compression "compressed" \
#     --start_at_idx 25000 \
#     --end_at_idx 25000 \
#     --skip_if_processed

# ── Output .npz keys ──────────────────────────────────────────────────────────
# hash_id, sample_id, question, answer, generated_response, is_correct,
# activation_states [L, d] float16  ← post-residual hidden state
# attention_states  [L, d] float16  ← MHSA output before residual
# ff_states         [L, d] float16  ← FFN  output before residual
# num_layers, hidden_dim, boundaries (JSON), input_length,
# num_generated_tokens, token_ids, token_strs

# ── Compatible with II_train.py and II_eval.py unchanged ─────────────────────