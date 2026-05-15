"""
Prompt Ensemble (PE) Paraphrase Generation Script
Generates multiple paraphrased versions of questions for VLM confidence estimation.

This script reads VLM extraction results and generates N paraphrased versions of each
question using GPT, then stores them alongside the original data for later inference.
"""

import os
import argparse
import numpy as np
import re
import logging
from pathlib import Path
from tqdm import tqdm
from openai import OpenAI
import traceback
import shutil
import json
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Tuple
import time
from datasets import load_from_disk
from PIL import Image
import io

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
    ]
)
logger = logging.getLogger(__name__)

# Suppress noisy HTTP/API client logs
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

# ============================================================================
# Constants
# ============================================================================

MAX_IMAGE_DIMENSION = 2048  # Default maximum allowed dimension (width or height)

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
    
    # Ensure RGB mode
    if pil_img.mode != 'RGB':
        pil_img = pil_img.convert('RGB')
    
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
    
    logger.debug(f"Resized image from {w}x{h} to {new_w}x{new_h}")
    
    return resized

# ============================================================================
# Argument Parser
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate paraphrased questions for Prompt Ensemble confidence estimation'
    )
    
    # Data configuration
    parser.add_argument('--dataset_path', type=str, required=True,
                        help='Path to the original dataset directory (e.g., ../data/VLCB/raw/test)')
    parser.add_argument('--output_path', type=str, required=True,
                        help='Output path where the PE dataset will be created (e.g., ../data/PE/)')
    parser.add_argument('--output_dataset', type=str, default=None,
                        help='Output dataset name (default: inferred from dataset_path + _PE)')
    
    # Paraphrase configuration
    parser.add_argument('--num_paraphrases', type=int, default=20,
                        help='Number of paraphrases to generate per question (default: 20)')
    
    # GPT configuration
    parser.add_argument('--openai-api-key', type=str, default=None,
                        help='OpenAI API key (if not provided, uses OPENAI_API_KEY environment variable)')
    parser.add_argument('--gpt-model', type=str, default='gpt-4o-mini',
                        help='GPT model to use for paraphrase generation (default: gpt-4o-mini)')
    parser.add_argument('--reasoning-effort', type=str, default='low',
                        choices=['low', 'medium', 'high'],
                        help='Reasoning effort for GPT model (default: low)')
    
    # Processing configuration
    parser.add_argument('--skip-if-exists', action='store_true',
                        help='Skip samples that already have paraphrases in output directory')
    parser.add_argument('--overwrite', action='store_true',
                        help='Overwrite existing output dataset if it exists')
    parser.add_argument('--max-concurrent', type=int, default=10,
                        help='Maximum number of concurrent API requests (default: 10)')
    parser.add_argument('--max-file-concurrent', type=int, default=10,
                        help='Maximum number of concurrent file operations (default: 10, reduce if getting "too many open files" errors)')
    parser.add_argument('--max-image-dim', type=int, default=MAX_IMAGE_DIMENSION,
                        help='Maximum image dimension (images larger than this will be resized, default: 2048)')
    parser.add_argument('--allow-fewer-paraphrases', action='store_true',
                        help='Allow fewer paraphrases than requested. By default, samples with incorrect count will fail.')
    
    # Debug configuration
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug mode with detailed printing')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Maximum number of samples to process per dataset (for testing)')
    
    return parser.parse_args()


# ============================================================================
# Paraphrase Generation Prompt
# ============================================================================

PARAPHRASE_SYSTEM_PROMPT = """You are an expert linguist and domain specialist generating alternative phrasings for a Visual Question Answering (VQA) task. Your goal is to generate variations of a question while preserving its exact semantic meaning and expected answer.

### Instructions:
- Generate the requested number of alternative ways to phrase the question.
- **Preserve the Answer:** Ensure that for any given image, the answer to your new questions would be *identical* to the answer of the original question.
- **Maintain Domain Precision:**
  - If the question is **Medical**, preserve the correct clinical terminology.
  - If the question is **Financial**, keep the technical intent clear.
- **Maintain Question Type:**
  - If the original is a **Yes/No** question, your rephrasings must remain Yes/No questions.
  - If the original asks for a **Count**, your rephrasings must ask for a number.
- **Multiple Choice Handling (Crucial):**
  - If the original question includes multiple options (e.g., A, B, C, D):
    1. Rephrase **only** the question stem (the text asking the question).
    2. **Append the EXACT same options** in the **EXACT same order** to the end of your rephrased question.
    3. Do **not** shuffle, reword, or modify the options in any way. 
- **Allowed Changes:** You may vary word order, sentence structure, and use strict synonyms for the question text.
- **Prohibited Changes:** Do **not** add new constraints, remove location details, or introduce ambiguity.

### Output Format:
Each rephrased question should be wrapped in numbered tags like this:
[question_1] Rephrased question stem? (A) Option 1 (B) Option 2... [/question_1]
[question_2] Rephrased question stem? (A) Option 1 (B) Option 2... [/question_2]
...and so on for each question."""


def get_paraphrase_user_prompt(question, num_paraphrases):
    """Generate the user prompt for paraphrase generation."""
    return f"""### Original Question: '{question}'

Please generate **{num_paraphrases}** alternative phrasings for this question.

### Output Format:
[question_1] ... [/question_1]
[question_2] ... [/question_2]
...
[question_{num_paraphrases}] ... [/question_{num_paraphrases}]"""


# ============================================================================
# GPT Paraphrase Generation
# ============================================================================

def parse_paraphrases(response_text, num_expected):
    """
    Parse paraphrases from GPT response using numbered tags.
    
    Args:
        response_text: Raw text response from GPT
        num_expected: Expected number of paraphrases
    
    Returns:
        list: List of parsed paraphrase strings
    """
    paraphrases = []
    
    # Pattern to match [question_N] ... [/question_N]
    pattern = r'\[question_(\d+)\]\s*(.*?)\s*\[/question_\1\]'
    matches = re.findall(pattern, response_text, re.DOTALL | re.IGNORECASE)
    
    for match in matches:
        idx, text = match
        paraphrases.append(text.strip())
    
    # If regex didn't find enough, try alternative patterns
    if len(paraphrases) < num_expected:
        # Try pattern without closing tag
        alt_pattern = r'\[question_(\d+)\]\s*([^\[]+)'
        alt_matches = re.findall(alt_pattern, response_text, re.IGNORECASE)
        
        for match in alt_matches:
            idx, text = match
            text = text.strip().rstrip('?') + '?'  # Ensure question mark
            if text not in paraphrases:
                paraphrases.append(text)
    
    # If still not enough, try numbered list pattern
    if len(paraphrases) < num_expected:
        # Pattern for "1. Question?" or "1) Question?"
        list_pattern = r'(?:^|\n)\s*\d+[\.\)]\s*([^\n]+\?)'
        list_matches = re.findall(list_pattern, response_text)
        
        for text in list_matches:
            text = text.strip()
            if text not in paraphrases:
                paraphrases.append(text)
    
    return paraphrases[:num_expected]  # Return at most num_expected


def generate_paraphrases_with_gpt(question, num_paraphrases, openai_api_key=None, 
                                   gpt_model='gpt-4o-mini', reasoning_effort='low',
                                   debug=False):
    """
    Generate paraphrases for a question using GPT (synchronous version).
    
    Args:
        question: The original question string
        num_paraphrases: Number of paraphrases to generate
        openai_api_key: OpenAI API key (if None, uses environment variable)
        gpt_model: GPT model to use
        reasoning_effort: Reasoning effort level
        debug: Whether to print debug information
    
    Returns:
        tuple: (paraphrases_list, raw_response_text) or (None, None) on error
    """
    if openai_api_key is None:
        openai_api_key = os.getenv("OPENAI_API_KEY")
    
    if openai_api_key is None:
        logger.error("No OpenAI API key provided. Cannot generate paraphrases.")
        return None, None
    
    client = OpenAI(api_key=openai_api_key)
    
    # Prepare messages
    messages = [
        {"role": "system", "content": PARAPHRASE_SYSTEM_PROMPT},
        {"role": "user", "content": get_paraphrase_user_prompt(question, num_paraphrases)}
    ]
    
    try:
        # Use responses API with reasoning for o-series models
        response = client.responses.create(
            model=gpt_model,
            reasoning={"effort": reasoning_effort},
            input=messages,
            max_output_tokens=25000,
        )
        output_text = response.output_text
        
        if debug:
            logger.info(f"GPT Response:\n{output_text[:500]}...")
        
        # Parse paraphrases from response
        paraphrases = parse_paraphrases(output_text, num_paraphrases)
        
        if len(paraphrases) < num_paraphrases:
            logger.warning(f"Only parsed {len(paraphrases)}/{num_paraphrases} paraphrases")
        
        return paraphrases, output_text
            
    except Exception as e:
        logger.error(f"Failed to generate paraphrases with GPT: {e}")
        if debug:
            logger.error(traceback.format_exc())
        return None, None


async def generate_paraphrases_with_gpt_async(question, num_paraphrases, openai_api_key=None, 
                                               gpt_model='gpt-4o-mini', reasoning_effort='low',
                                               debug=False):
    """
    Generate paraphrases for a question using GPT (async version).
    
    Args:
        question: The original question string
        num_paraphrases: Number of paraphrases to generate
        openai_api_key: OpenAI API key (if None, uses environment variable)
        gpt_model: GPT model to use
        reasoning_effort: Reasoning effort level
        debug: Whether to print debug information
    
    Returns:
        tuple: (paraphrases_list, raw_response_text) or (None, None) on error
    """
    if openai_api_key is None:
        openai_api_key = os.getenv("OPENAI_API_KEY")
    
    if openai_api_key is None:
        logger.error("No OpenAI API key provided. Cannot generate paraphrases.")
        return None, None
    
    # Run synchronous client in thread pool
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor() as executor:
        result = await loop.run_in_executor(
            executor,
            generate_paraphrases_with_gpt,
            question, num_paraphrases, openai_api_key, gpt_model, reasoning_effort, debug
        )
    return result


# ============================================================================
# NPZ File Operations
# ============================================================================

def load_sample_npz(filepath):
    """
    Load a sample .npz file and return its contents as a dict.
    
    Args:
        filepath: Path to .npz file
    
    Returns:
        dict: Sample data with all fields
    """
    with np.load(filepath, allow_pickle=True) as data:
        # Convert to regular dict
        sample = {}
        for key in data.files:
            sample[key] = data[key]
            
            # Convert numpy scalars to Python types
            if isinstance(sample[key], np.ndarray) and sample[key].shape == ():
                sample[key] = sample[key].item()
    
    return sample


def save_sample_with_paraphrases(filepath, original_sample, paraphrases):
    """
    Save a sample with paraphrases to a new .npz file.
    Preserves all existing fields and adds paraphrases.
    
    Args:
        filepath: Path to save .npz file
        original_sample: Dictionary with original sample data
        paraphrases: List of paraphrase strings
    """
    # Create save dict with all original fields
    save_dict = {}
    for key, value in original_sample.items():
        save_dict[key] = value
    
    # Add paraphrases as numpy array of strings
    save_dict['paraphrases'] = np.array(paraphrases, dtype=object)
    save_dict['num_paraphrases'] = len(paraphrases)
    
    # Save
    np.savez_compressed(filepath, **save_dict)


def check_has_paraphrases(filepath):
    """
    Check if a sample .npz file already has paraphrases.
    
    Args:
        filepath: Path to .npz file
    
    Returns:
        bool: True if has paraphrases, False otherwise
    """
    try:
        with np.load(filepath, allow_pickle=True) as data:
            # Check if paraphrases field exists and is not empty
            if 'paraphrases' not in data.files:
                return False
            
            paraphrases = data['paraphrases']
            return paraphrases is not None and len(paraphrases) > 0
        
    except Exception as e:
        logger.error(f"Error checking paraphrases in {filepath}: {e}")
        return False


# ============================================================================
# JSON Storage Operations
# ============================================================================

def save_raw_response_json(filepath, hash_id, question, raw_response, metadata=None):
    """
    Save raw GPT response as JSON.
    
    Args:
        filepath: Path to save JSON file
        hash_id: Hash ID of the sample
        question: Original question
        raw_response: Raw GPT response text
        metadata: Optional metadata dict
    """
    data = {
        'hash_id': hash_id,
        'question': question,
        'raw_response': raw_response,
        'timestamp': time.time(),
        'metadata': metadata or {}
    }
    
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.flush()  # Ensure data is written immediately


def save_paraphrases_json(filepath, hash_id, question, paraphrases, raw_response=None, metadata=None):
    """
    Save processed paraphrases as JSON.
    
    Args:
        filepath: Path to save JSON file
        hash_id: Hash ID of the sample
        question: Original question
        paraphrases: List of paraphrase strings
        raw_response: Optional raw GPT response text
        metadata: Optional metadata dict
    """
    data = {
        'hash_id': hash_id,
        'question': question,
        'paraphrases': paraphrases,
        'num_paraphrases': len(paraphrases),
        'raw_response': raw_response,
        'timestamp': time.time(),
        'metadata': metadata or {}
    }
    
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.flush()  # Ensure data is written immediately


def load_paraphrases_json(filepath):
    """
    Load processed paraphrases from JSON.
    
    Args:
        filepath: Path to JSON file
    
    Returns:
        dict: Paraphrases data or None on error
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading paraphrases JSON from {filepath}: {e}")
        return None


def check_has_paraphrases_json(filepath):
    """
    Check if a JSON file with paraphrases exists and is valid.
    
    Args:
        filepath: Path to JSON file
    
    Returns:
        bool: True if has valid paraphrases, False otherwise
    """
    if not os.path.exists(filepath):
        return False
    
    data = load_paraphrases_json(filepath)
    if data is None:
        return False
    
    return 'paraphrases' in data and len(data.get('paraphrases', [])) > 0


# ============================================================================
# Dataset Processing
# ============================================================================

def get_all_sample_files(samples_dir):
    """
    Get all .npz sample files in a directory.
    
    Args:
        samples_dir: Path to samples directory
    
    Returns:
        list: List of .npz file paths
    """
    if not os.path.exists(samples_dir):
        return []
    
    sample_files = []
    for filename in os.listdir(samples_dir):
        if filename.endswith('.npz'):
            filepath = os.path.join(samples_dir, filename)
            sample_files.append(filepath)
    
    return sorted(sample_files)


def setup_output_directory(output_dataset, output_path, overwrite=False):
    """
    Setup the output directory for the paraphrased dataset.
    
    Args:
        output_dataset: Output dataset name
        output_path: Base output path where datasets will be created
        overwrite: Whether to overwrite existing directory
    
    Returns:
        tuple: (output_samples_dir, raw_responses_dir, paraphrases_json_dir)
    """
    dataset_output_path = os.path.join(output_path, output_dataset)
    output_samples_dir = os.path.join(dataset_output_path, 'samples')
    raw_responses_dir = os.path.join(dataset_output_path, 'raw_responses')
    paraphrases_json_dir = os.path.join(dataset_output_path, 'paraphrases_json')
    
    if os.path.exists(dataset_output_path):
        if overwrite:
            logger.warning(f"Overwriting existing output directory: {dataset_output_path}")
            shutil.rmtree(dataset_output_path)
        else:
            logger.info(f"Output directory exists: {dataset_output_path}")
            # Don't remove, we'll skip or overwrite individual files based on flags
    
    # Create output directory structure
    os.makedirs(output_samples_dir, exist_ok=True)
    os.makedirs(raw_responses_dir, exist_ok=True)
    os.makedirs(paraphrases_json_dir, exist_ok=True)
    
    return output_samples_dir, raw_responses_dir, paraphrases_json_dir


def get_already_processed_hash_ids(paraphrases_json_dir):
    """
    Get set of hash_ids that already have paraphrases JSON files.
    
    Args:
        paraphrases_json_dir: Directory containing paraphrases JSON files
    
    Returns:
        set: Set of hash_ids (without .json extension) that are already processed
    """
    already_processed = set()
    
    if not os.path.exists(paraphrases_json_dir):
        return already_processed
    
    # List all JSON files in the directory
    for filename in os.listdir(paraphrases_json_dir):
        if filename.endswith('.json'):
            # Extract hash_id from filename (remove .json extension)
            hash_id = filename[:-5]  # Remove '.json'
            # Verify it's actually a valid paraphrases file
            json_filepath = os.path.join(paraphrases_json_dir, filename)
            if check_has_paraphrases_json(json_filepath):
                already_processed.add(hash_id)
    
    return already_processed


async def process_single_sample(sample, output_samples_dir, 
                                 raw_responses_dir, paraphrases_json_dir, args, file_semaphore):
    """
    Process a single sample: generate paraphrases and store as JSON.
    
    Args:
        sample: Dataset sample dict with 'hash_id' and 'question'
        output_samples_dir: Output samples directory (for final .npz)
        raw_responses_dir: Directory for raw response JSON files
        paraphrases_json_dir: Directory for paraphrases JSON files
        args: Command line arguments
        file_semaphore: Semaphore to limit concurrent file operations
    
    Returns:
        tuple: (success: bool, hash_id: str, error_message: str or None)
    """
    try:
        # Get hash_id and question
        hash_id = str(sample['hash_id'])
        question = str(sample['question'])
        
        # Check if already processed (if skip flag is set) - use file semaphore
        json_filepath = os.path.join(paraphrases_json_dir, f"{hash_id}.json")
        if args.skip_if_exists:
            async with file_semaphore:
                if check_has_paraphrases_json(json_filepath):
                    return (True, None, 'skipped')
        
        if args.debug:
            logger.info(f"\nProcessing: {hash_id}")
            logger.info(f"Question: {question}")
        
        # Generate paraphrases with GPT (async)
        paraphrases, raw_response = await generate_paraphrases_with_gpt_async(
            question,
            args.num_paraphrases,
            args.openai_api_key,
            args.gpt_model,
            args.reasoning_effort,
            args.debug
        )
        
        if paraphrases is None or len(paraphrases) == 0:
            return (False, hash_id, "Failed to generate paraphrases")
        
        # Validate paraphrase count
        num_generated = len(paraphrases)
        num_expected = args.num_paraphrases
        
        if num_generated != num_expected:
            if not args.allow_fewer_paraphrases:
                error_msg = f"Expected {num_expected} paraphrases but got {num_generated} for {hash_id}"
                logger.error(error_msg)
                return (False, hash_id, error_msg)
            else:
                logger.warning(f"Expected {num_expected} paraphrases but got {num_generated} for {hash_id} (allowing due to --allow-fewer-paraphrases)")
        
        if args.debug:
            logger.info(f"Generated {num_generated}/{num_expected} paraphrases for {hash_id}")
        
        # Save files with file semaphore to limit concurrent file operations
        async with file_semaphore:
            # Save raw response as JSON
            raw_response_filepath = os.path.join(raw_responses_dir, f"{hash_id}_raw.json")
            save_raw_response_json(
                raw_response_filepath,
                hash_id,
                question,
                raw_response or "",
                metadata={'gpt_model': args.gpt_model, 'reasoning_effort': args.reasoning_effort}
            )
            
            # Save paraphrases as JSON
            save_paraphrases_json(
                json_filepath,
                hash_id,
                question,
                paraphrases,
                raw_response=raw_response,
                metadata={'gpt_model': args.gpt_model, 'reasoning_effort': args.reasoning_effort}
            )
        
        return (True, hash_id, None)
        
    except Exception as e:
        error_msg = f"Error processing hash_id {hash_id}: {e}"
        logger.error(error_msg)
        if args.debug:
            logger.error(traceback.format_exc())
        return (False, None, error_msg)


async def process_samples_parallel_by_index(dataset, indices, output_samples_dir, raw_responses_dir, 
                                            paraphrases_json_dir, args):
    """
    Process all samples in parallel using asyncio, accessing by index to avoid keeping all in memory.
    
    Args:
        dataset: HuggingFace dataset object
        indices: List of indices to process
        output_samples_dir: Output samples directory (for final .npz)
        raw_responses_dir: Directory for raw response JSON files
        paraphrases_json_dir: Directory for paraphrases JSON files
        args: Command line arguments
    
    Returns:
        dict: Statistics about processing
    """
    stats = {
        'processed': 0,
        'skipped': 0,
        'failed': 0,
        'total': len(indices)
    }
    
    # Create semaphores to limit concurrent operations
    api_semaphore = asyncio.Semaphore(args.max_concurrent)
    file_semaphore = asyncio.Semaphore(args.max_file_concurrent)
    
    async def process_with_semaphore(idx):
        # Access sample by index (avoids keeping entire dataset in memory)
        try:
            sample = dataset[idx]
        except Exception as e:
            logger.error(f"Error accessing sample at index {idx}: {e}")
            return (False, None, f"Error accessing sample: {e}")
        
        async with api_semaphore:
            return await process_single_sample(
                sample, output_samples_dir,
                raw_responses_dir, paraphrases_json_dir, args, file_semaphore
            )
    
    # Process all samples with progress bar
    # Create tasks properly
    tasks = [asyncio.create_task(process_with_semaphore(idx)) for idx in indices]
    
    try:
        with tqdm(total=len(tasks), desc=f"Generating paraphrases") as pbar:
            # Use as_completed to process results as they come in
            # This will yield tasks as they complete
            for done_task in asyncio.as_completed(tasks):
                try:
                    success, hash_id, error_msg = await done_task
                    
                    if error_msg == 'skipped':
                        stats['skipped'] += 1
                    elif success:
                        stats['processed'] += 1
                    else:
                        stats['failed'] += 1
                    
                    pbar.update(1)
                    pbar.set_postfix({
                        'processed': stats['processed'],
                        'skipped': stats['skipped'],
                        'failed': stats['failed']
                    })
                except asyncio.CancelledError:
                    # Task was cancelled, count as failed
                    stats['failed'] += 1
                    pbar.update(1)
                except Exception as e:
                    stats['failed'] += 1
                    logger.error(f"Error in task completion: {e}")
                    if args.debug:
                        logger.error(traceback.format_exc())
                    pbar.update(1)
    
    except KeyboardInterrupt:
        logger.warning("Interrupted by user. Cancelling remaining tasks...")
        # Cancel all remaining tasks
        for task in tasks:
            if not task.done():
                task.cancel()
        # Wait for cancellations to complete (suppress errors)
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        raise
    
    finally:
        # Ensure all tasks are done (wait for any stragglers)
        # This prevents "Task was destroyed but it is pending" errors
        remaining = [t for t in tasks if not t.done()]
        if remaining:
            logger.debug(f"Waiting for {len(remaining)} remaining tasks...")
            await asyncio.gather(*remaining, return_exceptions=True)
    
    return stats


def create_final_dataset_from_json(dataset, paraphrases_json_dir, 
                                    output_samples_dir, args):
    """
    Create final .npz dataset from dataset samples and JSON paraphrases.
    
    Args:
        dataset: Original dataset (HuggingFace dataset object)
        paraphrases_json_dir: Directory with paraphrases JSON files
        output_samples_dir: Directory to save final .npz files
        args: Command line arguments
    
    Returns:
        dict: Statistics about dataset creation
    """
    stats = {
        'created': 0,
        'missing_json': 0,
        'failed': 0,
        'incorrect_count': 0
    }
    
    logger.info(f"\nCreating final dataset from {len(dataset)} samples...")
    
    # Process by index to avoid keeping dataset handles open
    for idx in tqdm(range(len(dataset)), desc="Creating final dataset"):
        try:
            # Access sample by index
            sample = dataset[idx]
            hash_id = str(sample['hash_id'])
            json_filepath = os.path.join(paraphrases_json_dir, f"{hash_id}.json")
            output_filepath = os.path.join(output_samples_dir, f"{hash_id}.npz")
            
            # Check if JSON exists
            if not os.path.exists(json_filepath):
                stats['missing_json'] += 1
                if args.debug:
                    logger.warning(f"Missing JSON for hash_id {hash_id}")
                continue
            
            # Load paraphrases from JSON
            json_data = load_paraphrases_json(json_filepath)
            if json_data is None or 'paraphrases' not in json_data:
                stats['missing_json'] += 1
                continue
            
            paraphrases = json_data['paraphrases']
            num_paraphrases = len(paraphrases)
            num_expected = args.num_paraphrases
            
            # Validate paraphrase count
            if num_paraphrases != num_expected:
                if not args.allow_fewer_paraphrases:
                    stats['incorrect_count'] += 1
                    logger.error(f"JSON file {hash_id}.json has {num_paraphrases} paraphrases, expected {num_expected}. Skipping.")
                    continue
                else:
                    logger.warning(f"JSON file {hash_id}.json has {num_paraphrases}/{num_expected} paraphrases (allowing due to --allow-fewer-paraphrases)")
            
            # Convert sample to dict and resize image if needed
            sample_dict = dict(sample)
            
            # Resize image if present
            if 'image' in sample_dict:
                try:
                    sample_dict['image'] = resize_image_if_needed(sample_dict['image'], max_dim=args.max_image_dim)
                except Exception as e:
                    logger.warning(f"Failed to resize image for {hash_id}: {e}")
                    # Continue with original image
            
            save_sample_with_paraphrases(output_filepath, sample_dict, paraphrases)
            stats['created'] += 1
            
        except Exception as e:
            stats['failed'] += 1
            logger.error(f"Error creating final dataset entry for hash_id {hash_id}: {e}")
            if args.debug:
                logger.error(traceback.format_exc())
            continue
    
    return stats


def process_dataset(dataset_path, output_dataset, output_path, args):
    """
    Process dataset with parallel processing and JSON storage.
    
    Args:
        dataset_path: Path to the original dataset directory
        output_dataset: Output dataset name
        output_path: Base output path where datasets will be created
        args: Command line arguments
    
    Returns:
        dict: Statistics about processing (processed, skipped, failed)
    """
    # Load dataset
    logger.info(f"Loading dataset from: {dataset_path}")
    if os.path.isdir(dataset_path):
        dataset = load_from_disk(dataset_path)
    else:
        from datasets import load_dataset as hf_load_dataset
        dataset = hf_load_dataset(dataset_path)
    
    logger.info(f"Loaded {len(dataset)} samples")
    
    if len(dataset) == 0:
        logger.warning("Dataset is empty")
        return {'processed': 0, 'skipped': 0, 'failed': 0, 'total': 0}
    
    # Setup output directories
    output_samples_dir, raw_responses_dir, paraphrases_json_dir = setup_output_directory(
        output_dataset, output_path, args.overwrite
    )
    
    if output_samples_dir is None:
        logger.error("Failed to setup output directory")
        return {'processed': 0, 'skipped': 0, 'failed': 0, 'total': 0}
    
    # Determine indices to process
    total_samples = len(dataset)
    if args.max_samples is not None:
        total_samples = min(total_samples, args.max_samples)
    
    all_indices = list(range(total_samples))
    
    # Pre-filter already-processed samples if skip-if-exists is enabled
    indices_to_process = all_indices
    pre_skipped_count = 0
    
    if args.skip_if_exists:
        logger.info("Gathering already-processed samples...")
        already_processed = get_already_processed_hash_ids(paraphrases_json_dir)
        logger.info(f"Found {len(already_processed)} already-processed samples")
        
        # Filter indices: only process samples that don't have JSON files yet
        indices_to_process = []
        for idx in all_indices:
            try:
                sample = dataset[idx]
                hash_id = str(sample['hash_id'])
                if hash_id not in already_processed:
                    indices_to_process.append(idx)
                else:
                    pre_skipped_count += 1
            except Exception as e:
                logger.warning(f"Error checking sample at index {idx}: {e}")
                # Include it in processing list if we can't check
                indices_to_process.append(idx)
        
        logger.info(f"Pre-filtered: {pre_skipped_count} already processed, {len(indices_to_process)} remaining to process")
    
    logger.info(f"Processing {len(indices_to_process)} samples")
    logger.info(f"Using {args.max_concurrent} concurrent API requests")
    logger.info(f"Using {args.max_file_concurrent} concurrent file operations")
    
    # Process all samples in parallel (async)
    # Access samples by index to avoid keeping entire dataset in memory
    logger.info("\nPhase 1: Generating paraphrases in parallel...")
    stats = asyncio.run(process_samples_parallel_by_index(
        dataset, indices_to_process, output_samples_dir, raw_responses_dir, 
        paraphrases_json_dir, args
    ))
    
    # Add pre-skipped count to stats
    stats['skipped'] += pre_skipped_count
    stats['total'] = len(all_indices)  # Total includes pre-skipped
    
    # Create final dataset from JSON files
    logger.info("\nPhase 2: Creating final dataset from JSON files...")
    dataset_stats = create_final_dataset_from_json(
        dataset, paraphrases_json_dir, output_samples_dir, args
    )
    
    stats['dataset_created'] = dataset_stats['created']
    stats['dataset_missing_json'] = dataset_stats['missing_json']
    stats['dataset_incorrect_count'] = dataset_stats['incorrect_count']
    stats['dataset_failed'] = dataset_stats['failed']
    
    return stats


def verify_output_dataset(output_dataset, output_path, expected_num_paraphrases):
    """
    Verify that the output dataset is complete and valid.
    
    Args:
        output_dataset: Output dataset name
        output_path: Base output path where datasets are created
        expected_num_paraphrases: Expected number of paraphrases per sample
    
    Returns:
        dict: Verification results
    """
    output_samples_dir = os.path.join(output_path, output_dataset, 'samples')
    
    if not os.path.exists(output_samples_dir):
        return {'total': 0, 'valid': 0, 'invalid': 0, 'missing_paraphrases': []}
    
    sample_files = get_all_sample_files(output_samples_dir)
    
    results = {
        'total': len(sample_files),
        'valid': 0,
        'invalid': 0,
        'incomplete': 0,
        'missing_paraphrases': [],
        'incomplete_files': []
    }
    
    for filepath in sample_files:
        try:
            with np.load(filepath, allow_pickle=True) as data:
                if 'paraphrases' not in data.files:
                    results['invalid'] += 1
                    results['missing_paraphrases'].append(os.path.basename(filepath))
                else:
                    paraphrases = data['paraphrases']
                    if len(paraphrases) < expected_num_paraphrases:
                        results['incomplete'] += 1
                        results['incomplete_files'].append(
                            (os.path.basename(filepath), len(paraphrases))
                        )
                    else:
                        results['valid'] += 1
                    
        except Exception as e:
            results['invalid'] += 1
            results['missing_paraphrases'].append(os.path.basename(filepath))
    
    return results


# ============================================================================
# Main Function
# ============================================================================

def main():
    args = parse_args()
    
    # Validate API key
    api_key = args.openai_api_key or os.getenv("OPENAI_API_KEY")
    if api_key is None:
        logger.error("No OpenAI API key provided. Use --openai-api-key or set OPENAI_API_KEY environment variable.")
        return
    
    # Set output dataset name if not provided
    if args.output_dataset is None:
        # Infer from dataset_path (e.g., ../data/VLCB/raw/test -> test_PE)
        dataset_name = os.path.basename(args.dataset_path.rstrip('/'))
        args.output_dataset = f"{dataset_name}_PE"
    
    logger.info("="*80)
    logger.info("Prompt Ensemble (PE) Paraphrase Generation Script")
    logger.info("="*80)
    logger.info(f"Dataset path: {args.dataset_path}")
    logger.info(f"Output path: {args.output_path}")
    logger.info(f"Output dataset: {args.output_dataset}")
    logger.info(f"Number of paraphrases: {args.num_paraphrases}")
    logger.info(f"GPT model: {args.gpt_model}")
    logger.info(f"Strict count validation: {not args.allow_fewer_paraphrases} (samples with incorrect count will {'fail' if not args.allow_fewer_paraphrases else 'be allowed'})")
    logger.info(f"Skip if exists: {args.skip_if_exists}")
    logger.info(f"Overwrite: {args.overwrite}")
    
    # Process dataset
    stats = process_dataset(
        args.dataset_path,
        args.output_dataset,
        args.output_path,
        args
    )
    
    # Print summary
    logger.info(f"\n{'#'*80}")
    logger.info("Processing Summary:")
    logger.info(f"{'#'*80}")
    logger.info(f"  Total samples: {stats['total']}")
    logger.info(f"  Processed (paraphrases generated): {stats['processed']}")
    logger.info(f"  Skipped (already exists): {stats['skipped']}")
    logger.info(f"  Failed: {stats['failed']}")
    logger.info(f"  Final dataset created: {stats.get('dataset_created', 0)}")
    if stats.get('dataset_missing_json', 0) > 0:
        logger.warning(f"  Missing JSON files: {stats['dataset_missing_json']}")
    if stats.get('dataset_incorrect_count', 0) > 0:
        logger.warning(f"  Samples with incorrect paraphrase count: {stats['dataset_incorrect_count']}")
    if stats.get('dataset_failed', 0) > 0:
        logger.warning(f"  Dataset creation failures: {stats['dataset_failed']}")
    
    # Verify output dataset
    logger.info(f"\n{'#'*80}")
    logger.info("Verifying output dataset...")
    logger.info(f"{'#'*80}")
    verification = verify_output_dataset(
        args.output_dataset,
        args.output_path,
        args.num_paraphrases
    )
    
    logger.info(f"  Total output files: {verification['total']}")
    logger.info(f"  Valid (with {args.num_paraphrases} paraphrases): {verification['valid']}")
    logger.info(f"  Incomplete (< {args.num_paraphrases} paraphrases): {verification['incomplete']}")
    logger.info(f"  Invalid (missing paraphrases): {verification['invalid']}")
    
    if verification['invalid'] > 0:
        logger.warning(f"  Found {verification['invalid']} files without paraphrases!")
        if args.debug and verification['missing_paraphrases']:
            logger.warning(f"  Missing paraphrases in: {verification['missing_paraphrases'][:10]}")
    
    if verification['incomplete'] > 0:
        logger.warning(f"  Found {verification['incomplete']} files with incomplete paraphrases!")
        if args.debug and verification['incomplete_files']:
            for fname, count in verification['incomplete_files'][:10]:
                logger.warning(f"    {fname}: {count}/{args.num_paraphrases}")
    
    if verification['valid'] == verification['total'] and verification['total'] > 0:
        logger.info(f"  ✓ All samples have complete paraphrases!")
    
    # Final summary
    logger.info(f"\n{'='*80}")
    logger.info("Paraphrase Generation Complete - Final Summary")
    logger.info(f"{'='*80}")
    logger.info(f"  Total: {stats['total']}, Processed: {stats['processed']}, "
               f"Skipped: {stats['skipped']}, Failed: {stats['failed']}")
    logger.info(f"\n{'='*80}")
    logger.info(f"Output dataset saved to: {os.path.join(args.output_path, args.output_dataset)}")
    logger.info(f"Each sample now contains 'paraphrases' field with {args.num_paraphrases} variations")
    logger.info(f"{'='*80}")


if __name__ == "__main__":
    main()


# ============================================================================
# Example Usage
# ============================================================================

# python PE_paraphrase_generation.py \
#     --dataset_path "../data/VLCB/raw/test" \
#     --output_path "../data/PE/" \
#     --output_dataset "test_PE" \
#     --num_paraphrases 10 \
#     --gpt-model "gpt-5-mini" \
#     --reasoning-effort "medium" \
#     --max-concurrent 10 \
#     --max_samples 20 \
#     --skip-if-exists

# Test run with few samples:
# python PE_paraphrase_generation.py \
#     --dataset_path "../data/VLCB/raw/test" \
#     --output_path "../data/PE/" \
#     --num_paraphrases 5 \
#     --max-concurrent 5 \
#     --max_samples 3 \
#     --debug