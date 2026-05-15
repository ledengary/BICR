"""
Correctness Labeling Script for VLM Extraction Results
Adds GPT-based correctness labels to already-extracted .npz files.
"""

import os
import argparse
import numpy as np
import json
import logging
from pathlib import Path
from tqdm import tqdm
from openai import OpenAI
import traceback
import base64
import io
from PIL import Image
from datasets import load_from_disk
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
    ]
)
logger = logging.getLogger(__name__)

# ============================================================================
# Argument Parser
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description='Add GPT correctness labels to extracted VLM representations'
    )
    
    # Model configuration
    parser.add_argument('--model_names', type=str, nargs='+', required=True,
                        help='Model directory names to process (e.g., Qwen3-VL-8B-Instruct)')
    
    # Data configuration
    parser.add_argument('--base_path', type=str, required=True,
                        help='Base path to extraction results (e.g., ../../data/extraction/raw/)')
    parser.add_argument('--target_datasets', type=str, nargs='+', required=True,
                        help='List of target dataset names to process (e.g., train validation)')
    
    # GPT Judge configuration
    parser.add_argument('--openai-api-key', type=str, default=None,
                        help='OpenAI API key for GPT correctness judge (if not provided, uses OPENAI_API_KEY environment variable)')
    parser.add_argument('--gpt-model', type=str, default='gpt-5-mini',
                        help='GPT model to use for correctness assessment')
    parser.add_argument('--reasoning-effort', type=str, default='low',
                        choices=['low', 'medium', 'high'],
                        help='Reasoning effort for GPT model')
    parser.add_argument('--assess-with-image', action='store_true',
                        help='Include image in GPT correctness assessment (requires --dataset-path)')
    parser.add_argument('--image-assessment-detail', type=str, default='auto',
                        choices=['low', 'high', 'auto'],
                        help='Image detail level for GPT assessment (low/high/auto, default: auto)')
    parser.add_argument('--dataset-path', type=str, default=None,
                        help='Base path to raw datasets directory (required if --assess-with-image is set, e.g., ../../data/VLCB/raw/)')
    
    # Processing configuration
    parser.add_argument('--skip-if-labeled', action='store_true',
                        help='Skip samples that already have correctness labels')
    parser.add_argument('--verify-all-labeled', action='store_true',
                        help='After labeling, verify all samples have labels')
    parser.add_argument('--n-parallel', type=int, default=1,
                        help='Number of parallel requests to make (default: 1 for sequential processing)')
    
    # Debug configuration
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug mode with detailed printing')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Maximum number of samples to process per dataset (for testing)')
    
    return parser.parse_args()


# ============================================================================
# Dataset Loading for Image Access
# ============================================================================

def load_vlcb_dataset(dataset_path, dataset_name):
    """
    Load VLCB dataset and create hash_id to index mapping.
    
    Args:
        dataset_path: Base path to datasets directory
        dataset_name: Dataset name (e.g., train)
    
    Returns:
        tuple: (dataset, hash_id_to_idx dict)
    """
    full_path = os.path.join(dataset_path, dataset_name)
    logger.info(f"Loading VLCB dataset from: {full_path}")
    dataset = load_from_disk(full_path)
    
    # Create hash_id to index mapping
    hash_id_to_idx = {}
    for idx in range(len(dataset)):
        hash_id = dataset[idx]['hash_id']
        hash_id_to_idx[str(hash_id)] = idx
    
    logger.info(f"Loaded {len(dataset)} samples, created hash_id mapping")
    return dataset, hash_id_to_idx


def get_image_by_hash_id(hash_id, dataset, hash_id_to_idx, image_column='image'):
    """
    Get image from dataset by hash_id.
    
    Args:
        hash_id: Hash ID string
        dataset: Loaded dataset
        hash_id_to_idx: Dictionary mapping hash_id to dataset index
        image_column: Column name for images (default: 'image')
    
    Returns:
        PIL Image or None if not found
    """
    hash_id_str = str(hash_id)
    idx = hash_id_to_idx.get(hash_id_str)
    
    if idx is None:
        return None
    
    try:
        image = dataset[idx][image_column]
        # Ensure it's a PIL Image
        if isinstance(image, Image.Image):
            return image
        elif isinstance(image, bytes):
            return Image.open(io.BytesIO(image))
        elif isinstance(image, dict) and 'bytes' in image:
            return Image.open(io.BytesIO(image['bytes']))
        else:
            logger.warning(f"Unknown image format for hash_id {hash_id_str}")
            return None
    except Exception as e:
        logger.error(f"Error loading image for hash_id {hash_id_str}: {e}")
        return None


def convert_image_to_data_url(image):
    """
    Convert PIL Image to base64 data URL.
    
    Args:
        image: PIL Image object
    
    Returns:
        str: Data URL (data:image/png;base64,...)
    """
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"


# ============================================================================
# GPT Correctness Assessment
# ============================================================================

def create_openai_client(openai_api_key=None):
    """
    Create a shared OpenAI client instance.
    
    Args:
        openai_api_key: OpenAI API key (if None, uses environment variable)
    
    Returns:
        OpenAI client instance, or None if no API key available
    """
    if openai_api_key is None:
        openai_api_key = os.getenv("OPENAI_API_KEY")
    
    if openai_api_key is None:
        logger.error("No OpenAI API key provided. Cannot create client.")
        return None
    
    return OpenAI(api_key=openai_api_key)


def assess_correctness_with_gpt(question, ground_truth_answer, generated_response, 
                                client, gpt_model='gpt-5-mini', 
                                reasoning_effort='low', image=None, image_detail='auto',
                                debug=False):
    """
    Queries GPT to assess whether the generated response is correct compared to ground truth.
    
    Args:
        question: The question string
        ground_truth_answer: The ground truth answer string
        generated_response: The student's generated response string
        client: Shared OpenAI client instance
        gpt_model: GPT model to use
        reasoning_effort: Reasoning effort level
        image: Optional PIL Image object to include in assessment
        image_detail: Image detail level ('low', 'high', or 'auto')
        debug: Whether to print debug information
    
    Returns:
        bool: True if the generated response is correct, False otherwise, None on error
    """
    if client is None:
        logger.error("No OpenAI client provided. Cannot assess correctness.")
        return None
    
    # Choose system prompt based on whether image is provided
    if image is not None:
        # System prompt with image context
        system_prompt = """You are an expert answer evaluator. Your task is to determine if a student's answer to a question is correct by comparing it to the ground truth answer. The student was asked the question based on an attached image, which is provided to you.

1. Read the question carefully and consider all provided context, including the image.
2. Compare the student's answer to the ground truth answer.
3. Judge correctness based on practical semantic equivalence as a human would. Answers should be marked correct if they convey the same essential meaning or would reasonably be accepted, even if they differ in specificity, category boundaries, or everyday terminology.
4. Return ONLY "yes" if the answer is correct, or "no" if it is incorrect.
5. Be lenient with minor variations in wording, capitalization, punctuation, or reasonable interpretation.

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
    else:
        # System prompt without image context
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
    
    # Prepare text content
    text_content = f"Question: {question}\n\nGround Truth Answer: {ground_truth_answer}\n\nStudent Answer: {generated_response}\n\nIs the student's answer correct? (yes/no):"
    
    # Prepare messages - use list format if image is present, string format otherwise
    if image is not None:
        # Convert image to data URL
        image_data_url = convert_image_to_data_url(image)
        
        # Use list format with image
        # Note: For OpenAI responses API, image_url should be a string, not an object
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": text_content
                    },
                    {
                        "type": "input_image",
                        "image_url": image_data_url
                    }
                ]
            }
        ]
    else:
        # Use string format without image
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text_content}
        ]
    
    try:
        # Make OpenAI API call
        response = client.responses.create(
            model=gpt_model,
            reasoning={"effort": reasoning_effort},
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
        if debug:
            logger.error(traceback.format_exc())
        return None


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
    # Use context manager to ensure file is closed
    with np.load(filepath, allow_pickle=True) as data:
        # Convert to regular dict
        sample = {}
        for key in data.files:
            sample[key] = data[key]
            
            # Convert numpy scalars to Python types
            if isinstance(sample[key], np.ndarray) and sample[key].shape == ():
                sample[key] = sample[key].item()
    
    return sample


def update_sample_npz(filepath, is_correct):
    """
    Update an existing .npz file with correctness label.
    Preserves all existing fields and adds/updates is_correct.
    
    Args:
        filepath: Path to .npz file
        is_correct: Boolean correctness label (or None)
    """
    # Load existing data with context manager to ensure file is closed
    with np.load(filepath, allow_pickle=True) as data:
        # Convert to dict
        save_dict = {}
        for key in data.files:
            save_dict[key] = data[key]
    
    # Update is_correct field
    save_dict['is_correct'] = is_correct
    
    # Save back (overwrite original file)
    np.savez_compressed(filepath, **save_dict)


def check_has_correctness_label(filepath):
    """
    Check if a sample .npz file already has a correctness label.
    
    Args:
        filepath: Path to .npz file
    
    Returns:
        bool: True if has label, False otherwise
    """
    try:
        # Use context manager to ensure file is closed
        with np.load(filepath, allow_pickle=True) as data:
            # Check if is_correct field exists
            if 'is_correct' not in data.files:
                return False
            
            # Check if is_correct is not None
            is_correct = data['is_correct']
            if isinstance(is_correct, np.ndarray) and is_correct.shape == ():
                is_correct = is_correct.item()
            
            return is_correct is not None
        
    except Exception as e:
        logger.error(f"Error checking correctness label in {filepath}: {e}")
        return False


# ============================================================================
# Dataset Processing
# ============================================================================

def process_single_sample(filepath, args, client, dataset=None, hash_id_to_idx=None):
    """
    Process a single sample file.
    
    Args:
        filepath: Path to the .npz sample file
        args: Command line arguments
        client: Shared OpenAI client instance
        dataset: Optional dataset for image access
        hash_id_to_idx: Optional hash_id to index mapping
    
    Returns:
        dict: Result with keys 'status' ('labeled', 'skipped', 'failed'), 'hash_id', 'error' (if failed)
    """
    try:
        # Check if already labeled (if skip flag is set)
        if args.skip_if_labeled and check_has_correctness_label(filepath):
            if args.debug:
                logger.info(f"Skipping already labeled: {os.path.basename(filepath)}")
            return {'status': 'skipped', 'hash_id': os.path.basename(filepath)}
        
        # Load sample
        sample = load_sample_npz(filepath)
        
        # Extract required fields
        question = str(sample['question'])
        answer = str(sample['answer'])
        generated_response = str(sample['generated_response'])
        
        # Get hash_id for logging
        hash_id = str(sample.get('hash_id', os.path.basename(filepath)))
        
        if args.debug:
            logger.info(f"\nProcessing: {hash_id}")
            logger.info(f"Question: {question[:100]}...")
            logger.info(f"Answer: {answer[:100]}...")
            logger.info(f"Generated: {generated_response[:100]}...")
        
        # Get image if image assessment is enabled
        image = None
        if args.assess_with_image and dataset is not None and hash_id_to_idx is not None:
            image = get_image_by_hash_id(hash_id, dataset, hash_id_to_idx)
            if image is None:
                logger.warning(f"Could not load image for hash_id {hash_id}, proceeding without image")
        
        # Assess correctness with GPT (using shared client)
        is_correct = assess_correctness_with_gpt(
            question,
            answer,
            generated_response,
            client,
            args.gpt_model,
            args.reasoning_effort,
            image=image,
            image_detail=args.image_assessment_detail,
            debug=args.debug
        )
        
        if is_correct is None:
            logger.warning(f"Failed to assess correctness for {hash_id}")
            return {'status': 'failed', 'hash_id': hash_id, 'error': 'GPT assessment returned None'}
        
        # Update the .npz file
        update_sample_npz(filepath, is_correct)
        
        if args.debug:
            logger.info(f"Labeled as: {'CORRECT' if is_correct else 'INCORRECT'}")
        
        return {'status': 'labeled', 'hash_id': hash_id}
    
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Error processing {filepath}: {error_msg}")
        if args.debug:
            logger.error(traceback.format_exc())
        hash_id = os.path.basename(filepath)
        return {'status': 'failed', 'hash_id': hash_id, 'error': error_msg}


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


def process_dataset(model_name, dataset_name, base_path, args):
    """
    Process a single dataset for a single model.
    
    Args:
        model_name: Model directory name
        dataset_name: Dataset directory name
        base_path: Base path to extraction results
        args: Command line arguments
    
    Returns:
        dict: Statistics about processing (labeled, skipped, failed)
    """
    # Construct path to samples directory
    dataset_path = os.path.join(base_path, model_name, dataset_name)
    samples_dir = os.path.join(dataset_path, 'samples')
    
    if not os.path.exists(samples_dir):
        logger.warning(f"Samples directory not found: {samples_dir}")
        return {'labeled': 0, 'skipped': 0, 'failed': 0, 'total': 0}
    
    # Get all sample files
    sample_files = get_all_sample_files(samples_dir)
    
    if len(sample_files) == 0:
        logger.warning(f"No sample files found in: {samples_dir}")
        return {'labeled': 0, 'skipped': 0, 'failed': 0, 'total': 0}
    
    logger.info(f"Found {len(sample_files)} samples in {samples_dir}")
    
    # Load dataset and create hash_id mapping if image assessment is enabled
    dataset = None
    hash_id_to_idx = None
    if args.assess_with_image:
        if args.dataset_path is None:
            logger.error("--dataset-path is required when --assess-with-image is set")
            return {'labeled': 0, 'skipped': 0, 'failed': 0, 'total': 0}
        
        try:
            dataset, hash_id_to_idx = load_vlcb_dataset(args.dataset_path, dataset_name)
        except Exception as e:
            logger.error(f"Failed to load dataset for image access: {e}")
            logger.error(traceback.format_exc())
            return {'labeled': 0, 'skipped': 0, 'failed': 0, 'total': 0}
    
    # Limit samples if specified
    if args.max_samples is not None:
        sample_files = sample_files[:args.max_samples]
        logger.info(f"Limited to {len(sample_files)} samples")
    
    # Statistics
    stats = {
        'labeled': 0,
        'skipped': 0,
        'failed': 0,
        'total': len(sample_files)
    }
    
    # Create a single shared OpenAI client (reuses HTTP connection pool across threads)
    client = create_openai_client(args.openai_api_key)
    if client is None:
        logger.error("Failed to create OpenAI client. Aborting.")
        return stats
    
    # Process samples sequentially or in parallel based on n_parallel
    if args.n_parallel == 1:
        # Sequential processing (original behavior)
        iterator = tqdm(sample_files, desc=f"Labeling {model_name}/{dataset_name}")
        
        for filepath in iterator:
            result = process_single_sample(filepath, args, client, dataset, hash_id_to_idx)
            stats[result['status']] += 1
    else:
        # Parallel processing with bounded concurrency
        # Submit in batches to avoid opening too many files at once
        logger.info(f"Processing {len(sample_files)} samples with {args.n_parallel} parallel workers")
        
        with tqdm(total=len(sample_files), desc=f"Labeling {model_name}/{dataset_name}") as pbar:
            with ThreadPoolExecutor(max_workers=args.n_parallel) as executor:
                # Only keep at most n_parallel * 2 futures in flight at a time
                # to bound the number of open file descriptors
                pending = set()
                file_iter = iter(sample_files)
                max_in_flight = args.n_parallel * 2
                
                # Seed the initial batch
                for filepath in file_iter:
                    pending.add(
                        executor.submit(process_single_sample, filepath, args, client, dataset, hash_id_to_idx)
                    )
                    if len(pending) >= max_in_flight:
                        break
                
                # As futures complete, submit new ones
                while pending:
                    done, pending = wait_first(pending)
                    for future in done:
                        result = future.result()
                        stats[result['status']] += 1
                        pbar.update(1)
                        
                        # Submit next task if available
                        try:
                            next_filepath = next(file_iter)
                            pending.add(
                                executor.submit(process_single_sample, next_filepath, args, client, dataset, hash_id_to_idx)
                            )
                        except StopIteration:
                            pass
    
    return stats


def wait_first(pending):
    """
    Wait for at least one future to complete and return (done, still_pending).
    
    Args:
        pending: Set of futures
    
    Returns:
        tuple: (set of completed futures, set of still-pending futures)
    """
    done, still_pending = wait(pending, return_when=FIRST_COMPLETED)
    return done, still_pending


def verify_all_labeled(model_name, dataset_name, base_path):
    """
    Verify that all samples in a dataset have correctness labels.
    
    Args:
        model_name: Model directory name
        dataset_name: Dataset directory name
        base_path: Base path to extraction results
    
    Returns:
        dict: Verification results (total, labeled, unlabeled)
    """
    samples_dir = os.path.join(base_path, model_name, dataset_name, 'samples')
    
    if not os.path.exists(samples_dir):
        return {'total': 0, 'labeled': 0, 'unlabeled': 0, 'unlabeled_files': []}
    
    sample_files = get_all_sample_files(samples_dir)
    
    unlabeled_files = []
    labeled_count = 0
    
    for filepath in sample_files:
        if check_has_correctness_label(filepath):
            labeled_count += 1
        else:
            unlabeled_files.append(os.path.basename(filepath))
    
    return {
        'total': len(sample_files),
        'labeled': labeled_count,
        'unlabeled': len(unlabeled_files),
        'unlabeled_files': unlabeled_files
    }


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
    
    # Validate arguments
    if args.assess_with_image and args.dataset_path is None:
        logger.error("--dataset-path is required when --assess-with-image is set")
        return
    
    logger.info("="*80)
    logger.info("VLM Correctness Labeling Script")
    logger.info("="*80)
    logger.info(f"Models: {args.model_names}")
    logger.info(f"Base path: {args.base_path}")
    logger.info(f"Target datasets: {args.target_datasets}")
    logger.info(f"GPT model: {args.gpt_model}")
    logger.info(f"Reasoning effort: {args.reasoning_effort}")
    logger.info(f"Assess with image: {args.assess_with_image}")
    if args.assess_with_image:
        logger.info(f"Image assessment detail: {args.image_assessment_detail}")
        logger.info(f"Dataset path: {args.dataset_path}")
    logger.info(f"Parallel workers: {args.n_parallel}")
    logger.info(f"Skip if labeled: {args.skip_if_labeled}")
    logger.info(f"Verify all labeled: {args.verify_all_labeled}")
    
    # Process each model and dataset combination
    all_stats = {}
    
    for model_name in args.model_names:
        logger.info(f"\n{'#'*80}")
        logger.info(f"Processing model: {model_name}")
        logger.info(f"{'#'*80}")
        
        model_stats = {}
        
        for dataset_name in args.target_datasets:
            logger.info(f"\n{'='*80}")
            logger.info(f"Processing dataset: {dataset_name}")
            logger.info(f"{'='*80}")
            
            # Process dataset
            stats = process_dataset(model_name, dataset_name, args.base_path, args)
            model_stats[dataset_name] = stats
            
            # Print summary
            logger.info(f"\nDataset Summary for {dataset_name}:")
            logger.info(f"  Total samples: {stats['total']}")
            logger.info(f"  Labeled: {stats['labeled']}")
            logger.info(f"  Skipped (already labeled): {stats['skipped']}")
            logger.info(f"  Failed: {stats['failed']}")
            
            # Verify if requested
            if args.verify_all_labeled:
                logger.info(f"\nVerifying all samples are labeled...")
                verification = verify_all_labeled(model_name, dataset_name, args.base_path)
                
                logger.info(f"Verification results:")
                logger.info(f"  Total samples: {verification['total']}")
                logger.info(f"  Labeled: {verification['labeled']}")
                logger.info(f"  Unlabeled: {verification['unlabeled']}")
                
                if verification['unlabeled'] > 0:
                    logger.warning(f"Found {verification['unlabeled']} unlabeled samples!")
                    if args.debug:
                        logger.warning(f"Unlabeled files: {verification['unlabeled_files'][:10]}")
                else:
                    logger.info("✓ All samples have correctness labels!")
        
        all_stats[model_name] = model_stats
    
    # Final summary
    logger.info(f"\n{'='*80}")
    logger.info("Labeling Complete - Final Summary")
    logger.info(f"{'='*80}")
    
    for model_name, model_stats in all_stats.items():
        logger.info(f"\nModel: {model_name}")
        for dataset_name, stats in model_stats.items():
            logger.info(f"  {dataset_name}:")
            logger.info(f"    Total: {stats['total']}, Labeled: {stats['labeled']}, "
                       f"Skipped: {stats['skipped']}, Failed: {stats['failed']}")
    
    logger.info(f"\n{'='*80}")
    logger.info("All processing complete!")
    logger.info(f"{'='*80}")


if __name__ == "__main__":
    main()

# Qwen/Qwen3-VL-8B-Instruct
# llava-hf/llava-v1.6-vicuna-13b-hf - 13B
# OpenGVLab/InternVL3_5-14B-HF
# google/gemma-3-27b-it - 27B
# deepseek-ai/deepseek-vl2 - 27B

# Example Usage (without image):
# python correctness_labeling.py \
#     --model_names "Qwen3-VL-8B-Instruct" \
#     --base_path "../../data/extraction/raw/" \
#     --target_datasets "train" "validation" "test" \
#     --gpt-model "gpt-5-mini" \
#     --reasoning-effort "low" \
#     --skip-if-labeled \
#     --verify-all-labeled \
#     --max_samples 5 \
#     --debug

# Example Usage (with image):
# python correctness_labeling.py \
#     --model_names "InternVL3_5-14B-HF" \
#     --base_path "../../data/generation_extraction_v3/" \
#     --target_datasets "train" "validation" "test" \
#     --gpt-model "gpt-5-mini" \
#     --reasoning-effort "low" \
#     --assess-with-image \
#     --image-assessment-detail "auto" \
#     --dataset-path "../../data/VLCB/raw/" \
#     --skip-if-labeled \
#     --verify-all-labeled \
#     --n-parallel 10 \
#     --max_samples 5 \
#     --debug