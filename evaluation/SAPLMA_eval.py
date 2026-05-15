#!/usr/bin/env python3
"""
SAPLMA_paper_eval.py

Evaluation script for SAPLMA models trained using SAPLMA_paper_train.py.

Key properties:
- Loads 3 trained runs
- Computes predictions for each run
- Averages confidence scores across runs
- Uses fixed architecture (256,128,64)
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple
import logging

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from datasets import load_from_disk

sys.path.append(str(Path(__file__).parent / "../utils"))
from eval import calculate_all_metrics, save_evaluation_results

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ---------------------------------------------------------
# Model
# ---------------------------------------------------------

class SAPLMAModel(nn.Module):

    def __init__(self, input_dim: int):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_layers = (256,128,64)

        self.classifier = nn.Sequential(
            nn.Linear(input_dim,256),
            nn.ReLU(),
            nn.Linear(256,128),
            nn.ReLU(),
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Linear(64,1)
        )

    def forward(self,x):
        logits = self.classifier(x)
        return logits.squeeze(-1)


# ---------------------------------------------------------
# Evaluator
# ---------------------------------------------------------

class SAPLMAEvaluator:

    def __init__(self, model_dir:Path, data_dir:Path, model_name:str, test_dataset_name:str, dataset_path:Path=None):

        self.model_dir = model_dir
        self.data_dir = data_dir
        self.model_name = model_name
        self.test_dataset_name = test_dataset_name
        self.dataset_path = dataset_path

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.models = self._load_models()

        self.hash_id_to_dataset = {}

        if dataset_path:
            self._load_dataset_mapping()


    # -----------------------------------------------------

    def _load_models(self):
        """Load the best checkpoint produced by SAPLMA_train.py."""
        models = []
        best_dir = self.model_dir / "best"
        if not (best_dir / "model.pth").exists():
            raise RuntimeError(
                f"No trained model found in {best_dir}. "
                f"Run SAPLMA_train.py first.")
        state_dict = torch.load(best_dir / "model.pth", map_location="cpu")
        input_dim = list(state_dict.values())[0].shape[1]
        model = SAPLMAModel(input_dim)
        model.load_state_dict(state_dict)
        model.to(self.device).eval()
        models.append(model)
        logger.info(f"Loaded best/model.pth from {best_dir}")
        return models


    # -----------------------------------------------------

    def _load_dataset_mapping(self):

        try:

            possible_paths = [
                self.dataset_path / self.test_dataset_name,
                self.dataset_path / "raw" / self.test_dataset_name,
                self.dataset_path / self.test_dataset_name.replace("_raw","")
            ]

            dataset_dir = None

            for p in possible_paths:
                if p.exists():
                    dataset_dir = p
                    break

            if dataset_dir is None:
                logger.warning(f"Dataset path not found. Tried: {possible_paths}. Dataset property will not be included.")
                return

            dataset = load_from_disk(str(dataset_dir))
            logger.info(f"Loading dataset mapping from {dataset_dir}")

            # Create hash_id -> dataset mapping
            for idx in range(len(dataset)):
                sample = dataset[idx]
                hash_id = sample.get('hash_id')
                dataset_name = sample.get('dataset', 'unknown')
                if hash_id:
                    # Convert to string to match the format used in load_test_data()
                    self.hash_id_to_dataset[str(hash_id)] = dataset_name

            logger.info(f"Loaded dataset mapping for {len(self.hash_id_to_dataset)} samples")
            if len(self.hash_id_to_dataset) == 0:
                logger.warning("Dataset mapping is empty! Check if dataset has 'hash_id' and 'dataset' fields.")

        except Exception as e:

            logger.warning(f"Failed to load dataset mapping: {e}. Dataset property will not be included.")
            import traceback
            logger.warning(traceback.format_exc())
            self.hash_id_to_dataset = {}


    # -----------------------------------------------------

    def load_test_data(self):

        model_name_part = self.model_name.split("/")[-1]

        samples_dir = self.data_dir / model_name_part / self.test_dataset_name / "samples"

        hidden_states_list=[]
        labels_list=[]
        sample_ids=[]

        files = list(samples_dir.glob("*.npz"))

        logger.info(f"Loading {len(files)} test samples")

        for f in tqdm(files):

            try:

                data = np.load(f,allow_pickle=True)

                hidden = data["hidden_states"][-1]
                label = data["is_correct"]

                if label is None:
                    continue

                # Extract sample ID - handle numpy arrays/scalars
                hash_id_raw = data["hash_id"]
                if isinstance(hash_id_raw, np.ndarray):
                    hash_id = str(hash_id_raw.item() if hash_id_raw.shape == () else hash_id_raw)
                else:
                    hash_id = str(hash_id_raw)

                hidden_states_list.append(hidden)
                labels_list.append(bool(label))
                sample_ids.append(hash_id)

            except Exception:
                continue

        X=np.stack(hidden_states_list)
        y=np.array(labels_list,dtype=bool)

        logger.info(f"Loaded {len(X)} test samples "
                    f"(correct={int(y.sum())}, incorrect={int((~y).sum())})")
        return X,y,sample_ids


    # -----------------------------------------------------

    def predict(self,hidden_states):

        hidden_states = torch.FloatTensor(hidden_states).to(self.device)

        batch_size=32

        run_predictions=[]

        with torch.no_grad():

            for model in self.models:

                preds=[]

                for i in range(0,len(hidden_states),batch_size):

                    batch = hidden_states[i:i+batch_size]

                    logits=model(batch)

                    conf=torch.sigmoid(logits)

                    preds.append(conf.cpu().numpy())

                preds=np.concatenate(preds)

                run_predictions.append(preds)

        run_predictions=np.stack(run_predictions)

        # average across runs
        confidence_scores = run_predictions.mean(axis=0)

        return confidence_scores


    # -----------------------------------------------------

    def evaluate(self,output_dir:Path):

        X,y,ids = self.load_test_data()

        conf = self.predict(X)

        metrics = calculate_all_metrics(y.astype(float),conf)

        records=[]
        matched_count = 0

        for i,sid in enumerate(ids):

            dataset_name=self.hash_id_to_dataset.get(sid,"unknown")
            if dataset_name != 'unknown':
                matched_count += 1

            records.append({
                "sample_id":sid,
                "ground_truth_correctness":int(y[i]),
                "confidence_score":float(conf[i]),
                "dataset":dataset_name
            })

        if matched_count == 0 and len(self.hash_id_to_dataset) > 0:
            logger.warning(f"No hash_id matches found! Mapping has {len(self.hash_id_to_dataset)} entries but none matched.")
            # Log first few sample_ids and mapping keys for debugging
            logger.warning(f"First 5 sample_ids: {ids[:5]}")
            logger.warning(f"First 5 mapping keys: {list(self.hash_id_to_dataset.keys())[:5]}")
        elif matched_count < len(ids) and len(self.hash_id_to_dataset) > 0:
            logger.warning(f"Only {matched_count}/{len(ids)} samples matched dataset mapping.")

        output_dir.mkdir(parents=True,exist_ok=True)

        labels_path = output_dir / "test_labels.json"

        with open(labels_path,"w") as f:
            json.dump(records,f,indent=2)

        results = {
            "overall":{
                "n_samples":len(y),
                **metrics
            },
            "metadata":{
                "model_name":self.model_name,
                "test_dataset_name":self.test_dataset_name,
                "evaluation_timestamp":str(np.datetime64("now"))
            }
        }

        results_path = output_dir / "test_results.json"

        save_evaluation_results(results,results_path)

        logger.info("Evaluation complete")

        return results


# ---------------------------------------------------------
# CLI
# ---------------------------------------------------------

def parse_arguments():

    parser = argparse.ArgumentParser()

    parser.add_argument("--data-dir",type=str,required=True)
    parser.add_argument("--model-name",type=str,required=True)
    parser.add_argument("--test-dataset-name",type=str,required=True)
    parser.add_argument("--train-dataset-name",type=str,required=True)
    parser.add_argument("--trained-model-path",type=str,required=True)
    parser.add_argument("--dataset-path",type=str,default=None)
    parser.add_argument("--output-dir",type=str,required=True)
    parser.add_argument("--seed",type=int,default=None,
                        help="If set: treat --trained-model-path and --output-dir as literal leaves (no {MODEL}/{DATASET} injection).")

    return parser.parse_args()


def main():

    args=parse_arguments()

    model_name_part=args.model_name.split("/")[-1]

    if args.seed is not None:
        trained_model_path = Path(args.trained_model_path)
        output_dir = Path(args.output_dir) / args.test_dataset_name
    else:
        trained_model_path = Path(args.trained_model_path) / model_name_part / args.train_dataset_name
        output_dir = Path(args.output_dir) / model_name_part / args.test_dataset_name

    dataset_path = Path(args.dataset_path) if args.dataset_path else None

    if dataset_path:
        logger.info(f"Dataset path: {dataset_path}")
    else:
        logger.warning("No --dataset-path provided. Dataset property will be 'unknown' for all samples.")

    evaluator = SAPLMAEvaluator(
        model_dir=trained_model_path,
        data_dir=Path(args.data_dir),
        model_name=args.model_name,
        test_dataset_name=args.test_dataset_name,
        dataset_path=dataset_path
    )

    results=evaluator.evaluate(output_dir)

    if results:
        print(f"\nEvaluation complete. Results saved to {output_dir}")
    else:
        print("Evaluation failed")
        sys.exit(1)


if __name__=="__main__":
    main()

# Qwen/Qwen3-VL-8B-Instruct
# llava-hf/llava-v1.6-vicuna-13b-hf
# OpenGVLab/InternVL3_5-14B-HF
# google/gemma-3-27b-it
# deepseek-ai/deepseek-vl2

## Example Usage:
# python SAPLMA_paper_eval.py \
#   --data-dir ../data/extraction/raw/ \
#   --dataset-path ../data/VLCB \
#   --model-name deepseek-ai/deepseek-vl2 \
#   --trained-model-path ../trained_models/SAPLMA_paper \
#   --output-dir ../results/SAPLMA_paper \
#   --test-dataset-name test \
#   --train-dataset-name train \
#   --dataset-path ../data/VLCB