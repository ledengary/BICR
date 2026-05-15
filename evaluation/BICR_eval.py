#!/usr/bin/env python3
"""
BICR (Blind-Image Contrastive Ranking) — Evaluation Script

Loads a trained BICR checkpoint and evaluates on test data.
Saves test_results.json + test_labels.json to results/SPARROW/BICR/

Usage:
    python evaluation/BICR_eval.py --gpu 0 --model-name Qwen/Qwen3-VL-8B-Instruct --seed 23
    python evaluation/BICR_eval.py --gpu 0 --model-name Qwen/Qwen3-VL-8B-Instruct --seed 23 --ablation no_rank
"""
import sys, os, json, argparse, numpy as np, torch, torch.nn as nn
from pathlib import Path
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'utils'))
from eval import calculate_all_metrics, save_evaluation_results

ROOT = Path(__file__).resolve().parent.parent


def parse_args():
    parser = argparse.ArgumentParser(description='BICR Evaluation')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--model-name', type=str, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--data-dir', type=str, default=None)
    parser.add_argument('--checkpoint-dir', type=str, default=None)
    parser.add_argument('--results-dir', type=str, default=None)
    parser.add_argument('--test-dataset', type=str, default='test')
    parser.add_argument('--ablation', type=str, default=None,
                        choices=['no_rank', 'no_brier', 'bce_only'])
    return parser.parse_args()


# ============================================================================
# Model (must match BICR_train.py exactly)
# ============================================================================

class BICRModel(nn.Module):
    def __init__(self, h_dim, clf_layers_str, dropout=0.0):
        super().__init__()
        if clf_layers_str == "0":
            self.net = nn.Linear(h_dim, 1)
        else:
            hidden = [int(x) for x in clf_layers_str.split(',') if x.strip()]
            dims = [h_dim] + hidden
            blocks = []
            for i in range(len(dims) - 1):
                blocks += [nn.Linear(dims[i], dims[i + 1]), nn.ReLU()]
                if dropout > 0:
                    blocks.append(nn.Dropout(dropout))
            blocks.append(nn.Linear(dims[-1], 1))
            self.net = nn.Sequential(*blocks)

    def forward(self, h):
        return self.net(h).squeeze(-1)


# ============================================================================
# Data loading
# ============================================================================

def load_test_split(data_dir, model_short, split):
    """Load h_base + labels + sample_ids for test evaluation.

    Labels are stored as 1 = correct, 0 = incorrect; this convention is fixed
    across training and evaluation.
    """
    sdir = data_dir / model_short / split / 'samples'
    files = sorted(f for f in os.listdir(sdir) if f.endswith('.npz'))
    h_base_l, labels, sample_ids = [], [], []
    for f in tqdm(files, desc=f'Loading {split}', leave=False):
        d = dict(np.load(sdir / f, allow_pickle=True))
        ic = d.get('is_correct')
        if hasattr(ic, 'item'): ic = ic.item()
        if ic is None: continue
        h_base_l.append(d['h_base'].astype(np.float32))
        labels.append(int(bool(ic)))
        sid = d.get('sample_id', f[:-4])
        if hasattr(sid, 'item'): sid = str(sid.item())
        elif isinstance(sid, np.ndarray): sid = str(sid)
        sample_ids.append(sid)
    labels_arr = np.array(labels, dtype=np.float32)
    return {
        'h_base': np.stack(h_base_l),
        'labels': labels_arr,
        'sample_ids': sample_ids,
    }


# ============================================================================
# Main
# ============================================================================

def main():
    args = parse_args()
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    model_short = args.model_name.split('/')[-1]
    data_dir = Path(args.data_dir) if args.data_dir else ROOT / 'data' / 'extraction' / 'BICR'

    # Checkpoint path
    if args.checkpoint_dir:
        ckpt_dir = Path(args.checkpoint_dir)
    elif args.ablation:
        ckpt_dir = ROOT / 'trained_models' / 'SPARROW' / 'BICR_abl' / args.ablation / model_short / f'seed_{args.seed}' / 'best'
    else:
        ckpt_dir = ROOT / 'trained_models' / 'SPARROW' / 'BICR' / model_short / f'seed_{args.seed}' / 'best'

    # Results path
    if args.results_dir:
        results_dir = Path(args.results_dir)
    elif args.ablation:
        results_dir = ROOT / 'results' / 'SPARROW' / 'BICR_abl' / args.ablation / model_short / f'seed_{args.seed}' / args.test_dataset
    else:
        results_dir = ROOT / 'results' / 'SPARROW' / 'BICR' / model_short / f'seed_{args.seed}' / args.test_dataset

    # Skip if already done
    if (results_dir / 'test_results.json').exists() and (results_dir / 'test_labels.json').exists():
        logger.info(f"SKIP: results exist at {results_dir}")
        return

    # Check checkpoint exists
    if not (ckpt_dir / 'model.pth').exists():
        logger.error(f"Checkpoint not found: {ckpt_dir / 'model.pth'}")
        return
    if not (ckpt_dir / 'model_info.json').exists():
        logger.error(f"Model info not found: {ckpt_dir / 'model_info.json'}")
        return

    # Load model info
    with open(ckpt_dir / 'model_info.json') as f:
        model_info = json.load(f)

    input_dim = model_info['input_dim']
    clf_layers = model_info['classifier_layers']
    dropout = model_info['dropout']

    logger.info(f"BICR Eval — {model_short}, seed {args.seed}")
    logger.info(f"  Checkpoint: {ckpt_dir}")
    logger.info(f"  Architecture: {input_dim} -> {clf_layers} -> 1, dropout={dropout}")
    logger.info(f"  Params: {model_info['n_parameters']:,}")

    # Load model
    model = BICRModel(input_dim, clf_layers, dropout)
    state_dict = torch.load(ckpt_dir / 'model.pth', map_location='cpu', weights_only=True)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    te = load_test_split(data_dir, model_short, args.test_dataset)
    logger.info(f"  Test samples: {len(te['labels'])} "
                f"(correct={int(te['labels'].sum())}, "
                f"incorrect={int((te['labels'] == 0).sum())})")

    # Evaluate
    with torch.no_grad():
        scores = torch.sigmoid(
            model(torch.FloatTensor(te['h_base']).to(device))
        ).cpu().numpy()

    metrics = calculate_all_metrics(te['labels'], scores)

    logger.info(f"  AUROC={metrics['auroc']:.4f} AUCPR={metrics['aucpr']:.4f} "
                f"ECE={metrics['ece']:.4f} Brier={metrics['brier']:.4f}")

    # Save results
    results_dir.mkdir(parents=True, exist_ok=True)

    results_obj = {
        'overall': {**metrics, 'n_samples': len(te['labels'])},
        'metadata': {
            'method': f'BICR{"/" + args.ablation if args.ablation else ""}',
            'model_name': args.model_name,
            'model_name_short': model_short,
            'seed': args.seed,
            'test_dataset': args.test_dataset,
            'checkpoint_dir': str(ckpt_dir),
            'n_parameters': model_info['n_parameters'],
            'classifier_layers': clf_layers,
            'config': model_info['config'],
            'ablation': args.ablation,
        }
    }
    save_evaluation_results(results_obj, str(results_dir / 'test_results.json'))

    test_labels = [
        {'sample_id': te['sample_ids'][i],
         'ground_truth_correctness': int(te['labels'][i]),
         'confidence_score': float(scores[i])}
        for i in range(len(scores))
    ]
    json.dump(test_labels, open(results_dir / 'test_labels.json', 'w'))

    logger.info(f"  Results saved to {results_dir}")


if __name__ == '__main__':
    main()
