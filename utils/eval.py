#!/usr/bin/env python3
"""
Comprehensive evaluation functions for confidence estimation methods.
Calculates all core metrics with proper zero-division handling.
"""

import numpy as np
from sklearn.metrics import (
    roc_auc_score, brier_score_loss, precision_recall_curve, auc,
    precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
)
from typing import Dict, Any, List
import json


def calculate_ece(y_true: np.ndarray, y_conf: np.ndarray, n_bins: int = 10) -> float:
    """Calculate Expected Calibration Error (ECE)."""
    if len(y_true) == 0:
        return 1.0  # Worst possible ECE
    
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0.0
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (y_conf > bin_lower) & (y_conf <= bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            accuracy_in_bin = y_true[in_bin].mean()
            avg_confidence_in_bin = y_conf[in_bin].mean()
            ece += abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    
    return float(ece)


def calculate_all_metrics(y_true: np.ndarray, y_conf: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    """
    Calculate all evaluation metrics with proper zero-division handling.
    
    Args:
        y_true: Binary ground truth labels (0 or 1)
        y_conf: Confidence scores between 0 and 1
        threshold: Threshold for binary classification
        
    Returns:
        Dictionary containing all metrics
    """
    if len(y_true) == 0:
        return {
            'n_samples': 0,
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0,
            'sensitivity': 0.0,
            'specificity': 0.0,
            'ece': 1.0,
            'brier': 1.0,
            'auroc': 0.5,
            'aucpr': 0.0
        }
    
    # Convert confidence to binary predictions
    y_pred = (y_conf >= threshold).astype(int)
    
    # Basic metrics
    metrics = {
        'n_samples': len(y_true),
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'ece': calculate_ece(y_true, y_conf),
        'brier': float(brier_score_loss(y_true, y_conf))
    }
    
    # Handle edge cases for classification metrics
    if len(np.unique(y_true)) < 2:
        # All samples have the same label
        if np.all(y_true == y_pred):
            metrics.update({
                'precision': 1.0,
                'recall': 1.0,
                'f1': 1.0,
                'sensitivity': 1.0,
                'specificity': 1.0
            })
        else:
            metrics.update({
                'precision': 0.0,  # Worst case
                'recall': 0.0,     # Worst case
                'f1': 0.0,         # Worst case
                'sensitivity': 0.0, # Worst case
                'specificity': 0.0  # Worst case
            })
    else:
        # Calculate with proper zero division handling
        metrics['precision'] = float(precision_score(y_true, y_pred, zero_division=0.0))
        metrics['recall'] = float(recall_score(y_true, y_pred, zero_division=0.0))
        metrics['f1'] = float(f1_score(y_true, y_pred, zero_division=0.0))
        
        # Calculate sensitivity and specificity
        try:
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
            metrics['sensitivity'] = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
            metrics['specificity'] = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        except ValueError:
            metrics['sensitivity'] = 0.0
            metrics['specificity'] = 0.0
    
    # AUC metrics (handle edge cases)
    if len(np.unique(y_true)) < 2:
        metrics['auroc'] = 0.5  # Random performance
        metrics['aucpr'] = float(np.mean(y_true))  # Baseline AUCPR
    else:
        try:
            metrics['auroc'] = float(roc_auc_score(y_true, y_conf))
            precision, recall, _ = precision_recall_curve(y_true, y_conf)
            metrics['aucpr'] = float(auc(recall, precision))
        except Exception:
            metrics['auroc'] = 0.5
            metrics['aucpr'] = float(np.mean(y_true))
    
    return metrics


def evaluate_by_groups(y_true: np.ndarray, y_conf: np.ndarray, 
                      datasets: np.ndarray, categories: np.ndarray) -> Dict[str, Any]:
    """
    Evaluate metrics across all data, by dataset, and by dataset-category combinations.
    
    Args:
        y_true: Binary ground truth labels (0 or 1)
        y_conf: Confidence scores between 0 and 1
        datasets: Dataset labels for each sample
        categories: Category labels for each sample
        
    Returns:
        Dictionary containing results for all groups
    """
    results = {}
    
    # Overall results
    print("Calculating overall metrics...")
    results['overall'] = calculate_all_metrics(y_true, y_conf)
    
    # Results by dataset
    print("Calculating metrics by dataset...")
    results['by_dataset'] = {}
    unique_datasets = np.unique(datasets)
    
    for dataset in unique_datasets:
        dataset_mask = datasets == dataset
        dataset_results = calculate_all_metrics(
            y_true[dataset_mask], 
            y_conf[dataset_mask]
        )
        results['by_dataset'][str(dataset)] = dataset_results
    
    # Results by dataset and category
    print("Calculating metrics by dataset and category...")
    results['by_dataset_category'] = {}
    
    for dataset in unique_datasets:
        dataset_mask = datasets == dataset
        dataset_categories = categories[dataset_mask]
        unique_categories = np.unique(dataset_categories)
        
        results['by_dataset_category'][str(dataset)] = {}
        
        for category in unique_categories:
            category_mask = dataset_categories == category
            full_mask = dataset_mask.copy()
            full_mask[dataset_mask] = category_mask
            
            category_results = calculate_all_metrics(
                y_true[full_mask],
                y_conf[full_mask]
            )
            results['by_dataset_category'][str(dataset)][str(category)] = category_results
    
    return results


def save_evaluation_results(results: Dict[str, Any], output_path: str) -> None:
    """Save evaluation results to JSON file."""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Evaluation results saved to {output_path}")


if __name__ == "__main__":
    # Example usage
    np.random.seed(23)
    
    # Generate test data
    n_samples = 1000
    y_true = np.random.binomial(1, 0.6, n_samples)
    y_conf = np.random.uniform(0.2, 0.9, n_samples)
    datasets = np.random.choice(['dataset1', 'dataset2'], n_samples)
    categories = np.random.choice(['cat1', 'cat2', 'cat3'], n_samples)
    
    # Evaluate
    results = evaluate_by_groups(y_true, y_conf, datasets, categories)
    
    # Print overall results
    print("Overall Results:")
    for metric, value in results['overall'].items():
        print(f"  {metric}: {value:.4f}")
    
    # Save results
    save_evaluation_results(results, 'test_evaluation_results.json')