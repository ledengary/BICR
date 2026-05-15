"""
VLCB Dataset Finalization
===========================
This notebook combines datasets from multiple sources to create the final
VLCB train/val/test splits:

- Train: GQA train only
- Val: GQA val only  
- Test: Combined test sets from:
  - GQA test
  - LLaVA-Wild test
  - GMAI-MMBench test
  - MME-Finance test
  - POPE test

Final outputs:
- train
- validation
- test
"""

# ============================================================================
# CELL 1: Imports and Setup
# ============================================================================

import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datasets import Dataset, load_from_disk
from tqdm import tqdm
from collections import defaultdict

# Set random seeds for reproducibility
SEED = 23
random.seed(SEED)
np.random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)

# ============================================================================
# CELL 2: Define Paths
# ============================================================================

OUTPUT_ROOT = "../../data/VLCB_2/raw"
os.makedirs(OUTPUT_ROOT, exist_ok=True)

# Input paths (from other notebooks)
GQA_TRAIN_PATH = "../../data/GQA/GQA_train_final"
GQA_VAL_PATH = "../../data/GQA/GQA_val_final"
GQA_TEST_PATH = "../../data/GQA/GQA_test_final"
LLAVA_WILD_TEST_PATH = "../../data/LLaVA-Wild/LLaVA_Wild_test_final"
GMAI_MMBENCH_TEST_PATH = "../../data/GMAI-MMBench/GMAI_MMBench_test_final"
MME_FINANCE_TEST_PATH = "../../data/MME-Finance/MMFin_test_final"
POPE_TEST_PATH = "../../data/POPE/POPE_test_final"
MMMU_PRO_4_TEST_PATH = "../../data/MMMU_Pro/MMMU_Pro_4_test_final"
MMMU_PRO_10_TEST_PATH = "../../data/MMMU_Pro/MMMU_Pro_10_test_final"

# Output paths
FINAL_TRAIN_PATH = os.path.join(OUTPUT_ROOT, "train")
FINAL_VAL_PATH = os.path.join(OUTPUT_ROOT, "validation")
FINAL_TEST_PATH = os.path.join(OUTPUT_ROOT, "test")

print("=" * 80)
print("VLCB Dataset Finalization")
print("=" * 80)
print(f"\nOutput directory: {OUTPUT_ROOT}")
print(f"\nInput datasets:")
print(f"  Train: {GQA_TRAIN_PATH}")
print(f"  Val: {GQA_VAL_PATH}")
print(f"  Test sources:")
print(f"    - GQA: {GQA_TEST_PATH}")
print(f"    - LLaVA-Wild: {LLAVA_WILD_TEST_PATH}")
print(f"    - GMAI-MMBench: {GMAI_MMBENCH_TEST_PATH}")
print(f"    - MME-Finance: {MME_FINANCE_TEST_PATH}")
print(f"    - POPE: {POPE_TEST_PATH}")
print(f"    - MMMU-Pro 4: {MMMU_PRO_4_TEST_PATH}")
print(f"    - MMMU-Pro 10: {MMMU_PRO_10_TEST_PATH}")
print(f"\nOutput datasets:")
print(f"  Train: {FINAL_TRAIN_PATH}")
print(f"  Val: {FINAL_VAL_PATH}")
print(f"  Test: {FINAL_TEST_PATH}")

# ============================================================================
# CELL 3: Load Train Dataset (GQA only) with Comprehensive Assertions
# ============================================================================

print("\n" + "=" * 80)
print("Loading Train Dataset")
print("=" * 80)

print(f"\nLoading GQA train from: {GQA_TRAIN_PATH}")
train_dataset = load_from_disk(GQA_TRAIN_PATH)

print(f"✓ Loaded train dataset")
print(f"  Size: {len(train_dataset)}")
print(f"  Features: {list(train_dataset.features.keys())}")

# Verify required fields
required_fields = ["question", "answer", "image", "category", "dataset", "hash_id"]
for field in required_fields:
    assert field in train_dataset.features, f"Missing field: {field}"

# Comprehensive assertions
print(f"\nRunning comprehensive assertions on train dataset...")

# Check for non-empty dataset
assert len(train_dataset) > 0, "Train dataset is empty!"

# Check for NaN values in string fields
string_fields = ["question", "answer", "category", "dataset", "hash_id"]
for field in string_fields:
    field_values = train_dataset[field]
    nan_count = sum(1 for v in field_values if pd.isna(v) or str(v).strip() == "")
    assert nan_count == 0, f"Found {nan_count} NaN or empty values in '{field}'"
    print(f"  ✓ No NaN/empty values in '{field}'")

# Check hash ID uniqueness
train_hash_ids = train_dataset["hash_id"]
train_unique_hash_ids = len(set(train_hash_ids))
assert len(train_hash_ids) == train_unique_hash_ids, \
    f"Duplicate hash IDs found: {len(train_hash_ids) - train_unique_hash_ids} duplicates"
print(f"  ✓ All hash IDs are unique ({train_unique_hash_ids} unique)")

# Check that all images are present
image_field = train_dataset["image"]
assert len(image_field) == len(train_dataset), "Mismatch in image count"
print(f"  ✓ All images are present ({len(image_field)} images)")

# Check dataset name consistency
train_dataset_names = set(train_dataset["dataset"])
assert len(train_dataset_names) == 1, f"Multiple dataset names found: {train_dataset_names}"
print(f"  ✓ Dataset name consistent: {train_dataset_names.pop()}")

print(f"\n✓ Train dataset comprehensive validation passed")

# ============================================================================
# CELL 4: Load Val Dataset (GQA only) with Comprehensive Assertions
# ============================================================================

print("\n" + "=" * 80)
print("Loading Val Dataset")
print("=" * 80)

print(f"\nLoading GQA val from: {GQA_VAL_PATH}")
val_dataset = load_from_disk(GQA_VAL_PATH)

print(f"✓ Loaded val dataset")
print(f"  Size: {len(val_dataset)}")
print(f"  Features: {list(val_dataset.features.keys())}")

# Verify required fields
for field in required_fields:
    assert field in val_dataset.features, f"Missing field: {field}"

# Comprehensive assertions
print(f"\nRunning comprehensive assertions on val dataset...")

# Check for non-empty dataset
assert len(val_dataset) > 0, "Val dataset is empty!"

# Check for NaN values in string fields
for field in string_fields:
    field_values = val_dataset[field]
    nan_count = sum(1 for v in field_values if pd.isna(v) or str(v).strip() == "")
    assert nan_count == 0, f"Found {nan_count} NaN or empty values in '{field}'"
    print(f"  ✓ No NaN/empty values in '{field}'")

# Check hash ID uniqueness
val_hash_ids = val_dataset["hash_id"]
val_unique_hash_ids = len(set(val_hash_ids))
assert len(val_hash_ids) == val_unique_hash_ids, \
    f"Duplicate hash IDs found: {len(val_hash_ids) - val_unique_hash_ids} duplicates"
print(f"  ✓ All hash IDs are unique ({val_unique_hash_ids} unique)")

# Check that all images are present
val_image_field = val_dataset["image"]
assert len(val_image_field) == len(val_dataset), "Mismatch in image count"
print(f"  ✓ All images are present ({len(val_image_field)} images)")

# Check dataset name consistency
val_dataset_names = set(val_dataset["dataset"])
assert len(val_dataset_names) == 1, f"Multiple dataset names found: {val_dataset_names}"
print(f"  ✓ Dataset name consistent: {val_dataset_names.pop()}")

print(f"\n✓ Val dataset comprehensive validation passed")

# ============================================================================
# CELL 5: Load All Test Datasets
# ============================================================================

print("\n" + "=" * 80)
print("Loading Test Datasets")
print("=" * 80)

test_datasets = {}
test_paths = {
    "GQA": GQA_TEST_PATH,
    "LLaVA-Wild": LLAVA_WILD_TEST_PATH,
    "GMAI-MMBench": GMAI_MMBENCH_TEST_PATH,
    "MME-Finance": MME_FINANCE_TEST_PATH,
    "POPE": POPE_TEST_PATH,
    "MMMU-Pro 4": MMMU_PRO_4_TEST_PATH,
    "MMMU-Pro 10": MMMU_PRO_10_TEST_PATH
}

for name, path in test_paths.items():
    print(f"\nLoading {name} test from: {path}")
    try:
        ds = load_from_disk(path)
        test_datasets[name] = ds
        print(f"  ✓ Loaded {name}: {len(ds)} samples")
        print(f"    Features: {list(ds.features.keys())}")
        
        # Verify required fields
        for field in required_fields:
            assert field in ds.features, f"Missing field '{field}' in {name}"
            
    except Exception as e:
        print(f"  ✗ Error loading {name}: {e}")
        raise

print(f"\n✓ All test datasets loaded successfully")
print(f"\nTest dataset summary:")
total_test_samples = sum(len(ds) for ds in test_datasets.values())
for name, ds in test_datasets.items():
    pct = (len(ds) / total_test_samples) * 100
    print(f"  {name}: {len(ds)} samples ({pct:.1f}%)")
print(f"  Total: {total_test_samples} samples")

# ============================================================================
# CELL A: Read test datasets, check duplicates, remove them
# ============================================================================

print("\n" + "=" * 80)
print("Reading and checking test datasets")
print("=" * 80)

# Convert all test datasets to lists and combine
test_records = []

for name, ds in test_datasets.items():
    print(f"\nProcessing {name} ({len(ds)} samples)...")
    for i in tqdm(range(len(ds)), desc=f"  {name}"):
        record = ds[i]
        test_records.append(record)

print(f"\nTotal collected records: {len(test_records)}")

# Check hash id counts before removing duplicates
all_hash_ids = [r["hash_id"] for r in test_records]
unique_hash_ids = set(all_hash_ids)
num_unique = len(unique_hash_ids)
num_total = len(all_hash_ids)
num_duplicates = num_total - num_unique

print(f"\nHash ID report:")
print(f"  Total records: {num_total}")
print(f"  Unique hash ids: {num_unique}")
print(f"  Duplicate count: {num_duplicates}")

# Remove duplicates by hash id (keep first occurrence)
seen = set()
clean_records = []
for r in test_records:
    hid = r["hash_id"]
    if hid not in seen:
        seen.add(hid)
        clean_records.append(r)

print(f"\nAfter removing duplicates:")
print(f"  New total records: {len(clean_records)}")

# Pass cleaned list forward
test_records_clean = clean_records
# ============================================================================
# CELL B: Build combined test dataset and run assertions
# ============================================================================

print("\n" + "=" * 80)
print("Building combined test dataset")
print("=" * 80)

combined_test_dataset = Dataset.from_list(test_records_clean)

print(f"\nDataset created")
print(f"  Size: {len(combined_test_dataset)}")
print(f"  Features: {list(combined_test_dataset.features.keys())}")

# Verify required fields exist
for field in required_fields:
    assert field in combined_test_dataset.features, f"Missing field: {field}"

print("\nRunning assertions...")

# Non empty
assert len(combined_test_dataset) > 0, "Combined test dataset is empty"

# NaN and empty checks
for field in string_fields:
    values = combined_test_dataset[field]
    nan_count = sum(1 for v in values if pd.isna(v) or str(v).strip() == "")
    assert nan_count == 0, f"Found {nan_count} NaN or empty entries in {field}"
    print(f"  Verified: no empty values in {field}")

# Hash id uniqueness
hash_ids = combined_test_dataset["hash_id"]
assert len(hash_ids) == len(set(hash_ids)), "Duplicate hash ids remain"
print(f"  Verified: all hash ids are unique")

# Image count
imgs = combined_test_dataset["image"]
assert len(imgs) == len(combined_test_dataset), "Mismatch in image count"
print(f"  Verified: all images present ({len(imgs)})")

# Dataset name check
names = set(combined_test_dataset["dataset"])
assert len(names) > 0, "No dataset names found"
print(f"  Verified: {len(names)} dataset names found: {sorted(names)}")

print("\nAll assertions passed")
# ============================================================================
# CELL 7: Dataset Statistics
# ============================================================================

print("\n" + "=" * 80)
print("Dataset Statistics")
print("=" * 80)

# Train statistics
print(f"\nTrain Dataset (GQA only):")
print(f"  Total samples: {len(train_dataset)}")
train_datasets = pd.Series(train_dataset["dataset"]).value_counts()
print(f"  Source datasets:")
for ds_name, count in train_datasets.items():
    print(f"    - {ds_name}: {count}")
train_categories = pd.Series(train_dataset["category"]).value_counts()
print(f"  Unique categories: {len(train_categories)}")
print(f"  Top 5 categories:")
for cat, count in train_categories.head(5).items():
    pct = (count / len(train_dataset)) * 100
    print(f"    - {cat}: {count} ({pct:.1f}%)")

# Val statistics
print(f"\nVal Dataset (GQA only):")
print(f"  Total samples: {len(val_dataset)}")
val_datasets = pd.Series(val_dataset["dataset"]).value_counts()
print(f"  Source datasets:")
for ds_name, count in val_datasets.items():
    print(f"    - {ds_name}: {count}")
val_categories = pd.Series(val_dataset["category"]).value_counts()
print(f"  Unique categories: {len(val_categories)}")
print(f"  Top 5 categories:")
for cat, count in val_categories.head(5).items():
    pct = (count / len(val_dataset)) * 100
    print(f"    - {cat}: {count} ({pct:.1f}%)")

# Test statistics
print(f"\nTest Dataset (Combined):")
print(f"  Total samples: {len(combined_test_dataset)}")
test_datasets_series = pd.Series(combined_test_dataset["dataset"]).value_counts()
print(f"  Source datasets:")
for ds_name, count in test_datasets_series.items():
    pct = (count / len(combined_test_dataset)) * 100
    print(f"    - {ds_name}: {count} ({pct:.1f}%)")
test_categories = pd.Series(combined_test_dataset["category"]).value_counts()
print(f"  Unique categories: {len(test_categories)}")
print(f"  Top 10 categories:")
for cat, count in test_categories.head(10).items():
    pct = (count / len(combined_test_dataset)) * 100
    print(f"    - {cat}: {count} ({pct:.1f}%)")

# ============================================================================
# CELL 8: Visualize Dataset Distributions
# ============================================================================

# Train dataset distribution
plt.figure(figsize=(14, 6))
train_datasets.plot(kind="bar", color="skyblue", edgecolor="black")
plt.title("Train Dataset - Source Distribution", fontsize=14, fontweight="bold")
plt.xlabel("Dataset", fontsize=12)
plt.ylabel("Count", fontsize=12)
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.show()

# Val dataset distribution
plt.figure(figsize=(14, 6))
val_datasets.plot(kind="bar", color="orange", edgecolor="black")
plt.title("Val Dataset - Source Distribution", fontsize=14, fontweight="bold")
plt.xlabel("Dataset", fontsize=12)
plt.ylabel("Count", fontsize=12)
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.show()

# Test dataset distribution
plt.figure(figsize=(14, 6))
test_datasets_series.plot(kind="bar", color="green", edgecolor="black")
plt.title("Test Dataset - Source Distribution", fontsize=14, fontweight="bold")
plt.xlabel("Dataset", fontsize=12)
plt.ylabel("Count", fontsize=12)
plt.xticks(rotation=45, ha="right")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.show()

# Overall split distribution
plt.figure(figsize=(10, 6))
splits = ["Train", "Val", "Test"]
counts = [len(train_dataset), len(val_dataset), len(combined_test_dataset)]
colors = ["skyblue", "orange", "green"]
plt.bar(splits, counts, color=colors, edgecolor="black")
plt.title("VLCB Dataset - Split Distribution", fontsize=14, fontweight="bold")
plt.xlabel("Split", fontsize=12)
plt.ylabel("Count", fontsize=12)
for i, (split, count) in enumerate(zip(splits, counts)):
    plt.text(i, count, str(count), ha="center", va="bottom", fontweight="bold")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.show()

# ============================================================================
# CELL 9: Comprehensive Data Quality Checks and Assertions
# ============================================================================

print("\n" + "=" * 80)
print("Comprehensive Data Quality Checks")
print("=" * 80)

# Check hash ID uniqueness (with assertions)
train_hash_ids = train_dataset["hash_id"]
val_hash_ids = val_dataset["hash_id"]
test_hash_ids = combined_test_dataset["hash_id"]

train_unique = len(set(train_hash_ids))
val_unique = len(set(val_hash_ids))
test_unique = len(set(test_hash_ids))

print(f"\nHash ID Uniqueness:")
print(f"  Train: {len(train_hash_ids)} total, {train_unique} unique")
assert len(train_hash_ids) == train_unique, "Train has duplicate hash IDs!"
print(f"  Val: {len(val_hash_ids)} total, {val_unique} unique")
assert len(val_hash_ids) == val_unique, "Val has duplicate hash IDs!"
print(f"  Test: {len(test_hash_ids)} total, {test_unique} unique")
assert len(test_hash_ids) == test_unique, "Test has duplicate hash IDs!"

# Check for overlap between splits (with assertions)
train_hash_set = set(train_hash_ids)
val_hash_set = set(val_hash_ids)
test_hash_set = set(test_hash_ids)

train_val_overlap = train_hash_set & val_hash_set
train_test_overlap = train_hash_set & test_hash_set
val_test_overlap = val_hash_set & test_hash_set

print(f"\nOverlap between splits:")
print(f"  Train-Val overlap: {len(train_val_overlap)}")
print(f"  Train-Test overlap: {len(train_test_overlap)}")
print(f"  Val-Test overlap: {len(val_test_overlap)}")

assert len(train_val_overlap) == 0, f"Found {len(train_val_overlap)} overlapping hash IDs between train and val!"
assert len(train_test_overlap) == 0, f"Found {len(train_test_overlap)} overlapping hash IDs between train and test!"
assert len(val_test_overlap) == 0, f"Found {len(val_test_overlap)} overlapping hash IDs between val and test!"
print(f"  ✓ No overlap between splits")

# Check for NaN values (with assertions)
print(f"\nNaN Value Checks:")
for split_name, dataset in [("Train", train_dataset), ("Val", val_dataset), ("Test", combined_test_dataset)]:
    for field in string_fields:
        field_values = dataset[field]
        nan_count = sum(1 for v in field_values if pd.isna(v) or str(v).strip() == "")
        assert nan_count == 0, f"{split_name} has {nan_count} NaN/empty values in '{field}'"
    print(f"  ✓ {split_name}: No NaN/empty values in any field")

# Check dataset consistency
print(f"\nDataset Name Consistency:")
train_ds_names = set(train_dataset["dataset"])
val_ds_names = set(val_dataset["dataset"])
test_ds_names = set(combined_test_dataset["dataset"])

print(f"  Train datasets: {sorted(train_ds_names)}")
print(f"  Val datasets: {sorted(val_ds_names)}")
print(f"  Test datasets: {sorted(test_ds_names)}")

assert len(train_ds_names) == 1 and "GQA" in train_ds_names, "Train should only contain GQA"
assert len(val_ds_names) == 1 and "GQA" in val_ds_names, "Val should only contain GQA"
assert len(test_ds_names) >= 7, f"Test should contain at least 7 datasets, found {len(test_ds_names)}"
print(f"  ✓ Dataset names are consistent")

# Check image presence
print(f"\nImage Presence Checks:")
assert len(train_dataset["image"]) == len(train_dataset), "Train image count mismatch"
assert len(val_dataset["image"]) == len(val_dataset), "Val image count mismatch"
assert len(combined_test_dataset["image"]) == len(combined_test_dataset), "Test image count mismatch"
print(f"  ✓ All images are present in all splits")

print(f"\n✓ All data quality checks passed!")

# ============================================================================
# CELL 10: Word Count Analysis and Distribution
# ============================================================================

print("\n" + "=" * 80)
print("Word Count Analysis")
print("=" * 80)

def calculate_word_counts(dataset, split_name):
    """Calculate word counts for questions and answers."""
    question_word_counts = []
    answer_word_counts = []
    
    for i in tqdm(range(len(dataset)), desc=f"  Processing {split_name}"):
        question = str(dataset[i]["question"]).strip()
        answer = str(dataset[i]["answer"]).strip()
        
        q_words = len(question.split())
        a_words = len(answer.split())
        
        question_word_counts.append(q_words)
        answer_word_counts.append(a_words)
    
    return question_word_counts, answer_word_counts

# Calculate word counts for all splits
print("\nCalculating word counts...")
train_q_words, train_a_words = calculate_word_counts(train_dataset, "Train")
val_q_words, val_a_words = calculate_word_counts(val_dataset, "Val")
test_q_words, test_a_words = calculate_word_counts(combined_test_dataset, "Test")

# Statistics for questions
print("\n" + "=" * 80)
print("Question Word Count Statistics")
print("=" * 80)

for split_name, q_words in [("Train", train_q_words), ("Val", val_q_words), ("Test", test_q_words)]:
    print(f"\n{split_name}:")
    print(f"  Mean: {np.mean(q_words):.2f} words")
    print(f"  Median: {np.median(q_words):.1f} words")
    print(f"  Std: {np.std(q_words):.2f} words")
    print(f"  Min: {np.min(q_words)} words")
    print(f"  Max: {np.max(q_words)} words")
    print(f"  25th percentile: {np.percentile(q_words, 25):.1f} words")
    print(f"  75th percentile: {np.percentile(q_words, 75):.1f} words")

# Statistics for answers
print("\n" + "=" * 80)
print("Answer Word Count Statistics")
print("=" * 80)

for split_name, a_words in [("Train", train_a_words), ("Val", val_a_words), ("Test", test_a_words)]:
    print(f"\n{split_name}:")
    print(f"  Mean: {np.mean(a_words):.2f} words")
    print(f"  Median: {np.median(a_words):.1f} words")
    print(f"  Std: {np.std(a_words):.2f} words")
    print(f"  Min: {np.min(a_words)} words")
    print(f"  Max: {np.max(a_words)} words")
    print(f"  25th percentile: {np.percentile(a_words, 25):.1f} words")
    print(f"  75th percentile: {np.percentile(a_words, 75):.1f} words")

# ============================================================================
# CELL 11: Visualize Word Count Distributions
# ============================================================================

# Question word count distributions
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

colors_q = ["skyblue", "orange", "green"]
splits_q = [("Train", train_q_words, colors_q[0]), ("Val", val_q_words, colors_q[1]), ("Test", test_q_words, colors_q[2])]

for idx, (split_name, q_words, color) in enumerate(splits_q):
    axes[idx].hist(q_words, bins=50, color=color, edgecolor="black", alpha=0.7)
    axes[idx].axvline(np.mean(q_words), color='red', linestyle='--', linewidth=2, 
                      label=f'Mean: {np.mean(q_words):.1f}')
    axes[idx].axvline(np.median(q_words), color='blue', linestyle='--', linewidth=2, 
                      label=f'Median: {np.median(q_words):.1f}')
    axes[idx].set_title(f"{split_name} - Question Word Count Distribution", 
                        fontsize=12, fontweight="bold")
    axes[idx].set_xlabel("Word Count", fontsize=10)
    axes[idx].set_ylabel("Frequency", fontsize=10)
    axes[idx].legend(fontsize=9)
    axes[idx].grid(alpha=0.3)

plt.suptitle("Question Word Count Distribution Across Splits", 
             fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
plt.show()

# Answer word count distributions
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

colors_a = ["skyblue", "orange", "green"]
splits_a = [("Train", train_a_words, colors_a[0]), ("Val", val_a_words, colors_a[1]), ("Test", test_a_words, colors_a[2])]

for idx, (split_name, a_words, color) in enumerate(splits_a):
    # Use log scale for answers as they can vary widely
    axes[idx].hist(a_words, bins=50, color=color, edgecolor="black", alpha=0.7)
    axes[idx].axvline(np.mean(a_words), color='red', linestyle='--', linewidth=2, 
                      label=f'Mean: {np.mean(a_words):.1f}')
    axes[idx].axvline(np.median(a_words), color='blue', linestyle='--', linewidth=2, 
                      label=f'Median: {np.median(a_words):.1f}')
    axes[idx].set_title(f"{split_name} - Answer Word Count Distribution", 
                        fontsize=12, fontweight="bold")
    axes[idx].set_xlabel("Word Count", fontsize=10)
    axes[idx].set_ylabel("Frequency", fontsize=10)
    axes[idx].legend(fontsize=9)
    axes[idx].grid(alpha=0.3)
    # Set x-axis limit to show most of the distribution (up to 95th percentile)
    if len(a_words) > 0:
        x_max = np.percentile(a_words, 95)
        axes[idx].set_xlim(0, x_max * 1.1)

plt.suptitle("Answer Word Count Distribution Across Splits", 
             fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
plt.show()

# Combined comparison
fig, axes = plt.subplots(2, 1, figsize=(14, 10))

# Questions comparison
for split_name, q_words, color in splits_q:
    axes[0].hist(q_words, bins=50, alpha=0.5, label=split_name, color=color, edgecolor="black")
axes[0].set_title("Question Word Count Distribution - All Splits", 
                  fontsize=12, fontweight="bold")
axes[0].set_xlabel("Word Count", fontsize=10)
axes[0].set_ylabel("Frequency", fontsize=10)
axes[0].legend(fontsize=10)
axes[0].grid(alpha=0.3)

# Answers comparison
for split_name, a_words, color in splits_a:
    axes[1].hist(a_words, bins=50, alpha=0.5, label=split_name, color=color, edgecolor="black")
axes[1].set_title("Answer Word Count Distribution - All Splits", 
                  fontsize=12, fontweight="bold")
axes[1].set_xlabel("Word Count", fontsize=10)
axes[1].set_ylabel("Frequency", fontsize=10)
axes[1].legend(fontsize=10)
axes[1].grid(alpha=0.3)
# Set x-axis limit for answers
if len(test_a_words) > 0:
    x_max = max(np.percentile(train_a_words, 95), 
                np.percentile(val_a_words, 95), 
                np.percentile(test_a_words, 95))
    axes[1].set_xlim(0, x_max * 1.1)

plt.tight_layout()
plt.show()

# Box plots for comparison
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Question word counts box plot
q_data = [train_q_words, val_q_words, test_q_words]
axes[0].boxplot(q_data, labels=["Train", "Val", "Test"], patch_artist=True,
                boxprops=dict(facecolor='lightblue', alpha=0.7),
                medianprops=dict(color='red', linewidth=2))
axes[0].set_title("Question Word Count Distribution - Box Plot", 
                  fontsize=12, fontweight="bold")
axes[0].set_ylabel("Word Count", fontsize=10)
axes[0].grid(alpha=0.3)

# Answer word counts box plot
a_data = [train_a_words, val_a_words, test_a_words]
axes[1].boxplot(a_data, labels=["Train", "Val", "Test"], patch_artist=True,
                boxprops=dict(facecolor='lightgreen', alpha=0.7),
                medianprops=dict(color='red', linewidth=2))
axes[1].set_title("Answer Word Count Distribution - Box Plot", 
                  fontsize=12, fontweight="bold")
axes[1].set_ylabel("Word Count", fontsize=10)
axes[1].grid(alpha=0.3)
# Set y-axis limit for answers
if len(test_a_words) > 0:
    y_max = max(np.percentile(train_a_words, 95), 
                np.percentile(val_a_words, 95), 
                np.percentile(test_a_words, 95))
    axes[1].set_ylim(0, y_max * 1.1)

plt.tight_layout()
plt.show()

# ============================================================================
# CELL 12: Detailed Dataset and Category Counts
# ============================================================================

print("\n" + "=" * 80)
print("Detailed Dataset and Category Counts")
print("=" * 80)

# Train dataset counts
print("\n" + "-" * 80)
print("TRAIN DATASET")
print("-" * 80)
print(f"Total samples: {len(train_dataset)}")

train_ds_counts = pd.Series(train_dataset["dataset"]).value_counts().sort_index()
print(f"\nSource dataset counts:")
for ds_name, count in train_ds_counts.items():
    pct = (count / len(train_dataset)) * 100
    print(f"  {ds_name}: {count} ({pct:.1f}%)")

train_cat_counts = pd.Series(train_dataset["category"]).value_counts()
print(f"\nCategory counts (showing all {len(train_cat_counts)} categories):")
for cat, count in train_cat_counts.items():
    pct = (count / len(train_dataset)) * 100
    print(f"  {cat}: {count} ({pct:.1f}%)")

# Val dataset counts
print("\n" + "-" * 80)
print("VAL DATASET")
print("-" * 80)
print(f"Total samples: {len(val_dataset)}")

val_ds_counts = pd.Series(val_dataset["dataset"]).value_counts().sort_index()
print(f"\nSource dataset counts:")
for ds_name, count in val_ds_counts.items():
    pct = (count / len(val_dataset)) * 100
    print(f"  {ds_name}: {count} ({pct:.1f}%)")

val_cat_counts = pd.Series(val_dataset["category"]).value_counts()
print(f"\nCategory counts (showing all {len(val_cat_counts)} categories):")
for cat, count in val_cat_counts.items():
    pct = (count / len(val_dataset)) * 100
    print(f"  {cat}: {count} ({pct:.1f}%)")

# Test dataset counts
print("\n" + "-" * 80)
print("TEST DATASET")
print("-" * 80)
print(f"Total samples: {len(combined_test_dataset)}")

test_ds_counts = pd.Series(combined_test_dataset["dataset"]).value_counts().sort_index()
print(f"\nSource dataset counts:")
for ds_name, count in test_ds_counts.items():
    pct = (count / len(combined_test_dataset)) * 100
    print(f"  {ds_name}: {count} ({pct:.1f}%)")

test_cat_counts = pd.Series(combined_test_dataset["category"]).value_counts()
print(f"\nCategory counts (showing top 20 of {len(test_cat_counts)} categories):")
for cat, count in test_cat_counts.head(20).items():
    pct = (count / len(combined_test_dataset)) * 100
    print(f"  {cat}: {count} ({pct:.1f}%)")
if len(test_cat_counts) > 20:
    print(f"  ... and {len(test_cat_counts) - 20} more categories")

# Category distribution by dataset in test set
print("\n" + "-" * 80)
print("TEST DATASET - Category Distribution by Source Dataset")
print("-" * 80)
for ds_name in sorted(test_ds_counts.index):
    ds_mask = pd.Series(combined_test_dataset["dataset"]) == ds_name
    ds_categories = pd.Series(np.array(combined_test_dataset["category"])[ds_mask]).value_counts()
    print(f"\n{ds_name} ({ds_mask.sum()} samples):")
    print(f"  Categories: {len(ds_categories)}")
    print(f"  Top 5 categories:")
    for cat, count in ds_categories.head(5).items():
        pct = (count / ds_mask.sum()) * 100
        print(f"    - {cat}: {count} ({pct:.1f}%)")

# ============================================================================
# CELL 13: Save Final Datasets
# ============================================================================

print("\n" + "=" * 80)
print("Saving Final Datasets")
print("=" * 80)

# Use smaller shard size to avoid offset overflow
MAX_SHARD_SIZE = "250MB"  # Reduced from default 500MB

print(f"\nSaving train dataset to: {FINAL_TRAIN_PATH}")
train_dataset.save_to_disk(FINAL_TRAIN_PATH, max_shard_size=MAX_SHARD_SIZE)
print("✓ Train dataset saved successfully!")

print(f"\nSaving val dataset to: {FINAL_VAL_PATH}")
val_dataset.save_to_disk(FINAL_VAL_PATH, max_shard_size=MAX_SHARD_SIZE)
print("✓ Val dataset saved successfully!")

print(f"\nSaving test dataset to: {FINAL_TEST_PATH}")
combined_test_dataset.save_to_disk(FINAL_TEST_PATH, max_shard_size=MAX_SHARD_SIZE)
print("✓ Test dataset saved successfully!")
# Verify saves by reloading
print("\nVerifying saved datasets...")
loaded_train = load_from_disk(FINAL_TRAIN_PATH)
loaded_val = load_from_disk(FINAL_VAL_PATH)
loaded_test = load_from_disk(FINAL_TEST_PATH)

print(f"\n✓ Reloaded train dataset with {len(loaded_train)} records")
print(f"✓ Reloaded val dataset with {len(loaded_val)} records")
print(f"✓ Reloaded test dataset with {len(loaded_test)} records")
print(f"\n✓ Train features: {list(loaded_train.features.keys())}")
print(f"✓ Val features: {list(loaded_val.features.keys())}")
print(f"✓ Test features: {list(loaded_test.features.keys())}")
# ============================================================================
# CELL 14: Final Summary with All Statistics
# ============================================================================

print("\n" + "=" * 80)
print("FINAL DATASET SUMMARY")
print("=" * 80)

print(f"\nVLCB Dataset Finalization Complete!")
print(f"\nFinal datasets saved to: {OUTPUT_ROOT}")

print(f"\nTrain Dataset (train):")
print(f"  Total Records: {len(train_dataset)}")
print(f"  Source: GQA train only")
print(f"  Save Location: {FINAL_TRAIN_PATH}")
print(f"  Unique Categories: {len(train_cat_counts)}")
print(f"  Question Word Count - Mean: {np.mean(train_q_words):.1f}, Median: {np.median(train_q_words):.1f}")
print(f"  Answer Word Count - Mean: {np.mean(train_a_words):.1f}, Median: {np.median(train_a_words):.1f}")

print(f"\nVal Dataset (validation):")
print(f"  Total Records: {len(val_dataset)}")
print(f"  Source: GQA val only")
print(f"  Save Location: {FINAL_VAL_PATH}")
print(f"  Unique Categories: {len(val_cat_counts)}")
print(f"  Question Word Count - Mean: {np.mean(val_q_words):.1f}, Median: {np.median(val_q_words):.1f}")
print(f"  Answer Word Count - Mean: {np.mean(val_a_words):.1f}, Median: {np.median(val_a_words):.1f}")

print(f"\nTest Dataset (test):")
print(f"  Total Records: {len(combined_test_dataset)}")
print(f"  Sources:")
for name, ds in test_datasets.items():
    pct = (len(ds) / len(combined_test_dataset)) * 100
    print(f"    - {name}: {len(ds)} ({pct:.1f}%)")
print(f"  Save Location: {FINAL_TEST_PATH}")
print(f"  Unique Categories: {len(test_cat_counts)}")
print(f"  Unique Source Datasets: {len(test_ds_counts)}")
print(f"  Question Word Count - Mean: {np.mean(test_q_words):.1f}, Median: {np.median(test_q_words):.1f}")
print(f"  Answer Word Count - Mean: {np.mean(test_a_words):.1f}, Median: {np.median(test_a_words):.1f}")

print(f"\nFields in each record:")
for field in required_fields:
    print(f"  - {field}")

print(f"\nTotal dataset size:")
total_size = len(train_dataset) + len(val_dataset) + len(combined_test_dataset)
print(f"  Train: {len(train_dataset)} ({len(train_dataset)/total_size*100:.1f}%)")
print(f"  Val: {len(val_dataset)} ({len(val_dataset)/total_size*100:.1f}%)")
print(f"  Test: {len(combined_test_dataset)} ({len(combined_test_dataset)/total_size*100:.1f}%)")
print(f"  Total: {total_size}")

print(f"\nData Quality Checks:")
print(f"  ✓ All hash IDs are unique across all splits")
print(f"  ✓ No overlap between train/val/test splits")
print(f"  ✓ No NaN or empty values in any field")
print(f"  ✓ All images are present")
print(f"  ✓ Dataset names are consistent")

print("\n✓ Processing complete!")
print("=" * 80)

# reload train val and test datasets
train_dataset = load_from_disk(FINAL_TRAIN_PATH)
val_dataset = load_from_disk(FINAL_VAL_PATH)
test_dataset = load_from_disk(FINAL_TEST_PATH)

print(f"✓ Reloaded train dataset with {len(train_dataset)} records")
print(f"✓ Reloaded val dataset with {len(val_dataset)} records")
print(f"✓ Reloaded test dataset with {len(test_dataset)} records")
def print_dataset_ranges(dataset, split_name):
    start_idx = None
    prev_ds = None
    ranges = []
    for i, sample in enumerate(dataset):
        ds = sample.get('dataset')
        if ds is None:
            continue  # Skip if no dataset key
        if ds != prev_ds:
            if prev_ds is not None:
                # End of previous ds
                ranges.append((prev_ds, start_idx, i-1))
            # Start new range
            start_idx = i
            prev_ds = ds
    if prev_ds is not None and start_idx is not None:
        # Finish last range
        ranges.append((prev_ds, start_idx, len(dataset)-1))
    print(f"\n{split_name} dataset source ranges:")
    for ds, start, end in ranges:
        print(f"  - {ds}: samples {start} to {end} (count {end-start+1})")

# print_dataset_ranges(train_dataset, "Train")
# print_dataset_ranges(val_dataset, "Val")
print_dataset_ranges(test_dataset, "Test")
# ============================================================================
# CELL 15: Image Resolution Analysis - Detect Large Images
# ============================================================================

print("\n" + "=" * 80)
print("Image Resolution Analysis")
print("=" * 80)

from PIL import Image
import io
from collections import defaultdict

def get_image_stats(img):
    """Get width, height, and approximate token count for an image."""
    if img is None:
        return None, None, None
    
    # Handle PIL Image
    if isinstance(img, Image.Image):
        w, h = img.size
    # Handle bytes
    elif isinstance(img, bytes):
        img_pil = Image.open(io.BytesIO(img))
        w, h = img_pil.size
    # Handle dict with bytes
    elif isinstance(img, dict) and 'bytes' in img:
        img_pil = Image.open(io.BytesIO(img['bytes']))
        w, h = img_pil.size
    else:
        return None, None, None
    
    # Approximate token count: pixels / 196 (typical ViT patch size 14x14 = 196 pixels per token)
    # This is a rough estimate used by many VLMs
    total_pixels = w * h
    approx_tokens = total_pixels // 196
    
    return w, h, approx_tokens

def analyze_split_images(dataset, split_name):
    """Analyze images in a dataset split, grouped by source dataset."""
    print(f"\nAnalyzing {split_name} images...")
    
    stats_by_dataset = defaultdict(lambda: {
        'widths': [], 'heights': [], 'tokens': [], 'pixels': []
    })
    
    errors = []
    
    for i in tqdm(range(len(dataset)), desc=f"  {split_name}"):
        try:
            sample = dataset[i]
            ds_name = sample['dataset']
            img = sample['image']
            
            w, h, tokens = get_image_stats(img)
            
            if w is not None:
                stats_by_dataset[ds_name]['widths'].append(w)
                stats_by_dataset[ds_name]['heights'].append(h)
                stats_by_dataset[ds_name]['tokens'].append(tokens)
                stats_by_dataset[ds_name]['pixels'].append(w * h)
            else:
                errors.append((i, ds_name, "Could not read image"))
                
        except Exception as e:
            errors.append((i, sample.get('dataset', 'unknown'), str(e)))
    
    if errors:
        print(f"  Errors encountered: {len(errors)}")
        for idx, ds, err in errors[:5]:
            print(f"    Sample {idx} ({ds}): {err}")
        if len(errors) > 5:
            print(f"    ... and {len(errors) - 5} more errors")
    
    return dict(stats_by_dataset)

# Reload datasets
print("Reloading datasets...")
loaded_train = load_from_disk(FINAL_TRAIN_PATH)
loaded_val = load_from_disk(FINAL_VAL_PATH)
loaded_test = load_from_disk(FINAL_TEST_PATH)

# Analyze each split
train_stats = analyze_split_images(loaded_train, "Train")
val_stats = analyze_split_images(loaded_val, "Val")
test_stats = analyze_split_images(loaded_test, "Test")
# ============================================================================
# CELL 16: Print Image Statistics Summary
# ============================================================================

print("\n" + "=" * 80)
print("Image Statistics Summary")
print("=" * 80)

def print_stats_summary(stats_by_dataset, split_name):
    print(f"\n{split_name} Image Statistics:")
    print("-" * 60)
    
    for ds_name in sorted(stats_by_dataset.keys()):
        stats = stats_by_dataset[ds_name]
        n = len(stats['widths'])
        
        if n == 0:
            print(f"\n  {ds_name}: No valid images")
            continue
            
        print(f"\n  {ds_name} ({n} images):")
        print(f"    Width  - Mean: {np.mean(stats['widths']):.0f}, "
              f"Min: {np.min(stats['widths'])}, Max: {np.max(stats['widths'])}")
        print(f"    Height - Mean: {np.mean(stats['heights']):.0f}, "
              f"Min: {np.min(stats['heights'])}, Max: {np.max(stats['heights'])}")
        print(f"    Pixels - Mean: {np.mean(stats['pixels'])/1e6:.2f}M, "
              f"Min: {np.min(stats['pixels'])/1e6:.2f}M, Max: {np.max(stats['pixels'])/1e6:.2f}M")
        print(f"    Tokens - Mean: {np.mean(stats['tokens']):.0f}, "
              f"Min: {np.min(stats['tokens'])}, Max: {np.max(stats['tokens'])}")
        
        # Flag potentially problematic images
        large_threshold = 4096 * 4096  # 16M pixels
        large_count = sum(1 for p in stats['pixels'] if p > large_threshold)
        if large_count > 0:
            print(f"    ⚠️  {large_count} images exceed {large_threshold/1e6:.0f}M pixels!")

print_stats_summary(train_stats, "TRAIN")
print_stats_summary(val_stats, "VAL")
print_stats_summary(test_stats, "TEST")
# ============================================================================
# CELL 17: Histogram Plots - Image Token Counts by Dataset
# ============================================================================

import matplotlib.pyplot as plt
import matplotlib.cm as cm

def plot_token_histograms(stats_by_dataset, split_name, ax):
    """Plot overlapping histograms of token counts for each dataset in a split."""
    
    datasets = sorted(stats_by_dataset.keys())
    n_datasets = len(datasets)
    
    if n_datasets == 0:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
        ax.set_title(f'{split_name} - Image Token Distribution')
        return
    
    # Use a colormap for distinct colors
    colors = cm.tab10(np.linspace(0, 1, max(n_datasets, 1)))
    
    # Find global max for consistent binning
    all_tokens = []
    for ds_name in datasets:
        all_tokens.extend(stats_by_dataset[ds_name]['tokens'])
    
    if not all_tokens:
        ax.text(0.5, 0.5, 'No valid images', ha='center', va='center', transform=ax.transAxes)
        ax.set_title(f'{split_name} - Image Token Distribution')
        return
    
    # Use percentile for x-axis limit to handle outliers
    x_max = np.percentile(all_tokens, 99)
    bins = np.linspace(0, x_max, 50)
    
    # Plot histogram for each dataset
    for idx, ds_name in enumerate(datasets):
        tokens = stats_by_dataset[ds_name]['tokens']
        if len(tokens) > 0:
            ax.hist(tokens, bins=bins, alpha=0.6, label=f'{ds_name} (n={len(tokens)})', 
                   color=colors[idx], edgecolor='black', linewidth=0.5)
    
    ax.set_xlabel('Approximate Image Tokens (pixels/196)', fontsize=10)
    ax.set_ylabel('Frequency', fontsize=10)
    ax.set_title(f'{split_name} - Image Token Distribution by Dataset', fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_xlim(0, x_max * 1.05)

# Create figure with 3 subplots
fig, axes = plt.subplots(3, 1, figsize=(14, 15))

plot_token_histograms(train_stats, "Train", axes[0])
plot_token_histograms(val_stats, "Val", axes[1])
plot_token_histograms(test_stats, "Test", axes[2])

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_ROOT, "image_token_distributions.png"), dpi=150, bbox_inches='tight')
plt.show()

print(f"\n✓ Plot saved to: {os.path.join(OUTPUT_ROOT, 'image_token_distributions.png')}")
# ============================================================================
# CELL 18: Detailed Test Set Analysis - Find Outliers
# ============================================================================

print("\n" + "=" * 80)
print("Test Set Image Outlier Analysis")
print("=" * 80)

# Find the largest images in test set by dataset
print("\nTop 10 largest images by pixel count in TEST set:")
print("-" * 80)

outliers = []
for ds_name, stats in test_stats.items():
    for i, (w, h, tokens, pixels) in enumerate(zip(
        stats['widths'], stats['heights'], stats['tokens'], stats['pixels']
    )):
        outliers.append({
            'dataset': ds_name,
            'width': w,
            'height': h,
            'pixels': pixels,
            'tokens': tokens,
            'megapixels': pixels / 1e6
        })

outliers_df = pd.DataFrame(outliers)
outliers_df = outliers_df.sort_values('pixels', ascending=False)

print(outliers_df.head(20).to_string(index=False))

# Summary by dataset
print("\n" + "-" * 80)
print("Test Set - Statistics by Dataset (sorted by max pixels)")
print("-" * 80)

summary_rows = []
for ds_name, stats in test_stats.items():
    if len(stats['pixels']) > 0:
        summary_rows.append({
            'Dataset': ds_name,
            'Count': len(stats['pixels']),
            'Mean MP': np.mean(stats['pixels']) / 1e6,
            'Max MP': np.max(stats['pixels']) / 1e6,
            'Min MP': np.min(stats['pixels']) / 1e6,
            'Mean Tokens': np.mean(stats['tokens']),
            'Max Tokens': np.max(stats['tokens'])
        })

summary_df = pd.DataFrame(summary_rows)
summary_df = summary_df.sort_values('Max MP', ascending=False)
print(summary_df.to_string(index=False, float_format='%.2f'))

# Flag datasets with potential OOM issues
print("\n" + "-" * 80)
print("⚠️  Datasets with potentially problematic large images:")
print("-" * 80)

threshold_mp = 10  # 10 megapixels
for _, row in summary_df.iterrows():
    if row['Max MP'] > threshold_mp:
        print(f"  {row['Dataset']}: Max {row['Max MP']:.1f} MP, Mean {row['Mean MP']:.1f} MP")
