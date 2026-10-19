#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import shutil
import random
from pathlib import Path
from typing import List, Tuple


def get_all_files(directory: str) -> List[str]:
    """
    Get all file paths under a directory

    Args:
        directory: directory path

    Returns:
        List of file paths
    """
    files = []
    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)
        if os.path.isfile(item_path):
            files.append(item_path)
    return files


def sample_files_from_categories(source_dir: str, sample_ratio: float = 0.05) -> dict:
    """
    Randomly sample a given fraction of files from each class folder

    Args:
        source_dir: source directory path
        sample_ratio: sampling ratio, default 5%

    Returns:
        Dict mapping class name to the list of sampled file paths
    """
    sampled_data = {}

    # Iterate over all subfolders in the source directory
    for category in os.listdir(source_dir):
        category_path = os.path.join(source_dir, category)

        # Skip non-directory entries
        if not os.path.isdir(category_path):
            continue

        # Get all files in this class
        all_files = get_all_files(category_path)
        total_count = len(all_files)

        # Compute the number of files to sample
        sample_count = max(1, int(total_count * sample_ratio))

        # Random sampling
        sampled_files = random.sample(all_files, sample_count)
        sampled_data[category] = sampled_files

        print(f"Class {category}: total={total_count}, sampled={sample_count}")

    return sampled_data


def split_and_copy_files(sampled_data: dict, target_dir: str,
                         train_ratio, val_ratio, test_ratio):
    """
    Split the sampled files into train/val/test folders by ratio

    Args:
        sampled_data: dict of sampled files
        target_dir: target directory path
        train_ratio: train split ratio
        val_ratio: val split ratio
        test_ratio: test split ratio
    """
    # Create the target directory structure
    splits = ['train', 'val', 'test']
    for split in splits:
        split_path = os.path.join(target_dir, split)
        os.makedirs(split_path, exist_ok=True)

    # Compute the total ratio
    total_ratio = train_ratio + val_ratio + test_ratio

    # Process each class
    for category, files in sampled_data.items():
        # Create class folders under each split
        for split in splits:
            category_split_path = os.path.join(target_dir, split, category)
            os.makedirs(category_split_path, exist_ok=True)

        # Shuffle file order
        random.shuffle(files)

        total_files = len(files)

        # Compute the number of files per split
        train_count = int(total_files * train_ratio / total_ratio)
        val_count = int(total_files * val_ratio / total_ratio)
        # test_count takes all remaining files so none are left out

        # Split the file list
        train_files = files[:train_count]
        val_files = files[train_count:train_count + val_count]
        test_files = files[train_count + val_count:]

        # Copy files to the corresponding directories
        for file_path in train_files:
            filename = os.path.basename(file_path)
            target_path = os.path.join(target_dir, 'train', category, filename)
            shutil.copy2(file_path, target_path)

        for file_path in val_files:
            filename = os.path.basename(file_path)
            target_path = os.path.join(target_dir, 'val', category, filename)
            shutil.copy2(file_path, target_path)

        for file_path in test_files:
            filename = os.path.basename(file_path)
            target_path = os.path.join(target_dir, 'test', category, filename)
            shutil.copy2(file_path, target_path)

        print(f"Class {category}: train={len(train_files)}, val={len(val_files)}, test={len(test_files)}")


def main():
    """Main function"""
    # TODO: change to your actual paths
    SOURCE_DIR = "./all_RSCD"  # Source directory containing 27 class folders
    TARGET_DIR = "./5%_RSCD"  # Target directory; train/val/test will be created

    # Set random seed for reproducibility (optional)
    random.seed(42)

    print("=" * 50)
    print("Sampling 5% of the data from each class...")
    print("=" * 50)

    # Step 1: sample 5% of the data from each class
    sampled_data = sample_files_from_categories(SOURCE_DIR, sample_ratio=0.05)

    print("\n" + "=" * 50)
    print("Splitting data into train/val/test...")
    print("=" * 50)

    # Step 2: split the sampled data into train/val/test
    split_and_copy_files(sampled_data, TARGET_DIR,
                         train_ratio=80, val_ratio=10, test_ratio=10)

    print("\n" + "=" * 50)
    print("Data processing complete!")
    print("=" * 50)


if __name__ == "__main__":
    main()