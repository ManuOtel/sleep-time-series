"""
This module provides a PyTorch Dataset for sleep stage classification using multimodal time series data.

The main purpose is to load and prepare preprocessed sensor data (heart rate, motion, steps) and sleep
stage labels for training deep learning models. It handles:

The module contains a SleepDataset class that:
    1. Takes preprocessed HDF5 data files as input
    2. Creates fixed-length sequences with configurable stride for sequence models
    3. Performs configurable validation checks on each data stream, including:
        - Cross-validation 10-fold splitting
        - Train/validation/test set creation
        - Data normalization and preprocessing
        - Batch generation for training
    4. Reports detailed validation failures and statistics
    5. Can process individual subjects or entire datasets
"""

import json
import torch
import numpy as np
from pathlib import Path
try:
    from .reader import DataReader
except ImportError:
    try:
        from reader import DataReader
    except ImportError:
        raise ImportError(
            "Could not import DataReader from either .reader or reader")
from typing import Tuple, Dict, Optional
from torch.utils.data import Dataset, random_split


class SleepDataset(Dataset):
    """Dataset class for sleep stage classification using multimodal time series data.

    This class loads preprocessed sensor data (heart rate, motion, steps) and sleep stage labels
    from HDF5 files. It creates fixed-length sequences with optional stride for training
    sequence models like LSTM or Transformers.

    Args:
        data_dir: Directory containing preprocessed HDF5 files
        sequence_length: Length of sequences in samples (default 600 samples = 5 min)
        stride: Number of samples to stride between sequences (default 30 samples = 30 sec)
        fold_id: Which fold to use as test set (0-9), if None uses all data
        train_mode: Whether to return train or test set when using folds
        valid_ratio: Ratio of training sequences to use for validation
        seed: Random seed for reproducibility
        split: Which split to return ('train' or 'valid') when using validation
    """

    def __init__(self,
                 data_dir: str,
                 sequence_length: int = 600,
                 stride: int = 30,
                 fold_id: Optional[int] = None,
                 train_mode: bool = True,
                 valid_ratio: float = 0.1,
                 seed: int = 42,
                 split: str = 'train') -> None:
        """Initialize the SleepDataset.

        Args:
            data_dir: Path to directory containing preprocessed HDF5 data files
            sequence_length: Length of each sequence in samples (default 600 = 5 min at 2 Hz)
            stride: Number of samples to stride between sequences (default 30 = 30 sec)
            fold_id: Which cross-validation fold to use as test set (0-9), if None uses all data
            train_mode: If True, return training fold, if False return test fold
            valid_ratio: Fraction of training sequences to use for validation (0.0-1.0)
            seed: Random seed for reproducible data splits
            split: Which split to return when using validation ('train' or 'valid')

        Returns:
            None

        Raises:
            FileNotFoundError: If data directory or subject_ids.json not found
            ValueError: If invalid fold_id, valid_ratio or split parameters
        """

        # Add number of sleep stages (classes)
        self.num_classes = 5  # Assuming 5 sleep stages (0-4)

        self.reader = DataReader(data_dir)
        # Load subject IDs from JSON file in data directory
        subject_ids_file = Path(data_dir) / "subject_ids.json"
        with open(subject_ids_file, 'r') as f:
            all_subject_ids = json.load(f)

        # Split subjects into 10 folds
        np.random.seed(seed)
        shuffled_subjects = np.random.permutation(all_subject_ids)
        fold_size = len(shuffled_subjects) // 10
        subject_folds = [shuffled_subjects[i:i + fold_size]
                         for i in range(0, len(shuffled_subjects), fold_size)]

        # Select subjects based on fold_id
        if fold_id is not None:
            test_subjects = subject_folds[fold_id]
            train_subjects = [
                s for fold in subject_folds[:fold_id] + subject_folds[fold_id+1:] for s in fold]
            self.subject_ids = train_subjects if train_mode else test_subjects
        else:
            self.subject_ids = all_subject_ids

        self.sequence_length = sequence_length
        self.stride = stride

        # Calculate expected samples for each modality
        self.hr_samples = int(sequence_length * 0.2)
        self.motion_samples = int(sequence_length * 5)
        self.steps_samples = max(1, int(sequence_length * 0.002))

        # Load and preprocess all data
        self.sequences = []
        self.labels = []
        self.sequence_metadata = []

        for subject_id in self.subject_ids:
            # Load all data streams
            hr_data = self.reader.read_heart_rate(subject_id)
            motion_data = self.reader.read_motion(subject_id)
            steps_data = self.reader.read_steps(subject_id)
            label_data = self.reader.read_labels(subject_id)

            # Create sequences with stride
            total_duration = label_data.timestamps[-1] - \
                label_data.timestamps[0]
            for start_time in range(0, int(total_duration) - sequence_length + 1, stride):
                end_time = start_time + sequence_length

                # Get data for each modality in their natural frequency
                hr_mask = (hr_data.timestamps >= start_time) & (
                    hr_data.timestamps < end_time)
                motion_mask = (motion_data.timestamps >= start_time) & (
                    motion_data.timestamps < end_time)
                steps_mask = (steps_data.timestamps >= start_time) & (
                    steps_data.timestamps < end_time)

                hr_seq = hr_data.values[hr_mask]
                motion_seq = motion_data.values[motion_mask]
                steps_seq = steps_data.values[steps_mask]

                # Validate sequence lengths
                if (len(hr_seq) < self.hr_samples * 0.9 or  # Allow 10% tolerance
                    len(motion_seq) < self.motion_samples * 0.9 or
                        len(steps_seq) < self.steps_samples):
                    continue

                # Trim or pad to exact expected lengths
                hr_seq = self._adjust_sequence_length(hr_seq, self.hr_samples)
                motion_seq = self._adjust_sequence_length(
                    motion_seq, self.motion_samples)
                steps_seq = self._adjust_sequence_length(
                    steps_seq, self.steps_samples)

                # Get labels for this time window
                label_mask = (label_data.timestamps >= start_time) & (
                    label_data.timestamps < end_time)
                labels_in_window = label_data.values[label_mask]
                if len(labels_in_window) == 0 or -1 in labels_in_window:
                    continue

                # For a 600s window with 30s intervals, we expect 20 labels
                if len(labels_in_window) < 20:  # Skip if we don't have enough labels
                    continue

                metadata = {
                    'subject_id': subject_id,
                    'start_time': start_time,
                    'end_time': end_time,
                    # timestamp of prediction target
                    'timestamp': label_data.timestamps[label_mask][-1]
                }

                # Previous labels are all except the last one
                previous_labels = labels_in_window[:-1]  # Should be 19 labels
                # The 20th label to predict
                current_label = labels_in_window[-1]

                sequence = {
                    'heart_rate': torch.FloatTensor(hr_seq),      # [120, 1]
                    'motion': torch.FloatTensor(motion_seq),      # [3000, 3]
                    'steps': torch.FloatTensor(steps_seq),        # [1-2, 1]
                    # [19, 1]
                    'previous_labels': torch.LongTensor(previous_labels)
                }

                self.sequences.append(sequence)
                self.sequence_metadata.append(metadata)
                # Only storing the last label as target
                self.labels.append(current_label)
        # Convert labels to one-hot encoded tensors
        labels_one_hot = torch.zeros((len(self.labels), self.num_classes))
        for i, label in enumerate(self.labels):
            # Convert label 5 to 4 to make classes sequential 0-4
            label = 4 if label == 5 else label
            labels_one_hot[i][label] = 1
        self.labels = labels_one_hot

        # Split into train/valid if in training mode and using folds
        if fold_id is not None and train_mode and valid_ratio > 0:
            num_valid = int(len(self.sequences) * valid_ratio)
            num_train = len(self.sequences) - num_valid

            # Random split for validation
            train_dataset, valid_dataset = random_split(
                list(zip(self.sequences, self.labels)),
                [num_train, num_valid],
                generator=torch.Generator().manual_seed(seed)
            )

            # Select appropriate split based on parameter
            if split == 'train':
                self.sequences, self.labels = zip(*train_dataset)
            elif split == 'valid':
                self.sequences, self.labels = zip(*valid_dataset)

            self.sequences = list(self.sequences)
            self.labels = torch.stack(list(self.labels))

    def _adjust_sequence_length(self, seq: np.ndarray, target_length: int) -> np.ndarray:
        """Adjust sequence to target length by trimming or padding with zeros.

        Args:
            seq: Input sequence array to adjust
            target_length: Desired length of output sequence

        Returns:
            Adjusted sequence array with length equal to target_length:
            - If input is longer than target, returns trimmed sequence
            - If input is shorter than target, returns zero-padded sequence
            - If input equals target length, returns unchanged sequence
        """
        if len(seq) > target_length:
            return seq[:target_length]
        elif len(seq) < target_length:
            pad_width = [(0, target_length - len(seq))] + \
                [(0, 0)] * (seq.ndim - 1)
            return np.pad(seq, pad_width, mode='constant', constant_values=0)
        return seq

    def get_sequences_for_subject(self, subject_id: str) -> Dict:
        """Get all sequences and metadata for a specific subject.

        Args:
            subject_id: The subject ID to retrieve sequences for

        Returns:
            Dictionary containing:
                - sequences: List of sequence dictionaries (heart_rate, motion, steps, previous_labels)
                - labels: Tensor of labels for each sequence
                - timestamps: List of timestamps for each sequence
                - indices: Original indices in the dataset for each sequence
        """
        # Find all indices for this subject
        subject_indices = [
            idx for idx, meta in enumerate(self.sequence_metadata)
            if meta['subject_id'] == subject_id
        ]

        if not subject_indices:
            raise ValueError(f"No sequences found for subject {subject_id}")

        # Gather all data for these indices
        subject_sequences = [self.sequences[idx] for idx in subject_indices]
        subject_labels = self.labels[subject_indices]
        subject_timestamps = [self.sequence_metadata[idx]
                              ['timestamp'] for idx in subject_indices]

        return {
            'sequences': subject_sequences,
            'labels': subject_labels,
            'timestamps': subject_timestamps,
            'indices': subject_indices
        }

    def __len__(self) -> int:
        """Return number of sequences in dataset"""
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """Get sequence and label at index"""
        return self.sequences[idx], self.labels[idx]


if __name__ == "__main__":
    import time
    import argparse
    from torch.utils.data import DataLoader
    #### This is for dataloader ####
    import multiprocessing
    # Set start method to spawn
    multiprocessing.set_start_method('spawn', force=True)

    # Set up argument parser
    parser = argparse.ArgumentParser(
        description='Test dataset loading and processing')
    parser.add_argument('-d', '--data_dir', type=str, default='./data/preprocessed/',
                        help='Directory containing preprocessed data')
    parser.add_argument('-f', '--fold_id', type=int, default=0,
                        help='Fold ID for cross validation')
    parser.add_argument('-b', '--batch_size', type=int, default=512,
                        help='Batch size for data loading')
    parser.add_argument('-n', '--num_workers', type=int, default=4,
                        help='Number of worker processes for data loading')
    parser.add_argument('-v', '--valid_ratio', type=float, default=0.1,
                        help='Ratio of validation data split')
    args = parser.parse_args()

    # Time dataset creation
    start = time.time()
    train_dataset = SleepDataset(
        args.data_dir, fold_id=args.fold_id, train_mode=True,
        valid_ratio=args.valid_ratio, split='train')
    valid_dataset = SleepDataset(
        args.data_dir, fold_id=args.fold_id, train_mode=True,
        valid_ratio=args.valid_ratio, split='valid')
    test_dataset = SleepDataset(
        args.data_dir, fold_id=args.fold_id, train_mode=False)
    dataset_time = time.time() - start

    print("\nDataset Loading Summary")
    print("=" * 50)
    print(f"Dataset creation time: {dataset_time:.2f} seconds")
    print(f"Train sequences: {len(train_dataset)}")
    print(f"Valid sequences: {len(valid_dataset)}")
    print(f"Test sequences: {len(test_dataset)}")

    # Test dataloader speed
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=1,  # args.num_workers,
        persistent_workers=True,
        pin_memory=True
    )

    start = time.time()
    for batch_idx, (batch_data, batch_labels) in enumerate(train_loader):
        if batch_idx % 100 == 0:
            print(f"Processed {batch_idx} batches...")
    dataloader_time = time.time() - start

    print("\nDataloader Performance")
    print("=" * 50)
    print(f"Time to iterate all batches: {dataloader_time:.2f} seconds")
    print(f"Average time per batch: {
          dataloader_time/len(train_loader):.4f} seconds")

    # Test a sample
    sequence, label = train_dataset[0]
    print("\nSequence Structure")
    print("=" * 50)
    print(f"Heart Rate shape: {sequence['heart_rate'].shape}")
    print(f"Motion shape: {sequence['motion'].shape}")
    print(f"Steps shape: {sequence['steps'].shape}")
    print(f"Previous labels shape: {sequence['previous_labels'].shape}")
    print(f"Label: {label}")

    # Get sequences for a specific subject
    subject_id = "1066528"  # Example subject ID
    print("\nSequences for Subject Analysis")
    print("=" * 50)

    # Get all sequences for this subject
    subject_data = train_dataset.get_sequences_for_subject(subject_id)
    print(f"Number of sequences for subject {
          subject_id}: {len(subject_data['sequences'])}")

    if len(subject_data['sequences']) > 0:
        # Get first sequence for this subject
        seq = subject_data['sequences'][0]
        lbl = subject_data['labels'][0]

        print("\nExample Sequence Stats:")
        print(f"Heart Rate - Mean: {seq['heart_rate'].mean():.1f}, "
              f"Min: {seq['heart_rate'].min():.1f}, "
              f"Max: {seq['heart_rate'].max():.1f}")

        print(f"Motion Magnitude - Mean: {torch.norm(seq['motion'], dim=1).mean():.3f}, "
              f"Max: {torch.norm(seq['motion'], dim=1).max():.3f}")

        print(f"Steps: {seq['steps'].item()}")
        print(f"Previous Sleep Stages: {seq['previous_labels'].tolist()}")
        print(f"Current Sleep Stage: {torch.argmax(lbl).item()}")

    #### Print Example ####
    # Dataset Loading Summary
    # ==================================================
    # Dataset creation time: 37.54 seconds
    # Train sequences: 18998
    # Valid sequences: 2110
    # Test sequences: 2297
    # Processed 0 batches...

    # Dataloader Performance
    # ==================================================
    # Time to iterate all batches: 38.74 seconds
    # Average time per batch: 1.0195 seconds

    # Sequence Structure
    # ==================================================
    # Heart Rate shape: torch.Size([120])
    # Motion shape: torch.Size([3000, 3])
    # Steps shape: torch.Size([1])
    # Previous labels shape: torch.Size([19])
    # Label: tensor([0., 0., 1., 0., 0.])

    # Sequences for Subject Analysis
    # ==================================================
    # Number of sequences for subject 1066528: 929

    # Example Sequence Stats:
    # Heart Rate - Mean: 0.6, Min: 0.3, Max: 1.1
    # Motion Magnitude - Mean: 1.004, Max: 1.006
    # Steps: 0.0
    # Previous Sleep Stages: [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
    # Current Sleep Stage: 2
