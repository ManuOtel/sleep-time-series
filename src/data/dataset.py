import json
import torch
import numpy as np
from typing import List, Tuple, Dict, Optional
from torch.utils.data import Dataset, DataLoader, random_split
from .reader import DataReader
from pathlib import Path


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
                 split: str = 'train'):

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

                label = np.bincount(labels_in_window).argmax()

                # Convert label to one-hot encoded vector
                label_onehot = torch.zeros(self.num_classes)
                label_onehot[label] = 1.0

                sequence = {
                    'heart_rate': torch.FloatTensor(hr_seq),      # [120, 1]
                    'motion': torch.FloatTensor(motion_seq),      # [3000, 3]
                    'steps': torch.FloatTensor(steps_seq),        # [1-2, 1]
                }

                self.sequences.append(sequence)
                self.labels.append(label_onehot)

        # Changed from ByteTensor to stacked tensor
        self.labels = torch.stack(self.labels)

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

    def _adjust_sequence_length(self, seq, target_length):
        """Adjust sequence to target length by trimming or padding"""
        if len(seq) > target_length:
            return seq[:target_length]
        elif len(seq) < target_length:
            pad_width = [(0, target_length - len(seq))] + \
                [(0, 0)] * (seq.ndim - 1)
            return np.pad(seq, pad_width, mode='constant', constant_values=0)
        return seq

    def __len__(self) -> int:
        """Return number of sequences in dataset"""
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """Get sequence and label at index"""
        return self.sequences[idx], self.labels[idx]


if __name__ == "__main__":
    # Example usage
    data_dir = Path("data/preprocessed")

    # Create train/test/valid splits
    train_dataset = SleepDataset(
        data_dir, fold_id=0, train_mode=True, valid_ratio=0.1, split='train')
    valid_dataset = SleepDataset(
        data_dir, fold_id=0, train_mode=True, valid_ratio=0.1, split='valid')
    test_dataset = SleepDataset(data_dir, fold_id=0, train_mode=False)

    print(f"Train sequences: {len(train_dataset)}")
    print(f"Valid sequences: {len(valid_dataset)}")
    print(f"Test sequences: {len(test_dataset)}")

    # Test a sample
    sequence, label = train_dataset[0]
    # Should be [120, 1]
    print(f"\nHeart Rate shape: {sequence['heart_rate'].shape}")
    print(f"Motion shape: {sequence['motion'].shape}")  # Should be [3000, 3]
    # Should be [1, 1] or [2, 1]
    print(f"Steps shape: {sequence['steps'].shape}")
    print(f"Label: {label}")

    #### Print Example ####
    # Train sequences: 18998
    # Valid sequences: 2110
    # Test sequences: 2297

    # Heart Rate shape: torch.Size([120])
    # Motion shape: torch.Size([3000, 3])
    # Steps shape: torch.Size([1])
    # Label: tensor([1., 0., 0., 0., 0.])
