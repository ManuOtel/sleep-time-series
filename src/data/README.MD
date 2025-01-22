# Data Processing Directory

This directory contains scripts for data loading, preprocessing and management of sleep stage classification data.

## Key Files

### formator.py

Handles reformatting of raw Apple Watch sensor data into standardized HDF5 format:

- Converts raw text files containing:
  - Heart rate measurements
  - Motion/acceleration data (x,y,z)
  - Step counts
  - Sleep stage labels
- Performs extensive data validation:
  - Missing/corrupted file checks
  - Empty data stream detection
  - Timestamp monotonicity
  - Data gap identification
  - Value range validation
  - Stream alignment checks
  - Sampling rate consistency
- Creates compressed HDF5 files with standardized structure

Example usage:

```python
from data.formator import DataFormator

# Initialize formator
formator = DataFormator(
    data_dir="./data/original",
    output_dir="./data/formatted",
    verbose=True
)

# Convert single subject
formator.convert_subject_data("46343")

# Convert all subjects
formator.convert_all_subjects()
```

Output HDF5 structure:

```
subject_id.h5
├── heart_rate/
│   ├── timestamps  # [N] float64 array
│   └── values     # [N] int64 array
├── motion/
│   ├── timestamps  # [M] float64 array
│   └── values     # [M,3] float64 array (x,y,z)
├── steps/
│   ├── timestamps  # [P] int64 array
│   └── values     # [P] int64 array
└── labels/
    ├── timestamps  # [Q] int64 array
    └── values     # [Q] int64 array
```

The formatted HDF5 files provide:

- Efficient storage with compression
- Fast random access to data segments
- Standardized data organization
- Built-in data validation
- Reduced memory footprint
- Simplified data loading interface

## reader.py

Provides efficient access to formatted HDF5 data files:

- Clean interface for accessing sensor streams
- Type-safe data containers using dataclasses
- Consistent error handling
- Returns numpy arrays ready for model input

Example usage:

```python
from data.reader import DataReader

# Initialize reader
reader = DataReader(
    data_dir="./data/formatted",
    verbose=True
)

# Read data streams for a subject
heart_rate = reader.read_heart_rate("46343")
motion = reader.read_motion("46343")
steps = reader.read_steps("46343")
labels = reader.read_labels("46343")

# Access data fields
print(f"Heart rate timestamps: {heart_rate.timestamps.shape}")  # [N]
print(f"Heart rate values: {heart_rate.values.shape}")         # [N]

print(f"Motion timestamps: {motion.timestamps.shape}")          # [M]
print(f"Motion values: {motion.values.shape}")                 # [M,3]

print(f"Steps timestamps: {steps.timestamps.shape}")           # [P]
print(f"Steps values: {steps.values.shape}")                  # [P]

print(f"Label timestamps: {labels.timestamps.shape}")          # [Q]
print(f"Label values: {labels.values.shape}")                 # [Q]
```

The DataReader provides:

- TimeSeriesData class for timestamps/values
- Automatic file existence validation
- Consistent error messages
- Optional verbose logging
- Memory-efficient data loading
- Clean access to each data stream

## info.py

Analyzes preprocessed HDF5 data files to extract key statistical information:

- Missing/corrupted data detection
- Empty stream identification
- Timestamp validation
- Gap analysis and coverage metrics
- Value range validation
- Stream alignment checks
- Sampling rate consistency

Example usage:

```python
from data.info import DataInfo

# Initialize analyzer
analyzer = DataInfo(
    data_dir="./data/formatted",
    verbose=True
)

# Analyze all subjects
results = analyzer.analyze_all_subjects()

# Access individual subject stats
subject_stats = results['individual']['46343']
print(f"Heart rate samples: {subject_stats['heart_rate']['n_samples']}")
print(f"Motion samples: {subject_stats['motion']['n_samples']}")

# Access aggregate summary stats
summary = results['summary']
print(f"\nHeart Rate Missing Data:")
print(f"Mean: {summary['heart_rate']['missing_data']['mean_pct']:.1f}%")
print(f"Max Gap: {summary['heart_rate']['gap_statistics']['max_gap']:.1f}s")
```

The DataInfo analyzer provides:

- Per-subject statistical analysis
- Dataset-wide aggregate metrics
- Missing data quantification
- Gap detection and statistics
- Distribution analysis
- Sampling irregularity detection
- Detailed error reporting

## preprocess.py

Handles preprocessing and testing of formatted HDF5 data files:

- Trims data streams to relevant time periods around sleep labels
- Normalizes data using training set statistics (mean/std)
- Resamples data to consistent frequencies
- Fixes invalid values and non-monotonic timestamps
- Saves preprocessed data in HDF5 format

Example usage:

```python
from data.preprocess import DataPreprocessor

# Initialize preprocessor
preprocessor = DataPreprocessor(
    data_dir="./data/formatted",
    output_dir="./data/preprocessed",
    test_output_dir="./data/test",
    verbose=True
)

# Preprocess training data (with normalization)
preprocessor.preprocess_all_subjects(test=False)

# Process test data (without normalization)
preprocessor.preprocess_all_subjects(test=True)

# Process single subject
preprocessor.preprocess_subject_data("46343", test=False)
```

The DataPreprocessor provides:

- Training vs test data handling
- Data stream trimming and alignment
- Consistent resampling frequencies:
  - Labels: 1/30 Hz (30s intervals)
  - Heart rate: 0.2 Hz (5s intervals)
  - Motion: 5 Hz (0.2s intervals)
  - Steps: 1/500 Hz (500s intervals)
- Invalid value detection and fixing
- Non-monotonic timestamp correction
- Progress tracking and logging
- Memory-efficient processing

## checker.py

Validates preprocessed data files for completeness and quality:

- Checks for missing or corrupted data files
- Validates data stream presence and completeness
- Verifies value ranges and validity
- Checks timestamp monotonicity and sampling rates
- Detects data gaps and coverage issues
- Validates stream alignment and synchronization
- Reports detailed validation failures

Example usage:

```python
from data.checker import DataChecker

# Initialize checker
checker = DataChecker(
    data_dir="./data/preprocessed",
    verbose=True
)

# Check single subject
valid, reason = checker.check_subject_data(
    subject_id="46343",
    hr_range=(40, 150),  # Valid heart rate range
    motion_range=(-10, 10),  # Valid motion range
    max_allowed_gap=3600,  # Max 1 hour gap
    min_duration=10800  # Min 3 hours total
)

# Check all subjects and modify invalid files
results = checker.check_all_subjects(
    modify_invalid=True,  # Rename invalid files
    siamese_dir="./data/siamese"  # Optional siamese dir
)
```

The DataChecker provides:

- Comprehensive data validation checks
- Configurable validation parameters
- Individual and batch subject processing
- Invalid file handling and renaming
- Siamese data validation support
- Detailed failure reporting and statistics
- Progress tracking and logging

Key validation checks:

- Empty stream detection
- Value range validation
- Timestamp monotonicity
- Sampling rate consistency
- Data coverage and gaps
- Stream alignment
- File integrity

Expected sampling rates:

- Heart rate: 0.2 Hz (5s)
- Motion: 5 Hz (0.2s)
- Steps: 0.002 Hz (500s)
- Labels: 0.033 Hz (30s)

## dataset.py

Provides PyTorch Dataset class for loading and batching sleep stage data:

- Creates fixed-length sequences with configurable stride
- Handles train/validation/test splits
- Performs data validation and preprocessing
- Supports cross-validation folds
- Manages multimodal data streams

Example usage:

```python
from data.dataset import SleepDataset
from torch.utils.data import DataLoader

# Create dataset splits
train_dataset = SleepDataset(
    data_dir="./data/preprocessed",
    sequence_length=600,  # 5 min sequences
    stride=30,           # 30 sec stride
    fold_id=0,          # Use fold 0 for testing
    train_mode=True,    # Get training data
    valid_ratio=0.1,    # 10% validation
    split='train'       # Get training split
)

valid_dataset = SleepDataset(
    data_dir="./data/preprocessed",
    fold_id=0,
    train_mode=True,
    valid_ratio=0.1,
    split='valid'
)

test_dataset = SleepDataset(
    data_dir="./data/preprocessed",
    fold_id=0,
    train_mode=False    # Get test data
)

# Create data loaders
train_loader = DataLoader(
    train_dataset,
    batch_size=512,
    num_workers=4,
    pin_memory=True
)

# Access a single sequence
sequence, label = train_dataset[0]
print(f"Heart rate shape: {sequence['heart_rate'].shape}")     # [120]
print(f"Motion shape: {sequence['motion'].shape}")             # [3000, 3]
print(f"Steps shape: {sequence['steps'].shape}")               # [1]
print(f"Previous labels shape: {sequence['previous_labels'].shape}")  # [19]
print(f"Label shape: {label.shape}")                          # [5]

# Get sequences for a specific subject
subject_data = train_dataset.get_sequences_for_subject("1066528")
print(f"Number of sequences: {len(subject_data['sequences'])}")
print(f"Sequence timestamps: {subject_data['timestamps']}")
```

The SleepDataset provides:

- Fixed-length sequence creation
- Configurable sequence stride
- Cross-validation fold support
- Train/validation splitting
- Subject-specific sequence access
- Automatic data validation
- Memory-efficient data loading
- Multimodal data handling

Expected sequence shapes:

- Heart rate: [120] (5 min @ 0.2 Hz)
- Motion: [3000, 3] (5 min @ 5 Hz)
- Steps: [1-2] (5 min @ 0.002 Hz)
- Previous labels: [19] (Last 19 30s labels)
- Target label: [5] (One-hot encoded)
