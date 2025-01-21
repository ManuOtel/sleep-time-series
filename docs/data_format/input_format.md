# Sleep Classifier Input Format

This document describes the input data format required for the sleep stage classification models (LSTM and Transformer versions).

## Overview

The models take multimodal physiological data as input, including:
- Heart rate measurements
- Motion (acceleration) data  
- Step counts
- Previous sleep stage labels

## Input Dictionary Structure

The input to both models is a dictionary containing the following keys and tensor shapes:

```python
{
    'heart_rate': torch.FloatTensor,      # Shape: [batch_size, 120]
    'motion': torch.FloatTensor,          # Shape: [batch_size, 3000, 3] 
    'steps': torch.FloatTensor,           # Shape: [batch_size, 1]
    'previous_labels': torch.LongTensor   # Shape: [batch_size, 19]
}
```

## Data Streams Details

### Heart Rate
- Sampling rate: 0.2 Hz (every 5 seconds)
- Window size: 600 seconds (10 minutes)
- Resulting sequence length: 120 samples
- Values: Normalized heart rate in beats per minute (BPM)

### Motion
- Sampling rate: 5 Hz
- Window size: 600 seconds (10 minutes)
- Resulting sequence length: 3000 samples
- Values: Normalized tri-axial acceleration (x, y, z) in g units
- Shape: [3000, 3] where 3 represents the x, y, z acceleration components

### Steps
- Sampling rate: 0.002 Hz (every ~8.3 minutes)
- Window size: 600 seconds (10 minutes)
- Resulting sequence length: 1 sample
- Values: Normalized step count for the window

### Previous Labels
- Sequence of 19 previous sleep stage labels
- Each label is an integer in range [-1, 5]:
  - -1: Invalid/Unknown
  - 0: Wake
  - 1: N1 (Light sleep)
  - 2: N2 (Stable sleep)
  - 3: N3 (Deep sleep)
  - 5: REM sleep

## Output Format

The models output sleep stage predictions with shape [batch_size, num_classes], where num_classes=5 representing:
- 0: Wake
- 1: N1 (Light sleep)
- 2: N2 (Stable sleep)
- 3: N3 (Deep sleep)
- 4: REM sleep

Note: While input labels use 5 to represent REM sleep, the model output uses 4 to maintain zero-based consecutive class indices.

## Data Processing Pipeline

1. Raw data is collected from wearable devices
2. Data is formatted into HDF5 files (DataFormator)
3. Data is preprocessed and normalized (DataPreprocessor)
4. DataReader loads the processed data
5. SleepDataset creates the windowed sequences
6. Data is batched and fed to the models

## Example Usage

```python
# Create example input batch
batch_size = 2
example_data = {
    'heart_rate': torch.randn(batch_size, 120),        # [B, 120]
    'motion': torch.randn(batch_size, 3000, 3),        # [B, 3000, 3]
    'steps': torch.randn(batch_size, 1),               # [B, 1]
    'previous_labels': torch.randint(0, 4, (batch_size, 19))  # [B, 19]
}

# Get predictions
model = SleepClassifierLSTM(num_classes=5)
predictions = model(example_data)  # Shape: [2, 5]
```

## Data Quality Requirements

The DataChecker validates input data for:
- Missing or corrupted values
- Non-monotonic timestamps
- Sampling irregularities
- Data gaps and coverage
- Value range violations
- Misaligned endpoints between streams

See the DataChecker class for detailed validation criteria.