# Sleep Classifier Intermediate Data Format

This document describes the intermediate HDF5 data format used in the sleep stage classification pipeline.

## Overview

The intermediate format uses HDF5 (Hierarchical Data Format version 5) to store preprocessed physiological data and sleep labels. The format provides efficient storage and access for the training pipeline.

## HDF5 Structure

Each subject's data is stored in a separate HDF5 file named `{subject_id}.h5` with the following structure:

```text
{subject_id}.h5
├── heart_rate/
│   ├── timestamps  # Float64 array [N]
│   └── values     # Int64 array [N]
├── motion/
│   ├── timestamps  # Float64 array [N]
│   └── values     # Float64 array [N,3]
├── steps/
│   ├── timestamps  # Int64 array [N]
│   └── values     # Int64 array [N]
└── labels/
    ├── timestamps  # Int64 array [N]
    └── values     # Int64 array [N]
```

## Data Groups Details

### Heart Rate Data

**Values Dataset**:
- Type: Int64
- Shape: [N]
- Units: Beats per minute (BPM)
- Range: [30, 220]
- Sampling: Variable rate (~0.2 Hz)

**Timestamps Dataset**:
- Type: Float64
- Shape: [N]
- Units: Unix timestamp (seconds)

### Motion Data

**Values Dataset**:
- Type: Float64
- Shape: [N, 3]
- Units: g (acceleration)
- Range: [-16, 16]
- Columns: [acc_x, acc_y, acc_z]
- Sampling: Variable rate (~5 Hz)

**Timestamps Dataset**:
- Type: Float64
- Shape: [N]
- Units: Unix timestamp (seconds)

### Steps Data

**Values Dataset**:
- Type: Int64
- Shape: [N]
- Units: Step count
- Range: [0, inf)
- Sampling: Variable rate

**Timestamps Dataset**:
- Type: Int64
- Shape: [N]
- Units: Unix timestamp (seconds)

### Sleep Stage Labels

**Values Dataset**:
- Type: Int64
- Shape: [N]
- Values:
  - -1: Invalid/Unknown
  - 0: Wake
  - 1: N1 (Light sleep)
  - 2: N2 (Stable sleep)
  - 3: N3 (Deep sleep)
  - 5: REM sleep
- Sampling: 30-second epochs

**Timestamps Dataset**:
- Type: Int64
- Shape: [N]
- Units: Unix timestamp (seconds)

## Example Usage

Reading data from the HDF5 file:

```python
with h5py.File('46343.h5', 'r') as f:
    # Read heart rate data
    hr_timestamps = f['heart_rate/timestamps'][:]
    hr_values = f['heart_rate/values'][:]
    
    # Read motion data
    motion_timestamps = f['motion/timestamps'][:]
    motion_values = f['motion/values'][:]  # Shape: [N,3]
    
    # Read steps data
    steps_timestamps = f['steps/timestamps'][:]
    steps_values = f['steps/values'][:]
    
    # Read sleep stage labels
    label_timestamps = f['labels/timestamps'][:]
    label_values = f['labels/values'][:]
```

## Related Documentation

- Data Preprocessing Guide
- Data Quality Control Documentation
- Model Input Format Specification
