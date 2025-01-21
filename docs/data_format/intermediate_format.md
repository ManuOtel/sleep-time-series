# Sleep Classifier Intermediate Data Format

This document describes the intermediate HDF5 data format used in the sleep stage classification pipeline.

## Overview

The intermediate format uses HDF5 (Hierarchical Data Format version 5) to store preprocessed physiological data and sleep labels. This format serves as a bridge between raw device data and the final PyTorch tensors used for training.

## HDF5 Structure

```text
sleep_data.h5
├── subjects/
│ ├── subject_001/
│ │ ├── heart_rate/
│ │ │ ├── values # Float32 array [N]
│ │ │ └── timestamps # Int64 array [N]
│ │ ├── motion/
│ │ │ ├── values # Float32 array [N, 3]
│ │ │ └── timestamps # Int64 array [N]
│ │ ├── steps/
│ │ │ ├── values # Int32 array [N]
│ │ │ └── timestamps # Int64 array [N]
│ │ └── sleep_stages/
│ │ ├── labels # Int8 array [N]
│ │ └── timestamps # Int64 array [N]
```

## Data Groups Details

### Heart Rate Data

**Location**: `/subjects/{subject_id}/heart_rate/`

**Values Dataset**:

- Type: Float32
- Shape: [N]
- Units: Beats per minute (BPM)
- Range: [30, 220]

**Timestamps Dataset**:

- Type: Int64
- Shape: [N]
- Units: Unix timestamp (seconds)
- Sampling: 0.2 Hz (5-second intervals)

### Motion Data

**Location**: `/subjects/{subject_id}/motion/`

**Values Dataset**:

- Type: Float32
- Shape: [N, 3]
- Units: g (acceleration)
- Range: [-16, 16]
- Columns: [x, y, z] acceleration

**Timestamps Dataset**:

- Type: Int64
- Shape: [N]
- Units: Unix timestamp (seconds)
- Sampling: 5 Hz

### Steps Data

**Location**: `/subjects/{subject_id}/steps/`

**Values Dataset**:

- Type: Int32
- Shape: [N]
- Units: Step count
- Range: [0, inf)

**Timestamps Dataset**:

- Type: Int64
- Shape: [N]
- Units: Unix timestamp (seconds)
- Sampling: Variable (typically ~8.3 minutes)

### Sleep Stages

**Location**: `/subjects/{subject_id}/sleep_stages/`

**Labels Dataset**:

- Type: Int8
- Shape: [N]
- Values:
  - -1: Invalid/Unknown
  - 0: Wake
  - 1: N1 (Light sleep)
  - 2: N2 (Stable sleep)
  - 3: N3 (Deep sleep)
  - 5: REM sleep

**Timestamps Dataset**:

- Type: Int64
- Shape: [N]
- Units: Unix timestamp (seconds)
- Sampling: 30-second epochs

Example:

```json
{
  "heart_rate": {
    "filter_type": "median",
    "window_size": 5,
    "min_valid": 30,
    "max_valid": 220,
    "normalization": "standard"
  },
  "motion": {
    "filter_type": "butterworth",
    "cutoff_freq": 0.5,
    "order": 4,
    "normalization": "minmax"
  },
  "steps": {
    "aggregation": "sum",
    "window_size": 600,
    "normalization": "standard"
  }
}
```

## Data Quality Attributes

Each dataset group contains additional attributes for data quality tracking:

### Quality Metrics

**Missing Data Percentage**:

- Attribute: missing_pct
- Type: Float
- Range: [0, 100]

**Signal Quality Score**:

- Attribute: quality_score
- Type: Float
- Range: [0, 1]

**Valid Samples Count**:

- Attribute: valid_samples
- Type: Integer

### Data Statistics

**Summary Statistics**:

- Attribute: statistics
- Type: JSON string
- Contains: mean, std, min, max, quartiles

## Data Integrity Features

### Checksums

- Each dataset includes an MD5 checksum attribute
- Used to verify data integrity during transfers
- Stored in dataset attributes as md5_hash

### Compression

- Compression Type: GZIP
- Compression Level: 6
- Chunk Size: Optimized for typical reading patterns
  - Heart Rate: 1800 samples (15 minutes)
  - Motion: 600 samples (2 minutes)
  - Steps: 360 samples (1 hour)
  - Sleep Stages: 120 samples (1 hour)

## Reading Patterns

### Sequential Access

```python
with h5py.File('sleep_data.h5', 'r') as f:
    # Read a subject's heart rate data
    subject = f['subjects']['subject_001']
    hr_values = subject['heart_rate']['values'][:]
    hr_times = subject['heart_rate']['timestamps'][:]
```

### Random Access

```python
with h5py.File('sleep_data.h5', 'r') as f:
    # Read specific time window
    start_idx = 1000
    window_size = 120
    hr_window = subject['heart_rate']['values'][start_idx:start_idx+window_size]
```

## Data Validation Rules

### Timestamp Consistency

- Strictly monotonic increasing
- No duplicates
- No gaps larger than specified thresholds

### Value Ranges

- Heart Rate: 30-220 BPM
- Motion: ±16g per axis
- Steps: Non-negative integers
- Sleep Stages: Valid label set only

### Sampling Rate Tolerance

- Heart Rate: ±0.1 Hz from nominal
- Motion: ±0.5 Hz from nominal
- Steps: Variable but timestamped
- Sleep Stages: Exact 30-second epochs

### Missing Data Handling

- Gaps < 30s: Linear interpolation
- Gaps 30s-5min: Forward fill
- Gaps > 5min: Marked invalid

## Related Documentation

- Data Preprocessing Guide
- HDF5 Best Practices
- Quality Control Documentation
