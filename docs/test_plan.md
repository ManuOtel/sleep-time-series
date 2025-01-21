AppleWatch/
├── src/
│   ├── data/
│   │   ├── preprocess.py
│   │   ├── lines_of_code.py
│   │   └── reader.py
│   └── utils/
│       └── logger.py
└── tests/
    ├── data/
    │   ├── test_preprocess.py
    │   ├── test_lines_of_code.py
    │   └── test_reader.py
    └── utils/
        └── test_logger.py


Plan:
Overall Testing Strategy
Create a tests directory with subdirectories matching the source structure
Use pytest as the testing framework
Create fixtures for common test data and mocked dependencies
Follow AAA pattern (Arrange, Act, Assert) for test structure
Testing Plan by File

for scr/data/reader:
```
tests/data/test_reader.py:

1. Test TimeSeriesData dataclass
   - Test creation with valid data
   - Test creation with empty arrays
   - Test creation with invalid data types

2. Test DataReader class
   - Test initialization
   - Test file not found scenarios
   - Test read_heart_rate()
     - Valid data
     - Empty data
     - Corrupted data
   - Test read_motion()
     - Valid data
     - Empty data
     - Corrupted data
   - Test read_steps()
     - Valid data
     - Empty data
     - Corrupted data
   - Test read_labels()
     - Valid data
     - Empty data
     - Corrupted data
```

for src/data/checker
```
tests/data/test_checker.py:

1. Test DataChecker class
   - Test initialization
   - Test _check_empty_streams()
     - All streams present
     - Missing streams
     - Empty streams
   - Test _check_data_gaps()
     - No gaps
     - Small gaps
     - Large gaps
     - Missing ratio violations
   - Test _check_sampling_rates()
     - Regular sampling
     - Irregular sampling
     - Different tolerances
   - Test _check_aligned_endpoints()
     - Aligned data
     - Misaligned data
   - Test check_all_subjects()
     - All valid
     - Some invalid
     - All invalid
     - Test modify_invalid flag
```

for src/data/dataset
```
tests/data/test_dataset.py:

1. Test SleepDataset class
   - Test initialization
     - Different sequence lengths
     - Different strides
     - Different fold configurations
   - Test cross-validation splitting
     - Test fold creation
     - Test train/test split
   - Test validation split
     - Different ratios
     - Different seeds
   - Test sequence generation
     - Correct lengths
     - Correct strides
     - Handling edge cases
   - Test data normalization
   - Test label encoding
   - Test __len__ and __getitem__
   - Test batch generation
```

for src/data/formator
```
tests/data/test_formator.py:

1. Test DataFormator class
   - Test initialization
     - Valid directories
     - Invalid directories
     - Permission issues
   - Test convert_subject_data()
     - Valid data conversion
     - Missing files
     - Corrupted files
     - Invalid data formats
   - Test HDF5 file creation
     - Correct structure
     - Data integrity
     - Compression
   - Test convert_all_subjects()
     - Multiple subjects
     - Error handling
     - Progress tracking
```

for src/data/preprocess.py
```
tests/data/test_preprocess.py:

1. Test DataPreprocessor class
   - Test initialization
   - Test data trimming functions
     - _trim_data_to_labels()
     - _fix_monotonic_timestamps()
     - _fix_motion_range()
   - Test resampling functions
     - Different target frequencies
     - Different methods
   - Test normalization
   - Test synthetic data generation
   - Test preprocess_subject_data()
     - Complete pipeline
     - Error handling
   - Test preprocess_all_subjects()
     - Multiple subjects
     - Progress tracking
```

for src/data/visualization
```

#### 6. `src/data/visualization.py`
```
tests/data/test_visualization.py:

1. Test plot_subject_data function
   - Test plot generation
     - All data streams present
     - Missing data streams
     - Different time ranges
   - Test figure properties
     - Correct dimensions
     - Correct labels
     - Day boundaries
   - Test saving functionality
     - Different formats
     - Different resolutions
   - Test error handling
     - Invalid data
     - File system issues
```

Common Test Fixtures
```
tests/data/conftest.py:

1. Sample data fixtures
   - Mock heart rate data
   - Mock motion data
   - Mock steps data
   - Mock labels data
   - Mock HDF5 files

2. Mock file system fixtures
   - Temporary directories
   - Sample file structures

3. Configuration fixtures
   - Test parameters
   - Expected values
```