# Test Suite Documentation

**NOTE: This documentation is a proof-of-concept and is far from complete due to time constraints. It serves as an initial framework for documenting the test suite and will be expanded significantly as more tests are implemented.**


This directory contains test files for the sleep stage classification system. The test suite uses pytest as the testing framework.

## Current Test Coverage

Currently, the test suite includes:

- `data/checker_test.py`: Tests for data validation and checking functionality
  - Tests empty stream detection
  - Tests timestamp monotonicity checking 
  - Tests value range validation

- `data/dataset_test.py`: Tests for dataset loading and processing
  - Tests dataset initialization
  - Tests sequence generation
  - Tests tensor shapes and formats

## Planned Test Categories (Not Implemented)

Due to time constraints, the following test categories are planned but not yet implemented:

### Models
- Model architecture tests
- Forward pass tests
- Loss calculation tests
- Model loading/saving tests

### Training
- Training loop tests
- Optimizer tests
- Learning rate scheduler tests
- Checkpoint management tests

### Inference
- API endpoint tests
- Model prediction tests
- Input validation tests
- Error handling tests

### Utils
- Configuration loading tests
- Logging tests
- Metric calculation tests
- Data preprocessing tests

## Running Tests

To run the existing tests:
