# Source Code Directory

This directory contains the core source code for the sleep stage classification project.

## Directory Structure

- `/data/` - Data loading, preprocessing and dataset management scripts
- `/models/` - Neural network model architectures and definitions
- Root level files:
  - `inference.py` - FastAPI server for model inference and API endpoints
  - `test_inference.py` - Testing scripts for the inference API
  - `train.py` - Model training pipeline and experiment management

## Key Files

### inference.py
FastAPI server that provides REST endpoints for sleep stage classification:
- Handles data validation and preprocessing
- Serves model predictions via API
- Includes rate limiting and authentication
- Provides health check and version endpoints

### test_inference.py  
Test suite for the inference API:
- Tests endpoint functionality
- Validates model predictions
- Measures prediction accuracy
- Handles batch processing with rate limiting

### train.py
Training pipeline for sleep classification models:
- Manages complete training workflow
- Handles data loading and batching
- Tracks metrics and experiment history
- Supports parallel training runs
- Includes validation and testing

## Code Examples

Example training run:
```python
from train import run_single_experiment

params = {
    'num_epochs': 10,
    'learning_rate': 0.0003,
    'batch_size': 512,
    'fold_id': 0
}

history = run_single_experiment(
    model="lstm",
    params=params, 
    base_dir="./experiments",
    num_workers=4,
    data_dir="./data"
)
```

Example inference request:
```python
import requests

data = {
    "heart_rate": [75] * 120,  # Must contain 1-120 values
    "motion": [[0.1, -0.2, 9.8]] * 3000,  # Must contain 1-3000 vectors with x,y,z values
    "steps": 42,  # Non-negative step count
    "previous_labels": [0, 1, 2, 2, 3] * 4  # Must contain 1-19 labels between 0-5
}

response = requests.post(
    "http://localhost:8000/predict",  # Default FastAPI port
    json=data,
    headers={"X-API-Key": "your-api-key"}
)
prediction = response.json()
# Returns:
# {
#    "predicted_class": 2,  # Sleep stage (0=Wake, 1=N1, 2=N2, 3=N3, 5=REM)
#    "class_probabilities": [0.1, 0.2, 0.4, 0.2, 0.1]  # Probabilities for each class
# }
```


Example test inference:
```python
from test_inference import test_inference_endpoint, get_data

# Get test data for a subject
request_data = get_data(
    data_dir="./data/test",
    subject_id="1066528"  # Example subject ID
)

# Test the endpoint
response = test_inference_endpoint(
    request_data=request_data,
    api_url="http://localhost:6969/predict",
    api_key="your-api-key"
)

# Returns:
# {
#    "predicted_class": 2,  # Sleep stage prediction
#    "class_probabilities": [0.1, 0.2, 0.4, 0.2, 0.1],  # Class probabilities
#    "warning": "Optional warning if data was adjusted"
# }

# For batch testing with metrics:
from test_inference import SleepDataset, DataLoader
import asyncio

# Set up test dataset
test_dataset = SleepDataset(
    data_dir="./data/test",
    fold_id=6,
    train_mode=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=128,
    num_workers=4
)

# Process batches and get metrics
loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)

correct = 0
total = 0
for batch_data, batch_labels in test_loader:
    results = loop.run_until_complete(
        process_batch(batch_data, batch_labels, api_key, max_concurrent=5)
    )
    # Update accuracy metrics
    for pred_label, true_label in results:
        if pred_label is not None:
            total += 1
            if pred_label == true_label:
                correct += 1

accuracy = 100 * correct / total
print(f"Test Accuracy: {accuracy:.2f}%")
```

