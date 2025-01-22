# Models Directory

This directory contains the neural network model architectures used for sleep stage classification.

## Model Architectures

### lstm.py
LSTM-based sleep stage classifier that processes multimodal physiological data:
- Heart rate processing with CNN + Bidirectional LSTM
- Motion processing with 2D CNN + Bidirectional LSTM 
- Additional features:
  - Step count integration
  - Previous sleep stage label embedding
- Feature fusion via concatenation and dense layers
- Outputs 5-class sleep stage predictions

Example usage:
```python
from models.lstm import SleepClassifierLSTM

model = SleepClassifierLSTM(num_classes=5)
output = model({
    'heart_rate': heart_rate_tensor,      # [B, 120]
    'motion': motion_tensor,              # [B, 3000, 3] 
    'steps': steps_tensor,                # [B, 1]
    'previous_labels': prev_labels_tensor # [B, 19]
})
# Returns: [B, 5] class probabilities
```

### transformer.py
Transformer-based sleep stage classifier with similar architecture:
- Replaces LSTM layers with Transformer encoders
- Uses self-attention for temporal modeling
- Maintains same input/output interface as LSTM model
- Generally better at capturing long-range dependencies

Example usage:
```python
from models.transformer import SleepClassifierTransformer

model = SleepClassifierTransformer(num_classes=5) 
output = model({
    'heart_rate': heart_rate_tensor,      # [B, 120]
    'motion': motion_tensor,              # [B, 3000, 3]
    'steps': steps_tensor,                # [B, 1]
    'previous_labels': prev_labels_tensor # [B, 19]
})
# Returns: [B, 5] class probabilities
```

## Sleep Stage Classes

Both models output predictions for 5 sleep stages:
- 0: Wake (Awake State)
- 1: N1 (Light Sleep)
- 2: N2 (Light Sleep, Deeper than N1)
- 3: N3 (Deep Sleep/Slow-Wave Sleep)
- 4: REM Sleep

## Model Statistics

- LSTM Model:
  - Parameters: 195,365
  - Approx Memory: 0.75 MB

- Transformer Model:
  - Parameters: 236,901
  - Approx Memory: 0.90 MB
