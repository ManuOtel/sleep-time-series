# Sleep Stage Classification System

A deep learning system for real-time sleep stage prediction using consumer-grade wearable data. This project implements both Transformer and LSTM architectures to classify sleep stages (REM, Deep Sleep, Light Sleep, Awake) based on heart rate, motion, and step data.

## Overview

This system processes wearable device data from the [PhysioNet Sleep-Accel Dataset](https://physionet.org/content/sleep-accel/1.0.0/) to predict sleep stages in real-time. It includes:

- Data preprocessing and validation pipeline
- Model training with both Transformer and LSTM architectures
- Real-time inference API endpoint
- Visualization tools for data analysis
- Comprehensive testing and validation framework

## Quick Start

### Using Docker (Recommended)

1. Clone the repository:
```bash
git clone https://github.com/ManuOtel/sleep-time-series.git
cd sleep-time-series
```

2. Start the system using Docker Compose:
```bash
docker-compose up --build
```

This will:
- Download and preprocess the PhysioNet dataset
- Train the models
- Start the inference API server on port 6969

### Manual Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the pipeline:
```bash
chmod +x run.sh
./run.sh
```

## Project Structure

- `src/` - Source code for data processing, training, and inference
- `config/` - Configuration files for training and inference
- `data/` - Dataset storage and processing
- `models/` - Trained model checkpoints
- `visualizations/` - Generated data visualizations
- `logs/` - Training and inference logs
- `docker/` - Docker-related files and mounted volumes

## API Usage

After starting the server, you can make predictions using the REST API:

```bash
curl -X POST http://localhost:6969/predict \
  -H "X-API-Key: YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "heart_rate": [...],
    "motion": [[...]],
    "steps": ...,
    "previous_labels": [...]
  }'
```

## Model Performance

The system achieves competitive accuracy on sleep stage classification:
- Average accuracy: ~90-95%
- REM sleep detection accuracy: >70%
- Results validated across multiple training runs

## Development

### Environment Variables

Key environment variables (configurable in docker-compose.yml):
- `DATA_DIR`: Dataset location
- `MODEL_DIR`: Model checkpoint location
- `NUM_WORKERS`: Training worker processes
- `API_PORT`: Inference server port

### Testing
#### THIS IS ACTUALLY NOT WORKING ATM. TODO! 
Run the test suite:
```bash
pytest
```

