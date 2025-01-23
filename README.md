# Sleep Stage Classification System

A deep learning system for real-time sleep stage prediction using consumer-grade wearable data. This project implements both Transformer and LSTM architectures to classify sleep stages (REM, Deep Sleep, Light Sleep, Awake) based on heart rate, motion, and step data.

## Overview

This system processes wearable device data from the [PhysioNet Sleep-Accel Dataset](https://physionet.org/content/sleep-accel/1.0.0/) to predict sleep stages in real-time. It includes:

- Data preprocessing and validation pipeline
- Model training with both Transformer and LSTM architectures
- Real-time inference API endpoint
- Visualization tools for data analysis
- Comprehensive testing and validation framework
> 📖 For a detailed technical explanation of the system architecture, data processing pipeline, and model performance analysis, please see [README.MD](docs/README.MD).

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
- `docs/` - Detailed documentation and analysis

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

The system achieves strong performance on sleep stage classification:
- LSTM models: 93.74% average test accuracy
- Transformer models: 93.24% average test accuracy
- Best individual run: 96.92% test accuracy (LSTM)

> ⚠️ Note: These accuracy metrics are for single-step prediction using ground truth labels as context. See [README.MD](docs/README.MD) for important details about evaluation methodology.

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

