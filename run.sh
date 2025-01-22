#!/bin/bash

# Exit on any error
set -e

# Set base directories
DATA_DIR="data"
VISUALIZATION_DIR="visualizations"
MODEL_DIR="models"

# Add these missing variables
LOGS_DIR="logs"  # Used in STEPS 6, 7, and 8
NUM_WORKERS=4    # Used in STEP 7 for training
MAX_PARALLEL=2   # Used in STEP 7 for training
API_PORT=8000    # Used in STEP 8 for inference server



################################################################################
        ### STEP 1 - Download the sleep-accel dataset from PhysioNet ###
################################################################################
# Create data directories if they don't exist
mkdir -p "${DATA_DIR}/original"

# Download data from PhysioNet
echo "Downloading sleep-accel dataset from PhysioNet..."
# Check if the zip file or extracted data already exists
if [ ! -f "${DATA_DIR}/original/sleep-accel.zip" ] && [ ! -d "${DATA_DIR}/original/heart_rate" ] && [ ! -d "${DATA_DIR}/original/labels" ] && [ ! -d "${DATA_DIR}/original/motion" ] && [ ! -d "${DATA_DIR}/original/steps" ]; then
    echo "Downloading sleep-accel dataset from PhysioNet..."
    wget -O "${DATA_DIR}/original/sleep-accel.zip" https://physionet.org/static/published-projects/sleep-accel/motion-and-heart-rate-from-a-wrist-worn-wearable-and-labeled-sleep-from-polysomnography-1.0.0.zip
else
    echo "Sleep-accel dataset already exists, skipping download..."
fi
# wget -O "${DATA_DIR}/original/sleep-accel.zip" https://physionet.org/static/published-projects/sleep-accel/motion-and-heart-rate-from-a-wrist-worn-wearable-and-labeled-sleep-from-polysomnography-1.0.0.zip

# Extract files only if they haven't been extracted yet
if [ -f "${DATA_DIR}/original/sleep-accel.zip" ]; then
    # Extract files from zip archive to data/original
    unzip -o "${DATA_DIR}/original/sleep-accel.zip" -d "${DATA_DIR}/original/"
    rm "${DATA_DIR}/original/sleep-accel.zip"

    # Move files from nested folder to data/original
    mv "${DATA_DIR}/original/motion-and-heart-rate-from-a-wrist-worn-wearable-and-labeled-sleep-from-polysomnography-1.0.0/"* "${DATA_DIR}/original/"
    rm -r "${DATA_DIR}/original/motion-and-heart-rate-from-a-wrist-worn-wearable-and-labeled-sleep-from-polysomnography-1.0.0"

    # Clean up cuz I don't like how they look in there (:
    rm -f "${DATA_DIR}/original/LICENSE.txt"
    rm -f "${DATA_DIR}/original/SHA256SUMS.txt"
fi

echo "Data ready in ${DATA_DIR}/original/"
echo "Download complete. Data saved in ${DATA_DIR}/original/"



################################################################################
                   ### STEP 2 - Format the data for training ###
################################################################################
# Create formatted data directory
mkdir -p "${DATA_DIR}/formated"

# Run the data formatting script
echo "Formatting data..."
python3 src/data/formator.py \
    --data_dir "${DATA_DIR}/original" \
    --output_dir "${DATA_DIR}/formated" \
    --verbose

echo "Data formatting complete. Formatted data saved in ${DATA_DIR}/formated/"



################################################################################
        ### STEP 3 - Create Visualizations of the original data ###
################################################################################
# Create visualization directory for original data view
mkdir -p "${VISUALIZATION_DIR}/data/original_view"

# Generate visualizations for 5 random subjects
echo "Generating sample visualizations..."
python3 src/data/visualization.py \
    --data_dir "${DATA_DIR}/formated" \
    --output_dir "${VISUALIZATION_DIR}/data/original_view" \
    --num_subjects 5 \
    --subject_ids_file "${DATA_DIR}/formated/subject_ids.json" \
    --verbose

echo "Sample visualizations complete. Saved in ${VISUALIZATION_DIR}/data/original_view/"



################################################################################
                ### STEP 4 - Preprocess the formated data ###
################################################################################
# Create preprocessed data directories
mkdir -p "${DATA_DIR}/preprocessed"
mkdir -p "${DATA_DIR}/test"

# Run the data preprocessing script to create both datasets
echo "Preprocessing data..."
python3 src/data/preprocess.py \
    --data_dir "${DATA_DIR}/formated" \
    --output_dir "${DATA_DIR}/preprocessed" \
    --test_dir "${DATA_DIR}/test" \
    --mode all \
    --verbose

echo "Data preprocessing complete."
echo "Training data saved in ${DATA_DIR}/preprocessed/"
echo "Test data saved in ${DATA_DIR}/test/"



################################################################################
        ### STEP 5 - Create Visualizations of the preprocessed data ###
################################################################################
# Create visualization directory for preprocessed data view
mkdir -p "${VISUALIZATION_DIR}/data/preprocessed_view"

# Generate visualizations for 5 random subjects
echo "Generating sample visualizations..."
python3 src/data/visualization.py \
    --data_dir "${DATA_DIR}/preprocessed" \
    --output_dir "${VISUALIZATION_DIR}/data/preprocessed_view" \
    --num_subjects 5 \
    --subject_ids_file "${DATA_DIR}/formated/subject_ids.json" \
    --verbose

echo "Sample visualizations complete. Saved in ${VISUALIZATION_DIR}/data/preprocessed_view/"



################################################################################
            ### STEP 6 - Check the data for invalid samples ###
################################################################################
# Create directory for data check logs
mkdir -p "${LOGS_DIR}/data_check"

# Run data checker on test data and modify both test and preprocessed data
echo "Checking data quality..."
python3 src/data/checker.py \
    --data_dir "${DATA_DIR}/test" \
    --modify_invalid \
    --siamese_dir "${DATA_DIR}/preprocessed" \
    --verbose > "${LOGS_DIR}/data_check/check_results.txt"

echo "Data quality check complete. Results saved in ${LOGS_DIR}/data_check/check_results.txt"
echo "Invalid files have been renamed with 'INVALID_' prefix in both test and preprocessed directories"



################################################################################
        ### STEP 7 - Start the training process for the model ###
################################################################################
# Create directories for model outputs and logs
mkdir -p "${MODEL_DIR}"
mkdir -p "${LOGS_DIR}/training"

# Start training process
echo "Starting model training..."
python3 src/train.py \
    --data_dir "${DATA_DIR}/preprocessed" \
    --num_workers "${NUM_WORKERS}" \
    --max_parallel "${MAX_PARALLEL}" \
    --config config/training_config.yaml \
    --experiment manu_test \
    2>&1 | tee "${LOGS_DIR}/training/training.log" &

# Get the PID of the training process
TRAIN_PID=$!

echo "Training process started with PID: $TRAIN_PID"
echo "Training logs being written to ${LOGS_DIR}/training/training.log"

# Wait for training process to complete
echo "Waiting for training to complete..."
wait $TRAIN_PID

# Check if training completed successfully
if [ $? -eq 0 ]; then
    echo "Training completed successfully"
else
    echo "Training failed with exit code $?"
    exit 1
fi



################################################################################
        ### STEP 8 - Start an inference endpoint for the model ###
################################################################################
# Create directory for inference logs
mkdir -p "${LOGS_DIR}/inference"

# Generate API key
echo "Generating API key..."
python3 keygen.py --length 32
if [ $? -ne 0 ]; then
    echo "Failed to generate API key"
    exit 1
fi

# Display API key for user
API_KEY=$(grep MANU_API_KEY .env | cut -d"'" -f2)
echo "Generated API key: $API_KEY"
echo "This key has been saved to .env and will be needed for testing the inference endpoint"

# Start the inference API server
echo "Starting inference API server..."
python3 src/inference.py \
    --port "${API_PORT}" \
    2>&1 | tee "${LOGS_DIR}/inference/inference.log" &

# Get the PID of the inference process
INFERENCE_PID=$!

echo "Inference API server started with PID: $INFERENCE_PID"
echo "Inference logs being written to ${LOGS_DIR}/inference/inference.log"
echo "API server running at http://localhost:${API_PORT}"
echo "You can now use test_inference.py with the API key shown above"

# Wait for inference process
echo "Inference server is running... Press Ctrl+C to stop"
wait $INFERENCE_PID

# Check if inference server stopped cleanly
if [ $? -eq 0 ]; then
    echo "Inference server stopped successfully"
else
    echo "Inference server stopped with exit code $?"
    exit 1
fi
