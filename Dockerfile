# Use PyTorch with CUDA as base image
#FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime
# 🙏🙏 God Help me 🙏🙏


FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    wget \
    unzip \
    dos2unix \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project
COPY . .

# Convert line endings for Python files
RUN find . -type f -name "*.py" -exec dos2unix {} \;
RUN find . -type f -name "*.sh" -exec dos2unix {} \;


# Make the run script executable
RUN chmod +x run.sh

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Set environment variables
ENV DATA_DIR=/app/data \
    VISUALIZATION_DIR=/app/visualizations \
    MODEL_DIR=/app/models \
    LOGS_DIR=/app/logs \
    NUM_WORKERS=4 \
    MAX_PARALLEL=2 \
    API_PORT=6969

# Expose the port for the inference server
EXPOSE 6969

# Command to run the pipeline
CMD ["./run.sh"]