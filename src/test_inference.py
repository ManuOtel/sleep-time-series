import os
import torch
import asyncio
import aiohttp
import logging
import requests
from fastapi import HTTPException
from src.data.reader import DataReader
from torch.utils.data import DataLoader
from src.data.dataset import SleepDataset


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_data(
    data_dir: str = "./data/test/",
    subject_id: str = "1066528"
) -> dict:
    """Get data from HDF5 files and format it for the inference endpoint.

    Args:
        data_dir: Directory containing the HDF5 data files
        subject_id: ID of the subject to get data for

    Returns:
        Dictionary containing formatted data ready for inference:
            - heart_rate: List of 120 heart rate values
            - motion: List of 3000 motion vectors (each with x,y,z)
            - steps: Single float value for steps
            - previous_labels: List of 19 previous sleep stage labels
    """
    try:
        # Initialize data reader
        reader = DataReader(data_dir)

        # Read raw data
        heart_rate_data = reader.read_heart_rate(subject_id)
        motion_data = reader.read_motion(subject_id)
        steps_data = reader.read_steps(subject_id)
        labels = reader.read_labels(subject_id)

        # Format data for request
        request_data = {
            # First 120 values
            "heart_rate": heart_rate_data.values[:120].tolist(),
            # First 3000 motion vectors
            "motion": motion_data.values[:3000].tolist(),
            "steps": float(steps_data.values[0]),  # First steps value
            "previous_labels": labels.values[:19].tolist()  # First 19 labels
        }

        return request_data

    except Exception as e:
        logger.error(f"Failed to get data: {str(e)}")
        raise


def test_inference_endpoint(
    request_data: dict,
    api_url: str = "http://localhost:6969/predict",
    api_key: str = "your-api-key-here"
) -> dict:
    """Test the inference endpoint with provided request data.

    Args:
        request_data: Dictionary containing formatted data for inference:
            - heart_rate: List of 120 heart rate values 
            - motion: List of 3000 motion vectors (each with x,y,z)
            - steps: Single float value for steps
            - previous_labels: List of 19 previous sleep stage labels
        api_url: URL of the inference endpoint
        api_key: API key for authentication

    Returns:
        Dictionary containing model response with:
            - predicted_class: Integer class prediction
            - class_probabilities: List of class probabilities
            - warning: Optional warning message if data was adjusted
    """

    try:
        # Set up headers with API key
        headers = {
            "X-API-Key": api_key,
            "Content-Type": "application/json"
        }

        # Make request
        logger.info("Sending request to inference endpoint...")
        response = requests.post(api_url, json=request_data, headers=headers)

        # Check response
        if response.status_code == 200:
            result = response.json()
            predicted_class = result['predicted_class']
            logger.info(f"Prediction successful!")
            logger.info(f"Predicted class: {predicted_class}")
            logger.info(f"Class probabilities: {
                        result['class_probabilities']}")
            if "warning" in result:
                logger.warning(f"Warning from server: {result['warning']}")
            return result
        else:
            logger.error(f"Request failed with status code {
                         response.status_code}")
            logger.error(f"Error message: {response.text}")
            raise HTTPException(
                status_code=response.status_code, detail=response.text)

    except Exception as e:
        logger.error(f"Error during testing: {e}")
        raise


async def process_request(
    session: aiohttp.ClientSession,
    request_data: dict,
    true_label: int,
    api_key: str
) -> tuple[int | None, int]:
    """Process a single inference request asynchronously.

    Args:
        session: The aiohttp client session to use for the request
        request_data: Dictionary containing the data for inference with keys:
            - heart_rate: List of heart rate values
            - motion: List of motion vectors
            - steps: Float value for steps
            - previous_labels: List of previous labels
        true_label: The ground truth label for this request
        api_key: API key for authentication

    Returns:
        Tuple containing:
            - Predicted label (int) or None if request failed
            - True label (int)
    """
    headers = {
        "X-API-Key": api_key,
        "Content-Type": "application/json"
    }
    async with session.post("http://localhost:6969/predict", json=request_data, headers=headers) as response:
        if response.status == 200:
            result = await response.json()
            pred_label = result['predicted_class']
            pred_label = 4 if pred_label == 5 else pred_label
            return pred_label, true_label
        else:
            logger.error(f"Request failed with status {response.status}")
            return None, true_label


async def process_batch(
    batch_data: dict,
    batch_labels: torch.Tensor,
    api_key: str,
    max_concurrent: int = 1
) -> list[tuple[int | None, int]]:
    """Process a batch of requests in parallel with rate limiting.

    Args:
        batch_data: Dictionary containing batched input data with keys:
            - heart_rate: Tensor of heart rate values
            - motion: Tensor of motion vectors
            - steps: Tensor of step values
            - previous_labels: Tensor of previous labels
        batch_labels: Tensor of true labels for the batch
        api_key: API key for authentication
        max_concurrent: Maximum number of concurrent requests allowed. Defaults to 1.

    Returns:
        List of tuples, each containing:
            - Predicted label (int) or None if request failed
            - True label (int)
    """
    async with aiohttp.ClientSession() as session:
        tasks = []
        # Process data in chunks based on max_concurrent limit
        for i in range(0, len(batch_data['heart_rate']), max_concurrent):
            chunk_tasks = []
            # Create tasks for current chunk
            for j in range(i, min(i + max_concurrent, len(batch_data['heart_rate']))):
                request_data = {
                    'heart_rate': batch_data['heart_rate'][j].tolist(),
                    'motion': batch_data['motion'][j].tolist(),
                    'steps': float(batch_data['steps'][j].item()),
                    'previous_labels': batch_data['previous_labels'][j].tolist()
                }
                true_label = torch.argmax(batch_labels[j]).item()
                chunk_tasks.append(process_request(
                    session, request_data, true_label, api_key))

            # Wait for current chunk to complete before processing next chunk
            chunk_results = await asyncio.gather(*chunk_tasks)
            tasks.extend(chunk_results)

            # Increased delay between chunks to avoid rate limiting
            await asyncio.sleep(0.5)  # Increased from 0.1 to 0.5

        return tasks

if __name__ == "__main__":
    from dotenv import load_dotenv

    # Load environment variables
    load_dotenv()

    # Get API key from environment
    api_key = os.getenv("MANU_API_KEY")

    # Set up test dataset
    data_dir = "./data/test"
    test_dataset = SleepDataset(
        data_dir=data_dir,
        fold_id=6,
        train_mode=False
    )

    # Print dataset size
    print(f"Test dataset size: {len(test_dataset)} sequences")

    # Create dataloader
    test_loader = DataLoader(
        test_dataset,
        batch_size=128,
        num_workers=4,
        pin_memory=True
    )

    print(f"\nTesting {len(test_dataset)} sequences")
    print("=" * 50)

    # Track metrics
    correct = 0
    total = 0
    class_correct = [0] * 5
    class_total = [0] * 5

    # Process batches
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    for batch_idx, (batch_data, batch_labels) in enumerate(test_loader):
        # if batch_idx >= 100:
        #     break
        if batch_idx % 10 == 0:
            print(f"Processing batch {batch_idx}...")

        # Process batch with reduced concurrency and increased delay
        results = loop.run_until_complete(
            process_batch(batch_data, batch_labels, api_key, max_concurrent=5))

        # Update metrics
        for pred_label, true_label in results:
            if pred_label is not None:
                total += 1
                if pred_label == true_label:
                    correct += 1
                    class_correct[true_label] += 1
                class_total[true_label] += 1

    # Print results
    print("\nTest Results")
    print("=" * 50)
    print(f"Overall Accuracy: {100 * correct / total:.2f}%")
    print("\nPer-class Accuracy:")
    for i in range(5):
        if class_total[i] > 0:
            print(f"Class {i}: {100 * class_correct[i] / class_total[i]:.2f}%")

    # Run test
    # test_inference_endpoint(api_key=api_key)
