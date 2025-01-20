import requests
from data.reader import DataReader
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_inference_endpoint(
    data_dir: str = "./data/preprocessed/",
    subject_id: str = "1066528",
    api_url: str = "http://localhost:6969/predict",
    api_key: str = "your-api-key-here"
):
    """Test the inference endpoint with real data from HDF5 files."""

    try:
        # Initialize data reader
        reader = DataReader(data_dir)

        # Read data for the subject
        heart_rate_data = reader.read_heart_rate(subject_id)
        motion_data = reader.read_motion(subject_id)
        steps_data = reader.read_steps(subject_id)
        labels = reader.read_labels(subject_id)

        # Get actual label to test against
        actual_label = labels.values[19]  # 20th label

        # Prepare data for request (taking a slice of data)
        request_data = {
            # Take first 120 values
            "heart_rate": heart_rate_data.values[:120].tolist(),
            # Take first 3000 motion vectors
            "motion": motion_data.values[:3000].tolist(),
            "steps": float(steps_data.values[0]),  # Take first steps value
            # Take first 19 labels
            "previous_labels": labels.values[:19].tolist()
        }

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
            logger.info(f"Actual class: {actual_label}")
            logger.info(
                f"Prediction {'correct' if predicted_class == actual_label else 'incorrect'}")
            logger.info(f"Class probabilities: {
                        result['class_probabilities']}")
            if "warning" in result:
                logger.warning(f"Warning from server: {result['warning']}")
        else:
            logger.error(f"Request failed with status code {
                         response.status_code}")
            logger.error(f"Error message: {response.text}")

    except FileNotFoundError as e:
        logger.error(f"Data file not found: {e}")
    except Exception as e:
        logger.error(f"Error during testing: {e}")


if __name__ == "__main__":
    import os
    from dotenv import load_dotenv

    # Load environment variables
    load_dotenv()

    # Get API key from environment
    api_key = os.getenv("MANU_API_KEY")

    # Run test
    test_inference_endpoint(api_key=api_key)
