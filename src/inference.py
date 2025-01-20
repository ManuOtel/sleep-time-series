import os
import torch
import logging
from typing import List, Dict
from dotenv import load_dotenv
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi import Limiter, _rate_limit_exceeded_handler
from sleep_classifier import SleepClassifierLSTM
from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security.api_key import APIKey, APIKeyHeader
from pydantic import BaseModel, field_validator, Field

# Load environment variables
load_dotenv(override=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize rate limiter
limiter = Limiter(key_func=get_remote_address)

# Initialize FastAPI app
app = FastAPI(
    title="Sleep Stage Classification API",
    description="API for sleep stage classification using LSTM model",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Specify allowed origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add rate limiter error handler
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Configure API key authentication
API_KEY = os.getenv('MANU_API_KEY')
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=True)


async def verify_api_key(api_key: APIKey = Depends(api_key_header)):
    if api_key != API_KEY:
        raise HTTPException(
            status_code=403,
            detail="Invalid API key"
        )
    return api_key


# Define input data model with validatio
class SleepData(BaseModel):
    """Input data model for sleep stage classification.

    Validates and processes input data for the sleep classification model.
    All fields are required and have strict validation rules.

    Attributes:
        heart_rate: List of heart rate values in beats per minute (bpm).
            Must contain 1-120 values between 30-200 bpm.
        motion: List of 3D motion vectors (x,y,z accelerometer data).
            Must contain 1-3000 vectors with values between -20 and 20 g.
        steps: Number of steps taken in the time window.
            Must be non-negative.
        previous_labels: List of previous sleep stage classifications.
            Must contain 1-19 labels between 0-4 representing sleep stages.
            0=Wake, 1=N1, 2=N2, 3=N3, 4=REM
    """
    heart_rate: List[float] = Field(
        ...,
        min_items=1,
        max_items=120,
        title="Heart Rate",
        description="Heart rate values in beats per minute (bpm)",
        example=[75, 76, 74]
    )
    motion: List[List[float]] = Field(
        ...,
        min_items=1,
        max_items=3000,
        title="Motion Data",
        description="3D motion vectors from accelerometer (x,y,z) in g",
        example=[[0.1, -0.2, 9.8], [0.2, -0.1, 9.7]]
    )
    steps: float = Field(
        ...,
        ge=0,
        title="Step Count",
        description="Number of steps taken in the time window",
        example=42
    )
    previous_labels: List[int] = Field(
        ...,
        min_items=1,
        max_items=19,
        title="Previous Sleep Stages",
        description="Previous sleep stage classifications (0=Wake, 1=N1, 2=N2, 3=N3, 5=REM)",
        example=[0, 1, 2, 2, 3]
    )

    @field_validator('heart_rate')
    def validate_heart_rate(cls, v):
        if not all(30 <= x <= 200 for x in v):
            raise ValueError(
                "Heart rate values must be between 30 and 200 bpm")
        return v

    @field_validator('motion')
    def validate_motion(cls, v):
        if not all(len(x) == 3 for x in v):
            raise ValueError(
                "Each motion entry must have exactly 3 values (x,y,z)")
        if not all(-20 <= val <= 20 for row in v for val in row):
            raise ValueError("Motion values must be between -20 and 20 g")
        return v

    @field_validator('previous_labels')
    def validate_labels(cls, v):
        if not all(0 <= x <= 4 for x in v):
            raise ValueError("Sleep stage labels must be between 0 and 5")
        return v


def load_model(model_path: str = "./models/m_e10_lr0.0003_b512_f6/model.pth") -> torch.nn.Module:
    """Load and initialize the trained sleep classification model.

    Loads a trained PyTorch model from the specified path and prepares it for inference
    by setting it to evaluation mode.

    Args:
        model_path: Path to the saved model weights file (.pth format).
            Defaults to "./models/m_e10_lr0.0003_b512_f6/model.pth"

    Returns:
        The loaded PyTorch model in evaluation mode

    Raises:
        RuntimeError: If model loading fails due to file not found or incompatible architecture
    """
    try:
        # Initialize model architecture
        model = SleepClassifierLSTM(num_classes=5)

        # Load state dict and remove "_orig_mod." prefix ???? Pytorch?
        state_dict = torch.load(model_path, weights_only=True)
        fixed_state_dict = {
            k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}

        # Load fixed state dict
        model.load_state_dict(fixed_state_dict)
        model.eval()

        return model
    except Exception as e:
        logger.error(f"Model loading failed: {str(e)}")
        raise RuntimeError(f"Failed to load model: {str(e)}")


# Initialize model globally
try:
    model = load_model()
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load model at startup: {str(e)}")
    model = None  # Allow app to start without model for health checks etc.


@app.post("/predict")
@limiter.limit("100/minute")
async def predict(
    request: Request,
    data: SleepData,
    api_key: APIKey = Depends(verify_api_key)
) -> Dict[str, float | List[float]]:
    """Predict sleep stage from physiological data.

    Takes preprocessed heart rate, motion, steps and previous label data and returns
    predicted sleep stage class and class probabilities using trained LSTM model.

    Args:
        data: SleepData object containing:
            - heart_rate: List of heart rate values [120] 
            - motion: List of motion sensor values [3000, 3] for x,y,z axes
            - steps: Step count value [1]
            - previous_labels: List of previous sleep stage labels [19]

    Returns:
        Dictionary containing:
            - predicted_class: Integer indicating predicted sleep stage (0-4)
            - class_probabilities: List of probabilities for each sleep stage class
            - warning: Optional warning message if input shapes are non-ideal

    Raises:
        HTTPException: If model inference fails
    """
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Service temporarily unavailable."
        )
    try:
        # Convert input data to tensors and validate shapes
        heart_rate = torch.tensor(data.heart_rate).float()
        motion = torch.tensor(data.motion).float()
        steps = torch.tensor([data.steps]).float()
        previous_labels = torch.tensor(data.previous_labels).long()

        warning = None

        # Validate and adjust heart rate shape
        if heart_rate.shape[0] < 120:
            heart_rate = torch.nn.functional.pad(
                heart_rate, (0, 120 - heart_rate.shape[0]))
            warning = "Heart rate data padded to required length"
        elif heart_rate.shape[0] > 120:
            heart_rate = heart_rate[:120]
            warning = "Heart rate data truncated to required length"

        # Validate and adjust motion shape
        if motion.shape[0] < 3000 or motion.shape[1] != 3:
            motion = torch.nn.functional.pad(motion, (0, 3 - motion.shape[1] if motion.shape[1] < 3 else 0,
                                                      0, 3000 - motion.shape[0] if motion.shape[0] < 3000 else 0))
            warning = warning or "Motion data padded to required shape"
        elif motion.shape[0] > 3000:
            motion = motion[:3000, :3]
            warning = warning or "Motion data truncated to required shape"

        # Validate and adjust previous labels
        if previous_labels.shape[0] < 19:
            previous_labels = torch.nn.functional.pad(
                previous_labels, (0, 19 - previous_labels.shape[0]))
            warning = warning or "Previous labels padded to required length"
        elif previous_labels.shape[0] > 19:
            previous_labels = previous_labels[:19]
            warning = warning or "Previous labels truncated to required length"

        # Add batch dimension
        heart_rate = heart_rate.unsqueeze(0)
        motion = motion.unsqueeze(0)
        steps = steps.unsqueeze(0)
        previous_labels = previous_labels.unsqueeze(0)
        # Convert any class 5 labels to class 4 in previous labels
        previous_labels = torch.where(previous_labels == 5, 4, previous_labels)

        # Create input dictionary
        model_input = {
            'heart_rate': heart_rate,
            'motion': motion,
            'steps': steps,
            'previous_labels': previous_labels
        }

        # Get prediction
        with torch.no_grad():
            output = model(model_input)
            probabilities = torch.softmax(output, dim=1)
            predicted_class = torch.where(torch.argmax(
                probabilities, dim=1) == 4, 5, torch.argmax(probabilities, dim=1)).item()

        # Convert probabilities to list
        probs = probabilities[0].tolist()

        response = {
            "predicted_class": predicted_class,
            "class_probabilities": probs
        }
        if warning:
            response["warning"] = warning

        logger.info(f"Successful prediction with class {predicted_class}")
        return response

    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


@app.get("/version")
async def version():
    return {
        "version": app.version,
        "model_version": "1.0.0",
        "api_version": "1.0.0"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=6969)
