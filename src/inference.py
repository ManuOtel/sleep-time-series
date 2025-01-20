import json
import torch
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict
from sleep_classifier import SleepClassifierLSTM

# Initialize FastAPI app
app = FastAPI(
    title="Sleep Stage Classification API",
    description="API for sleep stage classification using LSTM model"
)

# Define input data model


class SleepData(BaseModel):
    heart_rate: List[float]
    motion: List[List[float]]
    steps: float
    previous_labels: List[int]

# Load model


def load_model(model_path: str = "./models/m_e10_lr0.0003_b512_f1/model.pth"):
    try:
        # Initialize model architecture
        model = SleepClassifierLSTM(num_classes=5)

        # Load trained weights
        model.load_state_dict(torch.load(model_path))
        model.eval()

        return model
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {str(e)}")


# Initialize model globally
model = load_model()


@app.post("/predict")
async def predict(data: SleepData):
    try:
        # Convert input data to tensors
        heart_rate = torch.tensor(data.heart_rate).float(
        ).unsqueeze(0)  # Add batch dimension
        motion = torch.tensor(data.motion).float().unsqueeze(0)
        steps = torch.tensor([data.steps]).float().unsqueeze(0)
        previous_labels = torch.tensor(
            data.previous_labels).long().unsqueeze(0)

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
            predicted_class = torch.argmax(probabilities, dim=1).item()

        # Convert probabilities to list
        probs = probabilities[0].tolist()

        return {
            "predicted_class": predicted_class,
            "class_probabilities": probs
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
