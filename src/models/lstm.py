"""
This module implements the LSTM-based sleep stage classifier model.

The SleepClassifierLSTM model processes multimodal physiological data to classify sleep stages:

Key components:
    1. Heart Rate Processing:
        - CNN layers extract temporal features from heart rate signals
        - Bidirectional LSTM captures sequential patterns
        
    2. Motion Processing:
        - 2D CNN processes tri-axial accelerometer data
        - Bidirectional LSTM models motion sequences
        
    3. Additional Features:
        - Step count integration
        - Previous sleep stage label embedding
        
    4. Feature Fusion:
        - Concatenates features from all modalities
        - Dense layers for final classification
        
The architecture prioritizes:
    - Efficient feature extraction through strided convolutions
    - Temporal modeling with bidirectional LSTMs
    - Regularization via dropout and batch normalization
    - Balanced model capacity through reduced hidden sizes
    
The model outputs 5-class sleep stage predictions:
    0: Wake
    1: Light Sleep (N1/N2)
    2: Deep Sleep (N2)
    3: N3 Sleep
    4: REM Sleep
"""

import torch
import torch.nn as nn


class SleepClassifierLSTM(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()

        # Simplified heart rate CNN
        self.hr_cnn = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, stride=2,
                      padding=2),  # Increased stride
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=4, stride=4),  # Larger pooling
            nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )

        # Simpler LSTM
        self.hr_lstm = nn.LSTM(
            input_size=64,
            hidden_size=32,  # Reduced size
            num_layers=1,    # Single layer
            batch_first=True,
            bidirectional=True
        )

        # Simplified motion CNN
        self.motion_cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(5, 3),
                      stride=(2, 1), padding=(2, 0)),  # Increased stride
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(4, 1), stride=(4, 1)),  # Larger pooling
            nn.Conv2d(32, 64, kernel_size=(3, 1),
                      stride=(1, 1), padding=(1, 0)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
            nn.Conv2d(64, 128, kernel_size=(3, 1),
                      stride=(1, 1), padding=(1, 0)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
        )

        # Simpler LSTM
        self.motion_lstm = nn.LSTM(
            input_size=128,
            hidden_size=64,  # Reduced size
            num_layers=1,    # Single layer
            batch_first=True,
            bidirectional=True
        )

        # Previous labels embedding with dropout and batch norm for regularization
        self.label_embedding = nn.Sequential(
            nn.Linear(19, 64),  # Wider layer for more capacity
            nn.BatchNorm1d(64),  # Normalize activations
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),  # Prevent overfitting
            nn.Linear(64, 32),  # Project down to final embedding size
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True)
        )

        # Simpler classifier
        # HR (32*2) + Motion (64*2) + Steps + Previous Labels
        combined_size = 64 + 128 + 1 + 32
        self.classifier = nn.Sequential(
            nn.Linear(combined_size, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes)  # No softmax needed for classification
        )

    def forward(self, x):
        # Heart rate processing
        hr = x['heart_rate'].unsqueeze(1)
        hr = self.hr_cnn(hr)
        hr = hr.transpose(1, 2)
        hr_out, _ = self.hr_lstm(hr)
        hr_feat = hr_out[:, -1, :]

        # Motion processing
        motion = x['motion'].unsqueeze(1)
        motion_cnn = self.motion_cnn(motion)
        motion_cnn = motion_cnn.squeeze(-1).transpose(1, 2)
        motion_out, _ = self.motion_lstm(motion_cnn)
        motion_feat = motion_out[:, -1, :]

        # Steps processing
        steps_feat = x['steps']

        # Previous labels 19 processing
        label_feat = self.label_embedding(
            x['previous_labels'].float())  # [B, 19] -> [B, 32]

        # Combine and classify
        combined = torch.cat(
            [hr_feat, motion_feat, steps_feat, label_feat], dim=1)
        return self.classifier(combined)


if __name__ == "__main__":
    # Create sample batch
    batch_size = 2
    example_data = {
        # Heart rate signal [B, 120]
        'heart_rate': torch.randn(batch_size, 120),
        # Motion data [B, 3000, 3]
        'motion': torch.randn(batch_size, 3000, 3),
        'steps': torch.randn(batch_size, 1),               # Step count [B, 1]
        # Previous stages [B, 19]
        'previous_labels': torch.randint(0, 4, (batch_size, 19))
    }

    # Initialize model and run forward pass
    model = SleepClassifierLSTM(num_classes=5)
    output = model(example_data)

    # Print model validation info
    print("SleepClassifierLSTM Model Validation")
    print("-" * 40)
    print("Input Tensor Shapes:")
    for key, tensor in example_data.items():
        print(f"{key:15s}: {list(tensor.shape)}")

    print("\nOutput Tensor Shape:", list(output.shape))
    print("\nSample Output Predictions:")
    print(output)
    print("\nValidation completed successfully.")

    lstm_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    lstm_size = lstm_params * 4  # Approximate size in bytes (32-bit floats)

    print("\nModel Statistics:")
    print("-" * 40)
    print(f"LSTM Model:")
    print(f"Parameters: {lstm_params:,}")
    print(f"Approx Memory: {lstm_size/1024/1024:.2f} MB")

    #### Print Example ####
    # SleepClassifierLSTM Model Validation
    # ----------------------------------------
    # Input Tensor Shapes:
    # heart_rate     : [2, 120]
    # motion         : [2, 3000, 3]
    # steps          : [2, 1]
    # previous_labels: [2, 19]

    # Output Tensor Shape: [2, 5]

    # Sample Output Predictions:
    # tensor([[ 1.3298e-01,  1.2268e-02, -4.4489e-02,  1.2485e-01,  1.1771e-01],
    #         [ 8.6893e-02, -7.1910e-05, -1.2649e-01,  4.2309e-03,  1.4474e-01]],
    #        grad_fn=<AddmmBackward0>)

    # Validation completed successfully.

    # Model Statistics:
    # ----------------------------------------
    # LSTM Model:
    # Parameters: 195,365
    # Approx Memory: 0.75 MB
