"""
The main purpose is to classify sleep stages using multimodal physiological data.

The module contains two model architectures for sleep stage classification:

1. SleepClassifierLSTM:
    - Takes preprocessed heart rate and motion data as input
    - Uses CNN layers for feature extraction from each modality
    - Employs bidirectional LSTM layers for temporal modeling
    - Processes previous labels through an embedding layer
    - Fuses multimodal features for final classification
    - Outputs sleep stage predictions (5 classes)

2. SleepClassifierTransformer:
    - Takes the same preprocessed inputs as the LSTM version
    - Uses identical CNN feature extractors for each modality
    - Uses Transformer encoder layers for temporal modeling (no decoder)
    - The encoder-only architecture focuses on self-attention patterns
    - Processes previous labels through an embedding layer
    - Fuses features from all modalities for classification
    - Outputs sleep stage predictions (5 classes)
    - Provides interpretable attention weights for analysis

Both architectures share:
    - CNN-based feature extraction pipelines
    - Previous label embedding components
    - Multi-modal feature fusion approach
    - Final classification head

The LSTM version offers efficient training and good performance on sequential data,
while the Transformer version can capture more complex temporal dependencies through
its self-attention mechanism, though using only the encoder portion for simplicity.
"""
import torch
import torch.nn as nn


class SleepClassifierTransformer(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()

        # Simplified heart rate branch
        self.hr_cnn = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=4, stride=4),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )

        # Simpler transformer with fewer layers and heads, only encodder
        self.hr_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=64,
                nhead=4,
                dim_feedforward=128,
                dropout=0.1,
                batch_first=True
            ),
            num_layers=1  # Single layer
        )

        # Simplified motion branch
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

        # Simpler transformer
        self.motion_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=128,
                nhead=4,  # Reduced heads
                dim_feedforward=256,  # Reduced feedforward
                dropout=0.1,
                batch_first=True
            ),
            num_layers=1  # Single layer
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
        combined_size = 64 + 128 + 1 + 32  # HR + Motion + Steps + Previous Labels
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
        hr_out = self.hr_transformer(hr)
        hr_feat = hr_out[:, -1, :]

        # Motion processing
        motion = x['motion'].unsqueeze(1)
        motion_cnn = self.motion_cnn(motion)
        motion_cnn = motion_cnn.squeeze(-1).transpose(1, 2)
        motion_out = self.motion_transformer(motion_cnn)
        motion_feat = motion_out[:, -1, :]

        # Steps processing
        steps_feat = x['steps']

        # Previous all 19 labels processing
        label_feat = self.label_embedding(
            x['previous_labels'].float())  # [B, 19] -> [B, 32]

        # Combine and classify
        combined = torch.cat(
            [hr_feat, motion_feat, steps_feat, label_feat], dim=1)
        return self.classifier(combined)


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
    # Test both LSTM and Transformer versions
    batch_size = 2
    example_data = {
        'heart_rate': torch.randn(batch_size, 120),      # [B, 120]
        'motion': torch.randn(batch_size, 3000, 3),      # [B, 3000, 3]
        'steps': torch.randn(batch_size, 1),             # [B, 1]
        'previous_labels': torch.randint(0, 4, (batch_size, 19))  # [B, 19]
    }

    # Test LSTM version
    print("Testing LSTM version:")
    model_lstm = SleepClassifierLSTM(num_classes=5)
    output_lstm = model_lstm(example_data)
    print(f"Input shapes:")
    print(f"Heart rate: {example_data['heart_rate'].shape}")
    print(f"Motion: {example_data['motion'].shape}")
    print(f"Steps: {example_data['steps'].shape}")
    print(f"Previous labels: {example_data['previous_labels'].shape}")
    print(f"\nOutput shape: {output_lstm.shape}")  # Should be [2, 5]
    print(f"Output example:\n{output_lstm}")

    # Test Transformer version
    print("\nTesting Transformer version:")
    model_transformer = SleepClassifierTransformer(num_classes=5)
    output_transformer = model_transformer(example_data)
    print(f"Input shapes:")
    print(f"Heart rate: {example_data['heart_rate'].shape}")
    print(f"Motion: {example_data['motion'].shape}")
    print(f"Steps: {example_data['steps'].shape}")
    print(f"\nOutput shape: {output_transformer.shape}")  # Should be [2, 5]
    print(f"Output example:\n{output_transformer}")

    #### Print Example ####
    # Input shapes:
    # Heart rate: torch.Size([2, 120])
    # Motion: torch.Size([2, 3000, 3])
    # Steps: torch.Size([2, 1])

    # Output shape: torch.Size([2, 5])
    # Output example:
    # tensor([[-0.0577, -0.0627, -0.1462, -0.0631,  0.0485],
    #         [-0.0983, -0.1588, -0.1296, -0.1075,  0.0866]],
    #     grad_fn=<AddmmBackward0>)
