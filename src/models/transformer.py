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
    model = SleepClassifierTransformer(num_classes=5)
    output = model(example_data)

    # Print model validation info
    print("SleepClassifierTransformer Model Validation")
    print("-" * 40)
    print("Input Tensor Shapes:")
    for key, tensor in example_data.items():
        print(f"{key:15s}: {list(tensor.shape)}")

    print("\nOutput Tensor Shape:", list(output.shape))
    print("\nSample Output Predictions:")
    print(output)
    print("\nValidation completed successfully.")
    # Print parameter counts and memory footprint

    transformer_params = sum(p.numel()
                             for p in model.parameters() if p.requires_grad)
    # Approximate size in bytes (32-bit floats)
    transformer_size = transformer_params * 4

    print("\nModel Statistics:")
    print("-" * 40)
    print(f"Transformer Model:")
    print(f"Parameters: {transformer_params:,}")
    print(f"Approx Memory: {transformer_size/1024/1024:.2f} MB")

    #### Print Example ####
    # SleepClassifierTransformer Model Validation
    # ----------------------------------------
    # Input Tensor Shapes:
    # heart_rate     : [2, 120]
    # motion         : [2, 3000, 3]
    # steps          : [2, 1]
    # previous_labels: [2, 19]

    # Output Tensor Shape: [2, 5]

    # Sample Output Predictions:
    # tensor([[-0.1817,  0.0102,  0.1498,  0.5020,  0.1781],
    #         [-0.1213,  0.0157, -0.0608,  0.6189,  0.0165]],
    #        grad_fn=<AddmmBackward0>)

    # Validation completed successfully.

    # Model Statistics:
    # ----------------------------------------
    # Transformer Model:
    # Parameters: 236,901
    # Approx Memory: 0.90 MB
