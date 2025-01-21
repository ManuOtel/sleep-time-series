import pytest
import numpy as np
import torch
from pathlib import Path
import json
from src.data.dataset import SleepDataset
from src.data.reader import TimeSeriesData


@pytest.fixture
def sample_data_dir(tmp_path):
    """Create temporary directory with sample data files"""
    # Create subject IDs file with at least 10 subjects to avoid division by zero
    subject_ids = [f"subject{i}" for i in range(1, 11)]
    with open(tmp_path / "subject_ids.json", "w") as f:
        json.dump(subject_ids, f)
    return str(tmp_path)


@pytest.fixture
def mock_reader(mocker):
    """Mock DataReader to return sample data"""

    def mock_read_data(self, subject_id, *args):
        if "heart_rate" in str(subject_id):
            # Heart rate data: ~5000 samples at ~2Hz
            # Match real data timestamps
            timestamps = np.arange(-600000, 0) * 0.5
            # HR between 60-80
            values = 60 + np.random.rand(len(timestamps)) * 20
            return TimeSeriesData(timestamps=timestamps, values=values)
        elif "motion" in str(subject_id):
            # Motion data: ~1M samples at 100Hz with 3 axes
            # Match real data timestamps
            timestamps = np.arange(-125000, 0, 0.01)
            values = np.random.randn(
                len(timestamps), 3) * 0.5  # 3-axis acceleration
            return TimeSeriesData(timestamps=timestamps, values=values)
        elif "steps" in str(subject_id):
            # Steps data: ~1400 samples at 0.1Hz
            # Match real data timestamps
            timestamps = np.arange(-605000, 0, 600)
            values = np.random.randint(0, 100, len(timestamps))  # Step counts
            return TimeSeriesData(timestamps=timestamps, values=values)
        else:  # labels
            # Labels: ~600 samples at 1/30Hz
            timestamps = np.arange(0, 18000, 30)  # Match real data timestamps
            values = np.random.randint(
                0, 5, len(timestamps))  # Sleep stages 0-4
            # Convert some labels to 5 to match real data
            values[values == 4] = 5
            return TimeSeriesData(timestamps=timestamps, values=values)

    mocker.patch("src.data.reader.DataReader.read_heart_rate",
                 side_effect=mock_read_data)
    mocker.patch("src.data.reader.DataReader.read_motion",
                 side_effect=mock_read_data)
    mocker.patch("src.data.reader.DataReader.read_steps",
                 side_effect=mock_read_data)
    mocker.patch("src.data.reader.DataReader.read_labels",
                 side_effect=mock_read_data)


def test_dataset_initialization(sample_data_dir, mock_reader):
    """Test basic dataset initialization"""
    dataset = SleepDataset(
        data_dir=sample_data_dir,
        sequence_length=600,  # 10 minutes
        stride=30,  # 30 second stride
        fold_id=0  # Use first fold for testing
    )

    # Basic checks
    assert len(dataset) > 0  # Should have sequences

    # Test getting one item
    sequence, label = dataset[0]

    # Check sequence structure
    assert isinstance(sequence, dict)
    assert 'heart_rate' in sequence
    assert 'motion' in sequence
    assert 'steps' in sequence
    assert 'previous_labels' in sequence

    # Check tensor shapes based on sampling rates
    assert sequence['heart_rate'].shape == torch.Size([120])  # 0.2Hz * 600s
    assert sequence['motion'].shape == torch.Size(
        [3000, 3])  # 5Hz * 600s, 3 axes
    assert sequence['steps'].shape == torch.Size([1])  # 0.002Hz * 600s
    assert sequence['previous_labels'].shape == torch.Size(
        [19])  # 19 previous labels

    # Check label
    assert isinstance(label, torch.Tensor)
    assert label.shape == (5,)  # One-hot encoded with 5 classes
    assert label.sum() == 1  # One-hot encoding should sum to 1
