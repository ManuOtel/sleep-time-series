import pytest
from pathlib import Path
import numpy as np
from src.data.checker import DataChecker
from src.data.reader import TimeSeriesData


@pytest.fixture
def sample_data():
    """Create sample time series data for testing"""
    timestamps = np.arange(0, 100, 5)  # 20 samples, 5 second intervals
    values = np.random.rand(20)
    return TimeSeriesData(timestamps=timestamps, values=values)


@pytest.fixture
def checker(tmp_path):
    """Create DataChecker instance with temporary directory"""
    return DataChecker(str(tmp_path))


def test_check_empty_streams(checker, sample_data):
    """Test empty stream detection"""
    # All streams have data
    valid, msg = checker._check_empty_streams(
        sample_data, sample_data, sample_data, sample_data, "test_subject"
    )
    assert valid
    assert msg == ""

    # Empty heart rate data
    empty_data = TimeSeriesData(timestamps=np.array([]), values=np.array([]))
    valid, msg = checker._check_empty_streams(
        empty_data, sample_data, sample_data, sample_data, "test_subject"
    )
    assert not valid
    assert msg == "Empty heart rate data"


def test_check_timestamps(checker, sample_data):
    """Test timestamp monotonicity checking"""
    # All timestamps monotonic
    valid, msg = checker._check_timestamps(
        sample_data, sample_data, sample_data, sample_data, "test_subject"
    )
    assert valid
    assert msg == ""

    # Non-monotonic heart rate timestamps
    bad_timestamps = np.array([0, 5, 2, 10])  # Out of order
    bad_data = TimeSeriesData(
        timestamps=bad_timestamps, values=np.random.rand(4))
    valid, msg = checker._check_timestamps(
        bad_data, sample_data, sample_data, sample_data, "test_subject"
    )
    assert not valid
    assert msg == "Non-monotonic timestamps in heart rate data"


def test_check_value_ranges(checker, sample_data):
    """Test value range validation"""
    # Create data within normal ranges
    hr_data = TimeSeriesData(
        timestamps=np.arange(5),
        values=np.array([60, 65, 70, 75, 80])  # Normal heart rates
    )
    motion_data = TimeSeriesData(
        timestamps=np.arange(5),
        values=np.array([0.1, 0.2, 0.3, 0.4, 0.5])  # Normal motion values
    )
    steps_data = TimeSeriesData(
        timestamps=np.arange(5),
        values=np.array([100, 200, 300, 400, 500])  # Valid step counts
    )
    labels_data = TimeSeriesData(
        timestamps=np.arange(5),
        values=np.array([0, 1, 2, 3, 4])  # Valid sleep stages
    )

    valid, msg = checker._check_value_ranges(
        hr_data, motion_data, steps_data, labels_data, "test_subject",
        hr_range=(40, 200), motion_range=(-3, 3)
    )
    assert valid
    assert msg == ""

    # Test invalid heart rate values
    hr_data_invalid = TimeSeriesData(
        timestamps=np.arange(5),
        values=np.array([30, 65, 70, 75, 250])  # Outside normal range
    )
    valid, msg = checker._check_value_ranges(
        hr_data_invalid, motion_data, steps_data, labels_data, "test_subject",
        hr_range=(40, 200), motion_range=(-3, 3)
    )
    assert not valid
    assert "Heart rate values outside normal range" in msg
