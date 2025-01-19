"""
This module provides functionality for reading and accessing preprocessed HDF5 data files.

The main purpose is to efficiently load different data streams (heart rate, motion, steps, labels) 
from the preprocessed HDF5 files created by the DataPreprocessor. This module provides:

- Clean interface for accessing different sensor data streams
- Type-safe data containers using dataclasses
- Consistent error handling for missing files/data
- Numpy array outputs ready for model training

The module contains:
    1. TimeSeriesData dataclass for storing timestamps and values together
    2. DataReader class that provides methods to read each data stream
    3. Input validation and error handling

The DataReader class is used by the SleepDataset class to load data during model training.
"""

import h5py
import numpy as np
from pathlib import Path
from dataclasses import dataclass


@dataclass
class TimeSeriesData:
    """Container for time series data with timestamps"""
    timestamps: np.ndarray
    values: np.ndarray


class DataReader:
    """Class for reading different types of time series data from HDF5 files"""

    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)

    def read_heart_rate(self, subject_id: str) -> TimeSeriesData:
        """Read heart rate data for a given subject"""
        file_path = self.data_dir / f"{subject_id}.h5"
        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")

        with h5py.File(file_path, 'r') as f:
            return TimeSeriesData(
                timestamps=f['heart_rate/timestamps'][:],
                values=f['heart_rate/values'][:]
            )

    def read_motion(self, subject_id: str) -> TimeSeriesData:
        """Read acceleration data for a given subject"""
        file_path = self.data_dir / f"{subject_id}.h5"
        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")

        with h5py.File(file_path, 'r') as f:
            return TimeSeriesData(
                timestamps=f['motion/timestamps'][:],
                values=f['motion/values'][:]
            )

    def read_steps(self, subject_id: str) -> TimeSeriesData:
        """Read steps data for a given subject"""
        file_path = self.data_dir / f"{subject_id}.h5"
        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")

        with h5py.File(file_path, 'r') as f:
            return TimeSeriesData(
                timestamps=f['steps/timestamps'][:],
                values=f['steps/values'][:]
            )

    def read_labels(self, subject_id: str) -> TimeSeriesData:
        """Read sleep stage labels for a given subject"""
        file_path = self.data_dir / f"{subject_id}.h5"
        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")

        with h5py.File(file_path, 'r') as f:
            return TimeSeriesData(
                timestamps=f['labels/timestamps'][:],
                values=f['labels/values'][:]
            )


if __name__ == "__main__":
    # Example usage
    data_reader = DataReader("./data/formated/")
    subject_id = "1066528"  # Example subject ID

    # Read different types of data
    heart_rate_data = data_reader.read_heart_rate(subject_id)
    motion_data = data_reader.read_motion(subject_id)
    steps_data = data_reader.read_steps(subject_id)
    labels = data_reader.read_labels(subject_id)

    # Print some basic info
    print(f"Heart rate data shape: {heart_rate_data.values.shape}")
    print(f"Motion data shape: {motion_data.values.shape}")
    print(f"Steps data shape: {steps_data.values.shape}")
    print(f"Labels shape: {labels.values.shape}")

    n_samples = 50

    # Print first n_samples from each data stream
    print(f"\nFirst {
          min(n_samples, len(heart_rate_data.timestamps))} heart rate samples:")
    for i in range(min(n_samples, len(heart_rate_data.timestamps))):
        t, v = heart_rate_data.timestamps[i], heart_rate_data.values[i]
        print(f"Time: {t:.2f}s, Heart rate: {v}")

    print(
        f"\nFirst {min(n_samples, len(motion_data.timestamps))} motion samples:")
    for i in range(min(n_samples, len(motion_data.timestamps))):
        t, v = motion_data.timestamps[i], motion_data.values[i]
        print(f"Time: {t:.2f}s, Acceleration (x,y,z): {v}")

    print(
        f"\nFirst {min(n_samples, len(steps_data.timestamps))} steps samples:")
    for i in range(min(n_samples, len(steps_data.timestamps))):
        t, v = steps_data.timestamps[i], steps_data.values[i]
        print(f"Time: {t:.2f}s, Steps: {v}")

    print(f"\nFirst {min(n_samples, len(labels.timestamps))} label samples:")
    for i in range(min(n_samples, len(labels.timestamps))):
        t, v = labels.timestamps[i], labels.values[i]
        print(f"Time: {t:.2f}s, Sleep stage: {v}")

    #### Print Example ####

    # Heart rate data shape: (4878,)
    # Motion data shape: (982000, 3)
    # Steps data shape: (1403,)
    # Labels shape: (567,)

    # First 10 heart rate samples:
    # Time: -556410.36s, Heart rate: 57
    # Time: -556408.36s, Heart rate: 56
    # Time: -556403.36s, Heart rate: 56
    # Time: -556399.36s, Heart rate: 57
    # Time: -556389.36s, Heart rate: 59
    # Time: -556163.88s, Heart rate: 61
    # Time: -555327.01s, Heart rate: 98
    # Time: -554832.63s, Heart rate: 90
    # Time: -554635.13s, Heart rate: 94
    # Time: -554380.12s, Heart rate: 88

    # First 10 motion samples:
    # Time: -124489.16s, Acceleration (x,y,z): [ 0.0174866 -0.5867004 -0.8057709]
    # Time: -124489.12s, Acceleration (x,y,z): [ 0.0189819 -0.5896759 -0.8091583]
    # Time: -124489.12s, Acceleration (x,y,z): [ 0.0209656 -0.5808868 -0.8150482]
    # Time: -124489.11s, Acceleration (x,y,z): [ 0.0194855 -0.5808716 -0.8135834]
    # Time: -124489.10s, Acceleration (x,y,z): [ 0.0169983 -0.587204  -0.8062592]
    # Time: -124489.08s, Acceleration (x,y,z): [ 0.0199585 -0.5930939 -0.8061981]
    # Time: -124489.07s, Acceleration (x,y,z): [ 0.0243988 -0.5862579 -0.8115845]
    # Time: -124489.05s, Acceleration (x,y,z): [ 0.0179291 -0.565567  -0.8039551]
    # Time: -124489.03s, Acceleration (x,y,z): [ 0.0189667 -0.5793762 -0.8106842]
    # Time: -124489.01s, Acceleration (x,y,z): [ 0.0332489 -0.5921173 -0.8071136]

    # First 10 steps samples:
    # Time: -604539.00s, Steps: 0
    # Time: -603939.00s, Steps: 0
    # Time: -603339.00s, Steps: 0
    # Time: -602739.00s, Steps: 0
    # Time: -602139.00s, Steps: 0
    # Time: -601539.00s, Steps: 0
    # Time: -600939.00s, Steps: 0
    # Time: -600339.00s, Steps: 0
    # Time: -599739.00s, Steps: 0
    # Time: -599139.00s, Steps: 28

    # First 10 label samples:
    # Time: 0.00s, Sleep stage: -1
    # Time: 30.00s, Sleep stage: -1
    # Time: 60.00s, Sleep stage: -1
    # Time: 90.00s, Sleep stage: -1
    # Time: 120.00s, Sleep stage: -1
    # Time: 150.00s, Sleep stage: -1
    # Time: 180.00s, Sleep stage: -1
    # Time: 210.00s, Sleep stage: -1
    # Time: 240.00s, Sleep stage: -1
    # Time: 270.00s, Sleep stage: -1
