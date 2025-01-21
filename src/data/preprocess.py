"""
This module handles preprocessing and testing of formatted HDF5 data files.

The module processes two types of data:

1. Training Data (Preprocessing):
   - Trims data streams to relevant time periods around sleep labels
   - Normalizes data using training set statistics (mean/std)
   - Used for model training and validation
   - Saved as preprocessed HDF5 files

2. Testing/API Data:
   - Same trimming and cleaning as training data
   - No normalization applied since mean/std unknown at inference time
   - Used for model evaluation and real-world predictions
   - Saved as separate test HDF5 files

The module contains a DataPreprocessor class that:
    1. Takes formatted HDF5 files as input
    2. For preprocessing training data:
        - Trims each data stream to a window around the sleep labels
        - Normalizes data streams using training set statistics
        - Saves the preprocessed data in a new HDF5 file
    3. For testing/API data:
        - Trims data streams similarly but skips normalization step
        - Saves the test data in a separate HDF5 file
    4. Processes entire dataset maintaining same structure for both modes
"""

import h5py
import json
import logging
import numpy as np
from tqdm import tqdm
from pathlib import Path
from reader import DataReader


class TqdmLoggingHandler(logging.Handler):
    def __init__(self, level=logging.NOTSET):
        super().__init__(level)

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)
            # self.flush()
        except Exception:
            self.handleError(record)


# Configure logger with custom handler
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
tqdm_handler = TqdmLoggingHandler()
tqdm_handler.setFormatter(formatter)
logger.addHandler(tqdm_handler)


def set_logging_level(verbose: bool):
    """Set logging level based on verbose flag"""
    if verbose:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.WARNING)


class DataPreprocessor:
    """Preprocesses formatted HDF5 files by trimming to relevant time periods"""

    def __init__(self, data_dir: str,
                 output_dir: str,
                 test_output_dir: str = None,
                 verbose: bool = False):
        """Initialize the DataPreprocessor.

        This constructor sets up a DataPreprocessor instance by:
        1. Creating a DataReader to load the raw data files
        2. Setting up the output directory for preprocessed files
        3. Setting up the output directory for test files (optional)
        4. Creating the output directory if it doesn't exist. 

        Args:
            data_dir: Path to directory containing the raw data files
            output_dir: Path where preprocessed files will be saved
            test_output_dir: Path where test files will be saved (optional)
            verbose: If True, show progress bar and logging info

        Note:
            The output directory will be created if it doesn't exist,
            including any necessary parent directories.
        """
        self.data_reader = DataReader(data_dir, verbose=False)
        self.output_dir = Path(output_dir)
        self.verbose = verbose
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if test_output_dir:
            self.test_output_dir = Path(test_output_dir)
            self.test_output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Initialized DataPreprocessor with data_dir={data_dir} and output_dir={output_dir} and test_output_dir={
                    test_output_dir}") if test_output_dir else logger.info(f"Initialized DataPreprocessor with data_dir={data_dir} and output_dir={output_dir}")

    def _trim_data_to_labels(self, timestamps: np.ndarray, values: np.ndarray,
                             first_label_time: float, last_label_time: float,
                             before_window: float = 0,
                             after_window: float = 0) -> tuple[np.ndarray, np.ndarray]:
        """Trim data arrays to a window around the labels period.

        Args:
            timestamps: Array of timestamps for the data points
            values: Array of corresponding data values (1D or 2D)
            first_label_time: Start time of the label period
            last_label_time: End time of the label period
            before_window: Time in seconds to include before first label (default: 0)
            after_window: Time in seconds to include after last label (default: 0)

        Returns:
            tuple containing:
                - np.ndarray: Trimmed timestamps array
                - np.ndarray: Trimmed values array (same shape as input values)
        """
        start_time = first_label_time - before_window
        end_time = last_label_time + after_window

        # Find indices within the window
        mask = (timestamps >= start_time) & (timestamps <= end_time)

        if self.verbose:
            logger.info(f"Trimming data from {
                        len(timestamps)} to {np.sum(mask)} points")

        return timestamps[mask], values[mask] if len(values.shape) == 1 else values[mask, :]

    def _fix_monotonic_timestamps(self, timestamps: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Fix non-monotonic timestamps by sorting both timestamps and values arrays.

        Args:
            timestamps: Array of timestamps that may not be monotonically increasing
            values: Array of values corresponding to the timestamps

        Returns:
            Tuple of (sorted timestamps array, reordered values array)
        """
        # Get sorting indices based on timestamps
        sort_idx = np.argsort(timestamps)

        # Sort both arrays using these indices
        sorted_timestamps = timestamps[sort_idx]
        sorted_values = values[sort_idx] if len(
            values.shape) == 1 else values[sort_idx, :]

        if self.verbose:
            logger.info("Fixed non-monotonic timestamps")
        return sorted_timestamps, sorted_values

    def _fix_unrealistic_hr_changes(self, timestamps: np.ndarray, values: np.ndarray,
                                    max_change: int = 100) -> tuple[np.ndarray, np.ndarray]:
        """Fix unrealistic heart rate changes by interpolating between valid values.

        When heart rate changes exceed the maximum allowed change between consecutive readings,
        interpolate between the last valid value and next valid value to create more
        realistic transitions.

        Args:
            timestamps: Array of heart rate timestamps
            values: Array of heart rate values in bpm
            max_change: Maximum allowed heart rate change between readings in bpm

        Returns:
            Tuple of (timestamps array, smoothed values array)
        """
        # Find indices where heart rate changes exceed max_change
        hr_changes = np.abs(np.diff(values))
        invalid_idx = np.where(hr_changes > max_change)[0]

        if len(invalid_idx) == 0:
            return timestamps, values

        # Create copy of values to modify
        smoothed_values = values.copy()

        # For each invalid change, interpolate between valid points
        for idx in invalid_idx:
            # Find next valid value
            next_valid_idx = idx + 1
            while (next_valid_idx < len(values)-1 and
                   abs(values[next_valid_idx] - values[idx]) > max_change):
                next_valid_idx += 1

            # Linearly interpolate between valid points
            num_points = next_valid_idx - idx + 1
            smoothed_values[idx:next_valid_idx+1] = np.linspace(
                values[idx], values[next_valid_idx], num_points)

        if self.verbose:
            logger.info(f"Fixed {len(invalid_idx)
                                 } unrealistic heart rate changes")
        return timestamps, smoothed_values

    def _resample_timeseries(self, timestamps: np.ndarray, values: np.ndarray,
                             target_interval: float,
                             method: str = 'linear',
                             max_gap: float = None) -> tuple[np.ndarray, np.ndarray]:
        """Resample time series data to a target interval.

        Args:
            timestamps: Array of timestamps
            values: Array of values corresponding to timestamps
            target_interval: Desired interval between samples in seconds
            method: Method for interpolation:
                   'linear' - Linear interpolation(good for continuous data like HR)
                   'cubic' - Cubic interpolation(good for smooth continuous data)
                   'ffill' - Forward fill(good for categorical data)
                   'nearest' - Nearest neighbor(good for discrete data like steps)
                   'zero' - Zero interpolation(good for sparse event data like steps)
            max_gap: Maximum gap ( in seconds) to interpolate across. Gaps larger than
                    this will be filled with NaN.

        Returns:
            Tuple of(resampled timestamps array, resampled values array)
        """
        if len(timestamps) < 2:
            return timestamps, values

        # Round start and end times to nearest target interval
        start_time = np.floor(
            timestamps[0] / target_interval) * target_interval
        end_time = np.ceil(timestamps[-1] / target_interval) * target_interval
        new_timestamps = np.arange(
            start_time, end_time + target_interval/2, target_interval)

        # For forward fill, use a different approach
        if method == 'ffill':
            indices = np.searchsorted(
                timestamps, new_timestamps, side='right') - 1
            indices = np.clip(indices, 0, len(timestamps) - 1)
            new_values = values[indices]

            if max_gap is not None:
                # Find gaps larger than max_gap
                time_diffs = np.diff(timestamps)
                for i, gap in enumerate(time_diffs):
                    if gap > max_gap:
                        mask = (new_timestamps > timestamps[i]) & (
                            new_timestamps < timestamps[i+1])
                        new_values[mask] = np.nan

            return new_timestamps, new_values

        # For nearest neighbor interpolation
        if method == 'nearest':
            indices = np.searchsorted(timestamps, new_timestamps)
            indices = np.clip(indices, 0, len(timestamps) - 1)
            # Compare distances to left and right points
            left_distances = np.abs(
                new_timestamps - timestamps[np.maximum(indices-1, 0)])
            right_distances = np.abs(
                new_timestamps - timestamps[np.minimum(indices, len(timestamps)-1)])
            use_right = right_distances < left_distances
            indices[use_right] = np.minimum(
                indices[use_right], len(timestamps)-1)
            indices[~use_right] = np.maximum(indices[~use_right]-1, 0)
            new_values = values[indices]

        # For zero interpolation (specific for sparse event data like steps)
        elif method == 'zero':
            if len(values.shape) == 1:
                new_values = np.zeros_like(new_timestamps)
                for i in range(len(timestamps)):
                    # Find the closest timestamp in new grid
                    idx = np.abs(new_timestamps - timestamps[i]).argmin()
                    new_values[idx] = values[i]
            else:
                new_values = np.zeros((len(new_timestamps), values.shape[1]))
                for i in range(len(timestamps)):
                    idx = np.abs(new_timestamps - timestamps[i]).argmin()
                    new_values[idx] = values[i]

        # For linear and cubic interpolation
        elif method in ['linear', 'cubic']:
            if len(values.shape) == 1:
                # For 1D data (e.g., heart rate, steps)
                new_values = np.interp(new_timestamps, timestamps, values)
            else:
                # For multi-dimensional data (e.g., motion)
                new_values = np.zeros((len(new_timestamps), values.shape[1]))
                for j in range(values.shape[1]):
                    new_values[:, j] = np.interp(
                        new_timestamps, timestamps, values[:, j])

        else:
            raise ValueError(f"Unsupported interpolation method: {method}")

        # Handle max_gap for all methods except ffill (already handled)
        if max_gap is not None and method != 'ffill':
            time_diffs = np.diff(timestamps)
            for i, gap in enumerate(time_diffs):
                if gap > max_gap:
                    mask = (new_timestamps > timestamps[i]) & (
                        new_timestamps < timestamps[i+1])
                    if len(values.shape) == 1:
                        new_values[mask] = np.nan
                    else:
                        new_values[mask, :] = np.nan

        if self.verbose:
            logger.info(f"Resampled data from {len(timestamps)} to {len(new_timestamps)} points "
                        f"at {1/target_interval:.3f} Hz using {method} interpolation")

        return new_timestamps, new_values

    def _fix_motion_range(self, timestamps: np.ndarray, values: np.ndarray,
                          motion_range: tuple[float, float] = (-10, 10)) -> tuple[np.ndarray, np.ndarray]:
        """Fix motion values that are outside the expected range by clipping.

        Args:
            timestamps: Array of motion timestamps
            values: Array of motion values in g
            motion_range: Valid range for motion values as (min, max) tuple

        Returns:
            Tuple of(timestamps array, clipped values array)
        """
        # Find values outside valid range
        invalid_mask = ((values < motion_range[0]) | (
            values > motion_range[1]))
        num_invalid = np.sum(invalid_mask)

        if num_invalid == 0:
            return timestamps, values

        # Clip values to valid range
        clipped_values = np.clip(values, motion_range[0], motion_range[1])

        if self.verbose:
            logger.info(f"Clipped {num_invalid} motion values outside range [{
                        motion_range[0]}, {motion_range[1]}]")

        return timestamps, clipped_values

    def _generate_empty_steps_data(self, start_time: float, end_time: float,
                                   sampling_rate: float = 0.002) -> tuple[np.ndarray, np.ndarray]:
        """Generate synthetic steps data filled with zeros when original data is empty

        Args:
            start_time: Start timestamp for synthetic data
            end_time: End timestamp for synthetic data
            sampling_rate: Desired sampling rate in Hz(default 0.002 Hz from info.py)

        Returns:
            Tuple of(timestamps array, values array) for synthetic steps data
        """
        # Calculate number of samples needed
        duration = end_time - start_time
        num_samples = int(duration * sampling_rate)

        # Generate evenly spaced timestamps
        timestamps = np.linspace(start_time, end_time, num_samples)

        # Generate array of zeros for values
        values = np.zeros(num_samples)

        if self.verbose:
            logger.info(f"Generated synthetic steps data with {
                        num_samples} points")
        return timestamps, values

    def _fix_invalid_labels(self, timestamps: np.ndarray, values: np.ndarray,
                            min_duration: float = 120.0) -> tuple[np.ndarray, np.ndarray]:
        """Fix invalid labels(-1) and remove short duration label changes.
        Also maps label 5 to 4 to ensure consistent label range[0-4].

        Args:
            timestamps: Array of label timestamps
            values: Array of label values where - 1 indicates invalid labels
            min_duration: Minimum duration in seconds for a label change to be kept

        Returns:
            Tuple of(timestamps array, fixed values array)
        """
        if len(values) == 0:
            return timestamps, values

        fixed_values = values.copy()

        # First map any 5's to 4's
        fixed_values[fixed_values == 5] = 4

        # Handle invalid labels (-1)
        invalid_mask = (fixed_values == -1)
        num_invalid = np.sum(invalid_mask)

        if -1 not in fixed_values:
            return timestamps, fixed_values

        # Find valid indices
        valid_indices = np.where(~invalid_mask)[0]

        if len(valid_indices) == 0:
            logger.warning("All labels are invalid (-1), cannot fix")
            return timestamps, values

        # For each invalid index, find nearest valid neighbor
        invalid_indices = np.where(invalid_mask)[0]
        for idx in invalid_indices:
            # Find distances to all valid indices
            distances = np.abs(valid_indices - idx)
            nearest_valid_idx = valid_indices[np.argmin(distances)]
            fixed_values[idx] = values[nearest_valid_idx]

        # Remove short duration label changes
        changes = np.where(np.diff(fixed_values) != 0)[0] + 1
        if len(changes) > 1:
            segments = np.split(fixed_values, changes)
            segment_times = np.split(timestamps, changes)

            for i in range(1, len(segments)-1):
                duration = segment_times[i][-1] - segment_times[i][0]
                if duration < min_duration:
                    # Replace short segment with previous label value
                    segments[i][:] = segments[i-1][0]

            fixed_values = np.concatenate(segments)

        num_smoothed = np.sum(fixed_values != values)
        if self.verbose:
            logger.info(f"Fixed {num_invalid} invalid labels and smoothed {
                        num_smoothed-num_invalid} short duration changes")
        return timestamps, fixed_values

    def _normalize_data(self, data: np.ndarray, mean: float = None, std: float = None) -> np.ndarray:
        """Normalize data using mean and standard deviation

        Args:
            data: Array to normalize
            mean: Optional mean value to use. If None, computed from data
            std: Optional standard deviation to use. If None, computed from data

        Returns:
            Normalized array
        """
        # Calculate statistics if not provided
        mean = np.mean(data) if mean is None else mean
        std = np.std(data) if std is None else std

        # Avoid division by zero
        if std == 0:
            logger.warning(
                "Standard deviation is zero, returning zero-centered data")
            return data - mean

        if self.verbose:
            logger.info(f"Normalizing data with mean={
                        mean:.4f}, std={std:.4f}")

        # Apply z-score normalization
        normalized_data = (data - mean) / std
        return normalized_data

    def preprocess_subject_data(self, subject_id: str, test: bool = True):
        """Preprocess and save all data streams for a subject.

        This function:
        1. Reads raw heart rate, motion, steps and sleep label data for the subject
        2. Trims data streams to relevant time periods around sleep labels
        3. Fixes invalid values and non-monotonic timestamps
        4. Resamples data to consistent frequencies:
           - Labels: 1/30 Hz (every 30 seconds)
           - Heart rate: 0.2 Hz (every 5 seconds)
           - Motion/steps: Original frequency
        5. Saves preprocessed data to HDF5 file
        6. Normalizes numeric data streams using z-score normalization
        7. Handles edge cases like missing data and invalid values
        8. Logs progress and any issues encountered during processing

        Args:
            subject_id (str): Unique identifier for the subject to process
            test (bool): If True, preprocess test data without normalization

        Returns:
            None

        Raises:
            FileNotFoundError: If raw data files for subject cannot be found
            ValueError: If data streams contain invalid values that cannot be fixed
        """
        if self.verbose:
            logger.info(f"Starting preprocessing for subject {subject_id}")

        # Determine output path based on whether this is test or training data
        output_file = (
            self.test_output_dir if test else self.output_dir) / f"{subject_id}.h5"
        if self.verbose:
            logger.info(f"Writing preprocessed data to {output_file}")

        try:
            # Read all data streams
            hr_data = self.data_reader.read_heart_rate(subject_id)
            motion_data = self.data_reader.read_motion(subject_id)
            steps_data = self.data_reader.read_steps(subject_id)
            labels_data = self.data_reader.read_labels(subject_id)

            # Get label time bounds
            first_label_time = 0  # Start at time 0
            last_label_time = labels_data.timestamps[-1]

            # Trim each data stream
            with h5py.File(output_file, 'w') as hf:
                # Trim labels to first and last label times
                label_times, label_vals = self._trim_data_to_labels(
                    labels_data.timestamps, labels_data.values,
                    first_label_time, last_label_time
                )

                # Fix any invalid labels (-1) using nearest neighbor interpolation
                label_times, label_vals = self._fix_invalid_labels(
                    label_times, label_vals)

                # Resample labels to 1/30 Hz (every 30 seconds)
                label_times, label_vals = self._resample_timeseries(
                    label_times, label_vals,
                    target_interval=30.0,
                    method='nearest'
                )
                hf.create_dataset('labels/timestamps', data=label_times)
                hf.create_dataset('labels/values', data=label_vals)

                # Process and save heart rate
                hr_times, hr_vals = self._trim_data_to_labels(
                    hr_data.timestamps, hr_data.values,
                    first_label_time, last_label_time
                )
                hr_times, hr_vals = self._fix_monotonic_timestamps(
                    hr_times, hr_vals)
                hr_times, hr_vals = self._fix_unrealistic_hr_changes(
                    hr_times, hr_vals)

                # Resample heart rate to 0.2 Hz
                hr_times, hr_vals = self._resample_timeseries(
                    hr_times, hr_vals,
                    target_interval=5.0,
                    method='nearest'
                )

                # Normalize heart rate data
                if not test:
                    hr_vals = self._normalize_data(hr_vals)

                hf.create_dataset('heart_rate/timestamps', data=hr_times)
                hf.create_dataset('heart_rate/values', data=hr_vals)
                if self.verbose:
                    logger.info(f"Heart rate data reduced from {
                        len(hr_data.timestamps)} to {len(hr_times)} points")

                # Process and save motion
                motion_times, motion_vals = self._trim_data_to_labels(
                    motion_data.timestamps, motion_data.values,
                    first_label_time, last_label_time
                )
                motion_times, motion_vals = self._fix_monotonic_timestamps(
                    motion_times, motion_vals)
                motion_times, motion_vals = self._fix_motion_range(
                    motion_times, motion_vals)

                # Resample motion to 5 Hz because it is too much redundant information
                # leading to overfitting and increased computational cost
                motion_times, motion_vals = self._resample_timeseries(
                    motion_times, motion_vals,
                    target_interval=0.2,  # 1/5 Hz
                    method='nearest'
                )
                hf.create_dataset('motion/timestamps', data=motion_times)
                hf.create_dataset('motion/values', data=motion_vals)
                if self.verbose:
                    logger.info(f"Motion data reduced from {
                        len(motion_data.timestamps)} to {len(motion_times)} points")

                # Process and save steps
                steps_times, steps_vals = self._trim_data_to_labels(
                    steps_data.timestamps, steps_data.values,
                    first_label_time, last_label_time
                )
                if len(steps_times) <= 5:
                    steps_times, steps_vals = self._generate_empty_steps_data(
                        first_label_time, last_label_time)
                else:
                    steps_times, steps_vals = self._fix_monotonic_timestamps(
                        steps_times, steps_vals)

                    # Resample steps to 1/500 Hz (every minute)
                    steps_times, steps_vals = self._resample_timeseries(
                        steps_times, steps_vals,
                        target_interval=500.0,
                        method='nearest'
                    )
                hf.create_dataset('steps/timestamps', data=steps_times)
                hf.create_dataset('steps/values', data=steps_vals)
                if self.verbose:
                    logger.info(f"Steps data reduced from {
                        len(steps_data.timestamps)} to {len(steps_times)} points")
            if self.verbose:
                logger.info(
                    f"Successfully preprocessed data for subject {subject_id}")

        except Exception as e:
            logger.error(f"Error preprocessing data for subject {
                         subject_id}: {str(e)}")

    def preprocess_all_subjects(self, test: bool = True) -> None:
        """Preprocess data for all subjects in the input directory.

        Args:
            test: If True, process data for test set without normalization.
                 If False, process training data with normalization.
        """
        if self.verbose:
            logger.info("Starting preprocessing for all subjects")

        # Get list of all subject files
        subject_files = list(Path(self.data_reader.data_dir).glob("*.h5"))
        subject_ids = [f.stem for f in subject_files]

        if self.verbose:
            logger.info(f"Found {len(subject_ids)} subjects to process")

        # Save subject IDs to JSON file
        subject_ids_path = (Path(self.test_output_dir) if test else Path(
            self.output_dir)) / "subject_ids.json"

        with open(subject_ids_path, "w") as f:
            json.dump(subject_ids, f)

        # Use tqdm for progress bar if verbose
        subjects_iter = tqdm(
            subject_ids, desc="Processing subjects") if self.verbose else subject_ids

        for subject_id in subjects_iter:
            self.preprocess_subject_data(subject_id, test=test)

        if self.verbose:
            logger.info("Completed preprocessing for all subjects")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description='Preprocess and test formatted data')
    parser.add_argument('-d', '--data_dir', type=str, default='./data/formated/',
                        help='Directory containing formatted data')
    parser.add_argument('-o', '--output_dir', type=str, default='./data/preprocessed/',
                        help='Directory to save preprocessed data')
    parser.add_argument('-t', '--test_dir', type=str, default='./data/test/',
                        help='Directory to save test data')
    parser.add_argument('-v', '--verbose', action='store_true', default=False,
                        help='Show progress bar and logging info')
    parser.add_argument('-m', '--mode', type=str, choices=['train', 'test', 'all'],
                        default='all', help='Run test data processing, validation data processing, or both')

    args = parser.parse_args()

    set_logging_level(args.verbose)

    preprocessor = DataPreprocessor(
        args.data_dir, args.output_dir, args.test_dir, verbose=args.verbose)

    if args.mode == 'test':
        preprocessor.preprocess_all_subjects(test=True)
    elif args.mode == 'train':
        preprocessor.preprocess_all_subjects(test=False)
    else:
        preprocessor.preprocess_all_subjects(test=False)
        preprocessor.preprocess_all_subjects(test=True)
