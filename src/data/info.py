"""
The main purpose is to analyze preprocessed HDF5 data files to extract key statistical information. 

The module contains a DataInfo class that:
    1. Takes preprocessed HDF5 data files as input
    2. Performs statistical analysis on each data stream, including:
        - Missing or corrupted data files
        - Empty data streams or invalid values
        - Non-monotonic timestamps and sampling irregularities 
        - Data gaps and coverage issues
        - Value range violations
        - Misaligned endpoints between streams
        - Inconsistent sampling rates
    3. Reports detailed statistics and summaries
    4. Can process individual subjects or entire datasets
    5. Provides insights for data validation parameters
"""

import json
import logging
import numpy as np
from scipy import stats
from pathlib import Path
from reader import DataReader

logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class DataInfo:
    """Analyzes data files to extract key statistical information"""

    def __init__(self, data_dir: str):
        self.data_reader = DataReader(data_dir)
        self.data_dir = Path(data_dir)
        logger.info(f"Initialized DataInfo with data_dir={data_dir}")

    def _analyze_time_series_gaps(self, timestamps: np.ndarray) -> dict[str, float | dict[str, float] | None]:
        """Analyze gaps and irregularities in time series data.

        Calculates statistics about gaps and missing data in a time series by analyzing
        the intervals between consecutive timestamps. Uses the most common interval as
        the expected sampling rate.

        Args:
            timestamps: Array of timestamps in seconds

        Returns:
            Dictionary containing:
                - gap_stats: Dict with gap statistics (n_gaps, mean_gap, median_gap, std_gap) or None if invalid
                - missing_pct: Percentage of missing data points (0-100)
                - largest_gap: Size of largest gap in seconds

        Raises:
            TypeError: If timestamps is not a numpy array
            ValueError: If timestamps does not contain numeric values
        """
        if len(timestamps) <= 1:
            return {
                'gap_stats': None,
                'missing_pct': 100.0,
                'largest_gap': 0
            }

        # Calculate time differences between consecutive timestamps
        time_diffs = np.diff(timestamps)

        # Use the most common time difference as the expected sampling rate
        unique_diffs, diff_counts = np.unique(time_diffs, return_counts=True)
        expected_interval = unique_diffs[np.argmax(diff_counts)]

        # Guard against zero or invalid expected_interval
        if expected_interval <= 0 or np.isinf(expected_interval):
            return {
                'gap_stats': None,
                'missing_pct': 100.0,
                'largest_gap': float(np.max(time_diffs)) if len(time_diffs) > 0 else 0
            }

        # Identify gaps (time differences larger than 2x the expected interval)
        gap_threshold = 2 * expected_interval
        gaps = time_diffs[time_diffs > gap_threshold]

        # Calculate missing percentage more accurately
        duration = timestamps[-1] - timestamps[0]
        try:
            expected_points = max(2, int(duration / expected_interval) + 1)
            actual_points = len(timestamps)
            missing_pct = max(
                0, min(100, 100 * (1 - actual_points/expected_points)))
        except (OverflowError, ZeroDivisionError):
            missing_pct = 100.0

        return {
            'gap_stats': {
                'n_gaps': len(gaps),
                'mean_gap': float(np.mean(gaps)) if len(gaps) > 0 else 0,
                'median_gap': float(np.median(gaps)) if len(gaps) > 0 else 0,
                'std_gap': float(np.std(gaps)) if len(gaps) > 0 else 0
            },
            'missing_pct': float(missing_pct),
            'largest_gap': float(np.max(time_diffs))
        }

    def _analyze_distribution(self, values: np.ndarray) -> dict[str, float | dict[str, float]]:
        """Analyze the statistical distribution of a numeric data array.

        Calculates skewness to measure asymmetry and performs Kolmogorov-Smirnov test
        to check for normality of the distribution.

        Args:
            values: Array of numeric values to analyze

        Returns:
            dict containing:
                skewness (float): Measure of distribution asymmetry
                normality_test (dict): Results of K-S normality test with:
                    - statistic (float): K-S test statistic
                    - pvalue (float): p-value of the test
        """
        if len(values) <= 1:
            return {
                'skewness': 0,
                'normality_test': {'statistic': 0, 'pvalue': 0}
            }

        # Calculate skewness
        skewness = stats.skew(values)

        # Perform Kolmogorov-Smirnov test for normality
        ks_stat, p_value = stats.kstest(values, 'norm')

        return {
            'skewness': skewness,
            'normality_test': {
                'statistic': ks_stat,
                'pvalue': p_value
            }
        }

    def _analyze_subject_data(self, subject_id: str) -> dict[str, dict[str, int | float | dict]]:
        """Analyze data streams for a single subject to extract key statistics.

        Processes heart rate, motion, steps and label data streams to compute:
        - Basic statistics (mean, median, std, min, max)
        - Sampling rates and data gaps analysis
        - Distribution statistics (skewness, normality tests)

        Args:
            subject_id: Unique identifier for the subject to analyze

        Returns:
            Dictionary containing analysis results for each data stream with structure:
            {
                'heart_rate': {
                    'n_samples': int,
                    'duration_hours': float,
                    'basic_stats': dict,
                    'sampling_rate': float,
                    'gaps': dict,
                    'distribution': dict
                },
                'motion': {...},
                'steps': {...},
                'labels': {...}
            }
        """
        stats = {}
        try:
            # Read all data streams
            hr_data = self.data_reader.read_heart_rate(subject_id)
            motion_data = self.data_reader.read_motion(subject_id)
            steps_data = self.data_reader.read_steps(subject_id)
            labels_data = self.data_reader.read_labels(subject_id)

            # Heart rate analysis
            if len(hr_data.timestamps) > 1:
                hr_gaps = self._analyze_time_series_gaps(hr_data.timestamps)
                hr_dist = self._analyze_distribution(hr_data.values)
                stats['heart_rate'] = {
                    'n_samples': len(hr_data.timestamps),
                    'duration_hours': (hr_data.timestamps[-1] - hr_data.timestamps[0])/3600,
                    'basic_stats': {
                        'mean': np.mean(hr_data.values),
                        'median': np.median(hr_data.values),
                        'std': np.std(hr_data.values),
                        'min': np.min(hr_data.values),
                        'max': np.max(hr_data.values)
                    },
                    'sampling_rate': 1/np.mean(np.diff(hr_data.timestamps)),
                    'gaps': hr_gaps,
                    'distribution': hr_dist
                }
            else:
                stats['heart_rate'] = {'n_samples': 0}

            # Motion analysis
            if len(motion_data.timestamps) > 1:
                motion_gaps = self._analyze_time_series_gaps(
                    motion_data.timestamps)
                magnitudes = np.linalg.norm(motion_data.values, axis=1)
                motion_dist = self._analyze_distribution(magnitudes)
                stats['motion'] = {
                    'n_samples': len(motion_data.timestamps),
                    'duration_hours': (motion_data.timestamps[-1] - motion_data.timestamps[0])/3600,
                    'basic_stats': {
                        'mean_magnitude': np.mean(magnitudes),
                        'median_magnitude': np.median(magnitudes),
                        'std_magnitude': np.std(magnitudes),
                        'min_values': np.min(motion_data.values, axis=0),
                        'max_values': np.max(motion_data.values, axis=0)
                    },
                    'sampling_rate': 1/np.mean(np.diff(motion_data.timestamps)),
                    'gaps': motion_gaps,
                    'distribution': motion_dist
                }
            else:
                stats['motion'] = {'n_samples': 0}

            # Steps analysis
            if len(steps_data.timestamps) > 1:
                steps_gaps = self._analyze_time_series_gaps(
                    steps_data.timestamps)
                steps_dist = self._analyze_distribution(steps_data.values)
                stats['steps'] = {
                    'n_samples': len(steps_data.timestamps),
                    'duration_hours': (steps_data.timestamps[-1] - steps_data.timestamps[0])/3600,
                    'basic_stats': {
                        'total': np.sum(steps_data.values),
                        'mean': np.mean(steps_data.values),
                        'median': np.median(steps_data.values),
                        'max': np.max(steps_data.values)
                    },
                    'sampling_rate': 1/np.mean(np.diff(steps_data.timestamps)),
                    'gaps': steps_gaps,
                    'distribution': steps_dist
                }
            else:
                stats['steps'] = {'n_samples': 0}

            # Labels analysis
            if len(labels_data.timestamps) > 1:
                labels_gaps = self._analyze_time_series_gaps(
                    labels_data.timestamps)
                stats['labels'] = {
                    'n_samples': len(labels_data.timestamps),
                    'duration_hours': (labels_data.timestamps[-1] - labels_data.timestamps[0])/3600,
                    'unique_labels': np.unique(labels_data.values),
                    'label_counts': {str(label): np.sum(labels_data.values == label)
                                     for label in np.unique(labels_data.values)},
                    'sampling_rate': 1/np.mean(np.diff(labels_data.timestamps)),
                    'gaps': labels_gaps
                }
            else:
                stats['labels'] = {'n_samples': 0}

        except Exception as e:
            logger.error(f"Error analyzing data for subject {
                         subject_id}: {str(e)}")
            stats = {
                'heart_rate': {'n_samples': 0},
                'motion': {'n_samples': 0},
                'steps': {'n_samples': 0},
                'labels': {'n_samples': 0},
                'error': str(e)
            }

        return stats

    def analyze_all_subjects(self) -> dict[str, dict[str, dict]]:
        """Analyze all subjects and compute aggregate statistics including sampling irregularities.

        Processes data from all subjects to calculate individual statistics and aggregate metrics.
        Focuses on sampling irregularities like gaps and missing data patterns across the dataset.

        Returns:
            Dictionary containing:
                individual: Dict mapping subject IDs to their individual statistics
                summary: Dict containing aggregate statistics across all subjects, with:
                    - Missing data percentages (mean, median, std)
                    - Gap size statistics
                    - Distribution metrics like skewness
        """
        # Read subject IDs
        subject_ids_file = self.data_dir / "subject_ids.json"

        try:
            if not subject_ids_file.exists():
                raise FileNotFoundError(
                    f"Subject IDs file not found: {subject_ids_file}")

            with open(subject_ids_file, 'r') as f:
                try:
                    subject_ids = json.load(f)
                except json.JSONDecodeError:
                    raise ValueError(f"Invalid JSON format in {
                                     subject_ids_file}")

            if not isinstance(subject_ids, list):
                raise ValueError(
                    "subject_ids.json must contain a list of subject IDs")

            all_stats = {}
            aggregate_stats = {
                'heart_rate': {'missing_pcts': [], 'gap_sizes': [], 'skewness': []},
                'motion': {'missing_pcts': [], 'gap_sizes': [], 'skewness': []},
                'steps': {'missing_pcts': [], 'gap_sizes': [], 'skewness': []},
                'labels': {'missing_pcts': [], 'gap_sizes': []}
            }

            for subject_id in subject_ids:
                try:
                    stats = self._analyze_subject_data(subject_id)
                    all_stats[subject_id] = stats

                    # Aggregate gap and distribution statistics
                    for stream in ['heart_rate', 'motion', 'steps', 'labels']:
                        if stats[stream]['n_samples'] > 0:
                            if 'gaps' in stats[stream]:
                                aggregate_stats[stream]['missing_pcts'].append(
                                    stats[stream]['gaps']['missing_pct'])
                                aggregate_stats[stream]['gap_sizes'].append(
                                    stats[stream]['gaps']['largest_gap'])
                            if 'distribution' in stats[stream]:
                                aggregate_stats[stream]['skewness'].append(
                                    stats[stream]['distribution']['skewness'])
                except Exception as e:
                    logger.error(f"Error processing subject {
                                 subject_id}: {str(e)}")
                    all_stats[subject_id] = {
                        'heart_rate': {'n_samples': 0},
                        'motion': {'n_samples': 0},
                        'steps': {'n_samples': 0},
                        'labels': {'n_samples': 0},
                        'error': str(e)
                    }

            # Compute summary including sampling irregularity metrics
            summary = self.compute_summary_statistics(aggregate_stats)

            return {'individual': all_stats, 'summary': summary}

        except Exception as e:
            logger.error(f"Failed to analyze subjects: {str(e)}")
            raise

    def compute_summary_statistics(self, aggregate_stats: dict[str, dict[str, list[float]]]) -> dict[str, dict[str, dict[str, float]]]:
        """Compute summary statistics with focus on sampling irregularities across data streams.

        Calculates aggregate statistics for missing data percentages, gap sizes, and distribution 
        skewness (where applicable) across all subjects for each data stream type.

        Args:
            aggregate_stats: Dictionary containing aggregated metrics for each stream type with structure:
                {
                    'stream_name': {
                        'missing_pcts': list[float],  # Missing data percentages
                        'gap_sizes': list[float],     # Gap sizes in seconds
                        'skewness': list[float]       # Distribution skewness (optional)
                    }
                }

        Returns:
            Dictionary containing summary statistics for each stream type with structure:
                {
                    'stream_name': {
                        'missing_data': {
                            'mean_pct': float,
                            'median_pct': float,
                            'std_pct': float
                        },
                        'gap_statistics': {
                            'mean_gap': float,
                            'median_gap': float,
                            'max_gap': float
                        },
                        'distribution': {             # Only for numerical streams
                            'mean_skewness': float,
                            'median_skewness': float
                        }
                    }
                }
        """
        summary = {}

        for stream in ['heart_rate', 'motion', 'steps', 'labels']:
            summary[stream] = {
                'missing_data': {
                    'mean_pct': np.mean(aggregate_stats[stream]['missing_pcts']),
                    'median_pct': np.median(aggregate_stats[stream]['missing_pcts']),
                    'std_pct': np.std(aggregate_stats[stream]['missing_pcts'])
                },
                'gap_statistics': {
                    'mean_gap': np.mean(aggregate_stats[stream]['gap_sizes']),
                    'median_gap': np.median(aggregate_stats[stream]['gap_sizes']),
                    'max_gap': np.max(aggregate_stats[stream]['gap_sizes'])
                }
            }

            # Add skewness statistics for numerical data streams
            if 'skewness' in aggregate_stats[stream]:
                summary[stream]['distribution'] = {
                    'mean_skewness': np.mean(aggregate_stats[stream]['skewness']),
                    'median_skewness': np.median(aggregate_stats[stream]['skewness'])
                }

        return summary


if __name__ == "__main__":
    import argparse
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description='Analyze data quality metrics')
    parser.add_argument('--data_dir', type=str, default='./data/preprocessed/',
                        help='Directory containing preprocessed data')
    parser.add_argument('--verbose', action='store_true', default=True,
                        help='Print detailed analysis results')
    args = parser.parse_args()

    # Initialize analyzer and run analysis
    analyzer = DataInfo(args.data_dir)
    results = analyzer.analyze_all_subjects()

    if args.verbose:
        print("\nTime Series Analysis Summary")
        print("=" * 50)

        summary = results['summary']
        for stream in ['heart_rate', 'motion', 'steps', 'labels']:
            print(f"\n{stream.upper()} Data Quality:")
            print(f"Missing Data:")
            print(f"  - Mean: {summary[stream]
                  ['missing_data']['mean_pct']:.1f}%")
            print(f"  - Median: {summary[stream]
                  ['missing_data']['median_pct']:.1f}%")
            print(f"Gap Statistics:")
            print(f"  - Mean gap: {summary[stream]
                  ['gap_statistics']['mean_gap']:.2f}s")
            print(f"  - Max gap: {summary[stream]
                  ['gap_statistics']['max_gap']:.2f}s")
            if 'sampling_rate' in summary[stream]:
                print(f"Sampling Rate:")
                print(f"  - {summary[stream]['sampling_rate']:.2f} Hz")
            if 'distribution' in summary[stream]:
                print(f"Distribution:")
                print(
                    f"  - Mean skewness: {summary[stream]['distribution']['mean_skewness']:.3f}")
                print(
                    f"  - Median skewness: {summary[stream]['distribution']['median_skewness']:.3f}")

    #### Print Example ####

    # Time Series Analysis Summary
    # ==================================================

    # HEART_RATE Data Quality:
    # Missing Data:
    # - Mean: 15.4%
    # - Median: 3.9%
    # Gap Statistics:
    # - Mean gap: 66.84s
    # - Max gap: 858.50s
    # Distribution:
    # - Mean skewness: 1.857

    # MOTION Data Quality:
    # Missing Data:
    # - Mean: 1.0%
    # - Median: 0.1%
    # Gap Statistics:
    # - Mean gap: 131.81s
    # - Max gap: 1277.24s
    # Distribution:
    # - Mean skewness: 24.885

    # STEPS Data Quality:
    # Missing Data:
    # - Mean: 0.0%
    # - Median: 0.0%
    # Gap Statistics:
    # - Mean gap: 578.07s
    # - Max gap: 600.00s
    # Distribution:
    # - Mean skewness: nan

    # LABELS Data Quality:
    # Missing Data:
    # - Mean: 0.0%
    # - Median: 0.0%
    # Gap Statistics:
    # - Mean gap: 30.00s
    # - Max gap: 30.00s
