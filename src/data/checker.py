"""
This module checks data files for completeness and validity.

The main purpose is to verify that all required data streams are present and contain valid data.
It checks for:
- Missing data files
- Empty or corrupted data streams
- Invalid values or timestamps
- Inconsistent data lengths

The module contains a DataChecker class that:
    1. Takes HDF5 data files as input
    2. Checks each data stream for completeness and validity 
    3. Reports any issues found
    4. Can process individual subjects or entire dataset
"""

import json
import logging
import numpy as np
from pathlib import Path
from reader import DataReader, TimeSeriesData

logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class DataChecker:
    """Checks data files for completeness and validity"""

    def __init__(self, data_dir: str):
        self.data_reader = DataReader(data_dir)
        self.data_dir = Path(data_dir)
        logger.info(f"Initialized DataChecker with data_dir={data_dir}")

    def _check_empty_streams(self,
                             hr_data: TimeSeriesData,
                             motion_data: TimeSeriesData,
                             steps_data: TimeSeriesData,
                             labels_data: TimeSeriesData,
                             subject_id: str) -> tuple[bool, str]:
        """Check for empty data streams"""
        if len(hr_data.timestamps) == 0:
            logger.warning(f"Subject {subject_id}: Empty heart rate data")
            return False, "Empty heart rate data"
        if len(motion_data.timestamps) == 0:
            logger.warning(f"Subject {subject_id}: Empty motion data")
            return False, "Empty motion data"
        if len(steps_data.timestamps) == 0:
            logger.warning(f"Subject {subject_id}: Empty steps data")
            return False, "Empty steps data"
        if len(labels_data.timestamps) == 0:
            logger.warning(f"Subject {subject_id}: Empty labels data")
            return False, "Empty labels data"
        return True, ""

    def _check_aligned_endpoints(self,
                                 hr_data: TimeSeriesData,
                                 motion_data: TimeSeriesData,
                                 steps_data: TimeSeriesData,
                                 labels_data: TimeSeriesData,
                                 subject_id: str) -> tuple[bool, str]:
        """Check if all data streams have aligned start and end times, accounting for sampling frequency"""
        # Calculate typical intervals between samples for each stream
        # Static intervals based on preprocessed data analysis
        hr_interval = 2.01 * 5.0  # 5 seconds between heart rate samples
        motion_interval = 2.01 * 0.2  # 0.2 seconds between motion samples
        steps_interval = 2.01 * 500.0  # 500 seconds between step count windows
        labels_interval = 2.01 * 30.0  # 30 seconds between sleep stage labels

        # Get start and end times for each data stream
        start_times = {
            'heart_rate': hr_data.timestamps[0],
            'motion': motion_data.timestamps[0],
            'steps': steps_data.timestamps[0],
            'labels': labels_data.timestamps[0]
        }

        end_times = {
            'heart_rate': hr_data.timestamps[-1],
            'motion': motion_data.timestamps[-1],
            'steps': steps_data.timestamps[-1],
            'labels': labels_data.timestamps[-1]
        }

        # Log start and end times for debugging
        for name in start_times:
            logger.info(f"Subject {subject_id}: {name} data starts at {
                        start_times[name]:.1f}s and ends at {end_times[name]:.1f}s")

        # Find max difference between end times
        max_end_diff = max(end_times.values()) - min(end_times.values())
        if max_end_diff > max(hr_interval, motion_interval, steps_interval, labels_interval):
            logger.warning(f"Subject {
                           subject_id}: Data streams have misaligned end times (max difference: {max_end_diff:.1f}s)")
            return False, "Misaligned data stream end times"

        # Check start times are close to 0
        max_start = max(start_times.values())
        if max_start > min(hr_interval, motion_interval, steps_interval, labels_interval):
            logger.warning(f"Subject {
                           subject_id}: Data streams have delayed start times (max start: {max_start:.1f}s)")
            return False, "Delayed data stream start times"

        return True, ""

    def _check_value_ranges(self,
                            hr_data: TimeSeriesData,
                            motion_data: TimeSeriesData,
                            steps_data: TimeSeriesData,
                            labels_data: TimeSeriesData,
                            subject_id: str,
                            hr_range: tuple[int, int],
                            motion_range: tuple[float, float]) -> tuple[bool, str]:
        """Check for invalid values in data streams"""
        if np.any((hr_data.values < hr_range[0]) | (hr_data.values > hr_range[1])):
            logger.warning(f"Subject {subject_id}: Heart rate values outside normal range ({
                           hr_range[0]}-{hr_range[1]} bpm)")
            return False, "Heart rate values outside normal range"

        if np.any((motion_data.values < motion_range[0]) | (motion_data.values > motion_range[1])):
            logger.warning(f"Subject {subject_id}: Motion values outside normal range ({
                           motion_range[0]}g to {motion_range[1]}g)")
            return False, "Motion values outside normal range"

        if np.any(steps_data.values < 0):
            logger.warning(f"Subject {subject_id}: Negative step counts found")
            return False, "Negative step counts found"

        if np.any((labels_data.values < -1) | (labels_data.values > 5)):
            logger.warning(
                f"Subject {subject_id}: Invalid sleep stage labels found")
            return False, "Invalid sleep stage labels"

        return True, ""

    def _check_timestamps(self,
                          hr_data: TimeSeriesData,
                          motion_data: TimeSeriesData,
                          steps_data: TimeSeriesData,
                          labels_data: TimeSeriesData,
                          subject_id: str) -> tuple[bool, str]:
        """Check timestamp ordering"""
        for data, name in [(hr_data, "heart rate"), (motion_data, "motion"),
                           (steps_data, "steps"), (labels_data, "labels")]:
            if not np.all(np.diff(data.timestamps) >= 0):
                logger.warning(
                    f"Subject {subject_id}: Non-monotonic timestamps in {name} data")
                return False, f"Non-monotonic timestamps in {name} data"
        return True, ""

    def _check_data_gaps(self,
                         hr_data: TimeSeriesData,
                         motion_data: TimeSeriesData,
                         steps_data: TimeSeriesData,
                         labels_data: TimeSeriesData,
                         subject_id: str,
                         max_allowed_gap: float,
                         max_missing_ratio: float) -> tuple[bool, str]:
        """Check for large gaps and missing data ratio"""
        for data, name in [(hr_data, "heart rate"), (motion_data, "motion"),
                           (steps_data, "steps"), (labels_data, "labels")]:
            if len(data.timestamps) <= 1:
                continue

            gaps = np.diff(data.timestamps)
            large_gaps = gaps > max_allowed_gap

            if np.any(large_gaps):
                logger.warning(f"Subject {subject_id}: Large gaps (>{
                               max_allowed_gap}s) found in {name} data")
                return False, f"Large gaps in {name} data"

            total_duration = data.timestamps[-1] - data.timestamps[0]
            missing_duration = np.sum(gaps[large_gaps])
            missing_ratio = missing_duration / total_duration

            if missing_ratio > max_missing_ratio:
                logger.warning(f"Subject {subject_id}: Too much missing data in {
                               name} ({missing_ratio:.1%})")
                return False, f"Excessive missing data in {name}"
        return True, ""

    def _check_sampling_rates(self,
                              hr_data: TimeSeriesData,
                              motion_data: TimeSeriesData,
                              steps_data: TimeSeriesData,
                              labels_data: TimeSeriesData,
                              subject_id: str,
                              expected_rates: dict[str, float],
                              rate_tolerances: dict[str, float]) -> tuple[bool, str]:
        """Check data sampling rates"""
        for data, name in [(hr_data, "heart rate"), (motion_data, "motion"),
                           (steps_data, "steps"), (labels_data, "labels")]:
            if len(data.timestamps) > 1:
                avg_rate = 1.0 / np.mean(np.diff(data.timestamps))
                expected_rate = expected_rates[name]
                tolerance = rate_tolerances[name]
                if abs(avg_rate - expected_rate) > tolerance * expected_rate:
                    expected_interval = 1/expected_rate
                    actual_interval = 1/avg_rate
                    logger.warning(f"Subject {subject_id}: Irregular sampling rate in {name} data "
                                   f"(expected: {expected_rate:.3f} Hz ({
                        expected_interval:.1f}s), "
                        f"actual: {avg_rate:.3f} Hz ({actual_interval:.1f}s))")
                    return False, f"Irregular sampling rate in {name} data"
        return True, ""

    def _check_coverage(self,
                        hr_data: TimeSeriesData,
                        motion_data: TimeSeriesData,
                        labels_data: TimeSeriesData,
                        subject_id: str,
                        min_duration: float) -> tuple[bool, str]:
        """Check data alignment and coverage"""
        min_times = [data.timestamps[0]
                     for data in [hr_data, motion_data, labels_data]]
        max_times = [data.timestamps[-1]
                     for data in [hr_data, motion_data, labels_data]]

        time_range_diff = max(max_times) - min(min_times)
        if time_range_diff < min_duration:
            logger.warning(f"Subject {subject_id}: Recording duration too short (<{
                           min_duration/3600:.1f}h)")
            return False, "Recording duration too short"

        latest_start = max(min_times)
        earliest_end = min(max_times)
        expected_coverage = 0.7 * \
            (max(max_times) - min(min_times))  # Expect 70% coverage

        if earliest_end - latest_start < expected_coverage:
            logger.warning(
                f"Subject {subject_id}: Insufficient overlapping data between streams")
            return False, "Insufficient overlapping data"
        return True, ""

    def _check_aligned_endpoints(self,
                                 hr_data: TimeSeriesData,
                                 motion_data: TimeSeriesData,
                                 # steps_data: TimeSeriesData,
                                 labels_data: TimeSeriesData,
                                 subject_id: str,
                                 ) -> tuple[bool, str]:
        """Check that all data streams start and end at approximately the same time"""
        # Tolerance for start/end time misalignment based on sampling rates
        # Use 2x the longest sampling interval as tolerance
        tolerances = {
            "heart_rate": 2000.0,
            "motion": 2000.0,
            "labels": 60.0
        }
        # Use max tolerance across streams
        tolerance = max(tolerances.values())

        start_times = [data.timestamps[0]
                       for data in [hr_data, motion_data, labels_data]]
        end_times = [data.timestamps[-1]
                     for data in [hr_data, motion_data, labels_data]]

        # Log start and end times for each stream
        # streams = ["heart rate", "motion", "labels"]
        # for stream, start, end in zip(streams, start_times, end_times):
        #     logger.info(f"Subject {subject_id}: {stream} data starts at {
        #                 start:.1f}s and ends at {end:.1f}s")

        start_diff = max(start_times) - min(start_times)
        end_diff = max(end_times) - min(end_times)

        if start_diff > tolerance:
            logger.warning(f"Subject {subject_id}: Data streams have misaligned start times "
                           f"(max difference: {start_diff:.1f}s)")
            return False, "Misaligned start times"

        if end_diff > tolerance:
            logger.warning(f"Subject {subject_id}: Data streams have misaligned end times "
                           f"(max difference: {end_diff:.1f}s)")
            return False, "Misaligned end times"

        return True, ""

    def check_subject_data(self, subject_id: str,
                           hr_range: tuple[int, int] = (40, 150),
                           motion_range: tuple[float, float] = (-10, 10),
                           max_allowed_gap: int = 3600,  # 1 hour max gap
                           heart_rate_rate: float = 0.2,  # 5Hz, 0.2s
                           motion_rate: float = 5.0,  # 5Hz, 0.2s
                           steps_rate: float = 0.002,  # 0.002Hz, 500s, 8.3m
                           labels_rate: float = 0.033,  # 0.033Hz, 30s
                           min_duration: int = 3 * 3600,  # 3 hours
                           max_missing_ratio: float = 0.3) -> tuple[bool, str]:
        """Check all data streams for a single subject"""
        logger.info(f"Checking data for subject {subject_id}")

        try:
            # Read all data streams
            hr_data = self.data_reader.read_heart_rate(subject_id)
            motion_data = self.data_reader.read_motion(subject_id)
            steps_data = self.data_reader.read_steps(subject_id)
            labels_data = self.data_reader.read_labels(subject_id)

            # Run all checks
            checks = [
                self._check_empty_streams(
                    hr_data, motion_data, steps_data, labels_data, subject_id),
                self._check_value_ranges(
                    hr_data, motion_data, steps_data, labels_data, subject_id, hr_range, motion_range),
                self._check_timestamps(
                    hr_data, motion_data, steps_data, labels_data, subject_id),
                self._check_data_gaps(hr_data, motion_data, steps_data,
                                      labels_data, subject_id, max_allowed_gap, max_missing_ratio),
                self._check_sampling_rates(hr_data, motion_data, steps_data, labels_data, subject_id,
                                           {"heart rate": heart_rate_rate, "motion": motion_rate,
                                            "steps": steps_rate, "labels": labels_rate},
                                           {"heart rate": 0.05, "motion": 0.05,
                                               "steps": 0.20, "labels": 0.10}),
                self._check_coverage(hr_data, motion_data,
                                     labels_data, subject_id, min_duration),
                self._check_aligned_endpoints(
                    hr_data, motion_data, labels_data, subject_id)
            ]

            # Return first failure or success
            for valid, reason in checks:
                if not valid:
                    return False, reason

            logger.info(f"All checks passed for subject {subject_id}")
            return True, ""

        except FileNotFoundError:
            logger.error(f"Data file not found for subject {subject_id}")
            return False, "Data file not found"
        except Exception as e:
            logger.error(f"Error checking data for subject {
                         subject_id}: {str(e)}")
            return False, f"Error checking data: {str(e)}"

    def check_all_subjects(self, verbose: bool = True, modify_invalid: bool = False) -> dict:
        """
        Check data for all subjects in the input directory

        Args:
            verbose: If True, print summary of failures. If False, just return results.
            modify_invalid: If True, rename invalid subject files and remove from subject_ids.json

        Returns:
            Dictionary mapping subject IDs to validation results (bool)
        """
        # Read subject IDs from JSON file
        subject_ids_file = self.data_dir / "subject_ids.json"
        if not subject_ids_file.exists():
            raise FileNotFoundError(
                f"Subject IDs file not found: {subject_ids_file}")

        with open(subject_ids_file, 'r') as f:
            subject_ids = json.load(f)

        results = {}
        failed_subjects = {}
        failure_counts = {}
        valid_subject_ids = []

        for subject_id in subject_ids:
            valid, failure_reason = self.check_subject_data(subject_id)
            results[subject_id] = valid

            if valid:
                valid_subject_ids.append(subject_id)
            else:
                failed_subjects[subject_id] = failure_reason
                failure_counts[failure_reason] = failure_counts.get(
                    failure_reason, 0) + 1

                if modify_invalid:
                    # Rename invalid subject file
                    orig_file = self.data_dir / f"{subject_id}.h5"
                    invalid_file = self.data_dir / f"INVALID_{subject_id}.h5"
                    if orig_file.exists():
                        orig_file.rename(invalid_file)
                        logger.info(
                            f"Renamed invalid file for subject {subject_id}")

        if modify_invalid:
            # Update subject_ids.json with only valid subjects
            with open(subject_ids_file, 'w') as f:
                json.dump(valid_subject_ids, f)
            logger.info("Updated subject_ids.json with valid subjects only")

        if verbose:
            # Print summary of failures
            print("\n" + "="*50)
            print("DATA CHECK FAILURES")
            print("="*50)

            total_subjects = len(subject_ids)
            failed_count = len(failed_subjects)
            fail_rate = (failed_count / total_subjects) * 100

            print(f"\nOverall Statistics:")
            print(f"Total Subjects: {total_subjects}")
            print(f"Failed Subjects: {failed_count}")
            print(f"Failure Rate: {fail_rate:.1f}%")

            if failed_subjects:
                print("\nFailure Reasons Breakdown:")
                for reason, count in failure_counts.items():
                    percentage = (count / failed_count) * 100
                    print(f"- {reason}: {count} subjects ({percentage:.1f}%)")

                print("\nDetailed Failures:")
                for subject_id, failure in failed_subjects.items():
                    print(f"\nSubject {subject_id} failed:")
                    print(f"- {failure}")
            else:
                print("\nAll subjects passed data checks")

            print("\n" + "="*50)

        return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description='Check data files for completeness and validity')
    parser.add_argument('--data_dir', type=str, default='./data/preprocessed/',
                        help='Directory containing data files to check')
    parser.add_argument('--modify_invalid', action='store_true', default=False,
                        help='If set, rename invalid files and update subject_ids.json')

    args = parser.parse_args()

    checker = DataChecker(args.data_dir)
    checker.check_all_subjects(modify_invalid=args.modify_invalid)

    # Print Exmaple
    # ==================================================
    # DATA CHECK FAILURES
    # ==================================================

    # Overall Statistics:
    # Total Subjects: 31
    # Failed Subjects: 18
    # Failure Rate: 58.1%

    # Failure Reasons Breakdown:
    # - Error checking data: index 0 is out of bounds for axis 0 with size 0: 6 subjects (33.3%)
    # - Non-monotonic timestamps in motion data: 9 subjects (50.0%)
    # - Irregular sampling rate in motion data: 3 subjects (16.7%)

    # Detailed Failures:

    # Subject 1066528 failed:
    # - Error checking data: index 0 is out of bounds for axis 0 with size 0

    # Subject 1360686 failed:
    # - Non-monotonic timestamps in motion data

    # Subject 1449548 failed:
    # - Non-monotonic timestamps in motion data

    # Subject 1818471 failed:
    # - Non-monotonic timestamps in motion data

    # Subject 2598705 failed:
    # - Non-monotonic timestamps in motion data

    # Subject 2638030 failed:
    # - Non-monotonic timestamps in motion data

    # Subject 3509524 failed:
    # - Irregular sampling rate in motion data

    # Subject 4018081 failed:
    # - Non-monotonic timestamps in motion data

    # Subject 4314139 failed:
    # - Non-monotonic timestamps in motion data

    # Subject 4426783 failed:
    # - Error checking data: index 0 is out of bounds for axis 0 with size 0

    # Subject 5383425 failed:
    # - Irregular sampling rate in motion data

    # Subject 7749105 failed:
    # - Irregular sampling rate in motion data

    # Subject 781756 failed:
    # - Error checking data: index 0 is out of bounds for axis 0 with size 0

    # Subject 8000685 failed:
    # - Non-monotonic timestamps in motion data

    # Subject 8258170 failed:
    # - Error checking data: index 0 is out of bounds for axis 0 with size 0

    # Subject 8530312 failed:
    # - Non-monotonic timestamps in motion data

    # Subject 9618981 failed:
    # - Error checking data: index 0 is out of bounds for axis 0 with size 0

    # Subject 9961348 failed:
    # - Error checking data: index 0 is out of bounds for axis 0 with size 0

    # ==================================================

    #### Another Example ####
    # 2025-01-19 15:23:15 - __main__ - WARNING - Subject 7749105: Insufficient overlapping data between streams

    # ==================================================
    # DATA CHECK FAILURES
    # ==================================================

    # Overall Statistics:
    # Total Subjects: 31
    # Failed Subjects: 1
    # Failure Rate: 3.2%

    # Failure Reasons Breakdown:
    # - Insufficient overlapping data: 1 subjects (100.0%)

    # Detailed Failures:

    # Subject 7749105 failed:
    # - Insufficient overlapping data

    # ==================================================
