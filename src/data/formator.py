"""
This module handles the reformatting of raw apple watch data into a standardized HDF5 format.

The main purpose is to convert raw text files containing heart rate, motion, steps and sleep stage label data 
into a more efficient HDF5 file format. The HDF5 format provides:

- Faster data access and loading compared to text files 
- Hierarchical organization of different data streams
- Efficient storage of large numerical arrays
- Built-in compression

The module contains a DataFormator class that:
    1. Takes raw txt files from different sensors as input
    2. Converts the data to HDF5 format
    3. Saves it in a structured format with timestamps and values for each data stream
    4. Organizes data by subject ID

The formatted HDF5 files are then used by the DataReader class for efficient data loading.
"""

import h5py
import logging
import pandas as pd
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class DataFormator:
    """Converts raw txt files to HDF5 format for more efficient storage and access"""

    def __init__(self, data_dir: str, output_dir: str):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Initialized DataPreprocessor with data_dir={
                    data_dir} and output_dir={output_dir}")

    def convert_subject_data(self, subject_id: str):
        """Convert all data files for a single subject to HDF5 format"""
        logger.info(f"Starting data conversion for subject {subject_id}")

        output_file = self.output_dir / f"{subject_id}.h5"
        logger.info(f"Writing data to {output_file}")

        with h5py.File(output_file, 'w') as hf:
            # Convert heart rate data
            try:
                hr_path = self.data_dir / "heart_rate" / \
                    f"{subject_id}_heartrate.txt"  # Fixed filename
                logger.debug(f"Reading heart rate data from {hr_path}")
                hr_data = pd.read_csv(hr_path, header=None, names=[
                                      'timestamp', 'heart_rate'])
                hf.create_dataset('heart_rate/timestamps',
                                  data=hr_data['timestamp'].values)
                hf.create_dataset('heart_rate/values',
                                  data=hr_data['heart_rate'].values)
                logger.info(f"Successfully converted heart rate data: {
                            hr_data.shape[0]} samples")
            except FileNotFoundError:
                logger.error(f"Heart rate data file not found at {hr_path}")
            except Exception as e:
                logger.error(f"Error processing heart rate data: {str(e)}")

            # Convert motion data
            try:
                motion_path = self.data_dir / "motion" / \
                    f"{subject_id}_acceleration.txt"
                logger.debug(f"Reading motion data from {motion_path}")
                motion_data = pd.read_csv(motion_path, header=None,
                                          names=['timestamp', 'acc_x',
                                                 'acc_y', 'acc_z'],
                                          delimiter=' ')
                hf.create_dataset('motion/timestamps',
                                  data=motion_data['timestamp'].values)
                hf.create_dataset(
                    'motion/values', data=motion_data[['acc_x', 'acc_y', 'acc_z']].values)
                logger.info(f"Successfully converted motion data: {
                            motion_data.shape[0]} samples")
            except FileNotFoundError:
                logger.error(f"Motion data file not found at {motion_path}")
            except Exception as e:
                logger.error(f"Error processing motion data: {str(e)}")

            # Convert steps data
            try:
                steps_path = self.data_dir / \
                    "steps" / f"{subject_id}_steps.txt"
                logger.debug(f"Reading steps data from {steps_path}")
                steps_data = pd.read_csv(steps_path, header=None, names=[
                                         'timestamp', 'steps'])
                hf.create_dataset('steps/timestamps',
                                  data=steps_data['timestamp'].values)
                hf.create_dataset(
                    'steps/values', data=steps_data['steps'].values)
                logger.info(f"Successfully converted steps data: {
                            steps_data.shape[0]} samples")
            except FileNotFoundError:
                logger.error(f"Steps data file not found at {steps_path}")
            except Exception as e:
                logger.error(f"Error processing steps data: {str(e)}")

            # Convert labels data
            try:
                labels_path = self.data_dir / "labels" / \
                    f"{subject_id}_labeled_sleep.txt"
                logger.debug(f"Reading labels data from {labels_path}")
                labels_data = pd.read_csv(labels_path, header=None,
                                          names=['timestamp', 'sleep_stage'],
                                          delimiter=' ')
                hf.create_dataset('labels/timestamps',
                                  data=labels_data['timestamp'].values)
                hf.create_dataset(
                    'labels/values', data=labels_data['sleep_stage'].values)
                logger.info(f"Successfully converted labels data: {
                            labels_data.shape[0]} samples")
            except FileNotFoundError:
                logger.error(f"Labels data file not found at {labels_path}")
            except Exception as e:
                logger.error(f"Error processing labels data: {str(e)}")

        logger.info(f"Completed data conversion for subject {subject_id}")

    def convert_all_subjects(self):
        """Convert data for all subjects found in the data directory"""
        logger.info("Starting conversion for all subjects")

        # Get unique subject IDs from heart rate directory
        # Fixed filename pattern
        hr_files = list(self.data_dir.glob("heart_rate/*_heartrate.txt"))
        subject_ids = [f.stem.split('_')[0] for f in hr_files]

        logger.info(f"Found {len(subject_ids)} subjects to process")

        for i, subject_id in enumerate(subject_ids, 1):
            logger.info(f"Processing subject {
                        i}/{len(subject_ids)}: {subject_id}")
            self.convert_subject_data(subject_id)

        logger.info("Completed conversion for all subjects")


if __name__ == "__main__":
    preprocessor = DataFormator("./data/original", "./data/formated")
    preprocessor.convert_all_subjects()

    print("\nExample processed data structure:")
    print("-" * 50)
    with h5py.File('./processed_data/46343.h5', 'r') as f:
        print("\nDataset structure:")

        def print_structure(name, obj):
            if isinstance(obj, h5py.Dataset):
                print(f"{name}: shape={obj.shape}, dtype={obj.dtype}")
        f.visititems(print_structure)

        print("\nSample data:")
        print("\nHeart rate:")
        print("Timestamps:", f['heart_rate/timestamps'][:5])
        print("Values:", f['heart_rate/values'][:5])

        print("\nMotion:")
        print("Timestamps:", f['motion/timestamps'][:5])
        print("Values:", f['motion/values'][:5])

        print("\nSteps:")
        print("Timestamps:", f['steps/timestamps'][:5])
        print("Values:", f['steps/values'][:5])

        print("\nLabels:")
        print("Timestamps:", f['labels/timestamps'][:5])
        print("Values:", f['labels/values'][:5])

    #### Print example ####

    # Dataset structure:
    #     heart_rate/timestamps: shape=(4878,), dtype=float64
    #     heart_rate/values: shape=(4878,), dtype=int64
    #     labels/timestamps: shape=(567,), dtype=int64
    #     labels/values: shape=(567,), dtype=int64
    #     motion/timestamps: shape=(982000,), dtype=float64
    #     motion/values: shape=(982000, 3), dtype=float64
    #     steps/timestamps: shape=(1403,), dtype=int64
    #     steps/values: shape=(1403,), dtype=int64

    # Sample data:
    #     Heart rate:
    #         Timestamps: [-556410.36066 -556408.36062 -556403.36062 -556399.36062 -556389.36062]
    #         Values: [57 56 56 57 59]
    #     Motion:
    #         Timestamps: [-124489.16105  -124489.116395 -124489.115548 -124489.114691 -124489.0977]
    #         Values: [[ 0.0174866 -0.5867004 -0.8057709]
    #                  [ 0.0189819 -0.5896759 -0.8091583]
    #                  [ 0.0209656 -0.5808868 -0.8150482]
    #                  [ 0.0194855 -0.5808716 -0.8135834]
    #                  [ 0.0169983 -0.587204  -0.8062592]]
    #     Steps:
    #         Timestamps: [-604539 -603939 -603339 -602739 -602139]
    #         Values: [0 0 0 0 0]
    #     Labels:
    #         Timestamps: [  0  30  60  90 120]
    #         Values: [-1 -1 -1 -1 -1]
