"""
This module provides visualization functionality for the preprocessed physiological data.

The main purpose is to create comprehensive multi-panel visualizations showing the 
different data streams (heart rate, motion, steps, sleep stages) aligned in time. 
This module provides:

- Functions to plot multiple time series data streams
- Consistent styling and formatting across plots
- Clear visualization of circadian patterns via day boundaries
- Metadata annotations about recording duration and quality

The visualizations help with:
    1. Data quality assessment and validation
    2. Understanding temporal patterns and relationships
    3. Identifying potential issues or anomalies
    4. Communicating findings. 

The plot_subject_data() function is the main entry point used to generate plots
for individual subjects or batches of subjects.
"""

import numpy as np
from tqdm import tqdm
from reader import DataReader
import matplotlib.pyplot as plt
import logging

# Configure logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler = logging.StreamHandler()
handler.setFormatter(formatter)
logger.addHandler(handler)


def plot_subject_data(subject_id: str, data_reader: DataReader, save_path: str, verbose: bool = True):
    """
    Create a comprehensive visualization of physiological and behavioral data streams for a subject.

    This function generates a multi-panel figure showing:
    - Heart rate measurements over time (beats per minute)
    - Tri-axial acceleration data showing physical movement
    - Step counts per time window showing activity level  
    - Sleep stage classifications (-1: invalid, 0: wake, 1: N1, 2: N2, 3: N3, 5: REM)

    Vertical dashed lines indicate day boundaries (midnight) to show circadian patterns.
    Data points are semi-transparent to better show density.

    Args:
        subject_id: ID of the subject to plot
        data_reader: DataReader instance to load the data
        save_path: Path where to save the plot
        verbose: If True, print warnings about empty data streams
    """
    # Read all data streams
    hr_data = data_reader.read_heart_rate(subject_id)
    motion_data = data_reader.read_motion(subject_id)
    steps_data = data_reader.read_steps(subject_id)
    labels_data = data_reader.read_labels(subject_id)

    # Check if any data stream is empty
    if verbose:
        for name, data in [('heart rate', hr_data), ('motion', motion_data),
                           ('steps', steps_data), ('labels', labels_data)]:
            if len(data.timestamps) == 0:
                logger.warning(f"Empty {name} data for subject {subject_id}")

    # Skip plotting if any stream is empty
    if any(len(data.timestamps) == 0 for data in [hr_data, motion_data, steps_data, labels_data]):
        if verbose:
            logger.warning(f"Skipping plot for subject {
                           subject_id} due to missing data")
        return

    # Create figure with subplots
    fig, axs = plt.subplots(4, 1, figsize=(15, 12), sharex=True)
    fig.suptitle(f'Physiological Data for Subject {subject_id}',
                 fontsize=14, fontweight='bold', y=0.95)

    # Calculate time range and day boundaries
    timestamps = [data.timestamps for data in [
        hr_data, motion_data, steps_data, labels_data]]
    min_time = min(t[0] for t in timestamps)
    max_time = max(t[-1] for t in timestamps)
    duration_hrs = (max_time - min_time) / 3600

    # Calculate day lines only if recording > 24h
    day_lines = []
    if duration_hrs >= 24:
        day_seconds = 24 * 60 * 60
        day_lines = np.arange(
            int(min_time) - (int(min_time) % day_seconds),
            int(max_time),
            day_seconds
        )

    # Plot heart rate with reduced points
    hr_min = min(hr_data.values)
    hr_max = max(hr_data.values)
    hr_buffer = (hr_max - hr_min) * 0.1  # Add 10% padding

    axs[0].plot(hr_data.timestamps[::2], hr_data.values[::2],
                'b.', alpha=0.3, markersize=2, label='Heart Rate')
    axs[0].set_ylabel('Heart Rate\n(bpm)', fontsize=10)
    axs[0].grid(True, alpha=0.3)
    axs[0].set_ylim(hr_min - hr_buffer, hr_max + hr_buffer)

    # Plot motion with reduced points
    step = max(1, len(motion_data.timestamps) // 10000)  # Limit to ~10k points
    for i, color in enumerate(['r', 'g', 'b']):
        axs[1].plot(motion_data.timestamps[::step], motion_data.values[::step, i],
                    f'{color}.', alpha=0.3, markersize=1, label=f'{["X", "Y", "Z"][i]}-axis')
    axs[1].set_ylabel('Acceleration\n(g)', fontsize=10)
    axs[1].legend(loc='upper right', fontsize=8)
    axs[1].grid(True, alpha=0.3)

    # Plot steps
    axs[2].plot(steps_data.timestamps, steps_data.values,
                'g.', alpha=0.5, label='Steps')
    axs[2].set_ylabel('Steps per\nTime Window', fontsize=10)
    axs[2].grid(True, alpha=0.3)

    # Plot sleep stages
    axs[3].plot(labels_data.timestamps, labels_data.values,
                'k.', alpha=0.5, markersize=2)
    axs[3].set_ylabel('Sleep Stage', fontsize=10)
    # Support both raw and processed label formats
    if -1 in labels_data.values:  # Raw data format
        axs[3].set_yticks([-1, 0, 1, 2, 3, 5])
        axs[3].set_yticklabels(['Invalid', 'Wake', 'N1', 'N2', 'N3', 'REM'])
    else:  # Processed data format
        axs[3].set_yticks([0, 1, 2, 3, 4])
        axs[3].set_yticklabels(['Wake', 'N1', 'N2', 'N3', 'REM'])
    axs[3].grid(True, alpha=0.3)

    # Add day boundary lines to all plots
    for ax in axs:
        for day_line in day_lines:
            ax.axvline(x=day_line, color='gray', linestyle='--', alpha=0.3)
        ax.tick_params(labelsize=8)

    # Set common x label
    axs[3].set_xlabel('Time (seconds from start)', fontsize=10)

    # Add metadata
    metadata = (f'Total duration: {duration_hrs:.1f} hours ({len(day_lines)} days)\n'
                f'Recording start: {min_time:.0f}s, end: {max_time:.0f}s')
    fig.text(0.02, 0.98, metadata, fontsize=8, va='top')

    # Save and close
    plt.tight_layout()
    plt.savefig(f"{save_path}/{subject_id}_data.png",
                dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    import json
    import random
    import argparse

    # Set up argument parser
    parser = argparse.ArgumentParser(
        description='Generate plots for subject data')
    parser.add_argument('--data_dir', type=str, default='./data/formated/',
                        help='Directory containing formatted data')
    parser.add_argument('--output_dir', type=str, default='./data_visualization/',
                        help='Directory to save plots')
    parser.add_argument('--num_subjects', type=int, default=-1,
                        help='Number of random subjects to plot')
    parser.add_argument('--subject_ids_file', type=str, default='./data/formated/subject_ids.json',
                        help='JSON file containing subject IDs')
    parser.add_argument('--verbose', action='store_true', default=True,
                        help='Print warnings about empty data streams')
    args = parser.parse_args()

    # Initialize data reader
    data_reader = DataReader(args.data_dir)

    # Read subject IDs from JSON file
    with open(args.subject_ids_file, 'r') as f:
        subject_ids = json.load(f)

    # Select subjects based on num_subjects argument
    if args.num_subjects == -1:
        selected_subjects = subject_ids
    else:
        selected_subjects = random.sample(subject_ids, args.num_subjects)

    # Generate plots for selected subjects with progress bar
    for subject_id in tqdm(selected_subjects, desc="Generating plots"):
        try:
            plot_subject_data(subject_id,
                              data_reader,
                              args.output_dir,
                              verbose=args.verbose)
        except FileNotFoundError as e:
            logger.error(f"Invalid subject at {subject_id}")
            continue
        except Exception as e:
            logger.error(f"Error processing subject {subject_id}: {str(e)}")
            continue
