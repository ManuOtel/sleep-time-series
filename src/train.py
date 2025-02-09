"""
This module provides functionality for training and evaluating sleep stage classification models.

The main purpose is to handle the complete training pipeline including:
- Data loading and batching
- Model training and validation
- Metric tracking and logging
- Experiment management
- Parallel training runs

Key components:
    1. train_model(): Core training loop with metric tracking
    2. run_single_experiment(): Executes one training run with given parameters
    3. run_experiment(): Manages parallel execution of multiple training runs
    4. Logging utilities for thread-safe progress tracking
    5. TensorBoard integration for visualization

The training pipeline supports:
    - Multi-fold cross validation
    - Hyperparameter grid search
    - GPU acceleration when available
    - Parallel training runs
    - Progress logging and visualization
    - Model checkpointing and history saving
"""

import gc
import sys
import yaml
import json
import torch
import logging
import itertools
import threading
import traceback
import torch.nn as nn
import multiprocessing
import concurrent.futures
from pathlib import Path
from data.dataset import SleepDataset
from torch.utils.data import DataLoader
from models.lstm import SleepClassifierLSTM
from models.transformer import SleepClassifierTransformer
from torch.utils.tensorboard import SummaryWriter
# import torch._dynamo
# torch._dynamo.config.suppress_errors = True

# Set up logging with timestamp and thread safety
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    force=True
)

# Get logger instance
logger = logging.getLogger(__name__)

# Create thread lock for safe logging
log_lock = threading.Lock()


def load_config(config_path: str, experiment_name: str | None = None) -> dict:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to config file
        experiment_name: Name of specific parameter grid to use (e.g., 'quick_test')
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    if experiment_name and experiment_name in config:
        config['param_grid'] = config[experiment_name]

    return config


def safe_log(message: str) -> None:
    """
    Thread-safe logging function.

    Args:
        message: The message to log
    """
    with log_lock:
        logger.info(message)


def train_model(model: nn.Module,
                train_loader: DataLoader,
                valid_loader: DataLoader,
                test_loader: DataLoader,
                num_epochs: int = 50,
                learning_rate: float = 0.001,
                device: str = "cuda" if torch.cuda.is_available() else "cpu",
                writer: SummaryWriter | None = None,
                run_name: str = "",
                verbose: bool = False
                ) -> dict[str, list[float]]:
    """
    Train a sleep classification model using provided data loaders.

    Performs training loop with validation and testing. Tracks metrics like loss and accuracy.
    Optionally logs to TensorBoard and prints verbose progress.

    Args:
        model: Neural network model to train
        train_loader: DataLoader containing training data batches
        valid_loader: DataLoader containing validation data batches  
        test_loader: DataLoader containing test data batches
        num_epochs: Number of complete passes through training data
        learning_rate: Step size for optimizer updates
        device: Device to run training on ('cuda' or 'cpu')
        writer: TensorBoard SummaryWriter for logging metrics
        run_name: Identifier string for this training run
        verbose: If True, print detailed progress messages

    Returns:
        Dictionary containing training history with keys:
            - train_loss: List of training losses per epoch
            - train_acc: List of training accuracies per epoch  
            - valid_loss: List of validation losses per epoch
            - valid_acc: List of validation accuracies per epoch
            - test_loss: List of test losses per epoch
            - test_acc: List of test accuracies per epoch

    Raises:
        RuntimeError: If training fails due to GPU memory or other runtime issues
        ValueError: If input data dimensions don't match model expectations
    """
    try:
        if verbose:
            safe_log(f"[{run_name}] Setting up loss and optimizer...")

        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        # Training history
        history = {
            'train_loss': [],
            'train_acc': [],
            'valid_loss': [],
            'valid_acc': [],
            'test_loss': [],
            'test_acc': []
        }

        # Training loop
        for epoch in range(num_epochs):
            safe_log(f"[{run_name}] Starting epoch {epoch+1}")
            if verbose:
                safe_log(f"[{run_name}] Setting model to train mode...")

            model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0

            # Training
            if verbose:
                safe_log(f"[{run_name}] Starting training batches...")

            for batch_idx, (batch_data, batch_labels) in enumerate(train_loader):
                try:
                    if verbose:
                        safe_log(f"[{run_name}] Moving batch {batch_idx+1} to device...")

                    # Move data to device
                    batch_data = {k: v.to(device)
                                  for k, v in batch_data.items()}
                    batch_labels = batch_labels.to(device)

                    if verbose:
                        safe_log(f"[{run_name}] Forward pass...")

                    # Forward pass
                    outputs = model(batch_data)
                    loss = criterion(outputs, batch_labels)

                    if verbose:
                        safe_log(
                            f"[{run_name}] Backward pass and optimization...")

                    # Backward pass and optimize
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    # Statistics
                    train_loss += loss.item()
                    _, predicted = outputs.max(1)
                    _, true_labels = batch_labels.max(1)
                    train_total += true_labels.size(0)
                    train_correct += predicted.eq(true_labels).sum().item()

                    if verbose:
                        safe_log(f'[{run_name}] Completed batch {batch_idx+1}')

                    print(f'[{run_name}] Batch {batch_idx+1} loss: {loss.item():.4f}')

                except Exception as e:
                    safe_log(f'[{run_name}] Error in training batch {batch_idx}: {str(e)}')
                    raise

            # Validation
            if verbose:
                safe_log(f"[{run_name}] Starting validation...")
                safe_log(f"[{run_name}] Setting model to eval mode...")

            model.eval()
            valid_loss = 0
            valid_correct = 0
            valid_total = 0
            test_loss = 0
            test_correct = 0
            test_total = 0

            with torch.no_grad():
                for batch_idx, (batch_data, batch_labels) in enumerate(valid_loader):
                    try:
                        if verbose:
                            safe_log(f"[{run_name}] Processing validation batch {batch_idx+1}...")

                        batch_data = {k: v.to(device)
                                      for k, v in batch_data.items()}
                        batch_labels = batch_labels.to(device)

                        outputs = model(batch_data)
                        loss = criterion(outputs, batch_labels)

                        valid_loss += loss.item()
                        _, predicted = outputs.max(1)
                        _, true_labels = batch_labels.max(1)
                        valid_total += true_labels.size(0)
                        valid_correct += predicted.eq(true_labels).sum().item()

                    except Exception as e:
                        safe_log(f'[{run_name}] Error in validation batch {batch_idx}: {str(e)}')
                        raise

                # Testing
                for batch_idx, (batch_data, batch_labels) in enumerate(test_loader):
                    try:
                        if verbose:
                            safe_log(f"[{run_name}] Processing test batch {batch_idx+1}...")

                        batch_data = {k: v.to(device)
                                      for k, v in batch_data.items()}
                        batch_labels = batch_labels.to(device)

                        outputs = model(batch_data)
                        loss = criterion(outputs, batch_labels)

                        test_loss += loss.item()
                        _, predicted = outputs.max(1)
                        _, true_labels = batch_labels.max(1)
                        test_total += true_labels.size(0)
                        test_correct += predicted.eq(true_labels).sum().item()

                    except Exception as e:
                        safe_log(f'[{run_name}] Error in test batch {batch_idx}: {str(e)}')
                        raise

            if verbose:
                safe_log(f"[{run_name}] Computing epoch metrics...")

            # Record metrics
            train_loss = train_loss / len(train_loader)
            train_acc = 100. * train_correct / train_total
            valid_loss = valid_loss / len(valid_loader)
            valid_acc = 100. * valid_correct / valid_total
            test_loss = test_loss / len(test_loader)
            test_acc = 100. * test_correct / test_total

            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['valid_loss'].append(valid_loss)
            history['valid_acc'].append(valid_acc)
            history['test_loss'].append(test_loss)
            history['test_acc'].append(test_acc)

            if verbose:
                safe_log(f"[{run_name}] Logging to TensorBoard...")

            # Log epoch metrics to TensorBoard
            if writer:
                writer.add_scalar(
                    f'{run_name}/Train/Epoch_Loss', train_loss, epoch)
                writer.add_scalar(
                    f'{run_name}/Train/Epoch_Accuracy', train_acc, epoch)
                writer.add_scalar(
                    f'{run_name}/Valid/Epoch_Loss', valid_loss, epoch)
                writer.add_scalar(
                    f'{run_name}/Valid/Epoch_Accuracy', valid_acc, epoch)
                writer.add_scalar(
                    f'{run_name}/Test/Epoch_Loss', test_loss, epoch)
                writer.add_scalar(
                    f'{run_name}/Test/Epoch_Accuracy', test_acc, epoch)

            safe_log(f'[{run_name}] Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | Valid Loss: {valid_loss:.4f} | Valid Acc: {valid_acc:.2f}% | Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%')

    except Exception as e:
        safe_log(f'[{run_name}] Fatal error during training: {str(e)}')
        raise

    finally:
        if verbose:
            safe_log(f"[{run_name}] Cleaning up...")

        # Clean up memory
        if writer:
            try:
                writer.flush()
                writer.close()
            except Exception as e:
                safe_log(f'[{run_name}] Error closing writer: {str(e)}')

        # Clear GPU memory if used
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception as e:
                safe_log(f'[{run_name}] Error clearing GPU memory: {str(e)}')

        gc.collect()

    return history


def run_single_experiment(model: str,
                          params: dict[str, float | int],
                          base_dir: Path,
                          num_workers: int,
                          data_dir: str | Path,
                          verbose: bool = True
                          ) -> dict[str, list[float]]:
    """Run a single training experiment with the given hyperparameters and configuration.

    Args:
        model: Name of the model to train
        params: Dictionary containing training hyperparameters:
            - num_epochs: Number of training epochs
            - learning_rate: Learning rate for optimizer
            - batch_size: Batch size for training
            - fold_id: Cross validation fold ID
        base_dir: Base directory path to save experiment outputs
        num_workers: Number of worker processes for data loading
        data_dir: Directory containing the dataset files
        verbose: If True, print detailed progress messages

    Returns:
        Dictionary containing training history with metrics per epoch:
            - train_loss: Training losses
            - train_acc: Training accuracies
            - valid_loss: Validation losses
            - valid_acc: Validation accuracies
            - test_loss: Test losses
            - test_acc: Test accuracies

    Raises:
        RuntimeError: If experiment fails due to memory or other runtime issues
        FileNotFoundError: If data directory does not exist
        ValueError: If invalid parameter values are provided
    """
    try:
        run_name = f"{'t' if model == 'transformer' else 'm'}_e{params['num_epochs']}_lr{params['learning_rate']}_b{params['batch_size']}_f{params['fold_id']}"
        run_dir = base_dir / run_name
        run_dir.mkdir(exist_ok=True)

        # Initialize TensorBoard writer
        if verbose:
            safe_log(f"[{run_name}] Initializing TensorBoard...")
        writer = SummaryWriter(log_dir=str(run_dir / 'tensorboard'))

        # Create datasets
        if verbose:
            safe_log(f"[{run_name}] Creating datasets...")
        train_dataset = SleepDataset(
            data_dir, fold_id=params['fold_id'], train_mode=True, split='train')
        valid_dataset = SleepDataset(
            data_dir, fold_id=params['fold_id'], train_mode=True, split='valid')
        test_dataset = SleepDataset(
            data_dir, fold_id=params['fold_id'], train_mode=False)

        # Create dataloaders
        if verbose:
            safe_log(f"[{run_name}] Creating dataloaders...")
        train_loader = DataLoader(
            train_dataset,
            batch_size=params['batch_size'],
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=True,
            # worker_init_fn=lambda worker_id: np.random.seed(np.random.get_state()[1][0] + worker_id)
        )
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=params['batch_size'],
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=True,
            # worker_init_fn=lambda worker_id: np.random.seed(np.random.get_state()[1][0] + worker_id)
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=params['batch_size'],
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=True,
            # worker_init_fn=lambda worker_id: np.random.seed(np.random.get_state()[1][0] + worker_id)
        )

        # Initialize model
        if verbose:
            safe_log(f"[{run_name}] Initializing model...")
        if model == "lstm":
            model_class = SleepClassifierLSTM
        else:
            model_class = SleepClassifierTransformer
        model = model_class(num_classes=5)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if verbose:
            safe_log(f"[{run_name}] Training on device: {device}")
        model = model.to(device)

        # Compile model I WOULD LOVE TO USE THIS BUT SOO MANY ISSUES....
        if verbose:
            safe_log(f"[{run_name}] Compiling model...")
        model = torch.compile(model, backend='eager')  # , mode='max-autotune')
        # Train model
        if verbose:
            safe_log(f"\n[{run_name}] Starting run\n[{run_name}] Parameters: {params}\n[{run_name}] Training model...")

        history = train_model(
            model=model,
            train_loader=train_loader,
            valid_loader=valid_loader,
            test_loader=test_loader,
            num_epochs=params['num_epochs'],
            learning_rate=params['learning_rate'],
            writer=writer,
            run_name=run_name,
            verbose=False
        )

        # Save training history
        if verbose:
            safe_log(f"[{run_name}] Saving training history...")
        history_path = run_dir / 'history.json'
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=4)
        if verbose:
            safe_log(f"[{run_name}] History saved to {history_path}")

        # Save model
        if verbose:
            safe_log(f"[{run_name}] Saving model...")
        model_path = run_dir / 'model.pth'
        torch.save(model.state_dict(), model_path)
        if verbose:
            safe_log(f"[{run_name}] Model saved to {model_path}")
            safe_log(f"[{run_name}] Closing writer...")
        writer.close()

    except Exception as e:
        safe_log(f'[{run_name}] Fatal error in experiment: {str(e)}')
        traceback.print_exc()
        raise
    finally:
        try:
            if 'train_loader' in locals():
                del train_loader
            if 'valid_loader' in locals():
                del valid_loader
            if 'test_loader' in locals():
                del test_loader
            if 'model' in locals():
                del model
            torch.cuda.empty_cache()
            gc.collect()
        except Exception as e:
            safe_log(f'Error during cleanup: {str(e)}')


def run_experiment(model: str = "lstm",
                   num_workers: int = 5,
                   data_dir: str = "data/preprocessed",
                   param_grid: dict[str, list[int | float]] = {'num_epochs': [10, 25, 50, 100],
                                                               'learning_rate': [0.0001, 0.0003, 0.001, 0.003],
                                                               'batch_size': [64, 128, 256, 512],
                                                               'fold_id': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]},
                   max_parallel: int = 1) -> None:
    """Run experiments with different parameter combinations in parallel.

    This function executes a grid search over the provided parameter combinations,
    training multiple models in parallel. For each parameter combination, it:
    1. Creates a unique model directory
    2. Initializes data loaders and model
    3. Trains the model and tracks metrics
    4. Saves the model weights and training history

    Args:
        model: Name of the model to train. Defaults to 'lstm'.
        num_workers: Number of worker processes for data loading. Defaults to 5.
        data_dir: Path to directory containing preprocessed HDF5 data files. 
            Defaults to "data/preprocessed".
        param_grid: Dictionary mapping parameter names to lists of values to try.
            Must contain 'num_epochs', 'learning_rate', 'batch_size', and 'fold_id'.
            Defaults to standard parameter ranges.
        max_parallel: Maximum number of training runs to execute simultaneously.
            Defaults to 1 (sequential execution).

    Raises:
        RuntimeError: If training fails for any parameter combination
        ValueError: If param_grid is missing required parameters
        FileNotFoundError: If data_dir does not exist
    """
    # Create base model directory
    base_dir = Path("./models")
    base_dir.mkdir(exist_ok=True)

    # Generate all combinations of parameters
    param_combinations = [dict(zip(param_grid.keys(), v))
                          for v in itertools.product(*param_grid.values())]

    # Filter out combinations where model already exists
    filtered_combinations = []
    for params in param_combinations:
        run_name = f"{'t' if model == 'transformer' else 'm'}_e{params['num_epochs']}_lr{params['learning_rate']}_b{params['batch_size']}_f{params['fold_id']}"
        model_path = base_dir / run_name / 'model.pth'
        if not model_path.exists():
            filtered_combinations.append(params)
        else:
            safe_log(f"Skipping {run_name} - model already exists")

    # Run experiments in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_parallel) as executor:
        futures = [
            executor.submit(
                run_single_experiment,
                model,
                params,
                base_dir,
                num_workers,
                data_dir
            )
            for params in filtered_combinations
        ]
        concurrent.futures.wait(futures)


if __name__ == "__main__":
    try:
        # Set start method to spawn
        # multiprocessing.set_start_method('spawn', force=True)
        import argparse
        # Set up argument parser
        parser = argparse.ArgumentParser(
            description='Train sleep stage classification models')
        parser.add_argument('-d', '--data_dir', type=str, default='./data/preprocessed/',
                            help='Directory containing preprocessed data')
        parser.add_argument('-n', '--num_workers', type=int, default=4,
                            help='Number of worker processes for data loading')
        parser.add_argument('-p', '--max_parallel', type=int, default=1,
                            help='Maximum number of training runs to execute simultaneously')
        parser.add_argument('-c', '--config', type=str, default='config/training_config.yaml',
                            help='Path to configuration file')
        parser.add_argument('-e', '--experiment', type=str,
                            help='Specific experiment configuration to use (e.g., quick_test)')
        args = parser.parse_args()

        # Load configuration
        config = load_config(args.config, args.experiment)

        run_experiment(
            model=config['training']['model'],
            num_workers=args.num_workers if args.num_workers else config['training']['num_workers'],
            data_dir=config['training']['data_dir'],
            param_grid=config[args.experiment] if args.experiment else config['manu_test'],
            max_parallel=args.max_parallel if args.max_parallel else config['training']['max_parallel']
        )
    except Exception as e:
        print(f"Fatal error: {str(e)}")
        traceback.print_exc()
        sys.exit(1)
    finally:
        # Final cleanup
        for p in multiprocessing.active_children():
            p.terminate()
        torch.cuda.empty_cache()
        gc.collect()
    
    
    #### Print Example ####
    # [m_e10_lr0.0003_b512_f0] Batch 28 loss: 0.4473
    # [m_e10_lr0.0003_b512_f0] Batch 29 loss: 0.3828
    # [m_e10_lr0.0003_b512_f0] Batch 30 loss: 0.4333
    # [m_e10_lr0.0003_b512_f0] Batch 31 loss: 0.4504
    # [m_e10_lr0.0003_b512_f0] Batch 32 loss: 0.4372
    # [m_e10_lr0.0003_b512_f0] Batch 33 loss: 0.4963
    # [m_e10_lr0.0003_b512_f0] Batch 34 loss: 0.4618
    # [m_e10_lr0.0003_b512_f0] Batch 35 loss: 0.3977
    # [m_e10_lr0.0003_b512_f0] Batch 36 loss: 0.4447
    # [m_e10_lr0.0003_b512_f0] Batch 37 loss: 0.3901
    # [m_e10_lr0.0003_b512_f0] Batch 38 loss: 0.3660
    # 2025-01-20 13:38:22 | [m_e10_lr0.0003_b512_f0] Epoch 4/10 | Train Loss: 0.4631 | Train Acc: 87.45% | Valid Loss: 0.4135 | Valid Acc: 88.86% | Test Loss: 0.4626 | Test Acc: 84.33%
    # 2025-01-20 13:38:22 | [m_e10_lr0.0003_b512_f0] Starting epoch 5
    # [m_e10_lr0.0003_b512_f0] Batch 1 loss: 0.5099
    # [m_e10_lr0.0003_b512_f0] Batch 2 loss: 0.3412
    # [m_e10_lr0.0003_b512_f0] Batch 3 loss: 0.4415
    # [m_e10_lr0.0003_b512_f0] Batch 4 loss: 0.3700
