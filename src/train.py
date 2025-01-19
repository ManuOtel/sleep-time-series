from data.dataset import SleepDataset
from sleep_classifier import SleepClassifierLSTM as SleepClassifier
import json
import gc
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import logging
from torch.utils.tensorboard import SummaryWriter
import itertools
import concurrent.futures
import threading
import torch._dynamo
torch._dynamo.config.suppress_errors = True


# Configure thread-safe logging
# ============================

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


def safe_log(message: str) -> None:
    """
    Thread-safe logging function.

    Args:
        message: The message to log
    """
    with log_lock:
        logger.info(message)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    num_epochs: int = 50,
    learning_rate: float = 0.001,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    writer: SummaryWriter = None,
    run_name: str = "",
    verbose: bool = False
) -> dict:
    """
    Train the sleep classification model

    Args:
        model: The neural network model
        train_loader: DataLoader for training data
        valid_loader: DataLoader for validation data
        num_epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        device: Device to train on ('cuda' or 'cpu')
        writer: TensorBoard writer
        run_name: Name of the current run for logging
        verbose: Whether to print detailed progress

    Returns:
        dict containing training history
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
            'valid_acc': []
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
                        safe_log(f"[{run_name}] Moving batch {
                                 batch_idx+1} to device...")

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

                    print(f'[{run_name}] Batch {
                          batch_idx+1} loss: {loss.item():.4f}')

                except Exception as e:
                    safe_log(f'[{run_name}] Error in training batch {
                             batch_idx}: {str(e)}')
                    raise

            # Validation
            if verbose:
                safe_log(f"[{run_name}] Starting validation...")
                safe_log(f"[{run_name}] Setting model to eval mode...")

            model.eval()
            valid_loss = 0
            valid_correct = 0
            valid_total = 0

            with torch.no_grad():
                for batch_idx, (batch_data, batch_labels) in enumerate(valid_loader):
                    try:
                        if verbose:
                            safe_log(f"[{run_name}] Processing validation batch {
                                     batch_idx+1}...")

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
                        safe_log(f'[{run_name}] Error in validation batch {
                                 batch_idx}: {str(e)}')
                        raise

            if verbose:
                safe_log(f"[{run_name}] Computing epoch metrics...")

            # Record metrics
            train_loss = train_loss / len(train_loader)
            train_acc = 100. * train_correct / train_total
            valid_loss = valid_loss / len(valid_loader)
            valid_acc = 100. * valid_correct / valid_total

            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['valid_loss'].append(valid_loss)
            history['valid_acc'].append(valid_acc)

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

            safe_log(f'[{run_name}] Epoch {epoch+1}/{num_epochs}:')
            safe_log(f'[{run_name}] Train Loss: {
                     train_loss:.4f} | Train Acc: {train_acc:.2f}%')
            safe_log(f'[{run_name}] Valid Loss: {
                     valid_loss:.4f} | Valid Acc: {valid_acc:.2f}%')

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


def run_single_experiment(params, base_dir, num_workers, data_dir, num_classes, verbose=True):
    """Run a single experiment with given parameters"""
    try:
        # Create run directory
        run_name = f"m_e{params['num_epochs']}_lr{params['learning_rate']}_b{
            params['batch_size']}_f{params['fold_id']}"
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

        # Create dataloaders
        if verbose:
            safe_log(f"[{run_name}] Creating dataloaders...")
        train_loader = DataLoader(
            train_dataset,
            batch_size=params['batch_size'],
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            prefetch_factor=4,
            persistent_workers=True
        )
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=params['batch_size'],
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            prefetch_factor=4,
            persistent_workers=True
        )

        # Initialize model
        if verbose:
            safe_log(f"[{run_name}] Initializing model...")
        model = SleepClassifier(num_classes=num_classes)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if verbose:
            safe_log(f"[{run_name}] Training on device: {device}")
        model = model.to(device)

        # Compile model
        # if verbose:
        #     safe_log(f"[{run_name}] Compiling model...")
        # model = torch.compile(model)
        # Train model
        if verbose:
            safe_log(f"\n[{run_name}] Starting run")
            safe_log(f"[{run_name}] Parameters: {params}")
            safe_log(f"[{run_name}] Training model...")

        history = train_model(
            model=model,
            train_loader=train_loader,
            valid_loader=valid_loader,
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
        raise


def run_experiment(
    num_workers: int = 4,
    data_dir: str = "data/preprocessed",
    num_classes: int = 5,
    param_grid: dict = {
        # Train for longer to ensure convergence
        'num_epochs': [50, 25, 100, 10],  # Start with moderate epochs
        'learning_rate': [0.0003, 0.001, 0.0001],  # Start with middle LR
        'batch_size': [128, 256, 64],  # Start with moderate batch size
        'fold_id': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]  # 10-fold cross-validation
    },
    max_parallel: int = 3
):
    """
    Run experiments with different parameter combinations in parallel

    Args:
        num_workers: Number of data loading workers
        data_dir: Directory containing preprocessed data
        num_classes: Number of sleep stage classes
        param_grid: Dictionary containing parameter combinations to try
        max_parallel: Maximum number of parallel training runs
    """

    # Create base model directory
    base_dir = Path("./models")
    base_dir.mkdir(exist_ok=True)

    # Generate all combinations of parameters
    param_combinations = [dict(zip(param_grid.keys(), v))
                          for v in itertools.product(*param_grid.values())]

    # Run experiments in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_parallel) as executor:
        futures = [
            executor.submit(
                run_single_experiment,
                params,
                base_dir,
                num_workers,
                data_dir,
                num_classes
            )
            for params in param_combinations
        ]
        concurrent.futures.wait(futures)


if __name__ == "__main__":
    run_experiment()
