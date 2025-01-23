"""
This module provides functionality for analyzing model training results and performance.

The main purpose is to analyze training histories from different model runs to understand:
- Final model performance across architectures and hyperparameters
- Training dynamics like convergence speed and overfitting
- Cross-fold validation results and statistical significance
- Model comparison insights

The module contains:
    1. TrainingAnalyzer class for computing performance metrics and statistics
    2. Command line interface for analyzing model directories
    3. Visualization and reporting of key findings

The analysis results help identify:
    - Best performing model architectures and configurations
    - Training stability and convergence patterns  
    - Cross-validation robustness
    - Areas for potential improvement

The results are used to select optimal models and inform future architecture decisions.
"""


import os
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from typing import Dict, List, Any

class TrainingAnalyzer:
    def __init__(self, history: Dict[str, List[float]]) -> None:
        """Initialize the TrainingAnalyzer with a training history.
        
        Args:
            history: Dictionary containing training metrics over epochs.
                    Expected keys are of form '{phase}_{metric}' where
                    phase is train/valid/test and metric is loss/acc.
        """
        self.history = history
        self.metrics = ['loss', 'acc']
        self.phases = ['train', 'valid', 'test']
        
    def final_performance_stats(self) -> Dict[str, float]:
        """Calculate final performance metrics and their statistics.
        
        Returns:
            Dictionary containing:
                - Final values for each metric/phase combination
                - Standard deviations over last 3 epochs
                
        Example returned dict:
            {
                'final_train_acc': 0.95,
                'final_valid_acc': 0.93,
                'std_train_acc': 0.01,
                'std_valid_acc': 0.02,
                ...
            }
        """
        final_stats = {}
        
        # Get final epoch metrics
        for phase in self.phases:
            for metric in self.metrics:
                key = f'{phase}_{metric}'
                if key in self.history:
                    final_stats[f'final_{key}'] = self.history[key][-1]
                    
        # Calculate standard deviations over last 3 epochs
        for phase in self.phases:
            for metric in self.metrics:
                key = f'{phase}_{metric}'
                if key in self.history:
                    final_3_epochs = self.history[key][-3:]
                    final_stats[f'std_{key}'] = np.std(final_3_epochs)
                    
        return final_stats

    def analyze_training_dynamics(self) -> Dict[str, float]:
        """Analyze training dynamics including convergence and overfitting.
        
        Returns:
            Dictionary containing:
                - Convergence epoch for each metric (epoch where 90% of final performance is reached)
                - Average and final train-validation accuracy gaps
                
        Example returned dict:
            {
                'convergence_epoch_train_acc': 5,
                'convergence_epoch_valid_acc': 6,
                'avg_train_valid_gap': 0.03,
                'final_train_valid_gap': 0.02,
                ...
            }
        """
        dynamics = {}
        
        # Calculate convergence rate (epochs to 90% of final performance)
        for phase in self.phases:
            for metric in self.metrics:
                key = f'{phase}_{metric}'
                if key in self.history:
                    final_value = self.history[key][-1]
                    target = 0.9 * final_value
                    epochs = np.where(np.array(self.history[key]) >= target)[0]
                    dynamics[f'convergence_epoch_{key}'] = epochs[0] if len(epochs) > 0 else len(self.history[key])

        # Calculate train-valid gaps
        if all(f'{phase}_acc' in self.history for phase in ['train', 'valid']):
            train_acc = np.array(self.history['train_acc'])
            valid_acc = np.array(self.history['valid_acc'])
            dynamics['avg_train_valid_gap'] = np.mean(train_acc - valid_acc)
            dynamics['final_train_valid_gap'] = train_acc[-1] - valid_acc[-1]

        return dynamics

    def stability_metrics(self) -> Dict[str, float]:
        """Calculate model training stability metrics.
        
        Returns:
            Dictionary containing for each metric/phase:
                - Improvement rate (fraction of epochs with improvement)
                - Variance across epochs
                - Trend consistency (Kendall's tau correlation with epochs)
                
        Example returned dict:
            {
                'improvement_rate_train_acc': 0.8,
                'variance_train_acc': 0.001,
                'trend_consistency_train_acc': 0.95,
                ...
            }
        """
        stability = {}
        
        for phase in self.phases:
            for metric in self.metrics:
                key = f'{phase}_{metric}'
                if key in self.history:
                    values = np.array(self.history[key])
                    
                    # Calculate improvement consistency
                    improvements = np.diff(values) > 0
                    stability[f'improvement_rate_{key}'] = np.mean(improvements)
                    
                    # Calculate variance across epochs
                    stability[f'variance_{key}'] = np.var(values)
                    
                    # Calculate trend consistency using Kendall's tau
                    epochs = np.arange(len(values))
                    tau, _ = stats.kendalltau(epochs, values)
                    stability[f'trend_consistency_{key}'] = tau
                    
        return stability

    def cross_validation_stats(self, all_histories: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
        """Calculate cross-validation statistics across folds.
        
        Args:
            all_histories: Dictionary mapping folder names to training histories
                         Format: {'model_config_fold0': {'history': {...}}, ...}
                         
        Returns:
            Dictionary containing cross-validation statistics for each model config:
                - Mean and std of final metrics (test/valid acc/loss)
                - P-values from t-tests against random baseline
                
        Example returned dict:
            {
                'lstm_mean_test_acc': 93.5,
                'lstm_std_test_acc': 2.1,
                'lstm_pvalue_test_acc': 0.001,
                ...
            }
        """
        cv_stats = {}
        
        # Group histories by model configuration (excluding fold number)
        model_groups = {}
        for folder, data in all_histories.items():
            # Create key without fold number
            config_key = '_'.join(folder.split('_')[:-1])  # Remove fold part
            if config_key not in model_groups:
                model_groups[config_key] = []
            model_groups[config_key].append(data['history'])
        
        # Calculate statistics for each model configuration
        for config, histories in model_groups.items():
            final_metrics = {
                'test_acc': [],
                'valid_acc': [],
                'test_loss': [],
                'valid_loss': []
            }
            
            # Collect final metrics from each fold
            for hist in histories:
                for metric in final_metrics.keys():
                    if metric in hist:
                        final_metrics[metric].append(hist[metric][-1])
            
            # Calculate statistics
            for metric, values in final_metrics.items():
                if values:
                    values = np.array(values)
                    cv_stats[f'{config}_mean_{metric}'] = np.mean(values)
                    cv_stats[f'{config}_std_{metric}'] = np.std(values)
                    
                    # Perform t-test against random baseline (50% for accuracy)
                    if 'acc' in metric:
                        t_stat, p_value = stats.ttest_1samp(values, 50.0)
                        cv_stats[f'{config}_pvalue_{metric}'] = p_value
        
        return cv_stats

    def plot_learning_curves(self, save_path: str = None) -> None:
        """Plot learning curves for all metrics.
        
        Args:
            save_path: Optional path to save the plot figure. If None, plot is only displayed.
            
        Returns:
            None
        """
        epochs = range(1, len(self.history['train_loss']) + 1)
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
        
        # Plot losses
        ax1.set_title('Model Loss')
        for phase in self.phases:
            key = f'{phase}_loss'
            if key in self.history:
                ax1.plot(epochs, self.history[key], label=phase)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Plot accuracies
        ax2.set_title('Model Accuracy')
        for phase in self.phases:
            key = f'{phase}_acc'
            if key in self.history:
                ax2.plot(epochs, self.history[key], label=phase)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
        plt.close()

    def run_analysis(self, all_histories: Dict = None) -> Dict:
        """Run complete analysis and return all metrics"""
        analysis_results = {
            'final_performance': self.final_performance_stats(),
            'training_dynamics': self.analyze_training_dynamics(),
            'stability': self.stability_metrics()
        }
        
        # Add cross-validation stats if all_histories is provided
        if all_histories:
            analysis_results['cross_validation'] = self.cross_validation_stats(all_histories)
        
        # Generate plots
        self.plot_learning_curves('./visualizations/analysis/learning_curves.png')
        
        return analysis_results

if __name__ == "__main__":
    import os
    import json
    import argparse

    # Set up argument parser
    parser = argparse.ArgumentParser(description='Analyze model training results')
    parser.add_argument('--models_dir', type=str, default='./models',
                        help='Directory containing model folders')
    parser.add_argument('--vis_dir', type=str, default='./visualizations/analysis',
                        help='Directory to save visualization plots')
    parser.add_argument('--top_k', type=int, default=5,
                        help='Number of top models to show in leaderboard')
    args = parser.parse_args()

    # Get all folders in models directory
    model_folders = [f for f in os.listdir(args.models_dir) if os.path.isdir(os.path.join(args.models_dir, f))]
    
    # Dictionary to store all histories
    all_histories = {}
    
    for folder in model_folders:
        history_path = os.path.join(args.models_dir, folder, 'history.json')
        
        # Check if history.json exists
        if os.path.exists(history_path):
            # Parse folder name
            # Example format: m_e10_lr0.0003_b128_f0
            parts = folder.split('_')
            model_info = {
                'model_type': 'lstm' if parts[0] == 'm' else 'transformer',
                'epochs': int(parts[1][1:]),  # Remove 'e' prefix
                'learning_rate': float(parts[2][2:]),  # Remove 'lr' prefix
                'batch_size': int(parts[3][1:]),  # Remove 'b' prefix
                'fold': int(parts[4][1:])  # Remove 'f' prefix
            }
            
            # Read history file
            with open(history_path, 'r') as f:
                history = json.load(f)
                
            # Store both model info and history
            all_histories[folder] = {
                'info': model_info,
                'history': history
            }
    
    print(f"Found {len(all_histories)} model histories for analysis")
    
    # Create visualization directory if it doesn't exist
    os.makedirs(args.vis_dir, exist_ok=True)
    
    # Analyze all models and collect results
    model_results = []
    for folder, data in all_histories.items():
        analyzer = TrainingAnalyzer(data['history'])
        results = analyzer.run_analysis(all_histories)
        
        # Collect key metrics for each model
        model_metrics = {
            'folder': folder,
            'model_type': data['info']['model_type'],
            'learning_rate': data['info']['learning_rate'],
            'batch_size': data['info']['batch_size'],
            'test_acc': results['final_performance']['final_test_acc'],
            'test_loss': results['final_performance']['final_test_loss'],
            'convergence_epoch': results['training_dynamics']['convergence_epoch_test_acc'],
            'train_valid_gap': results['training_dynamics']['final_train_valid_gap']
        }
        model_results.append(model_metrics)

    # Sort and display leaderboards
    print("\n=== Model Performance Leaderboard ===")
    print("\nTop Models by Test Accuracy:")
    acc_leaderboard = sorted(model_results, key=lambda x: x['test_acc'], reverse=True)
    for i, model in enumerate(acc_leaderboard[:5], 1):
        print(f"{i}. {model['model_type']} (lr={model['learning_rate']}, bs={model['batch_size']})")
        print(f"   Test Acc: {model['test_acc']:.2f}%, Test Loss: {model['test_loss']:.4f}")
        print(f"   Folder: {model['folder']}")

    print("\nFastest Converging Models:")
    conv_leaderboard = sorted(model_results, key=lambda x: x['convergence_epoch'])
    for i, model in enumerate(conv_leaderboard[:3], 1):
        print(f"{i}. {model['model_type']} (lr={model['learning_rate']}, bs={model['batch_size']})")
        print(f"   Convergence Epoch: {model['convergence_epoch']}")
        print(f"   Final Test Acc: {model['test_acc']:.2f}%")

    print("\n=== Overall Statistics ===")
    lstm_results = [m for m in model_results if m['model_type'] == 'lstm']
    transformer_results = [m for m in model_results if m['model_type'] == 'transformer']

    print("\nLSTM vs Transformer Performance:")
    print(f"LSTM Avg Test Acc: {np.mean([m['test_acc'] for m in lstm_results]):.2f}%")
    print(f"Transformer Avg Test Acc: {np.mean([m['test_acc'] for m in transformer_results]):.2f}%")

    print("\nFold Performance Analysis:")
    for fold in range(10):  # Assuming 10 folds
        fold_results = [m for m in model_results if m['folder'].endswith(f'f{fold}')]
        if fold_results:
            fold_acc = np.mean([m['test_acc'] for m in fold_results])
            fold_std = np.std([m['test_acc'] for m in fold_results])
            print(f"Fold {fold}:")
            print(f"   Avg Test Acc: {fold_acc:.2f}% (±{fold_std:.2f})")
            best_fold_model = max(fold_results, key=lambda x: x['test_acc'])
            print(f"   Best Model: {best_fold_model['model_type']} "
                  f"(lr={best_fold_model['learning_rate']}, bs={best_fold_model['batch_size']})")

    # Calculate cross-fold statistics
    fold_means = []
    for fold in range(10):
        fold_results = [m for m in model_results if m['folder'].endswith(f'f{fold}')]
        if fold_results:
            fold_means.append(np.mean([m['test_acc'] for m in fold_results]))
    
    print("\nCross-fold Statistics:")
    print(f"Mean across folds: {np.mean(fold_means):.2f}%")
    print(f"Std across folds: {np.std(fold_means):.2f}%")

    print("\nBest Hyperparameters:")
    best_model = acc_leaderboard[0]
    print(f"Model Type: {best_model['model_type']}")
    print(f"Learning Rate: {best_model['learning_rate']}")
    print(f"Batch Size: {best_model['batch_size']}")


    #### Print Example ####
    # Found 48 model histories for analysis

    # === Model Performance Leaderboard ===

    # Top Models by Test Accuracy:
    # 1. lstm (lr=0.0003, bs=128)
    #    Test Acc: 96.92%, Test Loss: 0.1491
    #    Folder: m_e10_lr0.0003_b128_f1
    # 2. transformer (lr=0.0003, bs=128)
    #    Test Acc: 96.70%, Test Loss: 0.1500
    #    Folder: t_e10_lr0.0003_b128_f1
    # 3. lstm (lr=0.0003, bs=256)
    #    Test Acc: 96.65%, Test Loss: 0.1900
    #    Folder: m_e10_lr0.0003_b256_f0
    # 4. lstm (lr=0.0003, bs=128)
    #    Test Acc: 96.60%, Test Loss: 0.1681
    #    Folder: m_e10_lr0.0003_b128_f0
    # 5. transformer (lr=0.0003, bs=256)
    #    Test Acc: 96.57%, Test Loss: 0.1559
    #    Folder: t_e10_lr0.0003_b256_f1

    # Fastest Converging Models:
    # 1. transformer (lr=0.0003, bs=128)
    #    Convergence Epoch: 0
    #    Final Test Acc: 87.45%
    # 2. transformer (lr=0.0003, bs=128)
    #    Convergence Epoch: 0
    #    Final Test Acc: 93.52%
    # 3. transformer (lr=0.0003, bs=128)
    #    Convergence Epoch: 0
    #    Final Test Acc: 91.93%

    # === Overall Statistics ===

    # LSTM vs Transformer Performance:
    # LSTM Avg Test Acc: 93.74%
    # Transformer Avg Test Acc: 93.24%

    # Fold Performance Analysis:
    # Fold 0:
    #    Avg Test Acc: 95.15% (±1.59)
    #    Best Model: lstm (lr=0.0003, bs=256)
    # Fold 1:
    #    Avg Test Acc: 95.96% (±0.99)
    #    Best Model: lstm (lr=0.0003, bs=128)
    # Fold 2:
    #    Avg Test Acc: 88.84% (±1.25)
    #    Best Model: lstm (lr=0.0003, bs=256)
    # Fold 3:
    #    Avg Test Acc: 93.33% (±0.28)
    #    Best Model: lstm (lr=0.0003, bs=256)
    # Fold 4:
    #    Avg Test Acc: 91.49% (±0.27)
    #    Best Model: transformer (lr=0.0003, bs=128)
    # Fold 5:
    #    Avg Test Acc: 93.54% (±0.18)
    #    Best Model: lstm (lr=0.0003, bs=256)
    # Fold 6:
    #    Avg Test Acc: 95.66% (±0.48)
    #    Best Model: transformer (lr=0.0003, bs=128)
    # Fold 7:
    #    Avg Test Acc: 93.51% (±0.53)
    #    Best Model: transformer (lr=0.0003, bs=128)
    # Fold 8:
    #    Avg Test Acc: 95.24% (±0.23)
    #    Best Model: transformer (lr=0.0003, bs=256)
    # Fold 9:
    #    Avg Test Acc: 90.47% (±0.19)
    #    Best Model: transformer (lr=0.0003, bs=256)

    # Cross-fold Statistics:
    # Mean across folds: 93.32%
    # Std across folds: 2.26%

    # Best Hyperparameters:
    # Model Type: lstm
    # Learning Rate: 0.0003
    # Batch Size: 128