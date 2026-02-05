
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.append('./source')

from onlineMROTauc_eval import OnlineMROTAD
from onlineMROTwassertein import OnlineMROTADWassertein
from utils import plot_auc_over_time, plot_auc_with_drift, split_data


class ExperimentConfig:
    """Configuration for a single experiment."""
    
    def __init__(
        self,
        dataset_path: str,
        dataset_name: str,
        n_train_samples: int = 2000,
        mrot_params: Optional[Dict] = None,
        window_size: int = 400,
        n_history: int = 5,
        m_barycenter: int = 3,
        theta_validation: float = 0.80,
        tau_anomaly: float = 0.75,
        K_retrain: int = 5,
        drift_threshold: float = 0.5
    ):
        self.dataset_path = dataset_path
        self.dataset_name = dataset_name
        self.n_train_samples = n_train_samples
        self.mrot_params = mrot_params or {}
        self.window_size = window_size
        self.n_history = n_history
        self.m_barycenter = m_barycenter
        self.theta_validation = theta_validation
        self.tau_anomaly = tau_anomaly
        self.K_retrain = K_retrain
        self.drift_threshold = drift_threshold


class ExperimentMetrics:
    """Store and manage experiment metrics."""
    
    def __init__(self, dataset_name: str, method_name: str):
        self.dataset_name = dataset_name
        self.method_name = method_name
        self.auc_scores = []
        self.drift_indicators = []
        self.wasserstein_scores = []
        self.mean_auc = None
        self.std_auc = None
        self.num_drifts = None
        self.execution_time = None
        
    def calculate_summary_stats(self):
        """Calculate summary statistics from collected metrics."""
        if self.auc_scores:
            self.mean_auc = np.mean(self.auc_scores)
            self.std_auc = np.std(self.auc_scores)
        if self.drift_indicators:
            self.num_drifts = sum(self.drift_indicators)
    
    def to_dict(self) -> Dict:
        """Convert metrics to dictionary for saving."""
        return {
            'dataset_name': self.dataset_name,
            'method_name': self.method_name,
            'auc_scores': self.auc_scores,
            'drift_indicators': self.drift_indicators,
            'wasserstein_scores': self.wasserstein_scores,
            'mean_auc': self.mean_auc,
            'std_auc': self.std_auc,
            'num_drifts': self.num_drifts,
            'execution_time': self.execution_time
        }


class MultiDatasetExperimentRunner:
    """Run experiments across multiple datasets and methods."""
    
    def __init__(self, output_dir: str = "./results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results = []
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def load_dataset(self, dataset_path: str) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
        """Load and prepare dataset."""
        print(f"Loading dataset: {dataset_path}")
        df = pd.read_csv(dataset_path)
        
        data = df.iloc[:, :-2]
        drift_labels = df.iloc[:, -1]
        true_labels = df.iloc[:, -2]
        
        return data, drift_labels, true_labels
    
    def run_online_mrotad_tumbling(
        self, 
        config: ExperimentConfig,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_true_train: pd.Series,
        y_true_test: pd.Series
    ) -> ExperimentMetrics:
        """Run Online MROTAD with tumbling window."""
        print(f"  Running Online MROTAD (Tumbling Window)...")
        
        metrics = ExperimentMetrics(config.dataset_name, "MROTAD_Tumbling")
        start_time = datetime.now()
        
        try:
            od = OnlineMROTAD(
                mrot_params=config.mrot_params,
                window_size=config.window_size,
                n_history=config.n_history,
                m_barycenter=config.m_barycenter,
                theta_validation=config.theta_validation,
                tau_anomaly=config.tau_anomaly,
                K_retrain=config.K_retrain,
                data_online=X_test,
                y_true_online=y_true_test,
                data_offline=X_train,
                y_true_offline=y_true_train,
                drift_threshold=config.drift_threshold
            )
            
            result = od.online_tumbling_window()
            metrics.auc_scores = result[0]
            metrics.drift_indicators = result[1]
            
        except Exception as e:
            print(f"    Error: {str(e)}")
            metrics.auc_scores = []
            metrics.drift_indicators = []
        
        metrics.execution_time = (datetime.now() - start_time).total_seconds()
        metrics.calculate_summary_stats()
        
        return metrics
    
    def run_online_mrotad_sliding(
        self, 
        config: ExperimentConfig,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_true_train: pd.Series,
        y_true_test: pd.Series
    ) -> ExperimentMetrics:
        """Run Online MROTAD with sliding window."""
        print(f"  Running Online MROTAD (Sliding Window)...")
        
        metrics = ExperimentMetrics(config.dataset_name, "MROTAD_Sliding")
        start_time = datetime.now()
        
        try:
            od = OnlineMROTAD(
                mrot_params=config.mrot_params,
                window_size=config.window_size,
                n_history=config.n_history,
                m_barycenter=config.m_barycenter,
                theta_validation=config.theta_validation,
                tau_anomaly=config.tau_anomaly,
                K_retrain=config.K_retrain,
                data_online=X_test,
                y_true_online=y_true_test,
                data_offline=X_train,
                y_true_offline=y_true_train,
                drift_threshold=config.drift_threshold
            )
            
            result = od.online_sliding_window()
            metrics.auc_scores = result[0]
            metrics.drift_indicators = result[1]
            
        except Exception as e:
            print(f"    Error: {str(e)}")
            metrics.auc_scores = []
            metrics.drift_indicators = []
        
        metrics.execution_time = (datetime.now() - start_time).total_seconds()
        metrics.calculate_summary_stats()
        
        return metrics
    
    def run_online_wasserstein_structured(
        self, 
        config: ExperimentConfig,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_true_train: pd.Series,
        y_true_test: pd.Series
    ) -> ExperimentMetrics:
        """Run Online MROTAD with Wasserstein (Structured)."""
        print(f"  Running Online Wasserstein (Structured)...")
        
        metrics = ExperimentMetrics(config.dataset_name, "Wasserstein_Structured")
        start_time = datetime.now()
        
        try:
            ow = OnlineMROTADWassertein(
                mrot_params=config.mrot_params,
                window_size=config.window_size,
                n_history=config.n_history,
                m_barycenter=config.m_barycenter,
                theta_validation=config.theta_validation,
                tau_anomaly=config.tau_anomaly,
                K_retrain=config.K_retrain,
                data_online=X_test,
                y_true_online=y_true_test,
                data_offline=X_train,
                y_true_offline=y_true_train,
                drift_threshold=config.drift_threshold
            )
            
            score_list, drift_detected_list, wasserstein_score = ow.online_wasserstein_structured()
            metrics.auc_scores = score_list
            metrics.drift_indicators = drift_detected_list
            metrics.wasserstein_scores = wasserstein_score
            
        except Exception as e:
            print(f"    Error: {str(e)}")
            metrics.auc_scores = []
            metrics.drift_indicators = []
            metrics.wasserstein_scores = []
        
        metrics.execution_time = (datetime.now() - start_time).total_seconds()
        metrics.calculate_summary_stats()
        
        return metrics
    
    def run_online_domain_adaptation(
        self, 
        config: ExperimentConfig,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_true_train: pd.Series,
        y_true_test: pd.Series
    ) -> ExperimentMetrics:
        """Run Online Domain Adaptation."""
        print(f"  Running Online Domain Adaptation...")
        
        metrics = ExperimentMetrics(config.dataset_name, "Domain_Adaptation")
        start_time = datetime.now()
        
        try:
            ow = OnlineMROTADWassertein(
                mrot_params=config.mrot_params,
                window_size=config.window_size,
                n_history=config.n_history,
                m_barycenter=config.m_barycenter,
                theta_validation=config.theta_validation,
                tau_anomaly=config.tau_anomaly,
                K_retrain=config.K_retrain,
                data_online=X_test,
                y_true_online=y_true_test,
                data_offline=X_train,
                y_true_offline=y_true_train,
                drift_threshold=config.drift_threshold
            )
            
            result = ow.online_domain_adaptation()
            metrics.auc_scores = result[0]
            metrics.drift_indicators = result[1]
            
        except Exception as e:
            print(f"    Error: {str(e)}")
            metrics.auc_scores = []
            metrics.drift_indicators = []
        
        metrics.execution_time = (datetime.now() - start_time).total_seconds()
        metrics.calculate_summary_stats()
        
        return metrics
    
    def run_single_dataset_experiment(
        self, 
        config: ExperimentConfig,
        methods: List[str] = None
    ) -> List[ExperimentMetrics]:
        """Run all methods on a single dataset."""
        print(f"\n{'='*60}")
        print(f"Dataset: {config.dataset_name}")
        print(f"{'='*60}")
        
        # Load dataset
        data, drift_labels, true_labels = self.load_dataset(config.dataset_path)
        
        # Split data
        X_train, X_test, y_drift_train, y_drift_test, y_true_train, y_true_test = split_data(
            data, drift_labels, true_labels, n_train_samples=config.n_train_samples
        )
        
        # Convert to DataFrame if needed
        X_train = pd.DataFrame(X_train, columns=data.columns)
        X_test = pd.DataFrame(X_test, columns=data.columns)
        
        # Default methods
        if methods is None:
            methods = ['tumbling', 'sliding', 'wasserstein', 'domain_adaptation']
        
        dataset_metrics = []
        
        # Run each method
        if 'tumbling' in methods:
            metrics = self.run_online_mrotad_tumbling(
                config, X_train, X_test, y_true_train, y_true_test
            )
            dataset_metrics.append(metrics)
            self.results.append(metrics)
        
        if 'sliding' in methods:
            metrics = self.run_online_mrotad_sliding(
                config, X_train, X_test, y_true_train, y_true_test
            )
            dataset_metrics.append(metrics)
            self.results.append(metrics)
        
        if 'wasserstein' in methods:
            metrics = self.run_online_wasserstein_structured(
                config, X_train, X_test, y_true_train, y_true_test
            )
            dataset_metrics.append(metrics)
            self.results.append(metrics)
        
        if 'domain_adaptation' in methods:
            metrics = self.run_online_domain_adaptation(
                config, X_train, X_test, y_true_train, y_true_test
            )
            dataset_metrics.append(metrics)
            self.results.append(metrics)
        
        return dataset_metrics
    
    def run_multiple_datasets(
        self, 
        configs: List[ExperimentConfig],
        methods: List[str] = None
    ):
        """Run experiments on multiple datasets."""
        print("\n" + "="*60)
        print("STARTING MULTI-DATASET EXPERIMENTS")
        print("="*60)
        
        for config in configs:
            self.run_single_dataset_experiment(config, methods)
        
        print("\n" + "="*60)
        print("ALL EXPERIMENTS COMPLETED")
        print("="*60)
    
    def save_results(self):
        """Save all results to files."""
        # Create experiment directory
        exp_dir = self.output_dir / f"experiment_{self.timestamp}"
        exp_dir.mkdir(parents=True, exist_ok=True)
        
        # Save detailed results as JSON
        results_dict = [m.to_dict() for m in self.results]
        with open(exp_dir / "detailed_results.json", 'w') as f:
            json.dump(results_dict, f, indent=2, default=str)
        
        # Save summary as CSV
        summary_data = []
        for m in self.results:
            summary_data.append({
                'Dataset': m.dataset_name,
                'Method': m.method_name,
                'Mean_AUC': m.mean_auc,
                'Std_AUC': m.std_auc,
                'Num_Drifts': m.num_drifts,
                'Execution_Time': m.execution_time
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(exp_dir / "summary_results.csv", index=False)
        
        print(f"\nResults saved to: {exp_dir}")
        return exp_dir
    
    def generate_comparison_plots(self, save_dir: Optional[Path] = None):
        """Generate comparison plots across datasets and methods."""
        if save_dir is None:
            save_dir = self.output_dir / f"experiment_{self.timestamp}" / "plots"
        save_dir.mkdir(parents=True, exist_ok=True)
        
        self._plot_auc_comparison(save_dir)
        
        self._plot_drift_comparison(save_dir)
        
        self._plot_execution_time(save_dir)
        
        self._plot_individual_auc_curves(save_dir)
        
        print(f"\nPlots saved to: {save_dir}")
    
    def _plot_auc_comparison(self, save_dir: Path):
        """Plot AUC comparison across methods and datasets."""
        data = []
        for m in self.results:
            if m.mean_auc is not None:
                data.append({
                    'Dataset': m.dataset_name,
                    'Method': m.method_name,
                    'Mean_AUC': m.mean_auc,
                    'Std_AUC': m.std_auc
                })
        
        if not data:
            return
        
        df = pd.DataFrame(data)
        
        # Group by dataset
        datasets = df['Dataset'].unique()
        n_datasets = len(datasets)
        
        fig, axes = plt.subplots(1, n_datasets, figsize=(6*n_datasets, 5), squeeze=False)
        axes = axes.flatten()
        
        for idx, dataset in enumerate(datasets):
            dataset_df = df[df['Dataset'] == dataset]
            ax = axes[idx]
            
            x = range(len(dataset_df))
            ax.bar(x, dataset_df['Mean_AUC'], yerr=dataset_df['Std_AUC'], 
                   capsize=5, alpha=0.7, color=sns.color_palette("husl", len(dataset_df)))
            ax.set_xticks(x)
            ax.set_xticklabels(dataset_df['Method'], rotation=45, ha='right')
            ax.set_ylabel('Mean AUC')
            ax.set_title(f'{dataset}')
            ax.set_ylim([0, 1])
            ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_dir / 'auc_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_drift_comparison(self, save_dir: Path):
        """Plot drift detection comparison."""
        data = []
        for m in self.results:
            if m.num_drifts is not None:
                data.append({
                    'Dataset': m.dataset_name,
                    'Method': m.method_name,
                    'Num_Drifts': m.num_drifts
                })
        
        if not data:
            return
        
        df = pd.DataFrame(data)
        
        plt.figure(figsize=(12, 6))
        sns.barplot(data=df, x='Dataset', y='Num_Drifts', hue='Method')
        plt.title('Number of Detected Drifts by Method and Dataset')
        plt.ylabel('Number of Drifts')
        plt.xticks(rotation=45, ha='right')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(save_dir / 'drift_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_execution_time(self, save_dir: Path):
        """Plot execution time comparison."""
        data = []
        for m in self.results:
            if m.execution_time is not None:
                data.append({
                    'Dataset': m.dataset_name,
                    'Method': m.method_name,
                    'Execution_Time': m.execution_time
                })
        
        if not data:
            return
        
        df = pd.DataFrame(data)
        
        plt.figure(figsize=(12, 6))
        sns.barplot(data=df, x='Dataset', y='Execution_Time', hue='Method')
        plt.title('Execution Time by Method and Dataset')
        plt.ylabel('Time (seconds)')
        plt.xticks(rotation=45, ha='right')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(save_dir / 'execution_time_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_individual_auc_curves(self, save_dir: Path):
        """Plot individual AUC curves over time."""
        for m in self.results:
            if not m.auc_scores:
                continue
            
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Plot AUC scores
            ax.plot(m.auc_scores, label='AUC', linewidth=2)
            
            # Mark drift points
            if m.drift_indicators:
                drift_points = [i for i, d in enumerate(m.drift_indicators) if d == 1]
                if drift_points:
                    ax.scatter(drift_points, [m.auc_scores[i] for i in drift_points],
                             color='red', s=100, marker='x', label='Drift Detected', zorder=5)
            
            ax.set_xlabel('Window Index')
            ax.set_ylabel('AUC Score')
            ax.set_title(f'{m.dataset_name} - {m.method_name}')
            ax.legend()
            ax.grid(alpha=0.3)
            
            filename = f"auc_curve_{m.dataset_name}_{m.method_name}.png"
            plt.savefig(save_dir / filename, dpi=300, bbox_inches='tight')
            plt.close()
    
    def print_summary(self):
        """Print summary of all results."""
        print("\n" + "="*80)
        print("EXPERIMENT SUMMARY")
        print("="*80)
        
        for m in self.results:
            print(f"\n{m.dataset_name} - {m.method_name}")
            print(f"  Mean AUC: {m.mean_auc:.4f} ± {m.std_auc:.4f}" if m.mean_auc else "  Mean AUC: N/A")
            print(f"  Drifts Detected: {m.num_drifts}")
            print(f"  Execution Time: {m.execution_time:.2f}s")
