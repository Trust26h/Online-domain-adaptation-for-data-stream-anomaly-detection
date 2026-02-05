

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from typing import List, Dict
from scipy import stats


class ResultsComparator:
    """Compare and analyze results from multiple experiments."""
    
    def __init__(self, result_dirs: List[str]):
        """
        Initialize with list of result directories.
        
        Args:
            result_dirs: List of paths to experiment result directories
        """
        self.result_dirs = [Path(d) for d in result_dirs]
        self.experiments = {}
        self.load_all_results()
    
    def load_all_results(self):
        """Load results from all experiment directories."""
        for result_dir in self.result_dirs:
            exp_name = result_dir.name
            
            # Load summary CSV
            summary_path = result_dir / "summary_results.csv"
            if summary_path.exists():
                self.experiments[exp_name] = {
                    'summary': pd.read_csv(summary_path),
                    'path': result_dir
                }
                print(f"Loaded: {exp_name}")
            else:
                print(f"Warning: No summary found in {result_dir}")
    
    def compare_methods_across_experiments(self) -> pd.DataFrame:
        """Compare method performance across all experiments."""
        all_data = []
        
        for exp_name, exp_data in self.experiments.items():
            df = exp_data['summary'].copy()
            df['Experiment'] = exp_name
            all_data.append(df)
        
        combined = pd.concat(all_data, ignore_index=True)
        return combined
    
    def get_best_methods(self) -> pd.DataFrame:
        """Get best performing method for each dataset across all experiments."""
        combined = self.compare_methods_across_experiments()
        
        # Find best method per dataset
        best_methods = combined.loc[
            combined.groupby(['Dataset', 'Experiment'])['Mean_AUC'].idxmax()
        ]
        
        return best_methods[['Dataset', 'Experiment', 'Method', 'Mean_AUC', 'Std_AUC']]
    
    def statistical_comparison(self, dataset_name: str, method1: str, method2: str):
        """Perform statistical comparison between two methods on a dataset."""
        results = {
            'dataset': dataset_name,
            'method1': method1,
            'method2': method2,
            'comparisons': []
        }
        
        for exp_name, exp_data in self.experiments.items():
            df = exp_data['summary']
            
            # Get metrics for both methods
            m1_data = df[(df['Dataset'] == dataset_name) & (df['Method'] == method1)]
            m2_data = df[(df['Dataset'] == dataset_name) & (df['Method'] == method2)]
            
            if not m1_data.empty and not m2_data.empty:
                m1_auc = m1_data['Mean_AUC'].values[0]
                m2_auc = m2_data['Mean_AUC'].values[0]
                
                results['comparisons'].append({
                    'experiment': exp_name,
                    'method1_auc': m1_auc,
                    'method2_auc': m2_auc,
                    'difference': m1_auc - m2_auc,
                    'better': method1 if m1_auc > m2_auc else method2
                })
        
        return results
    
    def plot_method_comparison(self, save_path: str = None):
        """Create comprehensive comparison plots."""
        combined = self.compare_methods_across_experiments()
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. AUC comparison by method
        ax = axes[0, 0]
        sns.boxplot(data=combined, x='Method', y='Mean_AUC', ax=ax)
        ax.set_title('AUC Distribution by Method (All Datasets)')
        ax.set_ylabel('Mean AUC')
        ax.set_xlabel('Method')
        ax.tick_params(axis='x', rotation=45)
        
        # 2. AUC comparison by dataset
        ax = axes[0, 1]
        sns.boxplot(data=combined, x='Dataset', y='Mean_AUC', hue='Method', ax=ax)
        ax.set_title('AUC Comparison by Dataset')
        ax.set_ylabel('Mean AUC')
        ax.set_xlabel('Dataset')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # 3. Execution time comparison
        ax = axes[1, 0]
        sns.barplot(data=combined, x='Method', y='Execution_Time', 
                   estimator=np.mean, ci='sd', ax=ax)
        ax.set_title('Average Execution Time by Method')
        ax.set_ylabel('Time (seconds)')
        ax.set_xlabel('Method')
        ax.tick_params(axis='x', rotation=45)
        
        # 4. Drift detection comparison
        ax = axes[1, 1]
        drift_data = combined.groupby('Method')['Num_Drifts'].mean().sort_values(ascending=False)
        drift_data.plot(kind='bar', ax=ax, color='skyblue')
        ax.set_title('Average Number of Drifts Detected')
        ax.set_ylabel('Average Drifts')
        ax.set_xlabel('Method')
        ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Comparison plot saved to: {save_path}")
        
        plt.show()
    
    def generate_summary_report(self, output_path: str = "comparison_report.txt"):
        """Generate text-based summary report."""
        combined = self.compare_methods_across_experiments()
        
        report = []
        report.append("="*80)
        report.append("MULTI-EXPERIMENT COMPARISON REPORT")
        report.append("="*80)
        report.append(f"\nTotal Experiments: {len(self.experiments)}")
        report.append(f"Total Datasets: {combined['Dataset'].nunique()}")
        report.append(f"Total Methods: {combined['Method'].nunique()}")
        report.append(f"Total Runs: {len(combined)}")
        
        # Overall best methods
        report.append("\n" + "="*80)
        report.append("OVERALL BEST METHODS")
        report.append("="*80)
        best_overall = combined.groupby('Method').agg({
            'Mean_AUC': ['mean', 'std'],
            'Num_Drifts': 'mean',
            'Execution_Time': 'mean'
        }).round(4)
        best_overall_sorted = best_overall.sort_values(('Mean_AUC', 'mean'), ascending=False)
        report.append(str(best_overall_sorted))
        
        # Best method per dataset
        report.append("\n" + "="*80)
        report.append("BEST METHOD PER DATASET")
        report.append("="*80)
        for dataset in combined['Dataset'].unique():
            dataset_data = combined[combined['Dataset'] == dataset]
            best_idx = dataset_data['Mean_AUC'].idxmax()
            best_row = dataset_data.loc[best_idx]
            
            report.append(f"\n{dataset}:")
            report.append(f"  Best Method: {best_row['Method']}")
            report.append(f"  AUC: {best_row['Mean_AUC']:.4f} ± {best_row['Std_AUC']:.4f}")
            report.append(f"  Drifts: {best_row['Num_Drifts']:.0f}")
            report.append(f"  Time: {best_row['Execution_Time']:.2f}s")
        
        # Method stability (std of AUC across datasets)
        report.append("\n" + "="*80)
        report.append("METHOD STABILITY (Lower is more consistent)")
        report.append("="*80)
        stability = combined.groupby('Method')['Mean_AUC'].std().sort_values()
        report.append(str(stability))
        
        # Efficiency (AUC / Time ratio)
        report.append("\n" + "="*80)
        report.append("METHOD EFFICIENCY (AUC / Time Ratio)")
        report.append("="*80)
        combined['Efficiency'] = combined['Mean_AUC'] / combined['Execution_Time']
        efficiency = combined.groupby('Method')['Efficiency'].mean().sort_values(ascending=False)
        report.append(str(efficiency))
        
        # Save report
        report_text = "\n".join(report)
        with open(output_path, 'w') as f:
            f.write(report_text)
        
        print(f"Report saved to: {output_path}")
        print("\n" + report_text)
        
        return report_text
    
    def create_latex_comparison_table(self, output_path: str = "comparison_table.tex"):
        """Create LaTeX table comparing methods across experiments."""
        combined = self.compare_methods_across_experiments()
        
        # Pivot for better view
        pivot = combined.pivot_table(
            index=['Dataset', 'Experiment'],
            columns='Method',
            values='Mean_AUC',
            aggfunc='first'
        ).reset_index()
        
        # Format for LaTeX
        pivot = pivot.round(4)
        
        # Save as LaTeX
        latex_str = pivot.to_latex(index=False, escape=False)
        
        with open(output_path, 'w') as f:
            f.write(latex_str)
        
        print(f"LaTeX table saved to: {output_path}")
        
        return latex_str


def main():
    """Example usage of ResultsComparator."""
    
    # Example: Compare results from multiple experiment runs
    # Replace these with your actual result directories
    result_directories = [
        "./results/experiment_20250205_120000",
        "./results/experiment_20250205_140000",
        # Add more result directories as needed
    ]
    
    # Check which directories exist
    existing_dirs = [d for d in result_directories if Path(d).exists()]
    
    if not existing_dirs:
        print("Error: No valid result directories found!")
        print("Please update the result_directories list with actual paths.")
        return
    
    print(f"Found {len(existing_dirs)} result directories")
    
    # Create comparator
    comparator = ResultsComparator(existing_dirs)
    
    # Generate comparison plots
    comparator.plot_method_comparison(save_path="method_comparison.png")
    
    # Generate summary report
    comparator.generate_summary_report(output_path="comparison_report.txt")
    
    # Create LaTeX table
    comparator.create_latex_comparison_table(output_path="comparison_table.tex")
    
    # Show best methods
    print("\n" + "="*80)
    print("BEST METHODS PER DATASET")
    print("="*80)
    best = comparator.get_best_methods()
    print(best.to_string(index=False))


if __name__ == "__main__":
    main()
