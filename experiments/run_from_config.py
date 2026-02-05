
import yaml
from pathlib import Path
from experiments.experiment_runner import MultiDatasetExperimentRunner, ExperimentConfig


def load_config_from_yaml(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_experiment_configs(yaml_config: dict) -> list:
    """Create ExperimentConfig objects from YAML configuration."""
    configs = []
    
    # Get global defaults
    global_settings = yaml_config.get('global_settings', {})
    
    # Process each dataset
    for dataset_config in yaml_config.get('datasets', []):
        # Merge with global settings (dataset settings take precedence)
        merged_config = {
            'n_train_samples': 2000,
            'window_size': 400,
            'n_history': 5,
            'm_barycenter': 3,
            'theta_validation': 0.80,
            'tau_anomaly': 0.75,
            'drift_threshold': 0.5,
            'K_retrain': 5,
        }
        
        # Override with dataset-specific settings
        merged_config.update(dataset_config)
        
        # Create ExperimentConfig
        exp_config = ExperimentConfig(
            dataset_path=merged_config['path'],
            dataset_name=merged_config['name'],
            n_train_samples=merged_config.get('n_train_samples', 2000),
            window_size=merged_config.get('window_size', 400),
            n_history=merged_config.get('n_history', 5),
            m_barycenter=merged_config.get('m_barycenter', 3),
            theta_validation=merged_config.get('theta_validation', 0.80),
            tau_anomaly=merged_config.get('tau_anomaly', 0.75),
            drift_threshold=merged_config.get('drift_threshold', 0.5),
            K_retrain=merged_config.get('K_retrain', 5),
        )
        
        configs.append(exp_config)
    
    return configs


def get_enabled_methods(yaml_config: dict) -> list:
    """Get list of enabled methods from configuration."""
    method_settings = yaml_config.get('method_settings', {})
    global_methods = yaml_config.get('global_settings', {}).get('methods', [])
    
    # Filter enabled methods
    enabled_methods = []
    for method in global_methods:
        method_config = method_settings.get(method, {})
        if method_config.get('enabled', True):  # Default to enabled
            enabled_methods.append(method)
    
    return enabled_methods if enabled_methods else None


def run_from_config(config_path: str = "config.yaml"):
    """Run experiments based on YAML configuration file."""
    
    print("="*80)
    print("Loading configuration from:", config_path)
    print("="*80)
    
    # Load YAML configuration
    yaml_config = load_config_from_yaml(config_path)
    
    # Display experiment info
    exp_info = yaml_config.get('experiment', {})
    print(f"\nExperiment: {exp_info.get('name', 'Unnamed')}")
    print(f"Description: {exp_info.get('description', 'No description')}")
    print(f"Author: {exp_info.get('author', 'Unknown')}")
    print(f"Date: {exp_info.get('date', 'Not specified')}")
    
    # Create experiment configurations
    configs = create_experiment_configs(yaml_config)
    print(f"\nConfigured {len(configs)} dataset(s):")
    for config in configs:
        print(f"  - {config.dataset_name}: {config.dataset_path}")
    
    # Get enabled methods
    methods = get_enabled_methods(yaml_config)
    print(f"\nMethods to run: {methods if methods else 'All methods'}")
    
    # Get output directory
    output_dir = yaml_config.get('global_settings', {}).get('output_dir', './results')
    
    # Initialize runner
    print(f"\nOutput directory: {output_dir}")
    runner = MultiDatasetExperimentRunner(output_dir=output_dir)
    
    # Run experiments
    print("\n" + "="*80)
    print("STARTING EXPERIMENTS")
    print("="*80)
    
    runner.run_multiple_datasets(configs, methods=methods)
    
    # Print summary
    runner.print_summary()
    
    # Save results
    exp_dir = runner.save_results()
    
    # Generate plots if enabled
    viz_settings = yaml_config.get('visualization', {})
    if viz_settings.get('generate_plots', True):
        print("\nGenerating visualizations...")
        runner.generate_comparison_plots()
    
    # Export settings
    export_settings = yaml_config.get('export', {})
    print(f"\nResults exported:")
    if export_settings.get('save_json', True):
        print(f"  ✓ JSON: {exp_dir}/detailed_results.json")
    if export_settings.get('save_csv', True):
        print(f"  ✓ CSV: {exp_dir}/summary_results.csv")
    if export_settings.get('save_latex', True):
        # Create LaTeX table
        import pandas as pd
        summary = pd.read_csv(exp_dir / "summary_results.csv")
        summary['AUC'] = summary.apply(
            lambda x: f"{x['Mean_AUC']:.3f} ± {x['Std_AUC']:.3f}", axis=1
        )
        final_table = summary[['Dataset', 'Method', 'AUC', 'Num_Drifts', 'Execution_Time']]
        latex_path = exp_dir / "results_table.tex"
        final_table.to_latex(latex_path, index=False)
        print(f"  ✓ LaTeX: {latex_path}")
    
    print("\n" + "="*80)
    print("EXPERIMENT COMPLETE!")
    print(f"All results saved to: {exp_dir}")
    print("="*80)
    
    return runner, exp_dir


if __name__ == "__main__":
    import sys
    
    # Allow specifying config file as command line argument
    config_file = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    
    try:
        run_from_config(config_file)
    except FileNotFoundError:
        print(f"Error: Configuration file '{config_file}' not found!")
        print(f"Usage: python {sys.argv[0]} [config_file.yaml]")
        sys.exit(1)
    except Exception as e:
        print(f"Error running experiments: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
