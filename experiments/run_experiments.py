

from experiments.experiment_runner import (
    MultiDatasetExperimentRunner, 
    ExperimentConfig
)

def main():

    configs = [
        # Dataset S1
        ExperimentConfig(
            dataset_path="./datasets/cao2025/S1.csv",
            dataset_name="S1",
            n_train_samples=2000,
            window_size=400,
            n_history=5,
            m_barycenter=3,
            theta_validation=0.80,
            tau_anomaly=0.75,
            K_retrain=5,
            drift_threshold=0.5
        ),
        
        # Dataset S2 (example - adjust path as needed)
        # ExperimentConfig(
        #     dataset_path="./datasets/cao2025/S2.csv",
        #     dataset_name="S2",
        #     n_train_samples=2000,
        #     window_size=400,
        #     n_history=5,
        #     m_barycenter=3,
        #     theta_validation=0.80,
        #     tau_anomaly=0.75,
        #     K_retrain=5,
        #     drift_threshold=0.5
        # ),
        
        # Add more datasets as needed...
    ]
    
    # =========================================================================
    # 2. CHOOSE METHODS: Select which methods to run
    # =========================================================================
    
    # Available methods:
    # - 'tumbling': Online MROTAD with tumbling window
    # - 'sliding': Online MROTAD with sliding window
    # - 'wasserstein': Online Wasserstein (Structured)
    # - 'domain_adaptation': Online Domain Adaptation
    
    methods_to_run = [
        'tumbling',
        'sliding',
        'wasserstein',
        'domain_adaptation'
    ]
   
    # Initialize the experiment runner
    runner = MultiDatasetExperimentRunner(output_dir="./results")
    
    # Run experiments on all configured datasets
    runner.run_multiple_datasets(configs, methods=methods_to_run)
    
    # =========================================================================
    # 4. SAVE AND VISUALIZE RESULTS
    # =========================================================================
    
    # Print summary to console
    runner.print_summary()
    
    # Save results to files (JSON + CSV)
    exp_dir = runner.save_results()
    
    # Generate comparison plots
    runner.generate_comparison_plots()
   
if __name__ == "__main__":
    main()
