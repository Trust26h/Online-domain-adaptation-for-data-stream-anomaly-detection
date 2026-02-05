# OnlineMROTAD


##  Project Structure

```
ONLINEMROTAD/
├── datasets/               # Dataset storage
│   ├── cao2025/           # CAO 2025 dataset
│   └── files/             # Additional data files
│       └── glass_shake_sudden_2.csv
├── experiments/           # Experiment configurations and runners
│   ├── compare_results.py
│   ├── config.yaml
│   ├── experiment_runner.py
│   ├── run_experiments.py
│   ├── run_from_config.py
│   └── setup.sh
├── results/               # Experimental results
│   ├── experiment_20260205_145501/
│   ├── experiment_20260205_152112/
│   └── SOTA/
├── source/                # Main source code
│   ├── __init__.py
│   ├── mrot.py           # Main MROT implementation
│   ├── offline.py        # Offline processing
│   ├── online.py         # Online processing
│   ├── onlineMROTauc_eval.py
│   ├── onlineMROTwasserstein.py
│   └── utils.py          # Utility functions
│   └── wasserstein.py    # Wasserstein distance calculations
├── notebooks/             # Jupyter notebooks for analysis
│   ├── datasets.ipynb
│   ├── experiments.ipynb
│   ├── mrot_copy.ipynb
│   ├── online.ipynb
│   └── Segment.ipynb
├── .gitignore
├── README.md
└── datasets.dvc          # DVC configuration for data versioning

```

##  Features

- **Multi-target Tracking**: Advanced algorithms for tracking multiple targets simultaneously
- **Online and Offline Modes**: Support for both real-time and batch processing
- **Anomaly Detection**: Integrated anomaly detection capabilities
- **Wasserstein Distance**: Implementation of Wasserstein metrics for evaluation
- **Experimental Framework**: Comprehensive experiment management and comparison tools
- **Configurable Pipeline**: YAML-based configuration system

##  Getting Started

### Prerequisites

- Python 3.8+
- pip or conda for package management

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd ONLINEMROTAD
```

2. Run the setup script:
```bash
run experiments.ipynb
```

3. (Optional) If using DVC for dataset management:
```bash
dvc pull
```

##  Datasets

The project supports multiple datasets stored in the `datasets/` directory:

- **CAO2025**: Located in `datasets/cao2025/`
- **Custom datasets**: Additional CSV files in `datasets/files/`

Example dataset: `glass_shake_sudden_2.csv`

##  Running Experiments

### Using Configuration Files

```bash
cd experiments
python run_from_config.py
```

Edit `experiments/config.yaml` to customize experimental parameters.

### Running Individual Experiments

```bash
python experiment_runner.py
```

### Comparing Results

```bash
python compare_results.py
```

Results are automatically saved to the `results/` directory with timestamps.

##  Notebooks

Interactive Jupyter notebooks are available for exploration and analysis:

- `online.ipynb`: Online processing demonstrations
- `experiments.ipynb`: Experiment visualization and analysis
- `datasets.ipynb`: Dataset exploration
- `Segment.ipynb`: Segmentation analysis
- `mrot_copy.ipynb`: MROT algorithm walkthrough

To launch notebooks:
```bash
jupyter notebook
```

##  Core Components

### Source Modules

- **mrot.py**: Main multi-target tracking implementation
- **online.py**: Real-time processing pipeline
- **offline.py**: Batch processing capabilities
- **wasserstein.py**: Wasserstein distance calculations
- **onlineMROTwasserstein.py**: Online Wasserstein-based evaluation
- **onlineMROTauc_eval.py**: AUC-based evaluation metrics
- **utils.py**: Shared utility functions

##  Results

Experimental results are organized by timestamp in the `results/` directory:
- Each experiment creates a new directory with format: `experiment_YYYYMMDD_HHMMSS/`
- SOTA results are stored separately in `results/SOTA/`

##  Configuration

Experiments are configured via `experiments/config.yaml`. Key parameters include:

- Dataset selection
- Algorithm parameters
- Evaluation metrics
- Output settings
