# OnlineMROTAD


##  Project Structure

```
ONLINEMROTAD/
├── datasets/               # Dataset storage
│   ├── cao2025/           # CAO 2025 dataset
│   └── files/             # Additional data files
│       └── glass_shake_sudden_2.csv

├── results/               # Experimental results
│   ├── experiment_20260205_145501/
│   ├── experiment_20260205_152112/
│   └── SOTA/
├── source/                # Main source code
│   ├── __init__.py
│   ├── mrot.py           # Main MROT implementation
│   ├── offline.py        # Offline processing
│   ├── onlineMROTauc_eval.py
│   └── utils.py          # Utility functions
├── notebooks/             # Jupyter notebooks for analysis
│   ├── datasets.ipynb
│   ├── experiments.ipynb
│   ├── mrot_copy.ipynb
│   ├── online.ipynb
├── .gitignore
├── README.md
└── datasets.dvc          # DVC configuration for data versioning

```

##  Features

- **Multi-target Tracking**: Advanced algorithms for tracking multiple targets simultaneously
- **Online and Offline Modes**: Support for both real-time and batch processing
- **Anomaly Detection**: Integrated anomaly detection capabilities
- **Experimental Framework**: Comprehensive experiment management and comparison tools

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


Results are automatically saved to the `results/` directory with timestamps.

##  Notebooks

Interactive Jupyter notebooks are available for exploration and analysis:

- `online.ipynb`: Online processing demonstrations
- `experiments.ipynb`: Experiment visualization and analysis
- `datasets.ipynb`: Dataset exploration
- `Segment.ipynb`: Segmentation analysis


To launch notebooks:
```bash
jupyter notebook
```

##  Core Components

### Source Modules

- **mrot.py**: Main multi-target tracking implementation
- **offline.py**: Batch processing capabilities
- **wasserstein.py**: Wasserstein distance calculations
- **onlineMROTrate_eval.py**: Online Anomalyrate-based evaluation
- **utils.py**: Shared utility functions

##  Results

Experimental results are organized by timestamp in the `results/` directory:
- Each experiment creates a new directory with format: `experiment_YYYYMMDD_HHMMSS/`
- SOTA results are stored separately in `results/SOTA/`

##  Configuration

Experiments are configured via `sota_` files and. Key parameters include:

- Dataset selection
- Algorithm parameters
- Evaluation metrics
- Output settings
