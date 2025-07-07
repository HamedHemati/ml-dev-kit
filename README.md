# Minimalistic ML Research DevKit



## 📁 Project Structure

```
ml-dev-kit/
├── configs/               # Configuration files
│   ├── algorithm/         # Algorithm-specific configs
│   ├── dataset/           # Dataset-specific configs
│   └── generic.yaml       # Main configuration file
├── data/                  # Data storage directory
├── notebooks/             # Jupyter notebooks
├── outputs/               # Experiment outputs 
├── scripts/               # Utility scripts
├── src/                   # Source code: core package
│   ├── algorithms/        # Training algorithms
│   ├── datasets/          # Dataset implementations
│   ├── models/            # Models architectures
│   └── utils/             # Utility functions
├── train.py               # Main training script
└── requirements.txt       # Python dependencies
```

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd ml-dev-kit
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Optional: Set up Weights & Biases** (for experiment tracking):
   ```bash
   wandb login
   ```

## 🚀 Quick Start

### Basic Training

Run a single training experiment:

```bash
python train.py
```

### Enable Logging

Enable Weights & Biases logging:

```bash
python train.py log=True
```

### Hyperparameter Sweeping

Run multiple experiments with different hyperparameters:

```bash
python train.py --multirun lr=0.1,0.01,0.001 batch_size=16,32
```

### Override Configuration

Override any configuration parameter:

```bash
python train.py num_epochs=10 lr=0.001 device=cuda
```
