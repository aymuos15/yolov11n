# YOLOv11n Pruning Experiments

## Quick Start

```bash
pip install -e .
```

## Run Experiments

```bash
# Full pipeline (train -> export -> compare)
./scripts/run_experiment.sh TXL
./scripts/run_experiment.sh Cellpose

# Individual steps
./scripts/run_experiment.sh TXL train
./scripts/run_experiment.sh TXL export
./scripts/run_experiment.sh TXL compare
```

## Configuration

The project uses a hierarchical configuration system with inheritance:

### **Configuration Structure**

```
config/
├── default.yaml              # Global defaults and model configurations
└── datasets/
    ├── TXL.yaml           # Dataset-specific overrides
    └── Cellpose.yaml       # Dataset-specific overrides
```

### **Model Sizes**

All YOLOv11 model sizes are supported:

- **n** (nano) - Fastest, smallest model
- **s** (small) - Balanced speed and accuracy  
- **m** (medium) - Good accuracy, moderate speed
- **l** (large) - High accuracy, slower
- **x** (extra large) - Highest accuracy, slowest

### **Configuration Inheritance**

Dataset configs inherit from `config/default.yaml`:

```yaml
# config/default.yaml (base defaults)
model_size: n              # Default model size
model_family: yolo11        # Model family
epochs_pre: 100            # Training epochs
batch_sizes: [4, 8]        # Batch sizes to test
prune_ratios: [20, 50]     # Pruning percentages
device: "0"                # GPU device
```

### **Usage Examples**

#### **Command Line (override defaults)**
```bash
# Use different model size
python -m src.run_train --config config/datasets/TXL.yaml --model-size s

# Use large model
python -m src.run_train --config config/datasets/TXL.yaml --model-size l
```

#### **Dataset Config Override**
```yaml
# config/datasets/TXL.yaml
dataset: TXL
data: data/TXL/data.yaml

# Override model size for this dataset
model_size: m              # Use medium model instead of nano

# Other parameters inherited from default.yaml
```

#### **Available Parameters**
- `model_size`: n, s, m, l, x (model size)
- `model_family`: yolo11 (model family)
- `epochs_pre`: Pre-pruning training epochs
- `epochs_post`: Post-pruning training epochs  
- `batch_sizes`: List of batch sizes to test
- `prune_ratios`: List of pruning percentages
- `device`: GPU device ID or "cpu"

## Adding a Dataset

1. Create `data/<DATASET_NAME>/` with YOLO structure:
   ```
   data/<DATASET_NAME>/
   ├── images/{train,val,test}/
   ├── labels/{train,val,test}/
   └── data.yaml
   ```

2. Create `config/datasets/<DATASET_NAME>.yaml` (inherits from `config/default.yaml`):
   ```yaml
   # Dataset Configuration (inherits from config/default.yaml)
   dataset: <DATASET_NAME>
   data: data/<DATASET_NAME>/data.yaml

   # Optional: Override model size for this dataset
   # model_size: s              # Use small model instead of nano
   
   # Optional: Override training parameters
   # epochs_pre: 50             # Custom pre-pruning epochs
   # batch_sizes: [2, 4]        # Custom batch sizes
   ```

   **Note**: All parameters inherit from `config/default.yaml` unless overridden here.

3. Run experiments:
   ```bash
   # Use default model size (n)
   python -m src.run_train --config config/datasets/<DATASET_NAME>.yaml
   
   # Use specific model size
   python -m src.run_train --config config/datasets/<DATASET_NAME>.yaml --model-size m
   ```
