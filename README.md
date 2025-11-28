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

Each dataset has its own config in `config/datasets/`:

```
config/datasets/
├── TXL.yaml
└── Cellpose.yaml
```

Edit these files to customize training parameters (epochs, batch sizes, prune ratios, etc.).

## Adding a Dataset

1. Create `data/<DATASET_NAME>/` with YOLO structure:
   ```
   data/<DATASET_NAME>/
   ├── images/{train,val,test}/
   ├── labels/{train,val,test}/
   └── data.yaml
   ```

2. Create `config/datasets/<DATASET_NAME>.yaml` (see `config/datasets/TXL.yaml` for example):
   ```yaml
   dataset: <DATASET_NAME>
   data: data/<DATASET_NAME>/data.yaml

   model: yolo11n.pt
   epochs_pre: 100
   epochs_post: 50
   batch_sizes: [4, 8]
   prune_ratios: [20, 50]
   device: "0"
   ```

3. Run experiments:
   ```bash
   ./scripts/run_experiment.sh <DATASET_NAME>
   ./scripts/run_experiment.sh <DATASET_NAME> train    # training only
   ```
