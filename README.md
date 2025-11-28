# YOLOv11n Pruning Experiments

## Quick Start

```bash
pip install -e .
```

## Run Experiments

```bash
# Full pipeline (train -> export -> compare)
./scripts/run_experiment.sh

# Individual steps
./scripts/run_experiment.sh train
./scripts/run_experiment.sh export
./scripts/run_experiment.sh compare
```

## Configuration

Edit `config/experiment.yaml` to customize training parameters.

## Adding a Dataset

1. Create `data/<DATASET_NAME>/` with YOLO structure:
   ```
   data/<DATASET_NAME>/
   ├── images/{train,val,test}/
   ├── labels/{train,val,test}/
   └── data.yaml
   ```

2. Update `config/experiment.yaml`:
   ```yaml
   dataset: <DATASET_NAME>
   data: data/<DATASET_NAME>/data.yaml
   ```
