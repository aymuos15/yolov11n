"""
Export trained models to ONNX format.

Usage:
    python -m src.run_export --dataset TXL           # Export all models for dataset
    python -m src.run_export --model runs/.../best.pt  # Export single model
"""

import argparse
import re
from pathlib import Path

from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).parent.parent


def discover_models(runs_dir: Path, dataset: str) -> list[Path]:
    """Find all trained models for a dataset."""
    pattern_baseline = re.compile(rf"{re.escape(dataset)}_batch(\d+)_baseline$")
    pattern_pruned = re.compile(rf"{re.escape(dataset)}_batch(\d+)_pruned(\d+)$")

    models = []
    for child in runs_dir.iterdir():
        if not child.is_dir():
            continue

        weights = child / "weights" / "best.pt"
        if not weights.exists():
            continue

        name = child.name
        if pattern_baseline.match(name) or pattern_pruned.match(name):
            models.append(weights)

    return sorted(models)


def export_to_onnx(model_path: Path, imgsz: int = 640, force: bool = False) -> Path:
    """Export model to ONNX with dynamic batch size. Returns path to exported file."""
    onnx_path = model_path.with_suffix(".onnx")

    if onnx_path.exists() and not force:
        print(f"  Already exists: {onnx_path.name}")
        return onnx_path

    model = YOLO(str(model_path))
    exported = model.export(format="onnx", imgsz=imgsz, dynamic=True)

    if isinstance(exported, (str, Path)):
        result = Path(exported)
    elif isinstance(exported, dict):
        result = Path(exported.get("model") or exported.get("path", ""))
    else:
        result = next(model_path.parent.glob("*.onnx"), None)

    if not result or not result.exists():
        raise RuntimeError(f"ONNX export failed for {model_path}")

    print(f"  Exported: {result.name}")
    return result


def main():
    parser = argparse.ArgumentParser(description="Export models to ONNX")
    parser.add_argument("--dataset", type=str, help="Dataset name - exports all models for dataset")
    parser.add_argument("--model", type=Path, help="Single model path to export")
    parser.add_argument("--runs-dir", type=Path, default=PROJECT_ROOT / "runs")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--force", action="store_true", help="Re-export even if ONNX exists")
    args = parser.parse_args()

    if not args.dataset and not args.model:
        parser.error("Either --dataset or --model is required")

    print(f"=== ONNX Export ===")

    if args.model:
        # Single model export
        if not args.model.exists():
            print(f"Model not found: {args.model}")
            return
        print(f"Model: {args.model}")
        export_to_onnx(args.model, args.imgsz, args.force)
    else:
        # Dataset export
        print(f"Dataset: {args.dataset}")
        print(f"Discovering models in {args.runs_dir}...")

        models = discover_models(args.runs_dir, args.dataset)
        if not models:
            print("No models found.")
            return

        print(f"Found {len(models)} models")

        for model_path in models:
            print(f"\n[{model_path.parent.parent.name}]")
            export_to_onnx(model_path, args.imgsz, args.force)

    print(f"\n=== Export Done ===")


if __name__ == "__main__":
    main()
