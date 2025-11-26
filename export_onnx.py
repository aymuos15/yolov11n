import argparse
import json
import os
from time import perf_counter
from pathlib import Path

import numpy as np
import torch
from ultralytics import YOLO


def export_to_onnx(model_path: str, imgsz: int, dynamic: bool, simplify: bool) -> Path:
    """Export a YOLO .pt to ONNX and return the ONNX file path."""
    model = YOLO(model_path)
    exported = model.export(format="onnx", imgsz=imgsz, dynamic=dynamic, simplify=simplify)
    # Normalize to Path
    if isinstance(exported, dict):
        p = exported.get("model") or exported.get("path")
        onnx_path = Path(p) if p else None
    elif isinstance(exported, (str, Path)):
        onnx_path = Path(exported)
    elif isinstance(exported, (list, tuple)) and exported:
        # pick first *.onnx
        onnx_path = next((Path(x) for x in exported if str(x).endswith(".onnx")), None)
    else:
        onnx_path = None
    # Fallback: look in same folder as model
    if not onnx_path or not onnx_path.exists():
        parent = Path(model_path).parent
        cand = next(parent.glob("*.onnx"), None)
        onnx_path = Path(cand) if cand else None
    if not onnx_path or not onnx_path.exists():
        raise RuntimeError(f"ONNX export failed: no .onnx found for {model_path}")
    return onnx_path


def benchmark_pytorch(model_path: str, imgsz: int, batch: int, iters: int, device: str) -> float:
    """Average PyTorch forward latency in ms for a given batch size."""
    model = YOLO(model_path)
    torch_model = model.model.to(device).eval()
    inp = torch.randn(batch, 3, imgsz, imgsz, device=device)
    # Warmup
    with torch.inference_mode():
        for _ in range(min(10, max(2, iters // 10))):
            _ = torch_model(inp)
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = perf_counter()
    with torch.inference_mode():
        for _ in range(iters):
            _ = torch_model(inp)
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()
    return (perf_counter() - t0) * 1000.0 / iters


def benchmark_onnx(onnx_path: Path, imgsz: int, batch: int, iters: int):
    """Benchmark ONNX model and return (ms_per_image, used_provider)."""
    import onnxruntime as ort
    providers = []
    if 'CUDAExecutionProvider' in ort.get_available_providers():
        providers.append('CUDAExecutionProvider')
    providers.append('CPUExecutionProvider')

    session = ort.InferenceSession(str(onnx_path), providers=providers)
    input_info = session.get_inputs()[0]
    input_name = input_info.name
    input_shape = input_info.shape
    # Determine effective batch: If model fixed batch (int) and differs from requested, use model batch.
    model_batch = input_shape[0] if isinstance(input_shape[0], int) else batch
    effective_batch = model_batch
    if isinstance(input_shape[0], int) and input_shape[0] != batch:
        pass  # silent adjust
    h = imgsz
    w = imgsz
    inp = np.random.randn(effective_batch, 3, h, w).astype(np.float32)
    # Warmup
    for _ in range(min(10, max(2, iters // 10))):
        _ = session.run(None, {input_name: inp})
    t0 = perf_counter()
    for _ in range(iters):
        _ = session.run(None, {input_name: inp})
    elapsed = perf_counter() - t0
    ms_per_image = (elapsed / iters) / effective_batch * 1000.0
    used_provider = session.get_providers()[0]
    return ms_per_image, used_provider


def main():
    parser = argparse.ArgumentParser(description="Export original and pruned YOLO model to ONNX and benchmark")
    parser.add_argument('--orig', required=True, help='Path to original model .pt')
    parser.add_argument('--pruned', required=True, help='Path to pruned model .pt')
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--batch', type=int, default=1)
    parser.add_argument('--iters', type=int, default=100)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--output', default='onnx_benchmark_results.json')
    parser.add_argument('--dynamic', action='store_true', help='Enable dynamic axes in ONNX export')
    parser.add_argument('--simplify', action='store_true', help='Enable ONNX graph simplification')
    args = parser.parse_args()

    print(f"Using device: {args.device}")
    orig_onnx = export_to_onnx(args.orig, args.imgsz, args.dynamic, args.simplify)
    pruned_onnx = export_to_onnx(args.pruned, args.imgsz, args.dynamic, args.simplify)
    orig_torch_ms = benchmark_pytorch(args.orig, args.imgsz, args.batch, args.iters, args.device)
    pruned_torch_ms = benchmark_pytorch(args.pruned, args.imgsz, args.batch, args.iters, args.device)
    try:
        orig_onnx_ms, orig_provider = benchmark_onnx(orig_onnx, args.imgsz, args.batch, args.iters)
    except Exception as e:
        print(f"[WARN] ONNX original benchmark error: {e}; retrying without dynamic/simplify")
        orig_onnx = export_to_onnx(args.orig, args.imgsz, False, False)
        orig_onnx_ms, orig_provider = benchmark_onnx(orig_onnx, args.imgsz, args.batch, args.iters)
    try:
        pruned_onnx_ms, pruned_provider = benchmark_onnx(pruned_onnx, args.imgsz, args.batch, args.iters)
    except Exception as e:
        print(f"[WARN] ONNX pruned benchmark error: {e}; retrying without dynamic/simplify")
        pruned_onnx = export_to_onnx(args.pruned, args.imgsz, False, False)
        pruned_onnx_ms, pruned_provider = benchmark_onnx(pruned_onnx, args.imgsz, args.batch, args.iters)

    results = {
        'config': {
            'image_size': args.imgsz,
            'batch': args.batch,
            'iterations': args.iters,
            'device': args.device,
        },
        'paths': {
            'orig_pt': os.path.abspath(args.orig),
            'pruned_pt': os.path.abspath(args.pruned),
            'orig_onnx': str(orig_onnx),
            'pruned_onnx': str(pruned_onnx),
        },
        'latency_ms': {
            'pytorch_original_avg_ms': orig_torch_ms,
            'pytorch_pruned_avg_ms': pruned_torch_ms,
            'onnx_original_avg_ms': orig_onnx_ms,
            'onnx_pruned_avg_ms': pruned_onnx_ms,
        },
        'providers': {
            'onnx_original_provider': orig_provider,
            'onnx_pruned_provider': pruned_provider,
        },
        'speedup': {
            'pytorch_pruned_vs_orig': orig_torch_ms / pruned_torch_ms if pruned_torch_ms else None,
            'onnx_pruned_vs_orig': orig_onnx_ms / pruned_onnx_ms if pruned_onnx_ms else None,
            'onnx_orig_vs_pytorch_orig': orig_torch_ms / orig_onnx_ms if orig_onnx_ms else None,
            'onnx_pruned_vs_pytorch_pruned': pruned_torch_ms / pruned_onnx_ms if pruned_onnx_ms else None,
        }
    }

    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)

    print("=== Summary ===")
    print(json.dumps(results, indent=2))
    print(f"Saved: {args.output}")


if __name__ == '__main__':
    main()
