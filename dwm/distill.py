"""Distill planner/BC -> student, ONNX export, INT8 quantisation, latency."""
import time

import numpy as np
import torch
from torch import nn

from .models import obs_feat


def distill(student, teacher_fn, dataset_path, epochs=5, lr=1e-3):
    from .data import DrivingDataset
    ds = DrivingDataset(dataset_path, seq=2)
    opt = torch.optim.Adam(student.parameters(), lr); loss_val = 0.0
    for _ in range(epochs):
        for i in range(len(ds)):
            b = ds[i]
            with torch.no_grad():
                target = teacher_fn(b["bev"], b["scan"], b["ego"], b["goal"]) if isinstance(teacher_fn, nn.Module) else b["action"]
            feats = torch.stack([obs_feat({k: b[k][t] for k in ("bev", "scan", "ego", "goal")}) for t in range(2)])
            tgt = target if isinstance(teacher_fn, nn.Module) else b["action"]
            loss = nn.functional.mse_loss(student(feats.float()), tgt.float())
            opt.zero_grad(); loss.backward(); opt.step(); loss_val = float(loss)
    return loss_val


def export_onnx(student, path="artifacts/student.onnx"):
    import os; os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    student.eval()
    dummy = torch.zeros(1, student.net[0].in_features)
    torch.onnx.export(student, dummy, path, input_names=["obs"], output_names=["action"],
                      dynamic_axes={"obs": {0: "batch"}, "action": {0: "batch"}}, opset_version=17)
    return path


def quantize_and_benchmark(onnx_path, n=200):
    """Dynamic INT8 quantisation + CPU latency (p50/p99 ms). Falls back to fp32 session."""
    import onnxruntime as ort
    qpath = onnx_path.replace(".onnx", "_int8.onnx")
    try:
        from onnxruntime.quantization import QuantType, quantize_dynamic
        quantize_dynamic(onnx_path, qpath, weight_type=QuantType.QInt8)
    except Exception:
        qpath = onnx_path
    out = {}
    for name, p in [("fp32", onnx_path), ("int8", qpath)]:
        sess = ort.InferenceSession(p, providers=["CPUExecutionProvider"])
        x = np.zeros((1, sess.get_inputs()[0].shape[1]), np.float32)
        times = []
        for _ in range(n):
            t0 = time.perf_counter(); sess.run(None, {"obs": x}); times.append((time.perf_counter() - t0) * 1e3)
        out[name] = {"p50_ms": float(np.percentile(times, 50)), "p99_ms": float(np.percentile(times, 99)), "path": p}
    return out
