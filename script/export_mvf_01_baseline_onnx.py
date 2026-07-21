#!/usr/bin/env python3
"""Export the seed-8240 5-view raw-frame classifier to ONNX.

The exported model has two inputs:
  views:     float32 or float16 [batch, 5, 512, 3]  (XYZ points per view)
  view_mask: bool    [batch, 5]          (True for present views)

Only the classification path is exported.  The training-time geometry/ICP
branch does not affect the classifier logits and is intentionally omitted.

Run from the PointNeXt repository:
  python script/export_mvf_01_baseline_onnx.py
  python script/export_mvf_01_baseline_onnx.py --fp16 --verify
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


POINTNEXT_ROOT = Path(__file__).resolve().parents[1]
if str(POINTNEXT_ROOT) not in sys.path:
    sys.path.insert(0, str(POINTNEXT_ROOT))

from openpoints.models import build_model_from_cfg  # noqa: E402
from openpoints.utils import EasyConfig, load_checkpoint  # noqa: E402


RUN_DIR = POINTNEXT_ROOT / (
    "log/mvf_10class_ablation_shared_encoder_gt_pose/"
    "mvf_10class_ablation_shared_encoder_gt_pose-raw_frames_classification-"
    "01_baseline_5views_512_geometry_gradients-ngpus1-seed8240-"
    "20260711-213551-m2474PJB27sDKzaaskrhxf"
)
DEFAULT_CFG = RUN_DIR / "01_baseline_5views_512_geometry_gradients.yaml"
DEFAULT_CHECKPOINT = RUN_DIR / "checkpoint" / (RUN_DIR.name + "_ckpt_best.pth")
DEFAULT_OUTPUT = RUN_DIR / "onnx" / "mvf_01_baseline_5views_512.onnx"


class ClassificationONNXWrapper(torch.nn.Module):
    """Expose the dict-based PointNeXt model as tensor-only ONNX inputs."""

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, views: torch.Tensor, view_mask: torch.Tensor) -> torch.Tensor:
        batch_size, num_views, num_points, _ = views.shape
        flat_views = views.reshape(batch_size * num_views, num_points, 3)
        view_data = {"pos": flat_views, "x": flat_views.transpose(1, 2).contiguous()}
        view_logits = self.model.prediction(self.model.encoder.forward_cls_feat(view_data))
        view_logits = view_logits.reshape(batch_size, num_views, -1)
        weights = view_mask.to(dtype=view_logits.dtype).unsqueeze(-1)
        return (view_logits * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)


def onnx_furthest_point_sample(xyz: torch.Tensor, npoint: int) -> torch.Tensor:
    """FPS expressed with regular tensor operators instead of the CUDA extension."""
    batch_size, num_points, _ = xyz.shape
    npoint = min(npoint, num_points)
    centroids = torch.zeros((batch_size, npoint), dtype=torch.long, device=xyz.device)
    distance = torch.full((batch_size, num_points), torch.finfo(xyz.dtype).max, dtype=xyz.dtype, device=xyz.device)
    farthest = torch.zeros((batch_size,), dtype=torch.long, device=xyz.device)
    batch_indices = torch.arange(batch_size, dtype=torch.long, device=xyz.device)
    for index in range(npoint):
        centroids[:, index] = farthest
        centroid = xyz[batch_indices, farthest].unsqueeze(1)
        squared_distance = ((xyz - centroid) ** 2).sum(dim=-1)
        distance = torch.minimum(distance, squared_distance)
        farthest = distance.argmax(dim=1)
    return centroids


def onnx_grouping_operation(features: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    batch_size, channels, _ = features.shape
    _, num_query, num_neighbors = indices.shape
    flattened = indices.long().reshape(batch_size, -1)
    expanded = flattened.unsqueeze(1).expand(-1, channels, -1)
    return features.gather(2, expanded).reshape(batch_size, channels, num_query, num_neighbors)


def onnx_ball_query(radius: float, nsample: int, xyz: torch.Tensor, new_xyz: torch.Tensor) -> torch.Tensor:
    """Ball query using ONNX-supported matrix arithmetic and TopK."""
    query_squared = (new_xyz ** 2).sum(dim=-1, keepdim=True)
    support_squared = (xyz ** 2).sum(dim=-1).unsqueeze(1)
    distances = query_squared + support_squared - 2.0 * torch.matmul(new_xyz, xyz.transpose(1, 2))
    valid = distances <= radius * radius
    masked_distances = torch.where(valid, distances, torch.full_like(distances, float("inf")))
    indices = torch.topk(masked_distances, k=nsample, dim=-1, largest=False).indices
    nearest = distances.argmin(dim=-1, keepdim=True).expand(-1, -1, nsample)
    invalid = ~torch.gather(valid, -1, indices)
    return torch.where(invalid, nearest, indices).int()


def enable_onnx_safe_pointops(model: torch.nn.Module) -> None:
    """Replace PointNet++ CUDA autograd functions with traceable tensor ops."""
    import openpoints.models.layers.group as group

    group.ball_query = onnx_ball_query
    group.grouping_operation = onnx_grouping_operation
    for module in model.modules():
        if hasattr(module, "sample_fn"):
            module.sample_fn = onnx_furthest_point_sample


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export MultiViewLateFusionCls to ONNX.")
    parser.add_argument("--cfg", type=Path, default=DEFAULT_CFG)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--fp16", action="store_true", help="Exportiert Gewichte und views-Eingabe in Float16.")
    parser.add_argument("--verify", action="store_true", help="Verify with onnxruntime when installed.")
    return parser.parse_args()


def load_model(cfg_path: Path, checkpoint: Path, fp16: bool = False) -> torch.nn.Module:
    cfg = EasyConfig()
    cfg.load(str(cfg_path.resolve()), recursive=True)
    model = build_model_from_cfg(cfg.model).cpu()
    load_checkpoint(model, str(checkpoint.resolve()))

    # Geometry is used only for auxiliary losses/cloud export.  Its ICP loop is
    # not ONNX-friendly and cannot change object_logits (see cls_base.py).
    model.geometry_model = None
    enable_onnx_safe_pointops(model)
    if fp16:
        model.half()
    model.eval()
    return model


def verify_with_onnxruntime(
    output: Path, wrapper: torch.nn.Module, views: torch.Tensor, view_mask: torch.Tensor
) -> None:
    try:
        import onnxruntime as ort
    except ImportError:
        print("onnxruntime nicht installiert: ONNX-Datei wurde erzeugt, Verifikation übersprungen.")
        return

    with torch.inference_mode():
        torch_logits = wrapper(views, view_mask).numpy()
    session = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"])
    onnx_logits = session.run(
        ["logits"], {"views": views.numpy(), "view_mask": view_mask.numpy()}
    )[0]
    max_abs_error = float(abs(torch_logits - onnx_logits).max())
    rtol, atol = (1e-2, 1e-2) if views.dtype == torch.float16 else (1e-3, 1e-4)
    if not torch.allclose(torch.from_numpy(torch_logits), torch.from_numpy(onnx_logits), rtol=rtol, atol=atol):
        raise RuntimeError(f"ONNX-Verifikation fehlgeschlagen (max. absoluter Fehler: {max_abs_error:.6g}).")
    print(f"ONNXRuntime-Verifikation OK (max. absoluter Fehler: {max_abs_error:.6g}).")


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size muss mindestens 1 sein.")
    if not args.cfg.is_file():
        raise FileNotFoundError(f"Config nicht gefunden: {args.cfg}")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint nicht gefunden: {args.checkpoint}")

    if args.fp16 and args.output == DEFAULT_OUTPUT:
        args.output = DEFAULT_OUTPUT.with_name(DEFAULT_OUTPUT.stem + "_fp16.onnx")
    model = load_model(args.cfg, args.checkpoint, fp16=args.fp16)
    wrapper = ClassificationONNXWrapper(model).eval()
    views = torch.zeros(args.batch_size, 5, 512, 3, dtype=torch.float16 if args.fp16 else torch.float32)
    view_mask = torch.ones(args.batch_size, 5, dtype=torch.bool)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with torch.inference_mode():
        torch.onnx.export(
            wrapper,
            (views, view_mask),
            str(args.output),
            input_names=["views", "view_mask"],
            output_names=["logits"],
            dynamic_axes={"views": {0: "batch"}, "view_mask": {0: "batch"}, "logits": {0: "batch"}},
            opset_version=args.opset,
            do_constant_folding=True,
        )

    print(f"ONNX exportiert: {args.output}")
    precision = "float16" if args.fp16 else "float32"
    print(f"Eingaben: views={precision}[B,5,512,3], view_mask=bool[B,5]; Ausgabe: logits={precision}[B,10]")
    if args.verify:
        verify_with_onnxruntime(args.output, wrapper, views, view_mask)


if __name__ == "__main__":
    main()
