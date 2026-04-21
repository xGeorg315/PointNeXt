import logging
import os

import numpy as np
import torch
import torch.nn as nn
import wandb
from torch import distributed as dist
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# tensorboard==2.8 expects np.bool8 which is removed in numpy>=2
if not hasattr(np, "bool8"):
    np.bool8 = np.bool_

from examples.classification.dataloader import get_completion_dataloader
from examples.classification.train import (
    _class_name,
    _get_device,
    _log_confusion_matrix,
    _log_per_class_acc,
    _max_batches_for_split,
    _pcd_to_wandb_object3d,
    _apply_fast_run_overrides,
)
from openpoints.models import build_model_from_cfg
from openpoints.optim import build_optimizer_from_cfg
from openpoints.scheduler import build_scheduler_from_cfg
from openpoints.utils import (
    AverageMeter,
    ConfusionMatrix,
    Wandb,
    cal_model_parm_nums,
    load_checkpoint,
    resume_checkpoint,
    save_checkpoint,
    set_random_seed,
    setup_logger_dist,
)


def _move_batch_to_device(data, device):
    non_blocking = device.type == "cuda"
    for key, value in list(data.items()):
        if torch.is_tensor(value):
            data[key] = value.to(device, non_blocking=non_blocking)
    return data


def _prepare_completion_batch(data, cfg):
    partial = data["partial"].float().contiguous()
    data["pos"] = partial[:, :, :3].contiguous()
    data["x"] = partial[:, :, :cfg.model.in_channels].transpose(1, 2).contiguous()
    data["complete"] = data["complete"].float().contiguous()
    return data


def _build_loaders(cfg):
    custom_root = cfg.get("custom_dataset_root", None)
    if not custom_root:
        raise ValueError("Completion training requires cfg.custom_dataset_root.")

    num_workers = cfg.dataloader.get("num_workers", 4) if cfg.get("dataloader", None) else 4
    val_bs = cfg.get("val_batch_size", cfg.batch_size)
    num_complete = int(cfg.get("num_complete", 2048))
    target_strategy = cfg.get("completion_target_strategy", "aggregate")
    partial_choice = cfg.get("completion_partial_choice", "random_far")
    min_dist_delta = float(cfg.get("completion_min_dist_delta", 0.0))

    train_loader = get_completion_dataloader(
        custom_root,
        "train",
        cfg.batch_size,
        cfg.num_points,
        num_complete,
        True,
        num_workers,
        target_strategy=target_strategy,
        partial_choice=partial_choice,
        completion_min_dist_delta=min_dist_delta,
    )
    val_loader = get_completion_dataloader(
        custom_root,
        "val",
        val_bs,
        cfg.num_points,
        num_complete,
        False,
        num_workers,
        target_strategy=target_strategy,
        partial_choice="farthest",
        completion_min_dist_delta=min_dist_delta,
    )
    try:
        test_loader = get_completion_dataloader(
            custom_root,
            "test",
            val_bs,
            cfg.num_points,
            num_complete,
            False,
            num_workers,
            target_strategy=target_strategy,
            partial_choice="farthest",
            completion_min_dist_delta=min_dist_delta,
        )
    except FileNotFoundError:
        logging.warning("No 'test' split under custom_dataset_root. Reusing 'val' loader as test loader.")
        test_loader = val_loader

    _warn_object_id_overlap(train_loader.dataset, val_loader.dataset, "train", "val")
    if test_loader is not val_loader:
        _warn_object_id_overlap(train_loader.dataset, test_loader.dataset, "train", "test")
        _warn_object_id_overlap(val_loader.dataset, test_loader.dataset, "val", "test")

    cfg.classes = list(val_loader.dataset.classes)
    cfg.num_classes = len(cfg.classes)
    return train_loader, val_loader, test_loader


def _object_keys(dataset):
    keys = set()
    for sample in dataset.samples:
        if isinstance(sample, dict):
            keys.add((sample["class_name"], sample["object_id"]))
    return keys


def _warn_object_id_overlap(left_dataset, right_dataset, left_name, right_name):
    overlap = _object_keys(left_dataset) & _object_keys(right_dataset)
    if overlap:
        logging.warning(
            "Found %d object-id overlaps between %s and %s splits. "
            "This may leak completion targets across evaluation.",
            len(overlap),
            left_name,
            right_name,
        )


def _model_module(model):
    return model.module if hasattr(model, "module") else model


@torch.no_grad()
def _per_sample_chamfer(model, pred, target):
    criterion = _model_module(model).recon_criterion
    vals = []
    for idx in range(pred.shape[0]):
        vals.append(float(criterion(pred[idx:idx + 1], target[idx:idx + 1]).item()))
    return vals


def _update_meters(meters, losses, batch_size):
    for name, meter in meters.items():
        meter.update(float(losses[name].item()), batch_size)


def _meter_dict():
    return {
        "loss": AverageMeter(),
        "fine_loss": AverageMeter(),
        "coarse_loss": AverageMeter(),
        "cls_loss": AverageMeter(),
    }


def _log_completion_examples(cfg, epoch, split, examples):
    if cfg.rank != 0 or not cfg.wandb.use_wandb or wandb.run is None or not examples:
        return

    table = wandb.Table(
        columns=["id", "true_label", "pred_label", "partial", "target", "prediction"]
    )
    for item in examples:
        table.add_data(
            item["id"],
            item["true_label"],
            item["pred_label"],
            _pcd_to_wandb_object3d(item["partial"]),
            _pcd_to_wandb_object3d(item["target"]),
            _pcd_to_wandb_object3d(item["prediction"]),
        )
    wandb.log({"epoch": epoch, f"{split}/completion_examples": table})


def _log_per_class_chamfer(writer, prefix, class_chamfer, cfg, epoch):
    if writer is None:
        return
    for idx, value in enumerate(class_chamfer):
        cname = _class_name(cfg.classes, idx)
        writer.add_scalar(f"{prefix}/chamfer_per_class/{cname}", float(value), epoch)


def _print_results(prefix, metrics, cfg, epoch):
    parts = [
        f"{prefix} @E{epoch}",
        f"fine {metrics['fine_loss']:.5f}",
        f"coarse {metrics['coarse_loss']:.5f}",
        f"loss {metrics['loss']:.5f}",
        f"OA {metrics['oa']:.2f}",
        f"mAcc {metrics['macc']:.2f}",
    ]
    logging.info(" | ".join(parts))
    if metrics.get("class_chamfer") is not None:
        lines = ["Per-class Chamfer-L1"]
        for name, value in zip(cfg.classes, metrics["class_chamfer"]):
            lines.append(f"{name:20}: {value:.5f}")
        logging.info("\n".join(lines))


def main(gpu, cfg, profile=False):
    device = _get_device(cfg.rank)
    cfg.device = str(device)
    if cfg.distributed and (not torch.cuda.is_available()) and cfg.dist_backend == "nccl":
        cfg.dist_backend = "gloo"
    if cfg.distributed:
        if cfg.mp:
            cfg.rank = gpu
        dist.init_process_group(
            backend=cfg.dist_backend,
            init_method=cfg.dist_url,
            world_size=cfg.world_size,
            rank=cfg.rank,
        )
        dist.barrier()

    setup_logger_dist(cfg.log_path, cfg.rank, name=cfg.dataset.common.NAME)
    if cfg.rank == 0:
        Wandb.launch(cfg, cfg.wandb.use_wandb)
        writer = SummaryWriter(log_dir=cfg.run_dir)
    else:
        writer = None

    set_random_seed(cfg.seed + cfg.rank, deterministic=cfg.deterministic)
    torch.backends.cudnn.enabled = True
    _apply_fast_run_overrides(cfg)
    logging.info(cfg)

    train_loader, val_loader, test_loader = _build_loaders(cfg)
    if not cfg.model.get("criterion_args", False):
        cfg.model.criterion_args = cfg.criterion_args
    if cfg.model.get("in_channels", None) is None:
        cfg.model.in_channels = cfg.model.encoder_args.in_channels

    model = build_model_from_cfg(cfg.model).to(device)
    logging.info(model)
    logging.info("Number of params: %.4f M" % (cal_model_parm_nums(model) / 1e6))

    if cfg.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        logging.info("Using Synchronized BatchNorm ...")
    if cfg.distributed:
        if device.type == "cuda":
            torch.cuda.set_device(device)
            model = nn.parallel.DistributedDataParallel(
                model, device_ids=[device.index], output_device=device.index
            )
        else:
            model = nn.parallel.DistributedDataParallel(model)
        logging.info("Using Distributed Data parallel ...")

    optimizer = build_optimizer_from_cfg(model, lr=cfg.lr, **cfg.optimizer)
    scheduler = build_scheduler_from_cfg(cfg, optimizer)

    if cfg.pretrained_path is not None:
        module = _model_module(model)
        if cfg.mode == "resume":
            resume_checkpoint(cfg, model, optimizer, scheduler, pretrained_path=cfg.pretrained_path)
        elif cfg.mode in {"test", "val"}:
            epoch, _ = load_checkpoint(model, pretrained_path=cfg.pretrained_path)
            loader = test_loader if cfg.mode == "test" else val_loader
            metrics = validate(model, loader, cfg, epoch=epoch, split=cfg.mode)
            _print_results(cfg.mode, metrics, cfg, epoch)
            return True
        elif cfg.mode == "finetune_encoder":
            load_checkpoint(module.encoder, cfg.pretrained_path)
        elif cfg.mode == "finetune":
            load_checkpoint(model, cfg.pretrained_path)
    else:
        logging.info("Training completion/classification model from scratch")

    logging.info("length of training dataset: %d", len(train_loader.dataset))
    logging.info("length of validation dataset: %d", len(val_loader.dataset))
    logging.info("number of classes: %d, partial points: %d, complete points: %d",
                 cfg.num_classes, cfg.num_points, cfg.get("num_complete", 2048))

    best_val = float("inf")
    best_epoch = cfg.start_epoch - 1
    for epoch in range(cfg.start_epoch, cfg.epochs + 1):
        if cfg.distributed:
            train_loader.sampler.set_epoch(epoch)

        train_metrics = train_one_epoch(model, train_loader, optimizer, scheduler, epoch, cfg)

        val_metrics = None
        is_best = False
        if epoch % cfg.val_freq == 0:
            val_metrics = validate(model, val_loader, cfg, epoch=epoch, split="val")
            is_best = val_metrics["fine_loss"] < best_val
            if is_best:
                best_val = val_metrics["fine_loss"]
                best_epoch = epoch
                logging.info("Found a better completion ckpt @E%d", epoch)

        lr = optimizer.param_groups[0]["lr"]
        logging.info(
            "Epoch %d LR %.6f train_fine %.5f train_oa %.2f best_val_fine %.5f",
            epoch,
            lr,
            train_metrics["fine_loss"],
            train_metrics["oa"],
            best_val,
        )
        _print_results("train", train_metrics, cfg, epoch)
        if val_metrics is not None:
            _print_results("val", val_metrics, cfg, epoch)

        if writer is not None:
            writer.add_scalar("epoch", epoch, epoch)
            writer.add_scalar("lr", lr, epoch)
            for key in ["loss", "fine_loss", "coarse_loss", "cls_loss", "oa", "macc"]:
                writer.add_scalar(f"train/{key}", train_metrics[key], epoch)
                if val_metrics is not None:
                    writer.add_scalar(f"val/{key}", val_metrics[key], epoch)
            _log_per_class_acc(writer, "train", train_metrics["accs"], cfg, epoch)
            _log_confusion_matrix(writer, "train", train_metrics["cm"], cfg, epoch)
            if train_metrics.get("class_chamfer") is not None:
                _log_per_class_chamfer(writer, "train", train_metrics["class_chamfer"], cfg, epoch)
            if val_metrics is not None:
                _log_per_class_acc(writer, "val", val_metrics["accs"], cfg, epoch)
                _log_confusion_matrix(writer, "val", val_metrics["cm"], cfg, epoch)
                _log_per_class_chamfer(writer, "val", val_metrics["class_chamfer"], cfg, epoch)

        if cfg.sched_on_epoch:
            scheduler.step(epoch)
        if cfg.rank == 0:
            save_checkpoint(
                cfg,
                model,
                epoch,
                optimizer,
                scheduler,
                additioanl_dict={"best_val": best_val},
                is_best=is_best,
            )

    best_path = os.path.join(cfg.ckpt_dir, f"{cfg.run_name}_ckpt_best.pth")
    if os.path.exists(best_path):
        best_epoch, _ = load_checkpoint(model, pretrained_path=best_path)
    test_metrics = validate(model, test_loader, cfg, epoch=best_epoch, split="test")
    _print_results("test", test_metrics, cfg, best_epoch)

    if writer is not None:
        for key in ["loss", "fine_loss", "coarse_loss", "cls_loss", "oa", "macc"]:
            writer.add_scalar(f"test/{key}", test_metrics[key], best_epoch)
        writer.close()
    if cfg.distributed and dist.is_initialized():
        dist.destroy_process_group()


def train_one_epoch(model, train_loader, optimizer, scheduler, epoch, cfg):
    meters = _meter_dict()
    cm = ConfusionMatrix(num_classes=cfg.num_classes)
    device = torch.device(cfg.device)
    model.train()
    model.zero_grad()

    max_batches = _max_batches_for_split(cfg, "train")
    total_batches = train_loader.__len__() if max_batches is None else min(train_loader.__len__(), max_batches)
    pbar = tqdm(enumerate(train_loader), total=total_batches)
    num_iter = 0
    for idx, data in pbar:
        data = _prepare_completion_batch(_move_batch_to_device(data, device), cfg)
        batch_size = data["y"].shape[0]
        outputs = model(data)
        losses = _model_module(model).get_loss(outputs, data)
        losses["loss"].backward()
        num_iter += 1

        if num_iter >= cfg.step_per_update:
            if cfg.get("grad_norm_clip") is not None and cfg.grad_norm_clip > 0.:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_norm_clip, norm_type=2)
            optimizer.step()
            model.zero_grad()
            num_iter = 0
            if not cfg.sched_on_epoch:
                scheduler.step(epoch)

        _update_meters(meters, losses, batch_size)
        cm.update(outputs["logits"].argmax(dim=1), data["y"])

        if idx % cfg.print_freq == 0:
            pbar.set_description(
                f"Train Epoch [{epoch}/{cfg.epochs}] "
                f"Fine {meters['fine_loss'].val:.4f} CE {meters['cls_loss'].val:.4f}"
            )
        if max_batches is not None and (idx + 1) >= max_batches:
            break

    if num_iter > 0:
        if cfg.get("grad_norm_clip") is not None and cfg.grad_norm_clip > 0.:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_norm_clip, norm_type=2)
        optimizer.step()
        model.zero_grad()
        if not cfg.sched_on_epoch:
            scheduler.step(epoch)

    macc, oa, accs = cm.all_acc()
    return {
        "loss": meters["loss"].avg,
        "fine_loss": meters["fine_loss"].avg,
        "coarse_loss": meters["coarse_loss"].avg,
        "cls_loss": meters["cls_loss"].avg,
        "macc": macc,
        "oa": oa,
        "accs": accs,
        "cm": cm,
        "class_chamfer": None,
    }


@torch.no_grad()
def validate(model, val_loader, cfg, epoch=None, split="val"):
    meters = _meter_dict()
    cm = ConfusionMatrix(num_classes=cfg.num_classes)
    class_chamfer_sum = np.zeros(cfg.num_classes, dtype=np.float64)
    class_chamfer_count = np.zeros(cfg.num_classes, dtype=np.int64)
    examples = []

    device = torch.device(cfg.device)
    model.eval()
    vis_max = int(cfg.get("wandb_vis_max_samples", 4))
    max_batches = _max_batches_for_split(cfg, split)
    total_batches = val_loader.__len__() if max_batches is None else min(val_loader.__len__(), max_batches)
    pbar = tqdm(enumerate(val_loader), total=total_batches)

    for idx, data in pbar:
        data = _prepare_completion_batch(_move_batch_to_device(data, device), cfg)
        batch_size = data["y"].shape[0]
        outputs = model(data)
        losses = _model_module(model).get_loss(outputs, data)
        _update_meters(meters, losses, batch_size)

        pred_cls = outputs["logits"].argmax(dim=1)
        cm.update(pred_cls, data["y"])

        per_sample = _per_sample_chamfer(model, outputs["pred_complete"], data["complete"])
        labels_cpu = data["y"].detach().cpu().numpy()
        for cls_idx, chamfer in zip(labels_cpu, per_sample):
            class_chamfer_sum[int(cls_idx)] += chamfer
            class_chamfer_count[int(cls_idx)] += 1

        if len(examples) < vis_max:
            pred_cpu = pred_cls.detach().cpu()
            for batch_idx in range(batch_size):
                if len(examples) >= vis_max:
                    break
                gt_idx = int(labels_cpu[batch_idx])
                pred_idx = int(pred_cpu[batch_idx].item())
                object_id = data.get("object_id", ["unknown"] * batch_size)[batch_idx]
                examples.append(
                    {
                        "id": str(object_id),
                        "true_label": _class_name(cfg.classes, gt_idx),
                        "pred_label": _class_name(cfg.classes, pred_idx),
                        "partial": data["partial"][batch_idx].detach().cpu(),
                        "target": data["complete"][batch_idx].detach().cpu(),
                        "prediction": outputs["pred_complete"][batch_idx].detach().cpu(),
                    }
                )

        if idx % cfg.print_freq == 0:
            pbar.set_description(
                f"{split} Fine {meters['fine_loss'].val:.4f} CE {meters['cls_loss'].val:.4f}"
            )
        if max_batches is not None and (idx + 1) >= max_batches:
            break

    macc, oa, accs = cm.all_acc()
    class_chamfer = np.divide(
        class_chamfer_sum,
        np.clip(class_chamfer_count, a_min=1, a_max=None),
    )
    _log_completion_examples(cfg, epoch or 0, split, examples)
    return {
        "loss": meters["loss"].avg,
        "fine_loss": meters["fine_loss"].avg,
        "coarse_loss": meters["coarse_loss"].avg,
        "cls_loss": meters["cls_loss"].avg,
        "macc": macc,
        "oa": oa,
        "accs": accs,
        "cm": cm,
        "class_chamfer": class_chamfer,
    }
