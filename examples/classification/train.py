import os, logging, csv, json, numpy as np, wandb
from tqdm import tqdm
import torch, torch.nn as nn
from torch import distributed as dist
# tensorboard==2.8 expects np.bool8 which is removed in numpy>=2
if not hasattr(np, "bool8"):
    np.bool8 = np.bool_
from torch.utils.tensorboard import SummaryWriter
from openpoints.utils import set_random_seed, save_checkpoint, load_checkpoint, load_checkpoint_inv, resume_checkpoint, setup_logger_dist, \
    cal_model_parm_nums, Wandb
from openpoints.utils import AverageMeter, ConfusionMatrix, get_mious
from openpoints.transforms import build_transforms_from_cfg
from openpoints.optim import build_optimizer_from_cfg
from openpoints.scheduler import build_scheduler_from_cfg
# from openpoints.loss import build_criterion_from_cfg
from openpoints.models import build_model_from_cfg
from examples.classification.dataloader import (
    get_dataloader as get_local_dataloader,
    get_mixed_classification_dataloaders,
    get_modelnet40_off_classification_dataloader,
    get_raw_frames_classification_dataloader,
    get_review_classification_dataloader,
)
try:
    from openpoints.models.layers import furthest_point_sample, fps
except Exception:
    furthest_point_sample = None
    fps = None

def _get_device(rank):
    if torch.cuda.is_available():
        ngpu = max(1, torch.cuda.device_count())
        return torch.device(f"cuda:{rank % ngpu}")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _move_batch_to_device(data, device):
    non_blocking = device.type == "cuda"
    for key in data.keys():
        if hasattr(data[key], "to"):
            data[key] = data[key].to(device, non_blocking=non_blocking)
    return data


def _random_resample_points(points, npoints):
    bsz, ncurr, channels = points.shape
    idx = torch.randint(ncurr, (bsz, npoints), device=points.device)
    return torch.gather(points, 1, idx.unsqueeze(-1).expand(-1, -1, channels))


def _to_training_batch(data):
    # Local dataloader returns (points, label), openpoints loader returns dict.
    if isinstance(data, (tuple, list)) and len(data) == 2:
        points, label = data
        if points.dim() == 4:
            return {
                'views': points,
                'x': points[:, 0],
                'pos': points[:, 0, :, :3].contiguous(),
                'y': label
            }
        return {
            'x': points,
            'pos': points[:, :, :3].contiguous(),
            'y': label
        }
    return data

def _as_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return False


def _classification_augmentation_kwargs(cfg):
    return dict(
        augment_train=_as_bool(cfg.get('augment_train', False)),
        augment_rotation_deg=cfg.get('augment_rotation_deg', 20.0),
        augment_scale=cfg.get('augment_scale', (0.8, 1.2)),
        augment_jitter_sigma=cfg.get('augment_jitter_sigma', 0.01),
        augment_jitter_clip=cfg.get('augment_jitter_clip', 0.05),
        augment_dropout=cfg.get('augment_dropout', (0.2, 0.5)),
        augment_translate=cfg.get('augment_translate', 0.1),
        augment_random_points_ratio=cfg.get('augment_random_points_ratio', 0.0),
        augment_random_points_scale=cfg.get('augment_random_points_scale', 1.0),
        use_normals=_as_bool(cfg.get('use_normals', False)),
        normal_k=cfg.get('normal_k', 16),
        preload_data=_as_bool(cfg.get('preload_data', False)),
        multi_view=_as_bool(cfg.get('multi_view', False)),
        multi_view_axes=cfg.get('multi_view_axes', ('xy', 'xz', 'yz')),
        multi_view_num_points=cfg.get('multi_view_num_points', 512),
        multi_view_bins=cfg.get('multi_view_bins', 256),
        obj_features_include_sensor_dist=_as_bool(cfg.get('obj_features_include_sensor_dist', True)),
    )


def _configure_point_feature_channels(cfg):
    if not _as_bool(cfg.get('use_normals', False)):
        return
    if cfg.model.get('encoder_args', None) is not None:
        cfg.model.encoder_args.in_channels = 6
    cfg.model.in_channels = 6


def _apply_fast_run_overrides(cfg):
    if not _as_bool(cfg.get("fast_run", False)):
        return

    cfg.fast_run = True
    cfg.epochs = int(cfg.get("fast_run_epochs", 1))
    cfg.val_freq = 1
    cfg.print_freq = 1
    cfg.fast_run_train_batches = int(cfg.get("fast_run_train_batches", 4))
    cfg.fast_run_val_batches = int(cfg.get("fast_run_val_batches", 2))
    cfg.fast_run_test_batches = int(cfg.get("fast_run_test_batches", 2))
    logging.warning(
        "FAST RUN enabled: epochs=%d, train_batches=%d, val_batches=%d, test_batches=%d",
        cfg.epochs,
        cfg.fast_run_train_batches,
        cfg.fast_run_val_batches,
        cfg.fast_run_test_batches,
    )


def _fast_run_enabled(cfg):
    return _as_bool(cfg.get("fast_run", False))


def _max_batches_for_split(cfg, split):
    if not _fast_run_enabled(cfg):
        return None
    if split == "train":
        return int(cfg.get("fast_run_train_batches", 4))
    if split == "test":
        return int(cfg.get("fast_run_test_batches", cfg.get("fast_run_val_batches", 2)))
    return int(cfg.get("fast_run_val_batches", 2))


def _wandb_is_active(cfg):
    return cfg.rank == 0 and _as_bool(cfg.wandb.use_wandb) and wandb.run is not None


def _wandb_per_class_metrics(prefix, accs, cfg):
    if accs is None:
        return {}
    class_names = cfg.get('classes', None)
    metrics = {}
    for i, acc in enumerate(accs):
        cname = class_names[i] if class_names is not None and i < len(class_names) else f"class_{i}"
        metrics[f"{prefix}/acc_per_class/{cname}"] = float(acc)
    return metrics


def _log_wandb_epoch_metrics(cfg, epoch, metrics):
    if not _wandb_is_active(cfg):
        return
    wandb.log(metrics)


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def _log_local_epoch_metrics(cfg, epoch, metrics):
    if cfg.rank != 0 or not _as_bool(cfg.get("save_local_metrics", True)):
        return
    metrics_dir = os.path.join(cfg.run_dir, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)
    payload = {"epoch": int(epoch)}
    payload.update(metrics)
    jsonl_path = os.path.join(metrics_dir, "epoch_metrics.jsonl")
    with open(jsonl_path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, default=_json_default, sort_keys=True) + "\n")


def _classification_dataset_format(cfg):
    return str(cfg.get("classification_dataset_format", cfg.get("dataset_format", ""))).lower()


def _raw_frames_classification_mode(cfg):
    mode = str(cfg.get("mode", "")).lower()
    return mode in {"raw_frames", "raw_frames_classification"} or _classification_dataset_format(cfg) == "raw_frames"


def _review_classification_mode(cfg):
    mode = str(cfg.get("mode", "")).lower()
    return mode in {"review", "review_classification"} or _classification_dataset_format(cfg) == "review"


def _mixed_classification_mode(cfg):
    mode = str(cfg.get("mode", "")).lower()
    return mode in {"mixed", "mixed_classification", "review_raw_frames"} or _classification_dataset_format(cfg) in {"mixed", "review_raw_frames"}


def _modelnet40_off_classification_mode(cfg):
    mode = str(cfg.get("mode", "")).lower()
    return mode in {"modelnet40_off", "modelnet40_off_classification"} or _classification_dataset_format(cfg) == "modelnet40_off"


def _randomized_labels_mode(cfg):
    mode = str(cfg.get("mode", "")).lower()
    return mode in {"random_labels", "randomized_labels", "train_random_labels"}


def _randomize_train_labels(cfg):
    return _randomized_labels_mode(cfg) or _as_bool(cfg.get("randomize_train_labels", False))


def _random_label_seed(cfg):
    seed = cfg.get("random_label_seed", None)
    return cfg.seed if seed in {None, ""} else int(seed)


def _random_label_kwargs(cfg, split):
    return {
        "randomize_labels": split == "train" and _randomize_train_labels(cfg),
        "random_label_seed": _random_label_seed(cfg),
        "random_label_mode": cfg.get("random_label_mode", "permute"),
    }


def _configure_randomized_labels_mode(cfg):
    if not _randomize_train_labels(cfg):
        return
    cfg.randomize_train_labels = True
    cfg.random_label_seed = _random_label_seed(cfg)
    cfg.random_label_mode = cfg.get("random_label_mode", "permute")
    logging.warning(
        "Randomized train labels enabled: mode=%s, seed=%s. Validation/test labels stay unchanged.",
        cfg.random_label_mode,
        cfg.random_label_seed,
    )


def _configure_raw_frames_mode(cfg):
    if not _raw_frames_classification_mode(cfg):
        return
    if cfg.get("raw_frames_root", None) and not cfg.get("custom_dataset_root", None):
        cfg.custom_dataset_root = cfg.raw_frames_root
    cfg.classification_dataset_format = "raw_frames"
    cfg.save_local_metrics = _as_bool(cfg.get("save_local_metrics", True))
    cfg.wandb_log_confusion_matrices = _as_bool(cfg.get("wandb_log_confusion_matrices", False))
    cfg.wandb_log_pcd_examples = _as_bool(cfg.get("wandb_log_pcd_examples", False))
    cfg.wandb_vis_max_samples = int(cfg.get("wandb_vis_max_samples", 0))
    cfg.wandb_vis_max_wrong_samples = int(cfg.get("wandb_vis_max_wrong_samples", 0))
    logging.info("Raw-frames classification mode enabled: using only raw_frames PCD samples.")


def _loader_dataset(loader):
    if isinstance(loader, dict):
        return next(iter(loader.values())).dataset
    return loader.dataset


def _set_cfg_classes_from_loader(cfg, loader):
    dataset = _loader_dataset(loader)
    idx_to_class = {v: k for k, v in dataset.class_to_idx.items()}
    cfg.classes = [idx_to_class[i] for i in range(len(idx_to_class))]
    cfg.num_classes = len(cfg.classes)
    if cfg.model.get('cls_args', None) is not None:
        cfg.model.cls_args.num_classes = cfg.num_classes


def _log_loader_lengths(prefix, loader):
    if isinstance(loader, dict):
        for name, sub_loader in loader.items():
            logging.info(f"length of {prefix} {name} dataset: {len(sub_loader.dataset)}")
    else:
        logging.info(f"length of {prefix} dataset: {len(loader.dataset)}")


def _dataset_class_counts(dataset):
    counts = {class_name: 0 for class_name in getattr(dataset, "classes", [])}
    classes = list(getattr(dataset, "classes", []))
    for sample in getattr(dataset, "samples", []):
        if isinstance(sample, dict):
            class_name = sample.get("class_name")
            if class_name is None and "label" in sample and int(sample["label"]) < len(classes):
                class_name = classes[int(sample["label"])]
        elif isinstance(sample, tuple) and len(sample) >= 2 and int(sample[1]) < len(classes):
            class_name = classes[int(sample[1])]
        else:
            class_name = None
        if class_name is not None:
            counts[class_name] = counts.get(class_name, 0) + 1
    return counts


def _log_class_sample_overview(loaders):
    split_counts = {split: _dataset_class_counts(_loader_dataset(loader)) for split, loader in loaders.items()}
    class_names = sorted({name for counts in split_counts.values() for name in counts})
    logging.info("Sample overview before training:")
    logging.info("  class_name | train | val | test | total")
    for class_name in class_names:
        train_count = split_counts.get("train", {}).get(class_name, 0)
        val_count = split_counts.get("val", {}).get(class_name, 0)
        test_count = split_counts.get("test", {}).get(class_name, 0)
        total = train_count + val_count + test_count
        logging.info("  %s | %d | %d | %d | %d", class_name, train_count, val_count, test_count, total)


def _build_loaders(cfg):
    custom_root = cfg.get('custom_dataset_root', None)
    if _mixed_classification_mode(cfg):
        normal_root = cfg.get('normal_dataset_root', cfg.get('review_dataset_root', cfg.get('custom_dataset_root', None)))
        raw_root = cfg.get('raw_frames_root', None)
        if not normal_root or not raw_root:
            raise ValueError("mixed classification mode needs normal_dataset_root/custom_dataset_root and raw_frames_root")
        num_workers = cfg.dataloader.get('num_workers', 4) if cfg.get('dataloader', None) else 4
        val_bs = cfg.get('val_batch_size', cfg.batch_size)
        exclude_classes = cfg.get('exclude_classes', cfg.get('review_exclude_classes', ("reject",)))
        common_kwargs = dict(
            normal_root=normal_root,
            raw_frames_root=raw_root,
            num_points=cfg.num_points,
            num_workers=num_workers,
            normal_start_date=cfg.get('review_start_date', None),
            normal_min_points=cfg.get('review_min_points', 0),
            normal_split_ratios=cfg.get('review_split_ratios', (0.8, 0.1, 0.1)),
            normal_buckets=cfg.get('review_buckets', None),
            normal_source_dir=cfg.get('review_source_dir', 'pred'),
            normal_exclude_classes=exclude_classes,
            raw_frames_start_date=cfg.get('raw_frames_start_date', cfg.get('review_start_date', None)),
            raw_frames_min_points=cfg.get('raw_frames_min_points', cfg.get('review_min_points', 0)),
            raw_frames_split_ratios=cfg.get('raw_frames_split_ratios', cfg.get('review_split_ratios', (0.8, 0.1, 0.1))),
            raw_frames_exclude_classes=cfg.get('raw_frames_exclude_classes', exclude_classes),
            min_points_exempt_classes=cfg.get(
                'review_min_points_exempt_classes',
                ("TLS_VEHICLE_MOTORBIKE", "TLS_VEHICLE_TRAILER"),
            ),
            balanced_max_repeat_per_sample=cfg.get('balanced_max_repeat_per_sample', None),
            **_classification_augmentation_kwargs(cfg),
        )
        train_loader = get_mixed_classification_dataloaders(
            split='train',
            batch_size=cfg.batch_size,
            shuffle=True,
            class_balanced_batches=_as_bool(cfg.get('class_balanced_batches', True)),
            **_random_label_kwargs(cfg, 'train'),
            **common_kwargs,
        )
        val_loader = get_mixed_classification_dataloaders(
            split='val',
            batch_size=val_bs,
            shuffle=False,
            class_balanced_batches=False,
            **common_kwargs,
        )
        test_loader = get_mixed_classification_dataloaders(
            split='test',
            batch_size=val_bs,
            shuffle=False,
            class_balanced_batches=False,
            **common_kwargs,
        )
        _set_cfg_classes_from_loader(cfg, train_loader)
        return train_loader, val_loader, test_loader

    if _raw_frames_classification_mode(cfg):
        raw_root = cfg.get('raw_frames_root', custom_root)
        if not raw_root:
            raise ValueError("raw_frames mode needs raw_frames_root or custom_dataset_root")
        num_workers = cfg.dataloader.get('num_workers', 4) if cfg.get('dataloader', None) else 4
        val_bs = cfg.get('val_batch_size', cfg.batch_size)
        split_ratios = cfg.get('raw_frames_split_ratios', cfg.get('review_split_ratios', (0.8, 0.1, 0.1)))
        raw_exclude_classes = cfg.get('raw_frames_exclude_classes', cfg.get('review_exclude_classes', ("reject",)))
        exclude_classes = list(raw_exclude_classes)
        for class_name in cfg.get('exclude_classes', ()):
            if class_name not in exclude_classes:
                exclude_classes.append(class_name)
        common_kwargs = dict(
            dataset_root=raw_root,
            num_points=cfg.num_points,
            num_workers=num_workers,
            start_date=cfg.get('raw_frames_start_date', cfg.get('review_start_date', None)),
            min_points=cfg.get('raw_frames_min_points', cfg.get('review_min_points', 0)),
            split_ratios=split_ratios,
            exclude_classes=exclude_classes,
            min_points_exempt_classes=cfg.get(
                'raw_frames_min_points_exempt_classes',
                cfg.get('review_min_points_exempt_classes', ("TLS_VEHICLE_MOTORBIKE", "TLS_VEHICLE_TRAILER")),
            ),
            class_balanced_batches=_as_bool(cfg.get('class_balanced_batches', True)),
            balanced_max_repeat_per_sample=cfg.get('balanced_max_repeat_per_sample', None),
            forced_classes=cfg.get('raw_frames_classes', None),
            frame_selection=cfg.get('raw_frames_frame_selection', cfg.get('frame_selection', 'all')),
            object_multi_view=_as_bool(cfg.get('raw_frames_object_multi_view', False)),
            max_views=cfg.get('raw_frames_max_views', 5),
            view_selection=cfg.get('raw_frames_view_selection', 'uniform'),
            pose_metadata_root=cfg.get('raw_frames_pose_root', None),
            pose_required=_as_bool(cfg.get('raw_frames_pose_required', False)),
            shared_view_normalization=_as_bool(
                cfg.get('raw_frames_shared_view_normalization', False)
            ),
            return_shared_geometry_views=_as_bool(
                cfg.get('raw_frames_return_shared_geometry_views', False)
            ),
            **_classification_augmentation_kwargs(cfg),
        )
        train_loader = get_raw_frames_classification_dataloader(
            split='train', batch_size=cfg.batch_size, shuffle=True,
            **_random_label_kwargs(cfg, 'train'), **common_kwargs)
        val_loader = get_raw_frames_classification_dataloader(
            split='val', batch_size=val_bs, shuffle=False, **common_kwargs)
        test_loader = get_raw_frames_classification_dataloader(
            split='test', batch_size=val_bs, shuffle=False, **common_kwargs)

        idx_to_class = {v: k for k, v in val_loader.dataset.class_to_idx.items()}
        cfg.classes = [idx_to_class[i] for i in range(len(idx_to_class))]
        cfg.num_classes = len(cfg.classes)
        if cfg.model.get('cls_args', None) is not None:
            cfg.model.cls_args.num_classes = cfg.num_classes
        return train_loader, val_loader, test_loader

    if _modelnet40_off_classification_mode(cfg):
        modelnet_root = cfg.get('modelnet40_root', custom_root)
        if not modelnet_root:
            raise ValueError("modelnet40_off mode needs modelnet40_root or custom_dataset_root")
        num_workers = cfg.dataloader.get('num_workers', 4) if cfg.get('dataloader', None) else 4
        val_bs = cfg.get('val_batch_size', cfg.batch_size)
        common_kwargs = dict(
            dataset_root=modelnet_root,
            num_points=cfg.num_points,
            num_workers=num_workers,
            class_balanced_batches=_as_bool(cfg.get('class_balanced_batches', False)),
            balanced_max_repeat_per_sample=cfg.get('balanced_max_repeat_per_sample', None),
            **_classification_augmentation_kwargs(cfg),
        )
        train_loader = get_modelnet40_off_classification_dataloader(
            split='train', batch_size=cfg.batch_size, shuffle=True, **common_kwargs)
        val_loader = get_modelnet40_off_classification_dataloader(
            split='val', batch_size=val_bs, shuffle=False, **common_kwargs)
        test_loader = get_modelnet40_off_classification_dataloader(
            split='test', batch_size=val_bs, shuffle=False, **common_kwargs)
        _set_cfg_classes_from_loader(cfg, train_loader)
        return train_loader, val_loader, test_loader

    if _review_classification_mode(cfg):
        review_root = cfg.get('review_dataset_root', custom_root)
        if not review_root:
            raise ValueError("review classification mode needs review_dataset_root or custom_dataset_root")
        num_workers = cfg.dataloader.get('num_workers', 4) if cfg.get('dataloader', None) else 4
        val_bs = cfg.get('val_batch_size', cfg.batch_size)
        common_kwargs = dict(
            dataset_root=review_root,
            num_points=cfg.num_points,
            num_workers=num_workers,
            start_date=cfg.get('review_start_date', None),
            min_points=cfg.get('review_min_points', 0),
            split_ratios=cfg.get('review_split_ratios', (0.8, 0.1, 0.1)),
            buckets=cfg.get('review_buckets', None),
            source_dir=cfg.get('review_source_dir', 'pred'),
            exclude_classes=cfg.get('exclude_classes', cfg.get('review_exclude_classes', ("reject",))),
            min_points_exempt_classes=cfg.get(
                'review_min_points_exempt_classes',
                ("TLS_VEHICLE_MOTORBIKE", "TLS_VEHICLE_TRAILER"),
            ),
            class_balanced_batches=_as_bool(cfg.get('class_balanced_batches', True)),
            balanced_max_repeat_per_sample=cfg.get('balanced_max_repeat_per_sample', None),
            class_rebalance_mode=cfg.get('class_rebalance_mode', None),
            rebalance_max_repeat_per_sample=cfg.get('rebalance_max_repeat_per_sample', None),
            max_samples_per_class=cfg.get('max_samples_per_class', None),
            **_classification_augmentation_kwargs(cfg),
        )
        train_loader = get_review_classification_dataloader(
            split='train', batch_size=cfg.batch_size, shuffle=True,
            **_random_label_kwargs(cfg, 'train'), **common_kwargs)
        val_loader = get_review_classification_dataloader(
            split='val', batch_size=val_bs, shuffle=False, **common_kwargs)
        test_loader = get_review_classification_dataloader(
            split='test', batch_size=val_bs, shuffle=False, **common_kwargs)

        idx_to_class = {v: k for k, v in val_loader.dataset.class_to_idx.items()}
        cfg.classes = [idx_to_class[i] for i in range(len(idx_to_class))]
        cfg.num_classes = len(cfg.classes)
        if cfg.model.get('cls_args', None) is not None:
            cfg.model.cls_args.num_classes = cfg.num_classes
        return train_loader, val_loader, test_loader

    if not custom_root:
        if _randomize_train_labels(cfg):
            raise ValueError(
                "randomize_train_labels/mode=random_labels is only wired for the local "
                "classification dataloaders. Set custom_dataset_root/review/raw_frames data roots."
            )
        if cfg.get('exclude_classes', None):
            raise ValueError(
                "exclude_classes is only supported for the local normal dataset path. "
                "Set custom_dataset_root=/path/to/train-val-test-npy-dataset, or use "
                "pointnext-raw-frames.yaml with raw_frames_exclude_classes for raw_frames."
            )
        from openpoints.dataset import build_dataloader_from_cfg
        val_loader = build_dataloader_from_cfg(cfg.get('val_batch_size', cfg.batch_size),
                                               cfg.dataset,
                                               cfg.dataloader,
                                               datatransforms_cfg=cfg.datatransforms,
                                               split='val',
                                               distributed=cfg.distributed)
        test_loader = build_dataloader_from_cfg(cfg.get('val_batch_size', cfg.batch_size),
                                                cfg.dataset,
                                                cfg.dataloader,
                                                datatransforms_cfg=cfg.datatransforms,
                                                split='test',
                                                distributed=cfg.distributed)
        train_loader = build_dataloader_from_cfg(cfg.batch_size,
                                                 cfg.dataset,
                                                 cfg.dataloader,
                                                 datatransforms_cfg=cfg.datatransforms,
                                                 split='train',
                                                 distributed=cfg.distributed)
        return train_loader, val_loader, test_loader

    num_workers = cfg.dataloader.get('num_workers', 4) if cfg.get('dataloader', None) else 4
    val_bs = cfg.get('val_batch_size', cfg.batch_size)
    exclude_classes = cfg.get('exclude_classes', cfg.get('normal_exclude_classes', ()))
    balanced = _as_bool(cfg.get('class_balanced_batches', False))
    train_loader = get_local_dataloader(
        custom_root, 'train', cfg.batch_size, cfg.num_points, True, num_workers,
        class_balanced_batches=balanced,
        balanced_max_repeat_per_sample=cfg.get('balanced_max_repeat_per_sample', None),
        exclude_classes=exclude_classes,
        **_classification_augmentation_kwargs(cfg),
        **_random_label_kwargs(cfg, 'train'),
    )
    val_loader = get_local_dataloader(
        custom_root, 'val', val_bs, cfg.num_points, False, num_workers,
        class_balanced_batches=False,
        exclude_classes=exclude_classes,
    )
    try:
        test_loader = get_local_dataloader(
            custom_root, 'test', val_bs, cfg.num_points, False, num_workers,
            class_balanced_batches=False,
            exclude_classes=exclude_classes,
        )
    except FileNotFoundError:
        logging.warning("No 'test' split under custom_dataset_root. Reusing 'val' loader as test loader.")
        test_loader = val_loader

    idx_to_class = {v: k for k, v in val_loader.dataset.class_to_idx.items()}
    cfg.classes = [idx_to_class[i] for i in range(len(idx_to_class))]
    cfg.num_classes = len(cfg.classes)
    if cfg.model.get('cls_args', None) is not None:
        cfg.model.cls_args.num_classes = cfg.num_classes
    return train_loader, val_loader, test_loader


def _log_per_class_acc(writer, prefix, accs, cfg, epoch):
    if writer is None or accs is None:
        return
    class_names = cfg.get('classes', None)
    for i, acc in enumerate(accs):
        cname = class_names[i] if class_names is not None and i < len(class_names) else f"class_{i}"
        writer.add_scalar(f"{prefix}/acc_per_class/{cname}", float(acc), epoch)


def _build_confusion_matrix_figure(mat_norm, class_names, title):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(mat_norm, interpolation='nearest', cmap='Blues')
    ax.set_title(title)
    ax.set_xlabel("Pred")
    ax.set_ylabel("True")

    ncls = len(class_names)
    if ncls <= 40:
        ax.set_xticks(np.arange(ncls))
        ax.set_yticks(np.arange(ncls))
        ax.set_xticklabels(class_names, rotation=90, fontsize=7)
        ax.set_yticklabels(class_names, fontsize=7)

    # Draw normalized value in every cell.
    thresh = float(mat_norm.max()) * 0.6 if mat_norm.size > 0 else 0.0
    for i in range(mat_norm.shape[0]):
        for j in range(mat_norm.shape[1]):
            value = float(mat_norm[i, j])
            txt_color = "white" if value > thresh else "black"
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", color=txt_color, fontsize=6)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig


def _save_confusion_matrix_csv(prefix, cm, cfg, epoch):
    if cfg.rank != 0 or cm is None or not _as_bool(cfg.get("save_local_metrics", True)):
        return None, None, None

    class_names = cfg.get('classes', None) or [f"class_{i}" for i in range(cfg.num_classes)]
    mat = cm.value.detach().cpu().numpy().astype(np.float32)
    row_sum = mat.sum(axis=1, keepdims=True)
    mat_norm = np.divide(mat, np.clip(row_sum, a_min=1.0, a_max=None))

    cm_dir = os.path.join(cfg.run_dir, "confusion_matrix")
    os.makedirs(cm_dir, exist_ok=True)
    raw_path = os.path.join(cm_dir, f"{prefix}_epoch_{epoch}_raw.csv")
    norm_path = os.path.join(cm_dir, f"{prefix}_epoch_{epoch}_normalized.csv")

    with open(raw_path, "w", encoding="utf-8", newline="") as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(["true/pred"] + class_names)
        for i, row in enumerate(mat.tolist()):
            csv_writer.writerow([class_names[i]] + row)

    with open(norm_path, "w", encoding="utf-8", newline="") as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(["true/pred"] + class_names)
        for i, row in enumerate(mat_norm.tolist()):
            csv_writer.writerow([class_names[i]] + [f"{v:.4f}" for v in row])

    return mat_norm, class_names, (raw_path, norm_path)


def _log_confusion_matrix(writer, prefix, cm, cfg, epoch):
    if cm is None:
        return

    saved = _save_confusion_matrix_csv(prefix, cm, cfg, epoch)
    if writer is None or not _as_bool(cfg.get("tensorboard_log_confusion_matrices", True)):
        return

    mat_norm, class_names, _ = saved
    if mat_norm is None:
        return

    try:
        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except Exception as e:
        logging.warning(
            "Skipping confusion matrix plot: matplotlib import failed (%s). "
            "Use numpy<2 with matplotlib==3.5.1 (see requirements.txt).", e
        )
        return

    fig = _build_confusion_matrix_figure(
        mat_norm, class_names, f"{prefix} Confusion Matrix (normalized)"
    )
    writer.add_figure(f"{prefix}/confusion_matrix", fig, global_step=epoch)
    plt.close(fig)


def _class_name(class_names, idx):
    idx = int(idx)
    if class_names is not None and 0 <= idx < len(class_names):
        return str(class_names[idx])
    return f"class_{idx}"


def _pcd_to_wandb_object3d(points):
    # wandb.Object3D expects Nx3 (or Nx6/7) array; we keep xyz only.
    xyz = points.detach().cpu().float()
    if xyz.dim() == 2 and xyz.size(1) > 3:
        xyz = xyz[:, :3]
    return wandb.Object3D(xyz.numpy())



def _base_model(model):
    return model.module if hasattr(model, 'module') else model


def _consolidate_observed_points(points, confidence, voxel_size, min_confidence):
    keep = confidence >= min_confidence
    points = points[keep]
    confidence = confidence[keep]
    if points.numel() == 0:
        return points.reshape(0, 3)
    if voxel_size <= 0:
        return points
    voxel_keys = torch.floor(points / voxel_size).to(torch.int64)
    _, inverse = torch.unique(voxel_keys, dim=0, return_inverse=True)
    representatives = []
    for voxel_idx in range(int(inverse.max().item()) + 1):
        candidates = torch.nonzero(inverse == voxel_idx, as_tuple=False).flatten()
        best = candidates[confidence[candidates].argmax()]
        representatives.append(points[best])
    return torch.stack(representatives, dim=0)


def _write_ascii_ply(path, points):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    xyz = points.detach().cpu().float().numpy()
    if xyz.ndim != 2 or xyz.shape[1] < 3:
        raise ValueError(f"PLY export expects Nx3+ points, got shape {xyz.shape}")
    xyz = xyz[:, :3]
    with open(path, 'w', encoding='utf-8') as handle:
        handle.write('ply\nformat ascii 1.0\n')
        handle.write(f'element vertex {len(xyz)}\n')
        handle.write('property float x\nproperty float y\nproperty float z\nend_header\n')
        np.savetxt(handle, xyz, fmt='%.7f %.7f %.7f')



def _confidence_to_rgb(confidence, kept):
    conf = confidence.detach().cpu().float().clamp(0.0, 1.0).numpy()
    keep = kept.detach().cpu().bool().numpy()
    rgb = np.zeros((conf.shape[0], 3), dtype=np.uint8)
    rgb[:, 0] = np.round(255.0 * (1.0 - conf)).astype(np.uint8)
    rgb[:, 1] = np.round(255.0 * conf).astype(np.uint8)
    rgb[:, 2] = 32
    rgb[~keep] = np.array([255, 32, 32], dtype=np.uint8)
    return rgb


def _write_confidence_ply(path, points, confidence, kept):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    xyz = points.detach().cpu().float().numpy()
    conf = confidence.detach().cpu().float().clamp(0.0, 1.0).numpy()
    keep = kept.detach().cpu().bool().numpy()
    if xyz.ndim != 2 or xyz.shape[1] < 3:
        raise ValueError(f"PLY export expects Nx3+ points, got shape {xyz.shape}")
    if xyz.shape[0] != conf.shape[0] or xyz.shape[0] != keep.shape[0]:
        raise ValueError(
            "points, confidence and kept mask must have equal length, "
            f"got {xyz.shape[0]}, {conf.shape[0]}, {keep.shape[0]}"
        )
    xyz = xyz[:, :3]
    rgb = _confidence_to_rgb(confidence, kept)
    header = [
        "ply",
        "format ascii 1.0",
        f"element vertex {len(xyz)}",
        "property float x",
        "property float y",
        "property float z",
        "property uchar red",
        "property uchar green",
        "property uchar blue",
        "property float confidence",
        "property uchar kept",
        "end_header",
    ]
    with open(path, 'w', encoding='utf-8') as handle:
        handle.write("\n".join(header) + "\n")
        for point, color, value, is_kept in zip(xyz, rgb, conf, keep):
            handle.write(
                f'{point[0]:.7f} {point[1]:.7f} {point[2]:.7f} '
                f'{int(color[0])} {int(color[1])} {int(color[2])} '
                f'{float(value):.6f} {int(is_kept)}\n'
            )



def _write_confidence_histogram(path, confidence, kept, threshold=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    conf = confidence.detach().cpu().float().clamp(0.0, 1.0).numpy()
    keep = kept.detach().cpu().bool().numpy()
    try:
        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except Exception as e:
        logging.warning("Skipping confidence histogram export: matplotlib import failed (%s).", e)
        return

    bins = np.linspace(0.0, 1.0, 21)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(conf, bins=bins, color="#4c9f70", edgecolor="#1f2933", alpha=0.85)
    if conf.size:
        mean = float(conf.mean())
        median = float(np.median(conf))
        ax.axvline(mean, color="#1f4e79", linewidth=2, label=f"mean {mean:.3f}")
        ax.axvline(median, color="#805ad5", linewidth=1.5, linestyle="--", label=f"median {median:.3f}")
    if threshold is not None:
        ax.axvline(float(threshold), color="#d62728", linewidth=2, linestyle=":", label=f"threshold {float(threshold):.3f}")
    rejected = int((~keep).sum())
    total = int(keep.shape[0])
    ax.set_title(f"Point confidence distribution ({rejected}/{total} rejected)")
    ax.set_xlabel("confidence")
    ax.set_ylabel("points")
    ax.set_xlim(0.0, 1.0)
    ax.set_xticks(np.linspace(0.0, 1.0, 11))
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)

def _write_raw_frame_exports(base_path, data, batch_idx, valid_views):
    views = data.get('views')
    if views is None:
        return
    sample_views = views[batch_idx]
    valid_indices = torch.nonzero(valid_views, as_tuple=False).flatten().tolist()
    stem, extension = os.path.splitext(base_path)
    for output_idx, view_idx in enumerate(valid_indices):
        raw_path = f'{stem}_raw_frame_{output_idx:02d}{extension}'
        _write_ascii_ply(raw_path, sample_views[view_idx])


def _collect_fused_cloud_exports(model, data, epoch, cfg, exported_per_class):
    if cfg.rank != 0 or not _as_bool(cfg.get('export_fused_clouds', False)):
        return
    output = getattr(_base_model(model), 'last_output', None)
    if not output or 'transformed_points' not in output:
        return
    max_per_class = int(cfg.get('fused_clouds_per_class', 2))
    voxel_size = float(cfg.get('fusion_voxel_size', 0.02))
    points = output['transformed_points'].detach()
    pre_icp_points = output.get('pre_icp_points')
    if pre_icp_points is not None:
        pre_icp_points = pre_icp_points.detach()
    pre_residual_points = output.get('pre_residual_points')
    if pre_residual_points is not None:
        pre_residual_points = pre_residual_points.detach()
    confidence = output['point_confidence'].detach()
    point_mask = output.get('point_mask')
    if point_mask is not None:
        point_mask = point_mask.detach()
    view_mask = output['view_mask'].detach()
    labels = data['y'].detach().cpu()
    class_names = cfg.get('classes', None)
    for batch_idx, label in enumerate(labels.tolist()):
        class_name = _class_name(class_names, label)
        if exported_per_class.get(class_name, 0) >= max_per_class:
            continue
        valid_views = view_mask[batch_idx]
        all_points = points[batch_idx][valid_views].reshape(-1, 3)
        all_confidence = confidence[batch_idx][valid_views].reshape(-1)
        if point_mask is not None:
            selected = point_mask[batch_idx][valid_views].reshape(-1)
        else:
            selected = torch.ones_like(all_confidence, dtype=torch.bool)
        valid_points = all_points[selected]
        valid_confidence = all_confidence[selected]
        merged = _consolidate_observed_points(
            valid_points, valid_confidence, voxel_size, 0.0
        )
        sample_idx = exported_per_class.get(class_name, 0)
        safe_class_name = class_name.replace('/', '_').replace(' ', '_')
        path = os.path.join(
            cfg.run_dir, 'fused_clouds', f'epoch_{epoch:04d}',
            safe_class_name, f'sample_{sample_idx:02d}.ply'
        )
        _write_ascii_ply(path, merged)
        if (
            pre_icp_points is not None
            and _as_bool(cfg.get('export_pre_icp_clouds', False))
        ):
            pre_icp_all_points = pre_icp_points[batch_idx][
                valid_views
            ].reshape(-1, 3)
            pre_icp_merged = _consolidate_observed_points(
                pre_icp_all_points[selected], valid_confidence,
                voxel_size, 0.0,
            )
            stem, extension = os.path.splitext(path)
            _write_ascii_ply(
                f'{stem}_pre_icp{extension}', pre_icp_merged
            )
        if (
            pre_residual_points is not None
            and _as_bool(cfg.get('export_pre_residual_clouds', False))
        ):
            pre_residual_all_points = pre_residual_points[batch_idx][
                valid_views
            ].reshape(-1, 3)
            pre_residual_merged = _consolidate_observed_points(
                pre_residual_all_points[selected], valid_confidence,
                voxel_size, 0.0,
            )
            stem, extension = os.path.splitext(path)
            _write_ascii_ply(
                f'{stem}_pre_residual{extension}', pre_residual_merged
            )
        if _as_bool(cfg.get('export_fused_confidence_clouds', cfg.get('export_confidence_clouds', False))):
            stem, extension = os.path.splitext(path)
            confidence_path = f'{stem}_confidence{extension}'
            _write_confidence_ply(
                confidence_path, all_points, all_confidence, selected
            )
            if _as_bool(cfg.get('export_fused_confidence_histograms', True)):
                threshold = None
                if cfg.get('model', None) is not None:
                    threshold = cfg.model.get('confidence_threshold', None)
                _write_confidence_histogram(
                    f'{stem}_confidence_hist.png', all_confidence, selected, threshold
                )
        if _as_bool(cfg.get('export_fused_raw_frames', False)):
            _write_raw_frame_exports(path, data, batch_idx, valid_views)
        exported_per_class[class_name] = sample_idx + 1


def _log_wandb_val_pcds(cfg, epoch, split, samples, wrong_samples):
    if cfg.rank != 0 or not cfg.wandb.use_wandb or wandb.run is None or not _as_bool(cfg.get("wandb_log_pcd_examples", True)):
        return
    if len(samples) == 0 and len(wrong_samples) == 0:
        return

    all_table = wandb.Table(columns=["id", "true_label", "pred_label", "correct", "point_cloud"])
    for item in samples:
        all_table.add_data(
            item["id"],
            item["true_label"],
            item["pred_label"],
            item["correct"],
            _pcd_to_wandb_object3d(item["points"]),
        )

    log_payload = {"epoch": epoch, f"{split}/pcd_examples": all_table}
    if len(wrong_samples) > 0:
        wrong_table = wandb.Table(columns=["id", "true_label", "pred_label", "point_cloud"])
        for item in wrong_samples:
            wrong_table.add_data(
                item["id"],
                item["true_label"],
                item["pred_label"],
                _pcd_to_wandb_object3d(item["points"]),
            )
        log_payload[f"{split}/pcd_misclassified"] = wrong_table

    wandb.log(log_payload)


def _log_wandb_confusion_matrix(cfg, epoch, split, cm):
    if cfg.rank != 0 or not cfg.wandb.use_wandb or wandb.run is None or cm is None or not _as_bool(cfg.get("wandb_log_confusion_matrices", True)):
        return
    try:
        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except Exception as e:
        logging.warning("Skipping W&B confusion matrix image: matplotlib import failed (%s)", e)
        return

    class_names = cfg.get('classes', None) or [f"class_{i}" for i in range(cfg.num_classes)]
    mat = cm.value.detach().cpu().numpy().astype(np.float32)
    row_sum = mat.sum(axis=1, keepdims=True)
    mat_norm = np.divide(mat, np.clip(row_sum, a_min=1.0, a_max=None))

    fig = _build_confusion_matrix_figure(
        mat_norm, class_names, f"{split} Confusion Matrix (normalized)"
    )
    wandb.log({
        "epoch": epoch,
        f"{split}/confusion_matrix": wandb.Image(fig),
    })
    plt.close(fig)


def get_features_by_keys(input_features_dim, data):
    if input_features_dim == 3:
        features = data['pos']
    elif input_features_dim == 4:
        features = torch.cat(
            (data['pos'], data['heights']), dim=-1)
        raise NotImplementedError("error")
    return features.transpose(1, 2).contiguous()


def write_to_csv(oa, macc, accs, best_epoch, cfg, write_header=True):
    accs_table = [f'{item:.2f}' for item in accs]
    header = ['method', 'OA', 'mAcc'] + \
        cfg.classes + ['best_epoch', 'log_path', 'wandb link']
    data = [cfg.exp_name, f'{oa:.3f}', f'{macc:.2f}'] + accs_table + [
        str(best_epoch), cfg.run_dir, wandb.run.get_url() if cfg.wandb.use_wandb else '-']
    with open(cfg.csv_path, 'a', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(header)
        writer.writerow(data)
        f.close()


def print_cls_results(oa, macc, accs, epoch, cfg):
    s = f'\nClasses\tAcc\n'
    for name, acc_tmp in zip(cfg.classes, accs):
        s += '{:10}: {:3.2f}%\n'.format(name, acc_tmp)
    s += f'E@{epoch}\tOA: {oa:3.2f}\tmAcc: {macc:3.2f}\n'
    logging.info(s)


def _validate_loader_group(model, loaders, cfg, epoch=None, split='val'):
    cfg._last_eval_results = None
    if not isinstance(loaders, dict):
        return validate(model, loaders, cfg, epoch=epoch, split=split)

    results = {}
    for name, loader in loaders.items():
        metric_split = f"{split}_{name}"
        macc, oa, accs, cm = validate(model, loader, cfg, epoch=epoch, split=metric_split)
        results[name] = {
            'macc': macc,
            'oa': oa,
            'accs': accs,
            'cm': cm,
            'split': metric_split,
        }
        logging.info(f"{metric_split}: OA {oa:.2f}, mAcc {macc:.2f}")

    cfg._last_eval_results = results
    best_metric = str(cfg.get('mixed_best_metric', 'mean_oa')).lower()
    if best_metric in results:
        selected = results[best_metric]
        return selected['macc'], selected['oa'], selected['accs'], selected['cm']

    mean_oa = float(np.mean([item['oa'] for item in results.values()]))
    mean_macc = float(np.mean([item['macc'] for item in results.values()]))
    first = next(iter(results.values()))
    return mean_macc, mean_oa, first['accs'], first['cm']


def _metrics_for_loader_group(prefix, cfg):
    results = getattr(cfg, '_last_eval_results', None)
    if not isinstance(results, dict):
        return {}
    metrics = {}
    for name, item in results.items():
        metric_prefix = f"{prefix}_{name}"
        metrics[f"{metric_prefix}_oa"] = float(item['oa'])
        metrics[f"{metric_prefix}_macc"] = float(item['macc'])
        metrics.update(_wandb_per_class_metrics(metric_prefix, item['accs'], cfg))
    return metrics




def _sample_label_for_auto_weights(sample):
    if isinstance(sample, tuple):
        return int(sample[1])
    return int(sample["label"])


def _loader_epoch_class_counts(loader):
    batch_sampler = getattr(loader, "batch_sampler", None)
    if hasattr(batch_sampler, "epoch_class_counts"):
        return batch_sampler.epoch_class_counts()
    dataset = _loader_dataset(loader)
    counts = {}
    for sample in getattr(dataset, "samples", []):
        label = _sample_label_for_auto_weights(sample)
        counts[label] = counts.get(label, 0) + 1
    return counts


def _configure_auto_class_weights(cfg, train_loader):
    mode = str(cfg.get("auto_class_weights", "")).strip().lower()
    if mode in {"", "false", "0", "none", "off"}:
        return
    if isinstance(train_loader, dict):
        logging.warning("auto_class_weights is skipped for grouped train loaders.")
        return

    counts_by_label = _loader_epoch_class_counts(train_loader)
    if not counts_by_label:
        return
    num_classes = int(cfg.get("num_classes", max(counts_by_label) + 1))
    counts = np.asarray([counts_by_label.get(idx, 0) for idx in range(num_classes)], dtype=np.float64)
    if np.any(counts <= 0):
        missing = [idx for idx, count in enumerate(counts) if count <= 0]
        raise ValueError(f"auto_class_weights cannot handle classes without samples: {missing}")

    strategy = str(cfg.get("auto_class_weight_strategy", "inverse_sqrt")).strip().lower()
    if strategy in {"inverse", "inv"}:
        weights = 1.0 / counts
    elif strategy in {"none", "uniform"}:
        weights = np.ones_like(counts)
    else:
        weights = 1.0 / np.sqrt(counts)
    weights = weights / weights.mean()
    weights = [round(float(item), 6) for item in weights]

    criterion_args = cfg.get("criterion_args", {})
    criterion_args["weight"] = weights
    cfg.criterion_args = criterion_args
    if cfg.model.get("criterion_args", None):
        cfg.model.criterion_args.weight = weights

    class_names = cfg.get("classes", [str(idx) for idx in range(num_classes)])
    summary = ", ".join(
        f"{class_names[idx]}={int(counts[idx])}:{weights[idx]:.6f}"
        for idx in range(num_classes)
    )
    logging.info("Auto class weights from effective epoch counts: %s", summary)

def _freeze_encoder_for_finetune(model, cfg):
    should_freeze = _as_bool(cfg.get("finetune_freeze_encoder", False))
    if str(cfg.get("mode", "")).lower() == "finetune_head":
        should_freeze = True
    if not should_freeze:
        return

    base_model = model.module if hasattr(model, "module") else model
    if not hasattr(base_model, "encoder"):
        raise ValueError("finetune_freeze_encoder=True requires the model to expose an encoder module.")

    for param in base_model.encoder.parameters():
        param.requires_grad = False

    frozen = sum(param.numel() for param in base_model.encoder.parameters())
    trainable = sum(param.numel() for param in base_model.parameters() if param.requires_grad)
    logging.info(
        "Encoder frozen for finetuning: frozen encoder params=%d, trainable params=%d",
        frozen,
        trainable,
    )


def main(gpu, cfg, profile=False):
    device = _get_device(cfg.rank)
    cfg.device = str(device)
    if cfg.distributed and (not torch.cuda.is_available()) and cfg.dist_backend == 'nccl':
        cfg.dist_backend = 'gloo'
    if cfg.distributed:
        if cfg.mp:
            cfg.rank = gpu
        dist.init_process_group(backend=cfg.dist_backend,
                                init_method=cfg.dist_url,
                                world_size=cfg.world_size,
                                rank=cfg.rank)
        dist.barrier()
    # logger
    setup_logger_dist(cfg.log_path, cfg.rank, name=cfg.dataset.common.NAME)
    if cfg.rank == 0 :
        Wandb.launch(cfg, cfg.wandb.use_wandb)
        writer = SummaryWriter(log_dir=cfg.run_dir)
    else:
        writer = None
    set_random_seed(cfg.seed + cfg.rank, deterministic=cfg.deterministic)
    torch.backends.cudnn.enabled = True
    _apply_fast_run_overrides(cfg)
    _configure_randomized_labels_mode(cfg)
    _configure_raw_frames_mode(cfg)
    logging.info(cfg)

    # build dataset before the model so filtered datasets can set num_classes.
    train_loader, val_loader, test_loader = _build_loaders(cfg)
    _log_class_sample_overview({"train": train_loader, "val": val_loader, "test": test_loader})
    _log_loader_lengths("validation", val_loader)
    val_dataset = _loader_dataset(val_loader)
    num_classes = val_dataset.num_classes if hasattr(
        val_dataset, 'num_classes') else None
    if num_classes is None:
        num_classes = cfg.get('num_classes', None)
    num_points = val_dataset.num_points if hasattr(
        val_dataset, 'num_points') else None
    if num_classes is not None:
        cfg.num_classes = num_classes
        if cfg.model.get('cls_args', None) is not None:
            cfg.model.cls_args.num_classes = num_classes
    logging.info(f"number of classes of the dataset: {num_classes}, "
                 f"number of points sampled from dataset: {num_points}, "
                 f"number of points as model input: {cfg.num_points}")
    if cfg.get('classes', None):
        cfg.classes = cfg.classes
    elif hasattr(val_dataset, 'classes'):
        cfg.classes = val_dataset.classes
    elif num_classes is not None:
        cfg.classes = [str(i) for i in range(num_classes)]
    else:
        raise ValueError("Could not infer class names/num_classes from dataset. Set cfg.num_classes or cfg.classes.")
    validate_fn = eval(cfg.get('val_fn', 'validate'))
    _configure_point_feature_channels(cfg)
    _configure_auto_class_weights(cfg, train_loader)

    if not cfg.model.get('criterion_args', False):
        cfg.model.criterion_args = cfg.criterion_args
    model = build_model_from_cfg(cfg.model).to(device)
    model_size = cal_model_parm_nums(model)
    logging.info(model)
    logging.info('Number of params: %.4f M' % (model_size / 1e6))
    # criterion = build_criterion_from_cfg(cfg.criterion_args).cuda()
    if cfg.model.get('in_channels', None) is None:
        cfg.model.in_channels = cfg.model.encoder_args.in_channels
    _freeze_encoder_for_finetune(model, cfg)

    if cfg.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        logging.info('Using Synchronized BatchNorm ...')
    if cfg.distributed:
        if device.type == 'cuda':
            torch.cuda.set_device(device)
            model = nn.parallel.DistributedDataParallel(
                model, device_ids=[device.index], output_device=device.index)
        else:
            model = nn.parallel.DistributedDataParallel(model)
        logging.info('Using Distributed Data parallel ...')

    # optimizer & scheduler
    optimizer = build_optimizer_from_cfg(model, lr=cfg.lr, **cfg.optimizer)
    scheduler = build_scheduler_from_cfg(cfg, optimizer)

    # optionally resume from a checkpoint
    if cfg.pretrained_path is not None:
        if cfg.mode == 'resume':
            resume_checkpoint(cfg, model, optimizer, scheduler,
                              pretrained_path=cfg.pretrained_path)
            macc, oa, accs, cm = _validate_loader_group(model, val_loader, cfg, split='val')
            print_cls_results(oa, macc, accs, cfg.start_epoch, cfg)
        else:
            if cfg.mode == 'test':
                # test mode
                epoch, best_val = load_checkpoint(
                    model, pretrained_path=cfg.pretrained_path)
                macc, oa, accs, cm = _validate_loader_group(model, test_loader, cfg, split='test')
                print_cls_results(oa, macc, accs, epoch, cfg)
                return True
            elif cfg.mode == 'val':
                # validation mode
                epoch, best_val = load_checkpoint(model, cfg.pretrained_path)
                macc, oa, accs, cm = _validate_loader_group(model, val_loader, cfg, split='val')
                print_cls_results(oa, macc, accs, epoch, cfg)
                return True
            elif cfg.mode == 'finetune':
                # finetune the whole model
                logging.info(f'Finetuning from {cfg.pretrained_path}')
                load_checkpoint(
                    model,
                    cfg.pretrained_path,
                    skip_shape_mismatch=_as_bool(cfg.get("load_checkpoint_skip_shape_mismatch", False)),
                )
            elif cfg.mode == 'finetune_head':
                # finetune only the non-encoder layers
                logging.info(f'Finetuning head from {cfg.pretrained_path}')
                load_checkpoint(
                    model,
                    cfg.pretrained_path,
                    skip_shape_mismatch=_as_bool(cfg.get("load_checkpoint_skip_shape_mismatch", False)),
                )
            elif cfg.mode == 'finetune_encoder':
                # finetune the whole model
                logging.info(f'Finetuning from {cfg.pretrained_path}')
                load_checkpoint(model.encoder, cfg.pretrained_path)
            elif cfg.mode == 'finetune_encoder_inv':
                # finetune the whole model
                logging.info(f'Finetuning from {cfg.pretrained_path}')
                load_checkpoint_inv(model.encoder, cfg.pretrained_path)
    elif cfg.get('custom_dataset_root', None):
        logging.info(f"Training from scratch with custom dataset root: {cfg.custom_dataset_root}")
    else:
        logging.info('Training from scratch')
    _log_loader_lengths("training", train_loader)

    # ===> start training
    val_macc, val_oa, val_accs, best_val, macc_when_best, best_epoch = 0., 0., [], 0., 0., 0
    model.zero_grad()
    for epoch in range(cfg.start_epoch, cfg.epochs + 1):
        if cfg.distributed:
            train_loader.sampler.set_epoch(epoch)
        if hasattr(train_loader.dataset, 'epoch'):
            train_loader.dataset.epoch = epoch - 1
        train_loss, train_macc, train_oa, train_accs, train_cm = \
            train_one_epoch(model, train_loader,
                            optimizer, scheduler, epoch, cfg)

        is_best = False
        if epoch % cfg.val_freq == 0:
            val_macc, val_oa, val_accs, val_cm = _validate_loader_group(
                model, val_loader, cfg, epoch=epoch, split='val')
            is_best = val_oa > best_val
            if is_best:
                best_val = val_oa
                macc_when_best = val_macc
                best_epoch = epoch
                logging.info(f'Find a better ckpt @E{epoch}')
                print_cls_results(val_oa, val_macc, val_accs, epoch, cfg)

        lr = optimizer.param_groups[0]['lr']
        logging.info(f'Epoch {epoch} LR {lr:.6f} '
                     f'train_oa {train_oa:.2f}, val_oa {val_oa:.2f}, best val oa {best_val:.2f}')
        wandb_metrics = {
            'epoch': epoch,
            'lr': float(lr),
            'train_loss': float(train_loss),
            'train_oa': float(train_oa),
            'train_macc': float(train_macc),
            'val_oa': float(val_oa),
            'val_macc': float(val_macc),
            'mAcc_when_best': float(macc_when_best),
            'best_val': float(best_val),
        }
        wandb_metrics.update(_wandb_per_class_metrics("train", train_accs, cfg))
        wandb_metrics.update({f'train_loss_{name}': float(value) for name, value in getattr(cfg, '_last_train_aux_losses', {}).items()})
        wandb_metrics.update({
            f'train_{name}': float(value)
            for name, value in getattr(cfg, '_last_train_diagnostics', {}).items()
        })
        if epoch % cfg.val_freq == 0:
            wandb_metrics.update(_wandb_per_class_metrics("val", val_accs, cfg))
            wandb_metrics.update(_metrics_for_loader_group("val", cfg))
        _log_wandb_epoch_metrics(cfg, epoch, wandb_metrics)
        _log_local_epoch_metrics(cfg, epoch, wandb_metrics)
        if writer is not None:
            writer.add_scalar('train_loss', train_loss, epoch)
            writer.add_scalar('train_oa', train_macc, epoch)
            writer.add_scalar('lr', lr, epoch)
            writer.add_scalar('val_oa', val_oa, epoch)
            writer.add_scalar('mAcc_when_best', macc_when_best, epoch)
            writer.add_scalar('best_val', best_val, epoch)
            writer.add_scalar('epoch', epoch, epoch)
            _log_per_class_acc(writer, "train", train_accs, cfg, epoch)
            _log_confusion_matrix(writer, "train", train_cm, cfg, epoch)
            _log_wandb_confusion_matrix(cfg, epoch, "train", train_cm)
            if epoch % cfg.val_freq == 0:
                _log_per_class_acc(writer, "val", val_accs, cfg, epoch)
                _log_confusion_matrix(writer, "val", val_cm, cfg, epoch)
                _log_wandb_confusion_matrix(cfg, epoch, "val", val_cm)
                grouped_results = getattr(cfg, '_last_eval_results', None)
                if isinstance(grouped_results, dict):
                    for name, item in grouped_results.items():
                        metric_split = item['split']
                        _log_per_class_acc(writer, metric_split, item['accs'], cfg, epoch)
                        _log_confusion_matrix(writer, metric_split, item['cm'], cfg, epoch)
                        _log_wandb_confusion_matrix(cfg, epoch, metric_split, item['cm'])

        if cfg.sched_on_epoch:
            scheduler.step(epoch)
        if cfg.rank == 0:
            save_checkpoint(cfg, model, epoch, optimizer, scheduler,
                            additioanl_dict={'best_val': best_val},
                            is_best=is_best
                            )
    # test the last epoch
    test_macc, test_oa, test_accs, test_cm = _validate_loader_group(model, test_loader, cfg, split='test')
    print_cls_results(test_oa, test_macc, test_accs, best_epoch, cfg)
    test_metrics = {
        'epoch': epoch,
        'test_oa': float(test_oa),
        'test_macc': float(test_macc),
    }
    test_metrics.update(_wandb_per_class_metrics('test', test_accs, cfg))
    test_metrics.update(_metrics_for_loader_group('test', cfg))
    _log_wandb_epoch_metrics(cfg, epoch, test_metrics)
    _log_local_epoch_metrics(cfg, epoch, test_metrics)
    if writer is not None:
        writer.add_scalar('test_oa', test_oa, epoch)
        writer.add_scalar('test_macc', test_macc, epoch)
        _log_confusion_matrix(writer, "test_last", test_cm, cfg, epoch)
        _log_wandb_confusion_matrix(cfg, epoch, "test_last", test_cm)

    # test the best validataion model
    best_epoch, _ = load_checkpoint(model, pretrained_path=os.path.join(
        cfg.ckpt_dir, f'{cfg.run_name}_ckpt_best.pth'))
    test_macc, test_oa, test_accs, test_cm = _validate_loader_group(model, test_loader, cfg, split='test')
    best_test_metrics = {
        'epoch': int(best_epoch),
        'best_test_oa': float(test_oa),
        'best_test_macc': float(test_macc),
    }
    best_test_metrics.update(_wandb_per_class_metrics('best_test', test_accs, cfg))
    best_test_metrics.update(_metrics_for_loader_group('best_test', cfg))
    _log_wandb_epoch_metrics(cfg, best_epoch, best_test_metrics)
    _log_local_epoch_metrics(cfg, best_epoch, best_test_metrics)
    if writer is not None:
        writer.add_scalar('test_oa', test_oa, best_epoch)
        writer.add_scalar('test_macc', test_macc, best_epoch)
        _log_confusion_matrix(writer, "test_best", test_cm, cfg, best_epoch)
        _log_wandb_confusion_matrix(cfg, best_epoch, "test_best", test_cm)
    print_cls_results(test_oa, test_macc, test_accs, best_epoch, cfg)

    if writer is not None:
        writer.close()
    if cfg.distributed and dist.is_initialized():
        dist.destroy_process_group()

def train_one_epoch(model, train_loader, optimizer, scheduler, epoch, cfg):
    loss_meter = AverageMeter()
    aux_loss_meters = {
        name: AverageMeter()
        for name in (
            'cls', 'align', 'icp_delta_reg', 'reg', 'overlap_point',
            'residual_correction', 'density', 'boundary', 'consistency',
            'cls_weighted', 'reg_weighted', 'residual_correction_weighted',
        )
    }
    diagnostic_meters = {
        name: AverageMeter()
        for name in (
            'confidence_mean', 'retained_point_ratio',
            'overlap_positive_ratio', 'rotation_deg_mean',
            'translation_norm_mean', 'icp_delta_translation_norm_mean',
            'residual_delta_translation_norm_mean',
            'pose_rotation_loss',
            'pose_translation_loss', 'pose_valid_ratio',
        )
    }
    exported_per_class = {}
    cm = ConfusionMatrix(num_classes=cfg.num_classes)
    npoints = cfg.num_points

    model.train()  # set model to training mode
    max_batches = _max_batches_for_split(cfg, "train")
    total_batches = train_loader.__len__() if max_batches is None else min(train_loader.__len__(), max_batches)
    pbar = tqdm(enumerate(train_loader), total=total_batches)
    num_iter = 0
    device = torch.device(cfg.device)
    for idx, data in pbar:
        data = _to_training_batch(data)
        data = _move_batch_to_device(data, device)
        num_iter += 1
        target = data['y']
        """ bebug
        from openpoints.dataset import vis_points
        vis_points(data['pos'].cpu().numpy()[0])
        """
        if 'views' in data:
            data['views'] = data['views'].contiguous()
            data['pos'] = data['views'][:, 0, :, :3].contiguous()
            data['x'] = data['pos'].transpose(1, 2).contiguous()
        else:
            points = data['x']
            num_curr_pts = points.shape[1]
            if num_curr_pts > npoints:  # point resampling strategy
                if npoints == 1024:
                    point_all = 1200
                elif npoints == 4096:
                    point_all = 4800
                elif npoints == 8192:
                    point_all = 8192
                else:
                    raise NotImplementedError()
                if  points.size(1) < point_all:
                    point_all = points.size(1)
                if device.type == 'cuda' and furthest_point_sample is not None:
                    fps_idx = furthest_point_sample(
                        points[:, :, :3].contiguous(), point_all)
                    fps_idx = fps_idx[:, np.random.choice(
                        point_all, npoints, False)]
                    points = torch.gather(
                        points, 1, fps_idx.unsqueeze(-1).long().expand(-1, -1, points.shape[-1]))
                else:
                    points = _random_resample_points(points, npoints)

            data['pos'] = points[:, :, :3].contiguous()
            data['x'] = points[:, :, :cfg.model.in_channels].transpose(1, 2).contiguous()
        logits, loss = model.get_logits_loss(data, target) if not hasattr(model, 'module') else model.module.get_logits_loss(data, target)
        model_output = getattr(_base_model(model), 'last_output', None)
        if model_output and 'losses' in model_output:
            for loss_name, meter in aux_loss_meters.items():
                meter.update(float(model_output['losses'][loss_name].detach().item()))
            for name, meter in diagnostic_meters.items():
                value = model_output.get('diagnostics', {}).get(name)
                if value is not None:
                    meter.update(float(value.detach().item()))
        _collect_fused_cloud_exports(model, data, epoch, cfg, exported_per_class)
        loss.backward()

        # optimize
        if num_iter == cfg.step_per_update:
            if cfg.get('grad_norm_clip') is not None and cfg.grad_norm_clip > 0.:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), cfg.grad_norm_clip, norm_type=2)
            num_iter = 0
            optimizer.step()
            model.zero_grad()
            if not cfg.sched_on_epoch:
                scheduler.step(epoch)

        # update confusion matrix
        cm.update(logits.argmax(dim=1), target)
        loss_meter.update(loss.item())
        if idx % cfg.print_freq == 0:
            pbar.set_description(f"Train Epoch [{epoch}/{cfg.epochs}] "
                                 f"Loss {loss_meter.val:.3f} Acc {cm.overall_accuray:.2f}")
            if _wandb_is_active(cfg):
                wandb.log({
                    "epoch": epoch,
                    "train/batch": idx,
                    "train/batch_loss": float(loss_meter.val),
                    "train/running_loss": float(loss_meter.avg),
                    "train/running_oa": float(cm.overall_accuray),
                })
        if max_batches is not None and (idx + 1) >= max_batches:
            break
    macc, overallacc, accs = cm.all_acc()
    cfg._last_train_aux_losses = {name: meter.avg for name, meter in aux_loss_meters.items()}
    cfg._last_train_diagnostics = {
        name: meter.avg for name, meter in diagnostic_meters.items()
    }
    return loss_meter.avg, macc, overallacc, accs, cm


@torch.no_grad()
def validate(model, val_loader, cfg, epoch=None, split='val'):
    model.eval()  # set model to eval mode
    cm = ConfusionMatrix(num_classes=cfg.num_classes)
    npoints = cfg.num_points
    device = torch.device(cfg.device)
    class_names = cfg.get('classes', None)
    vis_max_samples = int(cfg.get('wandb_vis_max_samples', 8))
    vis_max_wrong = int(cfg.get('wandb_vis_max_wrong_samples', vis_max_samples))
    vis_freq = int(cfg.get('wandb_vis_val_freq', 1))
    should_log_wandb_vis = (
        epoch is not None
        and split == 'val'
        and cfg.rank == 0
        and cfg.wandb.use_wandb
        and _as_bool(cfg.get('wandb_log_pcd_examples', True))
        and (epoch % vis_freq == 0)
        and (vis_max_samples > 0 or vis_max_wrong > 0)
    )
    pcd_samples, wrong_pcd_samples = [], []

    max_batches = _max_batches_for_split(cfg, split)
    total_batches = val_loader.__len__() if max_batches is None else min(val_loader.__len__(), max_batches)
    pbar = tqdm(enumerate(val_loader), total=total_batches)
    for idx, data in pbar:
        data = _to_training_batch(data)
        data = _move_batch_to_device(data, device)
        target = data['y']
        if 'views' in data:
            data['views'] = data['views'].contiguous()
            data['pos'] = data['views'][:, 0, :, :3].contiguous()
            data['x'] = data['pos'].transpose(1, 2).contiguous()
        else:
            points = data['x']
            points = points[:, :npoints]
            data['pos'] = points[:, :, :3].contiguous()
            data['x'] = points[:, :, :cfg.model.in_channels].transpose(1, 2).contiguous()
        logits = model(data)
        pred = logits.argmax(dim=1)
        cm.update(pred, target)

        if should_log_wandb_vis and (len(pcd_samples) < vis_max_samples or len(wrong_pcd_samples) < vis_max_wrong):
            pos_cpu = data['pos'].detach().cpu()
            pred_cpu = pred.detach().cpu()
            target_cpu = target.detach().cpu()
            bsz = target_cpu.shape[0]
            for b in range(bsz):
                gt_idx = int(target_cpu[b].item())
                pred_idx = int(pred_cpu[b].item())
                is_correct = gt_idx == pred_idx
                sample = {
                    "id": f"{split}_e{epoch}_i{idx}_b{b}",
                    "true_label": _class_name(class_names, gt_idx),
                    "pred_label": _class_name(class_names, pred_idx),
                    "correct": is_correct,
                    "points": pos_cpu[b],
                }
                if len(pcd_samples) < vis_max_samples:
                    pcd_samples.append(sample)
                if (not is_correct) and len(wrong_pcd_samples) < vis_max_wrong:
                    wrong_pcd_samples.append(sample)
                if len(pcd_samples) >= vis_max_samples and len(wrong_pcd_samples) >= vis_max_wrong:
                    break
        if max_batches is not None and (idx + 1) >= max_batches:
            break

    tp, count = cm.tp, cm.count
    if cfg.distributed:
        dist.all_reduce(tp), dist.all_reduce(count)
    macc, overallacc, accs = cm.cal_acc(tp, count)
    if should_log_wandb_vis:
        _log_wandb_val_pcds(cfg, epoch, split, pcd_samples, wrong_pcd_samples)
    return macc, overallacc, accs, cm
