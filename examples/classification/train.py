import os, logging, csv, numpy as np, wandb
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
from examples.classification.dataloader import get_dataloader as get_local_dataloader
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


def _build_loaders(cfg):
    custom_root = cfg.get('custom_dataset_root', None)
    if not custom_root:
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
    train_loader = get_local_dataloader(custom_root, 'train', cfg.batch_size, cfg.num_points, True, num_workers)
    val_loader = get_local_dataloader(custom_root, 'val', val_bs, cfg.num_points, False, num_workers)
    try:
        test_loader = get_local_dataloader(custom_root, 'test', val_bs, cfg.num_points, False, num_workers)
    except FileNotFoundError:
        logging.warning("No 'test' split under custom_dataset_root. Reusing 'val' loader as test loader.")
        test_loader = val_loader

    idx_to_class = {v: k for k, v in val_loader.dataset.class_to_idx.items()}
    cfg.classes = [idx_to_class[i] for i in range(len(idx_to_class))]
    cfg.num_classes = len(cfg.classes)
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


def _log_confusion_matrix(writer, prefix, cm, cfg, epoch):
    if writer is None or cm is None:
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

    class_names = cfg.get('classes', None) or [f"class_{i}" for i in range(cfg.num_classes)]
    mat = cm.value.detach().cpu().numpy().astype(np.float32)
    row_sum = mat.sum(axis=1, keepdims=True)
    mat_norm = np.divide(mat, np.clip(row_sum, a_min=1.0, a_max=None))

    fig = _build_confusion_matrix_figure(
        mat_norm, class_names, f"{prefix} Confusion Matrix (normalized)"
    )
    writer.add_figure(f"{prefix}/confusion_matrix", fig, global_step=epoch)
    plt.close(fig)

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


def _log_wandb_val_pcds(cfg, epoch, split, samples, wrong_samples):
    if cfg.rank != 0 or not cfg.wandb.use_wandb or wandb.run is None:
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

    log_payload = {f"{split}/pcd_examples": all_table}
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

    wandb.log(log_payload, step=epoch)


def _log_wandb_confusion_matrix(cfg, epoch, split, cm):
    if cfg.rank != 0 or not cfg.wandb.use_wandb or wandb.run is None or cm is None:
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
    wandb.log({f"{split}/confusion_matrix": wandb.Image(fig)}, step=epoch)
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
    logging.info(cfg)

    if not cfg.model.get('criterion_args', False):
        cfg.model.criterion_args = cfg.criterion_args
    model = build_model_from_cfg(cfg.model).to(device)
    model_size = cal_model_parm_nums(model)
    logging.info(model)
    logging.info('Number of params: %.4f M' % (model_size / 1e6))
    # criterion = build_criterion_from_cfg(cfg.criterion_args).cuda()
    if cfg.model.get('in_channels', None) is None:
        cfg.model.in_channels = cfg.model.encoder_args.in_channels

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

    # build dataset
    train_loader, val_loader, test_loader = _build_loaders(cfg)
    logging.info(f"length of validation dataset: {len(val_loader.dataset)}")
    num_classes = val_loader.dataset.num_classes if hasattr(
        val_loader.dataset, 'num_classes') else None
    if num_classes is None:
        num_classes = cfg.get('num_classes', None)
    num_points = val_loader.dataset.num_points if hasattr(
        val_loader.dataset, 'num_points') else None
    if num_classes is not None:
        assert cfg.num_classes == num_classes
    logging.info(f"number of classes of the dataset: {num_classes}, "
                 f"number of points sampled from dataset: {num_points}, "
                 f"number of points as model input: {cfg.num_points}")
    if cfg.get('classes', None):
        cfg.classes = cfg.classes
    elif hasattr(val_loader.dataset, 'classes'):
        cfg.classes = val_loader.dataset.classes
    elif num_classes is not None:
        cfg.classes = [str(i) for i in range(num_classes)]
    else:
        raise ValueError("Could not infer class names/num_classes from dataset. Set cfg.num_classes or cfg.classes.")
    validate_fn = eval(cfg.get('val_fn', 'validate'))

    # optionally resume from a checkpoint
    if cfg.pretrained_path is not None:
        if cfg.mode == 'resume':
            resume_checkpoint(cfg, model, optimizer, scheduler,
                              pretrained_path=cfg.pretrained_path)
            macc, oa, accs, cm = validate_fn(model, val_loader, cfg)
            print_cls_results(oa, macc, accs, cfg.start_epoch, cfg)
        else:
            if cfg.mode == 'test':
                # test mode
                epoch, best_val = load_checkpoint(
                    model, pretrained_path=cfg.pretrained_path)
                macc, oa, accs, cm = validate_fn(model, test_loader, cfg)
                print_cls_results(oa, macc, accs, epoch, cfg)
                return True
            elif cfg.mode == 'val':
                # validation mode
                epoch, best_val = load_checkpoint(model, cfg.pretrained_path)
                macc, oa, accs, cm = validate_fn(model, val_loader, cfg)
                print_cls_results(oa, macc, accs, epoch, cfg)
                return True
            elif cfg.mode == 'finetune':
                # finetune the whole model
                logging.info(f'Finetuning from {cfg.pretrained_path}')
                load_checkpoint(model, cfg.pretrained_path)
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
    logging.info(f"length of training dataset: {len(train_loader.dataset)}")

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
            val_macc, val_oa, val_accs, val_cm = validate_fn(
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

        if cfg.sched_on_epoch:
            scheduler.step(epoch)
        if cfg.rank == 0:
            save_checkpoint(cfg, model, epoch, optimizer, scheduler,
                            additioanl_dict={'best_val': best_val},
                            is_best=is_best
                            )
    # test the last epoch
    test_macc, test_oa, test_accs, test_cm = validate(model, test_loader, cfg)
    print_cls_results(test_oa, test_macc, test_accs, best_epoch, cfg)
    if writer is not None:
        writer.add_scalar('test_oa', test_oa, epoch)
        writer.add_scalar('test_macc', test_macc, epoch)

    # test the best validataion model
    best_epoch, _ = load_checkpoint(model, pretrained_path=os.path.join(
        cfg.ckpt_dir, f'{cfg.run_name}_ckpt_best.pth'))
    test_macc, test_oa, test_accs, test_cm = validate(model, test_loader, cfg)
    if writer is not None:
        writer.add_scalar('test_oa', test_oa, best_epoch)
        writer.add_scalar('test_macc', test_macc, best_epoch)
    print_cls_results(test_oa, test_macc, test_accs, best_epoch, cfg)

    if writer is not None:
        writer.close()
    if cfg.distributed and dist.is_initialized():
        dist.destroy_process_group()

def train_one_epoch(model, train_loader, optimizer, scheduler, epoch, cfg):
    loss_meter = AverageMeter()
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
        points = data['x']
        target = data['y']
        """ bebug
        from openpoints.dataset import vis_points
        vis_points(data['pos'].cpu().numpy()[0])
        """
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
        if max_batches is not None and (idx + 1) >= max_batches:
            break
    macc, overallacc, accs = cm.all_acc()
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
