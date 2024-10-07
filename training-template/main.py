"""Training template script based on classification. """
import importlib
import logging
import os
from pathlib import Path
from typing import Any, Callable, Dict, Tuple
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
import torch
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import Compose
from tqdm import tqdm


__author__ = 'William Zhang'


def build_instance(blueprint: DictConfig, updates: Dict = None) -> Any:
    """Build an arbitrary class instance.

    Args:
        blueprint (DictConfig): config w/keys 'module', 'class_name', 'kwargs'
        updates (Dict): additional/non-default kwargs

    Returns:
        Any: instance defined by <module>.<class_name>(**kwargs)
    """
    module = importlib.import_module(blueprint.module)
    instance_kwargs = {}
    if 'kwargs' in blueprint:
        instance_kwargs.update(OmegaConf.to_container(blueprint.kwargs))
    if updates:
        instance_kwargs.update(updates)
    return getattr(module, blueprint.class_name)(**instance_kwargs)


def calc_top1_acc(
    logits: torch.Tensor,
    labels: torch.Tensor
) -> float:
    """Calculate top-1 accuracy.

    Args:
        logits: network classification logits (batch_size, num_classes)
        labels: correct category lables (batch_size, )

    Returns:
        float: top-1 accuracy as a percent
    """
    assert logits.ndim == 2
    correct = logits.argmax(dim=-1).eq(labels)
    return 100 * correct.sum() / correct.numel()


def do_forward_pass(
    inp: torch.Tensor,
    labels: torch.Tensor,
    model: torch.nn.Module,
    criteria: Dict[str, Callable],
    metric: Dict[str, Callable] = None
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """Run minibatch through the model.

    Args:
        inp (torch.Tensor): input data
        target (torch.Tensor): network target values
        model (torch.nn.Module): neural network to train/validate
        criteria (Dict[str, Callable]): task loss functions
        metric (Dict[str, Callable]): task metrics

    Returns:
        Tuple[Dict, Dict]: loss values and metric values
    """
    device = next(model.parameters()).device
    outp = model(inp.to(device))

    # compute losses and metrics
    losses = {
        key: loss_func(outp, labels.to(device))
        for key, loss_func in criteria.items()
    }
    metric_vals = {}
    if metric is not None:
        metric_vals = {
            key: metric_func(outp.cpu(), labels)
            for key, metric_func in metric.items()
        }
    return losses, metric_vals


def log_dict(to_log: Dict) -> None:
    """Log dictionary key and values.

    Args:
        to_log (Dict): information to log
    """
    for key, val in to_log.items():
        logging.info('%s: %f', key, val)


def add_dict_to_board(
    scalars: Dict[str, float],
    board: SummaryWriter,
    step: int
) -> None:
    """Add scalars to tensorboard.

    Args:
        board (SummaryWriter): tensorboard to write to
        scalars (Dict[str, float]): information to add
        step (int): global step
    """
    for key, val in scalars.items():
        board.add_scalar(key, val, step)


@torch.no_grad()
def do_eval_epoch(
    loader: torch.utils.data.DataLoader,
    model: torch.nn.Module,
    criteria: Dict[str, Callable],
    metric: Dict[str, Callable] = None
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """Run validation data through the model.

    Args:
        loader (torch.utils.data.DataLoader): validation batch loader
        model (torch.nn.Module): neural network to train/validate
        criteria (Dict[str, Callable]): task loss functions
        metric (Dict[str, Callable]): task metrics

    Returns:
        Tuple[Dict, Dict]: loss values and metric values
    """
    device = next(model.parameters()).device

    # accumulate outputs
    outp = []
    labels = []
    for inp, target in tqdm(loader, leave=False):
        outp.append(model(inp.to(device)).cpu())
        labels.append(target)
    outp, labels = torch.cat(outp), torch.cat(labels)

    # compute losses and metrics
    losses = {
        key: loss_func(outp, labels)
        for key, loss_func in criteria.items()
    }
    metric_vals = {}
    if metric is not None:
        metric_vals = {
            key: metric_func(outp, labels)
            for key, metric_func in metric.items()
        }
    return losses, metric_vals


@hydra.main(version_base=None, config_path='config', config_name='config')
def main(config: DictConfig = None) -> None:
    """Train a classifier.

    Args:
        config (DictConfig): script configuration
    """
    logging.info('running with torch %s', torch.__version__)
    outdir = Path(HydraConfig.get().runtime.output_dir)
    os.makedirs(outdir / 'checkpoints', exist_ok=True)
    train_board = SummaryWriter(log_dir=f'tensorboard/train-{outdir.stem}')
    valid_board = SummaryWriter(log_dir=f'tensorboard/valid-{outdir.stem}')
    device = torch.device(config.device)

    # get dataset and batch loaders
    preprocess = Compose(
        [build_instance(blueprint) for blueprint in config.dataset.preprocess]
    )
    train_data = build_instance(
        config.dataset.train,
        {'transform': preprocess}
    )
    train_loader = torch.utils.data.DataLoader(
        dataset=train_data,
        batch_size=config.batch_size,
        shuffle=True
    )
    valid_data = build_instance(
        config.dataset.valid,
        {'transform': preprocess}
    )
    valid_loader = torch.utils.data.DataLoader(
        dataset=valid_data,
        batch_size=config.batch_size,
        shuffle=False
    )
    imgs, _ = next(iter(train_loader))

    # get model and optimizers
    model = build_instance(
        config.model,
        {'num_classes': len(train_data.classes)}
    ).to(device)
    if 'weights' in config.model and config.model.weights is not None:
        model.load_state_dict(torch.load(config.model.weights))
    criteria = build_instance(config.optimization.criteria)
    metric = calc_top1_acc
    optimizer = build_instance(
        config.optimization.optimizer,
        {'params': model.parameters()}
    )
    if 'scheduler' in config.optimization:
        scheduler = build_instance(
            config.optimization.scheduler,
            {'optimizer': optimizer}
        )

    # main training loop
    if config.plumbing:
        progbar = tqdm(range(5))
    else:
        progbar = tqdm(range(config.num_epochs))
    num_batches = len(train_loader)
    step = 0
    for epoch in progbar:
        for i_batch, (imgs, labels) in enumerate(train_loader):
            step += 1

            # single gradient descent step
            with torch.autocast(device_type=device.type):
                losses, metric_vals = do_forward_pass(
                    imgs,
                    labels,
                    model,
                    {config.optimization.criteria.class_name: criteria},
                    {'top1': metric}
                )
            optimizer.zero_grad()
            total_loss = 0
            for curr in losses.values():
                total_loss = total_loss + curr
            total_loss.backward()
            optimizer.step()

            # logging and user feedback
            add_dict_to_board(losses, train_board, step)
            add_dict_to_board(metric_vals, train_board, step)
            progbar.set_postfix({'batch': f'{i_batch + 1}/{num_batches}'})

            if config.plumbing:
                break

        if 'scheduler' in locals():
            scheduler.step()

        # periodic validation and checkpoint during training
        model.eval()
        if (epoch + 1) % config.epochs_per_valid == 0:
            losses, metric_vals = do_eval_epoch(
                valid_loader,
                model,
                {config.optimization.criteria.class_name: criteria},
                {'top1': metric}
            )
            add_dict_to_board(losses, valid_board, step)
            add_dict_to_board(metric_vals, valid_board, step)

            # save checkpoint
            checkpoint = {
                'epoch': epoch + 1,
                'step': step,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': None
            }
            if 'scheduler' in locals():
                checkpoint['scheduler'] = scheduler.state_dict()
            torch.save(
                checkpoint,
                outdir / 'checkpoints' / f'{epoch + 1:03d}.pt'
            )
        model.train()

    # final validation
    model.eval()
    losses, metric_vals = do_eval_epoch(
        valid_loader,
        model,
        {config.optimization.criteria.class_name: criteria},
        {'top1': metric}
    )
    logging.info('FINAL LOSSES')
    log_dict(losses)
    logging.info('FINAL METRICS')
    log_dict(metric_vals)

    # save final model
    logging.info('export final weights to ONNX')
    torch.onnx.export(
        model,
        imgs.to(device),
        outdir / 'final.onnx',
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch', 2: 'row', 3: 'col'}
        }
    )

    logging.info('script ran to completion.')


if __name__ == '__main__':
    main()
