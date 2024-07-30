"""Training template script based on classification. """
import importlib
import logging
from pathlib import Path
from typing import Any, Callable, Dict, Tuple
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
import torch
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
    for i_epoch in progbar:
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
            progbar.set_postfix({'batch': f'{i_batch + 1}/{num_batches}'})

            if config.plumbing:
                break

        if 'scheduler' in locals():
            scheduler.step()

    # validation
    model.eval()
    losses, metric_vals = do_eval_epoch(
        valid_loader,
        model,
        {config.optimization.criteria.class_name: criteria},
        {'top1': metric}
    )
    logging.info('FINAL LOSSES')
    for key, val in losses.items():
        logging.info('%s: %f', key, val)
    logging.info('FINAL METRICS')
    for key, val in metric_vals.items():
        logging.info('%s: %f', key, val)

    # save final model
    if not config.plumbing:
        torch.onnx.export(
            model,
            imgs.to(device),
            outdir / 'final.onnx',
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch', 1: 'row', 2: 'col'}
            }
        )


if __name__ == '__main__':
    main()
    logging.info('script ran to completion.')
