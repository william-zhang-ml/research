"""Training template script based on classification. """
import importlib
import logging
from pathlib import Path
from typing import Any, Dict
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
import torch
from torchvision.transforms import Compose


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


@hydra.main(version_base=None, config_path='config', config_name='config')
def main(config: DictConfig = None) -> None:
    """Train a classifier.

    Args:
        config (DictConfig): script configuration
    """
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
    imgs, _ = next(iter(train_loader))

    # get model and optimizers
    model = build_instance(
        config.model,
        {'num_classes': len(train_data.classes)}
    ).to(device)

    # save final model
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
