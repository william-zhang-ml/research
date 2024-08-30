"""Fine tune ImageNet weights for CIFAR. """
import logging
import hydra
from omegaconf import DictConfig
import torch
from torchvision.datasets import CIFAR100
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.transforms import (
    Compose,
    Resize,
    ToTensor,
    RandomHorizontalFlip,
    RandomVerticalFlip
)
from tqdm import tqdm


@hydra.main(version_base=None, config_path='.', config_name='config')
def main(config: DictConfig = None) -> None:
    """Fine tune ImageNet-pretrained ResNet50 to CIFAR100.

    Args:
        config (DictConfig): script configuration
    """
    preproc = Compose([ToTensor(), Resize((224, 224))])
    augment = Compose([RandomHorizontalFlip(), RandomVerticalFlip()])
    train_data = CIFAR100(
        root=config.path_to_data, 
        train=True,
        transform=Compose([preproc, augment])
    )
    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=32,
        shuffle=True
    )
    valid_data = CIFAR100(
        root=config.path_to_data,
        train=False,
        transform=Compose([preproc, augment])
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_data,
        batch_size=128,
        shuffle=False
    )
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2).to(config.device)
    model.fc = torch.nn.Linear(model.fc.in_features, 100)
    optimizer = torch.optim.SGD(model.fc.parameters(), lr=1e-3)
    criteria = torch.nn.CrossEntropyLoss()
    for train_imgs, train_labels in tqdm(train_loader):
        logits = model(train_imgs.to(config.device))
        loss = criteria(logits, train_labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        for valid_imgs, valid_labels in tqdm(valid_loader):
            logits = model(valid_imgs.to(config.device))
            acc = (logits.argmax(dim=1) == valid_labels).float().sum()
    logging.info(acc.item())


if __name__ == '__main__':
    main()
