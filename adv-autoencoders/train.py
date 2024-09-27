"""Training adversarial autoencoders. """
import logging
import hydra
from omegaconf import DictConfig
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, Resize, ToTensor


@hydra.main(version_base=None, config_path='.', config_name='config')
def main(cfg: DictConfig = None) -> None:
    """_summary_

    Args:
        cfg (DictConfig): _description_
    """
    if cfg.user_note:
        logging.info('user note: % s', cfg.user_note)

    logging.info('load dataset')
    preproc = Compose([ToTensor(), Resize((32, 32))])
    data_train = MNIST(root=cfg.DATA_DIR, train=True, transform=preproc)
    data_valid = MNIST(root=cfg.DATA_DIR, train=False, transform=preproc)
    loader_train = DataLoader(data_train, batch_size=100, shuffle=True)

    # AAE components are nets w/2 x 1000-wide layers (AAE paper Appendix A1)
    logging.info('init autoencoder and discriminator')
    encoder = nn.Sequential(
        nn.Linear(32 * 32, 1000),
        nn.ReLU(),
        nn.Linear(1000, 1000),
        nn.ReLU(),
        nn.Linear(1000, 8),
    ).to(cfg.device)
    decoder = nn.Sequential(
        nn.Linear(8, 1000),
        nn.ReLU(),
        nn.Linear(1000, 1000),
        nn.ReLU(),
        nn.Linear(1000, 32 * 32),
    ).to(cfg.device)
    discriminator = nn.Sequential(
        nn.Linear(8, 1000),
        nn.ReLU(),
        nn.Linear(1000, 1000),
        nn.ReLU(),
        nn.Linear(1000, 1),
    ).to(cfg.device)

    logging.info('init training objects')
    recon_criteria = nn.MSELoss()
    gan_criteriria = nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(
        nn.ModuleList([encoder, decoder, discriminator]),
        lr=1e-3,
        momentum=0.9,
        weight_decay=1e-3
    )

    logging.info('training')
    for _ in range(1):
        for imgs, _ in loader_train:
            break
        break

    logging.info('main ran to completion')


if __name__ == '__main__':
    main()
