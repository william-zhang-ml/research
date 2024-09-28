"""Training adversarial autoencoders. """
import logging
import sys
import hydra
from omegaconf import DictConfig
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, Resize, ToPILImage, ToTensor
from tqdm.auto import tqdm


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
    data_train = MNIST(root=cfg.data_dir, train=True, transform=preproc)
    data_valid = MNIST(root=cfg.data_dir, train=False, transform=preproc)
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
        nn.Sigmoid()
    ).to(cfg.device)

    logging.info('init training objects')
    recon_criteria = nn.MSELoss()
    autoenc_optimizer = optim.SGD(
        nn.ModuleList([encoder, decoder]).parameters(),
        lr=1e-1,
        momentum=0.9,
        weight_decay=1e-3
    )
    autoenc_scheduler = optim.lr_scheduler.StepLR(
        autoenc_optimizer,
        step_size=25
    )

    logging.info('training')
    progbar = tqdm(range(75))
    for _ in progbar:
        for imgs, _ in tqdm(loader_train, leave=False):
            imgs = imgs.view(imgs.shape[0], -1).to(cfg.device)
            latent = encoder(imgs.to(cfg.device))
            recon = decoder(latent)
            recon_loss = recon_criteria(recon, imgs)
            loss = recon_loss
            autoenc_optimizer.zero_grad()
            loss.backward()
            autoenc_optimizer.step()

            if torch.isnan(loss):
                logging.error('nan loss')
                sys.exit()
        autoenc_scheduler.step()
        progbar.set_postfix({
                'loss': loss.item(),
                'recon loss': recon_loss.item(),
            })
        ToPILImage()(recon[0].detach().cpu().view(32, 32)).save('example.png')

    logging.info('main ran to completion')


if __name__ == '__main__':
    main()
