"""Training adversarial autoencoders. """
import logging
import hydra
from omegaconf import DictConfig
from torch import nn


@hydra.main(version_base=None, config_path='.', config_name='config')
def main(cfg: DictConfig = None) -> None:
    """_summary_

    Args:
        cfg (DictConfig): _description_
    """
    if cfg.user_note:
        logging.info('user note: % s', cfg.user_note)

    # AAE components are nets w/2 x 1000-wide layers (AAE paper Appendix A1)
    logging.info('init autoencoder and discriminator')
    encoder = nn.Sequential(
        nn.Linear(32 * 32, 1000),
        nn.ReLU(),
        nn.Linear(1000, 1000),
        nn.ReLU(),
        nn.Linear(1000, 8),
    )
    decoder = nn.Sequential(
        nn.Linear(8, 1000),
        nn.ReLU(),
        nn.Linear(1000, 1000),
        nn.ReLU(),
        nn.Linear(1000, 32 * 32),
    )
    discriminator = nn.Sequential(
        nn.Linear(8, 1000),
        nn.ReLU(),
        nn.Linear(1000, 1000),
        nn.ReLU(),
        nn.Linear(1000, 1),
    )

    logging.info('training')

    logging.info('main ran to completion')


if __name__ == '__main__':
    main()
