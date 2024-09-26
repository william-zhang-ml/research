"""Training adversarial autoencoders. """
import logging
import hydra
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path='.', config_name='config')
def main(cfg: DictConfig = None) -> None:
    """_summary_

    Args:
        cfg (DictConfig): _description_
    """
    if cfg.user_note:
        logging.info('user note: % s', cfg.user_note)

    logging.info('init autoencoder and discriminator')
    encoder = None
    decoder = None
    discriminator = None

    logging.info('training')

    logging.info('main ran to completion')


if __name__ == '__main__':
    main()
