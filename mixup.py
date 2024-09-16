"""MixUp augmentation and alternative versions. """
from typing import Tuple
import torch


class MixUp(torch.nn.Module):
    """
    Apply MixUp at random to a classificiation batch.
    - MixUp: https://arxiv.org/abs/1710.09412
    - Based on the RandomMixup class in the official PyTorch repo

    Key difference in this code is that each batch sample has its own lambda.
    """
    def __init__(
        self,
        num_classes: int,
        prob_apply: float = 0.5,
        alpha: float = 1.,
    ) -> None:
        """
        Args:
            num_classes (int): number of classification labels
            prob_apply (float): probability of applying MixUp to a batch
            alpha (float): distribution hyperparameter ... Beta(alpha, alpha)
        """
        super().__init__()

        # check arguments
        if not isinstance(num_classes, int):
            raise TypeError('num_classes must be int')
        if num_classes < 1:
            raise ValueError('must set num_classes > 0')
        if alpha <= 0:
            raise ValueError('must set alpha > 0')

        self.num_classes = num_classes
        self.prob_apply = prob_apply
        self.alpha = alpha

    def __repr__(self) -> str:
        return f'MixUp({self.num_classes}, {self.prob_apply}, {self.alpha})'

    @classmethod
    def from_dict(cls, some_args):
        """Instantiate from a pool of arguments. """
        instance_args = {}
        instance_args['num_classes'] = some_args['num_classes']
        if 'prob_apply' in some_args:
            instance_args['prob_apply'] = some_args['prob_apply']
        if 'alpha' in some_args:
            instance_args['alpha'] = some_args['alpha']
        return cls(**instance_args)

    def forward(
        self,
        images: torch.FloatTensor,
        labels: torch.LongTensor,
        inplace: bool = False
    ) -> Tuple[torch.FloatTensor, torch.LongTensor]:
        """Apply MixUp at random.

        Args:
            images (torch.FloatTensor): batch of images
            labels (torch.FloatTensor): batch of labels
            inplace (bool): whether or not to forgo argument cloning

        Returns:
            Tuple[torch.FloatTensor, torch.LongTensor]: mixed images/labels
        """
        # check arguments
        if images.ndim != 4:
            raise ValueError(f'images.ndim must be 4, not {images.ndim}')
        if not images.is_floating_point():
            raise TypeError(f'images.dtype must be float, not {images.dtype}')
        if labels.ndim != 1:
            raise ValueError(f'labels.ndim must be 1, not {labels.ndim}')
        if labels.dtype != torch.int64:
            raise TypeError(f'labels.dtype must be int64, not {labels.dtype}')

        # format arguments
        if not inplace:
            images, labels = images.clone(), labels.clone()
        labels = torch.nn.functional.one_hot(
            labels,
            num_classes=self.num_classes
        ).to(dtype=images.dtype)

        # apply MixUp at random
        if torch.rand(1).item() < self.prob_apply:
            # assign image pairs, roll is faster than shuffle
            other_images = images.roll(1, 0)
            other_labels = labels.roll(1, 0)

            # sample PAIR-SPECIFIC mix amounts (lambda in paper)
            mix_amount = torch._sample_dirichlet(
                self.alpha * torch.ones(images.size(0), 2, dtype=images.dtype)
            )[:, :1]
            mix_amount = mix_amount.to(images.device)

            # mix, final = lambda * current + (1 - lambda) * other
            labels.mul_(mix_amount).add_(other_labels.mul_(1 - mix_amount))
            mix_amount = mix_amount.view(images.size(0), 1, 1, 1)
            images.mul_(mix_amount).add_(other_images.mul_(1 - mix_amount))

        return images, labels


class MixUp2(torch.nn.Module):
    """
    Apply MixUp at random to a classificiation batch.
    - MixUp: https://arxiv.org/abs/1710.09412
    - Based on the RandomMixup class in the official PyTorch repo

    Key difference in this code is that each batch sample has its own lambda.

    The original draws lambda from Beta(alpha, alpha).
    This implementation draws lambda from Uniform(alpha, 1).
    This code does not mix labels and assumes alpha is close to 1.
    """
    def __init__(
        self,
        num_classes: int,
        prob_apply: float = 0.5,
        alpha: float = 0.9,
    ) -> None:
        """
        Args:
            num_classes (int): number of classification labels
            prob_apply (float): probability of applying MixUp to a batch
            alpha (float): distribution hyperparameter ... Uniform(alpha, 1)
        """
        super().__init__()

        # check arguments
        if not isinstance(num_classes, int):
            raise TypeError('num_classes must be int')
        if num_classes < 1:
            raise ValueError('must set num_classes > 0')
        if alpha <= 0 or alpha >= 1:
            raise ValueError('must set 0 < alpha < 1')

        self.num_classes = num_classes
        self.prob_apply = prob_apply
        self.alpha = alpha

    def __repr__(self) -> str:
        return f'MixUp2({self.num_classes}, {self.prob_apply}, {self.alpha})'

    @classmethod
    def from_dict(cls, some_args):
        """Instantiate from a pool of arguments. """
        instance_args = {}
        instance_args['num_classes'] = some_args['num_classes']
        if 'prob_apply' in some_args:
            instance_args['prob_apply'] = some_args['prob_apply']
        if 'alpha' in some_args:
            instance_args['alpha'] = some_args['alpha']
        return cls(**instance_args)

    def forward(
        self,
        images: torch.FloatTensor,
        *_,  # this version CAN accept labels but ignore them
        inplace: bool = False
    ) -> torch.FloatTensor:
        """Apply MixUp at random.

        Args:
            images (torch.FloatTensor): batch of images
            inplace (bool): whether or not to forgo argument cloning

        Returns:
            torch.FloatTensor: mixed images
        """
        # check arguments
        if images.ndim != 4:
            raise ValueError(f'images.ndim must be 4, not {images.ndim}')
        if not images.is_floating_point():
            raise TypeError(f'images.dtype must be float, not {images.dtype}')

        # format arguments
        if not inplace:
            images = images.clone()

        # apply MixUp at random
        if torch.rand(1).item() < self.prob_apply:
            # assign image pairs, roll is faster than shuffle
            other_images = images.roll(1, 0)

            # sample PAIR-SPECIFIC mix amounts (lambda in paper)
            mix_amount = torch.rand(images.size(0))
            mix_amount.mul_(1 - self.alpha).add_(self.alpha)
            mix_amount = mix_amount.to(images.device)

            # mix, final = lambda * current + (1 - lambda) * other
            mix_amount = mix_amount.view(images.size(0), 1, 1, 1)
            images.mul_(mix_amount).add_(other_images.mul_(1 - mix_amount))

        return images
