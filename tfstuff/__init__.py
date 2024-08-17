"""Misc tensorflow code for reference. """
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.applications import ResNet50


def init_resnet50(num_classes: int, weights: str = 'imagenet') -> tf.Module:
    """Initialize a resnet50 model.

    Args:
        num_classes (int): number of target classes
        weights (str): name of trained weights or path to weight file

    Returns:
        tf.Module: adjusted resnet50 model
    """
    backbone = ResNet50(weights=weights, include_top=False)  # downsamples 1/32
    x = backbone.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(1024, activation='relu')(x)
    pred = layers.Dense(num_classes, activation='softmax')(x)
    model = tf.keras.models.Model(inputs=backbone.input, outputs=pred)
    return model
