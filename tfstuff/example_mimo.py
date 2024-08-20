"""Example training script for a multi-input, multi-output model. """
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.applications import ResNet50


if __name__ == '__main__':
    # data for a multi-input model for classification and an auxilary task
    images = np.random.rand(128, 64, 64, 3)
    labels = tf.keras.utils.to_categorical(
        np.random.randint(4, size=(128, ))
    )
    features = np.random.rand(128, 7)
    targ_emb = np.random.rand(128, 3)

    # model
    backbone = ResNet50(weights='imagenet', include_top=False)  # downsamp 32
    img_inp = backbone.input
    fork = layers.GlobalAveragePooling2D()(backbone.output)
    aux_inp = layers.Input(shape=(7, ))
    fork = layers.Concatenate()([fork, aux_inp])
    cls_out = layers.Dense(1024, activation='relu')(fork)
    cls_out = layers.Dense(4, activation='softmax')(cls_out)
    aux_out = layers.Dense(1024, activation='relu')(fork)
    aux_out = layers.Dense(3)(aux_out)
    model = tf.keras.models.Model(
        inputs={
            'img_inp': img_inp,
            'aux_inp': aux_inp
        },
        outputs={
            'cls_out': cls_out,
            'aux_out': aux_out
        }
    )

    # training
    model.compile(
        optimizer='sgd',
        loss={
            'cls_out': 'categorical_crossentropy',
            'aux_out': 'mse'
        }
    )
    _ = model.fit(
        {
            'img_inp': images,
            'aux_inp': features
        },
        {
            'cls_out': labels,
            'aux_out': targ_emb
        },
        batch_size=32
    )
