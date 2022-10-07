# -*- coding: utf-8 -*-
# @Time    : 2022-05-16 23.42
# @Author  : Nuwan Madhusanka
# @Email   : nuwan@xdoto.io
# @File    : predict.py
# @Software: PyCharm
import os

from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_wavelets.Layers.DWT as DWT
import tensorflow as tf
import tensorflow.keras.backend as K
from keras.preprocessing.image import load_img

def svm_gradient_ramp_loss(layer):
    weights = layer.weights[0]
    weights_tf = tf.convert_to_tensor(weights)

    def categorical_hinge_loss(y_true, y_pred):
        gradient = 0.02
        ramp = 0.02
        pos = K.sum(y_true * y_pred, axis=-1)
        neg = K.max((1.0 - y_true) * y_pred, axis=-1)
        grad_ratio = gradient #0.02
        print("grad ratio",grad_ratio)
        print("ramp",ramp)
        hinge_loss = grad_ratio * K.mean(K.maximum(0.0, neg - pos + 1), axis=-1)
        # Do the ramp process using maximum
        hinge_loss_ramp = K.minimum(ramp,hinge_loss)

        regularization_loss = 0.00001 * (tf.reduce_sum(tf.square(weights_tf)))
        return regularization_loss +   hinge_loss_ramp

    return categorical_hinge_loss

def model_():
    model = keras.Sequential(
        [
            keras.Input(shape=(1,1,4096)),

            # RandomFourierFeatures(
            #     output_dim=4096, scale=10.0, kernel_initializer="gaussian"
            # ),
            # WaveTFFactory.build('db2', 'smooth'),

            # Wavelet1D(batch_size=2,trainable=True),

            # DWT.DWT(name='coif',input_dim=1,concat=0), # ,concat=1

            # layers.Dense(4096,activation = wavelet_activation),

            # DWT.DWT('db1', 'smooth'),
            DWT.DWT(name = 'haar',concat=1),  # ,concat=1
            layers.Flatten(),
            layers.Dense(units=4096), # new update
            layers.Dense(units=8192), # new update
            layers.Dense(units=8192),
            layers.BatchNormalization(),
            layers.Dense(units=4096),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(units=2048),
            layers.Dense(units=512),
            layers.Dense(units=128),

            layers.Dense(units=32),
            layers.Dense(units=2, name='svm'),
        ]
    )
    print(model.summary())
    optimizer = keras.optimizers.Adam(learning_rate=1e-5)  # tf.keras.optimizers.RMSprop(lr=2e-3, decay=1e-5)
    # optimizer = tf.keras.optimizers.RMSprop(learning_rate=1e-5, decay=1e-5)
    # loss = my_loss_fn, wrapper(model.layers[0].get_weights()),#keras.losses.hinge,

    # svm_robust_loss
    # svm_ramp_loss
    # svm_loss
    # svm_robust_with_ramp_loss
    # svm_upper_ramp_loss
    # svm_gradient_ramp_loss


    loss = svm_gradient_ramp_loss(model.get_layer('svm')) #
    print("svm_gradient_ramp_loss")
    # metrics = [keras.metrics.CategoricalAccuracy(name="acc")],['accuracy']
    metrics = ['accuracy']
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=['accuracy']
    )

    return model

model = model_()
model.compile()
load_status = model.load_weights("ckpt")
# load_status.compile()
x = load_img("images/processed/Acarpet_d.png", target_size=(224, 224))
print(load_status.predict())