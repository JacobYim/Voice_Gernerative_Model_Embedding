from keras.layers import Input
from keras.optimizers import RMSprop
from keras.models import Model
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.models import load_model
from keras import backend as K
from keras.layers import Activation, Dense, Input
from keras.layers import Conv2D, Flatten, Lambda
from keras.layers import Reshape, Conv2DTranspose
from keras.layers import LeakyReLU
from keras.layers import BatchNormalization
from keras.layers.merge import concatenate
from keras.utils import plot_model

import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import os
import random
from const import kernel_size, filters, latent_dim
from keras.losses import mse, binary_crossentropy
from keras.optimizers import SGD
from keras.layers import Conv2D, Flatten, Lambda,Conv1D, MaxPool2D, Dropout

def sampling(args):
    """Reparameterization trick by sampling fr an isotropic unit Gaussian.
    # Arguments:
        args (tensor): mean and log of variance of Q(z|X)
    # Returns:
        z (tensor): sampled latent vector
    """

    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean=0 and std=1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

def encoder(inputs, latent_dim, filters) :
    x = inputs
    for i in range(2):
        filters *= 2
        x = Conv2D(filters=filters,
                kernel_size=kernel_size,
                activation='relu',
                strides=2,
                padding='same')(x)

    # shape info needed to build decoder model
    shape = K.int_shape(x)

    # generate latent vector Q(z|X)
    x = Flatten()(x)
    x = Dense(16, activation='relu')(x)
    z_mean = Dense(latent_dim, name='z_mean')(x)
    z_log_var = Dense(latent_dim, name='z_log_var')(x)

    # use reparameterization trick to push the sampling out as input
    # note that "output_shape" isn't necessary with the TensorFlow backend
    z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

    # instantiate encoder model
    encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
    encoder.summary()
    # plot_model(encoder, to_file='vae_cnn_encoder.png', show_shapes=True)

    return encoder, shape, z_log_var, z_mean

def decoder(latent_dim, shape, filters, kernel_size) :
    latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
    x = Dense(shape[1] * shape[2] * shape[3], activation='relu')(latent_inputs)
    x = Reshape((shape[1], shape[2], shape[3]))(x)

    for i in range(2):
        x = Conv2DTranspose(filters=filters,
                            kernel_size=kernel_size,
                            activation='relu',
                            strides=2,
                            padding='same')(x)
        filters //= 2

    outputs = Conv2DTranspose(filters=3,
                            kernel_size=kernel_size,
                            activation='sigmoid',
                            padding='same',
                            name='decoder_output')(x)

    # instantiate decoder model
    decoder = Model(latent_inputs, outputs, name='decoder')
    decoder.summary()
    # plot_model(decoder, to_file='vae_cnn_decoder.png', show_shapes=True)

    return decoder

def vae(inputs, encoder, decoder, image_shape, z_log_var, z_mean) :
    outputs = decoder(encoder(inputs)[2])
    vae = Model(inputs, outputs, name='vae')
    # if args.mse:
    reconstruction_loss = mse(K.flatten(inputs), K.flatten(outputs))
    #     else:
    # reconstruction_loss = binary_crossentropy(K.flatten(inputs), K.flatten(outputs))

    reconstruction_loss *= image_shape[1] * image_shape[2] * image_shape[3]
    kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    vae_loss = K.mean(reconstruction_loss + kl_loss)
    vae.add_loss(vae_loss)
    vae.compile(optimizer='rmsprop')
    vae.summary()
    return vae

def classifier(inputs, shape, filters, num_labels) :
    x = inputs
    x = Dense(shape[1] * shape[2] * shape[3], activation='relu')(x)
    x = Reshape((shape[1], shape[2], shape[3]))(x)
    x = Conv2D(filters=filters, kernel_size=16, activation='relu', strides=2, padding='same')(x)
    x = MaxPool2D(pool_size=(3, 3))(x)
    x = Dropout(0.25)(x)
    x = Flatten()(x)
    x = Dense(2, activation='relu')(x)
    x = Dropout(0.25)(x)
    x = Dense(num_labels, activation = "softmax")(x)
    classifier = Model(inputs, x, name='classifier')
    classifier.summary()
    # plot_model(classifier, to_file='mfcc_classifier.png', show_shapes=True)

    opt = SGD(lr=0.0001, momentum=0.0, decay=0.0, nesterov=False)
    classifier.compile(loss='categorical_crossentropy', optimizer=opt)

    return classifier
