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

from utils import read_mfcc_data 
#####

x, y = read_mfcc_data('VCTK')
x, y = shuffle(x, y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.2, random_state=True, shuffle=True)

# reshape to (28, 28, 1) and normalize input images
image_size = x_train.shape[1]
x_train = np.reshape(x_train, [-1, x_train.shape[1], x_train.shape[2], x_train.shape[3]])
x_test = np.reshape(x_test, [-1,  x_test.shape[1], x_test.shape[2], x_test.shape[3]])
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

lable_color_dict = {}
for p in list(set(y)) :
   lable_color_dict[p] = "#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])

# network parameters
input_shape = (x_train.shape[1], x_train.shape[2], x_train.shape[3])
kernel_size = 3
filters = 16
latent_dim = 2

###

# reparameterization trick
# instead of sampling from Q(z|X), sample eps = N(0,I)
# then z = z_mean + sqrt(var)*eps
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

###

inputs = Input(shape=input_shape, name='encoder_input')
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

###

# build decoder model
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

###

outputs = decoder(encoder(inputs)[2])
vae = Model(inputs, outputs, name='vae')

models = (encoder, decoder)
data = (x_test, y_test)
from keras.losses import mse, binary_crossentropy
# if args.mse:
#         reconstruction_loss = mse(K.flatten(inputs), K.flatten(outputs))
#     else:
reconstruction_loss = binary_crossentropy(K.flatten(inputs), K.flatten(outputs))

reconstruction_loss *= image_size * image_size
kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
kl_loss = K.sum(kl_loss, axis=-1)
kl_loss *= -0.5
vae_loss = K.mean(reconstruction_loss + kl_loss)
vae.add_loss(vae_loss)
vae.compile(optimizer='rmsprop')
vae.summary()

###

vae.fit(x_train,epochs=1000,batch_size=2, validation_data=(x_test, None))
vae.save_weights('vae_cnn_mnist.h5')

###

def plot_results(models,
                 data,
                 batch_size=128,
                 model_name="vae_mnist",
                 lable_color_dict):
    """Plots labels and MNIST digits as function of 2-dim latent vector
    # Arguments:
        models (tuple): encoder and decoder models
        data (tuple): test data and label
        batch_size (int): prediction batch size
        model_name (string): which model is using this function
    """

    encoder, decoder = models
    x_test, y_test = data
    os.makedirs(model_name, exist_ok=True)

    filename = os.path.join(model_name, "vae_mean.png")
    # display a 2D plot of the digit classes in the latent space
    z_mean, _, _ = encoder.predict(x_test,
                                   batch_size=batch_size)
    plt.figure(figsize=(12, 10))
    color_test = list(map(lambda x : lable_color_dict[x], y_test))
    plt.scatter(z_mean[:, 0], z_mean[:, 1], c=color_test)
    plt.colorbar()
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.savefig(filename)
    plt.show()

    # filename = os.path.join(model_name, "digits_over_latent.png")
    # # display a 30x30 2D manifold of digits
    # n = 30
    # digit_size = 28
    # figure = np.zeros((digit_size * n, digit_size * n))
    # # linearly spaced coordinates corresponding to the 2D plot
    # # of digit classes in the latent space
    # grid_x = np.linspace(-4, 4, n)
    # grid_y = np.linspace(-4, 4, n)[::-1]

    # for i, yi in enumerate(grid_y):
    #     for j, xi in enumerate(grid_x):
    #         z_sample = np.array([[xi, yi]])
    #         x_decoded = decoder.predict(z_sample)
    #         digit = x_decoded[0].reshape(digit_size, digit_size)
    #         figure[i * digit_size: (i + 1) * digit_size,
    #                j * digit_size: (j + 1) * digit_size] = digit

    # plt.figure(figsize=(10, 10))
    # start_range = digit_size // 2
    # end_range = n * digit_size + start_range + 1
    # pixel_range = np.arange(start_range, end_range, digit_size)
    # sample_range_x = np.round(grid_x, 1)
    # sample_range_y = np.round(grid_y, 1)
    # plt.xticks(pixel_range, sample_range_x)
    # plt.yticks(pixel_range, sample_range_y)
    # plt.xlabel("z[0]")
    # plt.ylabel("z[1]")
    # plt.imshow(figure, cmap='Greys_r')
    # plt.savefig(filename)
    # plt.show()

###

plot_results(models, data, batch_size=2, model_name="vae_cnn")
