from keras import regularizers
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping
from keras.callbacks import  History, ReduceLROnPlateau, CSVLogger
from keras.models import Model, Sequential
from keras.layers import Dense, Embedding, LSTM
from keras.layers import Input, Flatten, Dropout, Activation, BatchNormalization
from keras.layers import Conv1D, MaxPooling1D, AveragePooling1D
from keras.preprocessing import sequence
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils import np_utils
from keras.utils import to_categorical


## Sklearn
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder

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

from keras.layers import Input
from keras.optimizers import RMSprop
from keras.models import Model
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.models import load_model
from keras import backend as K
from keras.layers import Activation, Dense, Input
from keras.layers import Conv2D, Flatten, Lambda,Conv1D, MaxPool2D, Dropout
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

####

def read_mfcc_data() :
  with open('mfcc_data.npy', 'rb') as f:
    x = np.load(f)
    y = np.load(f)
  return x, y

x_data, y_data = read_mfcc_data()
x_data, y_data = shuffle(x_data, y_data)

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size= 0.2, random_state=True, shuffle=True)

# reshape to (28, 28, 1) and normalize input images
image_size = x_train.shape[1]
x_train = np.reshape(x_train, [-1, x_train.shape[1], x_train.shape[2], x_train.shape[3]])
x_test = np.reshape(x_test, [-1,  x_test.shape[1], x_test.shape[2], x_test.shape[3]])
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

lable_color_dict = {}
for p in list(set(y_data)) :
   lable_color_dict[p] = "#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])

####

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
for i in range(4):
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
plot_model(encoder, to_file='vae_cnn_encoder.png', show_shapes=True)

###

# build decoder model
latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
x = Dense(shape[1] * shape[2] * shape[3], activation='relu')(latent_inputs)
x = Reshape((shape[1], shape[2], shape[3]))(x)

for i in range(4):
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
plot_model(decoder, to_file='vae_cnn_decoder.png', show_shapes=True)

###

outputs = decoder(encoder(inputs)[2])
vae = Model(inputs, outputs, name='vae')
from keras.losses import mse, binary_crossentropy
# if args.mse:
reconstruction_loss = mse(K.flatten(inputs), K.flatten(outputs))
#     else:
# reconstruction_loss = binary_crossentropy(K.flatten(inputs), K.flatten(outputs))
reconstruction_loss *= image_size * image_size
kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
kl_loss = K.sum(kl_loss, axis=-1)
kl_loss *= -0.5
vae_loss = K.mean(reconstruction_loss + kl_loss)
vae.add_loss(vae_loss)
vae.compile(optimizer='rmsprop')
vae.summary()

###

vae.load_weights('vae_cnn_mnist.h5')

###

z_mean, z_log_var, z = encoder.predict(x_data, batch_size=12)
classifier_input_shape = z.shape
classifier_input_shape = (2,)
y_dt = to_categorical(y_data)
num_labels = len(y_dt[0])
filters = 3
shape = (None, 18, 27, 256)
x_train, x_test, y_train, y_test = train_test_split(z, y_dt, test_size= 0.2, random_state=True, shuffle=True)

###

def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def fscore(y_true, y_pred):
    if K.sum(K.round(K.clip(y_true, 0, 1))) == 0:
        return 0

    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    f_score = 2 * (p * r) / (p + r + K.epsilon())
    return f_score

def get_lr_metric(optimizer):
    def lr(y_true, y_pred):
        return optimizer.lr

###

inputs = Input(shape=classifier_input_shape, name='classifier_input')
x = inputs
x = Dense(shape[1] * shape[2] * shape[3], activation='relu')(x)
x = Reshape((shape[1], shape[2], shape[3]))(x)
x = Conv2D(filters=filters, kernel_size=16, activation='relu', strides=2, padding='same')(x)
x = MaxPool2D(pool_size=(3, 3))(x)
x = Dropout(0.25)(x)
x = Flatten(100)(x)
x = Dense(225, activation='relu')(x)
x = Dropout(0.25)(x)
x = Dense(num_labels, activation = "softmax")(x)
classifier = Model(inputs, x, name='classifier')
classifier.summary()
plot_model(classifier, to_file='mfcc_classifier.png', show_shapes=True)



###

from keras.optimizers import SGD
opt = SGD(lr=0.0001, momentum=0.0, decay=0.0, nesterov=False)
classifier.compile(loss='categorical_crossentropy', optimizer=opt)
# lr_reduce = ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=20, min_lr=0.000001)
# Please change the model name accordingly.
# mcp_save = ModelCheckpoint('model/aug_noiseNshift_2class2_np.h5', save_best_only=True, monitor='val_loss', mode='min')
classifier.fit(x_train, y_train, batch_size=2, epochs=7, validation_data=(x_test, y_test))


###

