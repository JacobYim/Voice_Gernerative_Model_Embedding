from vae import encoder, decoder, vae
from const import kernel_size, filters, latent_dim
from keras.layers import Input
from utils import read_mfcc_data, plot_results
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

import numpy as np
import random

group = 'IMOECAP'
x, y = read_mfcc_data(group)
x, y = shuffle(x, y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.2, random_state=True, shuffle=True)

# reshape to (28, 28, 1) and normalize input images
image_shape = x_train.shape
x_train = np.reshape(x_train, [-1, x_train.shape[1], x_train.shape[2], x_train.shape[3]])
x_test = np.reshape(x_test, [-1,  x_test.shape[1], x_test.shape[2], x_test.shape[3]])
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

lable_color_dict = {}
for p in list(set(y)) :
   lable_color_dict[p] = "#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])

# network parameters
input_shape = (x_train.shape[1], x_train.shape[2], x_train.shape[3])

inputs = Input(shape=input_shape, name='encoder_input')
encoder, shape, z_log_var, z_mean = encoder(inputs, latent_dim, filters)
decoder = decoder(latent_dim, shape, filters, kernel_size)
vae = vae(inputs, encoder, decoder, image_shape, z_log_var, z_mean)

vae.fit(x_train,epochs=1,batch_size=2, validation_data=(x_test, None))
vae.save_weights('vae_'+group+'.h5')

models = (encoder, decoder)
data = (x_test, y_test)
plot_results(models, data, lable_color_dict, model_name='vae_'+group+'.h5')

