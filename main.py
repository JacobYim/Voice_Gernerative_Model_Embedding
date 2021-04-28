from vae import encoder, decoder, vae, classifier
from const import kernel_size, filters, latent_dim
from keras.layers import Input
from utils import read_mfcc_data, plot_results
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.layers import Conv2D, Flatten, Lambda,Conv1D, MaxPool2D, Dropout

import numpy as np
import random

def init_vae(group) :
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
   encoder_t, shape, z_log_var, z_mean = encoder(inputs, latent_dim, filters)
   decoder_t = decoder(latent_dim, shape, filters, kernel_size)
   vae_t = vae(inputs, encoder_t, decoder_t, image_shape, z_log_var, z_mean)

   return  vae_t, encoder_t, decoder_t,  x, y, x_train, x_test, y_test, lable_color_dict, group

def train_vae (vae, encoder, decoder,  x, y, x_train, x_test, y_test, lable_color_dict, group, additional=False) :
   if additional :
      vae.load_weights('vae_'+group+'.h5')

   vae.fit(x_train, epochs=50,batch_size=10, validation_data=(x_test, None))
   vae.save_weights('vae_'+group+'.h5', overwrite=True)

   models = (encoder, decoder)
   data = (x, y)
   plot_results(models, data, lable_color_dict, model_name='vae_'+group+'.h5')


def train_classifier(vae, encoder, decoder,  x, y, x_train, x_test, y_test, lable_color_dict, group, additional=False) :
   
   vae.load_weights('vae_'+group+'.h5')

   z_mean, z_log_var, z = encoder.predict(x, batch_size=12)
   classifier_input_shape = z.shape
   classifier_input_shape = (2,)
   y_dt = to_categorical(y)
   num_labels = len(y_dt[0])
   
   filters = 3
   shape = (None, 18, 27, 256)
   x_train, x_test, y_train, y_test = train_test_split(z, y_dt, test_size= 0.2, random_state=True, shuffle=True)

   inputs = Input(shape=classifier_input_shape, name='classifier_input')
   classifier_t = classifier(inputs, shape, filters, num_labels)
   
   if additional :
      vae.load_weights('classifier_'+group+'.h5')

   classifier_t.fit(x_train, y_train, batch_size=2, epochs=7, validation_data=(x_test, y_test))
   classifier_t.save_weights('classifier_'+group+'.h5', overwrite=True)
   # vae.fit(x_train, y_train, batch_size=2, epochs=7, validation_data=(x_test, y_test))
   # vae.save_weights('classifier_'+group+'.h5', overwrite=True)

if __name__ == "__main__" : 

   # vae
   (vae_t, encoder_t, decoder_t,  x, y, x_train, x_test, y_test, lable_color_dict, group) = init_vae('RES')

   # train_vae(vae_t, encoder_t, decoder_t,  x, y, x_train, x_test, y_test, lable_color_dict, group, True)

   train_classifier(vae_t, encoder_t, decoder_t,  x, y, x_train, x_test, y_test, lable_color_dict, group, False)
   

