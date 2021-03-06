import librosa
import librosa.display
import matplotlib.pyplot as plt
from matplotlib import image
import numpy as np
from sklearn.metrics import f1_score
import os

audio_paths = []

VCTK_PATH = '../dataset/VCTK/wav48/'
IMOECAP_PATH = '../dataset/IMOECAP/'
RES_PATH = '../dataset/RES/'

def covert2mfcc(group) :

    if group == 'VCTK' :
        paths = [VCTK_PATH]
    elif group == 'IMOECAP' :
        paths = [IMOECAP_PATH]
    elif group == 'RES' :
        paths = [RES_PATH]

    audio_paths = []
    for path in paths :
        for folder in os.listdir(path) :
            for file in os.listdir(path+folder) :
                audio_paths.append(path+folder+'/'+file)
    
    for audio_path in audio_paths :
        for audio_path in audio_paths : 
            (xf, sr) = librosa.load(audio_path)    
            mfccs = librosa.feature.mfcc(y=xf, sr=sr, n_mfcc=4)
            librosa.display.specshow(mfccs)
            plt.savefig('../dataset/mfcc/'+group+'/'+audio_path.split('/')[-1]+'.jpeg')
            print('../dataset/mfcc/'+group+'/'+audio_path.split('/')[-1]+'.jpeg')

def save_mfcc_np(group) :
  mfcc_x_data = []
  mfcc_y_data = []
  mfcc_e_data = []
  GROUP_PATH = '../dataset/mfcc/'+group
  mfcc_list = os.listdir(GROUP_PATH)
  for file in mfcc_list :
    mfcc_x_data.append(image.imread(GROUP_PATH+'/'+file))
    if group == "RES" :
        mfcc_y_data.append(file.split('-')[-1].split('.')[0])
        mfcc_e_data.append(file.split('-')[2])
    else :
        mfcc_y_data.append(file.split('_')[0].split('p')[-1])

  mfcc_x_data = np.array(mfcc_x_data)
  mfcc_y_data = np.array(mfcc_y_data)
  mfcc_e_data = np.array(mfcc_e_data)

  if group == "RES" :
    filename = 'mfcc_data_'+group+'1.npy'
  else :
    filename = 'mfcc_data_'+group+'.npy'
  with open(filename, 'wb') as f:
    np.save(f, mfcc_x_data)
    np.save(f, mfcc_y_data)
    if group == "RES" :
        np.save(f, mfcc_e_data)

def read_mfcc_data(group) :
  if group != "RES" :
    with open('mfcc_data_'+group+'.npy', 'rb') as f:
        x = np.load(f)
        y = np.load(f)
    return x, y
  else  :
    with open('mfcc_data_'+group+'1.npy', 'rb') as f:
        x = np.load(f)
        y = np.load(f)
        e = np.load(f)
    return x, y, e

def classifier_result_intepretation(group) :
    y_pred = []
    y_true = []
    with open('classifier_'+group+'.npy', 'rb') as f:
        x = np.load(f)
        y = np.load(f)
        for i, j in zip(x,y) :
            i = list(i)
            j = list(j)
            y_pred.append(i.index(max(i)))
            y_true.append(j.index(max(j)))
            # print(i_max, j_max)
    
    print(f1_score(y_true, y_pred, average='macro'))
    print(f1_score(y_true, y_pred, average='micro'))
    print(f1_score(y_true, y_pred, average='weighted'))

def plot_results (models, data, lable_color_dict, batch_size=128, model_name="vae_mnist"):
    """Plots labels and MNIST digits as function of 2-dim latent vector
    # Arguments:
        models (tuple): encoder and decoder models
        data (tuple): test data and label
        batch_size (int): prediction batch size
        model_name (string): which model is using this function
    """

    encoder, decoder = models
    x_test, y_test = data
    # os.makedirs(model_name, exist_ok=True)

    filename = "./vae_mean.png"
    # display a 2D plot of the digit classes in the latent space
    z_mean, _, _ = encoder.predict(x_test, batch_size=batch_size)
    plt.figure(figsize=(12, 10))
    color_test = list(map(lambda x : lable_color_dict[x], y_test))
    plt.scatter(z_mean[:, 0], z_mean[:, 1], c=color_test)
    plt.colorbar()
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.savefig(filename)
    # plt.show()

if __name__ == "__main__" :
    # covert2mfcc('VCTK')
    # covert2mfcc('IMOECAP')
    # covert2mfcc('RES')
    # save_mfcc_np('RES')
    print(read_mfcc_data('RES'))
    # save_mfcc_np('IMOECAP')
    # classifier_result_intepretation('RES')
    

    