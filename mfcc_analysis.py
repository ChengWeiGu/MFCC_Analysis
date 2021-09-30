import numpy as np
import pandas as  pd
import wave
import matplotlib.pyplot as plt
import os
from os.path import join, basename
import json
from sklearn.preprocessing import StandardScaler
import librosa


# read 16-bit wave file
def read_wav_data(filename):
    wav = wave.open(filename, "rb")
    num_frame = wav.getnframes()
    num_channel = wav.getnchannels()
    framerate = wav.getframerate()
    num_sample_width = wav.getsampwidth()
    str_data = wav.readframes(num_frame)
    wav.close()
    wave_data = np.fromstring(str_data, dtype=np.short)
    wave_data.shape = -1, num_channel
    wave_data = wave_data.T
    return wave_data, framerate


# plot stft images from wav data in a folder
def generate_stft_maps(dir_path):
    files = os.listdir(dir_path)
    files = [f for f in files if f.endswith(".wav")]
    filenames = [join(dir_path, f) for f in files]
    log_dir = './log/' # A folder to save stft image
    try:
        os.makedirs(log_dir)
    except OSError:
        if not os.path.isdir(log_dir):
            raise
    
    for filename in filenames:
        print(basename(filename),' is being processed...')
        wave_data, framerate = read_wav_data(filename)
        fig, axes = plt.subplots(2,1,figsize = (12,5))
        
        for i , ax in enumerate(axes):
            # print(im)
            im = ax.specgram(wave_data[i],NFFT = 512, Fs = 44100, noverlap = 128, vmin = -100, vmax = 20, cmap = plt.cm.jet)[3]
            if i == 1: 
                ax.set_xlabel("time(s)") # set xlabel under the bottom image
            if i == 0:
                ax.set_title(basename(filename)) # set title
            ax.set_ylabel("Freq(Hz)") # set ylabel for each image
            fig.colorbar(im, ax = ax)
        
        plt.savefig(log_dir + basename(filename) + '.png')
        # plt.show()



# show the data distribution in the feature space
def show_mfcc_scatter(index_pair=(0,1)):
    
    # extract features by mfcc method
    def get_mfcc_data_from_folder(dir_path, n_feat = 40):
        files = os.listdir(dir_path)
        files = [f for f in files if f.endswith(".wav")]
        filenames = [join(dir_path, f) for f in files]
        X = np.array([])
        for filename in filenames:
            print(basename(filename),' is being processed...')
            wave_data, framerate = read_wav_data(filename)
            data = wave_data[1].astype('float32')
            mfccs = librosa.feature.mfcc(y=data, sr=framerate, n_mfcc=n_feat)
            mfccs = mfccs.T
            mfccs = np.average(mfccs, 0) # take mfcc average over time
            X = np.append(X, mfccs)
        X = X.reshape(-1, n_feat)
        return X
    
    
    # normalize data with respect to training data
    def std_normalize(X, *args):
        scaler = StandardScaler()
        scaler.fit(X)
        data_list = [scaler.transform(X)] + [scaler.transform(args[i]) for i in range(len(args))]
        return data_list, scaler.mean_, scaler.var_
    
    
    #-----------------first batch of data------------------------#
    X = get_mfcc_data_from_folder(r"train_1")
    X_OK = get_mfcc_data_from_folder(r"test_1\OK")
    X_NG = get_mfcc_data_from_folder(r"test_1\NG")
    #-----------------second batch of data------------------------#
    X_2nd_mix = get_mfcc_data_from_folder(r"test_2\MIX")
    
    [X, X_OK, X_NG, X_2nd_mix], mu, sigma2 =  std_normalize(X, X_OK, X_NG, X_2nd_mix)
    # print(mu)
    # print(sigma2)
    
    f1 = index_pair[0] # pick uo a feature as x
    f2 = index_pair[1] # pick uo a feature as y
    plt.scatter(X[:, f1], X[:, f2], marker = "x", c = 'b', label = 'train1')
    plt.scatter(X_OK[:, f1], X_OK[:, f2], marker = "D", c = 'g', label = 'test1_OK')
    plt.scatter(X_NG[:, f1], X_NG[:, f2], marker = "o", c = 'r', label = 'test1_NG')
    plt.scatter(X_2nd_mix[:, f1], X_2nd_mix[:, f2], marker = "*", c = 'orange', label = 'test2_MIX')
    
    plt.title('MFCCS space', fontsize = 18)
    plt.xlabel(f'feature{f1}')
    plt.ylabel(f'feature{f2}')
    plt.legend(loc = 1)
    plt.show()


if __name__ == '__main__':
    # generate_stft_maps(dir_path = "wave_16bit")
    show_mfcc_scatter(index_pair=(0,1))