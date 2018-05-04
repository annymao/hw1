'''
 Example code 4 :  classification using MFCC and SVM

 Utility function
 	getfeature : load audio data and transform it to MFCCs
	parselabel : parse the label information and generate feature and label data

 Xuei-Wei Liao
 '''

import numpy as np
import librosa
import re
import os

# function for computing the MFCCs
# return its mean vector and covariance vector


def _mfcc_cov(y, sr):
    # call the mfcc routine of librosa
    # where n_mfcc is the number of DCT components you want
    #		n_mels is the number of mel-filter banks
    Chroma = librosa.feature.chroma_stft(y=y, sr=sr,hop_length = librosa.time_to_samples(1,sr=sr))
    Chroma = Chroma/np.tile(np.sum(np.abs(Chroma)**2, axis=0)**(1./2), \
                        (Chroma.shape[0], 1))
    GAMA = 1
    Chroma = np.log10(1+GAMA *np.abs(Chroma))
    #sumChroma =np.sum(Chroma,axis = 0)
    return Chroma;

def _mfcc_d_dd(y, sr):
    # call the mfcc routine of librosa
    # where n_mfcc is the number of DCT components you want
    #		n_mels is the number of mel-filter banks
    """
    mfcc = librosa.feature.mfcc(y, sr)
    mean_mfcc = np.mean(mfcc, axis=-1)
    std_mfcc = np.std(mfcc, axis=-1)
    d_mfcc = np.mean(np.diff(mfcc, n=1, axis =-1), axis=-1)
    std_d_mfcc = np.std(np.diff(mfcc, n=1, axis =-1), axis=-1)
    dd_mfcc = np.mean(np.diff(mfcc, n=2, axis =-1), axis=-1)
    std_dd_mfcc = np.std(np.diff(mfcc, n=2, axis =-1), axis=-1)
    feature = np.hstack((mean_mfcc, d_mfcc, dd_mfcc,std_mfcc, std_d_mfcc, std_dd_mfcc))
    return feature
    """
    Chroma = librosa.feature.chroma_stft(y=y, sr=sr,hop_length = int(sr),n_fft = int(sr*0.5) )
    Chroma = Chroma/np.tile(np.sum(np.abs(Chroma)**2, axis=0)**(1./2), \
                        (Chroma.shape[0], 1))
    GAMA = 1
    Chroma = np.log10(1+GAMA *np.abs(Chroma))
    #sumChroma =np.sum(Chroma,axis = 0)
    return Chroma;

def parselabel(audiopath, feature_type='mfcc_delta'):
    # prepare the variable save feature and label
   # XChroma = np.empty((0, int(n_mfcc + n_mfcc * (n_mfcc + 1) / 2)))
    # Our label is retrieved from the name of audio. We can use
    # regular expression(re) to parse the useful information.
    
    XChroma = np.empty(( 2000));
    if feature_type == 'mfcc_cov':
        getChroma = _mfcc_cov
    elif feature_type == 'mfcc_delta':
        getChroma = _mfcc_d_dd
    try:
        y, sr = librosa.load(audiopath, sr=None)
        XChroma = getChroma(y, sr)
        empty = np.zeros((12, 2000-XChroma.shape[1]))
        XChroma = np.hstack((XChroma, empty))
        #print(XChroma.shape)
        return np.transpose(XChroma)
    except:
        print('File Not Found ! : ' + audiopath)
        return
def parseAnswer(answerPath):
    #ans =open(answerPath,'r').read()
    with open(answerPath) as f:
        data = f.readlines()
    data = [i.rstrip().split('\t', 1)[len(i.split('\t', 1))-1] for i in data]
    data.pop(0)

    for i in range(len(data),2000):
        data.append('')
    
    #print(data.shape)
    return np.asarray(data)

if __name__ == '__main__':
    
    X= parselabel('train/1/5.wav')
    #np.savez('unit_test.npz', x=X)
    Y=parseAnswer('answer_train/REF_key_5.txt')
    print(X.shape,len(Y))
    
   # 

