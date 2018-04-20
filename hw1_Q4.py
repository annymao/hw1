# -*- coding: utf-8 -*-
"""
Created on Fri Apr 20 22:00:04 2018

@author: anny
"""
import numpy as np
import librosa.feature
import scipy.io.wavfile as wav
import scipy.signal
import matplotlib 
import matplotlib.pyplot as plt

font = {'family' : 'sans-serif', 'sans-serif':'Arial', 'size'   : 18}
matplotlib.rc('font', **font)

# Generate major chord templates
Major_template = np.array([[6.35,2.23,3.48,2.33,4.38,4.09,2.52,5.19,2.39,3.66,2.29,2.88]])
# Generate monor chord templates
Minor_template = np.array([[6.33,2.68,3.52,5.38,2.60,3.53,2.54,4.75,3.98,2.69,3.34,3.17]])

Template = Major_template
for i in range(11):
    Template = np.append(Template, np.roll(Major_template, i+1), axis=0)    
for i in range(12):
    Template = np.append(Template, np.roll(Minor_template, i), axis=0)
 
#for debug
Key = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#',\
           'G', 'G#', 'A', 'A#', 'B',\
           'c', 'c#', 'd', 'd#', 'e', 'f', 'f#',\
           'g', 'g#', 'a', 'a#', 'b'] 
acc = 0;         
music_pieces = 100;
#manually change path to each genre and test each song     
GENRES = "rock"      
"""
---------------ACCURACY-------------------------------------------
            new Acc                        |
gama      100     |    10    |    1        |
ROCK       |   | 0.45816     |
HIPHOP     |   | 0.17407     |
POP        |   | 0.59468     |
BLUES      |   | 0.28673     |
METAL      |   | 0.39140     |
-----------------------------------------------------------------
"""
for fileLen in range(100):
    path = 'genres/{0}/{0}.00{1:03}.au'.format(GENRES,fileLen)
    x, fs = librosa.load(path, sr=None)
    ansKey = int(open('gtzan_key-master/gtzan_key/genres/{0}/{0}.00{1:03}.lerch.txt'.format(GENRES,fileLen),'r').read())

    
    if x.dtype != 'float32': # deal with the case of integer-valued data
        x = np.float32(x/32767.)
    if x.ndim > 1: # I want to deal only with single-channel signal now
        x = np.mean(x, axis = 1)
    
    Chroma = librosa.feature.chroma_stft(y=x, sr=fs)
    Chroma = Chroma/np.tile(np.sum(np.abs(Chroma)**2, axis=0)**(1./2),(Chroma.shape[0], 1))
    #Q1###
    GAMA = 1
    Chroma = np.log10(1+GAMA *np.abs(Chroma))
    #spectral smoothing
    """
    Len = 5
    for i in range(Chroma.shape[1]):
        Chroma[:,i] = np.sum(Chroma[:,i- int(Len/2):int(i+Len/2)],axis = 1)/Len
    """
    
    sumChroma =np.sum(Chroma,axis = 1)
    
    #substract mean x - x bar
    sumChroma -= np.sum(sumChroma) /sumChroma.shape[0]
    temp_ma = Template[0]
    temp_ma -= np.sum(temp_ma)/12
    temp_mi=Template[12]
    temp_mi -= np.sum(temp_mi)/12
    MajorCof = np.dot(temp_ma,sumChroma)/np.sqrt(np.multiply(np.dot(sumChroma,sumChroma),np.dot(temp_ma,temp_ma)))
    MinCof = np.dot(temp_mi,sumChroma)/np.sqrt(np.multiply(np.dot(sumChroma,sumChroma),np.dot(temp_mi,temp_mi)))
    for i in range(1,12):
        temp_ma = Template[i]
        temp_ma -= np.sum(temp_ma)/12
        temp_mi=Template[i+12]
        temp_mi -= np.sum(temp_mi)/12
        MajorCof = np.append(MajorCof,np.dot(temp_ma,sumChroma)/np.sqrt(np.multiply(np.dot(sumChroma,sumChroma),np.dot(temp_ma,temp_ma))))
         
        MinCof = np.append(MinCof,np.dot(temp_mi,sumChroma)/np.sqrt(np.multiply(np.dot(sumChroma,sumChroma),np.dot(temp_mi,temp_mi))))
    cof = MajorCof
    cof = np.append(cof,MinCof)
    key = np.argmax(cof)
    if(ansKey == -1):
        music_pieces -=1
        continue
    if(ansKey == 0):
        ansKey = 9
    elif(ansKey == 1):
        ansKey = 10
    elif (ansKey == 2):
        ansKey = 11
    elif(ansKey == 12):
        ansKey = 21
    elif (ansKey == 13):
        ansKey = 22
    elif (ansKey == 14):
        ansKey = 23
    else :
        ansKey = ansKey-3


    print(key,Key[key])
    if key == ansKey:
        acc = acc+1
    elif (key+7)%12 == ansKey:
        acc = acc + 0.5
        print("fifthPerfect")
    elif (key + 9)%12+12 == ansKey:   
        acc = acc + 0.3
        print("Relative")
    elif key + 12 == ansKey:
        acc = acc + 0.2
        print("Parallel")         
    print("Ans: ",Key[ansKey])
print("#music_pieces: ",music_pieces)
if music_pieces !=0:
    print('accuracy: ',acc/music_pieces)
