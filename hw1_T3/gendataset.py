# Using all xml file to generate dataset
# read the xml files in metainfo/
# generate a *.npz data in feature_label_data/

from util import parselabel,parseAnswer
from time import time
import numpy as np
import os

if __name__ == "__main__":
    # setting parameter for MFCCs
    n_mfcc = 13
    featuretype = 'mfcc_delta'
    # define the variables for feature and label
    if featuretype == 'mfcc_delta':
        train = np.empty((0,12))
        test = np.empty((0,12))
        valid = np.empty((0,12))
    elif featuretype == 'mfcc_cov':
        train = np.empty((0,12))
        test = np.empty((0,12))
        valid = np.empty((0,12))
    Y_train = np.empty((0,))
    Y_test = np.empty((0,))
    Y_valid = np.empty((0,))
    chroma = []
    # Use os.walk to go through all the audio file under folder "audio/"
    print("Start generating dataset with feature type " + featuretype + " ...")
    t0 = time()
    
    for dirPath, _, fileNames in os.walk("Train"):
        if fileNames == []:
            continue
        for f in fileNames:
            X = parselabel(os.path.join(
                os.path.join(dirPath, f)),featuretype)
            train = np.vstack((train, X))
            print(train.shape)
    
    for dirPath, _, fileNames in os.walk("Test"):
        if fileNames == []:
            continue
        for f in fileNames:
            X = parselabel(os.path.join(
                os.path.join(dirPath, f)),featuretype)
            test = np.vstack((test, X))
            print(test.shape)
    for dirPath, _, fileNames in os.walk("Valid"):
        if fileNames == []:
            continue
        for f in fileNames:
            X = parselabel(os.path.join(
                os.path.join(dirPath, f)),featuretype)
            valid = np.vstack((valid, X))
            print(valid.shape)   
    
    for dirPath, _, fileNames in os.walk("answer_train"):
        if fileNames == []:
            continue
        for f in fileNames:
            Y_train = np.hstack((Y_train,(parseAnswer(os.path.join(
                os.path.join(dirPath, f))))))
    for dirPath, _, fileNames in os.walk("answer_test"):
        if fileNames == []:
            continue
        for f in fileNames:
            Y_test = np.hstack((Y_test,(parseAnswer(os.path.join(
                os.path.join(dirPath, f))))))
    for dirPath, _, fileNames in os.walk("answer_valid"):
        if fileNames == []:
            continue
        for f in fileNames:
            Y_valid = np.hstack((Y_valid,(parseAnswer(os.path.join(
                os.path.join(dirPath, f))))))
    print(Y_train)
    print (train)
    print('Finished in {:4.2f} sec!'.format(time() - t0))
    print('Collect {:d} samples totally.'.format(test.shape[0]+train.shape[0]+valid.shape[0]))
    datasetNames = 'dataset_' + featuretype + '_' + str(n_mfcc) + '.npz'
    datasetpath = os.path.join('feature_label_data', datasetNames)
    print('Dataset is saved at "feature_label_data/'+ datasetNames +'".')
    
    np.savez(datasetpath, test=test,train=train,valid=valid,ans_train=Y_train,ans_test = Y_test,ans_valid = Y_valid)
    
