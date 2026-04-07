# -*- coding: utf-8 -*-
"""
Created on Tue May 19 18:01:11 2020

@author: 109457
"""
import tensorflow as tf
import numpy as np
from MODULES.MODEL.model_creation import architecture_info_initializer
from MODULES.MODEL.training import training_info_initializer, training_model, custom_loss
from MODULES.PREPROCESSING.preprocessing import preprocessing_interface
from MODULES.PREPROCESSING.preprocessing_tools import convert_to_npy
from MODULES.POSTPROCESSING.postprocessing import postprocessing_info_initializer, postprocessing_interface
from MODULES.POSTPROCESSING.Classification_results import make_predictions, classification_metrics, confusion_matrix, classification_results
import os

tf.keras.backend.clear_session()
tf.random.set_seed(4321)
# configure to run_functions in eagerly or not True only recomended for debug
# tf.config.run_functions_eagerly(True)
# ##############################################################################################


# INITIAL CONSIDERATIONS

n_s  = [4]

for j in range(len(n_s)):
    # INITIAL CONSIDERATIONS
    filename = "08May_3sensors_20Hz"
    Wts = 3#Numer of Wind turbines in the farm 
    n_features = 38  # number of input variables/features without wind vars
    n_steps =n_s[j] #at 5Hz of sampling freq. this corresponds to 4 seconds. And at 100 Hz --> 0.2 s
    # The number of steps (segment length) conditions the COnvolution Operations
    nol = round(n_steps/2) # number of overlapping points in the windows/segments
    epochs = 500
    batch_s = 1024
    LRate =1e-04
    Architecture = "Arch_3" 
    folder_name = filename + '_Farm'+str(Wts)+'WTs_'+str(n_features)+'features_'+Architecture+'_'+str(n_steps)+'steps_'+str(epochs)+'epoch_'+str(batch_s)+'batch'+str(LRate)+'LR'
    Xtrain_resc, Xval_resc, Xtest_resc, Ytrain, Yval, Ytest, Xmins, Xmaxs, A = preprocessing_interface(
        n_features, n_steps, nol)
    
    # Initialize architecture properties and select the architectures FROM the DICTIONARY
    architecture_info = architecture_info_initializer(
        DeepClassifier=Architecture,
        # input_dim = Xtrain_resc.shape[1], #for standard DNN
        input_dim=(Xtrain_resc.shape[1], Xtrain_resc.shape[2]),  # for 1D_CNN
        # input_dim =(Xtrain_resc.shape[1],Xtrain_resc.shape[2],1), #For 2D CNN
        # input_dim = (None, Xtrain_resc.shape[1],Xtrain_resc.shape[2]), #for CNNLSTM
        enc_dim=Ytrain.shape[1])
    
    ####################################################################################################################################
    # BUILD AND TRAIN THE MODELS
    # Initiailize the training properties
    training_info = training_info_initializer(
        n_epoch=epochs,
        batch_size=batch_s,
        LR=LRate,
        metrics="accuracy",
        loss="categorical_crossentropy",
        shuffle=True)
    # Build and train the models
    model, history = training_model(
        architecture_info, training_info, Xtrain_resc, Xval_resc, Ytrain, Yval, filename, folder_name)
    
    ###########################################################################################################################################################
    #POSTPROCESSING - RESULTS 
    #You must learn to load the model from its location and evaluate it 
    # model = tf.keras.models.load_model(os.path.join("Output",folder_name,"best_model.hdf5"))
    ######################################### MODEL EVALUATION #########################################
    target_names = ['Healthy', 'F1WT1', 'F2WT1','F3WT1', 'F1WT2', 'F2WT2','F3WT2', 'F1WT3', 'F2WT3', 'F3WT3']
    # target_names = ['Healthy', 'F1WT1', 'F2WT1','F3WT1', 'F4WT1',' F5WT1', 'F1WT2', 'F2WT2','F3WT2', 'F4WT2',' F5WT2', 'F1WT3', 'F2WT3','F3WT3', 'F4WT3',' F5WT3' ]
    # target_names = ['Healthy','F3WT2','F4WT2']
    # target_names = ['Healthy', 'F1WT2', 'F2WT2','F3WT2','F4WT2', 'F5WT2']
    # target_names = ['Healthy', 'F1WT1', 'F2WT1','F3WT1', 'F4WT1',' F5WT1']
    # target_names = ['Healthy', 'F1WT1', 'F2WT1','F3WT1', 'F4WT1',' F5WT1', 'F1WT2', 'F2WT2','F3WT2', 'F4WT2',' F5WT2']
    # target_names = ['Healthy', 'F1WT1', 'F2WT1','F3WT1', 'F4WT1',' F5WT1', 'F1WT2', 'F2WT2','F3WT2', 'F4WT2',' F5WT2', 'F1WT3', 'F2WT3','F3WT3', 'F4WT3',' F5WT3' ]
    target_names = ['Healthy', 'F1WT1', 'F2WT1','F3WT1','F1WT2', 'F2WT2','F3WT2', 'F1WT3', 'F2WT3','F3WT3']
    fnames = ['Train', 'Validation', 'Test']
    XJoint = [Xtrain_resc, Xval_resc, Xtest_resc]
    YJoint = [Ytrain, Yval, Ytest]
    
    for i in range(len(fnames)):
        fname = fnames[i]
        X = XJoint[i]
        y_true = YJoint[i]
        
        classification_results(model, X, y_true, target_names, folder_name, fname)
    
















#############################################################################################################################################################
# from tensorflow.python.client import device_lib
# print(device_lib.list_local_devices())
