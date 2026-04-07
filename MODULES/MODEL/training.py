# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 17:46:07 2020

@author: 110137
"""

import tensorflow as tf
# from MODULES.MODEL.submodels_creation import submodels
from MODULES.MODEL.model_creation import DeepClassifier_model_creation
from MODULES.POSTPROCESSING.postprocessing import plot_loss_evolution
import numpy as np
from tensorflow.keras import backend as K
import os
import pandas as pd
import json
from tensorflow.keras import optimizers, models, utils
# from visualkeras import layered_view

# from src.models import generate_forward_model, generate_inverse_model
###########################################################################################

class training_info_initializer():
    #definicion e inicialización de las variables que van dentro 
    def __init__(self, n_epoch = 20, batch_size = 512, LR = 1e-03, metrics = "accuracy", loss = "categorical_crossentropy", shuffle = True, train_flag = True,):        
        self.n_epoch = n_epoch
        self.Batch_size = batch_size
        self.LR = LR
        self.Metrics = metrics
        self.Loss = loss
        self.Shuffle = shuffle
        self.train_flag = train_flag
        if n_epoch == None or batch_size == None or LR == None or loss == None:
            print("************************************************************************")
            print("Please initialize the info for TRAINING!")
            print("************************************************************************")
            quit()


def training_model(arch_info, train_info, Xtrain_resc, Xval_resc, Ytrain, Yval, folder_name):
    Input_layer, model_DeepClassifier = DeepClassifier_model_creation(arch_info)
    model_DeepClassifier.compile(metrics=[train_info.Metrics], loss= train_info.Loss, optimizer = tf.keras.optimizers.Adam(learning_rate = train_info.LR)) #Define the optimizer (SGD, RMSprop, Adam, adaline*) 
    model_DeepClassifier.summary()
    filepath = os.path.join('Output',folder_name,'best_model.hdf5')
    my_callbacks = [ tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=1200),tf.keras.callbacks.ModelCheckpoint(filepath, save_best_only=True, monitor="val_loss")]
    train_flag = train_info.train_flag
    DeepClassifier_history  = []
    if train_flag == True:   
        DeepClassifier_history = model_DeepClassifier.fit(Xtrain_resc, Ytrain, epochs = train_info.n_epoch, batch_size = train_info.Batch_size, callbacks = my_callbacks, shuffle = train_info.Shuffle, validation_data = (Xval_resc, Yval))
        hist_path = os.path.join('Output',folder_name, 'Loss'+'.npy')
        np.save(hist_path, DeepClassifier_history)
        model_DeepClassifier.save_weights(os.path.join('Output', folder_name,"model_weights.h5"))

    return model_DeepClassifier, DeepClassifier_history


# def training_model(arch_info, train_info, Xtrain_resc, Xval_resc, Ytrain, Yval, filename, folder_name):
#     Input_layer, model_DeepClassifier = DeepClassifier_model_creation(arch_info)
#     model_DeepClassifier.compile(metrics=[train_info.Metrics], loss= train_info.Loss, optimizer = tf.keras.optimizers.Adam(learning_rate = train_info.LR)) #Define the optimizer (SGD, RMSprop, Adam, adaline*) 
#     model_DeepClassifier.summary()
#     filepath = os.path.join('Output',folder_name,'best_model.hdf5')
#     my_callbacks = [ tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=1200),tf.keras.callbacks.ModelCheckpoint(filepath, save_best_only=True, monitor="val_loss")]
#     #other callb  tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=20, min_lr=0.0001)
#     # Callbacks = tf.keras.callbacks.ModelCheckpoint(filepath, save_best_only=True, monitor="val_loss")
#     DeepClassifier_history = model_DeepClassifier.fit(Xtrain_resc, Ytrain, epochs = train_info.n_epoch, batch_size = train_info.Batch_size, callbacks = my_callbacks, shuffle = train_info.Shuffle, validation_data = (Xval_resc, Yval))
#     # plot_loss_evolution(DeepClassifier_history,filename, folder_name)
#     hist_path = os.path.join('Output',folder_name, 'loss_ev'+filename+'.npy')
#     np.save(hist_path, DeepClassifier_history)
#     return model_DeepClassifier, DeepClassifier_history


##################################################################################################################################
#SaVE AND SET WEIGHTS 

def save_weights(model, arch_name):
    # Guardar configuración JSON en el disco
    json_config = model.to_json()
    with open(os.path.join('weights', 'model_config_'+arch_name+'.json'), 'w') as json_file:
        json_file.write(json_config)
    # Guardar pesos en el disco
    model.save_weights(os.path.join('weights', arch_name+'.h5'))

def load_weights_and_model(arch_name):
    # Recargue el modelo de los 2 archivos que guardamos
    with open(os.path.join('weights', 'model_config_'+arch_name+'.json'), 'r') as json_file:
        json_config = json_file.read()
    
    model = models.model_from_json(json_config)
    model.load_weights(os.path.join('weights', arch_name+'.h5'))
    return model

def save_weights2(model, arch_name):
    weights = []
    for layer in range(0, len(model.layers)):
        weights.append(model.layers[layer].get_weights())

    df = pd.DataFrame(weights)
    df.to_csv(os.path.join('weights', arch_name+'.csv'), index=False)
    #with open(os.path.join('weights',arch_name+'.txt'), 'wb') as f:
    #    np.savetxt(f, weights)

def set_weights2(model, arch_name):

    w = pd.read_csv(os.path.join('weights'), arch_name+'.csv', header=None)
    with open(os.path.join('weights',arch_name+'.npy'), 'r') as f:
        weights = np.load(f)

    print(weights)
    for layer in range(0, len(model.layers)):
        model.layers[layer].set_weights(weights[layer])
# CUSTOM LOSSES DEFINITION 
def custom_loss(y_actual,y_pred):
    loss = tf.math.reduce_mean(tf.math.square(y_actual - y_pred), axis= None)
    return loss


def orthogonality_loss(y_actual,y_pred,W_e,W_d):
    Me = np.dot(W_e,W_e.T)
    Md = np.dpt(W_d, W_d.T)
    I = np.identity(Me.shape[0])
    
    loss_pred = tf.math.reduce_mean(tf.math.square(y_actual - y_pred), axis= None)
    loss_We = K.sqrt(K.sum(K.square(Me-I)))
    loss_Wd =  K.sqrt(K.sum(K.square(Md-I)))
    loss_orthogonality = loss_We + loss_Wd
    loss = loss_pred + loss_orthogonality
                 
    return loss