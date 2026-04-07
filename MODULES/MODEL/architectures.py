# -*- coding: utf-8 -*-
"""
Created on Tue May 19 19:48:49 2020

@author: 109457
"""
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np

import sklearn
from sklearn import datasets
import numpy as np
from sklearn import decomposition
import scipy
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, Layer, InputSpec
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras import regularizers, activations, initializers, constraints, Sequential
from tensorflow.keras import backend as K
from tensorflow.keras.constraints import UnitNorm, Constraint
# import tensorflow_probability as tfp 
# tfd = tfp.distributions
# tfpl = tfp.layers
#############SOME LAYER EXAMPLES USING CLASS ################################################################
class Linear_encoder(tf.keras.layers.Layer):
  def __init__(self, layer_output_dim, **kwargs):
    super(Linear_encoder, self).__init__()
    self.layer_output_dim = layer_output_dim

  def build(self, input_shape):
      A = np.array([[0.5361779 , 0.10770221],
                          [ 0.4954218 , -0.53083354 ],
                          [-0.50844705 , 0.30159527], 
                          [ 0.4570274 ,  0.78462416 ]],dtype="float32") #For MEXICO
      b = np.array([[0.1,0.1,0.1,0.2]], dtype = 'float32')
      self.A = tf.Variable(
            initial_value=A,
            trainable=False,
        )
      self.b = tf.Variable(initial_value = b, trainable = False)
      
  def call(self, inputs_layer):
    print()
    return tf.matmul(inputs_layer, self.A)+self.b


class MyDenseLayer(tf.keras.layers.Layer):
  def __init__(self, layer_output_dim, **kwargs):
    super(MyDenseLayer, self).__init__()
    self.layer_output_dim = layer_output_dim

  def build(self, input_shape):
    self.w = self.add_weight("weights", trainable=True, shape=[int(input_shape[-1]), self.layer_output_dim])
    self.b = self.add_weight("bias", trainable=True, shape=[self.layer_output_dim])

  def call(self, inputs_layer):
    print()
    return tf.matmul(inputs_layer, self.w) + self.b, self.w
###################################################################################################################
def architecture_DeepRegressor(input_dim):
    input1 = tf.keras.Input(shape =(input_dim), name = 'Classifier_input')
    lay1 = layers.Dense(20, activation = 'relu', name = 'H1')(input1) #Intermediate layers
    lay2 = layers.Dense(10, activation = 'relu', name = 'H2')(lay1) #Intermediate layers
    lay3 = layers.Dense(10, activation = 'relu', name = 'H3')(lay2) #Intermediate layers
    lay4 = layers.Dense(10, activation = 'relu')(lay3) #Intermediate layers
    output1 =tf.keras.layers.Dense(20, activation = 'relu' , name = 'output1')(lay4)
    return input1, output1

def architecture_DeepClassifier(input_dim, enc_dim):
    input1 = tf.keras.Input(shape =(input_dim), name = 'Classifier_input')
    lay1 = layers.Dense(200, activation = 'relu', name = 'H1')(input1) #Intermediate layers
    lay2 = layers.Dense(200, activation = 'relu', name = 'H2')(lay1) #Intermediate layers
    lay3 = layers.Dense(200, activation = 'relu', name = 'H3')(lay2) #Intermediate layers
    lay4 = layers.Dense(200, activation = 'relu')(lay3) #Intermediate layers
    lay4 = layers.Dense(200, activation = 'relu')(lay4) #Intermediate layers
    output1 =tf.keras.layers.Dense(enc_dim, activation = 'softmax' , name = 'output1')(lay4)
    return input1, output1

def architecture_1D_CNNClassifier(input_dim, enc_dim):  #Architecture that is currently working
    input1 = tf.keras.Input(shape =( input_dim), name = "CNN_input")
    conv1 = layers.Conv1D(filters=36, kernel_size=16, activation = 'relu', padding="same", name = 'Convo1')(input1)
    conv1 = layers.BatchNormalization()(conv1)
    
    conv3 = layers.Conv1D(filters=64, kernel_size=5, activation = 'relu', padding="same", name = 'Convo3')(conv1)
    conv3 = layers.Dropout(0.5)(conv3)
    conv3 = layers.BatchNormalization()(conv3)
    conv3 = layers.GlobalMaxPooling1D()(conv3)
    conv3 = layers.Flatten()(conv3)

    lay2 = layers.Dense(100, activation = 'relu', name = 'H1')(conv3) #Intermediate layers
    output1 =tf.keras.layers.Dense(enc_dim, activation = 'softmax' , name = 'output1')(lay2)
    
    return input1, output1

def architecture_1D_CNNClassifier_1F(input_dim, enc_dim): #Architecture to make tests and trials
    input1 = tf.keras.Input(shape =( input_dim), name = "CNN_input")
    conv1 = layers.Conv1D(filters=50, kernel_size=16, activation = 'relu', padding="same", name = 'Convo1')(input1)
    conv1 = layers.BatchNormalization()(conv1)
    conv2 = layers.Conv1D(filters=50, kernel_size=12, activation = 'relu', padding="same", name = 'Convo2')(conv1)
    conv2 = layers.BatchNormalization()(conv2)
    conv3 = layers.Conv1D(filters=50, kernel_size=9, activation = 'relu', padding="same", name = 'Convo3')(conv2)
    conv3 = layers.BatchNormalization()(conv3)
    conv4 = layers.Conv1D(filters=50, kernel_size=14, activation = 'relu', padding="same", name = 'Convo4')(conv3)
    conv4 = layers.BatchNormalization()(conv4)   
    conv5 = layers.Conv1D(filters=50, kernel_size=7, activation = 'relu', padding="same", name = 'Convo5')(conv4)
    conv5 = layers.BatchNormalization()(conv5)   
    conv6 = layers.Conv1D(filters=50, kernel_size=7, activation = 'relu', padding="same", name = 'Convo6')(conv5)
    conv6 = layers.BatchNormalization()(conv6)   
    conv7 = layers.Conv1D(filters=50, kernel_size=11, activation = 'relu', padding="same", name = 'Convo7')(conv6)
    conv7 = layers.BatchNormalization()(conv7)   
    convf = layers.Conv1D(filters=50, kernel_size=18, activation = 'relu', padding="same", name = 'Convof')(conv7)
    convf = layers.BatchNormalization()(convf)
    convf = layers.GlobalMaxPooling1D()(convf)
    convf = layers.Flatten()(convf)
    lay2 = layers.Dense(100, activation = 'relu', name = 'H1')(convf) #Intermediate layers
    output1 =tf.keras.layers.Dense(enc_dim, activation = 'softmax' , name = 'output1')(lay2)
    # output1 =tf.keras.layers.Dense(1, activation = 'linear' , name = 'output1')(lay2)
    return input1, output1

def architecture_1D_CNNDetector_F2(input_dim, enc_dim): #Architecture to make tests and trials
    input1 = tf.keras.Input(shape =( input_dim), name = "CNN_input")
    conv1 = layers.Conv1D(filters=20, kernel_size=3, activation = 'relu', padding="same", name = 'C1')(input1)
    conv1 = layers.BatchNormalization()(conv1)
    conv2 = layers.Conv1D(filters=20, kernel_size=2, activation = 'relu', padding="same", name = 'Convo2')(conv1)
    conv2 = layers.BatchNormalization()(conv2)

    conv3 = layers.Conv1D(filters=20, kernel_size=5, activation = 'relu', padding="same", name = 'Convo3')(conv2)
    conv3 = layers.BatchNormalization()(conv3)
    convf = layers.Conv1D(filters=20, kernel_size=3, activation = 'relu', padding="same", name = 'Convof')(conv2)
    convf = layers.BatchNormalization()(convf)
    convf = layers.GlobalMaxPooling1D()(convf)
    convf = layers.Flatten()(convf)
    lay2 = layers.Dense(150, activation = 'relu', name = 'H1')(convf) #Dense hidden layer after flattening
    lay2 = layers.Dense(150, activation = 'relu')(lay2) #Dense hidden layer after flattening
    lay2 = layers.Dense(150, activation = 'relu')(lay2) #Dense hidden layer after flattening

    lay2 = layers.Dense(150, activation = 'relu')(lay2) #Dense hidden layer after flattening
    lay2 = layers.Dense(150, activation = 'relu')(lay2) #Dense hidden layer after flattening
    output1 =tf.keras.layers.Dense(enc_dim, activation = 'softmax' , name = 'output1')(lay2)
    return input1, output1


def architecture_1D_CNNClassifier_F2(input_dim, enc_dim): #Architecture to make tests and trials
    input1 = tf.keras.Input(shape =( input_dim), name = "CNN_input")
    conv1 = layers.Conv1D(filters=10, kernel_size=3, activation = 'relu', padding="same", name = 'Conv1')(input1)
    conv1 = layers.BatchNormalization()(conv1)
    conv2 = layers.Conv1D(filters=10, kernel_size=3, activation = 'relu', padding="same", name = 'Conv2')(conv1)
    conv2 = layers.BatchNormalization()(conv2)
    convf = layers.Conv1D(filters=10, kernel_size=3, activation = 'relu', padding="same", name = 'Conv3')(conv2)
    convf = layers.BatchNormalization()(convf)
    convf = layers.GlobalMaxPooling1D()(convf)
    convf = layers.Flatten()(convf)
    lay2 = layers.Dense(10, activation = 'relu', name = 'H1')(convf) #Dense hidden layer after flattening
    lay2 = layers.Dense(10, activation = 'relu')(lay2) #Dense hidden layer after flattening
    lay2 = layers.Dense(10, activation = 'relu')(lay2) #Dense hidden layer after flattening
    output1 =tf.keras.layers.Dense(enc_dim, activation = 'softmax' , name = 'output1')(lay2)
    return input1, output1


def architecture_1D_CNNClassifier_CAE(input_dim, enc_dim): #Architecture to make tests and trials
    input1 = tf.keras.Input(shape =( input_dim), name = "CNN_input")
    conv1 = layers.Conv1D(filters=10, kernel_size=11, activation = 'relu', padding="same", name = 'C1')(input1)
    conv1 = layers.MaxPooling1D(pool_size =2)(conv1)
    conv2 = layers.Conv1D(filters=10, kernel_size=5, activation = 'relu', padding="same", name = 'Convo2')(conv1)
    conv2 = layers.MaxPooling1D(pool_size = 2)(conv2)
    conv3 = layers.Dropout(0.5)(conv2)
    convf = layers.Flatten()(conv3)
    lay2 = layers.Dense(100, activation = 'relu', name = 'H1')(convf) #Dense hidden layer after flattening
    output1 =tf.keras.layers.Dense(enc_dim, activation = 'softmax' , name = 'output1')(lay2)
    return input1, output1

def architecture_1DCNN_LSTM(input_shape, enc_dim):
    input1 = tf.keras.Input(shape=input_shape, name="CNN_input")
    conv1 = layers.Conv1D(filters=20, kernel_size=3, activation='relu', padding="same", name='C1')(input1)
    conv1 = layers.BatchNormalization()(conv1)
    conv2 = layers.Conv1D(filters=20, kernel_size=5, activation='relu', padding="same", name='Convo2')(conv1)
    conv2 = layers.BatchNormalization()(conv2)
    conv3 = layers.Conv1D(filters=20, kernel_size=5, activation='relu', padding="same", name='Convo3')(conv2)
    conv3 = layers.BatchNormalization()(conv3)
    conv4 = layers.Conv1D(filters=20, kernel_size=5, activation='relu', padding="same", name='Convo4')(conv3)
    conv4 = layers.BatchNormalization()(conv4)
    conv5 = layers.Conv1D(filters=20, kernel_size=5, activation = 'relu', padding="same", name = 'Convo5')(conv4)
    conv5 = layers.BatchNormalization()(conv5)
    convf = layers.Conv1D(filters=20, kernel_size=5, activation='relu', padding="same", name='Convof')(conv5)
    convf = layers.BatchNormalization()(convf)
    convf = layers.GlobalMaxPooling1D()(convf)
    convf = layers.Flatten()(convf)
    lstm1 = layers.LSTM(32, return_sequences=True)(tf.expand_dims(convf, axis=-1))
    lstm2 = layers.LSTM(32)(lstm1)
    layf = layers.Dense(100, activation='relu', name='H1')(lstm2)
    output1 = layers.Dense(enc_dim, activation='softmax', name='output1')(layf)
    return input1, output1



def architecture_LSTM(input_dim, enc_dim):
    input1 = tf.keras.Input(shape =(input_dim), name = "LSTM_input")
    # Create the LSTM layer
    lstm = tf.keras.layers.LSTM(50)
    outputs = lstm(input1)

    # Flatten the LSTM outputs
    flattened_outputs = tf.keras.layers.Flatten()(outputs)
    # Create a fully connected layer with softmax activation for the final output
    lay4 = layers.Dense(250, activation = 'relu', name = 'H1')(flattened_outputs) #Intermediate layers
    lay4 = layers.Dense(100, activation = 'relu')(lay4) #Intermediate layers
    lay4 = layers.Dense(100, activation = 'relu')(lay4) #Intermediate layers
    lay4 = layers.Dense(100, activation = 'relu')(lay4) #Intermediate layers
    lay4 = layers.Dense(100, activation = 'relu')(lay4) #Intermediate layers

    output1 =tf.keras.layers.Dense(enc_dim, activation = 'softmax' , name = 'output1')(lay4)
    return input1, output1


def architecture_DeepRegressor(input_dim, enc_dim):
    input1 = tf.keras.Input(shape =(input_dim,), name = 'Classifier_input')
    lay1 = layers.Dense(32, activation = 'relu', name = 'H1')(input1) #Intermediate layers
    lay2 = layers.Dense(36, activation = 'relu', name = 'H2')(lay1) #Intermediate layers
    lay3 = layers.Dense(48, activation = 'relu', name = 'H3')(lay2) #Intermediate layers
    lay4 = layers.Dense(36, activation = 'relu')(lay3) #Intermediate layers
    lay5 = layers.Dense(48, activation = 'relu')(lay4) #Intermediate layers
    lay6 = layers.Dense(12, activation = 'relu')(lay5) #Intermediate layers
    output1 =tf.keras.layers.Dense(enc_dim,activation = 'linear', name = 'output1')(lay6)
    return input1, output1


#####################################################################################################



#function para que me devuelva diccionario de arquitecturas
def arch_dictionary():
    #Definir diccionariode arquitecturas
    architecture_dictionary = {
        "Arch_1": architecture_DeepClassifier,
        "Arch_2": architecture_1DCNN_LSTM,
        "Arch_3": architecture_1D_CNNClassifier_F2,
        "Arch_4": architecture_LSTM,
        "Arch_5": architecture_1D_CNNClassifier_CAE,
        "Arch_6": architecture_1D_CNNDetector_F2
        }
    return architecture_dictionary



######################################################################################################################
############################################ OLD STUFF ############################

# # ARchitecture that is working with OFfset but not with STUCK
# def architecture_1D_CNNClassifier_F2(input_dim, enc_dim): #Architecture to make tests and trials
#     input1 = tf.keras.Input(shape =( input_dim), name = "CNN_input")
#     conv1 = layers.Conv1D(filters=30, kernel_size=5, activation = 'relu', padding="same", name = 'C1')(input1)
#     conv1 = layers.BatchNormalization()(conv1)
#     conv2 = layers.Conv1D(filters=30, kernel_size=5, activation = 'relu', padding="same", name = 'Convo2')(conv1)
#     conv2 = layers.BatchNormalization()(conv2)
#     conv3 = layers.Conv1D(filters=30, kernel_size=5, activation = 'relu', padding="same", name = 'Convo3')(conv2)
#     conv3 = layers.BatchNormalization()(conv3)
#     conv4 = layers.Conv1D(filters=30, kernel_size=5, activation = 'relu', padding="same", name = 'Convo4')(conv3)
#     conv4 = layers.BatchNormalization()(conv4)
#     conv5 = layers.Conv1D(filters=10, kernel_size=5, activation = 'relu', padding="same", name = 'Convo5')(conv4)
#     conv5 = layers.BatchNormalization()(conv5)
#     # conv6 = layers.Conv1D(filters=10, kernel_size=5, activation = 'relu', padding="same", name = 'Convo6')(conv5)
#     # conv6 = layers.BatchNormalization()(conv6)
#     # conv7 = layers.Conv1D(filters=10, kernel_size=5, activation = 'relu', padding="same", name = 'Convo7')(conv6)
#     # conv7 = layers.BatchNormalization()(conv7)
#     # conv8 = layers.Conv1D(filters=10, kernel_size=5, activation = 'relu', padding="same", name = 'Convo8')(conv7)
#     # conv8 = layers.BatchNormalization()(conv8)
#     # conv9 = layers.Conv1D(filters=10, kernel_size=5, activation = 'relu', padding="same", name = 'Convo9')(conv8)
#     # conv9 = layers.BatchNormalization()(conv9)
#     conv10 = layers.Conv1D(filters=10, kernel_size=5, activation = 'relu', padding="same", name = 'Convo10')(conv5)
#     conv10 = layers.BatchNormalization()(conv10)
#     conv11 = layers.Conv1D(filters=10, kernel_size=5, activation = 'relu', padding="same", name = 'Convo11')(conv10)
#     conv11 = layers.BatchNormalization()(conv11)
#     convf = layers.Conv1D(filters=10, kernel_size=5, activation = 'relu', padding="same", name = 'Convof')(conv11)
#     convf = layers.BatchNormalization()(convf)
#     convf = layers.GlobalMaxPooling1D()(convf)
#     convf = layers.Flatten()(convf)
#     lay2 = layers.Dense(200, activation = 'relu', name = 'H1')(convf) #Dense hidden layer after flattening
#     output1 =tf.keras.layers.Dense(enc_dim, activation = 'softmax' , name = 'output1')(lay2)
#     return input1, output1
#Best architecture till now

#function para que me devuelva diccionario de arquitecturas
# def arch_dictionary():
#     #Definir diccionariode arquitecturas
#     architecture_dictionary = {
#         "arch_DeepClassifier": architecture_DeepClassifier,
#         "1D_CNN_classifier": architecture_1D_CNNClassifier_1F,
#         "arch_LSTM": architecture_LSTM,
#         "arch_2D_CNNClassifier": architecture_2D_CNNClassifierB,
#         "arch_DeepRegressor": architecture_DeepRegressor
#         }
#     return architecture_dictionary
