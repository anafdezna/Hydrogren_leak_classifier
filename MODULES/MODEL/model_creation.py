# -*- coding: utf-8 -*-
"""
Created on Wed May 20 08:45:54 2020

@author: 109457
"""
import tensorflow as tf
from MODULES.MODEL.architectures import arch_dictionary

class architecture_info_initializer():
    #definicion e inicialización de las variables q van dentro 
    def __init__(self, DeepClassifier = "arch_DeepClassifier", 
                 input_dim = None, enc_dim = None):        
        self.DeepClassifier = DeepClassifier
        self.input_dim = input_dim
        self.enc_dim = enc_dim
        if input_dim == None or enc_dim == None:
            print("************************************************************************")
            print("Please initialize input_dim and enc_dim in the architecture_info object!")
            print("************************************************************************")
            quit()


def model_features(arch_info):
    arch_d = arch_dictionary()
    Input_DeepClassifier, Output_DeepClassifier = arch_d[arch_info.DeepClassifier](arch_info.input_dim,arch_info.enc_dim)
    architecture_list = [Input_DeepClassifier, Output_DeepClassifier]
    return architecture_list

#create the  model
##########################################################################################################
def DeepClassifier_model_creation(arch_info):
    architecture_list = model_features(arch_info)
    input_layer = architecture_list[0]
    DeepClassifier = tf.keras.Model(input_layer, architecture_list[1], name = "DeepClassifier")
    output_DeepClassifier = DeepClassifier(input_layer)

    model_DeepClassifier = tf.keras.Model(inputs = input_layer, outputs = output_DeepClassifier, name = "model_DeepClassifier")
    return input_layer, model_DeepClassifier


# def models_creation(architecture_list):
#     input_layer = architecture_list[0]
#     linear_encoder =  tf.keras.Model(input_layer, architecture_list[1], name = 'encoder')
#     linear_decoder = tf.keras.Model(architecture_list[2],architecture_list[3], name = 'decoder')
#     nonlinear_encoder = tf.keras.Model(architecture_list[4], architecture_list[5], name = 'first_nonlinear_path')
#     nonlinear_decoder = tf.keras.Model(architecture_list[6], architecture_list[7], name = 'second_nonlinear_path')
#     return input_layer, linear_encoder, linear_decoder, nonlinear_encoder, nonlinear_decoder

# def submodels(arch_info):
#     #llamar a submodel features
#     architecture_list = submodel_features(arch_info)
#     #llamar submodels_Creation
#     input_layer, linear_encoder, linear_decoder, nonlinear_encoder, nonlinear_decoder = submodels_creation(architecture_list)
#     return input_layer, linear_encoder, linear_decoder, nonlinear_encoder, nonlinear_decoder



# def PCA_model_creation(input_layer, linear_encoder, linear_decoder):
#     output_encoder = linear_encoder(input_layer)
#     output_decoder = linear_decoder(output_encoder)
#     modelPCA_autoencoder = tf.keras.Model(inputs = input_layer, outputs = output_decoder, name = 'autoencoder')
#     return modelPCA_autoencoder

# ####################################################################################

# def ResNet_model_creation(input_layer, linear_encoder, linear_decoder, nonlinear_encoder, nonlinear_decoder):
#     output_linear_encoder = linear_encoder(input_layer)
#     output_nonlinear_encoder = nonlinear_encoder(input_layer)
#     output_encoder = output_linear_encoder + output_nonlinear_encoder
#     output_linear_decoder = linear_decoder(output_encoder)
#     output_nonlinear_decoder = nonlinear_decoder(output_encoder)
#     output_autoencoder = output_linear_decoder + output_nonlinear_decoder
#     modelResNet_autoencoder = tf.keras.Model(inputs = input_layer, outputs = output_autoencoder, name = 'autoencoder')
#     return modelResNet_autoencoder

