#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 14:31:02 2023

@author: afernandez
"""
#import packages
import os
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from MODULES.PREPROCESSING.preprocessing_tools import import_data, rem_zeros, standardization, rescaling, mode_UnitNormalizer, convert_to_npy
from MODULES.PREPROCESSING.LOAD_DATA import import_excel_file

#############################################################################################################################

# #creating the .npy files
# #create npy data: do this once and then comment (for loading the .mat files and transforming them into .npy)
# def generate_npy_files():
#     mat_directory = os.path.join("Data", "Data_current_matfiles")
#     output_folder = os.path.join("Data", "Data_current_npy_files")
#     header_label = "savevar" #depedens on the matlab file creation
#     convert_to_npy(mat_directory, output_folder, header_label)
    
# generate_npy_files()
################################################################

#Solve the preprocessing: obtain std datasets for train/val/test/dam_test
def load_data(n_features,n_steps):
    sheet_name = 'Hoja1'
    excel_directory = os.path.join('Data', 'Data_current_excel_files')
    filenames = []
    for fname in os.scandir(excel_directory):
        filenames.append(str(fname)[11:-2])#To cut the string and extract only the text of interest
        
    filenames = sorted(filenames) 
    Data = np.array([import_excel_file(os.path.join(excel_directory, fname), sheet_name)for fname in filenames])  
    #Cut or round the data to match with the n_steps as an exact number of samples according to n_steps. 
    #We do this since we are not considering variable window lenght. Otherwise this would be different
    K = n_steps*(round(Data.shape[1]/n_steps)-1) #We do this to use any n_steps for the segments
    Dataf = np.zeros(shape = (Data.shape[0], K, Data.shape[2]))
    for i in range(Data.shape[0]):
        Dataf[i,:,:] = Data[i,0:K,:]
    
    Dataf =  Dataf[:,0:500,:]
    return Dataf

        
# def load_data(n_features, n_steps):
#     npy_directory = os.path.join('Data','Data_current_npy_files')
#     filenames = []
#     for fname in os.scandir(npy_directory):
#         filenames.append(str(fname)[11:-2])#To cut the string and extract only the text of interest
#     filenames = sorted(filenames) 
#     # Data = np.array([np.load(os.path.join("Data","Data_current_npy_files", fname))[2*33001:,:]for fname in filenames]) #removing the lowest wind levels    
#     Data = np.array([np.load(os.path.join(npy_directory, fname))for fname in filenames])   
#     # Data = Data[1:,:] #Hopefully, we are removing the healthy dataset since there are healthy scenarios included in the faulty datasets.

#     #Cut or round the data to match with the n_steps as an exact number of samples according to n_steps. 
#     #We do this since we are not considering variable window lenght. Otherwise this would be different
#     K = n_steps*(round(Data.shape[1]/n_steps)-1) #We do this to use any n_steps for the segments
#     Dataf = np.zeros(shape = (Data.shape[0], K, Data.shape[2]))
#     for i in range(Data.shape[0]):
#         Dataf[i,:,:] = Data[i,0:K,:]
    
#     return Dataf

def segment_data_overlapping(Dataf, n_features, n_steps, nslide):
    #This function takes the data and divides it into segments with a certain overlapping
    N = Dataf.shape[1]
    Data = []
    for i in range(Dataf.shape[0]):
        for j in range(int((N-n_steps)/nslide)): 
            segment = Dataf[i,j*nslide: j*nslide+n_steps,:]
            Data.append(segment)
    Data = np.array(Data)
    #once we have divided the dataset into segments, we can randomly select some of the to reduce data volume:
    # random_rows = np.random.randint(0, Data.shape[0], 400000)
    # Data   = Data[random_rows, :, :]
    return Data
    
    

def select_segments(Data,n_features,n_steps):
    # DataF = []
    # # We need to keep only those segments that start with healthy label 
    # for i in range(Data.shape[0]):
        
    #     if Data[i, 0, n_features] == 0.0 or Data[i,n_steps-1,n_features] != 0.0: 
    #     # if Data[i,0,n_features] != 0.0 and Data[i,n_steps-1, n_features] != 0.0:
    #         DataF.append(Data[i,:,:])
    # DataF = np.array(DataF)
    DataF = Data
    
    # # #Assign a 2step classification label: 
    # for i in range(DataF.shape[0]):
    #     for j in range(DataF.shape[1]):
    #         if DataF[i,j,n_features] != 0.0: 
    #             DataF[i,j,n_features] = 1.0
    return DataF

def correct_labels01(DataF,n_features,n_steps):
    for i in range(DataF.shape[0]):
        for j in range(DataF.shape[1]):
            if DataF[i,j,n_features] == 0.0: 
                DataF[i,j,n_features] = 0.0
            else: 
                DataF[i,j,n_features] = 1.0
                
    
    return DataF
                
            

def Data_XYseparation(DataF, n_features, n_steps, nfp):
    #Includes the one hot encoding task 
    #OneHotEncoding classes:
    if n_steps > 1:
        X_dataF = DataF[:,:,(1,2,3,4)]                      
        #Can be applied in general, but mostly employed if healthy points coexist in damaged datasets. 
        Labels = np.zeros(shape = (DataF.shape[0],1))
        #TRy to classify as faulty a sample that contains at least nfp points with fault. 
        for j in range (DataF.shape[0]):
            if np.count_nonzero(DataF[j,:,n_features+1]) > nfp:
                # ind = np.nonzero(DataF[j,:,n_features]) #find the nonzero elements and their vlaue
                # fval = DataF[j,:,n_features][ind][0] #to take the first  value only since they are all the same 
                Labels[j,:] = 1.0 #assign the fault ID value to the Labels array    
            else:
                Labels[j,:] = 0.0
            # most_prob = np.bincount(DataF[j,:,n_features].astype(int)).argmax()
            # Labels[j,:] = most_prob 
        # Labels = DataF[:,n_steps-1,n_features].reshape(-1,1) #This is when only damaged points coexist. 
        OHE_model =  OneHotEncoder(sparse = False)
        Y_dataF = OHE_model.fit_transform(Labels)

        # #******# If we want to classify ONLY the Faults - remove healthy segments
    #     XdataF, YdataF  = [], []
    #     for i in range(X_dataF.shape[0]):
    #         if Y_dataF[i,0] == 0.0 :
    #             XdataF.append(X_dataF[i,:,:])
    #             YdataF.append(Y_dataF[i,:])
    #     X_dataF = np.array(XdataF)
    #     Y_dataF = np.array(YdataF)
    #     Y_dataF = Y_dataF[:,1:]
    # else:
    #     X_dataF = DataF[:,0:n_features]                      
    #     Labels = np.zeros(shape = (DataF.shape[0],1))
    #     for i in range(Labels.shape[0]):
    #         Labels[i,:] = DataF[i,n_features]
    #     # Labels = DataF[:,1,n_features].reshape(-1,1)
    #     OHE_model =  OneHotEncoder(sparse = False)
    #     Y_dataF = OHE_model.fit_transform(Labels)
        
        # # If we want to classify only the Faults - remove healthy segments
        # XdataF, YdataF  = [], []
        # for i in range(X_dataF.shape[0]):
        #     if Y_dataF[i,0] == 0.0 :
        #         XdataF.append(X_dataF[i,:])
        #         YdataF.append(Y_dataF[i,:])
        # X_dataF = np.array(XdataF)
        # Y_dataF = np.array(YdataF)
        # Y_dataF = Y_dataF[:,1:]
        

        
    return X_dataF, Y_dataF
    

def split_TrainValTest(X_dataF, Y_dataF):
    #Import  data
    X_train, Xtest, y_train, Ytest = train_test_split(X_dataF, Y_dataF, test_size=0.1, random_state=1234) #Traing(train +val) + test split
    Xtrain, Xval, Ytrain, Yval = train_test_split(X_train, y_train, test_size = 0.20, random_state=1234) #Training and validation split
    return Xtrain, Xval, Xtest, Ytrain, Yval, Ytest

def Data_scaler(Xtrain, n_steps):
    #to REscale in the interval [0.5, 1.5] the expression is for each variable: (x_k -xmin)/(xmax-xmin)+0.5 k = number of samples 
    A = (1.5-0.5)/2
    Xmins = np.min(Xtrain, axis = 0)
    Xmaxs = np.max(Xtrain, axis = 0)
    if n_steps > 1:
        Xmins = np.min(Xmins, axis =0)
        Xmaxs = np.min(Xmaxs, axis =0)  
    return Xmins, Xmaxs, A

def Data_rescaling(X_data, Xmins,Xmaxs,A):
    X_data_resc = np.zeros(shape = X_data.shape)
    for i in range(X_data.shape[0]):
        X_data_resc[i,:] = (X_data[i,:]- Xmins)/(Xmaxs - Xmins)+ A 
    return X_data_resc
#################################################################################################################################
def preprocessing_interface(n_features,n_steps, nslide, nfp):
    Data = load_data(n_features, n_steps)
    if n_steps > 1:
        Dataf = segment_data_overlapping(Data, n_features, n_steps, nslide)
    else:
        Dataf = Data.reshape(Data.shape[0]*Data.shape[1], Data.shape[2])
    
    DataF = select_segments(Dataf, n_features, n_steps)# select_segments() 
    DataF = correct_labels01(DataF, n_features, n_steps)
    X_dataF, Y_dataF = Data_XYseparation(DataF, n_features, n_steps, nfp)
    Xtrain, Xval, Xtest, Ytrain, Yval, Ytest = split_TrainValTest(X_dataF,Y_dataF)
    # Xtrain = Xtrain[0:300000,:,:] #activate for single fault classification
    # Ytrain = Ytrain[0:300000,:]
    # Xval = Xval[0:100000,:,:] #activate for single fault classification
    # Yval = Yval[0:100000,:]
    # Xval = Xval[0:30000,:] #For deepclassifier
    # Yval = Yval[0:30000,:]
    Xmins,Xmaxs, A = Data_scaler(Xtrain, n_steps)
    Xtrain_resc = Data_rescaling(Xtrain,Xmins,Xmaxs,A)
    Xval_resc = Data_rescaling(Xval,Xmins,Xmaxs,A)
    Xtest_resc = Data_rescaling(Xtest,Xmins,Xmaxs,A)
    return Xtrain_resc, Xval_resc, Xtest_resc, Ytrain, Yval, Ytest, Xmins,Xmaxs,A






