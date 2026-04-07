# -*- coding: utf-8 -*-
"""
Created on Thu May 21 09:18:33 2020

@author: 109457
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
#creating the .npy files
# #create npy data: do this once and then comment (for loading the .mat files and transforming them into .npy)
# def generate_npy_files():
#     mat_directory = os.path.join("Data", "Data_current_matfiles")
#     output_folder = os.path.join("Data", "Data_current_npy_files")
#     header_label = "savevar" #depedens on the matlab file creation
#     convert_to_npy(mat_directory, output_folder, header_label)
    
# generate_npy_files()
# ################################################################
# plt.figure()
# plt.plot(X[:,0], X[:,1:5], marker = '.')
# plt.legend(['Sensor 1', 'Sensor 2', 'Sensor 3', 'Sensor 4'])
# plt.xlabel("Time [s]", fontsize=14)
# plt.ylabel("H2 concentration (\%)", fontsize=14)
# plt.savefig(os.path.join("Output", "SEGURH2_results", "Signals_H2_old.png"), dpi = 500, bbox_inches='tight')


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
    
    Dataf = Dataf[:,0:300,:]# 310 when entire. selecting the part from the entire leak simulation that you want to use. 
    # Dataf = Dataf[:,0:250,:] # Time of approx. 80 secs. 
    # Dataf = Dataf[:,0:156,:] # Time of approx. 50 secs. 
    # Dataf = Dataf[:,0:63,:] # Time of approx. 20 secs. 
    
    
    # Selected_test_cases = [2,8,14,20,26,32] # same location of the leak for all teh six tanks
    # Selected_test_cases = [3,8,17,18,26,31] #randomly located leaks not the same location 
    Selected_test_cases = [2,7,14,18,26,31] #randomly located leaks not the same location 


    Dataf_test = Dataf[Selected_test_cases,:,:]
    Dataf1 = np.delete(Dataf, Selected_test_cases, axis=0)

    
    return Dataf1, Dataf_test


def segment_data_overlapping(Dataf, n_features, n_steps, nslide):
    N = Dataf.shape[1]
    Data = []
    for i in range(Dataf.shape[0]):
        for j in range(int((N-n_steps)/nslide)): 
            segment = Dataf[i,j*nslide: j*nslide+n_steps,:]
            Data.append(segment)
    Data = np.array(Data)
    nan_mask = np.isnan(Data).any(axis=(1, 2))

# Filter out rows with NaN values #REVIEW THIS 
    Data = Data[~nan_mask]
    # random_rows = np.random.randint(0, Data.shape[0], 400000)
    # Data   = Data[random_rows, :, :]
    return Data
    
    
def Data_XYseparation(DataF, n_features, n_steps, nfp, n_classes):

    X_dataF = DataF[:,:,(1,2,3,4)]       
    #Can be applied in general, but mostly employed if healthy points coexist in damaged datasets. 
    Y_dataF= np.zeros(shape = (DataF.shape[0],n_classes))
    #comment this if you want to consider the healthy state. this works for 6 classes (no healthy)
    for j in range (DataF.shape[0]):
        pos_label = DataF[j,0,n_features+1] #This works if no 0 (healthy) observations exist  
        Y_dataF[j,int(pos_label)-1] = 1    
   
    return X_dataF, Y_dataF
 


def split_TrainValTest(X_dataF, Y_dataF):
    #Import  data
    Xtrain, Xval, Ytrain, Yval  = train_test_split(X_dataF, Y_dataF, test_size=0.2, random_state=4321) #Traing(train +val) + test split
    return Xtrain, Xval, Ytrain, Yval   

# def split_TrainValTest(X_dataF, Y_dataF):
#     #Import  data
#     X_train, Xtest, y_train, Ytest = train_test_split(X_dataF, Y_dataF, test_size=0.1, random_state=4321) #Traing(train +val) + test split
#     Xtrain, Xval, Ytrain, Yval = train_test_split(X_train, y_train, test_size = 0.15, random_state=1) #Training and validation split
#     return Xtrain, Xval, Xtest, Ytrain, Yval, Ytest

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
def preprocessing_interface(n_features,n_steps, nslide, nfp, n_classes, folder_name):
    Data, Data_test = load_data(n_features, n_steps)
    if n_steps > 1:
        DataF = segment_data_overlapping(Data, n_features, n_steps, nslide)
        Datatest = segment_data_overlapping(Data_test, n_features, n_steps, nslide)
    else:
        DataF = Data.reshape(Data.shape[0]*Data.shape[1], Data.shape[2])
    
    # if n_steps > 1:
    #     DataF = select_segments(Dataf, n_features, n_steps)# select_segments() 
    # else:
    #     DataF = Dataf
    
    
    X_data, Y_data = Data_XYseparation(DataF, n_features, n_steps, nfp, n_classes)
    X_dataF, Y_dataF = X_data, Y_data
    
    Xtest, Ytest = Data_XYseparation(Datatest, n_features, n_steps, nfp, n_classes)

    Xtrain, Xval, Ytrain, Yval = split_TrainValTest(X_dataF,Y_dataF)

    Xtrain_resc, Xval_resc, Xtest_resc = Xtrain*1.5, Xval*1.5, Xtest*1.5 #We use a basic rescaling 
    
    # Typical rescaling in the interval [a,b]:
    # Xtrain_resc = Data_rescaling(Xtrain,Xmins,Xmaxs,A)
    # Xval_resc = Data_rescaling(Xval,Xmins,Xmaxs,A)
    # Xtest_resc = Data_rescaling(Xtest,Xmins,Xmaxs,A)
    Datafiles = {'Xtrain_resc': Xtrain_resc,
            'Xval_resc': Xval_resc,
            'Xtest_resc': Xtest_resc,
            'Ytrain': Ytrain,
            'Yval': Yval,
            'Ytest': Ytest
            }
    
    np.save(os.path.join('Output', folder_name, 'Datafiles_10Jan.npy'), Datafiles, allow_pickle=True)
    return Xtrain_resc, Xval_resc, Xtest_resc, Ytrain, Yval, Ytest, Data_test



# dt =  0.3288967
# freq_ac = 1/dt 
# data = Dataf[1,:]
# import numpy as np
# import matplotlib.pyplot as plt


# # --- Data Extraction ---
# # Extract the relevant columns for plotting
# time_vector = data[:, 0]
# sensor_1 = data[:, 1]
# sensor_2 = data[:, 2]
# sensor_3 = data[:, 3]
# sensor_4 = data[:, 4]


# # --- Plotting ---
# # Create a figure and a set of subplots
# plt.figure(figsize=(12, 7))
# # Plot each sensor's data with appropriate labels for the legend
# plt.plot(time_vector, sensor_1, label=r'$S_{1}$', linewidth = 2.5)
# plt.plot(time_vector, sensor_2, label=r'$S_{2}$', linewidth = 2.5)
# plt.plot(time_vector, sensor_3, label=r'$S_{3}$',linewidth = 2.5)
# plt.plot(time_vector, sensor_4, label=r'$S_{4}$',linewidth = 2.5)

# # --- Customization ---
# # Add labels for the x and y axes with a larger font size
# plt.xlabel('Time [s]', fontsize=22)
# plt.ylabel(r'$H_{2}$ concentration', fontsize=22)

# # You can also increase the size of the tick labels on the axes
# plt.xticks(fontsize=20)
# plt.yticks(fontsize=20)

# # Add a legend to identify each sensor's line, with a larger font size
# plt.legend(fontsize=20)

# # Add a grid for better readability of the plot
# plt.grid(True)


# # --- Save and Display ---
# # Save the figure to a high-resolution PNG file
# plt.savefig(os.path.join('MODULES', 'PREPROCESSING','sensor_measurements.png'), dpi=300)

# #We do not need this function here, but it is a pre-processing function we used in other project    
# def select_segments(Data,n_features,n_steps):
#     # DataF = []
#     # # We need to keep only those segments that start with healthy label 
#     # for i in range(Data.shape[0]):
        
#     #     if Data[i, 0, n_features] == 0.0 or Data[i,n_steps-1,n_features] != 0.0: 
#     #     # if Data[i,0,n_features] != 0.0 and Data[i,n_steps-1, n_features] != 0.0:
#     #         DataF.append(Data[i,:,:])
#     # DataF = np.array(DataF)
#     DataF = Data

#     return DataF
