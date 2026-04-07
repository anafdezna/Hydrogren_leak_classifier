#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 08:46:52 2023

@author: afernandez
"""
# PLOT DATASETS TO SHOW THE DATA

#Plot Datasets

import tensorflow as tf
import numpy as np
import os
import seaborn as sns
from MODULES.PREPROCESSING.LOAD_DATA import import_mat_file
from MODULES.POSTPROCESSING.postprocessing import plot_configuration
from MODULES.POSTPROCESSING.show_plots import plot_predicted_values_vs_ground_truth, plot_outliers_dam
from MODULES.MODEL.model_creation import architecture_info_initializer
from MODULES.MODEL.training import training_info_initializer, training_model, custom_loss
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import label_binarize
from matplotlib import pyplot as plt
from matplotlib import rc

from MODULES.PREPROCESSING.LOAD_DATA import import_mat_file, import_csv_file
from matplotlib import pyplot as plt


plot_configuration()

file_path = "AF29Sept_20Hz_10points_Farm3WTs_38features_Arch_3_10steps_1500epoch_512batch2e-05LR"
output_path = os.path.join("Output", file_path)


#############
header_label = "saveVar"
mat_directory = os.path.join('Data', 'Data_current_matfiles')
filenames = []
for fname in os.scandir(mat_directory):
    filenames.append(str(fname)[11:-2])#To cut the string and extract only the text of interest
    
filenames = sorted(filenames) 
Data = np.array([import_mat_file(os.path.join(mat_directory, fname), header_label)for fname in filenames])  
#Cut or round the data to ma







# x_values = data11[:,0]
# y_1 = data11[:,1]
# y_2 = data11[:,2]
# y_3 = data11[:,3]
# y_4 = data11[:,4]

# # Plotting
# dataplot = plt.figure()
# plt.plot(x_values, y_1, 'limegreen', linewidth = '2')
# plt.plot(x_values, y_2, 'blue', linewidth = '2')
# plt.plot(x_values, y_3, 'purple', linewidth = '2')
# plt.plot(x_values, y_3, 'orange', linewidth = '2')


# plt.xlabel('Time [s]')
# plt.ylabel('Hydrogen level')
# plt.grid(True)
# # Place the legend outside the plot
# plt.legend(['Sensor 1', 'Sensor 2', 'Sensor 3', 'Sensor 4'], loc='upper left', bbox_to_anchor=(1.05, 1))
# # plt.legend(['$\mathbf{x}_{1}$', '$\mathbf{x}_{2}$','$\mathbf{x}_{3}$'], loc='upper left', bbox_to_anchor=(1.05, 1))
# # dataplot.savefig(os.path.join(output_path, 'Dataset_portion_WT1Sensor_Gen_Offset.png'), dpi=500, bbox_inches='tight')
# plt.show()  # Show the plot








data = Data[6,:,:] #first pitch fault at wt1
# Select the specific range for plotting
start_index = 56000
end_index = 56500

# start_index = 56000
# end_index = 56500


sampling_frequency = 20  # Hz

start_time = start_index / sampling_frequency
end_time = end_index / sampling_frequency
# Extract the relevant data for x and y
# x_values = np.arange(start_index, end_index)
x_values = np.linspace(start_time, end_time, end_index - start_index)
y_1 = data[start_index:end_index, 0]
y_2 = data[start_index:end_index, 1]
y_3 = data[start_index:end_index, 2]


# Plotting
dataplot = plt.figure()
plt.plot(x_values, y_1, 'green', linewidth = '1', marker = 'o', markersize = '3.5')
plt.plot(x_values, y_2, 'cyan', linewidth = '2')
plt.plot(x_values, y_3, 'purple', linewidth = '2',linestyle='--')

plt.xlabel('Time [s]')
plt.ylabel('Angle [deg]')
plt.grid(True)
# Place the legend outside the plot
plt.legend(['$\\beta^{1}_{m}[t]$', '$\\beta^{2}_{m}[t]$','$\\beta^{3}_{m}[t]$'], loc='upper left', bbox_to_anchor=(1.05, 1))
# plt.legend(['$\mathbf{x}_{1}$', '$\mathbf{x}_{2}$','$\mathbf{x}_{3}$'], loc='upper left', bbox_to_anchor=(1.05, 1))
# dataplot.savefig(os.path.join(output_path, 'Dataset_portion_WT1Sensor_Gen_Offset.png'), dpi=500, bbox_inches='tight')
plt.show()  # Show the plot


#HEALTHY FAULTY PITCH 
dataH = Data[0,:,:]
data = Data[1,:,:] #Offset pitch fault at wt1
datan = Data[2,:,:] #Drift pitch fault at wt1

# Select the specific range for plotting
start_index = 56000
end_index = 56500

# start_index = 56000
# end_index = 56500


sampling_frequency = 20  # Hz

start_time = start_index / sampling_frequency
end_time = end_index / sampling_frequency
# Extract the relevant data for x and y
# x_values = np.arange(start_index, end_index)
x_values = np.linspace(start_time, end_time, end_index - start_index)
y_0 = dataH[start_index:end_index, 11]
y_1 = data[start_index:end_index, 11]
y_2 = datan[start_index:end_index, 11]

# Plotting
dataplot = plt.figure()
plt.plot(x_values, y_0, 'limegreen', linewidth = '2.5',  marker  = '*', markersize = 5)
# plt.plot(x_values, y_2, 'cyan', linewidth = '2')
plt.plot(x_values, y_1, 'orange', linewidth = '2')
plt.plot(x_values, y_2, 'purple', linewidth = '2',linestyle='--')

plt.xlabel('Time [s]')
plt.ylabel('$\\beta^{1,3}_{m}$ [deg]')
plt.grid(True)
# Place the legend outside t\mathcal{he plot
plt.legend(['Healthy', 'Offset', 'Drift'], loc='lower left', bbox_to_anchor=(0, 1), ncol =3, fontsize = 14)
dataplot.savefig(os.path.join(output_path, 'Dataset_portion_WT1Sensor_Pitch_Offoiseset_HOffNoi.png'), dpi=500, bbox_inches='tight')
plt.show()  # Show the plot




#GENERATOR SPEED HEATLHY AND FAULTS
dataH = Data[0,:,:]
dataf = Data[3,:,:] #first generator offset fault at wt1
datad = Data[4,:,:]
# Select the specific range for plotting
start_index = 56000
end_index = 56500

# start_index = 56000
# end_index = 56500


sampling_frequency = 20  # Hz

start_time = start_index / sampling_frequency
end_time = end_index / sampling_frequency
# Extract the relevant data for x and y
# x_values = np.arange(start_index, end_index)
x_values = np.linspace(start_time, end_time, end_index - start_index)
y_0 = dataH[start_index:end_index, 24]
y_1 = dataf[start_index:end_index, 24]
y_2 = datad[start_index:end_index, 24]

# Plotting
dataplot = plt.figure()
plt.plot(x_values, y_0, 'limegreen', linewidth = '2.5',  marker  = '*', markersize = 5)
plt.plot(x_values, y_1, 'orange', linewidth = '2')
plt.plot(x_values, y_2, 'purple', linewidth = '2',linestyle='--')

plt.xlabel('Time [s]')
plt.ylabel('$\\omega^{gen,1}_{m}$ [rad/s]')
plt.grid(True)
# Place the legend outside the plot
plt.legend(['Healthy', 'Offset', 'Drift'], loc='upper left', bbox_to_anchor=(0, 1.18), ncol =3, fontsize = 14)
# plt.legend(['$\\omega^{g,h}_{m}[t]$', '$\\omega^{g,{f}}_{m}[t]$'], loc='upper left', bbox_to_anchor=(1.05, 1))
# plt.legend(['$\mathbf{x}_{1}$', '$\mathbf{x}_{2}$','$\mathbf{x}_{3}$'], loc='upper left', bbox_to_anchor=(1.05, 1))
dataplot.savefig(os.path.join(output_path, 'Dataset_portion_WT1Sensor_Generator_HF_.png'), dpi=500, bbox_inches='tight')
plt.show()  # Show the plot




#####################################################################################################################################
### ACTUATOR FAULTS


#GENERATOR SPEED HEATLHY AND FAULTS
dataH = Data[0,:,:]
dataf = Data[5,:,:] #first generator offset fault at wt1
datad = Data[6,:,:]
# Select the specific range for plotting
start_index = 56000
end_index = 56500

# start_index = 56000
# end_index = 56500


sampling_frequency = 20  # Hz

start_time = start_index / sampling_frequency
end_time = end_index / sampling_frequency
# Extract the relevant data for x and y
# x_values = np.arange(start_index, end_index)
x_values = np.linspace(start_time, end_time, end_index - start_index)
y_01 = dataH[start_index:end_index, 2]
y_02 = dataH[start_index:end_index, 11]
y_11 = dataf[start_index:end_index, 2]
y_12 = datad[start_index:end_index, 11]
y_21 = dataf[start_index:end_index, 2]
y_22 = datad[start_index:end_index, 11]

# Plotting
dataplot = plt.figure()
plt.plot(x_values, y_01, 'lime', linewidth = '1.5',  marker  = '*', markersize = 3.5)
plt.plot(x_values, y_02, 'lime', linewidth = '1.5',  marker  = 'o', markersize = 3.5)
# plt.plot(x_values, y_11, 'blue', linewidth = '2', marker  = '*', markersize = 3.5)
# plt.plot(x_values, y_12, 'blue', linewidth = '2', marker  = 'o', markersize = 3.5)
# plt.plot(x_values, y_21, 'darkorange', linewidth = '2',linestyle='--',  marker  = '*', markersize = 3.5)
# plt.plot(x_values, y_21, 'darkorange', linewidth = '2',linestyle='--',  marker  = 'o', markersize = 3.5)


plt.xlabel('Time [s]', fontsize = 18)
plt.ylabel('$\\omega^{\text{gen},1}_{m}$ [rad/s]')
plt.grid(True)
# Place the legend outside the plot
plt.legend(['1', '2'], loc='upper left', bbox_to_anchor=(1.05, 1))
# plt.legend(['$\\omega^{g,h}_{m}[t]$', '$\\omega^{g,{f}}_{m}[t]$'], loc='upper left', bbox_to_anchor=(1.05, 1))
# plt.legend(['$\mathbf{x}_{1}$', '$\mathbf{x}_{2}$','$\mathbf{x}_{3}$'], loc='upper left', bbox_to_anchor=(1.05, 1))
dataplot.savefig(os.path.join(output_path, 'Dataset_portion_WT1Sensor_PitchActuator.png'), dpi=500, bbox_inches='tight')
plt.show()  # Show the plot




#####################################################################################################################################
### SEGMENTS
dataH = Data[0,:,:]
# dataf = Data[5,:,:] #first generator offset fault at wt1
# datad = Data[6,:,:]
# Select the specific range for plotting
start_index = 40000
end_index = 50000

# start_index = 56000
# end_index = 56500


sampling_frequency = 20  # Hz

start_time = start_index / sampling_frequency
end_time = end_index / sampling_frequency
# Extract the relevant data for x and y
# x_values = np.arange(start_index, end_index)
x_values = np.linspace(start_time, end_time, end_index - start_index)
y_01 = dataH[start_index:end_index, 11]


# Plotting
dataplot = plt.figure()
plt.plot(x_values, y_01, 'lime', linewidth = '1.5')
plt.xlabel('Time [s]')
plt.ylabel('$\\beta^{3}_{m}$ [deg]')
plt.grid(True)
# Place the legend outside the plot
# plt.legend(['1', '2'], loc='upper left', bbox_to_anchor=(1.05, 1))
# plt.legend(['$\\omega^{g,h}_{m}[t]$', '$\\omega^{g,{f}}_{m}[t]$'], loc='upper left', bbox_to_anchor=(1.05, 1))
# plt.legend(['$\mathbf{x}_{1}$', '$\mathbf{x}_{2}$','$\mathbf{x}_{3}$'], loc='upper left', bbox_to_anchor=(1.05, 1))
dataplot.savefig(os.path.join(output_path, 'Dataset_portion_WT1Sensor_segmentation.png'), dpi=500, bbox_inches='tight')
plt.show()  # Show the plot






####################################################################################################################################

y_values = data[start_index:end_index, 38]
# Plotting
yvarplot = plt.figure()
plt.plot(x_values, y_values, 'red', linewidth = '2')

plt.xlabel('Time [s]')
plt.ylabel('$\\hat{y}^{\ true}$')
plt.grid(True)
# Place the legend outside the plot
# plt.legend(['$\\hat{y}^{true}[t]$'], loc='upper left', bbox_to_anchor=(1.05, 1))
yvarplot.savefig(os.path.join(output_path, 'Yvar_portion.png'), dpi=500, bbox_inches='tight')
plt.show()  # Show the plot


#Drawing segments with overlapping

# Select the specific range for plotting
start_index = 0
end_index = 10
sampling_frequency = 20  # Hz

start_time = start_index / sampling_frequency
end_time = end_index / sampling_frequency
# Extract the relevant data for x and y
# x_values = np.arange(start_index, end_index)
x_values = np.linspace(start_time, end_time, end_index - start_index)
y_1 = data[start_index:end_index, 9]
y_2 = data[start_index:end_index, 10]
y_3 = data[start_index:end_index, 11]


# Plotting
dataplot = plt.figure()
plt.plot(x_values, y_1, 'green', linewidth = '1.5', linestyle = '--')
plt.plot(x_values, y_2, 'cyan', linewidth = '1.5', linestyle = '--')
plt.plot(x_values, y_3, 'purple', linewidth = '1.5', linestyle = '--')

plt.xlabel('Time [s]')
plt.ylabel('Angle [Deg]')
plt.grid(True)
# Place the legend outside the plot
plt.legend(['$\mathbf{x}_{1}$', '$\mathbf{x}_{2}$','$\mathbf{x}_{3}$'], loc='upper left', bbox_to_anchor=(1.05, 1))
dataplot.savefig(os.path.join(output_path, 'Signal_overlapping.png'), dpi=500, bbox_inches='tight')
plt.show()  # Show the plot


#Saving entire wind plot
# Select the specific range for plotting
start_index = 0
end_index = Data.shape[1]
sampling_frequency = 20  # Hz

start_time = start_index / sampling_frequency
end_time = end_index / sampling_frequency

x_values = np.linspace(start_time, end_time, end_index - start_index)
y_values = Data[0,start_index:end_index, 37]
y_wind = y_values
# Plotting
yvarplot = plt.figure()
plt.plot(x_values, y_values, 'blue', linewidth = '2')

plt.xlabel('Time [s]')
plt.ylabel('Speed [m/s]')
plt.grid(True)
# Place the legend outside the plot
plt.legend(['$v_{wind}^{MM_{1}}$'], loc='upper left', bbox_to_anchor=(0.0, 1))
yvarplot.savefig(os.path.join(output_path, 'Wind_plot.png'), dpi=500, bbox_inches='tight')
plt.show()  # Show the plot






# Plotting the short length test subset pitch angles (measured or refrence)
Data = long_set
sns.set(style="whitegrid")

start_index = 240
end_index = Data.shape[0]
sampling_frequency = 20  # Hz
start_time = start_index / sampling_frequency
end_time = end_index / sampling_frequency

x_values = np.linspace(start_time, end_time, end_index - start_index)
y0 = Data[start_index:end_index, 2]
y1 = Data[start_index:end_index, 11]
y2 = Data[start_index:end_index, 5]
y3 = Data[start_index:end_index, 14]
y4 = Data[start_index:end_index, 8]
y5 = Data[start_index:end_index, 17]


dataplot = plt.figure()
plt.plot(x_values, y1, 'tomato', linewidth = '2.2')
# plt.plot(x_values, y1, 'red', linewidth = '1.2', linestyle = '--')
plt.plot(x_values, y3, 'dodgerblue', linewidth = '2.2')
# plt.plot(x_values, y3, 'blue', linewidth = '1.2')
plt.plot(x_values, y5, 'fuchsia', linewidth = '2.2')
# plt.plot(x_values, y5, 'magenta', linewidth = '1.2')


plt.xlabel('Time [s]')
plt.ylabel('Pitch angle [deg]')
plt.grid(True)
# Place the legend outside the plot
plt.legend(['$\\beta^{1,3}_{m}[t]$', '$\\beta^{2,3}_{m}[t]$', '$\\beta^{3,3}_{m}[t]$'],fontsize = 14,loc='upper left', ncol = 3, bbox_to_anchor=(-0.07, 1.22))
dataplot.savefig(os.path.join(output_path, 'short_test_pitch_measured.png'), dpi=500, bbox_inches='tight')
plt.show()  # Show the plot


#Plotting the generator signals
z0 = Data[start_index:end_index, 24]
z1 = Data[start_index:end_index, 25]
z2 = Data[start_index:end_index, 26]


dataplot = plt.figure()
plt.plot(x_values, z0, 'tomato', linewidth = '2.2')
plt.plot(x_values, z1, 'dodgerblue', linewidth = '2.2')
plt.plot(x_values, z2, 'fuchsia', linewidth = '2.2')


plt.xlabel('Time [s]')
plt.ylabel('Generator speed [rad/s]')
plt.grid(True)
# Place the legend outside the plot
plt.legend(['$\\omega^{g,WT_{1}}_{m}$ ', '$\\omega^{g,WT_{2}}_{m}$', '$\\omega^{g,WT_{3}}_{m}$'],fontsize = 14,loc='upper left', ncol = 3, bbox_to_anchor=(0, 1.22))
dataplot.savefig(os.path.join(output_path, 'short_test_gen_measured.png'), dpi=500, bbox_inches='tight')
plt.show()  # Show the plot




