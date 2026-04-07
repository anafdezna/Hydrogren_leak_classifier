#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 18 12:08:16 2023

@author: afernandez
"""
import numpy as np
import os
# from MODULES.PREPROCESSING.LOAD_DATA import import_excel_file
from MODULES.POSTPROCESSING.postprocessing import plot_configuration
# from MODULES.POSTPROCESSING.show_plots import plot_predicted_values_vs_ground_truth, plot_outliers_dam
# from MODULES.MODEL.model_creation import architecture_info_initializer
# from MODULES.MODEL.training import training_info_initializer, training_model, custom_loss
# from sklearn.metrics import confusion_matrix
# from sklearn.preprocessing import label_binarize
from matplotlib import pyplot as plt
# from matplotlib import rc
# import matplotlib.patches as mpatches


plot_configuration()


from MODULES.PREPROCESSING.preprocessing import segment_data_overlapping
def Plot_time_domain_probability_evolution(Dataf_test, model, n_features, n_steps, nslide, output_path):
    Ntanks = 6
    for i in range(Ntanks):
        
        Test_dataset  = Dataf_test[i:i+1,:,1:5]
        Test_dataset = 1.5*Test_dataset
        Test_segments = segment_data_overlapping(Test_dataset,n_features, n_steps, nslide)
        
        
        Test_preds = model.predict(Test_segments)
        
        
        start_p = 0
        end_p = 200
        
        dt =  0.3288967
        freq_ac = 1/dt 
        time_vector = Dataf_test[3,start_p:end_p,0]
        npoints = n_steps # data points in time domain that form one window
        tdelay = npoints*dt 
        
        
        time_vector_initial = np.arange(0, end_p * dt, dt)
        # --- 2. Your Original Data Preparation ---
        # This is the code you provided.
        time_vector = time_vector_initial + tdelay
        all_tanks = Test_preds[start_p:end_p, 0:6] # More efficient to slice all tanks at once
        
        # --- 4. Generate the "History" Time Vector ---
        # This creates 50 new time points that count backwards from the original start time.
        # The start of this new vector will be: time_vector[0] - 50 * dt
        # The end will be just before time_vector[0].
        start_time_prepend = time_vector[0] - npoints * dt
        end_time_prepend = time_vector[0]
        time_prepend = np.arange(start_time_prepend, end_time_prepend, dt)
        
        # --- 5. Generate the "History" Zero-Value Data for Tanks ---
        # This creates a matrix of zeros with 50 rows (for npoints) and 6 columns (for the 6 tanks).
        zeros_prepend = np.zeros((npoints, all_tanks.shape[1]))
        
        # --- 6. Combine the "History" with the Original Data ---
        # np.concatenate joins the arrays. We add the new data to the beginning (axis=0).
        time_vector_extended = np.concatenate((time_prepend, time_vector))
        # time_vector_extended = time_vector_extended[:-1]
        
        all_tanks_extended = np.concatenate((zeros_prepend, all_tanks), axis=0)
        
        # --- 7. (Optional) Assign to Individual Variables ---
        # If you still need separate variables for each tank, you can easily slice them
        # from the combined matrix.
        tank_1_extended = all_tanks_extended[:, 0]
        tank_2_extended = all_tanks_extended[:, 1]
        tank_3_extended = all_tanks_extended[:, 2]
        tank_4_extended = all_tanks_extended[:, 3]
        tank_5_extended = all_tanks_extended[:, 4]
        tank_6_extended = all_tanks_extended[:, 5]
        
        
        
        filenam = f'Prob_evolution_tank_{i+1}_test_final'
        
        fig, ax = plt.subplots(figsize=(6, 4))
        
        # Set a light background color for the plot area
        ax.set_facecolor('#f5f5f5')
        
        # Plot the data for each tank using the extended variables
        ax.plot(time_vector_extended, tank_1_extended, linewidth='2', label='Tank 1', color='red')
        ax.plot(time_vector_extended, tank_2_extended, linewidth='2', label='Tank 2', color='limegreen')
        ax.plot(time_vector_extended, tank_3_extended, linewidth='2', label='Tank 3', color='darkblue')
        ax.plot(time_vector_extended, tank_4_extended, linewidth='2', label='Tank 4', color='orange')
        ax.plot(time_vector_extended, tank_5_extended, linewidth='2', label='Tank 5', color='cyan')
        ax.plot(time_vector_extended, tank_6_extended, linewidth='2', label='Tank 6', color='violet')
        
        # Add a vertical line to show where the original data begins
        ax.axvline(time_vector[0], color='black', linestyle='-.', lw=1.0, label='First segment')
        
        # Set titles and labels
        ax.set_title(f'Probability evolution for leak in Tank {i+1}')
        ax.set_ylabel('Probability', fontsize = 16)
        ax.set_xlabel('Time [s]', fontsize = 16)
        ax.tick_params(axis='both', which='major', labelsize=16)
        
        # Position the legend outside the plot area, slightly to the right
        ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5))
        
        # Ensure grid is visible
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        
        # Save the figure to the specified path
        # bbox_inches='tight' ensures the legend is not cut off
        plt.savefig(os.path.join(output_path, filenam + '.png'), dpi=500, bbox_inches='tight')
        
        # Display the plot
        plt.show()
    
    
    
    
    
 # BAR PLOTS
 

# file_path = "Trained_50points6TANKS4features_Arch_3_50steps_10000epoch_2048batch5e-05LR"
# output_path = os.path.join("Output", "SEGURH2_results", "share", file_path)


# # file_path = "WindowSelection_June3050points6TANKS4features_Arch_3_50steps_20000epoch_2048batch5e-05LR"
# # file_path = "Sol0_11Jan_20Hz_100points12Tanks_4features_Arch_3_100steps_1500epoch_2048batch5e-05LR"
# # output_path = os.path.join("Output", file_path)

# sampling_frequency = 3  # Hz
# # Load Data used during traning and rescaling information 
# Datafiles = np.load(os.path.join(output_path, 'Datafiles.npy'), allow_pickle = True).item()
# Xtrain_resc, Xval_resc, Xtest_resc, Ytrain, Yval, Ytest =  Datafiles['Xtrain_resc'],  Datafiles['Xval_resc'],  Datafiles['Xtest_resc'], Datafiles['Ytrain'], Datafiles['Yval'], Datafiles['Ytest']
# # Load model architecture and training info
# Problem_info = np.load(os.path.join(output_path, 'Problem_info.npy'), allow_pickle = True).item()
# n_features, n_steps, nslide, nfp, n_classes, epochs, batch_s, LRate, Architecture = Problem_info['n_features'],Problem_info['n_steps'], Problem_info['nslide'], Problem_info['nfp'], Problem_info['n_classes'], Problem_info['epochs'], Problem_info['batch_s'], Problem_info['LRate'], Problem_info['Arch']


# # Initialize architecture properties
# architecture_info = architecture_info_initializer(
#     DeepClassifier=Architecture,
#     input_dim=(Xtrain_resc.shape[1], Xtrain_resc.shape[2]),  # for 1D_CNN and LSTM
#     enc_dim = Ytrain.shape[1])

# # Initiailize the training properties
# training_info = training_info_initializer(
#     n_epoch=epochs,
#     batch_size=batch_s,
#     LR=LRate,
#     metrics="accuracy",
#     loss="categorical_crossentropy",
#     shuffle=True,
#     train_flag = False #IMPORTANT since we are plotting results and need not to train but build the model 
#     )


# folder_name = output_path
# model, history = training_model(
#     architecture_info, training_info, Xtrain_resc, Xval_resc, Ytrain, Yval, folder_name)
# model.build(input_shape = ())
# weights_path = os.path.join(output_path, "model_weights.h5" )
# model.load_weights(weights_path)

# yptest = model.predict(Xtest_resc)
# ######################

# ################################# TEST LONG Dataset ####################################################################

# from MODULES.PREPROCESSING.preprocessing import segment_data_overlapping, Data_XYseparation, Data_rescaling, select_segments


# #Altura 05m es  3
# #TESTING the leak detection 
# sheet_name = 'Hoja1'
# filenam= 'Jan15_deposito10.xlsx'
# test_path = os.path.join('Data', 'Data_test_files', filenam)
# Test_dataset = np.array([import_excel_file(os.path.join(test_path), sheet_name)])[:,:,1:5]
# #Rescaling according to the "value increase of *1.5 that we considere interesting for being the values very close to zero otherwise
# Test_dataset_resc = 1.5*Test_dataset
# Test_segments = segment_data_overlapping(Test_dataset,n_features,n_steps,nslide)




# Test_segments = Xtest_resc

# ###################################################################################################
# # # Define the class labels and corresponding colors
# # class_labels = ['Tank 1', 'Tank 2', 'Tank 3', 'Tank 4', 'Tank 5', 'Tank 6']
# marker_colors = ['forestgreen','maroon', 'sienna', 'gold', 'darkblue', 'darkviolet' ]
# # class_colors = ['lime', 'red', 'darkorange','yellow', 'cyan', 'mediumorchid']
    


# test_path = os.path.join("Output", "SEGURH2_results", "share", "Trained_50points6TANKS4features_Arch_3_50steps_10000epoch_2048batch5e-05LR")
# a = np.load(os.path.join(test_path, "Test_data.npy"),allow_pickle =True).item()
# Ytest = a['Ytest']
# Xtest = a['Xtest']

#################### selecting scenarios for the time-domain plot

# Test_dataset = Dataf[0:1,:,1:5] # scenrio with leak at tank 1

# Test_dataset = Dataf[7:8,:,1:5] # scenrio with leak at tank 2

# Test_dataset = Dataf[14:15,:,1:5] # scenrio with leak at tank 3

# Test_dataset = Dataf[21:22,:,1:5] # scenrio with leak at tank 4

# Test_dataset = Dataf[30:31,:,1:5] # scenrio with leak at tank 5

# Test_dataset = Dataf[35:36,:,1:5] # scenrio with leak at tank 6


# Test_dataset = 1.5*Test_dataset

# Test_segments = segment_data_overlapping(Test_dataset,n_features,n_steps,nslide)

    




# time_vector = time_vector + tdelay
# tank_1 = Test_preds[start_p:end_p, 0]
# tank_2 = Test_preds[start_p:end_p, 1]
# tank_3 = Test_preds[start_p:end_p, 2]
# tank_4 = Test_preds[start_p:end_p, 3]
# tank_5 = Test_preds[start_p:end_p, 4]
# tank_6 = Test_preds[start_p:end_p, 5]





# plt.figure()
# plt.plot(time_vector, tank_1, linewidth = '2', color ='red')
# plt.plot(time_vector, tank_2, linewidth = '2', color ='limegreen')
# plt.plot(time_vector, tank_3, linewidth = '2', color ='dodgerblue')
# plt.plot(time_vector, tank_4, linewidth = '2', color ='orange')
# plt.plot(time_vector, tank_5, linewidth = '2', color ='darkgreen')
# plt.plot(time_vector, tank_6, linewidth = '2', color ='darkblue')

# plt.legend(['Tank 1', 'Tank 2' , 'Tank 3', 'Tank 4', 'Tank 5', 'Tank 6'], loc = 'center left',  bbox_to_anchor=(1, 0.5))
# plt.ylabel('Probability')
# plt.xlabel('Time [s]')
# plt.savefig(os.path.join(output_path, filenam +'.png'), dpi=500, bbox_inches='tight')
# plt.show()



# # plt.plot(Test_preds[start_p:end_p,0], linewidth = '2', color ='red')
# # plt.plot(Test_preds[start_p:end_p,1], linewidth = '2', color = 'limegreen')
# # plt.plot(Test_preds[start_p:end_p,2], linewidth = '2', color = 'dodgerblue')
# # plt.plot(Test_preds[start_p:end_p,3], linewidth = '2', color = 'orange')
# # plt.plot(Test_preds[start_p:end_p,4], linewidth = '2', color = 'darkgreen')
# # plt.plot(Test_preds[start_p:end_p,5], linewidth = '2', color = 'darkblue')
# # plt.plot(Test_preds[start_p:end_p,6], linewidth = '2', color = 'coral')
# # plt.plot(Test_preds[start_p:end_p,7], linewidth = '2', color = 'gray')
# # plt.plot(Test_preds[start_p:end_p,8], linewidth = '2', color = 'brown')
# # plt.plot(Test_preds[start_p:end_p,9], linewidth = '2', color = 'cyan')
# # plt.plot(Test_preds[start_p:end_p,10], linewidth = '2', color = 'pink')
# # plt.plot(Test_preds[start_p:end_p,11], linewidth = '2', color  = 'red')





# ### Time-domain classification 
# gtruth = Y_long
# ypred = model.predict(X_long_resc)
# ypred = truncate_fh_preds(long_set_segments,ypred, n_features, n_steps)

# # Select the specific range for plotting
# start_indices = [1570,1850,2130,2470,2650, 2930]
# end_indices = [1665,1940,2230, 2565,2740, 3020]


# for j in range(len(start_indices)):
#     dam_ind = j  
#     start_index = start_indices[dam_ind]
#     end_index = end_indices[dam_ind]
#     start_time = start_index / sampling_frequency
#     end_time = end_index / sampling_frequency
#     x_values = np.linspace(start_time, end_time, end_index - start_index)
    
#     # Create the figure and axis
#     prob_plot = plt.figure()
#     ax = prob_plot.add_subplot(111)
    


#     class_labels = ['$Healthy$', '$Class 2$', '$Class 3$', '$Class 4$', '$Class 5$', '$Class 6$', '$Class 7$', '$Class 8$', '$Class 9$', '$Class 10$']
#     class_colors = ['lime', 'magenta', 'yellow', 'dodgerblue', 'green', 'red', 'orange', 'darkblue', 'magenta', 'darkblue']
    
    
#     # class_labels = ['$Healthy$', '$Class 2$', '$Class 3$', '$Class 4$', '$Class 5$', '$Class 6$', '$Class 7$', '$Class 8$', '$Class 9$', '$Class 10$']
#     # class_colors = ['lime', 'magenta', 'yellow', 'dodgerblue', 'green', 'red', 'orange', 'darkblue', 'magenta', 'darkblue']
    
#     # Plot the ground truth and prediction lines with labels
#     for i in range(10):
#         gt_color = class_colors[i]
#         pred_color = class_colors[i]  # Offset for prediction colors
#         marker_color = marker_colors[i]
#         # Ground truth lines
#         plt.plot(x_values, gtruth[start_index:end_index,i], color=gt_color, linewidth=1.5, label=class_labels[i])
        
#         # Prediction lines with markers
#         plt.plot(x_values, ypred[start_index:end_index,i], color=pred_color, linewidth=0.005, marker='o', markersize=5.5, markerfacecolor = marker_color,  label=f'{class_labels[i]} (Pred)')
    
#     # Set the legend outside the plot
#     legend_patches = [mpatches.Patch(color=color, label=label) for color, label in zip(class_colors, class_labels)]
#     ax.legend(handles=legend_patches, loc='upper left', fontsize = 12, bbox_to_anchor=(1,0.98))
    
#     # Set labels and title
#     plt.xlabel('Time [s]')
#     plt.ylabel('Probability')
#     # Save the figure
#     prob_plot.savefig(os.path.join(output_path, 'Test2s_fault'+ str(dam_ind +1)+'.png'), dpi=500, bbox_inches='tight')
#     # Show the plot
#     plt.show()

# #################### LOSS PLOT ##################################################################

# #Temporary plot_loss to adapt parameters
# def plot_loss_evolution(history, output_path):
#     loss_plot  = plt.figure()
#     loss = history.history['loss']
#     val_loss = history.history['val_loss']
#     # plt.plot(loss,color = '#072FD1',)
#     # plt.plot(val_loss, color = 'red')
#     plt.plot(np.log10(loss),color = '#072FD1',)
#     plt.plot(np.log10(val_loss), color = 'red')
#     plt.ylabel(r'$log_{10}(\mathcal{L}) $')
#     plt.xlabel('epoch')
#     xmin, xmax = plt.xlim()
#     # plt.grid(False)
#     # ymin, ymax = plt.ylim()
#     # ymin,ymax =0.01, 1
#     ymin,ymax = -0.4, 0.3
#     scale_factor = 1
#     plt.xticks()
#     plt.yticks()
#     plt.xlim(xmin *scale_factor, xmax * scale_factor)
#     plt.ylim(ymin * scale_factor, ymax * scale_factor)
#     plt.legend(['Train', 'Validation'], loc='upper right')
#     loss_plot.savefig(os.path.join(output_path, 'Log_Loss.png'),dpi = 500, bbox_inches='tight')
#     plt.show()

# Loss_data = np.load(os.path.join(output_path, "Loss.npy"), allow_pickle=True)
# history = Loss_data.item()
# plot_loss_evolution(history, output_path)




# Train_cm = np.load(os.path.join(output_path, "TrainConfusion_Matrix.npy"), allow_pickle = True)
# Val_cm = np.load(os.path.join(output_path, "ValidationConfusion_Matrix.npy"), allow_pickle = True)
# Test_cm = np.load(os.path.join(output_path, "TestConfusion_Matrix.npy"), allow_pickle = True)

# Train_Class_report  = np.load(os.path.join(output_path, "TrainClassification_Report.npy"), allow_pickle = True)
# Val_Class_report  = np.load(os.path.join(output_path, "ValidationClassification_Report.npy"), allow_pickle = True)
# Test_Class_report  = np.load(os.path.join(output_path, "TestClassification_Report.npy"), allow_pickle = True)



# # Example predicted probabilities (replace with your actual predicted probabilities)
# # Each row represents a test sample, and each column represents the predicted probability for a class
# pred_fname = "test_preds.npy"
# predicted_probs = np.load(os.path.join(output_path,pred_fname))

# # Example true labels (replace with your actual true labels)
# true_fname = "true_test.npy"
# true_labels = np.argmax(np.load(os.path.join(output_path, true_fname)), axis = 1)


    

# #PLOTTING THE CONFUSION MATRIX AS A BAR PLOT:
# # Example confusion matrix (replace this with your actual confusion matrix)

# confusion_matrix = Test_cm
# # Set a Seaborn style to improve the aesthetics
# # sns.set(style="whitegrid")

# # Define a custom color palette based on your 'colors' list
# colors = ['lime', 'dodgerblue', 'darkorange', 'purple', 'red', 'gold', 'cyan', 'brown', 'gray', 'magenta']
# # colors = class_colors

# custom_palette = sns.color_palette(colors)
# # custom_palette = sns.cubehelix_palette(len(class_labels), gamma=0.5)

# # Plotting
# barcm_plot = plt.figure()

# for i in range(len(class_labels)):
#     class_samples = confusion_matrix[i]
    
#     left = 0
#     for j in range(len(class_labels)):
#         plt.barh(class_labels[i], class_samples[j],  color=custom_palette[j], left=left,alpha=0.9)
#         left += class_samples[j]

# plt.xlabel("Number of Samples")
# plt.ylabel("Class Labels")

# # plt.title("Confusion Matrix Bar Plot")
# plt.legend(class_labels, loc='lower right', bbox_to_anchor=(1.35, 0.0))
# # plt.tight_layout()
# plt.xticks()
# plt.yticks()# Customize ticks and grid
# plt.xticks(np.arange(0, max(np.sum(confusion_matrix, axis=1)) + 500, step=1000))
# plt.gca().invert_yaxis()
# plt.grid(axis='x', linestyle='--', alpha=0.7)
# barcm_plot.savefig(os.path.join(output_path, 'BarCM_Test_plot.png'), dpi=500, bbox_inches='tight')
# plt.show() 







# #Vertical version 
# import numpy as np
# import matplotlib.pyplot as plt
# import os
# from matplotlib.patches import HatchPattern

# # Example confusion matrix (replace this with your actual confusion matrix)
# confusion_matrix = np.array([
#     [8682, 3, 0, 4, 0, 4, 0],
#     [134, 4027, 0, 102, 1, 95, 0],
#     [0, 0, 4211, 1, 1, 0, 0],
#     [89, 95, 1, 4061, 0, 81, 0],
#     [1, 0, 0, 0, 4380, 1, 2],
#     [110, 90, 0, 111, 1, 3996, 0],
#     [0, 0, 0, 0, 0, 0, 4269]
# ])

# class_labels = ["Healthy", "$F_{1}WT_{1}$", "$F_{2}WT_{1}$", "$F_{1}WT_{2}$", "$F_{2}WT_{2}$", "$F_{1}WT_{3}$", "$F_{2}WT_{3}$"]
# colors = ['lime', 'dodgerblue', 'darkorange', 'purple', 'gold', 'red', 'cyan']
# patterns = [None, None, '///', None, '///', None, '///']

# # Plotting
# barcm_plot = plt.figure(figsize=(12, 8))

# for i in range(len(class_labels)):
#     class_samples = confusion_matrix[i]
    
#     bottom = 0
#     for j in range(len(class_labels)):
#         color = colors[j]
#         pattern = patterns[j] if patterns[j] else None
#         plt.bar(class_labels[i], class_samples[j], color=color, bottom=bottom, hatch=pattern)
#         bottom += class_samples[j]

# plt.ylabel("Number of Samples")
# plt.xlabel("Class Labels")
# plt.title("Confusion Matrix Bar Plot")
# plt.legend(class_labels, loc='upper right')

# # Customize ticks and grid
# plt.yticks(np.arange(0, max(np.sum(confusion_matrix, axis=1)) + 500, step=1000))
# plt.grid(axis='y', linestyle='--', alpha=0.7)
# # Create custom legend for background patterns
# legend_elements = [HatchPattern(pattern='///', color='gray', label='Same WT Classes')]
# plt.legend(handles=legend_elements, loc='upper left')
# plt.tight_layout()
# barcm_plot.savefig(os.path.join(output_path, 'BarCM_plot_vertical.png'), dpi=500, bbox_inches='tight')
# plt.show()



# #TABLE OF CLASSIFICATION REPORT DURING TESTING
# Test_Cl_Report= np.load(os.path.join(output_path, "TestClassification_Report.npy"), allow_pickle=True)

# confusion_matrix = Test_cm

# # Create the color plot
# plt.figure(figsize=(10, 8))
# plt.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues)
# plt.colorbar()

# # Add labels
# # plt.title("Confusion Matrix")
# plt.xticks(np.arange(len(class_labels)), class_labels, rotation=45)
# plt.yticks(np.arange(len(class_labels)), class_labels)
# # plt.xlabel("Predicted")
# # plt.ylabel("Actual")

# # Add text for each cell in black
# for i in range(len(class_labels)):
#     for j in range(len(class_labels)):
#         text_color = 'black' if i == j else 'black'  # Make diagonal numbers black
#         plt.text(j, i, str(confusion_matrix[i, j]), ha="center", va="center", color=text_color)
# plt.savefig(os.path.join(output_path, 'Test_confusion_matrix.png'), dpi=500, bbox_inches='tight')
# plt.tight_layout()
# plt.show()



# # Create the color plot
# plt.figure(figsize=(10, 8))
# plt.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues)
# plt.colorbar()

# # Add labels
# plt.title("Confusion Matrix")
# plt.xticks(np.arange(len(class_labels)), class_labels, rotation=45)
# plt.yticks(np.arange(len(class_labels)), class_labels)
# plt.xlabel("Predicted")
# plt.ylabel("Actual")

# # Add text for each cell in black
# for i in range(len(class_labels)):
#     for j in range(len(class_labels)):
#         text_color = 'black' if i == j else 'black'  # Make diagonal numbers black
#         plt.text(j, i, str(confusion_matrix[i, j]), ha="center", va="center", color=text_color)

# plt.tight_layout()
# plt.show()







#OLD PLOTS:
    # gtruth = Y_long
    # ypred = model.predict(X_long_resc)

    # # Select the specific range for plotting
    # start_indices = [0,1550,1840,2110,2430,2610, 2890]
    # end_indices = [0,1700,1930,2200, 2540,2700, 2960]
    # dam_ind = 6
    # start_index = start_indices[dam_ind]
    # end_index = end_indices[dam_ind]


    # #For the thris
    # start_time = start_index / sampling_frequency +2.2
    # end_time = end_index / sampling_frequency+2.2
    # x_values = np.linspace(start_time, end_time, end_index - start_index)
    # prob_plot = plt.figure()
    # plt.plot(x_values, gtruth[start_index:end_index,0], 'lime', linewidth = '2')
    # plt.plot(x_values, ypred[start_index:end_index,0], 'lime', linewidth = '0.1', marker = 'o', markersize = '2', markerfacecolor = 'green')
    # plt.plot(x_values, gtruth[start_index:end_index,1], 'magenta', linewidth = '1.5')
    # plt.plot(x_values, ypred[start_index:end_index,1], 'magenta', linewidth = '0.5', marker = 'o', markersize = '2', markerfacecolor = 'red')
    # plt.plot(x_values, gtruth[start_index:end_index,2], 'magenta', linewidth = '1.5')
    # plt.plot(x_values, ypred[start_index:end_index,2], 'red', linewidth = '0.5', marker = 'o', markersize = '2')
    # plt.plot(x_values, gtruth[start_index:end_index,3], 'magenta', linewidth = '1.5')
    # plt.plot(x_values, ypred[start_index:end_index,3], 'red', linewidth = '0.5', marker = 'o', markersize = '2')
    # plt.plot(x_values, gtruth[start_index:end_index,4], 'yellow', linewidth = '1.5')
    # plt.plot(x_values, ypred[start_index:end_index,4], 'orange', linewidth = '0.5', marker = 'o', markersize = '2')
    # plt.plot(x_values, gtruth[start_index:end_index,5], 'yellow', linewidth = '1.5')
    # plt.plot(x_values, ypred[start_index:end_index,5], 'orange', linewidth = '0.5', marker = 'o', markersize = '2')
    # plt.plot(x_values, gtruth[start_index:end_index,6], 'yellow', linewidth = '1.5')
    # plt.plot(x_values, ypred[start_index:end_index,6], 'orange', linewidth = '0.5', marker = 'o', markersize = '2')
    # plt.plot(x_values, gtruth[start_index:end_index,7], 'dodgerblue', linewidth = '1.5')
    # plt.plot(x_values, ypred[start_index:end_index,7], 'darkblue', linewidth = '0.5', marker = 'o', markersize = '2')
    # plt.plot(x_values, gtruth[start_index:end_index,8], 'dodgerblue', linewidth = '1.5')
    # plt.plot(x_values, ypred[start_index:end_index,8], 'darkblue', linewidth = '0.5', marker = 'o', markersize = '2')
    # plt.plot(x_values, gtruth[start_index:end_index,9], 'dodgerblue', linewidth = '1.5')
    # plt.plot(x_values, ypred[start_index:end_index,9], 'darkblue', linewidth = '0.5', marker = 'o', markersize = '2')
    # plt.xlabel('Time [s]')
    # prob_plot.savefig(os.path.join(output_path, 'Test2s_fault6.png'), dpi=500, bbox_inches='tight')

# probs = np.argmax(Ypred,axis = 1)
# Y_long = np.argmax(Y_long, axis = 1)
#Establecer una condición para detectar fallos 
# for i in range(probs.shape[0]):
#start_index = 1500
# end_index = 3100
# start_time = start_index / sampling_frequency
# end_time = end_index / sampling_frequency
# x_values = np.linspace(start_time, end_time, end_index - start_index)
# y_0  = probs[start_index:end_index]
# y_1 = Y_long[start_index:end_index]
# # Plotting
# dataplot = plt.figure()
# plt.plot(x_values, y_1, 'lime', linewidth = '3', marker = 'o', markersize = 2)
# plt.plot(x_values, y_0, 'blue', linewidth = '1.5', linestyle = '--')
# plt.xlabel('Time [s]')
# plt.ylabel('Predicted class')
# plt.grid(True)
# # Place the legend outside the plot
# # plt.legend(['$\mathbf{x}_{1}$'], loc='upper left', bbox_to_anchor=(1.05, 1))
# # dataplot.savefig(os.path.join(output_path, 'Long_Set.png'), dpi=500, bbox_inches='tight')
# plt.show()  # Show the plot



##########################################################################################################################
###################### PLOT EACH OF THE 10 PROBABILITY VALUES (PROBABILITY OF EACH CLASS DRAW) ##################################



# # Calculate confidence scores for each class
# num_classes = predicted_probs.shape[1]
# class_confidence_scores = [[] for _ in range(num_classes)]
# for class_idx in range(num_classes):
#     class_mask = true_labels == class_idx
#     class_probs = predicted_probs[class_mask, class_idx]
#     class_confidence_scores[class_idx].extend(class_probs)

# # Class labels (replace with your actual class labels)
# class_labels = ['Healthy','$F_{1}WT_{1}$', '$F_{2}WT_{1}$', '$F_{3}WT_{1}$' ,'$F_{1}WT_{2}$', '$F_{2}WT_{2}$', '$F_{3}WT_{2}$', '$F_{1}WT_{3}$', '$F_{2}WT_{3}$', '$F_{3}WT_{3}$']

# # Set a Seaborn style to improve the aesthetics
# sns.set(style="whitegrid")

# # Define a custom color palette
# # colors = ['lime', 'dodgerblue', 'darkorange', 'purple', 'red', 'gold', 'cyan', 'brown', 'gray', 'magenta']
# colors = class_colors
# custom_palette = sns.color_palette(colors)
# # custom_palette = sns.color_palette("Set2", len(class_confidence_scores))


# # Create individual Confidence Distribution Plots for each class
# for class_idx, class_scores in enumerate(class_confidence_scores):
#     plt.figure()
    
#     # Create the histogram with a custom color
#     sns.histplot(class_scores, bins=30, kde=False, color=custom_palette[class_idx])
    
#     # Calculate summary metrics
    
#     # Calculate summary metrics
#     mean_score = np.mean(class_scores)
#     median_score = np.median(class_scores)
#     std_dev = np.std(class_scores)

#     # Add metrics as text annotations within the plot
#     plt.text(0.1, 0.75, f'Mean: {mean_score:.2f}\nMedian: {median_score:.2f}\nStd Dev: {std_dev:.2f}',
#              transform=plt.gca().transAxes, bbox=dict(facecolor='white', edgecolor='black',  alpha=0.9))

#     # plt.title(f'Confidence Distribution Plot - {class_labels[class_idx]}', fontsize = 18)
#     plt.xlabel('Confidence Score')
#     plt.ylabel('Frequency')
    
#     plt.xticks()
#     plt.yticks()
#     plt.tight_layout()

#     # Generate a unique filename based on the class label or index
#     filename = f'Confidence_{class_labels[class_idx]}.png'

#     # Save the figure with the generated filename
#     plt.savefig(os.path.join(output_path, filename), dpi=300, bbox_inches='tight')

#     plt.show()
    
    
    


