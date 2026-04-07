#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 31 10:37:27 2023

@author: afernandez
"""
from sklearn.metrics import confusion_matrix, classification_report
from matplotlib import pyplot as plt
import seaborn as sns 
import numpy as np
import os 


def make_predictions(model,X):
    y_pred = model.predict(X)
    return y_pred

def classification_metrics(y_true, y_pred,target_names, folder_name, fname):
    cr = classification_report(np.argmax(y_true,axis = 1), np.argmax(y_pred, axis = 1), target_names=target_names)
    print(cr)
    np.save(os.path.join('Output',folder_name,fname+'Classification_Report'), cr)

def confusion_matrix_(y_true, y_pred,folder_name,fname):
    fig_cm = plt.figure()
    cm = confusion_matrix(np.argmax(y_true,axis = 1), np.argmax(y_pred,axis = 1))
    minv = cm.min()
    maxv = cm.max()
    fig_cm = sns.heatmap(cm,annot = False, cmap = 'Blues', vmin=minv, vmax=maxv)
    
    fig_cm.get_figure().savefig(os.path.join('Output',folder_name, fname+'heatmap.png'), dpi=500)

    cm_path = os.path.join('Output', folder_name, fname+'Confusion_Matrix.npy')
    np.save(cm_path, cm)
    print('Confusion Matrix:')
    print(cm)
    return fig_cm


def classification_results(model, X, y_true, target_names, folder_name, fname):
    y_pred = make_predictions(model,X)
    classification_metrics(y_true, y_pred,target_names, folder_name, fname)
    confusion_matrix_(y_true, y_pred, folder_name, fname)

    

import matplotlib.pyplot as plt
import numpy as np
def get_text_color(background_color):
    """
    Determines if text should be black or white based on the background color's luminance.
    A simple luminance calculation is used.
    
    Args:
        background_color (tuple): RGBA color tuple.
        
    Returns:
        str: 'white' or 'black'.
    """
    # Unpack the RGBA color
    r, g, b, _ = background_color
    # Calculate luminance
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    # Return black for light backgrounds, white for dark backgrounds
    return 'black' if luminance > 0.5 else 'white'

def plot_confusion_matrix_bar(conf_matrix, class_names, output_path):
    """
    Generates and displays a stacked horizontal bar plot for a confusion matrix.

    Args:
        conf_matrix (np.array): A square confusion matrix.
        class_names (list): A list of strings representing the class names.
    """
    num_classes = len(class_names)
    # Convert the matrix to a NumPy array
    conf_matrix = np.array(conf_matrix)

    # --- Data Preparation ---
    # This helps in placing the text labels proportionally.
    row_sums = conf_matrix.sum(axis=1)
    conf_matrix_percent = conf_matrix / row_sums[:, np.newaxis]

    # --- Color Palette ---
    # A beautiful and colorblind-friendly palette from 'viridis'
    colors = plt.cm.viridis(np.linspace(0, 1, num_classes))

    # --- Plotting ---
    fig, ax = plt.subplots(figsize=(12, 8))

    # Y-axis positions for the bars
    y_pos = np.arange(num_classes)
    
    # Variable to keep track of the left position for stacking bars
    left_positions = np.zeros(num_classes)

    # Iterate through each class to plot its predictions as a segment in the stacked bar
    for i in range(num_classes):
        counts = conf_matrix[:, i]
        bar_color = colors[i]
        bars = ax.barh(y_pos, counts, left=left_positions, height=0.6, label=f"Pred. {class_names[i]}", color=bar_color, edgecolor='white')
        
        # Determine the best text color for contrast
        text_color = get_text_color(bar_color)
        
        # Add text labels inside the bar segments
        for j, bar in enumerate(bars):
            width = bar.get_width()
            # Only add text if the segment is large enough to be visible
            if width > 0:
                # Center the text within the segment
                x_pos = bar.get_x() + width / 2
                # Display the raw count with the determined color
                ax.text(x_pos, bar.get_y() + bar.get_height() / 2,
                        f'{int(width)}', ha='center', va='center', color=text_color, fontsize=18, fontweight='bold')
        
        # Update the left positions for the next stack
        left_positions += counts

    # --- Aesthetics and Labels ---
    ax.set_yticks(y_pos)
    ax.set_yticklabels(class_names, fontsize=18)
    ax.invert_yaxis()  # labels read top-to-bottom

    ax.set_xlabel('Number of Samples', fontsize=18)
    ax.set_ylabel('True Label', fontsize=18)
    ax.tick_params(axis='x', labelsize=18) # Specify fontsize for x-axis ticks


    # Add a legend to explain the colors
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), fancybox=True, shadow=True, ncol=num_classes // 2)

    # Remove the top and right spines for a cleaner look
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Add grid lines for better readability
    ax.xaxis.grid(True, linestyle='--', which='major', color='grey', alpha=.25)

    plt.tight_layout(rect=[0, 0.1, 1, 1]) # Adjust layout to make room for legend
    plt.savefig(os.path.join(output_path, 'Confusion_matrix_barplot.png'), dpi=500, bbox_inches='tight')
    plt.show()


# # The class names (including Tank 3, as the matrix is 6x6)
# output_path = os.path.join("Output", "A_final_outputs", 'ASIER_pruning_results' ) 
# labels = ['Tank 1', 'Tank 2', 'Tank 3', 'Tank 4', 'Tank 5', 'Tank 6']
# cm = np.array([
#     [125, 16, 0, 0, 36, 8],
#     [20, 101, 10, 5, 32, 2],
#     [0, 14, 108, 56, 8, 0],
#     [0, 0, 19, 141, 10, 0],
#     [4, 6, 7, 9, 120, 42],
#     [7, 0, 0, 1, 54, 118]
# ]) # esta es la de 5 filters removed in pruning

# cm = np.array([
#     [118,59, 1,0,0,7],
#     [19,113,34,0,2,1],
#     [0, 12, 160, 12,3,0],
#     [0,2,41,120,7,0],
#     [10,12,14,26,100,25],
#     [15,4,2,3,45,112]
    
#     ]) # Esta es la de 3 filters removed in pruning
# # Generate the plot
# plot_confusion_matrix_bar(cm, labels)


#################################################################################################################################################################33
#OLD stuff
# Ypred_test = model.predict(Xtest_resc)
# # target_names = ['Healthy', 'Faulty']
# # target_names = ['Healthy','F1WT1', 'F2WT1', 'F3WT1', 'F4WT1', 'F1WT2', 'F2WT2', 'F3WT2', 'F4WT2', 'F1WT3', 'F2WT3', 'F3WT3', 'F4WT3']
# target_names = ['Healthy', 'F1WT1','F2WT1', 'F3WT1', 'F4WT1',' F5WT1']
# # target_names = ['Healthy', 'F1WT1', 'F2WT1', 'F3WT1', 'F4WT1',' F5WT1']

# Ypred_train = model.predict(Xtrain_resc)
# Ypred_val = model.predict(Xval_resc)
# train_cr= classification_metrics(Ytrain, Ypred_train, target_names)
# np.save(os.path.join('Output',folder_name,'train_classification_report.npy'), train_cr)

# val_cr= classification_metrics(Yval, Ypred_val, target_names)
# np.save(os.path.join('Output',folder_name,'val_classification_report.npy'), val_cr)

# classification_metrics(Ytest, Ypred_test, target_names)

# # Convert predictions to labels
# y_pred_labels = np.argmax(Ypred_test, axis=1)

# # Postprocessing tools to evaluate classification task performance

# #With argmax operation we go from one-hot-enconding to single value.labels 
# # Confusion matrix
# cm = confusion_matrix(np.argmax(Ytest, axis  =1), y_pred_labels)
# cm_path = os.path.join('Output', folder_name, 'Test_confusion_M.npy')
# np.save(cm_path, cm)
# print('Confusion Matrix:')
# print(cm)


# sns.heatmap(cm, annot=True, cmap='Blues')
# plt.title('Confusion Matrix')
# plt.xlabel('Predicted')
# plt.ylabel('Actual')
# plt.show()


# # Classification report
# cr = classification_report(np.argmax(Ytest, axis  =1), y_pred_labels)
# cr_path = os.path.join('Output', folder_name, 'Test_Classification_report.npy')
# np.save(cr_path, cr)
# print('Classification Report:')
# print(cr)
# # #load testing data (from other file)
# from MODULES.PREPROCESSING.preprocessing_tools import import_data, rem_zeros, rescaling, convert_to_npy

# mat_directory  = os.path.join("Data","Test_data")
# output_folder  =os.path.join("Data", "Test_data", "npy_files")
# header_label = "saveVar"
# convert_to_npy(mat_directory, output_folder, header_label)
# filenames = ["Fault_0.npy", "Fault_1.npy", "Fault_2.npy", "Fault_3.npy"] #creo q no las lee en orden cuando hago el for fname in filenames ...
# lf = len(filenames)
# Data_original = np.array([np.load(os.path.join("Data","npy_files", fname)) for fname in filenames])
#     Data = Data_original.reshape(lf*Data_original.shape[1], Data_original.shape[2])
#     Data = Data[~np.isnan(Data).any(axis=1)] #Remove nan values from the chosen sensors
#     X_data = Data[:,0:input_shape]
#     #For the CLASSIFIER
#     Y_data = Data[:, input_shape+1:]
