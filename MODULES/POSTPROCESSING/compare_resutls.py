#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 20 10:26:47 2023

@author: afernandez
"""
import tensorflow as tf
import numpy as np
import os
import seaborn as sns
from MODULES.PREPROCESSING.preprocessing import preprocessing_interface
from MODULES.POSTPROCESSING.postprocessing import plot_configuration
from MODULES.POSTPROCESSING.show_plots import plot_predicted_values_vs_ground_truth, plot_outliers_dam
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import label_binarize
from matplotlib import pyplot as plt
from matplotlib import rc



file_path = "100secsSim_Windows_18Jul50points6TANKS4features_Arch_3_50steps_20000epoch_2048batch5e-05LR"
output_path = os.path.join("Output", 'A_final_outputs',  file_path)
Test_Class_report = np.load(os.path.join(output_path, "TestClassification_Report.npy"), allow_pickle = True)
print(Test_Class_report)


Cm = np.load(os.path.join(output_path, "TestConfusion_Matrix.npy"), allow_pickle = True)

print(Cm)

