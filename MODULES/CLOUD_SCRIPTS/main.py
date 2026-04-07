#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 12 15:07:58 2022

@author: ana
"""
#IMPORT PACKAGES
import os 
import numpy as np
# import pandas as pd
# import seaborn as sns
from MODULES.CLOUD_SCRIPTS.LOAD_DATA import import_excel_file, import_csv_file, import_txt_file, import_mat_file
from MODULES.CLOUD_SCRIPTS.AFDD import extract_sampling_features, Obtain_Freqs, Obtain_Modes



# loc = os.path.join("Data","20080101_prueba.npy")
# datanumpy  = np.load(loc) #to import from npy 
# filepath = os.path.join("Data", "2022-12-14-12-49_influxdb_data_PRUEBA0.csv")
filepath = os.path.join("Data", "03.mat")


shear = filepath.split('.')[1] 
if shear == 'mat': 
    Data = import_mat_file(filepath)
elif shear == 'xlsx': 
    Data = import_excel_file(filepath)
elif shear == 'csv':
    Data = import_csv_file(filepath)
elif shear =='txt':
    Data = import_txt_file(filepath)
else:
    print('Incorrect input data format')

#Signal specifications
# n, T, f_s = extract_sampling_features(Data[:,0])
f_s = 12.5
T = 1/f_s 
n = Data.shape[0]



Freqs, FDD = Obtain_Freqs(n, T, f_s, Data)
print(Freqs.shape)


# Freqs = [5.615, 20.02]
# final_Freqs, final_Modes = Obtain_Modes(Freqs,FDD)
