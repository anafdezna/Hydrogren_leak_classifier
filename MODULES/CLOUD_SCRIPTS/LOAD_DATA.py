# -*- coding: utf-8 -*-
"""
Created on Wed Dec 14 10:18:15 2022

@author: 110137
"""

import pandas as pd
import numpy as np
import time
import os
import scipy.io


def convert_to_npy(loc):
   df = np.array(pd.read_excel(loc, nrows = None))
   # np.save(filename, df)
   return df

#FOR EXCEL FILES 
def import_excel_file(loc):
    Data = np.array(pd.read_excel(loc, nrows = None))
    # Data = Data[~np.isnan(Data).any(axis=1)] #Remove nan value
    return Data


#FOR CSV FILES:
# loc = os.path.join("Data","2022-12-14-12-49_influxdb_data_PRUEBA0.csv" )

def import_csv_file(loc):
    data = pd.read_csv(loc).values
    # Acc1 = data[3:,6]
    # Acc2 = data[3:,6]
    # Data = np.transpose(np.vstack((Acc1,Acc2)))
    Data = data.astype(np.float32)
    Data = Data[~np.isnan(Data).any(axis=1)] #Remove nan value
    return Data

#FOR TXT FILES:
# loc = os.path.join("Data","ACC_UNDAMAGED.txt" )

def import_txt_file(loc):
    arr = np.loadtxt(loc)
    Data = arr.astype(np.float32)
    Data = Data[~np.isnan(Data).any(axis=1)] #Remove nan value
    return Data



#FOR .MAT files:
# loc = os.path.join("Data","03.mat")

def import_mat_file(loc):
    mat = scipy.io.loadmat(loc)
    matkeys = mat.keys()
    print(matkeys)
    Data = mat['I1730'][:,0:5]
    Data = Data[~np.isnan(Data).any(axis=1)] #Remove nan value
    return Data

#Adding a time vector
# t_len = Data.shape[0];
# s = 0.001
# Time = np.arange(0,t_len*s,s).reshape(-1,1)
# sig1 = arr[:,0]
# sig2 = arr[:,1]
# Acc = np.hstack((Time, Data))


