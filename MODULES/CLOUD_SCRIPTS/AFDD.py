# -*- coding: utf-8 -*-
"""
Created on Thu Oct 13 12:09:42 2022

@author: 110137
"""

#REQUIERED PACKAGES (this will require installing libraries for sure)
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
# import pandas as pd
# import seaborn as sns
# from scipy import linalg as LA
# from scipy.optimize import curve_fit
# from matplotlib.ticker import (MultipleLocator, FormatStrFormatter)
# import matplotlib.patches as patches
# import pyoma as oma
# import math

#############################################################################
# Encuentra los picos de la funcion csd
def peakpicking(csd_values):
    dist = 100
    prom = 0.0000002
    # prominence = 0.5

    peaks, properties = signal.find_peaks(csd_values, prominence = prom, distance = dist)
    return peaks, properties

# # Compute frequencies
# def frequencies_list(f_values, peaks):
#     return f_values[peaks]

# Extract temporal data from sampligs
def extract_sampling_features(data_time):
    t_n = data_time[len(data_time) - 1]  # Tiempo total del muestreo en segundos
    n = len(data_time)
    T = t_n / n
    f_s = 1 / T

    return t_n, n, T, f_s


#Definir iputs (data, sampling_freq )
# filename = "prueba0"
# path = os.path.join() #specify .csv path of the input signal to process
# data = np.load(path) #load the input data to process: prepare the .csv to an array 2D
#preparation and input information regarding the signal
# data = data[:,3] #Activate to use one single sensor
# n  = data.shape[0]
# f_s = 1000 #Sampling fequency of the acceleration record 
# f_s = 100
# T =1/f_s
############################################################################

def FDDsvp(data, fs, df, pov=0.5, window='hann'):
    '''
    This function perform the Frequency Domain Decomposition algorithm.
    
    The function return the plot of the singular values of the power spectral
    density. The cross power spectral density is estimated using 
    scipy.signal.csd() function, which in turn is based on Welch's method.
    Furthermore it returns a dictionary that contains the results needed
    by the function FDDmodEX().
    ----------
    Parameters
    ----------
    data : array
        The time history records (Ndata x Nchannels).
    fs : float
        The sampling frequency.
    df : float
        Desired frequency resolution. Default to 0.01 (Hz).
    '''  
    
    # ndat=data.shape[0] # Number of data points
    nch=data.shape[1] # Number of channels
    freq_max = fs/2 # Nyquist frequency
    nxseg = fs/df # number of point per segments
#    nseg = ndat // nxseg # number of segments
    noverlap = nxseg // (1/pov) # Number of overlapping points
    
    # Initialization
    PSD_matr = np.zeros((nch, nch, int((nxseg)/2+1)), dtype=complex) 
    S_val = np.zeros((nch, nch, int((nxseg)/2+1))) 
    S_vec = np.zeros((nch, nch, int((nxseg)/2+1)), dtype=complex) 
    
    # Calculating Auto e Cross-Spectral Density
    for _i in range(0, nch):
        for _j in range(0, nch):
            _f, _Pxy = signal.csd(data[:, _i],data[:, _j], fs=fs, nperseg=nxseg, noverlap=noverlap, window=window)
            PSD_matr[_i, _j, :] = _Pxy
            
    # Singular value decomposition     
    for _i in range(np.shape(PSD_matr)[2]):
        U1, S1, _V1_t = np.linalg.svd(PSD_matr[:,:,_i])
        U1_1=np.transpose(U1) 
        S1 = np.diag(S1)
        S_val[:,:,_i] = S1
        S_vec[:,:,_i] = U1_1

    
    Results={}
    Results['Data'] = {'Data': data}
    Results['Data']['Samp. Freq.'] = fs
    Results['Data']['Freq. Resol.'] = df
    Results['Singular Values'] = S_val
    Results['Singular Vectors'] = S_vec
    Results['PSD Matrix'] = PSD_matr
    Results['x_freqs'] = _f
    
    return Results

# Extract the modal properties 
def FDDmodEX(FreQ, Results, ndf=10):
    '''
    This function returns the modal parameters estimated according to the
    Frequency Domain Decomposition method.
    
    ----------
    Parameters
    ----------
    FreQ : array (or list)
        Array containing the frequencies, identified from the singular values
        plot, which we want to extract.
    Results : dictionary
        Dictionary of results obtained from FDDsvp().
    ndf : float
        Number of spectral lines in the proximity of FreQ[i] where the peak
        is searched.
    -------
    Returns
    -------
    fig1 : matplotlib figure
        Stabilisation diagram ...
    Results : dictionary
        Dictionary of results ...
    '''
    
#    data = Results['Data']['Data']
    fs = Results['Data']['Samp. Freq.']
    df = Results['Data']['Freq. Resol.']
    S_val = Results['Singular Values']
    S_vec = Results['Singular Vectors']
    deltaf=ndf*df
#    ndat=data.shape[0] #
#    nch=data.shape[1] #
    freq_max = fs/2 # Nyquist
#    nxseg = fs/df # 

    f = np.linspace(0, int(freq_max), int(freq_max*(1/df)+1)) # spectral lines
 
    Freq = []
    index = []
    Fi = []

    for _x in FreQ:
#        idx = np.argmin(abs(f-_x)) 
        lim = (_x - deltaf, _x + deltaf) # frequency bandwidth where the peak is searched
        idxlim = (np.argmin(abs(f-lim[0])), np.argmin(abs(f-lim[1])))
        # ratios between the first and second singular value 
        diffS1S2 = S_val[0,0,idxlim[0]:idxlim[1]]/S_val[1,1,idxlim[0]:idxlim[1]]
        maxDiffS1S2 = np.max(diffS1S2) # looking for the maximum difference
        idx1 = np.argmin(abs(diffS1S2 - maxDiffS1S2))
        idxfin = idxlim[0] + idx1 
# =============================================================================
        # Modal properties
        fr_FDD = f[idxfin] # Frequency
        fi_FDD = S_vec[0,:,idxfin] # Mode shape
        idx3 = np.argmax(abs(fi_FDD))
        fi_FDDn = fi_FDD/fi_FDD[idx3] # normalised (unity displacement)
        fiFDDn = np.array(fi_FDDn)
        
        Freq.append(fr_FDD)
        Fi.append(fiFDDn)
        index.append(idxfin)
        
    Freq = np.array(Freq)
    Fi = np.array(Fi)
    index = np.array(index)   
        
    Results={}
    Results['Frequencies'] = Freq
    Results['Mode Shapes'] = Fi.T
    Results['Freq. index'] = index

    return Results


def Obtain_Freqs(n, T, f_s, data):
    x = np.linspace(0.0, n*T, n)
    y = data
    df = f_s*2/n #resolution in frequency domain

    FDD = FDDsvp(y, f_s, df) 
    Results = FDD
    S_vals = Results['Singular Values']
    PSD =  Results['PSD Matrix']
    # Svalues = 10*np.log10(S_vals[0,0,:])
    PSDVals = PSD[1,1,:].real
    # PSDModule = math.sqrt(np.sum(PSD[0,0,:].real**2 + PSD[0,0].imag**2))
    x_freqs = Results['x_freqs']
    plt.plot(x_freqs, PSDVals)

    #Find peaks in CSD
    peaks, properties = peakpicking(PSDVals)

    FreQ = x_freqs[peaks]
    return FreQ, FDD


def Obtain_Modes(FreQ,FDD):
    Res_FDD = FDDmodEX(FreQ, FDD) # extracting modal properties using standard FDD
    final_Freqs = Res_FDD['Frequencies']
    Modes = Res_FDD['Mode Shapes'] #It is a complex number because modeshape has a variabiliyu (non totally linear system) . WE keep the linear part
    final_Modes = Modes.real
    print(final_Freqs)
    return final_Freqs, final_Modes


#####################################################################################

