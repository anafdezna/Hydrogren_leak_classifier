#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 13:17:56 2022

@author: ana
"""

#REQUIERED PACKAGES (this will require installing libraries for sure)
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import signal
import matplotlib.pyplot as plt
import pyoma as oma
import os 
from scipy.signal import welch, find_peaks, csd
from scipy.fft import fft, fftfreq, ifft

########################################CABECERA #####################################################
#path = os.path.join() #specify .csv path of the input signal to process
#data = np.load(path) #load the input data to process: prepare the .csv to an array 2D
#preparation and input information regarding the signal
# data = data[:,3] #Activate to use one single sensor
# n  = data.shape[0]
# f_s = 12.5
# # f_s = 100
# T =1/f_s
# x = np.linspace(0.0, n*T, n)
# y = data




###############################################################################
# path = os.path.join("5min_20nov.npy")
# path = os.path.join("Data","5min_20nov.npy")
path = os.path.join("oneDay","20080101_prueba.npy")

data = np.load(path)
data = data[:,3] #Activate to use one single sensor
n  = data.shape[0]
f_s = 12.5
# f_s = 100
T =1/f_s
x = np.linspace(0.0, n*T, n)
y = data
f, Pxy = signal.csd(x, y, fs = f_s)
FFT = fft(y)[:n//2]
xf = fftfreq(n, T)[:n//2]
# xfn = np.linspace(0.0, 1.0/(2.0*T), n//2) #¡¡¡this operation does what fftfreq does!
plt.plot(xf,np.abs(FFT))

FDD = oma.FDDsvp(y,  f_s)
Results = FDD[0]
S_vals = Results['Singular Values']
Svalues = 10*np.log10(S_vals[0,0,:])
x_freqs = Results['x_freqs']
# plt.plot(x_freqs, Svalues)


#Finding teh peaks from the SValues

def peakpicking(csd_values):
    # peaks, properties = find_peaks(csd_values,  distance=1)
    peaks, properties  = find_peaks(x, prominence=(None, 0.6)) #probar hasta que funcione OK
    return peaks, properties

[peak_positions, properties] = peakpicking(Svalues)
peaks = x_freqs[peak_positions]
FreQ = peaks

# FreQ = [0.810,1.135,1.405, 1.993] #list of elements

# Extract the modal properties 
Res_FDD = oma.FDDmodEX(FreQ, FDD[1]) # extracting modal properties using standard FDD
Res_EFDD = oma.EFDDmodEX(FreQ, FDD[1], method='EFDD') # " " " " Enhanced-FDD

Freqs = Res_FDD['Frequencies']
Modes = Res_FDD['Mode Shapes'] #It is a complex number because modeshape has a variabiliyu (non totally linear system) . WE keep the linear part


###########################################################
#OUTPUT 
#Save freqs
#Save modes


################################################################################################################################################

#  EXAMPLE WITH SOUND SIGNAL 

dt = 0.001
t = np.arange(0,2,dt)
f0 = 50
f1 = 250
t1 = 2
x = np.cos(2*np.pi*t*(f0+(f1-f0)*np.power(t,2)/(3*t1**2))) #This is the input signal 
fs = 1/dt
# sd.play = (2*x,fs)
plt.specgram(x, NFFT = 128, Fs = fs, noverlap = 120, cmap = 'jet_r')
plt.colorbar()
plt.show()


# OUR CASE 
path = os.path.join("5min_20nov.npy")
data = np.load(path)
data = data[:,3] #Activate to use one single sensor
n  = data.shape[0]
fs = 12.5
T =1/fs
t = np.linspace(0.0, n*T, n)
x = data
plt.specgram(x, NFFT = 250, Fs = fs, noverlap = 120, cmap = 'jet_r')
# cbar = plt.colorbar()
# cbar.set_label('Amplitude',rotation = 270)
# plt.xlabel('Time [s]')
# plt.ylabel('Frequency [Hz]')
plt.show()



#Read the .mat file through scipy.io:
import scipy.io as io
matr = io.loadmat('11nov.mat')
#Assuming that'data' in the dictionary is the data you want:
data = matr['data']
io.savemat('11nov.mat',{'data':numpy_file})








































############NO FUCNIONA CON MI SEÑAL######################################################################################
import numpy as np
import matplotlib.pyplot as plt
import pywt
import pycwt as wavelet
from pycwt.helpers import find

time, sst = pywt.data.nino()
dt = time[1] - time[0]


wavelet = 'cmor1.5-1.0'
scales = np.arange(1, 128)

[cfs, frequencies] = pywt.cwt(sst, scales, wavelet, dt)
power = (abs(cfs)) ** 2

period = 1. / frequencies
levels = [0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8]
f, ax = plt.subplots(figsize=(15, 10))
ax.contourf(time, np.log2(period), np.log2(power), np.log2(levels),
            extend='both')

ax.set_title('%s Wavelet Power Spectrum (%s)' % ('Nino1+2', wavelet))
ax.set_ylabel('Period (years)')
Yticks = 2 ** np.arange(np.ceil(np.log2(period.min())),
                        np.ceil(np.log2(period.max())))
ax.set_yticks(np.log2(Yticks))
ax.set_yticklabels(Yticks)
ax.invert_yaxis()
ylim = ax.get_ylim()
ax.set_ylim(ylim[0], -1)

plt.show()


