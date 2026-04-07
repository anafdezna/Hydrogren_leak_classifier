# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 10:52:37 2020

@author: 109457
"""
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 10:52:37 2020

@author: 109457
"""
import numpy as np 
import pandas
import seaborn
import scipy
import sklearn
import os 
import matplotlib
from matplotlib import pyplot as plt


#cumulated damage indicator with k 
def cumulated_errors(train_rec_error, val_rec_error, test_rec_error, test_rec_error_dam, k):
    Train_rec_error = np.zeros(shape = (len(train_rec_error)-k,1))
    for i in range(Train_rec_error.shape[0]):
        Train_rec_error[i,:] =  (np.sum(train_rec_error[i:(k+i),:]))/k
    
    Val_rec_error = np.zeros(shape = (len(val_rec_error)-k,1))
    for i in range(Val_rec_error.shape[0]):
        Val_rec_error[i,:] =  (np.sum(val_rec_error[i:(k+i),:]))/k
    
    Test_rec_error = np.zeros(shape = (len(test_rec_error)-k,1))
    for i in range(Test_rec_error.shape[0]):
        Test_rec_error[i,:] =  (np.sum(test_rec_error[i:(k+i),:]))/k
    
    Test_rec_error_dam= np.zeros(shape = (len(test_rec_error_dam)-k,1))
    for i in range(Test_rec_error_dam.shape[0]):
        Test_rec_error_dam[i,:] =  (np.sum(test_rec_error_dam[i:(k+i),:]))/k
    
    Test_rec_errors = np.concatenate((Test_rec_error,Test_rec_error_dam))
    return Train_rec_error, Val_rec_error, Test_rec_error, Test_rec_error_dam, Test_rec_errors

# --- MODIFICACIÓN AQUÍ: Desactivamos LaTeX externo y usamos MathText interno ---
plt.rcParams.update({
    "text.usetex": False,        # <--- CAMBIADO A FALSE
    "font.family": "serif",
    "mathtext.fontset": "cm",    # <--- AÑADIDO: Usa fuente estilo LaTeX interna
    "font.size": 18,
})

#Graph configuration function (font sizes)
def plot_configuration():
    # Configuración global de fuentes
    plt.rc('font', size=14, family='serif') 
    
    # --- MODIFICACIÓN CRÍTICA ---
    plt.rc('text', usetex=False) # <--- CAMBIADO A FALSE
    plt.rcParams['mathtext.fontset'] = 'cm' # <--- Estilo matemático
    # -----------------------------

    plt.rc('axes', titlesize = 16)     # fontsize of the axes title
    plt.rc('axes', labelsize = 14)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize = 14)    # fontsize of the tick labels
    plt.rc('ytick', labelsize = 14)    # fontsize of the tick labels
    plt.rc('legend', fontsize = 14)   # legend fontsize
    plt.rc('figure', titlesize= 14)   # fontsize of the figure title
    
    plt.rcParams.update({
        "text.usetex": False,       # <--- CAMBIADO A FALSE
        "font.family": "serif",
        "mathtext.fontset": "cm",   # <--- AÑADIDO
        "font.size": 14,
    })


#Código gráficos
def plot_predicted_values_vs_ground_truth(gt_array, pred_array, title_label, filename):
    min_val,max_val = -3,3 #for common scaling
    increment = max_val - min_val 
    limits = (min_val -0.05*increment, max_val + 0.05*increment)
    
    x,y = pandas.Series(gt_array,name = r'Ground Truth'),pandas.Series(pred_array, name = r'Predicted')
    g = seaborn.jointplot(x=x, y=y, kind = 'hex', color = '#1d6d68', joint_kws = {'gridsize':30,'bins':'log'},xlim=limits, ylim = limits)
    g.ax_joint.plot(np.linspace(limits[0], limits[1]), np.linspace(limits[0], limits[1]),'--r', linewidth=4)
    g.set_axis_labels(xlabel = r'Ground truth', ylabel = r'Predicted', fontsize = 18)
    g.fig.suptitle(title_label, y = 0.99)
    g.ax_joint.set_xticklabels([])
    g.ax_joint.set_yticklabels([])

    #############################################################################################
    #Introducir el r^2 
    
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(gt_array,pred_array)
    textstr = '\n'.join((
        r'$\mathrm{r^2}=%.4f$'%(r_value**2,),
        ))
    
    #These are matplotlib.patch.Patch properties
    props = dict(boxstyle = 'round', alpha = 0.5, facecolor = 'none')
    # Place a textbox in upper left in axes coords
    g.ax_joint.text(0.05,0.95, textstr, transform  = g.ax_joint.transAxes, fontsize = 18, verticalalignment = 'top', bbox=props)
    ###########################################################################################
    plt.tight_layout()
    # Asegurar que el directorio existe antes de guardar
    os.makedirs(os.path.join("Output","Figures","pruebas"), exist_ok=True)
    g.fig.savefig(os.path.join("Output","Figures","pruebas", filename),dpi = 600)


def plot_crossplots(Xtrain_std,Xtest_std,train_predictions,test_predictions):
    for i in range(Xtrain_std.shape[1]):
         plot_predicted_values_vs_ground_truth(Xtrain_std[:,i],train_predictions[:,i],'', 'Crossplot_TrainS'+str(i+1))
    for i in range(Xtest_std.shape[1]):
        plot_predicted_values_vs_ground_truth(Xtest_std[:,i],test_predictions[:,i],'', 'Crossplot_TestS'+str(i+1)) 


def plot_outliers_dam(Train_rec_error, Test_rec_errors, percentile, filename):
    x = np.arange(len(Test_rec_errors))
    Lim = np.percentile(Train_rec_error,percentile)
    col = np.where(Test_rec_errors<Lim,'g','r')
    C_chart, ax = plt.subplots()
    plt.axvspan(np.round(0.5*Test_rec_errors.shape[0]), Test_rec_errors.shape[0], facecolor='lightgrey', alpha=0.5, zorder = 0)
    for i in range( Test_rec_errors.shape[0]):
        plt.scatter(x[i], Test_rec_errors[i], s = 2, c=col[i])
    
    plt.ylabel('Damage indicator \u03C1 ')
    plt.xlabel('Measurement')
    text_ypos = Lim + 0.135
    C_chart.text(0.24,text_ypos, '\u03B1 = '+str(round(Lim,2)), color = '#0000FF')
    plt.plot(1.5)
    limit = np.ones(len(x))
    plt.plot(Lim*limit, color = '#0000FF', markersize = 8)
    
    os.makedirs(os.path.join("Output","Figures", "pruebas"), exist_ok=True)
    C_chart.savefig(os.path.join("Output","Figures", "pruebas", "Controlchart_D"+str(filename)+".png"), dpi = 500, bbox_inches='tight')
    plt.show()


def plot_loss_evolution(history, filename, folder_name):
    loss_plot  = plt.figure()
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    plt.plot(loss,color = '#072FD1',)
    plt.plot(val_loss, color = 'red')
    plt.ylabel(r'$\mathcal{L}_{\theta^{*}}$(log)')
    plt.xlabel('Epoch')
    xmin, xmax = plt.xlim()
    plt.yscale('log')
    plt.legend(['Train', 'Validation'], loc='lower left', fontsize = 18)
    
    os.makedirs(os.path.join('Output',folder_name), exist_ok=True)
    loss_plot.savefig(os.path.join('Output',folder_name,'Loss.png'), dpi=500, bbox_inches='tight')
    plt.show()


def plot_histogram(Train_rec_error, Lim, percentile):
    Histo = plt.figure()
    plt.hist(Train_rec_error, bins = 16, color = 'skyblue', alpha= 0.5, histtype='bar', ec='black')
    plt.axvline(x = Lim, ymax = 0.55, color = "red",linestyle='--', linewidth = '3')
    plt.axvline(x = Lim, ymin = 0.7, color = "red",linestyle='--', linewidth = '3')
    plt.xlabel('Reconstruction error')
    plt.ylabel('Frequency')
    Histo.text(0.32, 0.58, 'p-99 = '+str(round(Lim,2)), color = 'red')
    
    os.makedirs(os.path.join("Output","Figures", "pruebas"), exist_ok=True)
    Histo.savefig(os.path.join("Output","Figures", "pruebas","Train_hist"),dpi = 500, bbox_inches='tight')
    plt.show()
    return Lim

def calculate_errors(Xstd, predictions):
    rec_error = np.zeros(shape = [ Xstd.shape[0],1])
    for i in range(Xstd.shape[0]):
        rec_error[i] = 1/(Xstd.shape[1])*np.sum((Xstd[i,:] - predictions[i,:])**2)
    return rec_error


def calculate_metrics(Test_rec_error, Test_rec_error_dam, Lim):
    FP = len(np.where(Test_rec_error > Lim)[0])
    FN = len(np.where(Test_rec_error_dam < Lim)[0])
    TP = len(np.where(Test_rec_error_dam>Lim)[0])
    TN = len(np.where(Test_rec_error<Lim)[0])
    print(FP,FN,TP,TN)
    return FP,FN,TP,TN