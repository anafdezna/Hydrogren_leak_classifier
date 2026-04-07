# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 10:52:37 2020

@author: 109457
"""
import numpy as np
from MODULES.POSTPROCESSING.postprocessing_tools import plot_configuration, plot_loss_evolution, plot_histogram, plot_predicted_values_vs_ground_truth, plot_crossplots, plot_outliers_dam
from MODULES.POSTPROCESSING.postprocessing_tools import cumulated_errors, calculate_errors, calculate_metrics  

#Llamar a la función de configruación de gráficos para tamaños/fuentes
plot_configuration()
########################################################################################################################################

class postprocessing_info_initializer():
    #definicion e inicialización de las variables que van dentro 
    def __init__(self, k = 1, percentile = 99):        
        self.k = k
        self.Percentile = percentile
        if k == None or percentile == None:
            print("************************************************************************")
            print("Please initialize the info for POSTPROCESSING!")
            print("************************************************************************")
            quit()

##############################################################################################################################3

def make_predictions(my_model, Xtrain_std, Xval_std, Xtest_std, Xtest_dam_std,filename):
    #RECONSTRUCTION ERROR AS THE SINGLE VALUE DAMAGE INDICATOR 
    #FROM THIS DI WE THEN CALCULATE THE METRICS AND COMPARE METRICS TABLE
    train_predictions = my_model.predict(Xtrain_std)  
    train_rec_error = calculate_errors(Xtrain_std,train_predictions)
    captured = (1-np.mean(train_rec_error))*100
    val_predictions = my_model.predict(Xval_std)
    val_rec_error = calculate_errors(Xval_std, val_predictions)
    test_predictions = my_model.predict(Xtest_std)
    test_rec_error = calculate_errors(Xtest_std, test_predictions)
    test_predictions_dam = my_model.predict(Xtest_dam_std)
    test_rec_error_dam = calculate_errors(Xtest_dam_std, test_predictions_dam)
    print('% of information captured\n', captured)
    print('Train reconstrunction error\n', np.mean(train_rec_error))
    print('Test reconstrunction error\n', np.mean(test_rec_error), np.mean(val_rec_error))
    return train_predictions, val_predictions, test_predictions, test_predictions_dam, train_rec_error, val_rec_error, test_rec_error, test_rec_error_dam


def make_plots(Xtrain_std, Xtest_std, train_predictions, val_predictions, test_predictions, train_rec_error, val_rec_error, test_rec_error, test_rec_error_dam, k, percentile, filename):

    Train_rec_error, Val_rec_error, Test_rec_error, Test_rec_error_dam, Test_rec_errors = cumulated_errors(train_rec_error, val_rec_error, test_rec_error, test_rec_error_dam, k)
    
    #Plot Histogram with the threshold value from the training set
    Lim = np.percentile(Train_rec_error, percentile)
    # plot_histogram(Train_rec_error, Lim, percentile)
    
    #Plot outliers controlchart
    # plot_outliers_dam(Train_rec_error, Test_rec_errors, percentile, filename)
    
    #Plot CROSSPLOTS (Ground truth vs Predictions)
    # plot_crossplots(Xtrain_std, Xtest_std, train_predictions, test_predictions)
    return Train_rec_error, Val_rec_error, Test_rec_error, Test_rec_error_dam, Test_rec_errors, Lim


def obtain_metrics(Test_rec_error, Test_rec_error_dam, Lim):
    FP,FN,TP,TN = calculate_metrics(Test_rec_error, Test_rec_error_dam, Lim)
    # print('Accuracy \n', accuracy)
    # print('Precision\n', precision)
    # print('recall\n', recall) recall precsión\\\\
    # print('F1 score\n', f1_score)
    print(FP,FN,TP,TN)

def predictions_plots_metrics(my_model, Xtrain_std, Xval_std, Xtest_std, Xtest_dam_std, k, percentile, filename):
    train_predictions, val_predictions, test_predictions, test_predictions_dam, train_rec_error, val_rec_error, test_rec_error, test_rec_error_dam = make_predictions(my_model, Xtrain_std, Xval_std, Xtest_std, Xtest_dam_std, filename)
    Train_rec_error, Val_rec_error, Test_rec_error, Test_rec_error_dam, Test_rec_errors, Lim = make_plots(Xtrain_std, Xtest_std, train_predictions, val_predictions, test_predictions, train_rec_error, val_rec_error, test_rec_error, test_rec_error_dam, k,percentile, filename)
    obtain_metrics(Test_rec_error, Test_rec_error_dam, Lim)
    
    return Train_rec_error, Val_rec_error, Test_rec_error, Test_rec_error_dam, Test_rec_errors, Lim
    
def postprocessing_interface(PCA, Residual, models, Xtrain_std, Xval_std, Xtest_std, Xtest_dam_std, postpro_info, filename):
    if PCA:
        Train_rec_error, Val_rec_error, Test_rec_error, Test_rec_error_dam, Test_rec_errors, Lim = predictions_plots_metrics(models[0], Xtrain_std, Xval_std, Xtest_std, Xtest_dam_std, postpro_info.k, postpro_info.Percentile, filename)
    if Residual:
        Train_rec_error, Val_rec_error, Test_rec_error, Test_rec_error_dam, Test_rec_errors, Lim = predictions_plots_metrics(models[1], Xtrain_std, Xval_std, Xtest_std, Xtest_dam_std, postpro_info.k, postpro_info.Percentile, filename)
    
    return Train_rec_error, Val_rec_error, Test_rec_error, Test_rec_error_dam, Test_rec_errors, Lim


