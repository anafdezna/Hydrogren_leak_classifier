#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 22 13:49:01 2023

@author: afernandez
"""
def main():
    import os
    import tensorflow as tf
    import keras as K
    import numpy as np
    from MODULES.MODEL.model_creation import architecture_info_initializer
    from MODULES.MODEL.training import training_info_initializer, training_model, custom_loss
    from MODULES.PREPROCESSING.preprocessing import preprocessing_interface
    from MODULES.PREPROCESSING.preprocessing_tools import convert_to_npy
    from MODULES.POSTPROCESSING.postprocessing import postprocessing_info_initializer, postprocessing_interface
    from MODULES.POSTPROCESSING.Classification_results import make_predictions, classification_metrics, confusion_matrix, classification_results
    
    tf.keras.backend.clear_session()
    tf.random.set_seed(234)
    # configure to run_functions in eagerly or not True only recomended for debug
    # tf.config.run_functions_eagerly(True)
    # ##############################################################################################
    
    # INITIAL CONSIDERATIONS
    rand_seed = 1234
    n_s = 55
    nfps = [10] # nfp +1 =  Number of faulty points in a segment to assign fault label to that segment 

    # n_s  = [50] #  number of points in each segment (segment length) dependes on the sampling freq (100Hz by now)
    #f_s = 1/0.3s = 3.3Hz
    # 100 points son 30s de medida
    for j in range(len(nfps)):
        ssed = rand_seed
        tf.random.set_seed(ssed)
        # INITIAL CONSIDERATIONS
        filename = r"300p_12Feb2026"+str(n_s)+"steps"
        Ntanks = 6 #Number of  tanks in the system 
        n_features = 4  # number of  sensors
        n_steps =n_s # duration of the input sample (window)
        # The number of steps (segment length) conditions the COnvolution Operations
        nslide = 1 # number of sliding points for the overlapping in the windows/segments
        # nfp =  int(n_s[j]/2) 
        nfp = nfps[j] # nfp +1 =  Number of faulty points in a segment to assign fault label to that segment 
        n_classes = 6 #number of classes to be classified (the number of tanks considered)
        epochs = 80000
        batch_s = 1024
        LRate = 1e-05
        Architecture = "Arch_3" 
        Problem_info = {
            'n_features': n_features,
            'n_steps': n_steps,
            'nslide': nslide,
            'nfp': nfp,
            'n_classes' : n_classes,
            'epochs': epochs,
            'batch_s':batch_s,
            'LRate': LRate,
            'Arch': Architecture          
            }
        folder_name = filename +str(Ntanks)+'TANKS'+str(nfp)+'nfps'+str(n_steps)+'steps_'+str(epochs)+'epoch_'+str(batch_s)+'batch'+str(LRate)+'LR'
        directory_path = os.path.join('Output', folder_name)
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)
        
        output_path = os.path.join('Output', folder_name)
        
        np.save(os.path.join('Output', folder_name, 'Problem_info.npy'), Problem_info, allow_pickle=True)
        
        Xtrain_resc, Xval_resc, Xtest_resc, Ytrain, Yval, Ytest, Data_test = preprocessing_interface(
            n_features, n_steps, nslide, nfp, n_classes, folder_name)
       
        print(Xtrain_resc.shape)
        
        # Initialize architecture properties and select the architectures FROM the DICTIONARY
        architecture_info = architecture_info_initializer(
            DeepClassifier=Architecture,
            input_dim=(Xtrain_resc.shape[1], Xtrain_resc.shape[2]),  # for 1D_CNN and LSTM
            enc_dim = Ytrain.shape[1])
        
        ####################################################################################################################################
        # BUILD AND TRAIN THE MODELS
        # Initiailize the training properties
        training_info = training_info_initializer(
            n_epoch=epochs,
            batch_size=batch_s,
            LR=LRate,
            metrics="accuracy",
            loss="categorical_crossentropy",
            shuffle=True,
            train_flag = True
            )
    
        # Build and train the models
        model, history = training_model(
            architecture_info, training_info, Xtrain_resc, Xval_resc, Ytrain, Yval, folder_name)
        
        
        # --- ADD THIS SECTION TO SAVE THE MODEL ---
        model_filename = "final_model.h5" # Or .keras for newer TF versions
        model_save_path = os.path.join(directory_path, model_filename)

        # Save the full model (architecture + weights + optimizer state)
        model.save(model_save_path)
        print(f"Model successfully saved to: {model_save_path}")

   
        from MODULES.POSTPROCESSING.postprocessing_tools import plot_loss_evolution
        plot_loss_evolution(history,filename, folder_name)
        ###########################################################################################################################################################
        # POSTPROCESSING - RESULTS 
        Ytrain_pred = model.predict(Xtrain_resc)
        Yval_pred = model.predict(Xval_resc)
        Ytest_pred = model.predict(Xtest_resc)
    
        # You must learn to load the model from its location and evaluate it 
        # model = tf.keras.models.load_model(os.path.join("Output",folder_name,"best_model.hdf5"))
        ######################################## MODEL EVALUATION #########################################
        target_names = ['Tank 1', 'Tank 2', 'Tank 3', 'Tank 4', 'Tank 5', 'Tank6']
        Y_true = Ytest
        Y_pred = Ytest_pred
        TANK_SPACING = 4.25 #  from the layout description 

        
        from MODULES.POSTPROCESSING.Reviewer_comments_functions import calculate_spatial_metrics, plot_spatial_confusion_matrix, plot_error_distribution 
        spatial_results = calculate_spatial_metrics(Y_true, Y_pred, TANK_SPACING)
        plot_error_distribution(output_path, spatial_results, tank_spacing_meters=4.25)
        print(f"Mean Spatial Error: {spatial_results['Mean_Spatial_Error_m']:.2f} meters")
        print(f"Max Spatial Error: {spatial_results['Max_Spatial_Error_m']:.2f} meters")
        print(f"Neighbor Errors: {spatial_results['Percent_Errors_Are_Neighbors']:.1f}% of total errors are just 1 tank away")

        # Plot standard confusion matrix (you can save this)
        class_names = [f'T{i+1}' for i in range(6)]
        plot_spatial_confusion_matrix(Y_true, Y_pred, class_names,output_path)


    
        # print(Xtrain_resc.shape[0])
        # print(Xval_resc.shape[0])
        fnames = ['Train', 'Validation', 'Test']
        XJoint = [Xtrain_resc, Xval_resc, Xtest_resc]
        YJoint = [Ytrain, Yval, Ytest]
        
        for i in range(len(fnames)):
            fname = fnames[i]
            X = XJoint[i]
            y_true = YJoint[i]
            
            classification_results(model, X, y_true, target_names, folder_name, fname)
     
        
     
        
         #FInal savings
    
        
        train_path = os.path.join("Output",folder_name, "train_preds.npy")
        np.save(train_path, Ytrain_pred)
        val_path = os.path.join("Output",folder_name, "val_preds.npy")
        np.save(val_path, Yval_pred)
        test_path = os.path.join("Output",folder_name, "test_preds.npy")
        np.save(test_path, Ytest_pred)
        
        true_train_path = os.path.join("Output",folder_name, "true_train.npy")
        np.save(true_train_path, Ytrain)
        true_val_path = os.path.join("Output",folder_name, "true_val.npy")
        np.save(true_val_path, Yval)
        true_test_path = os.path.join("Output",folder_name, "true_test.npy")
        np.save(true_test_path, Ytest)
        
        
        from MODULES.POSTPROCESSING.plot_results import Plot_time_domain_probability_evolution
        Plot_time_domain_probability_evolution(Data_test, model, n_features, n_s, nslide, output_path)
        
        


############## This operation allows to exeute functions in the script     
if __name__ == "__main__":
    
    main()


        
        

    
        