#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 22 13:49:01 2023

@author: afernandez
"""
def mainGPU():
    import tensorflow as tf
    import numpy as np
    from MODULES.MODEL.model_creation import architecture_info_initializer
    from MODULES.MODEL.training import training_info_initializer, training_model, custom_loss
    from MODULES.PREPROCESSING.preprocessing import preprocessing_interface
    from MODULES.PREPROCESSING.preprocessing_tools import convert_to_npy
    from MODULES.POSTPROCESSING.postprocessing import postprocessing_info_initializer, postprocessing_interface
    from MODULES.POSTPROCESSING.Classification_results import make_predictions, classification_metrics, confusion_matrix, classification_results
    import os

    tf.keras.backend.clear_session()
    tf.random.set_seed(4321)
    # configure to run_functions in eagerly or not True only recomended for debug
    # tf.config.run_functions_eagerly(True)
    # ##############################################################################################

    # INITIAL CONSIDERATIONS

    n_s  = [10] #  number of points in each segment (segment length) dependes on the sampling freq (100Hz by now)
    for j in range(len(n_s)):
        # INITIAL CONSIDERATIONS
        filename = "17Aug_Classify_20Hz_shortened_types23_Correctsegments_nfp5"
        Wts = 3 #Numer of Wind turbines in the farm 
        n_features = 38  # number of input variables/features without wind vars
        n_steps =n_s[j] #at 5Hz of sampling freq. this corresponds to 4 seconds. And at 100 Hz --> 0.2 s
        # The number of steps (segment length) conditions the COnvolution Operations
        nslide = 3 # number of sliding points for the overlapping in the windows/segments
        # nfp =  int(n_s[j]/2) 
        nfp = 5# nfp +1 =  Number of faulty points in a segment to assign fault label to that segment 
        epochs = 1000
        batch_s = 1024
        LRate =1e-05
        Architecture = "Arch_3" 
        folder_name = filename + '_Farm'+str(Wts)+'WTs_'+str(n_features)+'features_'+Architecture+'_'+str(n_steps)+'steps_'+str(epochs)+'epoch_'+str(batch_s)+'batch'+str(LRate)+'LR'
        Xtrain_resc, Xval_resc, Xtest_resc, Ytrain, Yval, Ytest, Xmins, Xmaxs, A = preprocessing_interface(
            n_features, n_steps, nslide, nfp)
        #If working with one single wT, then retain only the corresponding features:
        # Xtrain_resc, Xval_resc, Xtest_resc = Xtrain_resc[:,:,(0,1,2,3,4,5,18,19,24,25,30,33,36,37)], Xval_resc[:,:,(0,1,2,3,4,5,18,19,24,25,30,33,36,37)], Xtest_resc[:,:,(0,1,2,3,4,5,18,19,24,25,30,33,36,37)]
        print(Xtrain_resc.shape)
        # we are working with one single WT
        # Initialize architecture properties and select the architectures FROM the DICTIONARY
        architecture_info = architecture_info_initializer(
            DeepClassifier=Architecture,
            # input_dim = Xtrain_resc.shape[1], #for standard DNN
            input_dim=(Xtrain_resc.shape[1], Xtrain_resc.shape[2]),  # for 1D_CNN and LSTM
            # input_dim =(Xtrain_resc.shape[1],Xtrain_resc.shape[2],1), #For 2D CNN
            # input_dim = (None, Xtrain_resc.shape[1],Xtrain_resc.shape[2]), #for CNNLSTM
            enc_dim=Ytrain.shape[1])
        
        ####################################################################################################################################
        # BUILD AND TRAIN THE MODELS
        # Initiailize the training properties
        training_info = training_info_initializer(
            n_epoch=epochs,
            batch_size=batch_s,
            LR=LRate,
            metrics="accuracy",
            loss="categorical_crossentropy",
            shuffle=True)
        # Build and train the models
        model, history = training_model(
            architecture_info, training_info, Xtrain_resc, Xval_resc, Ytrain, Yval, filename, folder_name)
        
        ###########################################################################################################################################################
        #POSTPROCESSING - RESULTS 
        #You must learn to load the model from its location and evaluate it 
        # model = tf.keras.models.load_model(os.path.join("Output",folder_name,"best_model.hdf5"))
        ######################################### MODEL EVALUATION #########################################
        # target_names = ['Healthy','Faulty']
        target_names = ['F1WT1', 'F2WT1', 'F1WT2', 'F2WT2', 'F1WT3', 'F2WT3']


        print(Xtrain_resc.shape[0])
        print(Xval_resc.shape[0])
        fnames = ['Train', 'Validation', 'Test']
        XJoint = [Xtrain_resc, Xval_resc, Xtest_resc]
        YJoint = [Ytrain, Yval, Ytest]
        
        for i in range(len(fnames)):
            fname = fnames[i]
            X = XJoint[i]
            y_true = YJoint[i]
            
            classification_results(model, X, y_true, target_names, folder_name, fname)
        
        
def create_npy_files():
    import os 
    import numpy as np
    from LOAD_DATA import import_mat_file, import_csv_file
    mat_directory= os.path.join("Data", "Data_mat")
    # mat_directory = os.path.join("Data", "mat_files")
    output_folder = os.path.join("Data","Data_current_npy_files")
    header_label = "saveVar" #depedens on the matlab file creation
    filenames = []
    count = 0
    for fname in os.scandir(mat_directory):
        filenames.append(str(fname)[11:-2])#To cut the string and extract only the text of interest
        filenames = sorted(filenames)
    for fname in filenames:#remove the Results folder that contains more scenarios
        mat_file = import_mat_file(os.path.join(mat_directory,fname), header_label)
        name = str(fname)[:11]
        # np.save(os.path.join(output_folder,  'Fault_'+str('%02d' % count)), mat_file)
        np.save(os.path.join(output_folder, name+'.npy'), mat_file)
        count = count+1
    
        