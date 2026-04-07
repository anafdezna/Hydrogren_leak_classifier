# -*- coding: utf-8 -*-
"""
Created on Sun Jan 25 12:56:02 2026

@author: anafd
"""

import numpy as np
import pandas as pd
import os
import tensorflow as tf
from MODULES.POSTPROCESSING.Reviewer_comments_functions import calculate_spatial_metrics, plot_spatial_confusion_matrix, plot_error_distribution 
# ==========================================================
# MOCK DATA GENERATION (Replace this with your actual loading)
# ==========================================================
# In your main_server.py, you would load your .npy files:
# Y_pred = np.load("Output/folder/test_preds.npy")
# Y_true = np.load("Output/folder/true_test.npy")

# print("--- 1. Loading Data (Mocking for Demo) ---")
# N_SAMPLES = 1000
# N_CLASSES = 6

# # Create dummy Ground Truth (One-hot)
# y_true_indices = np.random.randint(0, N_CLASSES, N_SAMPLES)
# Y_true = np.eye(N_CLASSES)[y_true_indices]

# # Create dummy Predictions (Probabilities)
# # Let's make them mostly correct but with some neighbor errors
# Y_pred = np.zeros_like(Y_true)
# for i in range(N_SAMPLES):
#     true_cls = y_true_indices[i]
#     # 80% chance correct
#     if np.random.rand() > 0.2:
#         Y_pred[i, true_cls] = 0.9
#         # Small probability to neighbors
#         if true_cls > 0: Y_pred[i, true_cls-1] = 0.05
#         if true_cls < 5: Y_pred[i, true_cls+1] = 0.05
#     else:
#         # 20% chance error (mostly neighbor)
#         noise = np.random.randint(-2, 3) # Error of -2 to +2
#         pred_cls = np.clip(true_cls + noise, 0, 5)
#         Y_pred[i, pred_cls] = 0.8
#         Y_pred[i, true_cls] = 0.2

TANK_SPACING = 4.25 # From paper

from MODULES.PREPROCESSING.preprocessing import preprocessing_interface
from MODULES.POSTPROCESSING.Reviewer_comments_functions import predict_with_best_model
# ==========================================================
# ADDRESSING COMMENT 1: SPATIAL ERROR
# ==========================================================
print("\n--- Addressing Comment 1: Spatial Metrics ---")
folder_name = "50secsSim_Windows_06Feb20266TANKS10nfps50steps_60000epoch_1024batch1e-05LR"
Y_pred = np.load(os.path.join("Output", folder_name, "test_preds.npy"),allow_pickle = True)
output_path  = os.path.join("Output", folder_name)
Problem_info = np.load(os.path.join(output_path, 'Problem_info.npy'), allow_pickle = True).item()
n_steps, n_slide, n_classes, nfp = Problem_info['n_steps'], Problem_info['nslide'], Problem_info['n_classes'], Problem_info['nfp']

Xtrain_resc, Xval_resc, Xtest_resc, Ytrain, Yval, Ytest, Data_test = preprocessing_interface(
    4, n_steps, n_slide, nfp, n_classes, folder_name)
   
Y_true = Ytest

model, test_preds = predict_with_best_model(output_path, Xtest_resc)
Y_pred = test_preds
spatial_results = calculate_spatial_metrics(Y_true, Y_pred, TANK_SPACING)
plot_error_distribution(output_path, spatial_results, tank_spacing_meters=4.25)
print(f"Mean Spatial Error: {spatial_results['Mean_Spatial_Error_m']:.2f} meters")
print(f"Max Spatial Error: {spatial_results['Max_Spatial_Error_m']:.2f} meters")
print(f"Neighbor Errors: {spatial_results['Percent_Errors_Are_Neighbors']:.1f}% of total errors are just 1 tank away")

# Plot standard confusion matrix (you can save this)
class_names = [f'T{i+1}' for i in range(6)]
plot_spatial_confusion_matrix(Y_true, Y_pred, class_names,output_path)

target_names = ['Tank 1', 'Tank 2', 'Tank 3', 'Tank 4', 'Tank 5', 'Tank6']
TANK_SPACING = 4.25 #  from the layout description 
from MODULES.POSTPROCESSING.Classification_results import make_predictions, classification_metrics, confusion_matrix, classification_results
fnames = ['Train', 'Validation', 'Test']
XJoint = [Xtrain_resc, Xval_resc, Xtest_resc]
YJoint = [Ytrain, Yval, Ytest]

for i in range(len(fnames)):
    fname = fnames[i]
    X = XJoint[i]
    y_true = YJoint[i]
    
    classification_results(model, X, y_true, target_names, folder_name, fname)
 
from matplotlib import pyplot as plt
import seaborn as sns

def plot_quantized_confusion_matrix(matrix_data, class_names, output_path):
    """
    Rounds a quantized confusion matrix and plots it using the requested style.
    """
    # 1. Round to nearest integer as requested
    cm_rounded = np.rint(matrix_data).astype(int)
    
    # 2. Setup Plotting Environment
    plt.figure(figsize=(10, 8))
    
    # 3. Create Heatmap
    # Using 'Blues' cmap and fmt='d' (decimal/integer) to match your template
    sns.heatmap(
        cm_rounded, 
        annot=True, 
        fmt='d', 
        cmap='Blues', 
        xticklabels=class_names, 
        yticklabels=class_names,
        annot_kws={"size": 12}
    )
    
    # 4. Labels (Matching your 'True tank' vs 'Predicted tank' labels)
    plt.ylabel('True tank', fontsize=12, fontweight='bold')
    plt.xlabel('Predicted tank', fontsize=12, fontweight='bold')
    
    # Ensure directory exists
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        
    # 5. Save and Show
    save_file = os.path.join(output_path, 'quantized_confusion_matrix_3filters.png')
    plt.savefig(save_file, dpi=500, bbox_inches='tight')
    print(f"Plot saved to: {save_file}")
    plt.show()

if __name__ == "__main__":
    # The provided Mean Confusion Matrix (quantized)
    mean_cm_quantized = np.array([
            [2.03e+02, 2.83e+01, 0.00e+00, 0.00e+00, 9.00e-01, 1.80e+00],
            [8.56e+01, 1.43e+02, 1.40e+00, 0.00e+00, 4.10e+00, 5.90e+00],
            [0.00e+00, 1.25e+01, 2.19e+02, 8.50e+00, 0.00e+00, 0.00e+00],
            [0.00e+00, 1.20e+00, 5.63e+01, 1.74e+02, 8.20e+00, 0.00e+00],
            [8.90e+00, 3.19e+01, 8.10e+00, 3.73e+01, 1.49e+02, 5.20e+00],
            [6.00e+00, 1.00e-01, 0.00e+00, 0.00e+00, 2.33e+01, 2.11e+02]
        ])
    # Configuration based on your provided context
    folder_name = '50secsSim_Windows_06Feb20266TANKS2nfps50steps_60000epoch_1024batch1e-05LR'
    output_path = os.path.join("Output", folder_name)
    class_names = [f'T{i+1}' for i in range(6)]

    # Execute plotting
    plot_quantized_confusion_matrix(mean_cm_quantized, class_names, output_path)
    
 
    
 
    
 


# ==========================================================
# ADDRESSING COMMENT 2: TEMPORAL ANALYSIS
# ==========================================================

from MODULES.POSTPROCESSING.Reviewer_comments_functions import analyze_temporal_performance, _calculate_single_sim_metrics, plot_ttfd_distribution, plot_stability_by_tank
# ==========================================
# CALLING CODE (Copy-Paste into your main script or console)
# ==========================================

# 1. Ensure your Datatest is loaded and has shape (1440, 50, 7)
# 2. Run the analysis (pass your trained 'model' object)
print("Starting Temporal Analysis...")
df_metrics = analyze_temporal_performance(model, Datatest, dt=0.32, n_simulations=6)

# 3. Print Summary Statistics for the paper
print("\n--- TEMPORAL METRICS SUMMARY ---")
print(f"Mean TTFD: {df_metrics['TTFD_seconds'].mean():.2f}s")
print(f"Median TTFD: {df_metrics['TTFD_seconds'].median():.2f}s")
print(f"Avg Stability: {df_metrics['Stability'].mean():.2f}")
print(f"Detection Success Rate: {(len(df_metrics[df_metrics['Status']=='Detected'])/len(df_metrics))*100:.1f}%")

# 4. Generate Plots
plot_ttfd_distribution(df_metrics)
plot_stability_by_tank(df_metrics)



print("\n--- Addressing Comment 2: Temporal Analysis ---")
# You need to group your data by Simulation ID. 
# Since your current script splits by windows, you might need to reconstruct this 
# or use your 'Data_test' object if it contains simulation IDs.

# Mocking a list of continuous simulations
# Imagine we have 10 simulations, each 50 time steps long
simulations_data = []
for sim_id in range(10):
    # True tank for this simulation
    true_tank = np.random.randint(0, 6)
    
    # Create a time series prediction
    # Initially confused, then becomes accurate
    preds = np.zeros((50, 6))
    for t in range(50):
        if t < 10: # First 10 steps: Random/Confused
             preds[t, np.random.randint(0,6)] = 1.0
        else: # After 10 steps: Mostly correct
             if np.random.rand() > 0.1: # Occasional flicker
                 preds[t, true_tank] = 1.0
             else:
                 preds[t, (true_tank+1)%6] = 1.0 # Wrong flicker
                 
    simulations_data.append((preds, true_tank))

# Calculate metrics
temp_summary, ttfd_dist = aggregate_temporal_results(simulations_data)

print(f"Mean Time to Detect: {temp_summary['Mean_Time_To_Detect_s']:.2f}s")
print(f"Stability Score: {temp_summary['Avg_Stability_Score']:.2f} (1.0 is perfect)")
print("Reviewer Tip: Plot a histogram of 'ttfd_dist' for the paper!")


# ==========================================================
# ADDRESSING COMMENT 3: SPLITTING CHECK
# ==========================================================
print("\n--- Addressing Comment 3: Splitting Validation ---")

# Mock simulation IDs for dataset
# 0-99 are from Sim 1, 100-199 from Sim 2...
simulation_ids = np.repeat(np.arange(10), 100) # 1000 samples total

# Mock indices for train/test (Random Split - The "Bad" Way)
indices = np.arange(1000)
np.random.shuffle(indices)
train_idx = indices[:800]
test_idx = indices[800:]

print("Checking Random Split (Current Method?):")
is_leaking = check_data_leakage(train_idx, test_idx, simulation_ids)

if is_leaking:
    print("\n>>> RECOMMENDATION FOR CODE <<<")
    print("You should use GroupKFold in your 'preprocessing_interface'.")
    print("Example Code:")
    print("from sklearn.model_selection import GroupKFold")
    print("gkf = GroupKFold(n_splits=5)")
    print("train_idx, val_idx = next(gkf.split(X, y, groups=simulation_ids))")


# ==========================================================
# ADDRESSING COMMENT 4: PARETO TABLE
# ==========================================================
print("\n--- Addressing Comment 4: Accuracy vs Delay ---")

# You likely have these numbers from your experiments (Table 1 in paper)
# I will create a dataframe representing them
pareto_data = {
    'nt': [5, 10, 30, 50, 100],
    'accuracy': [0.69, 0.74, 0.84, 0.87, 0.98], # Values from L_sim=300 column in your Table 1
    'dt': [0.32, 0.32, 0.32, 0.32, 0.32]
}
df_pareto = pd.DataFrame(pareto_data)
df_pareto['latency_s'] = df_pareto['nt'] * df_pareto['dt']

print(df_pareto[['nt', 'latency_s', 'accuracy']])
plot_pareto_accuracy_latency(df_pareto)