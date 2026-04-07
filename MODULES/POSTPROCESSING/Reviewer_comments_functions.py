# -*- coding: utf-8 -*-
"""
Created on Sun Jan 25 12:53:51 2026

@author: anafd
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import tensorflow as tf

# ==========================================
# COMMENT 1: SPATIAL ERROR METRIC
# ==========================================
# def calculate_spatial_metrics(y_true, y_pred_probs, tank_spacing_meters=4.25):
#     """
#     Calculates error metrics weighted by physical distance.
#     Assumes classes 0-5 correspond to T1-T6 in a linear column.
#     """
#     y_pred = np.argmax(y_pred_probs, axis=1)
#     y_true_idx = np.argmax(y_true, axis=1)
    
#     # Calculate index distance (how many tanks away?)
#     index_diff = np.abs(y_pred - y_true_idx)
    
#     # Convert to physical distance (meters)
#     physical_error = index_diff * tank_spacing_meters
    
#     # Metrics
#     mean_spatial_error = np.mean(physical_error)
#     max_spatial_error = np.max(physical_error)
    
#     # "Safe" Error Rate: % of errors that are only 1 neighbor away
#     errors_mask = index_diff > 0
#     if np.sum(errors_mask) > 0:
#         neighbor_errors = np.sum(index_diff[errors_mask] == 1)
#         pct_neighbor_errors = (neighbor_errors / np.sum(errors_mask)) * 100
#     else:
#         pct_neighbor_errors = 0.0
        
#     results = {
#         "Mean_Spatial_Error_m": mean_spatial_error,
#         "Max_Spatial_Error_m": max_spatial_error,
#         "Percent_Errors_Are_Neighbors": pct_neighbor_errors
#     }
    
#     return results


def predict_with_best_model(folder_path, X_new_data):
    """
    Loads best_model.hdf5 and generates predictions.
    """
    model_path = os.path.join(folder_path, "best_model.hdf5")
    
    if not os.path.exists(model_path):
        print(f"File not found: {model_path}")
        return None

    print(f"Attempting to load: {model_path}")

    # SCENARIO 1: Loading the Full Model (Recommended)
    # compile=False allows us to load the model structure and weights 
    # without needing the specific 'custom_loss' function definition.
    # This is perfect for prediction/inference.
    try:
        model = tf.keras.models.load_model(model_path, compile=False)
        print("Success: 'best_model.hdf5' loaded (compile=False).")
    except Exception as e:
        print(f"Error loading full model: {e}")
        return None

    # Predict
    print("Generating predictions on new data...")
    predictions = model.predict(X_new_data)
    return model, predictions

def plot_spatial_confusion_matrix(y_true, y_pred_probs, class_names, output_path):
    """
    Plots a confusion matrix where cell color intensity can represent 
    count, but we overlay the distance penalties.
    """
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true_idx = np.argmax(y_true, axis=1)
    
    cm = confusion_matrix(y_true_idx, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    # plt.title("Confusion matrix (Standard)")
    plt.ylabel('True tank')
    plt.xlabel('Predicted tank')
    plt.savefig(os.path.join(output_path, 'spatial_confusion_matrix.png'), dpi=500, bbox_inches='tight')
    plt.show()


def calculate_spatial_metrics(y_true, y_pred_probs, tank_spacing_meters=4.25):
    """
    Calculates spatial metrics and returns raw error data for plotting.
    
    Args:
        y_true: One-hot encoded truth (N_samples, N_classes)
        y_pred_probs: Predicted probabilities (N_samples, N_classes)
        tank_spacing_meters: Distance between adjacent tanks (default 4.25m)
    
    Returns:
        results: Dictionary containing summary metrics and raw arrays.
    """
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true_idx = np.argmax(y_true, axis=1)
    
    # Calculate index distance (0 = Correct, 1 = Neighbor, etc.)
    index_diff = np.abs(y_pred - y_true_idx)
    
    # Convert to physical distance (meters)
    physical_error = index_diff * tank_spacing_meters
    
    # --- Metrics Calculation ---
    mean_spatial_error = np.mean(physical_error)
    max_spatial_error = np.max(physical_error)
    
    # Filter for ONLY errors (distance > 0) to analyze misbehavior
    errors_mask = index_diff > 0
    if np.sum(errors_mask) > 0:
        neighbor_errors = np.sum(index_diff[errors_mask] == 1)
        pct_neighbor_errors = (neighbor_errors / np.sum(errors_mask)) * 100
        avg_error_distance_when_wrong = np.mean(physical_error[errors_mask])
    else:
        pct_neighbor_errors = 0.0
        avg_error_distance_when_wrong = 0.0
        
    results = {
        "Mean_Spatial_Error_m": mean_spatial_error,
        "Max_Spatial_Error_m": max_spatial_error,
        "Percent_Errors_Are_Neighbors": pct_neighbor_errors,
        "Avg_Error_Dist_When_Wrong_m": avg_error_distance_when_wrong,
        # Return raw data for plotting
        "raw_index_diff": index_diff,
        "raw_physical_error": physical_error
    }
    
    return results

def plot_error_distribution(output_path, results, tank_spacing_meters=4.25):
    """
    Plots the distribution of MISCLASSIFICATIONS only.
    This validates the claim that errors are spatially local.
    """
    # Get raw index differences
    diffs = results['raw_index_diff']
    
    # Filter to keep only errors (diff > 0)
    # If accuracy is very high (96%), plotting "Correct" (0) would dwarf the errors.
    # The reviewer wants to know about the severity of the *errors*.
    error_diffs = diffs[diffs > 0]
    
    if len(error_diffs) == 0:
        print("No errors to plot!")
        return

    # Count frequency of each error distance (1, 2, 3, 4, 5 tanks away)
    # We count up to max possible distance (N_classes - 1 = 5)
    max_dist_indices = 5 
    counts = np.bincount(error_diffs, minlength=max_dist_indices+1)
    
    # We only care about indices 1 to 5 (since 0 is correct)
    counts = counts[1:] 
    distances_m = np.arange(1, len(counts) + 1) * tank_spacing_meters
    
    # Calculate percentages relative to TOTAL ERRORS
    percentages = (counts / np.sum(counts)) * 100
    
    # Plotting
    plt.figure(figsize=(10, 7.5))
    bars = plt.bar(distances_m, percentages, width=tank_spacing_meters*0.6, 
                   color='#e74c3c', edgecolor='black', alpha=0.8)
    
    # Highlight the "Neighbor" bar (Distance = 1 unit)
    bars[0].set_color('#2ecc71') # Green for "Safe/Neighbor" error
    bars[0].set_label('Neighbor errors (safe)')
    bars[1].set_color('#e74c3c') # Red for "distant"" error
    bars[1].set_label('Distant errors')
    
    plt.xlabel(f'Error distance (meters)\n[Tank spacing = {tank_spacing_meters}m]')
    plt.ylabel('Percentage of misclassifications (%)')
    # plt.title('Spatial distribution of errors')
    plt.xticks(distances_m, [f'{d:.2f}m' for d in distances_m])
    
    # Annotate bars
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                     f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # Add a text box with the summary
    stats_text = (f"Total misclassifications: {len(error_diffs)}\n"
                  f"Neighbor errors (1 tank distance): {percentages[0]:.1f}%")
    plt.text(0.95, 0.95, stats_text, transform=plt.gca().transAxes, 
             fontsize=12, verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'Error_distribution.png'), dpi=500, bbox_inches='tight')
    plt.show()
# ==========================================
# COMMENT 2: TEMPORAL METRICS
# ==========================================
def analyze_temporal_performance(model, X_test, dt=0.32, n_simulations=6, label_idx=5):
    """
    Analyzes detection time and stability given the windowed dataset and model.
    
    Args:
        model: Trained Keras/TensorFlow model.
        X_test: (N_samples, n_steps, n_features) array. 
                - First 4 columns (indices 0-3) are sensor data.
                - Column at `label_idx` contains the True Tank ID (0-5).
        dt: Sampling time in seconds (0.32s).
        n_simulations: Number of distinct leak scenarios in the test set (default 6). 
        label_idx: Index in the last dimension containing the class label (default 5).
        
    Returns:
        metrics_df: DataFrame with TTFD, Stability, etc. for each simulation.
    """
    
    # 1. Prepare Data for Prediction (Sensors are 0,1,2,3)
    X_input = X_test[:, :, :4] 
    
    # 2. Generate Predictions
    print(f"Generating predictions for {len(X_input)} windows...")
    Y_pred_probs = model.predict(X_input, verbose=0)
    
    # 3. Extract Ground Truth
    # We take the label from the LAST time step of each window (index -1)
    # and the specific column `label_idx`.
    # Ensure it's cast to integer for comparison.
    Y_true_ids = X_test[:, -1, label_idx].astype(int)
    
    # 4. Reconstruct Simulations
    # We assume X_test is ordered: [Sim1 (240), Sim2 (240), ... Sim6 (240)]
    samples_per_sim = len(X_test) // n_simulations
    
    if len(X_test) % n_simulations != 0:
        print(f"WARNING: {len(X_test)} samples not divisible by {n_simulations} simulations.")
    
    print(f"Assumed structure: {n_simulations} simulations with {samples_per_sim} windows each.")
    
    results = []
    
    for i in range(n_simulations):
        start_idx = i * samples_per_sim
        end_idx = (i + 1) * samples_per_sim
        
        sim_preds = Y_pred_probs[start_idx:end_idx]
        
        # Get the true class for this simulation chunk.
        # We take the mode (most common value) to be robust against any noise, 
        # though it should be constant.
        chunk_labels = Y_true_ids[start_idx:end_idx]
        sim_true_id = np.bincount(chunk_labels).argmax()
        
        # --- Calculate Metrics for this chunk ---
        metrics = _calculate_single_sim_metrics(sim_preds, sim_true_id, dt)
        
        # Add metadata
        metrics["Simulation_ID"] = i
        metrics["True_Tank"] = f"Tank {sim_true_id + 1}"
        results.append(metrics)

    return pd.DataFrame(results)

def _calculate_single_sim_metrics(sim_preds, target_class, dt):
    """Helper to calculate metrics for one continuous probability sequence."""
    pred_indices = np.argmax(sim_preds, axis=1)
    
    # 1. Time to First Detection (TTFD)
    # Find the first window index where the model predicts the correct class
    correct_indices = np.where(pred_indices == target_class)[0]
    
    if len(correct_indices) > 0:
        first_correct_idx = correct_indices[0]
        
        # Time Calculation:
        # Time = index * dt (assuming 1-step slide). 
        # This is the time *since the start of the analysis window stream*.
        ttfd = first_correct_idx * dt 
        
        # 2. Stability
        # Check predictions AFTER the first detection
        # We start checking from the index immediately following detection
        subsequent_preds = pred_indices[first_correct_idx:]
        
        if len(subsequent_preds) > 0:
            stable_counts = np.sum(subsequent_preds == target_class)
            stability_score = stable_counts / len(subsequent_preds)
        else:
            stability_score = 1.0 # Detected at the very last step
            
        status = "Detected"
    else:
        ttfd = None
        stability_score = 0.0
        status = "Missed"
        
    return {
        "Status": status,
        "TTFD_seconds": ttfd,
        "Stability": stability_score,
        "Duration_Windows": len(pred_indices)
    }

def plot_ttfd_distribution(df_results):
    """Plots the distribution of Time-To-First-Detection."""
    detected_df = df_results[df_results["Status"] == "Detected"]
    
    if len(detected_df) == 0:
        print("No successful detections to plot.")
        return

    plt.figure(figsize=(10, 6))
    sns.histplot(data=detected_df, x="TTFD_seconds", bins=10, kde=True, color="teal")
    
    mean_val = detected_df["TTFD_seconds"].mean()
    median_val = detected_df["TTFD_seconds"].median()
    
    plt.axvline(mean_val, color='r', linestyle='--', label=f'Mean: {mean_val:.2f}s')
    plt.axvline(median_val, color='orange', linestyle='-', label=f'Median: {median_val:.2f}s')
    
    plt.title("Distribution of Time-to-First-Detection (TTFD)")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Count of Simulations")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

def plot_stability_by_tank(df_results):
    """Boxplot of stability scores grouped by Tank."""
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df_results, x="True_Tank", y="Stability", palette="viridis")
    plt.title("Detection Stability by Tank Source")
    plt.ylabel("Stability Score (1.0 = No flickering)")
    plt.ylim(-0.05, 1.05)
    plt.grid(True, axis='y', alpha=0.3)
    plt.show()

# ==========================================
# COMMENT 3: SPLITTING STRATEGY HELPERS
# ==========================================
def check_data_leakage(train_indices, test_indices, simulation_ids):
    """
    Checks if the same simulation ID appears in both Train and Test.
    """
    train_sims = set(simulation_ids[train_indices])
    test_sims = set(simulation_ids[test_indices])
    
    intersection = train_sims.intersection(test_sims)
    
    if len(intersection) > 0:
        print(f"CRITICAL WARNING: Data Leakage detected!")
        print(f"Simulations {list(intersection)[:5]}... appear in both sets.")
        return True
    else:
        print("Data Splitting looks valid. No simulation overlap.")
        return False

# ==========================================
# COMMENT 4: PARETO / KNEE POINT
# ==========================================
def plot_pareto_accuracy_latency(results_df):
    """
    Plots Accuracy vs Latency.
    results_df columns: ['nt', 'accuracy', 'latency_s']
    """
    plt.figure(figsize=(8, 6))
    
    # Plot points
    sns.scatterplot(data=results_df, x='latency_s', y='accuracy', s=100, color='red')
    
    # Connect them
    plt.plot(results_df['latency_s'], results_df['accuracy'], 'r--')
    
    # Annotate
    for i, row in results_df.iterrows():
        plt.text(row['latency_s'], row['accuracy']+0.005, 
                 f"nt={int(row['nt'])}", ha='center')
        
    plt.title("Pareto Front: Accuracy vs. Detection Latency")
    plt.xlabel("Latency (seconds) [Lower is Better]")
    plt.ylabel("Accuracy [Higher is Better]")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()