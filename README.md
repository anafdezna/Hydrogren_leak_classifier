Deep Learning Framework for Hydrogen Leak Detection and Localization

This repository contains the implementation and experimental framework for a 1D Convolutional Neural Network (CNN) designed to detect and localize hydrogen leaks in industrial storage facilities. The system processes sensor time-series data to identify which tank in a multi-tank array is experiencing a leak.

🚀 Project Overview

Industrial hydrogen storage requires rapid, reliable leak localization. Conventional point-based sensors lack spatial awareness, and manual monitoring is insufficient for large-scale facilities.

Objective: Real-time spatial identification of hydrogen leaks using a sparse sensor array.

Methodology: A supervised deep learning approach using 1D-CNNs trained on high-fidelity simulation data to interpret spatiotemporal concentration patterns.

📂 Repository Structure

main_server.py: Orchestration script for preprocessing, training, and evaluation.

architectures.py: Model definitions including the primary 1D-CNN (Arch_3), LSTM, and hybrid architectures.

model_creation.py: Logic for model instantiation and layer initialization.

training.py: Implementation of training loops, weight management, and custom loss logic.

Callbacks.py: Advanced callbacks including Loss-Dependent Adaptive Learning Rate and Early Stopping.

Custom_losses.py: Specialized metrics including Spatial Error, MSE, and Severity-based loss.

Classification_results.py: Visualization tools for barplots, heatmaps, and standard classification metrics.

Reviewer_comments_functions.py: Safety-critical analysis tools including Spatial Error distribution and TTFD (Time-to-First-Detection) analysis.

preprocessing_tools.py: Utilities for signal transformations, normalization, and windowing.

LOAD_DATA.py: Data ingestion modules for .mat, .csv, and .xlsx simulation files.

Output/: Directory for generated model weights (.h5), training logs (.npy), and visualization figures.

🏗️ Case Study & Dataset

The model is trained on data derived from high-fidelity 3D Computational Fluid Dynamics (CFD) simulations of a representative industrial layout.

Facility Layout

Domain: $100m \times 48m \times 13m$ open boundary storage area.

Tanks: 12 horizontal storage tanks arranged in a regular grid.

Sensors ($n_s=4$): A sparse array strategically placed to capture dispersion fields despite limited instrumentation.

Leak Scenarios: Constant release rate of $0.022 kg/s$ simulated at multiple longitudinal points per tank.

Data Optimization ($L_{sim}$ and $n_t$)

Research indicates that the "Transient Phase" data provides the most discriminative patterns for localization.

Simulation Length ($L_{sim}$): Capped at $100s$ to focus on the initial leak evolution and avoid ambiguous steady-state samples.

Segment Length ($n_t$): $50$ data points ($\approx 16s$ window) selected as the optimal balance between accuracy and detection delay.

Sampling: $f_{ac} = 3.11$ Hz (approximately $0.32s$ per sample).

📈 The 1D-CNN Architecture (Arch_3)

The framework utilizes a structured 1D-CNN designed to extract features from multi-channel sensor signals:

Layer

Type

Output Shape

Parameters

Input

Time-series

$(50, 4)$

-

Conv1D (x3)

Conv + BN

$(50, 10)$

10 filters, Kernel 3, ReLU

Pooling

Global Max

$(10,)$

Flattening operator

Dense (x3)

Fully Conn.

$(10,)$

ReLU activation

Output

Softmax

$(6,)$

Probability per tank

📊 Evaluation & Spatial Metrics

Performance is evaluated using both standard classification metrics and safety-critical spatial analysis.

Standard Metrics

Accuracy: $\approx 88\%$ on unseen test locations.

F1-Score: $\approx 0.87$ (Weighted average).

Spatial Safety Analysis

In hydrogen safety, a misclassification to an adjacent tank is significantly safer than a distant one.

Mean Spatial Error: $\approx 2.50$ meters.

Neighbor Errors: $\approx 80\%$ of errors are "Safe Errors" (predicted tank is within $4.25m$ of the true source).

Spatial Coherence: Validates that even when the model fails the exact match, it localizes the hazard to the immediate vicinity.

🛠️ Usage & Configuration

Prerequisites

pip install tensorflow numpy pandas matplotlib seaborn scikit-learn scipy


Execution Flow

Model Selection: In main_server.py, verify Architecture = "Arch_3".

Parameters: Ensure n_s = 50 (segment length) and LRate = 1e-05.

Training: Run the main script to start the pipeline:

python main_server.py


Results: The script generates model weights (.h5), training logs (.npy), and visualization plots in the Output/ folder.

🔍 Troubleshooting & FAQ

Q: Why does the model confuse adjacent tanks?
A: With only 4 sensors, gas plumes from neighboring tanks often create near-identical concentration signatures (the "ill-posed" problem). This is a known limitation of sparse instrumentation addressed in the spatial error analysis.

Q: What is the purpose of the Adaptive Learning Rate?
A: Found in Callbacks.py, this custom callback monitors loss and can reject weights if an epoch degrades performance, preventing gradient instability during training.

Authors: A. Garmendia-Orbegozo, A. Fernandez-Navamuel, T. Teijeiro, M. Minguez, M.A. Anton.
Affiliation: TECNALIA, Basque Research and Technology Alliance (BRTA).
Project: SEGURH2 (KK-2024/00068).
