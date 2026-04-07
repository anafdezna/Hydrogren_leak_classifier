# Deep Learning Framework for Hydrogen Leak Detection and Localization

This repository contains the implementation and experimental framework for a 1D Convolutional Neural Network (CNN) designed to detect and localize hydrogen leaks in industrial storage facilities. The system processes sensor time-series data to identify which tank in a multi-tank array is experiencing a leak.

## 1\. Project Overview

This project addresses a critical safety challenge in the transition to hydrogen energy: the rapid detection and spatial localization of unintended hydrogen releases in large-scale storage facilities. Conventional systems often rely on point-based sensors that lack spatial awareness or require dense, expensive networks.

This work implements a **1D Convolutional Neural Network (CNN)** framework designed to interpret spatiotemporal patterns from a sparse array of only four sensors. The system identifies which specific tank is leaking within a complex simulation environment, providing a scalable and intelligent safety monitoring solution.

## 2\. Repository Structure

├── main_server.py # Orchestration: Preprocessing, training, and evaluation  
├── architectures.py # 1D-CNN (Arch_3), LSTM, and hybrid model definitions  
├── model_creation.py # Model instantiation logic  
├── training.py # Training loops, weight management, and custom loss logic  
├── Callbacks.py # Advanced callbacks: Loss-Dependent Adaptive LR & Early Stopping  
├── Custom_losses.py # Specialized metrics (Spatial Error, MSE, Severity-based)  
├── Classification_results.py # Visualization: Barplots, Heatmaps, and Spatial Analysis  
├── postprocessing_tools.py # Plotting: Loss evolution, histograms, and crossplots  
├── MODULES/ # Sub-modules for signal preprocessing and transformations  
└── Output/ # Result directory (Models, .npy logs, and PNG figures)



## 3\. Experimental Case Study: Hydrogen Plant Simulation

The methodology is developed using data from a high-fidelity 3D computational fluid dynamics (CFD) simulation of a representative hydrogen storage facility.

### 3.1 Facility Layout

- **Dimensions:** \$100\\text{ m} \\times 48\\text{ m} \\times 13\\text{ m}\$ (Length \$\\times\$ Width \$\\times\$ Height).
- **Storage:** Twelve horizontal storage tanks (\$12\\text{ m}\$ long, \$2.5\\text{ m}\$ wide).
- **Grid Spacing:** \$3.25\\text{ m}\$ along the x-axis and \$4.25\\text{ m}\$ along the y-axis.
- **Symmetry:** For computational efficiency, simulations were focused on the left-hand column of six tanks (\$T_1\$ to \$T_6\$).

### 3.2 Leak Scenarios

- **Release Rate:** \$0.022\\text{ kg/s}\$ (representing a hazardous but credible release).
- **Leak Points:** Six potential leak points per tank (three on each side wall, \$1\\text{ m}\$ above ground).
- **Duration:** Each simulation spans \$\\approx 300\\text{ seconds}\$ (\$1,000\$ time steps) to capture the transient release and subsequent dispersion.

### 3.3 Instrumentation

- **Sensor Array:** 4 virtual sensors (\$n_s = 4\$).
- **Placement:** Positioned at the upper boundary (\$13\\text{ m}\$ height) where hydrogen, due to its buoyancy, naturally accumulates.
- **Data Acquisition:** Sampling frequency of \$\\approx 3.11\\text{ Hz}\$ (\$\\approx 0.32\\text{ s}\$ per time step).

## 4\. Data Engineering and Preprocessing

The raw CFD data is processed to create a robust dataset suitable for training a Deep Learning classifier.

### 4.1 Sliding Window Strategy

To enable continuous monitoring, we employ a sliding window approach:

- **Window Length (**\$n_t\$**):** 50 time steps (\$\\approx 16\\text{ seconds}\$).
- **Slide Width (**\$n\_{sl}\$**):** 1 time step (\$0.32\\text{ s}\$).
- **Input Dimension:** \$X_i \\in \\mathbb{R}^{n_t \\times n_s}\$ (a \$50 \\times 4\$ tensor per sample).

### 4.2 Label Encoding

The leak origin is treated as a multi-class classification problem (\$C = 6\$).

- **One-Hot Encoding:** A probability vector \$y_i^{true} \\in \[0, 1\]^C\$ is assigned, where the index of the leaking tank is marked as 1.

### 4.3 Dataset Optimization

Analysis was performed to balance accuracy and detection delay.

- **Simulation Length (**\$L\_{sim}\$**):** We selected data from the first \$100\\text{ seconds}\$ of each simulation. This focuses the model on the transient phase of the leak before the space becomes fully saturated, which provides the most discriminative spatial information.


## 5\. The 1D-CNN Architecture (Arch_3)

The framework utilizes a structured 1D-CNN designed to extract features from multi-channel sensor signals:

| **Layer**       | **Type**    | **Output Shape** | **Parameters**             |
| --------------- | ----------- | ---------------- | -------------------------- |
| **Input**       | Time-series | \$(50, 4)\$      | \-                         |
| ---             | ---         | ---              | ---                        |
| **Conv1D (x3)** | Conv + BN   | \$(50, 10)\$     | 10 filters, Kernel 3, ReLU |
| ---             | ---         | ---              | ---                        |
| **Pooling**     | Global Max  | \$(10,)\$        | Flattening operator        |
| ---             | ---         | ---              | ---                        |
| **Dense (x3)**  | Fully Conn. | \$(10,)\$        | ReLU activation            |
| ---             | ---         | ---              | ---                        |
| **Output**      | Softmax     | \$(6,)\$         | Probability per tank       |
| ---             | ---         | ---              | ---                        |

## 6\. Evaluation & Spatial Metrics

Performance is evaluated using both standard classification metrics and safety-critical spatial analysis:

### 6.1 Standard Metrics

- **Accuracy:** ~88% on unseen test locations.
- **F1-Score:** ~0.87 (Weighted average).

### 6.2 Spatial Safety Analysis

In hydrogen safety, a misclassification to an adjacent tank is significantly safer than a distant one.

- **Spatial Error Analysis:** Calculates the physical distance (meters) between the predicted tank and the true source.
- **Neighbor Error Rate:** Quantifies "safe" misclassifications (attributing the leak to a tank immediately adjacent, i.e., within \$4.25\\text{ m}\$).
- **Time to First Detection (TTFD):** Analyzes how many seconds the model takes to correctly identify the source after the leak starts.

## 7 Usage & Configuration

### Prerequisites

pip install tensorflow numpy pandas matplotlib seaborn scikit-learn scipy

### Execution Flow

- **Model Selection:** In main_server.py, verify Architecture = "Arch_3".
- **Parameters:** Ensure n_s = 50 (segment length) and LRate = 1e-05.
- **Training:** Run the main script to start the pipeline:  
   python main_server.py

- **Results:** The script generates model weights (.h5), training logs (.npy), and visualization plots in the Output/ folder.

## 🔍 Troubleshooting & FAQ

**Q: Why does the model confuse adjacent tanks?**

**A:** With only 4 sensors, gas plumes from neighboring tanks often create near-identical concentration signatures (the "ill-posed" problem). This is a known limitation of sparse instrumentation.

**Q: What is the purpose of the Adaptive Learning Rate?**

**A:** Found in Callbacks.py, this custom callback monitors loss and can **reject weights** if an epoch degrades performance, preventing gradient instability during training.

**Authors:** A. Garmendia-Orbegozo, A. Fernandez-Navamuel, T. Teijeiro, M. Minguez, M.A. Anton.

**Affiliation:** TECNALIA, Basque Research and Technology Alliance (BRTA).

**Project:** SEGURH2 (KK-2024/00068).
