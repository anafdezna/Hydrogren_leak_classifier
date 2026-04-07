# Deep Learning Framework for Hydrogen Leak Detection and Localization

This repository contains the implementation and experimental framework for a 1D Convolutional Neural Network (CNN) designed to detect and localize hydrogen leaks in industrial storage facilities. The system processes sensor time-series data to identify which tank in a multi-tank array is experiencing a leak.

## 🚀 Project Overview

Industrial hydrogen storage requires rapid, reliable leak localization. Conventional point-based sensors lack spatial awareness, and manual monitoring is insufficient for large-scale facilities.

- **Objective:** Real-time spatial identification of hydrogen leaks using a sparse sensor array.
- **Methodology:** A supervised deep learning approach using 1D-CNNs trained on high-fidelity simulation data to interpret spatiotemporal concentration patterns.

## 📂 Repository Structure

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

## 🏗️ Case Study & Dataset

The model is trained on data derived from high-fidelity 3D Computational Fluid Dynamics (CFD) simulations representing a representative industrial layout.

### Facility Layout

- **Domain:** \$100m \\times 48m \\times 13m\$ open boundary storage area.
- **Tanks:** 12 horizontal storage tanks arranged in a grid.
- **Sensors (**\$n_s=4\$**):** Sparse array strategically placed to capture dispersion fields.
- **Leak Scenarios:** Constant release rate of \$0.022 kg/s\$ simulated at multiple longitudinal points per tank.

### Data Optimization (\$L\_{sim}\$ and \$n_t\$)

Research indicates that "Transient Phase" data provides the most discriminative patterns for localization.

- **Simulation Length (**\$L\_{sim}\$**):** Capped at 100s to focus on the initial leak evolution and avoid ambiguous steady-state samples.
- **Segment Length (**\$n_t\$**):** 50 data points (~16s window) selected as the optimal balance between accuracy and detection delay.
- **Sampling:** \$f\_{ac} = 3.11\$ Hz (approx. 0.32s per sample).

## 📈 The 1D-CNN Architecture (Arch_3)

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

## 📊 Evaluation & Spatial Metrics

Performance is evaluated using both standard classification metrics and safety-critical spatial analysis:

### Standard Metrics

- **Accuracy:** ~88% on unseen test locations.
- **F1-Score:** ~0.87 (Weighted average).

### Spatial Safety Analysis

In hydrogen safety, a misclassification to an adjacent tank is significantly safer than a distant one.

- **Mean Spatial Error:** ~2.50 meters.
- **Neighbor Errors:** ~80% of errors are "Safe Errors" (predicted tank is within 4.25m of the true source).
- **Spatial Coherence:** Validates that even when the model fails the exact match, it localizes the hazard to the immediate vicinity.

## 🛠️ Usage & Configuration

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
