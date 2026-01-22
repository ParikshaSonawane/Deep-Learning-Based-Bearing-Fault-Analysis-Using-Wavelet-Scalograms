# Deep-Learning-Based-Bearing-Fault-Analysis-Using-Wavelet-Scalograms
The project performs bearing fault analysis by processing raw vibration signals through segmentation, noise augmentation, and wavelet-based denoising, followed by Continuous Wavelet Transform to generate time frequency scalogram images. The normalized scalograms are formatted as CNN compatible inputs, enabling DL based bearing fault classification.

# Bearing Fault Analysis Using Vibration Signal Processing and Wavelet Scalograms

## Overview
This project implements a bearing fault analysis pipeline using vibration signal data stored in `.mat` files.

The code processes raw vibration signals through segmentation, noise augmentation, wavelet-based denoising, and Continuous Wavelet Transform (CWT) to generate scalogram images. These images are then used to train and evaluate a convolutional neural network (CNN) for fault classification.

---

## Architecture Overview
The system follows a **pipeline-based architecture** implemented within a single Jupyter Notebook.

Data flows sequentially through preprocessing, feature generation, dataset preparation, and model training stages. Each stage performs a specific transformation on the vibration signal data.

---

## Core Components

| Component |                                    Responsibility |
|----------|----------------|
| **Data Extraction Module** | Mounts Google Drive, extracts the dataset, and loads `.mat` vibration signal files |
| **Signal Processing Module** | Performs signal segmentation, noise augmentation, and wavelet-based denoising |
| **Feature Generation Module** | Converts denoised signals into time–frequency scalograms using CWT |
| **Image Preparation Module** | Normalizes and resizes scalograms into CNN-compatible RGB images |
| **Modeling Module** | Defines, trains, and evaluates a CNN for fault classification |

---

## Major Modules / Features
- Loading vibration data from MATLAB `.mat` files
- Mapping bearing conditions to numeric class labels
- Sliding-window signal segmentation
- Additive White Gaussian Noise (AWGN) augmentation
- Discrete Wavelet Transform–based denoising
- Continuous Wavelet Transform using a complex Morlet wavelet
- Conversion of scalograms into 224 × 224 RGB images
- Saving and loading processed datasets in `.npz` format
- CNN-based multi-class classification
- Single-sample inference on segmented vibration signals

---

## System Workflow
1. Mount Google Drive and extract the compressed dataset.
2. Load vibration signals from `.mat` files and assign class labels.
3. Visualize raw vibration signals.
4. Segment signals using fixed window and step sizes.
5. Add AWGN noise to each signal segment.
6. Apply wavelet-based denoising to noisy segments.
7. Generate CWT-based scalograms from denoised signals.
8. Normalize, resize, and convert scalograms into RGB images.
9. Store processed images and labels in `.npz` files.
10. Split data into training, validation, and test sets.
11. Train a convolutional neural network on scalogram images.
12. Perform prediction on user-selected signal segments.

---

## Core Logic / Methodology
The core logic focuses on transforming non-stationary vibration signals into time–frequency representations using wavelet analysis.

- **Discrete Wavelet Transform (DWT)** is used for signal denoising.
- **Continuous Wavelet Transform (CWT)** with a complex Morlet wavelet is used to generate scalograms.
- Generated scalograms are treated as image inputs for CNN-based classification.

---

## Build & Setup Information
- **Execution Environment:** Google Colab (inferred from `google.colab` usage)

**Libraries Used:**
- numpy
- scipy
- matplotlib
- pywt (PyWavelets)
- opencv-python
- tqdm
- scikit-learn
- tensorflow / keras
- imbalanced-learn

Dataset paths and execution settings are hardcoded for Google Drive usage. No standalone setup script or environment configuration file is provided.

---
## Dataset Access
The vibration signal dataset used in this project consists of MATLAB `.mat` files and exceeds GitHub’s file size limits.

Due to this limitation, the dataset is hosted externally.

🔗 **Google Drive Link:** <https://drive.google.com/file/d/1kj02eBHbzFA3yuJUVTQwYZBqNPNgauaU/view?usp=sharing>  
**Access:** Public (View-only)

---

## Limitations & Assumptions
- File paths and dataset locations are hardcoded.
- Window size, step size, SNR value, and wavelet parameters are fixed in the code.
- No command-line interface or configuration abstraction is implemented.
- Dataset balancing is partially addressed using random undersampling.
- Data processing, training, and inference logic are combined in a single notebook.

---

## Future Scope
- Modularizing the pipeline into separate scripts or packages
- Externalizing hyperparameters and configuration settings
- Persisting model evaluation metrics
- Supporting real-time or streaming vibration data

---

## Contributing
Contributions are welcome.

Please fork the repository, create a feature branch, and submit a pull request with a clear description of the changes. Maintain consistency with the existing pipeline structure.

---


