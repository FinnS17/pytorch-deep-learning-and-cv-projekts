# PyTorch CNN for MNIST Classification (MPS Optimized)

This repository contains the complete implementation of a simple Convolutional Neural Network (CNN) built from scratch in PyTorch to classify handwritten digits from the MNIST dataset.

The project demonstrates core skills in Deep Learning architecture, training loop implementation, and utilization of the Apple Silicon (MPS) backend.

## 📈 Final Results

| Metric | Value | Observation |
| :--- | :--- | :--- |
| **Final Test Accuracy** | **98.4%** | Excellent generalization on unseen test data. |
| **Final Test Loss** | 0.0498 | Loss is minimal, indicating strong convergence. |
| **Hardware Used** | Apple M1 Pro (MPS) | Training utilizes GPU acceleration. |
| **Recommended Epochs** | 6-8 | The highest gains were made within the first 6 epochs. |

## 📐 Model Architecture (Custom CNN)

The custom network (`Cnn` class) uses a standard sequence of layers for feature extraction:

1.  **Conv1**: 1 Input Channel $\rightarrow$ 10 Feature Maps
2.  **Pool1**: MaxPool (2x2)
3.  **Conv2**: 10 Input Channels $\rightarrow$ 20 Feature Maps
4.  **Pool2**: MaxPool (2x2)
5.  **Flattening**: 3D Volume (4x4x20) $\rightarrow$ 1D Vector (320)
6.  **FC1 & FC2**: Fully Connected layers for final classification (320 $\rightarrow$ 50 $\rightarrow$ 10).

## 💻 Setup and Execution

### Prerequisites

The environment is managed via `conda` and `pip`. Create a new environment using Python 3.10:

```bash
conda create -n cnn_mnist python=3.10
conda activate cnn_mnist
```

Install the dependencies using the provided requirements.txt:
```bash 
pip install -r requirements.txt
```
### Execution
The training is executed via the `cnn_notebook.ipynb` file. The training loop calls functions from the `src/train_utils.py` module.