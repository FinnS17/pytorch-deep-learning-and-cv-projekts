# MNIST Experiments

Two projects share one repo:
- `mlp_from_scratch_mnist`: NumPy MLP with hand-written forward/backward. Layers (Linear, ReLU) and Softmax cross-entropy are implemented from scratch; weights get updated by plain SGD.
- `pytorch_cnn_mnist`: Small CNN in PyTorch with a couple of conv blocks and a linear head.
- `shared`: Helpers like the common MNIST loader used by both.

## Setup
- Python 3.10+
- From the repo root: `pip install -r requirements.txt`
- MNIST is downloaded once into `data/`.

## MLP (NumPy)
```bash
python mlp_from_scratch_mnist/train.py
```
Trains a 3-layer MLP on flattened 28x28 images, prints train/test accuracy each epoch, and uses a simple learning-rate drop halfway through.

## CNN (PyTorch)
```bash
python pytorch_cnn_mnist/train.py
```
Trains the CNN, evaluates after each epoch, and saves weights to `pytorch_cnn_mnist/cnn_mnist.pt`.

## Notes
- Both projects call into `shared/data.py` for MNIST.
- Everything runs as scripts; notebooks are optional.
