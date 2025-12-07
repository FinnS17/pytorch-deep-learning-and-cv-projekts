import sys
from pathlib import Path

import numpy as np
from mlp import MLP

# Allow importing shared helpers from the repository root
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shared import load_mnist_numpy  

def accuracy(model, X, y):
    logits = model.forward(X)
    preds = np.argmax(logits, axis=1)
    return np.mean(preds == y)

def calculate_n_params(model):
    count = 0
    for layer in model.layers:
        for p in layer.parameters():
            count += p.size
    return count


# data loading and training
X_train, y_train, X_test, y_test = load_mnist_numpy()


model = MLP(input_dim=784, hidden_dim=128, num_classes=10)
n_parameters = calculate_n_params(model)
print(f"Number of Parameters of the model: {n_parameters}")

lr = 1e-1
batch_size = 64
epochs = 25

N = X_train.shape[0]  # length of dataset

for epoch in range(epochs):
    # shuffle once per epoch and iterate in batches
    perm = np.random.permutation(N)
    for start in range(0, N, batch_size):
        idx = perm[start:start+batch_size]  # batch indices from permutation
        Xb = X_train[idx]
        yb = y_train[idx]

        logits, loss = model.forward(Xb, yb)
        model.backward()
        lr = 0.1 if epoch / epochs < 0.5  else 0.01
        model.update(lr)

    test_acc = accuracy(model, X_test, y_test)
    train_acc = accuracy(model, X_train, y_train)
    print(f"epoch: {epoch+1}/{epochs}; Loss: {loss:.4f}; Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}")
        

