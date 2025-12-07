import numpy as np
import loss, layers

class MLP:

    def __init__(self, input_dim, hidden_dim, num_classes):
        self.layers = [
            layers.Linear(input_dim, hidden_dim),
            layers.ReLU(),
            layers.Linear(hidden_dim, hidden_dim),
            layers.ReLU(),
            layers.Linear(hidden_dim, num_classes)
        ]
        self.loss_function = loss.SoftmaxCrossEntropy()

    def forward(self, X, y=None):
        out = X
        for layer in self.layers:
            out = layer.forward(out)

        if y is None:  # logits for prediction
            return out
        
        loss = self.loss_function.forward(out, y)  # calculate loss
        return out, loss
    
    def backward(self):
        dZ = self.loss_function.backward()
        for layer in reversed(self.layers):  # traverse layers from end to start
            dZ = layer.backward(dZ)  # propagate loss through the whole MLP
 

    def update(self, lr):
        for layer in self.layers:
            params = layer.parameters()
            grads = layer.grads()
            for p, g in zip(params, grads):
                p -= lr * g
