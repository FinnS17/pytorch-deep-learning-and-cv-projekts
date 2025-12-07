import numpy as np

class Linear:
    
    def __init__(self, input_dim, output_dim):
        self.W = np.random.randn(input_dim, output_dim) * 0.01
        self.b = np.zeros((1, output_dim))
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
        self.X = None  # cache input for backward
    
    def forward(self, X):  # (batch x input) @ (input x output) + (1 x output)
        self.X = X
        out = self.X @ self.W + self.b
        return out
    
    def backward(self, dZ):  # incoming gradient dL/dZ
        self.dW = self.X.T @ dZ  # (input_dim x batch) @ (batch x output_dim)
        self.db = dZ.sum(axis=0, keepdims=True)  # sum over batch
        dX = dZ @ self.W.T  # gradient w.r.t. input to pass left
        return dX

    def parameters(self):
        return [self.W, self.b]
    
    def grads(self):
        return [self.dW, self.db]


class ReLU:
    
    def __init__(self):
        self.mask = None  # saves where x > 0
    
    def forward(self, X):
        self.mask = X > 0
        return X * self.mask  # passes positive values, else 0
    
    def backward(self, dA):
        return dA * self.mask  # pass gradient where input was positive
    
    def parameters(self):
        return []

    def grads(self):
        return []

class Sigmoid:
    
    def __init__(self):
        self.s = None

    def forward(self, X):
        self.s = 1 / (1 + np.exp(-X))
        return self.s
    
    def backward(self, dA):
        return dA * self.s * (1 - self.s)  # incoming grad times local derivative
    
    def parameters(self):
        return []
    
    def grads(self):
        return []


        
