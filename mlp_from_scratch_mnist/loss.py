import numpy as np

class SoftmaxCrossEntropy:

    def __init__(self):
        self.probs = None
        self.y_true = None

    def forward(self, logits, y_true):
        # logits -> (batch x num_classes)
        # y_true -> (batch,) integer class labels
        shifted = logits - np.max(logits, axis=1, keepdims=True)  # stabilize exponent
        # softmax: exp(x) / sum(exp(x))
        exp = np.exp(shifted)
        self.probs = exp / np.sum(exp, axis=1, keepdims=True)  # batch x num_classes
        self.y_true = y_true

        # compute loss
        batch_size = logits.shape[0]
        correct = self.probs[np.arange(batch_size), y_true] + 1e-12  # -> (batch,)
        logprobs = -np.log(correct)
        loss = logprobs.mean()
        return loss
    
    def backward(self):
        # dL/dlogits = probs - one_hot(y); divide by batch for mean loss
        batch_size = self.probs.shape[0]  # probs.shape: (batch x num_classes)
        dZ = self.probs.copy()
        dZ[np.arange(batch_size), self.y_true] -= 1  # probs - 1 on true class
        dZ /= batch_size
        return dZ
