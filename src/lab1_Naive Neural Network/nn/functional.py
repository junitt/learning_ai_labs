import numpy as np
from .modules import Module
import math

class Sigmoid(Module):

    def forward(self, x):

        # TODO Implement forward propogation
        # of sigmoid function.
        self.x=x
        ret=np.zeros(x.shape)
        self.x=x
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                ret[i][j]=math.exp(x[i][j])/(math.exp(x[i][j])+1)
        return ret
        ...

        # End of todo

    def backward(self, dy):

        # TODO Implement backward propogation
        # of sigmoid function.
        x=self.x
        ret=np.zeros(x.shape)
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                ret[i][j]=dy[i][j]*math.exp(x[i][j])/(math.exp(x[i][j])+1)*(1-math.exp(x[i][j])/(math.exp(x[i][j])+1))
        return ret
        ...

        # End of todo


class Tanh(Module):

    def forward(self, x):

        # TODO Implement forward propogation
        # of tanh function.
        self.x=x
        return np.tanh(x)
        ...

        # End of todo

    def backward(self, dy):

        # TODO Implement backward propogation
        # of tanh function.
        return dy*(1-np.tanh(self.x)**2)
        ...

        # End of todo


class ReLU(Module):

    def forward(self, x):

        # TODO Implement forward propogation
        # of ReLU function.
        ret=np.zeros(x.shape)
        self.x=x
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                if(x[i][j]>0):
                    ret[i][j]=x[i][j]
        return ret
        ...

        # End of todo

    def backward(self, dy):

        # TODO Implement backward propogation
        # of ReLU function.
        x=self.x
        ret=np.zeros(x.shape)
        
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                if(x[i][j]>0):
                    ret[i][j]=dy[i][j]
        return ret
        ...

        # End of todo


class Softmax(Module):

    def forward(self, x):

        # TODO Implement forward propogation
        # of Softmax function.
        self.x=x
        dn=np.ones(x.shape)
        val=np.sum(np.exp(x),axis=1)
        for i in range(x.shape[0]):
            dn[i]=dn[i]*val[i]
        return np.exp(x)/dn
        ...

        # End of todo

    def backward(self, dy):

        # Omitted.
        ...


class Loss(object):
    """
    Usage:
        >>> criterion = Loss(n_classes)
        >>> ...
        >>> for epoch in n_epochs:
        ...     ...
        ...     probs = model(x)
        ...     loss = criterion(probs, target)
        ...     model.backward(loss.backward())
        ...     ...
    """
    def __init__(self, n_classes):
        self.n_classes = n_classes

    def __call__(self, probs, targets):
        self.probs = probs
        self.targets = targets
        ...
        return self

    def backward(self):
        ...


class SoftmaxLoss(Loss):

    def __call__(self, probs, targets):

        # TODO Calculate softmax loss.
        
        ...

        # End of todo

    def backward(self):

        # TODO Implement backward propogation
        # of softmax loss function.

        ...

        # End of todo


class CrossEntropyLoss(Loss):

    def __call__(self, probs, targets):

        # TODO Calculate cross-entropy loss.

        ...

        # End of todo

    def backward(self):

        # TODO Implement backward propogation
        # of cross-entropy loss function.

        ...

        # End of todo
