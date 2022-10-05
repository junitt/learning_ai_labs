import numpy as np
from .modules import Module
import math

class Sigmoid(Module):

    def forward(self, x):

        # TODO Implement forward propogation
        # of sigmoid function.
        self.x=x
        ret=np.zeros(x.shape)
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                ret[i][j]=math.exp(x[i][j])/(math.exp(x[i][j])+1)
        self.f=ret
        return ret
        ...

        # End of todo

    def backward(self, dy):

        # TODO Implement backward propogation
        # of sigmoid function.
        return dy*self.f*(1-self.f)
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
        self.x=x
        return np.maximum(x, 0)
        ...

        # End of todo

    def backward(self, dy):

        # TODO Implement backward propogation
        # of ReLU function.
        return np.where(self.x>0,dy,0)
        ...

        # End of todo


class Softmax(Module):

    def forward(self, x):

        # TODO Implement forward propogation
        # of Softmax function.
        self.x=x
        #dn=np.ones(x.shape)
        val=np.sum(np.exp(x),axis=1,keepdims=True)
        return np.exp(x)/val
        ...

        # End of todo

    def backward(self, dy):
        return dy
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
        
        self.probs = probs
        self.targets = targets
        temp=np.exp(probs)
        probs=temp/np.sum(temp,axis=1,keepdims=True)
        fac=np.zeros(self.n_classes)
        fac[targets]=1
        self.value=np.sum(-fac*np.log(probs))
        return self
        # End of todo

    def backward(self):

        # TODO Implement backward propogation
        # of softmax loss function.
        
        fac=np.zeros(self.n_classes)
        fac[self.targets]=1
        return self.probs-fac
        ...

        # End of todo


class CrossEntropyLoss(Loss):

    def __call__(self, probs, targets):

        # TODO Calculate cross-entropy loss.
        #val=np.sum(np.exp(probs),axis=1,keepdims=True)
        #np.exp(probs)/val
        self.probs=probs
        self.targets=targets
        self.value=np.mean(-np.eye(self.n_classes)[targets]*np.log(probs))
        return self

        ...

        # End of todo

    def backward(self):

        # TODO Implement backward propogation
        # of cross-entropy loss function.
        batch_size = self.probs.shape[0]
        class_num = self.n_classes
        target = np.eye(class_num)[self.targets]
        return (-(target - self.probs) / batch_size)
        ...

        # End of todo
