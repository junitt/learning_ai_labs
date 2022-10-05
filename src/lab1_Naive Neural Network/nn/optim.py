from .tensor import Tensor
from .modules import Module
import math

class Optim(object):

    def __init__(self, module, lr):
        self.module = module
        self.lr = lr

    def step(self):
        self._step_module(self.module)

    def _step_module(self, module):

        # TODO Traverse the attributes of `self.module`,
        # if is `Tensor`, call `self._update_weight()`,
        # else if is `Module` or `List` of `Module`,
        # call `self._step_module()` recursively.
        for inst in vars(module).values():
            if isinstance(inst,Tensor):
                if hasattr(inst,'grad'):
                    self._update_weight(inst)
            if isinstance(inst,Module):
                self._step_module(inst)
            if isinstance(inst,list):
                for u in inst:
                    self._step_module(u)
        ...

        # End of todo

    def _update_weight(self, tensor):
        tensor -= self.lr * tensor.grad


class SGD(Optim):

    def __init__(self, module, lr, momentum: float=0):
        super(SGD, self).__init__(module, lr)
        self.momentum = momentum

    def _update_weight(self, tensor):

        # TODO Update the weight of tensor
        # in SGD manner.
        tensor.v=self.momentum*tensor.v+self.lr*tensor.grad if 'v' in vars(tensor) else self.lr * tensor.grad
        tensor-=tensor.v

        ...

        # End of todo


class Adam(Optim):

    def __init__(self, module, lr):
        super(Adam, self).__init__(module, lr)

        # TODO Initialize the attributes
        # of Adam optimizer.
        self.betas=(0.9, 0.999)
        self.eps=1e-8,
        self.weight_decay=0,
        ...

        # End of todo

    def _update_weight(self, tensor):

        # TODO Update the weight of
        # tensor in Adam manner.
        
        ...

        # End of todo
