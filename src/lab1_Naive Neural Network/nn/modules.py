import numpy as np
from itertools import product
from . import tensor


class Module(object):
    """Base class for all neural network modules.
    """
    def __init__(self) -> None:
        """If a module behaves different between training and testing,
        its init method should inherit from this one."""
        self.training = True

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Defines calling forward method at every call.
        Should not be overridden by subclasses.
        """
        return self.forward(x)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Defines the forward propagation of the module performed at every call.
        Should be overridden by all subclasses.
        """
        ...

    def backward(self, dy: np.ndarray) -> np.ndarray:
        """Defines the backward propagation of the module.
        """
        return dy

    def train(self):
        """Sets the mode of the module to training.
        Should not be overridden by subclasses.
        """
        if 'training' in vars(self):
            self.training = True
        for attr in vars(self).values():
            if isinstance(attr, Module):
                Module.train()

    def eval(self):
        """Sets the mode of the module to eval.
        Should not be overridden by subclasses.
        """
        if 'training' in vars(self):
            self.training = False
        for attr in vars(self).values():
            if isinstance(attr, Module):
                Module.eval()


class Linear(Module):

    def __init__(self, in_length: int, out_length: int):
        """Module which applies linear transformation to input.

        Args:
            in_length: L_in from expected input shape (N, L_in).
            out_length: L_out from output shape (N, L_out).
        """

        # w[0] for bias and w[1:] for weight
        self.w = tensor.tensor((in_length + 1, out_length))

    def forward(self, x):
        """Forward propagation of linear module.

        Args:
            x: input of shape (N, L_in).
        Returns:
            out: output of shape (N, L_out).
        """

        # TODO Implement forward propogation
        # of linear module.
        self.x = x
        return np.dot(x, self.w[1:]) + self.w[0]

        ...

        # End of todo

    def backward(self, dy):
        """Backward propagation of linear module.

        Args:
            dy: output delta of shape (N, L_out).
        Returns:
            dx: input delta of shape (N, L_in).
        """

        # TODO Implement backward propogation
        # of linear module.
        self.w.grad=np.vstack((np.sum(dy,axis=0),np.dot(self.x.T,dy)))
        return np.dot(dy, self.w[1:].T)

        ...

        # End of todo


class BatchNorm1d(Module):

    def __init__(self, length: int, momentum: float=0.9):
        """Module which applies batch normalization to input.
        Args:
            length: L from expected input shape (N, L).
            momentum: default 0.9.
        """
        super(BatchNorm1d, self).__init__()

        # TODO Initialize the attributes
        # of 1d batchnorm module.
        self.last_mean=np.zeros((length,))
        self.last_var=np.zeros((length,))
        self.momentum=momentum
        self.eps=1e-5
        self.gamma=tensor.ones((length,))
        self.beta=tensor.zeros((length,))

        # End of todo

    def forward(self, x):
        """Forward propagation of batch norm module.
        Args:
            x: input of shape (N, L).
        Returns:
            out: output of shape (N, L).
        """

        # TODO Implement forward propogation
        # of 1d batchnorm module.

        if self.training:
            self.mean=np.mean(x,axis=0)
            self.var=np.var(x,axis=0)
            self.last_mean=self.momentum*self.last_mean+(1-self.momentum)*self.mean
            self.last_var=self.momentum*self.last_var+(1-self.momentum)*self.var
            self.x=(x-self.mean)/np.sqrt(self.var+self.eps)
        else:
            self.x=(x-self.last_mean)/np.sqrt(self.last_var+self.eps)
        return self.beta+self.x*self.gamma

        # End of todo

    def backward(self, dy):
        """Backward propagation of batch norm module.

        Args:
            dy: output delta of shape (N, L).
        Returns:
            dx: input delta of shape (N, L).
        """

        # TODO Implement backward propogation
        # of 1d batchnorm module.
        self.gamma.grad=np.sum(self.x*dy,axis=0)
        self.beta.grad=np.sum(dy,axis=0)
        N=dy.shape[0]
        dy*=self.gamma
        dx=N*dy-np.sum(dy,axis=0)-self.x*np.sum(dy*self.x,axis=0)
        return dx/N/np.sqrt(self.var+self.eps)

        # End of todo


class Conv2d(Module):

    def __init__(self, in_channels: int, channels: int, kernel_size: int=3,
                 stride: int=1, padding: int=0, bias: bool=True):
        """Module which applies 2D convolution to input.

        Args:
            in_channels: C_in from expected input shape (B, C_in, H_in, W_in).
            channels: C_out from output shape (B, C_out, H_out, W_out).
            kernel_size: default 3.
            stride: default 1.
            padding: default 0.
        """

        # TODO Initialize the attributes
        # of 2d convolution module.
        self.channels = channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.kernel=tensor.random((channels, in_channels, kernel_size, kernel_size))
        self.bias=tensor.zeros(channels) if bias else None
        ...

        # End of todo

    def forward(self, x):
        """Forward propagation of convolution module.

        Args:
            x: input of shape (B, C_in, H_in, W_in).
        Returns:
            out: output of shape (B, C_out, H_out, W_out).
        """

        # TODO Implement forward propogation
        # of 2d convolution module.
        B, C_in, H_in, W_in=x.shape
        H_out=int((H_in-self.kernel_size+2*self.padding)/self.stride+1)
        W_out=int((W_in-self.kernel_size+2*self.padding)/self.stride+1)
        base=np.ndarray((B,self.channels,H_out,W_out))
        self.x=x
        for b in range(B):
            for c in range(self.channels):
                for h in range(H_out):
                    for w in range(W_out):
                        base[b,c,h,w]=np.sum(self.kernel[c]*x[b,:,h*self.stride:h*self.stride+self.kernel_size,
                                            w*self.stride:w*self.stride+self.kernel_size])
        if self.bias is not None:
            ret=base+np.tile(self.bias,(B,H_out,W_out,1)).transpose(0,3,1,2)
        else:
            ret=base
        return ret
        ...

        # End of todo

    def backward(self, dy):
        """Backward propagation of convolution module.

        Args:
            dy: output delta of shape (B, C_out, H_out, W_out).
        Returns:
            dx: input delta of shape (B, C_in, H_in, W_in).
        """

        # TODO Implement backward propogation
        # of 2d convolution module.
        ret=np.zeros_like(self.x)
        self.kernel.grad=np.zeros_like(self.kernel)
        B, C_out, H_out, W_out=dy.shape
        for b in range(B):
            for c in range(C_out):
                for h in range(H_out):
                    for w in range(W_out):
                        ret[b,:,h*self.stride:h*self.stride+self.kernel_size,
                            w*self.stride:w*self.stride+self.kernel_size]+=dy[b,c,h,w]*self.kernel[c]
                        self.kernel.grad[c]+=dy[b,c,h,w]*self.x[b,:,h*self.stride:h*self.stride+self.kernel_size,
                            w*self.stride:w*self.stride+self.kernel_size]
        
        if self.bias is not None:
            self.bias.grad=np.sum(dy,axis=(0,2,3))
        return ret
        ...

        # End of todo


class Conv2d_im2col(Conv2d):#???????????????

    def forward(self, x):

        # TODO Implement forward propogation of
        # 2d convolution module using im2col method.
        B, C_in, H_in, W_in=x.shape
        H_out=int((H_in-self.kernel_size+2*self.padding)/self.stride+1)
        W_out=int((W_in-self.kernel_size+2*self.padding)/self.stride+1)
        base=np.ndarray((B,H_out,W_out,self.channels))
        cvtx=np.ndarray((B,H_out,W_out,C_in*self.kernel_size*self.kernel_size))
        kk=self.kernel.reshape((self.channels,C_in*self.kernel_size*self.kernel_size))
        for b in range(B):
            for h in range(H_out):
                for w in range(W_out):
                    cvtx[b,h,w]=x[b,:,h*self.stride:h*self.stride+self.kernel_size,
                            w*self.stride:w*self.stride+self.kernel_size].reshape((C_in*self.kernel_size*self.kernel_size,))
        for b in range(B):
            for h in range(H_out):
                base[b,h]=np.dot(cvtx[b,h],kk.T)
        base=base.transpose((0,3,1,2))
        if self.bias is not None:
            ret=base+np.tile(self.bias,(B,H_out,W_out,1)).transpose(0,3,1,2)
        else:
            ret=base
        return ret
        ...
        
        # End of todo


class AvgPool(Module):

    def __init__(self, kernel_size: int=2,
                 stride: int=2, padding: int=0):
        """Module which applies average pooling to input.

        Args:
            kernel_size: default 2.
            stride: default 2.
            padding: default 0.
        """

        # TODO Initialize the attributes
        # of average pooling module.
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        ...

        # End of todo

    def forward(self, x):
        """Forward propagation of average pooling module.

        Args:
            x: input of shape (B, C, H_in, W_in).
        Returns:
            out: output of shape (B, C, H_out, W_out).
        """

        # TODO Implement forward propogation
        # of average pooling module.
        self.x=x
        B, C, H_in, W_in=x.shape
        H_out=int((H_in-self.kernel_size+2*self.padding)/self.stride+1)
        W_out=int((W_in-self.kernel_size+2*self.padding)/self.stride+1)
        x_new=np.zeros((B,C,H_out,W_out))
        for h in range(H_out):
            for w in range(W_out):
                x_new[:, :, h, w] = np.mean(x[:, :,h*self.stride:h*self.stride+self.kernel_size,
                                            w*self.stride:w*self.stride+self.kernel_size],axis=(2,3))
        self.x_new=x_new
        return x_new
        ...

        # End of todo

    def backward(self, dy):
        """Backward propagation of average pooling module.

        Args:
            dy: output delta of shape (B, C, H_out, W_out).
        Returns:
            dx: input delta of shape (B, C, H_in, W_in).
        """

        # TODO Implement backward propogation
        # of average pooling module.
        x=self.x
        ret=np.zeros(x.shape)
        val=1/(self.kernel_size*self.kernel_size)
        B, C, H_out, W_out=dy.shape
        for h in range(H_out):
            for w in range(W_out):
                for l in range(B):
                    for r in range(C):
                        for i in range(h*self.stride,h*self.stride+self.kernel_size):
                            for j in range(w*self.stride,w*self.stride+self.kernel_size):
                                ret[l][r][i][j]+=dy[l][r][h][w]*val
        return ret
        ...

        # End of todo


class MaxPool(Module):

    def __init__(self, kernel_size: int=2,
                 stride: int=2, padding: int=0):
        """Module which applies max pooling to input.

        Args:
            kernel_size: default 2.
            stride: default 2.
            padding: default 0.
        """

        # TODO Initialize the attributes
        # of maximum pooling module.
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        ...

        # End of todo

    def forward(self, x):
        """Forward propagation of max pooling module.

        Args:
            x: input of shape (B, C, H_in, W_in).
        Returns:
            out: output of shape (B, C, H_out, W_out).
        """

        # TODO Implement forward propogation
        # of maximum pooling module.
        self.x=x
        B, C, H_in, W_in=x.shape
        H_out=int((H_in-self.kernel_size+2*self.padding)/self.stride+1)
        W_out=int((W_in-self.kernel_size+2*self.padding)/self.stride+1)
        x_new=np.zeros((B,C,H_out,W_out))
        for h in range(H_out):
            for w in range(W_out):
                x_new[:, :, h, w] = np.max(x[:, :,h*self.stride:h*self.stride+self.kernel_size,
                                            w*self.stride:w*self.stride+self.kernel_size],axis=(2,3))
        self.x_new=x_new
        return x_new
        ...

        # End of todo

    def backward(self, dy):
        """Backward propagation of max pooling module.

        Args:
            dy: output delta of shape (B, C, H_out, W_out).
        Returns:
            out: input delta of shape (B, C, H_in, W_in).
        """

        # TODO Implement backward propogation
        # of maximum pooling module.
        x=self.x
        ret=np.zeros(x.shape)
        B, C, H_out, W_out=dy.shape
        for h in range(H_out):
            for w in range(W_out):
                for l in range(B):
                    for r in range(C):
                        for i in range(h*self.stride,h*self.stride+self.kernel_size):
                            for j in range(w*self.stride,w*self.stride+self.kernel_size):
                                if(self.x_new[l][r][h][w]==self.x[l][r][i][j]):
                                    ret[l][r][i][j]+=dy[l][r][h][w]
        return ret


        ...

        # End of todo


class Dropout(Module):

    def __init__(self, p: float=0.5):

        # TODO Initialize the attributes
        # of dropout module.
        super().__init__()
        self.p=p#p??????????????????
        ...

        # End of todo

    def forward(self, x):

        # TODO Implement forward propogation
        # of dropout module.
        if self.training:
            self.mask=np.array(np.random.binomial(1,1-self.p,x.shape))
            return x*self.mask/(1-self.p)
        else:
            return x
        ...

        # End of todo

    def backard(self, dy):

        # TODO Implement backward propogation
        # of dropout module.
        return dy*self.mask/(1-self.p)
        ...

        # End of todo


if __name__ == '__main__':
    import pdb; pdb.set_trace()
