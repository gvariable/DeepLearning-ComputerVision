import numpy as np

from ..autograd_engine import Function, ContextManager


def unbrocast(arr, shape):
    if len(shape) == 1:
        sum_axis = tuple(range(0, len(arr.shape) - 1))
    else:
        sum_axis = tuple(i for i in range(len(shape)) if shape[i] == 1 and arr.shape[i] > 1) if arr.shape != (
            1,) else None
    return np.sum(arr, axis=sum_axis) if sum_axis else arr


class Add(Function):
    # TODO(gpl): add support for int/float etc.
    @staticmethod
    def forward(ctx: ContextManager, x0, x1):
        ctx.save_for_backward(x0.shape, x1.shape)
        return np.add(x0, x1)

    @staticmethod
    def backward(ctx: ContextManager, grad_output):
        x0_shape, x1_shape = ctx.saved_tensors
        return unbrocast(grad_output, x0_shape), unbrocast(grad_output, x1_shape)


class Sub(Function):

    @staticmethod
    def forward(ctx: ContextManager, x0, x1):
        ctx.save_for_backward(x0.shape, x1.shape)
        return np.subtract(x0, x1)

    @staticmethod
    def backward(ctx: ContextManager, grad_output):
        x0_shape, x1_shape = ctx.saved_tensors
        return unbrocast(grad_output, x0_shape), unbrocast(-grad_output, x1_shape)


class Mul(Function):

    @staticmethod
    def forward(ctx: ContextManager, x0, x1):
        ctx.save_for_backward(x0, x1)
        return np.multiply(x0, x1)

    @staticmethod
    def backward(ctx: ContextManager, grad_output):
        x0, x1 = ctx.saved_tensors
        return unbrocast(np.multiply(grad_output, x1), x0.shape), unbrocast(np.multiply(grad_output, x0), x1.shape)


class Div(Function):

    @staticmethod
    def forward(ctx: ContextManager, x0, x1):
        ctx.save_for_backward(x0, x1)
        return np.divide(x0, x1)

    @staticmethod
    def backward(ctx: ContextManager, grad_output):
        x0, x1 = ctx.saved_tensors
        return unbrocast(np.divide(grad_output, x1), x0.shape), unbrocast(
            np.divide(-np.multiply(grad_output, x0), np.multiply(x1, x1)),
            x1.shape)


class Neg(Function):

    @staticmethod
    def forward(ctx: ContextManager, x):
        return np.negative(x)

    @staticmethod
    def backward(ctx: ContextManager, grad_output):
        return np.negative(grad_output)


class Pow(Function):

    @staticmethod
    def forward(ctx: ContextManager, x0, x1):
        ctx.save_for_backward(x0, x1)
        return np.power(x0, x1)

    @staticmethod
    def backward(ctx: ContextManager, grad_output):
        x0, x1 = ctx.saved_tensors
        return unbrocast(np.multiply(np.multiply(np.power(x0, np.subtract(x1, 1), x1), grad_output), x0.shape),
                         unbrocast(np.multiply(np.multiply(np.log(x0), grad_output), x0), x1.shape))


class Transpose(Function):
    def __init__(self):
        super(Transpose, self).__init__()


class Reshape(Function):
    def __init__(self):
        super(Reshape, self).__init__()


class Log(Function):

    @staticmethod
    def forward(ctx: ContextManager, x):
        ctx.save_for_backward(x)
        return np.log(x)

    @staticmethod
    def backward(ctx: ContextManager, grad_output):
        x = ctx.saved_tensors[0]
        return np.divide(grad_output, x)


class Exp(Function):

    @staticmethod
    def forward(ctx: ContextManager, x):
        ctx.save_for_backward(x)
        return np.exp(x)

    @staticmethod
    def backward(ctx: ContextManager, grad_output):
        x = ctx.saved_tensors[0]
        return np.multiply(grad_output, x)


class Abs(Function):

    @staticmethod
    def forward(ctx: ContextManager, x):
        ctx.save_for_backward(x)
        return np.abs(x)

    @staticmethod
    def backward(ctx: ContextManager, grad_output):
        x = ctx.saved_tensors[0]
        return np.multiply(grad_output, np.sign(x))


class Matmul(Function):

    @staticmethod
    def forward(ctx: ContextManager, x0, x1):
        ctx.save_for_backward(x0, x1)
        return np.matmul(x0, x1)

    @staticmethod
    def backward(ctx: ContextManager, grad_output):
        x0, x1 = ctx.saved_tensors
        return np.matmul(grad_output, x1.T), np.matmul(x0.T, grad_output)


class Sum(Function):

    @staticmethod
    def forward(ctx: ContextManager, x, dim):
        ctx.save_for_backward(x.shape)
        if dim:
            return np.sum(x, axis=tuple(dim.data))
        else:
            return np.sum(x)

    @staticmethod
    def backward(ctx: ContextManager, grad_output):
        shape = ctx.saved_tensors[0]
        return (np.broadcast_to(grad_output, shape),)
