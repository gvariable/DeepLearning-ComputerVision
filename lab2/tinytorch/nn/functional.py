import numpy as np

from ..autograd_engine import Function, ContextManager


def unbrocast(arr, shape):
    if len(shape) == 1:
        sum_axis = tuple(range(0, len(arr.shape) - 1))
    else:
        if arr.shape != (1,):
            sum_axis = []
            for i in range(-1, -len(shape) - 1, -1):
                if shape[i] == 1 and arr.shape[i] > 1:
                    sum_axis.append(len(arr.shape) + i)
            sum_axis.extend(list(range(len(arr.shape) - len(shape))))
            sum_axis = tuple(sum_axis)
        else:
            sum_axis = None
    return np.sum(arr, axis=sum_axis, keepdims=True) if sum_axis else arr


def transpose(tensor):
    axes = tuple(range(len(tensor.shape) - 2)) + (len(tensor.shape) - 1, len(tensor.shape) - 2)
    return np.transpose(tensor, axes)


class Add(Function):
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


class Sqrt(Function):

    @staticmethod
    def forward(ctx: ContextManager, x):
        ctx.save_for_backward(x)
        return np.sqrt(x)

    @staticmethod
    def backward(ctx: ContextManager, grad_output):
        x = ctx.saved_tensors[0]
        return np.divide(np.multiply(0.5, grad_output), np.sqrt(x))


class Tanh(Function):

    @staticmethod
    def forward(ctx: ContextManager, x):
        ctx.save_for_backward(x)
        return np.tanh(x)

    @staticmethod
    def backward(ctx: ContextManager, grad_output):
        x = ctx.saved_tensors[0]
        return np.multiply(np.subtract(1, np.square(np.tanh(x))), grad_output)


class Sigmoid(Function):

    @staticmethod
    def forward(ctx: ContextManager, x):
        ctx.save_for_backward(x)
        return np.divide(1, (np.add(1, np.exp(np.negative(x)))))

    @staticmethod
    def backward(ctx: ContextManager, grad_output):
        def _sigmoid(x):
            return np.divide(1, (np.add(1, np.exp(np.negative(x)))))

        x = ctx.saved_tensors[0]
        return np.multiply(np.multiply(_sigmoid(x), np.subtract(1, _sigmoid(x))), grad_output)


class Relu(Function):

    @staticmethod
    def forward(ctx: ContextManager, x):
        ctx.save_for_backward(x)
        return np.maximum(0, x)

    @staticmethod
    def backward(ctx: ContextManager, grad_output):
        x = ctx.saved_tensors[0]
        return np.multiply(np.greater(x, 0), grad_output)


class Softmax(Function):

    @staticmethod
    def forward(ctx: ContextManager, x):
        ctx.save_for_backward(x)
        return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)

    @staticmethod
    def backward(ctx: ContextManager, grad_output):
        x = ctx.saved_tensors[0]
        return np.multiply(grad_output, np.subtract(np.exp(x), np.sum(np.exp(x), axis=1, keepdims=True)))


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
        return np.matmul(grad_output, np.transpose(x1)), np.matmul(np.transpose(x0), grad_output)


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


class Linear(Function):

    @staticmethod
    def forward(ctx: ContextManager, x, w, b=None):
        """
        Args:
            ctx: ContextManager
            x: (*, in_features)
            w: (out_features, in_features) or (in_features)
            b: (out_features) or None

        Returns:

        """
        if len(w.shape) == 1:
            shp = [1] * (len(x.shape) - 1) + [-1]
            w = np.reshape(w, shp)
        ctx.save_for_backward(x, w, b)
        if b:
            return np.add(np.matmul(x, transpose(w)), b)
        return np.matmul(x, transpose(w))

    @staticmethod
    def backward(ctx: ContextManager, grad_output):
        x, w, b = ctx.saved_tensors
        if b:
            return (unbrocast(np.matmul(grad_output, w), x.shape),
                    unbrocast(transpose(np.matmul(transpose(x), grad_output)), w.shape),
                    unbrocast(grad_output, b.shape))

        return (unbrocast(np.matmul(grad_output, w), x.shape),
                unbrocast(transpose(np.matmul(transpose(x), grad_output)), w.shape))
