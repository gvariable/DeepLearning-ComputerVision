import numpy as np

from .autograd_engine import backward
from .nn import functional as F


class Tensor(object):
    def __init__(self, data, requires_grad: bool = False):
        if not isinstance(data, np.ndarray):
            data = np.array(data)
        self.data = data
        self.requires_grad = requires_grad
        self.grad = None
        self.is_leaf = True
        self.grad_fn = None
        self.is_parameter = False

    def __repr__(self):
        if self.requires_grad and self.is_leaf:
            return f"Tensor({self.data}, requires_grad=True)"
        elif self.requires_grad and not self.is_leaf:
            return f"Tensor({self.data}, grad_fn={self.grad_fn})"
        else:
            return f"Tensor({self.data})"

    __str__ = __repr__

    def __array__(self):
        return self.data

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if method == "__call__":
            _args = [arg.data if isinstance(arg, self.__class__) else np.array(arg) for arg in inputs]
            return self.__class__(ufunc(*_args, **kwargs))
        else:
            return NotImplemented

    def __array_function__(self, func, types, args, kwargs):
        if func == np.ndarray.__array_ufunc__:
            return self.__array_ufunc__(*args, **kwargs)
        else:
            _args = []
            for arg in args:
                if isinstance(arg, Tensor):
                    _args.append(arg.data)
                else:
                    _args.append(arg)

            return self.__class__(func(*_args, **kwargs))

    def backward(self, grad=None):
        assert self.requires_grad, "Tensor does not require gradient"
        if not grad:
            grad = ones_like(self)

        backward(self, grad)

    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim

    @property
    def dtype(self):
        return self.data.dtype

    def size(self):
        return self.shape

    def asarray(self):
        return self.data


# """*********************Create Ops*********************"""
def zeros(*shape, dtype=None, requires_grad=False):
    return Tensor(np.zeros(shape, dtype=dtype), requires_grad)


def zeros_like(tensor, dtype=None, requires_grad=False):
    return Tensor(np.zeros_like(tensor.data, dtype=dtype), requires_grad)


def ones(*shape, dtype=None, requires_grad=False):
    return Tensor(np.ones(shape, dtype=dtype), requires_grad)


def ones_like(tensor, dtype=None, requires_grad=False):
    return Tensor(np.ones_like(tensor.data, dtype=dtype), requires_grad)


def arange(start, stop=None, step=1, dtype=None, requires_grad=False):
    return Tensor(np.arange(start, stop, step, dtype=dtype), requires_grad)


def range(start, stop, step=1, dtype=None, requires_grad=False):
    return Tensor(np.arange(start, stop, step, dtype=dtype), requires_grad)


def linspace(start, stop, num, dtype=None, requires_grad=False):
    return Tensor(np.linspace(start, stop, num, dtype=dtype), requires_grad)


def logspace(start, stop, num, dtype=None, requires_grad=False):
    return Tensor(np.logspace(start, stop, num, dtype=dtype), requires_grad)


def eye(dim, dtype=None, requires_grad=False):
    return Tensor(np.eye(dim).astype(dtype), requires_grad)


def empty(*shape, dtype=None, requires_grad=False):
    return Tensor(np.empty(shape, dtype=dtype), requires_grad)


def empty_like(tensor, dtype=None, requires_grad=False):
    return Tensor(np.empty_like(tensor.data, dtype=dtype), requires_grad)


def empty_strided(*shape, dtype=None, requires_grad=False):
    return Tensor(np.empty_strided(shape, dtype=dtype), requires_grad)


def full(*shape, fill_value, dtype=None, requires_grad=False):
    return Tensor(np.full(shape, fill_value, dtype=dtype), requires_grad)


def full_like(tensor, fill_value, dtype=None, requires_grad=False):
    return Tensor(np.full_like(tensor.data, fill_value, dtype=dtype), requires_grad)


def tensor(data, requires_grad=False):
    return Tensor(data, requires_grad)


# """*********************Random Sampling*********************"""
def seed(_seed):
    np.random.seed(_seed)


def manual_seed(_seed):
    np.random.seed(_seed)


def bernoulli(p, requires_grad=False):
    return Tensor(np.random.binomial(1, p, size=None), requires_grad)


def normal(mean=0, std=1, requires_grad=False):
    return Tensor(np.random.normal(mean, std, size=None), requires_grad)


def randn(*shape, dtype=None, requires_grad=False):
    return Tensor(np.random.randn(shape).astype(dtype), requires_grad)


def randn_like(tensor, dtype=None, requires_grad=False):
    return Tensor(np.random.randn(tensor.shape).astype(dtype), requires_grad)


def randint(low, high, shape=None, dtype=None, requires_grad=False):
    return Tensor(np.random.randint(low, high, size=shape).astype(dtype), requires_grad)


def randint_like(tensor, low, high, dtype=None, requires_grad=False):
    return Tensor(np.random.randint(low, high, size=tensor.shape).astype(dtype), requires_grad)


def uniform(low=0.0, high=1.0, size=None, dtype=None, requires_grad=False):
    return Tensor(np.random.uniform(low, high, size).astype(dtype), requires_grad)


# """*********************Operation*********************"""
def tensor_wrapper(func, *args, **kwargs):
    _args = [arg if isinstance(arg, Tensor) else Tensor(arg) for arg in args]
    import ipdb
    ipdb.set_trace()
    func.apply(*_args, **kwargs)


Tensor.__neg__ = lambda self: tensor_wrapper(F.Neg, self)
Tensor.__add__ = lambda self, other: tensor_wrapper(F.Add, self, other)
Tensor.__radd__ = Tensor.__add__
Tensor.__sub__ = lambda self, other: tensor_wrapper(F.Sub, self, other)
Tensor.__rsub__ = lambda self, other: tensor_wrapper(F.Sub, other, self)
Tensor.__mul__ = lambda self, other: tensor_wrapper(F.Mul, other, self)
Tensor.__rmul__ = Tensor.__mul__
Tensor.__truediv__ = lambda self, other: tensor_wrapper(F.Div, self, other)
Tensor.__rtruediv__ = lambda self, other: tensor_wrapper(F.Div, other, self)
Tensor.__pow__ = lambda self, other: tensor_wrapper(F.Pow, self, other)
Tensor.__rpow__ = lambda self, other: tensor_wrapper(F.Pow, other, self)

Tensor.log = lambda self: tensor_wrapper(F.Log, self)
Tensor.exp = lambda self: tensor_wrapper(F.Exp, self)
Tensor.abs = lambda self: tensor_wrapper(F.Abs, self)
Tensor.mm = lambda self, other: tensor_wrapper(F.Matmul, self, other)
Tensor.dot = Tensor.mm
Tensor.sum = lambda self, dim=None: tensor_wrapper(F.Sum, self, dim)

Tensor.__int__ = lambda self: int(self.data) if np.prod(self.shape) == 1 else None
Tensor.__float__ = lambda self: int(self.data) if np.prod(self.shape) == 1 else None
