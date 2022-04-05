from typing import Any, Union

import numpy as np


class Function(object):
    def __init__(self):
        ...

    @staticmethod
    def forward(ctx: Any, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("You must implement the forward function for custom functions")

    @staticmethod
    def backward(ctx: Any, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("You must implement the backward function for custom functions")

    @classmethod
    def apply(cls, *args, **kwargs):
        """
         1. Create a node object for the operation's output
         2. Run the operation on the input(s) to get an output tensor
         3. Store information on the node, which links it to the computation graph
         4, Store the node on the output tensor
         5. Return the output tensor
        """
        ctx = ContextManager()
        requires_grads = []
        _args = [tensor for tensor in args if hasattr(tensor, "requires_grad")]
        for tensor in args:
            if tensor and not isinstance(tensor, (int, float)):
                requires_grads.append(tensor.requires_grad)
                if not tensor.grad_fn and tensor.requires_grad and tensor.is_leaf:
                    setattr(tensor, "grad_fn", AccumulatedGrad(tensor))

        output = cls.forward(ctx, *args, **kwargs)
        import ipdb;
        ipdb.set_trace()
        if any(requires_grads):
            output.requires_grad = True
            output.is_leaf = False
            output.grad_fn = BackwardFunction(ctx, cls, _args)

        return output


class AccumulatedGrad(Function):
    def __init__(self, tensor):
        super().__init__()
        self.tensor = tensor

    def apply(self, grad):
        if not self.tensor.grad:
            self.tensor.grad = grad
        else:
            self.tensor.grad += grad


class ContextManager(object):

    def save_for_backward(self, *tensors):
        setattr(self, "_saved_tensors", tensors)

    @property
    def saved_tensors(self):
        return getattr(self, "_saved_tensors", None)


class BackwardFunction(Function):
    def __init__(self, ctx, operation, parents):
        super().__init__()
        self.ctx = ctx
        self.operation = operation
        self.parents = parents

    def __repr__(self):
        return f"<{self.operation.__name__}Backward>"

    def apply(self, grad_output):
        return self.operation.backward(self.ctx, grad_output)


def backward(tensor, grad_tensor=1.0):
    """DFS traversal of the graph to compute the gradients"""
    assert tensor.requires_grad, "Tensor must require gradients"
    if isinstance(grad_tensor, (int, float)):
        assert tensor.shape == (), "Shape of grad_tensor must be same as tensor"
    else:
        assert tensor.shape == grad_tensor.shape, "Shape of grad_tensor must be same as tensor"

    visited = set()

    def _dfs(node, grad_upstream=None):
        if node not in visited:
            visited.add(node)
            if isinstance(node, BackwardFunction):
                grads = node.apply(grad_upstream)
                for grad, tensor in zip(grads, node.parents):
                    if tensor.requires_grad:
                        if isinstance(tensor.grad_fn, BackwardFunction):
                            # tensor.grad = grad  # TODO(gpl): for debug usage
                            _dfs(tensor.grad_fn, grad)
                        elif isinstance(tensor.grad_fn, AccumulatedGrad):
                            tensor.grad_fn.apply(grad)
            elif isinstance(node, AccumulatedGrad):
                node.apply(grad_upstream)

    _dfs(tensor.grad_fn, grad_tensor)


def gradient_check(operation, *args, df: Union[float, np.ndarray] = 1, h: float = 1e-4):
    numerical_grads = []
    cls = None
    ctx = ContextManager()
    wrapped_args = None

    for arg in args:
        if arg:
            if isinstance(arg, (int, float)):
                continue
            if not cls:
                cls = arg.__class__
                wrapped_args = [arg if isinstance(arg, cls) else cls(arg) for arg in args]
            arr = arg.asarray()
            grad = np.zeros_like(arr)
            it = np.nditer(grad, flags=["multi_index"], op_flags=["readwrite"])
            while not it.finished:
                ix = it.multi_index
                oldval = arr[ix]

                arr[ix] = oldval + h
                output_plus = operation.forward(ctx, *wrapped_args).asarray()

                arr[ix] = oldval - h
                output_minus = operation.forward(ctx, *wrapped_args).asarray()
                arr[ix] = oldval

                grad[ix] = np.sum((output_plus - output_minus) * df) / (2 * h)
                it.iternext()

            numerical_grads.append(cls(grad))

    return numerical_grads


if __name__ == "__main__":
    ...
