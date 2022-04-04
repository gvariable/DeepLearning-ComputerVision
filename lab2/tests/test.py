import unittest
from itertools import product

import tinytorch
import tinytorch.nn.functional as F
from tinytorch import autograd_engine as autograd


class MyTestCase(unittest.TestCase):

    def compare(self, numerical_grads, backward_grads, *tensors):
        _backwards_grad = []
        for i, tensor in enumerate(tensors):
            if isinstance(tensor, tinytorch.Tensor):
                _backwards_grad.append(backward_grads[i])

        if_eqs = []
        for backward_grad, numerical_grad in zip(_backwards_grad, numerical_grads):
            if_eqs.append(tinytorch.allclose(backward_grad, numerical_grad))

        return all(if_eqs)

    def test_BinaryOps(self):
        a = tinytorch.randn(2, 2)
        # 1. multi tensor & tensor
        # 2. tensor & scalar
        # 3. tensor & vector
        bs = [tinytorch.randn(2, 2), 1, tinytorch.randn(2, 1)]
        operations = [F.Add, F.Sub, F.Mul, F.Div]
        grad_upstream = tinytorch.ones(2, 2)
        ctx = autograd.ContextManager()

        for b, operation in product(bs, operations):
            operation.forward(ctx, a, b if isinstance(b, tinytorch.Tensor) else tinytorch.tensor(b))
            backward_grads = operation.backward(ctx, grad_upstream)
            numerical_grads = autograd.gradient_check(operation, a, b, df=grad_upstream.asarray())
            self.assertTrue(self.compare(numerical_grads, backward_grads, a, b))

    def test_SingleOps(self):
        operations = [F.Neg, F.Sqrt, F.Exp, F.Log, F.Tanh, F.Sigmoid, F.Relu, F.Softmax]
        for operation in operations:
            a = tinytorch.randn(2, 2)
            grad_upstream = tinytorch.ones(2, 2)
            ctx = autograd.ContextManager()

            operation.forward(ctx, a)
            backward_grads = operation.backward(ctx, grad_upstream)
            numerical_grads = autograd.gradient_check(operation, a, df=grad_upstream.asarray())
            self.assertTrue(self.compare(numerical_grads, backward_grads, [a]))

    if __name__ == '__main__':
        unittest.main()
