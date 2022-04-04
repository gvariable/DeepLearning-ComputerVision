import unittest

import tinytorch
import tinytorch.nn.functional as F
from tinytorch import autograd_engine as autograd


class MyTestCase(unittest.TestCase):

    def test_add(self):
        operation = F.Add

        a = tinytorch.ones(2, 2)
        b = tinytorch.ones(2, 2)

        grad_upstream = tinytorch.ones(2, 2)
        ctx = autograd.ContextManager()
        operation.forward(ctx, a, b)
        backward_grads = operation.backward(ctx, grad_upstream)
        numerical_grads = autograd.gradient_check(operation, a, b, df=grad_upstream.asarray())

        assert backward_grads.allclose(numerical_grads)


if __name__ == '__main__':
    unittest.main()
