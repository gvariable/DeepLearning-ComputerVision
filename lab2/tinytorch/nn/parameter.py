import tinytorch


class Parameter(tinytorch.Tensor):
    def __init__(self, data=None, requires_grad=True):
        super(Parameter, self).__init__(data, requires_grad)
        ...

    def __repr__(self):
        return 'Parameter containing:\n' + super().__repr__()
