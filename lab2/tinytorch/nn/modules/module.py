from collections import OrderedDict
from typing import Union

import tinytorch
from ..parameter import Parameter


class Module(object):
    def __init__(self):
        self.training = True
        self._parameters = OrderedDict()
        self._modules = OrderedDict()
        self._buffers = OrderedDict()

    def register_buffer(self, name, tensor):
        if "_buffers" not in self.__dict__:
            raise AttributeError("cannot assign buffer before Module.__init__()")
        buffers = self.__dict__["_buffers"]
        if hasattr(self, name) and name not in self._buffers:
            raise KeyError("buffer '{}' already exists".format(name))

        if not isinstance(tensor, tinytorch.Tensor):
            raise TypeError("cannot assign '{}' as buffer(tinytorch.Tensor expected)".format(name))
        buffers[name] = tensor

    def register_parameter(self, name, param):
        if "_parameters" not in self.__dict__:
            raise AttributeError("cannot assign parameter before Module.__init__()")
        params = self.__dict__["_parameters"]

        if hasattr(self, name) and name not in self._parameters:
            raise KeyError("parameter '{}' already exists".format(name))

        if param is None:
            params[name] = None
        elif not isinstance(param, Parameter):
            raise TypeError("cannot assign '{}' as parameter(tinytorch.Tensor or None expected)".format(name))
        else:
            params[name] = param

    def register_module(self, name, module):
        if "_modules" not in self.__dict__:
            raise AttributeError("cannot assign module before Module.__init__()")
        modules = self.__dict__["_modules"]
        if hasattr(self, name) and name not in self._modules:
            raise KeyError("module '{}' already exists".format(name))

        if not isinstance(module, Module):
            raise TypeError("cannot assign '{}' as module(tinytorch.Module expected)".format(name))
        modules[name] = module

    def __setattr__(self, name: str, value: Union['Module', tinytorch.Tensor]):

        def remove_from(*dicts):
            for d in dicts:
                if name in d:
                    del d[name]

        params = self.__dict__.get('_parameters')
        if isinstance(value, Parameter):
            if params is None:
                raise AttributeError("cannot assign parameter before Module.__init__()")
            remove_from(self.__dict__, self._buffers, self._modules)
            params[name] = value
        elif params is not None and name in params:
            if value is not None:
                raise TypeError("cannot assign '{}' as parameter(tinytorch.Tensor or None expected)".format(name))
            params[name] = value
        else:
            modules = self.__dict__.get('_modules')
            if isinstance(value, Module):
                if modules is None:
                    raise AttributeError("cannot assign module before Module.__init__()")
                remove_from(self.__dict__, self._parameters, self._buffers)
                modules[name] = value
            elif modules is not None and name in modules:
                if value is not None:
                    raise TypeError("cannot assign '{}' as module".format(name))
                modules[name] = value
            else:
                buffers = self.__dict__.get('_buffers')
                if buffers is not None and name in buffers:
                    if value is not None and not isinstance(value, tinytorch.Tensor):
                        raise TypeError("cannot assign '{}' as buffer".format(name))
                    buffers[name] = value
                else:
                    object.__setattr__(self, name, value)

    def __getattr__(self, name: str):
        if '_parameters' in self.__dict__:
            _parameters = self.__dict__['_parameters']
            if name in _parameters:
                return _parameters[name]

        if '_modules' in self.__dict__:
            _modules = self.__dict__['_modules']
            if name in _modules:
                return _modules[name]

        if '_buffers' in self.__dict__:
            _buffers = self.__dict__['_buffers']
            if name in _buffers:
                return _buffers[name]
        raise AttributeError("'{}' object has no attribute '{}'".format(type(self).__name__, name))

    def __delattr__(self, name):
        if name in self._parameters:
            del self._parameters[name]
        elif name in self._modules:
            del self._modules[name]
        elif name in self._buffers:
            del self._buffers[name]
        else:
            object.__delattr__(self, name)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def backward(self, *grad_output):
        raise NotImplementedError
