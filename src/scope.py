from __future__ import annotations
import abc
from typing import Tuple, Type, TypeVar, Union
from collections import OrderedDict


import tensorflow as tf
from tensorflow import keras

BaseScope = TypeVar('BaseScope', bound='Scope')


class Scope(keras.layers.Layer, metaclass=abc.ABCMeta):

    _scopes = OrderedDict({})

    @property
    def scopes(self) -> OrderedDict:
        return Scope._scopes

    def __init__(self,
                 layer: Union[list[str], str]):
        super().__init__()
        if isinstance(layer, str):
            layer = [layer]
        self.layer = layer

        self._loss_weight = 1.0
        self.model = None

    @classmethod
    def __subclasshook__(cls, subclass):
        # TODO: define subclasshook
        return (hasattr(subclass, 'load_data_source') and
                callable(subclass.load_data_source) and
                hasattr(subclass, 'extract_text') and
                callable(subclass.extract_text))

    def get_model_layers(self, model) -> list[str]:
        return self._layers

    @abc.abstractmethod
    def call(self, *args, **kwargs):
        # TODO: add tensor type via tensorflow extension types
        """Define the scope access"""
        raise NotImplementedError()

    def __call__(self, *args, **kwargs):
        tensor, *args = args
        tensor = self._batch_outputs(tensor)
        super(self, Layer).__call__(tensor, *args, **kwargs)

    def _batch_outputs(self, tensor):
        """Process list of tensors into one Tensor"""
        tensor = self.model(tensor, training=False)
        # check possible tensor states
        # 1. Single Tensor [B, H, W, C] -> Nothing to do
        if len(tensor) == 1:
            pass
        else:
            # List of tensors [[B, H, W, C], [[B, H, W, C]]
            batch, height, width, channel = zip(*[list(*t.shape()) for t in tensor])
            # 3. Check for similar h,w dimensions and create uniform tensor
            if all(h == height[0] for h in height) and all(w == width[0] for w in width):
                # same dimensions -> check if concat by channel or batch
                # for now just concat by channel
                tensor = tf.concat([t.reshape((1, height[0], width[0], -1)) for t in tensor], axis=-1)
            # 4. create a ragged tensor
            tensor = tf.ragged.constant(tensor)
            tf.get_logger().info(f"Created ragged tensor of shape {tensor.shape.as_list()} during"
                                 f"scope call")
        return tensor

    @classmethod
    def _add_scope(cls: Type[BaseScope], scope: BaseScope, weight: float):
        BaseScope.scopes.update(
            {len(cls.scopes) + 1: {'item': scope,
                                   'weight': weight}}
        )

    def __mul__(self, other: Union[float, int]):
        if isinstance(float, int):
            self._loss_weight = float(other)
            return self
        else:
            raise ValueError("multiplier must be int or float")

    def __rmul__(self, other):
        return self.__mul__(other)

    def __add__(self,
                other: BaseScope):
        BaseScope._add_scope(self, self._loss_weight)
        BaseScope._add_scope(self, other._loss_weight)
        return self

    def __sub__(self,
                other: BaseScope):
        self + (-1*other)


class Block(Scope):

    def __init__(self,
                 layer: Union[list[str], str]):
        super(Block, self).__init__(layer=layer)
        layer_len = len(self.layer)
        if layer_len != 1 and layer_len % 2 != 0:
            raise ValueError("Layer names are not divisible into pairs of two")

    def _unpack(self,
                model: keras.Model,
                layer_list: list,
                begin: str = None,
                end: str = None,
                *args):
        if end is None:
            return layer_list
        end = model.layers.index(end)
        start = model.layers.index(begin) if begin is not None else 0
        layer_list.extend(model.layers[start:end])
        self._unpack(model, layer_list,  *args)

    def get_model_layers(self, model: keras.Model) -> list[str]:
        layer_list = []
        names = [None, *self.layer] if len(self.layer) == 1 else self.layer
        layer_list = self._unpack(model, layer_list, *names)
        return layer_list

    def call(self, tensor, *args, **kwargs):
        # No special access operation
        return tensor


class Layer(Scope):

    def __init__(self,
                 layer: Union[list[str], str]):
        super(Layer, self).__init__(layer=layer)

    def call(self, tensor, *args, **kwargs):
        # No special access operation
        return tensor

    def __getitem__(self,
                    item: Union[int, slice]) -> Channel:
        return Channel(self.layer, item)


class Channel(Scope):

    def __init__(self,
                 layer: Union[list[str], str],
                 index: Union[int, slice]):
        super(Channel, self).__init__(layer=layer)
        self.index = index

    def call(self, tensor, *args, **kwargs):
        return tensor[..., self.index]

    def __getitem__(self,
                    item: Tuple[Union[int, slice], Union[int, slice]]) -> Neuron:
        return Neuron(self.layer, item)


class Neuron(Scope):

    def __init__(self,
                 layer: Union[list[str], str],
                 index: Tuple[Union[int, slice], Union[int, slice]]):
        super(Neuron, self).__init__(layer=layer)
        self.index = index

    def call(self, tensor, *args, **kwargs):
        height, width = self.index
        return tensor[:, height, width, :]
