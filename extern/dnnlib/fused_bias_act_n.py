# Copyright (c) 2019, NVIDIA Corporation. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://nvlabs.github.io/stylegan2/license.html

"""Custom TensorFlow ops for efficient bias and activation."""

import numpy as np
import tensorflow as tf
from typing import Any


# Util classes
# ------------------------------------------------------------------------------------------

class EasyDict(dict):
    """Convenience class that behaves like a dict but allows access with the attribute syntax."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def __delattr__(self, name: str) -> None:
        del self[name]

#----------------------------------------------------------------------------

activation_funcs = {
    'linear':   EasyDict(func=lambda x, **_:        x,                          def_alpha=None, def_gain=1.0,           cuda_idx=1, ref='y', zero_2nd_grad=True),
    'relu':     EasyDict(func=lambda x, **_:        tf.nn.relu(x),              def_alpha=None, def_gain=np.sqrt(2),    cuda_idx=2, ref='y', zero_2nd_grad=True),
    'lrelu':    EasyDict(func=lambda x, alpha, **_: tf.nn.leaky_relu(x, alpha), def_alpha=0.2,  def_gain=np.sqrt(2),    cuda_idx=3, ref='y', zero_2nd_grad=True),
    'tanh':     EasyDict(func=lambda x, **_:        tf.nn.tanh(x),              def_alpha=None, def_gain=1.0,           cuda_idx=4, ref='y', zero_2nd_grad=False),
    'sigmoid':  EasyDict(func=lambda x, **_:        tf.nn.sigmoid(x),           def_alpha=None, def_gain=1.0,           cuda_idx=5, ref='y', zero_2nd_grad=False),
    'elu':      EasyDict(func=lambda x, **_:        tf.nn.elu(x),               def_alpha=None, def_gain=1.0,           cuda_idx=6, ref='y', zero_2nd_grad=False),
    'selu':     EasyDict(func=lambda x, **_:        tf.nn.selu(x),              def_alpha=None, def_gain=1.0,           cuda_idx=7, ref='y', zero_2nd_grad=False),
    'softplus': EasyDict(func=lambda x, **_:        tf.nn.softplus(x),          def_alpha=None, def_gain=1.0,           cuda_idx=8, ref='y', zero_2nd_grad=False),
    'swish':    EasyDict(func=lambda x, **_:        tf.nn.sigmoid(x) * x,       def_alpha=None, def_gain=np.sqrt(2),    cuda_idx=9, ref='x', zero_2nd_grad=False),
}

#----------------------------------------------------------------------------

def fused_bias_act(x, b=None, axis=1, act='linear', alpha=None, gain=None, impl='cuda'):
    r"""Fused bias and activation function.

    Adds bias `b` to activation tensor `x`, evaluates activation function `act`,
    and scales the result by `gain`. Each of the steps is optional. In most cases,
    the fused op is considerably more efficient than performing the same calculation
    using standard TensorFlow ops. It supports first and second order gradients,
    but not third order gradients.

    Args:
        x:      Input activation tensor. Can have any shape, but if `b` is defined, the
                dimension corresponding to `axis`, as well as the rank, must be known.
        b:      Bias vector, or `None` to disable. Must be a 1D tensor of the same type
                as `x`. The shape must be known, and it must match the dimension of `x`
                corresponding to `axis`.
        axis:   The dimension in `x` corresponding to the elements of `b`.
                The value of `axis` is ignored if `b` is not specified.
        act:    Name of the activation function to evaluate, or `"linear"` to disable.
                Can be e.g. `"relu"`, `"lrelu"`, `"tanh"`, `"sigmoid"`, `"swish"`, etc.
                See `activation_funcs` for a full list. `None` is not allowed.
        alpha:  Shape parameter for the activation function, or `None` to use the default.
        gain:   Scaling factor for the output tensor, or `None` to use default.
                See `activation_funcs` for the default scaling of each activation function.
                If unsure, consider specifying `1.0`.
        impl:   Name of the implementation to use. Can be `"ref"` or `"cuda"` (default).

    Returns:
        Tensor of the same shape and datatype as `x`.
    """

    return _fused_bias_act_ref(x=x, b=b, axis=axis, act=act, alpha=alpha, gain=gain)

#----------------------------------------------------------------------------

def _fused_bias_act_ref(x, b, axis, act, alpha, gain):
    """Slow reference implementation of `fused_bias_act()` using standard TensorFlow ops."""

    # Validate arguments.
    x = tf.convert_to_tensor(x)
    b = tf.convert_to_tensor(b) if b is not None else tf.constant([], dtype=x.dtype)
    act_spec = activation_funcs[act]
    assert b.shape.ndims == 1 and (b.shape[0] == 0 or b.shape[0] == x.shape[axis])
    assert b.shape[0] == 0 or 0 <= axis < x.shape.ndims
    if alpha is None:
        alpha = act_spec.def_alpha
    if gain is None:
        gain = act_spec.def_gain

    # Add bias.
    if b.shape[0] != 0:
        x += tf.reshape(b, [-1 if i == axis else 1 for i in range(x.shape.ndims)])

    # Evaluate activation function.
    x = act_spec.func(x, alpha=alpha)

    # Scale by gain.
    if gain != 1:
        x *= gain
    return x
