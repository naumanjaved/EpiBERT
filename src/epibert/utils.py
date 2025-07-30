import inspect
from typing import Any, Callable, Dict, Optional, Text, Union, Iterable
import numpy as np
import tensorflow.experimental.numpy as tnp
import tensorflow as tf
#tnp.experimental_enable_numpy_behavior(prefer_float32=True)

from tensorflow.keras import layers as kl

'''
Genformer helper functions
'''


@tf.keras.utils.register_keras_serializable()
def sinusoidal(input_shape):
    '''
    Generates sinusoidal PE as in Vasvani et al. 2017 given an input_shape
    where input_shape[-2] is the sequence length and input_shape[-1] is the
    channel dimension
    
    '''
    position = tnp.arange(0, input_shape[-2], dtype = tnp.float32)
    position = tnp.expand_dims(position, axis = 1)
    omega = tnp.exp(-(tnp.arange(0,
                                 input_shape[-1],
                                 2, dtype=tnp.float32)) *
            tf.constant(4.0,dtype=tf.float32) /
                            tf.constant(input_shape[-1], dtype=tf.float32))

    even_indices = tf.cast(tnp.sin(position * omega), dtype=tnp.float32)
    odd_indices = tf.cast(tnp.cos(position * omega), dtype=tnp.float32)

    PE = tf.cast(tf.reshape(
                    tf.concat([even_indices[...,tf.newaxis],
                                odd_indices[...,tf.newaxis]], axis=-1),
                    [tf.shape(even_indices)[0],-1]),dtype=tf.bfloat16)

    return PE

@tf.keras.utils.register_keras_serializable()
def gen_channels_list(num, end_channels):
    '''
    Given a number of channels and desired number of conv layers outputs
    an evenly spaced list of channels nubmers from end_channels // 2 to
    end_channels
    '''
    out = [tf.cast(end_channels // (2**i), dtype=tf.int32) for i in range(num)]
    return out[::-1]

@tf.keras.utils.register_keras_serializable()
def exponential_linspace_int(start, end, num, divisible_by=1):
    """Exponentially increasing values of integers. From enformer"""
    def _round(x):
        return tf.cast((tf.math.round(x / divisible_by) * divisible_by),dtype=tf.int32)

    base = tf.cast(tnp.exp(tnp.log(end / start) / (num - 1)), dtype=tf.float32)
    return [_round(start * base**i) for i in range(num)]

'''
@tf.function
def crop_tensor(input_tensor,
                crop_size_list,
                target_length_list):
    return tf.slice(input_tensor,
                    crop_size_list,
                    target_length_list)


@tf.function
def crop_tensor_1d(input_tensor,
                    crop_size,
                    target_length):
    return tf.slice(input_tensor,
                    [crop_size],
                    [target_length])


@tf.function
def subset_tensor(input_tensor,
                  mask):
    keep_indices = tf.reshape(tf.where(tf.equal(mask,1)), [-1])
    return tf.gather(input_tensor,
                     indices=keep_indices,
                     axis=1)
'''



