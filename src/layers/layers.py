# -*- coding: utf-8 -*-
"""Common building blocks (convolutions, FFN, Performer encoder, etc.) used by
EpiBERT models.

Only documentation and unused-import clean-ups were added—layer names, class
names, and call signatures are preserved so existing checkpoints remain
compatible.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import inspect
from typing import Any, Callable, Dict, Optional, Text, Union, Iterable

import tensorflow.experimental.numpy as tnp
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras import layers as kl
#import src.layers.fast_attention as fa
import src.layers.snnk_attention as snnk_attention
import src.utils as utils
from tensorflow.keras import regularizers
from tensorflow.keras.layers.experimental import SyncBatchNormalization as syncbatchnorm

#def custom_constant_initializer(shape, dtype=None):
#    return tf.constant(2.0, shape=shape, dtype=dtype)

@tf.keras.utils.register_keras_serializable()
class SoftmaxPooling1D(kl.Layer):
    """Softmax pooling layer."""
    def __init__(self,
                 pool_size=2, 
                 w_init_scale=2.0, 
                 kernel_init=None,
                 name='softmax_pooling', **kwargs):
        super().__init__(name=name, **kwargs)
        self.pool_size = pool_size
        self.w_init_scale = w_init_scale
        self.dense = kl.Dense(
            units = 1,
            use_bias=False,
            kernel_initializer=kernel_init if kernel_init is not None else 'lecun_normal')

    def call(self, inputs, **kwargs):
        _, length, num_features = inputs.shape
        # Reshape input for pooling
        inputs = tf.reshape(inputs, [-1, length//self.pool_size,
                                     self.pool_size, num_features])

        dense_out = self.dense(inputs)
        # Compute softmax weights
        dense_out = tf.nn.softmax(dense_out, axis=-2)
        return tf.reduce_sum(inputs * dense_out, axis=-2)

def conv_block(
    filters, 
    width=1,
    k_init=None, 
    b_init=None, 
    padding='same', 
    name='conv_block',
    beta_init=None, 
    gamma_init=None, 
    mean_init=None, 
    var_init=None, 
    dilation_rate=1, 
    stride=1,
    BN_momentum=0.90,
    **kwargs):
    """
    Creates a convolutional block with batch normalization and GELU activation.

    Parameters:
    - filters: Number of filters for the Conv1D layer.
    - width: Kernel size for the Conv1D layer.
    - k_init: Initializer for the kernel weights.
    - b_init: Initializer for the bias.
    - padding: Padding type for the Conv1D layer.
    - name: Name for the sequential model.
    - dilation_rate: Dilation rate for the Conv1D layer.
    - stride: Stride for the Conv1D layer.
    - **kwargs: Additional keyword arguments passed to BatchNormalization and Conv1D layers.

    Returns:
    - A TensorFlow Keras Sequential model comprising a batch normalization layer,
    a GELU activation layer, and a Conv1D layer.
    """
    return tf.keras.Sequential([
        tf.keras.layers.BatchNormalization(
            axis=-1, 
            synchronized=True, 
            center=True, 
            scale=True, 
            beta_initializer=beta_init if (beta_init is not None) else "zeros",
            gamma_initializer=gamma_init if (gamma_init is not None) else "ones",
            momentum=BN_momentum, 
            epsilon=1.0e-05, 
            moving_mean_initializer=mean_init if (mean_init is not None) else "zeros",
            moving_variance_initializer=var_init if (var_init is not None) else "ones", **kwargs),
        tfa.layers.GELU(),
        tf.keras.layers.Conv1D(
            filters, 
            width, 
            kernel_initializer=k_init if (k_init is not None) else 'lecun_normal',
            bias_initializer=b_init if (b_init is not None) else 'zeros',
            strides=stride, 
            dilation_rate=dilation_rate, 
            padding=padding, 
            **kwargs)
        ], name=name)

@tf.keras.utils.register_keras_serializable()
class pt_init(tf.keras.initializers.Initializer):
    """Pass-through initializer that returns a pre-computed numpy array/TF tensor.

    Useful for loading weights from NumPy before building a model in eager
    mode without registering a custom initializer separately.
    """
    def __init__(self, input_arr):
        self.input_arr = input_arr

    def __call__(self):
        return self.input_arr

@tf.keras.utils.register_keras_serializable()
class Residual(kl.Layer):
    def __init__(self,
                 layer: kl.Layer,
                 name: str = 'residual',
                 **kwargs):
        """Wrap *layer* with a residual skip connection (X + f(X))."""
        super().__init__(**kwargs,name=name)
        self._layer=layer

    def get_config(self):
        config = {
            "layer": self._layer
        }
        base_config = super().get_config()
        return {**base_config, **config}
    @classmethod
    def from_config(cls, config):
        return cls(**config)
    def call(self, inputs, training=None,**kwargs):
        return inputs + self._layer(inputs, training=training, **kwargs)


@tf.keras.utils.register_keras_serializable()
class FFN(kl.Layer):
    def __init__(self,
                 num_channels: int,
                 dropout_rate: float,
                   FFN_LN_gamma_init=None,
                   FFN_LN_beta_init=None,
                   FFN_kernel1_init=None,
                   FFN_bias1_init=None,
                   FFN_kernel2_init=None,
                   FFN_bias2_init=None,
                   load_init = True,
                 name: str = 'FFN',
                 **kwargs):
        super().__init__(name=name, **kwargs)
        """FFN/MLP layer for transformer block
        Args:
            num_channels: num output channels
            widening: scaling factor for how many channels to start w/
                      e.g. widening = 2, num_channels = 12 means start w/ 24
            dropout_rate: dropout rate used throughout network
            name: Module name.
        """
        self.ffn_channels = num_channels
        self.ffn_widening = 2
        self.ffn_dropout = dropout_rate
        self.load_init=load_init
        self.FFN_LN_gamma_init=FFN_LN_gamma_init,
        self.FFN_LN_beta_init=FFN_LN_beta_init
        self.FFN_kernel1_init=FFN_kernel1_init
        self.FFN_bias1_init=FFN_bias1_init
        self.FFN_kernel2_init=FFN_kernel2_init
        self.FFN_bias2_init=FFN_bias2_init

        self.FFN_layer_norm = kl.LayerNormalization(axis=-1,
                                                  scale=True,
                                                  center=True,
                                                    epsilon=1e-05,
                                                  beta_initializer=FFN_LN_beta_init if self.load_init else "zeros",
                                                  gamma_initializer=FFN_LN_gamma_init if self.load_init else "ones")
        self.FFN_dense_wide = kl.Dense(self.ffn_channels*self.ffn_widening,
                                       activation='linear',
                                       kernel_initializer=FFN_kernel1_init if self.load_init else 'lecun_normal',
                                       bias_initializer=FFN_bias1_init if self.load_init else 'lecun_normal',
                                       use_bias=True)
        self.dropout = kl.Dropout(rate=self.ffn_dropout,**kwargs) # default 0.20
        self.relu = kl.ReLU()
        self.FFN_dense_narrow = kl.Dense(self.ffn_channels,
                                         activation='linear',
                                         kernel_initializer=FFN_kernel2_init if self.load_init else 'lecun_normal',
                                         bias_initializer=FFN_bias2_init if self.load_init else 'lecun_normal',
                                         use_bias=True)

    def get_config(self):
        config = {
            "ffn_channels":self.ffn_channels,
            "ffn_widening":self.ffn_widening,
            "ffn_dropout":self.ffn_dropout,
            "load_init": self.load_init,
            "FFN_LN_gamma_init":self.FFN_LN_gamma_init,
            "FFN_LN_beta_init":self.FFN_LN_beta_init,
            "FFN_kernel1_init":self.FFN_kernel1_init,
            "FFN_bias1_init":self.FFN_bias1_init,
            "FFN_kernel2_init":self.FFN_kernel2_init,
            "FFN_bias2_init":self.FFN_bias2_init
        }
        base_config = super().get_config()
        return {**base_config,**config}
    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def call(self, inputs, training=None):
        x = self.FFN_layer_norm(inputs)
        x = self.FFN_dense_wide(x)
        x = self.dropout(x,training=training)
        x = self.relu(x)
        x = self.FFN_dense_narrow(x)
        x = self.dropout(x,training=training)
        return x

@tf.keras.utils.register_keras_serializable()
class Performer(kl.Layer):
    def __init__(self,
                 d_model,
                 normalize,
                 hidden_size: int,
                 num_heads: int,
                 seed: int,
                 dropout_rate: float,
                 numerical_stabilizer: float,
                 max_seq_length: int,
                 kernel_transformation: str = 'relu_kernel_transformation',
                 use_rot_emb: bool = True,
                 LN_gamma_init = None,
                 LN_beta_init= None,
                 q_init=None,
                 k_init=None,
                 v_init=None,
                 att_output=None,
                 FFN_LN_gamma_init=None,
                 FFN_LN_beta_init=None,
                 FFN_kernel1_init=None,
                 FFN_bias1_init=None,
                 FFN_kernel2_init=None,
                 FFN_bias2_init=None,
                 load_init: bool = False,
                 name = 'transformer_layer',
                 **kwargs):
        super().__init__(name=name, **kwargs)
        """
        Transformer block w/ performer attention
        Args:
            hidden size: ~channel dimension for transformer input
            num_heads: num attention heads
            numerical_stabilizer: small float for stability
            widening: scaling factor for how many channels to start w/
                      e.g. widening = 2, num_channels = 12 means start w/ 24
            dropout_rate: transformer MLP dropout rate
            kernel_transformation: softmax or relu kernel transform for fast att.
            positional_encoding_type: absolute sinusoidal or relative(rotary)
            name: Module name.
        """
        self.hidden_size=hidden_size
        self.num_heads=num_heads
        self.dropout_rate=dropout_rate
        self.kernel_transformation=kernel_transformation
        self.numerical_stabilizer=numerical_stabilizer
        self.max_seq_length = max_seq_length
        self.use_rot_emb=use_rot_emb
        self.d_model=d_model
        self.normalize=normalize
        self.seed=seed
        self.load_init=load_init
        self.FFN_LN_gamma_init=None,
        self.FFN_LN_beta_init=FFN_LN_beta_init
        self.FFN_kernel1_init=FFN_kernel1_init
        self.FFN_bias1_init=FFN_bias1_init
        self.FFN_kernel2_init=FFN_kernel2_init
        self.FFN_bias2_init=FFN_bias2_init

        self.layer_norm = kl.LayerNormalization(axis=-1,
                                                  scale=True,
                                                  center=True,
                                                    epsilon=1.0e-05,
                                                  beta_initializer="zeros",
                                                  gamma_initializer="ones")



        self.self_attention = snnk_attention.Attention(hidden_size=self.d_model,
                                               num_heads=self.num_heads,
                                               use_rot_emb=self.use_rot_emb,
                                               normalize=self.normalize,
                                               kernel_transformation=self.kernel_transformation,
                                               numerical_stabilizer=self.numerical_stabilizer,
                                               nb_random_features=256,
                                               seed=self.seed,
                                               q_init=q_init,
                                               k_init=k_init,
                                               v_init=v_init,
                                               att_output=att_output,
                                               load_init = self.load_init,
                                               **kwargs)
        self.dropout = kl.Dropout(rate=self.dropout_rate,**kwargs)
        self.FFN = FFN(num_channels=self.hidden_size,
                       dropout_rate=self.dropout_rate,
                       FFN_LN_gamma_init=FFN_LN_gamma_init,
                       FFN_LN_beta_init=FFN_LN_beta_init,
                       FFN_kernel1_init=FFN_kernel1_init,
                       FFN_bias1_init=FFN_bias1_init,
                       FFN_kernel2_init=FFN_kernel2_init,
                       FFN_bias2_init=FFN_bias2_init,
                       load_init = self.load_init,
                       name='FFN',
                       **kwargs)

    def get_config(self):
        config = {
            "hidden_size":self.hidden_size,
            "num_heads":self.num_heads,
            "numerical_stabilizer":self.numerical_stabilizer,
            "kernel_transformation":self.kernel_transformation,
            "max_seq_length":self.max_seq_length,
            "use_rot_emb":self.use_rot_emb,
            "d_model":self.d_model,
            "normalize":self.normalize,
            "seed":self.seed,
            "load_init": self.load_init,
            "FFN_LN_gamma_init":self.FFN_LN_gamma_init,
            "FFN_LN_beta_init":self.FFN_LN_beta_init,
            "FFN_kernel1_init":self.FFN_kernel1_init,
            "FFN_bias1_init":self.FFN_bias1_init,
            "FFN_kernel2_init":self.FFN_kernel2_init,
            "FFN_bias2_init":self.FFN_bias2_init
        }
        base_config = super().get_config()
        return{**base_config, **config}

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def call(self, inputs, rpe=None, training=None, **kwargs):
        x = self.layer_norm(inputs)
        x, k_prime, q_prime = self.self_attention(x,
                                                  x,
                                                  rpe,
                                                  training=training,
                                                  **kwargs)

        x = self.dropout(x, training=training) ## 0.40

        mha_output = x + inputs
        ## ffn
        FFN_out = self.FFN(mha_output,training=training,**kwargs)
        #return self.layer_norm(FFN_out + mha_output), k_prime, q_prime
        return (FFN_out + mha_output), k_prime, q_prime

@tf.keras.utils.register_keras_serializable()
class Performer_Encoder(kl.Layer):
    def __init__(self,
                 num_layers,
                 num_heads,
                 dim,
                 d_model,
                 max_seq_length,
                 hidden_size,
                 numerical_stabilizer,
                 dropout_rate = 0.40,
                 use_rot_emb=True,
                 normalize=True,
                 norm=True,
                 seed=42,
                 load_init=True,
                 inits=None,
                 kernel_transformation: str = 'relu_kernel_transformation',
                 name = 'performer_stack',
                 **kwargs):

        super().__init__(name=name, **kwargs)
        """Performer Encoder block
        Args:
            hidden size: ~channel dimension for transformer input
            num_heads: num attention heads
            attention_dropout: post attention layer dropout rate
            numerical_stabilizer: small float for stability
            widening: scaling factor for how many channels to start w/
                      e.g. widening = 2, num_channels = 12 means start w/ 24
            dropout_rate: transformer MLP dropout rate
            dropout_rate: dropout rate used throughout network
            kernel_transformation: softmax or relu kernel transform for fast att.
            positional_encoding_type: absolute sinusoidal or relative(rotary)
            name: Module name.
        """
        self.num_layers=num_layers
        self.num_heads=num_heads
        self.dim=dim
        self.hidden_size=hidden_size
        self.d_model=d_model
        self.max_seq_length=max_seq_length
        self.numerical_stabilizer=numerical_stabilizer
        self.use_rot_emb=use_rot_emb
        self.normalize=normalize
        self.norm=norm
        self.kernel_transformation=kernel_transformation
        self.seed=seed
        self.dropout_rate=dropout_rate
        self.load_init=load_init
        self.inits=inits

        self.layers = [Performer(d_model=self.d_model, #
                                 normalize=self.normalize, #
                                 hidden_size=self.hidden_size, #
                                 num_heads=self.num_heads, # 8
                                 dropout_rate=self.dropout_rate, #
                                 numerical_stabilizer=self.numerical_stabilizer, # 1.0e-03
                                 max_seq_length=self.max_seq_length, # 8192
                                 kernel_transformation=self.kernel_transformation, # relu
                                 seed=self.seed, # use whatever
                                 use_rot_emb=self.use_rot_emb, # True
                                 load_init=self.load_init,
                                 LN_gamma_init = inits["LN_g" + str(i)] if self.load_init else None,
                                 LN_beta_init=  inits["LN_b" + str(i)] if self.load_init else None,
                                 q_init= inits["SA_q" + str(i)] if self.load_init else None,
                                 k_init= inits["SA_k" + str(i)] if self.load_init else None,
                                 v_init= inits["SA_v" + str(i)] if self.load_init else None,
                                 att_output= inits["SA_O" + str(i)] if self.load_init else None,
                                 FFN_LN_gamma_init= inits["FFN_LN_g" + str(i)] if self.load_init else None,
                                 FFN_LN_beta_init= inits["FFN_LN_b" + str(i)] if self.load_init else None,
                                 FFN_kernel1_init= inits["FFN_wide_k" + str(i)] if self.load_init else None,
                                 FFN_bias1_init= inits["FFN_wide_b" + str(i)] if self.load_init else None,
                                 FFN_kernel2_init= inits["FFN_narr_k" + str(i)] if self.load_init else None,
                                 FFN_bias2_init= inits["FFN_narr_b" + str(i)] if self.load_init else None,
                                 **kwargs) for i in range(self.num_layers)]


        self.layer_norm = kl.LayerNormalization(axis=-1,
                                                  scale=True,
                                                  center=True,
                                                    epsilon=1.0e-05,
                                                  beta_initializer=self.inits["performer_encoder_LN_b"] if self.load_init else "zeros",
                                                  gamma_initializer=self.inits["performer_encoder_LN_g"] if self.load_init else "ones")

    def build(self, input_shape):
        N = input_shape[0]
        L = input_shape[1]

        #if self.use_rot_emb:
        self.pos_emb = FixedPositionalEmbedding(self.d_model, self.max_seq_length)
        self.layer_pos_emb = FixedPositionalEmbedding(self.dim, self.max_seq_length)

        super(Performer_Encoder,self).build(input_shape)

    def get_config(self):
        config = {
            "hidden_size":self.hidden_size,
            "num_heads":self.num_heads,
            "numerical_stabilizer":self.numerical_stabilizer,
            "kernel_transformation":self.kernel_transformation,
            "num_layers":self.num_layers,
            "dim":self.dim,
            "d_model":self.d_model,
            "max_seq_length":self.max_seq_length,
            "use_rot_emb":self.use_rot_emb,
            "normalize":self.normalize,
            "norm":self.norm,
            "seed":self.seed
        }

        base_config = super().get_config()
        return{**base_config, **config}

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def call(self, x, training=None, **kwargs):
        att_matrices={}
        for idx,layer in enumerate(self.layers):
            #x += self.pos_emb(x) # c/w with lucid rains implementation
            rpe = self.layer_pos_emb(x) ### check whether fixedpositionalembedding is c/w 
                                        ### apply_rotary_embedding + fixedposembedding in flaxformer
            x,k_prime,q_prime = layer(x, rpe=rpe, training=training)
            att_matrices['layer_' + str(idx)] = (k_prime,q_prime)
            ## relu, 256 hidden dimensions

        if self.norm:
            x = self.layer_norm(x)
        return x,att_matrices

@tf.keras.utils.register_keras_serializable()
class FixedPositionalEmbedding(tf.keras.layers.Layer):
    def __init__(self, dim, max_seq_len):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len

    def build(self, input_shape):
        self.inv_freq = 1. / (10000 ** (tf.range(start=0, limit=self.dim, delta=2, dtype='float32') / self.dim))
        self.position = tf.range(start=0, limit=self.max_seq_len, delta=1, dtype='float32')
        self.sinusoid_inp = tf.einsum("i,j->ij", self.position, self.inv_freq)
        self.emb = tf.concat((tf.math.sin(self.sinusoid_inp),
                              tf.math.cos(self.sinusoid_inp)), axis=-1)
    def call(self, x):
        return tf.cast(self.emb[None, :x.shape[1], :],
                       dtype=tf.bfloat16)

@tf.keras.utils.register_keras_serializable()
class TargetLengthCrop1D(kl.Layer):
    """Crop sequence to match the desired target length."""
    def __init__(self,
               uncropped_length: int = 4096,
               target_length: int = 4092,
               name: str = 'target_length_crop'):
        super().__init__(name=name)
        self._target_length = target_length
        self._uncropped_length = uncropped_length

    def call(self, inputs):
        if self._target_length is None:
            return inputs
        trim = (self._uncropped_length - self._target_length) // 2
        if trim < 0:
            raise ValueError('inputs longer than target length')
        elif trim == 0:
            return inputs
        else:
            return inputs[..., trim:-trim, :]

