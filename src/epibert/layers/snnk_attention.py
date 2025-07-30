# pylint: skip-file

import math
import numpy as np
import tensorflow as tf

# from spe_tf import *
from einops import rearrange, repeat
from functools import partial
#from util import *
import epibert.layers.util as util
BIG_CONSTANT = 1e8


def create_projection_matrix(m, d, seed=0, scaling=0, struct_mode=False):
    r"""Constructs the matrix of random projections.
  Constructs a matrix of random orthogonal projections. Each projection vector
  has direction chosen uniformly at random and either deterministic length
  \sqrt{d} or length taken from the \chi(d) distribution (in the latter case
  marginal distributions of the projections are d-dimensional Gaussian vectors
  with associated identity covariance matrix).
  Args:
    m: number of random projections.
    d: dimensionality of each random projection.
    seed: random seed used to construct projections.
    scaling: 1 if all the random projections need to be renormalized to have
      length \sqrt{d}, 0 if the lengths of random projections should follow
      \chi(d) distribution.
    struct_mode: if True then products of Givens rotations will be used to
      construct random orthogonal matrix. This bypasses Gram-Schmidt
      orthogonalization.
  Returns:
    The matrix of random projections of the shape [m, d].
    """
    nb_full_blocks = int(m / d)
    block_list = []
    current_seed = seed
    for _ in range(nb_full_blocks):
        if struct_mode:
            q = create_products_of_givens_rotations(d, seed)
        else:
            unstructured_block = tf.random.stateless_normal(shape=(d, d), seed=[current_seed,2], dtype=tf.float32)
            q, _ = tf.linalg.qr(unstructured_block)
            q = tf.transpose(q)
        block_list.append(q)
        current_seed += 1
    remaining_rows = m - nb_full_blocks * d
    if remaining_rows > 0:
        if struct_mode:
            q = create_products_of_givens_rotations(d, seed)
        else:
            unstructured_block = tf.random.stateless_normal(shape=(d, d), seed=[current_seed,5], dtype=tf.float32)
            q, _ = tf.linalg.qr(unstructured_block)
            q = tf.transpose(q)
        block_list.append(q[0:remaining_rows])
    final_matrix = tf.cast(tf.experimental.numpy.vstack(block_list),dtype=tf.float32)
    current_seed += 1

    if scaling == 0:
        multiplier = tf.norm(tf.random.stateless_normal(shape=(m, d), seed=[current_seed+3,6], dtype=tf.float32),
                             axis=1)
    elif scaling == 1:
        multiplier = tf.math.sqrt(tf.cast(d,dtype=tf.float32)) * tf.ones((m))
    else:
        raise ValueError("Scaling must be one of {0, 1}. Was %s" % scaling)

    return tf.linalg.matmul(tf.linalg.diag(multiplier), final_matrix)


def create_products_of_givens_rotations(dim, seed):
    r"""Constructs a 2D-tensor which is a product of Givens random rotations.
  Constructs a 2D-tensor of the form G_1 * ... * G_k, where G_i is a Givens
  random rotation. The resulting tensor mimics a matrix taken uniformly at
  random form the orthogonal group.
  Args:
    dim: number of rows/columns of the resulting 2D-tensor.
    seed: random seed.
  Returns:
    The product of Givens random rotations.
  """
    nb_givens_rotations = dim * int(math.ceil(math.log(float(dim))))
    q = np.eye(dim, dim)
    np.random.seed(seed)
    for _ in range(nb_givens_rotations):
        random_angle = math.pi * np.random.uniform()
        random_indices = np.random.choice(dim, 2)
        index_i = min(random_indices[0], random_indices[1])
        index_j = max(random_indices[0], random_indices[1])
        slice_i = q[index_i]
        slice_j = q[index_j]
        new_slice_i = math.cos(random_angle) * slice_i + math.sin(
            random_angle) * slice_j
        new_slice_j = -math.sin(random_angle) * slice_i + math.cos(
            random_angle) * slice_j
        q[index_i] = new_slice_i
        q[index_j] = new_slice_j
    return tf.cast(tf.constant(q), dtype=tf.float32)


def relu_kernel_transformation(data,
                               is_query,
                               projection_matrix=None,
                               numerical_stabilizer=0.001):
    """Computes features for the ReLU-kernel.
  Computes random features for the ReLU kernel from
  https://arxiv.org/pdf/2009.14794.pdf.
  Args:
    data: input data tensor of the shape [B, L, H, D], where: B - batch
      dimension, L - attention dimensions, H - heads, D - features.
    is_query: indicates whether input data is a query oor key tensor.
    projection_matrix: random Gaussian matrix of shape [M, D], where M stands
      for the number of random features and each D x D sub-block has pairwise
      orthogonal rows.
    numerical_stabilizer: small positive constant for numerical stability.
  Returns:
    Corresponding kernel feature map.
  """
    #del is_query
    #if projection_matrix is None:
    #  return tf.nn.relu(data) + numerical_stabilizer
    #else:
    ratio = 1.0 / tf.math.sqrt(
    tf.dtypes.cast(projection_matrix.shape[0], data.dtype))
    projection_matrix = tf.cast(projection_matrix, dtype=data.dtype)
    data_dash = ratio * tf.einsum("blhd,md->blhm", data, projection_matrix)
    return tf.nn.relu(data_dash) + numerical_stabilizer

def relu_kernel_transformation_q(data,
                               is_query,
                               projection_matrix=None,
                               numerical_stabilizer=0.001):
    """Computes features for the ReLU-kernel.
  Computes random features for the ReLU kernel from
  https://arxiv.org/pdf/2009.14794.pdf.
  Args:
    data: input data tensor of the shape [B, L, H, D], where: B - batch
      dimension, L - attention dimensions, H - heads, D - features.
    is_query: indicates whether input data is a query oor key tensor.
    projection_matrix: random Gaussian matrix of shape [M, D], where M stands
      for the number of random features and each D x D sub-block has pairwise
      orthogonal rows.
    numerical_stabilizer: small positive constant for numerical stability.
  Returns:
    Corresponding kernel feature map.
  """
    del is_query
    #if projection_matrix is None:
    #    return tf.nn.relu(data) + numerical_stabilizer
    #else:
    ratio = 1.0 / tf.math.sqrt(
    tf.dtypes.cast(projection_matrix.shape[0], tf.float32))
    data_dash = ratio * tf.einsum("blhd,md->blhm", data, projection_matrix)
    return tf.math.pow(tf.nn.relu(data_dash),4) + numerical_stabilizer


def softmax_kernel_transformation(data,
                                  is_query,
                                  projection_matrix,
                                  numerical_stabilizer=0.000001):
    """Computes random features for the softmax kernel using FAVOR+ mechanism.
  Computes random features for the softmax kernel using FAVOR+ mechanism from
  https://arxiv.org/pdf/2009.14794.pdf.
  Args:
    data: input data tensor of the shape [B, L, H, D], where: B - batch
      dimension, L - attention dimensions, H - heads, D - features.
    is_query: indicates whether input data is a query oor key tensor.
    projection_matrix: random Gaussian matrix of shape [M, D], where M stands
      for the number of random features and each D x D sub-block has pairwise
      orthogonal rows.
    numerical_stabilizer: small positive constant for numerical stability.
  Returns:
    Corresponding kernel feature map.
    """
  #changed the projection_matrix to not none
    data_normalizer = 1.0 / (
      tf.math.sqrt(tf.math.sqrt(tf.dtypes.cast(data.shape[-1], tf.float32))))
    data = data_normalizer * data
    ratio = 1.0 / tf.math.sqrt(
        tf.dtypes.cast(projection_matrix.shape[0], tf.float32))
    data_dash = tf.einsum("blhd,md->blhm", data, projection_matrix)
    diag_data = tf.math.square(data)
    diag_data = tf.math.reduce_sum(
        diag_data, axis=tf.keras.backend.ndim(data) - 1)
    diag_data = diag_data / 2.0
    diag_data = tf.expand_dims(diag_data, axis=tf.keras.backend.ndim(data) - 1)
    last_dims_t = (len(data_dash.shape) - 1,)
    attention_dims_t = (len(data_dash.shape) - 3,)
    if is_query:
        data_dash = ratio * (
        tf.math.exp(data_dash - diag_data - tf.math.reduce_max(
            data_dash, axis=last_dims_t, keepdims=True)) + numerical_stabilizer)
    else:
        data_dash = ratio * (
        tf.math.exp(data_dash - diag_data - tf.math.reduce_max(
            data_dash, axis=last_dims_t + attention_dims_t, keepdims=True)) +
        numerical_stabilizer)

    return data_dash


def noncausal_numerator(qs, ks, vs):
    """Computes not-normalized FAVOR noncausal attention AV.
  Args:
    qs: query_prime tensor of the shape [L,B,H,M].
    ks: key_prime tensor of the shape [L,B,H,M].
    vs: value tensor of the shape [L,B,H,D].
  Returns:
    Not-normalized FAVOR noncausal attention AV.
    """
    kvs = tf.einsum("lbhm,lbhd->bhmd", ks, vs)
    return tf.einsum("lbhm,bhmd->lbhd", qs, kvs)


def noncausal_denominator(qs, ks):
    """Computes FAVOR normalizer in noncausal attention.
  Args:
    qs: query_prime tensor of the shape [L,B,H,M].
    ks: key_prime tensor of the shape [L,B,H,M].
  Returns:
    FAVOR normalizer in noncausal attention.
    """
    all_ones = tf.ones([ks.shape[0]], dtype=tf.bfloat16)
    ks_sum = tf.einsum("lbhm,l->bhm", ks, all_ones)
    return tf.einsum("lbhm,bhm->lbh", qs, ks_sum)


@tf.custom_gradient
def causal_numerator(qs, ks, vs):
    """Computes not-normalized FAVOR causal attention A_{masked}V.
  Args:
    qs: query_prime tensor of the shape [L,B,H,M].
    ks: key_prime tensor of the shape [L,B,H,M].
    vs: value tensor of the shape [L,B,H,D].
  Returns:
    Not-normalized FAVOR causal attention A_{masked}V.
    """

    result = []
    sums = tf.zeros_like(tf.einsum("ijk,ijl->ijkl", ks[0], vs[0]))

    for index in range(qs.shape[0]):
        sums = sums + tf.einsum("ijk,ijl->ijkl", ks[index], vs[index])
        result.append(tf.einsum("ijkl,ijk->ijl", sums, qs[index])[None, Ellipsis])

    result = tf.concat(result, axis=0)

    def grad(res_grad):

        grads = tf.zeros_like(tf.einsum("ijk,ijl->ijkl", ks[0], vs[0]))

        gr_sums = sums

        q_grads = []
        k_grads = []
        v_grads = []

        for index in range(qs.shape[0] - 1, -1, -1):

            q_grads.append(
              tf.einsum("ijkl,ijl->ijk", gr_sums, res_grad[index])[None, Ellipsis])
            grads = grads + tf.einsum("ijk,ijl->ijkl", qs[index], res_grad[index])
            k_grads.append(tf.einsum("ijkl,ijl->ijk", grads, vs[index])[None, Ellipsis])
            v_grads.append(tf.einsum("ijkl,ijk->ijl", grads, ks[index])[None, Ellipsis])
            gr_sums = gr_sums - tf.einsum("ijk,ijl->ijkl", ks[index], vs[index])

        q_grads = tf.concat(q_grads[::-1], axis=0)
        k_grads = tf.concat(k_grads[::-1], axis=0)
        v_grads = tf.concat(v_grads[::-1], axis=0)

        return q_grads, k_grads, v_grads

    return result, grad


@tf.custom_gradient
def causal_denominator(qs, ks):
    """Computes FAVOR normalizer in causal attention.
  Args:
    qs: query_prime tensor of the shape [L,B,H,M].
    ks: key_prime tensor of the shape [L,B,H,M].
  Returns:
    FAVOR normalizer in causal attention.
    """

    result = []
    sums = tf.zeros_like(ks[0])

    for index in range(qs.shape[0]):
        sums = sums + ks[index]
        result.append(tf.reduce_sum(qs[index] * sums, axis=2)[None, Ellipsis])

    result = tf.concat(result, axis=0)

    def grad(res_grad):

        k_grad = tf.zeros_like(ks[0])

        gr_sums = sums

        q_grads = []
        k_grads = []

        for index in range(qs.shape[0] - 1, -1, -1):

            q_grads.append(
              tf.einsum("ijk,ij->ijk", gr_sums, res_grad[index])[None, Ellipsis])
            k_grad = k_grad + tf.einsum("ijk,ij->ijk", qs[index], res_grad[index])
            k_grads.append(k_grad[None, Ellipsis])
            gr_sums = gr_sums - ks[index]

        q_grads = tf.concat(q_grads[::-1], axis=0)
        k_grads = tf.concat(k_grads[::-1], axis=0)

        return q_grads, k_grads

    return result, grad


def favor_attention(query,
                    key,
                    value,
                    kernel_transformation,
                    causal,
                    projection_matrix):
    """Computes FAVOR normalized attention.
  Args:
    query: query tensor.
    key: key tensor.
    value: value tensor.
    kernel_transformation: transformation used to get finite kernel features.
    causal: whether attention is causal or not.
    projection_matrix: projection matrix to be used.
  Returns:
    FAVOR normalized attention.
    """
    query_prime = kernel_transformation(query, True,
                                        projection_matrix)  # [B,L,H,M]
    #print("qprime", query_prime)
    key_prime = kernel_transformation(key, False, projection_matrix)  # [B,L,H,M]
    #print("kprime", key_prime)
    query_prime = tf.transpose(query_prime, [1, 0, 2, 3])  # [L,B,H,M]
    key_prime = tf.transpose(key_prime, [1, 0, 2, 3])  # [L,B,H,M]
    value = tf.transpose(value, [1, 0, 2, 3])  # [L,B,H,D]

    if causal:
        av_attention = causal_numerator(query_prime, key_prime, value)
        attention_normalizer = causal_denominator(query_prime, key_prime)
    else:
        av_attention = noncausal_numerator(query_prime, key_prime, value)
        attention_normalizer = noncausal_denominator(query_prime, key_prime)

  # TODO(kchoro): Add more comments.
    av_attention = tf.transpose(av_attention, [1, 0, 2, 3])
    #print("avattn", av_attention.shape)
    attention_normalizer = tf.transpose(attention_normalizer, [1, 0, 2])

    attention_normalizer = tf.expand_dims(attention_normalizer,
                                        len(attention_normalizer.shape))
    return av_attention / attention_normalizer, key_prime, query_prime


@tf.keras.utils.register_keras_serializable()
class Attention(tf.keras.layers.Layer):
    """Performer-style scaled dot-product attention layer with random feature
    approximations.

    This implementation supports the FAVOR+ mechanism (Fast Attention via
    positive Orthogonal Random features) proposed in the Performer paper and
    retains all public attributes so previously-saved checkpoints remain
    loadable.  Only internal documentation has been improved and redundant
    comments removed—no behavioural changes were introduced.
    """

    def __init__(self,
                 hidden_size,
                 num_heads,
                 kernel_transformation=relu_kernel_transformation,
                 numerical_stabilizer=0.001,
                   causal=False,
                   nb_random_features=256,
                   use_rot_emb = True,
                   eps = 1e-6,
                   normalize = True,
                   seed=42,
                 q_init=None,
                 k_init=None,
                 v_init=None,
                 att_output=None,
                 load_init : bool = False
                   ):

#     """Initialize Attention.

#     Args:
#         hidden_size: int, output dim of hidden layer.
#         num_heads: int, number of heads to repeat the same attention structure.
#         attention_dropout: float, dropout rate inside attention for training.
#         kernel_transformation: transformation used to produce kernel features for
#             attention.
#         numerical_stabilizer: used to bound away from zero kernel values.
#         causal: whether attention is causal or not.
#         projection_matrix_type: None if Identity should be used, otherwise random
#             projection matrix will be applied.
#         nb_random_features: number of random features to be used (relevant only if
#             projection_matrix is not None).

#     """


        if hidden_size % num_heads:
            raise ValueError(
                "Hidden size ({}) must be divisible by the number of heads ({})."
                .format(hidden_size, num_heads))

        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.kernel_transformation = kernel_transformation
        self.numerical_stabilizer = numerical_stabilizer
        self.causal = causal
        self.nb_random_features = nb_random_features
        self.use_rot_emb = use_rot_emb
        self.eps = eps
        self.normalize = normalize
        self.seed = seed
        self.load_init=load_init
        self.q_init=q_init
        self.k_init=k_init
        self.v_init=v_init
        self.att_output=att_output

    def build(self, input_shape):
        """Builds the layer."""
    # Layers for linearly projecting the queries, keys, and values.
        size_per_head = self.hidden_size // self.num_heads

        def _glorot_initializer(fan_in, fan_out):
            limit = math.sqrt(6.0 / (fan_in + fan_out))
            return tf.keras.initializers.RandomUniform(minval=-limit, maxval=limit,seed=5)

        attention_initializer = _glorot_initializer(input_shape.as_list()[-1],
                                                    self.hidden_size)
        self.query_dense_layer = util.DenseEinsum(
            output_shape=(self.num_heads, size_per_head),
            kernel_initializer=self.q_init if self.load_init else attention_initializer,
            use_bias=False,
            name="query")
        self.key_dense_layer = util.DenseEinsum(
            output_shape=(self.num_heads, size_per_head),
            kernel_initializer=self.k_init if self.load_init else attention_initializer,
            use_bias=False,
            name="key")
        self.value_dense_layer = util.DenseEinsum(
            output_shape=(self.num_heads, size_per_head),
            kernel_initializer=self.v_init if self.load_init else attention_initializer,
            use_bias=False,
            name="value")

        output_initializer = _glorot_initializer(self.hidden_size, self.hidden_size)
        self.output_dense_layer = util.DenseEinsum(
            output_shape=self.hidden_size,
            num_summed_dimensions=2,
            kernel_initializer=self.att_output if self.load_init else output_initializer,
            use_bias=False,
            name="output_transform")

        seed=tf.cast(self.seed,tf.int32)
        self.projection_matrix = create_projection_matrix(
            self.nb_random_features, size_per_head, seed)

        super(Attention, self).build(input_shape)


    def get_config(self):
        config = {
            "hidden_size": self.hidden_size,
            "num_heads": self.num_heads,
            "attention_dropout": self.attention_dropout,
            "numerical_stabilizer":self.numerical_stabilizer,
            "nb_random_features":self.nb_random_features,
            "use_rot_emb":self.use_rot_emb,
            "causal":self.causal,
            'kernel_transformation':self.kernel_transformation,
            "eps":self.eps,
            "normalize":self.normalize,
            "seed":self.seed,
            "load_init":self.load_init
        }
    @classmethod
    def from_config(cls, config):
        return cls(**config)


    def call(self,
           query_input,
           source_input,
           rpe,
           training):
        """Apply attention mechanism to query_input and source_input.
    Args:
      query_input: A tensor with shape [batch_size, length_query, hidden_size].
      source_input: A tensor with shape [batch_size, length_source,
        hidden_size].
      bias: A tensor with shape [batch_size, 1, length_query, length_source],
        the attention bias that will be added to the result of the dot product.
      training: A bool, whether in training mode or not.
      cache: (Used during prediction) A dictionary with tensors containing
        results of previous attentions. The dictionary must have the items:
            {"k": tensor with shape [batch_size, i, heads, dim_per_head],
             "v": tensor with shape [batch_size, i, heads, dim_per_head]} where
               i is the current decoded length for non-padded decode, or max
               sequence length for padded decode.
      decode_loop_step: An integer, step number of the decoding loop. Used only
        for autoregressive inference on TPU.
    Returns:
      Attention layer output with shape [batch_size, length_query, hidden_size]
        """
    # Linearly project the query, key and value using different learned
    # projections. Splitting heads is automatically done during the linear
    # projections --> [batch_size, length, num_heads, dim_per_head].
        b, n, _ = query_input.shape
        h = self.num_heads

        q = self.query_dense_layer(query_input)
        k = self.key_dense_layer(source_input)
        v = self.value_dense_layer(source_input)

        if self.kernel_transformation == 'relu_kernel_transformation':
            kernel_transform = relu_kernel_transformation
            projection_matrix=None

        elif self.kernel_transformation == 'relu_kernel_transformation_q':
            kernel_transform = relu_kernel_transformation_q
            #projection_matrix=None

        else:
            kernel_transform = softmax_kernel_transformation
            #projection_matrix=self.projection_matrix

        dim = q.shape[-1]
        tgt_len = k.shape[1]

        if self.use_rot_emb:
            q,k = apply_rotary_pos_emb(q,k,rpe)
            attention_output, k_prime, q_prime = favor_attention(q, k, v,
                                       kernel_transform, self.causal,
                                       self.projection_matrix)
        else:
            attention_output, k_prime, q_prime = favor_attention(q, k, v,
                                       kernel_transform, self.causal,
                                       self.projection_matrix)

        attention_output = self.output_dense_layer(attention_output)
        return attention_output, k_prime, q_prime


@tf.keras.utils.register_keras_serializable()
class SelfAttention(Attention):
    """Multiheaded self-attention layer."""

    def call(self,
           query_input,
           rpe=None,
           training=True,
           cache=None,
           decode_loop_step=None):
        return super(SelfAttention, self).call(query_input, query_input, rpe,
                                               training, cache, decode_loop_step)
    def get_config(self):
        base_config = super(SelfAttention, self).get_config()
        return {**base_config}


def rotate_every_two(x):
    x = rearrange(x, '... (d j) -> ... d j', j = 2)
    x1, x2 = tf.unstack(x, axis = -1)
    x = tf.stack([-x2, x1], axis = -1)
    return rearrange(x, '... d j -> ... (d j)')

def apply_rotary_pos_emb(q, k, sinu_pos):

    sinu_pos = rearrange(sinu_pos, '() n (j d) -> n j d', j = 2)
    sin, cos = tf.unstack(sinu_pos, axis = -2)
    sin, cos = map(lambda t: repeat(t, 'b n -> b (n j)', j = 2), (sin, cos))

    sin = tf.repeat(tf.expand_dims(sin,axis=1),tf.shape(q)[2],axis=1)
    sin = tf.repeat(tf.expand_dims(sin,axis=0),tf.shape(q)[0],axis=0) # b, l, d
    cos = tf.repeat(tf.expand_dims(cos,axis=1),tf.shape(q)[2],axis=1)
    cos = tf.repeat(tf.expand_dims(cos,axis=0),tf.shape(q)[0],axis=0) # b, l, d
    q, k = map(lambda t: (t * cos) + (rotate_every_two(t) * sin), (q, k))
    return q, k

"""
def rotate_half(x):
    x = rearrange(x, r = 2)
    x1, x2 = tf.unstack(x, axis=-1)
    x = tf.stack((-x2, x1), axis=-1)
    return irearrange(x)


def apply_rotary_pos_emb(q,k,sinu_pos):
    sinu_pos = rearrange(sinu_pos, '() n (j d) -> n j d', j = 2)
    sin, cos = sinu_pos.unbind(dim = -2)
    sin, cos = map(lambda t: repeat(t, 'b n -> b (n j)', j = 2), (sin, cos))
    q, k = map(lambda t: (t * cos) + (rotate_every_two(t) * sin), (q, k))
    return q, k

def rearrange(x, r=2):
    b = tf.shape(x)
    b1 = b[:-1]
    b2 = b[-1, None]
    b3 = tf.constant([r], dtype=tf.int32)
    b4 = tf.cast(b2/b3, dtype=tf.int32)
    b_ = tf.concat([b1, b4, b3], axis=0)

    return tf.reshape(x, b_)

def irearrange(x):
    c = tf.shape(x)
    c1 = c[:-2]
    c2 = tf.reduce_prod(c[-2:])[None]
    c_ = tf.concat([c1, c2], axis=0)

    return tf.reshape(x, c_)

def repeat(x, r):
    c = tf.ones_like(tf.shape(x), dtype=tf.int32)
    c1 = c[:-1]
    c2 = c[-1][None] * r
    c_ = tf.concat([c1, c2], axis=0)

    return tf.tile(x, c_)
"""
