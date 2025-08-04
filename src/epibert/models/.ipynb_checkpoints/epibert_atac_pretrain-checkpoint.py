from typing import Any, Callable, Dict, Optional, Text, Union, Iterable
import tensorflow.experimental.numpy as tnp
import tensorflow as tf
from tensorflow.keras import layers as kl
from epibert.layers.layers import *
import tensorflow_addons as tfa
from tensorflow.keras import regularizers
from tensorflow.keras.layers.experimental import SyncBatchNormalization as syncbatchnorm

@tf.keras.utils.register_keras_serializable()
class epibert_atac_pretrain(tf.keras.Model):
    def __init__(self,
                 kernel_transformation = 'relu_kernel_transformation',
                 dropout_rate: float = 0.20,
                 pointwise_dropout_rate: float = 0.10,
                 input_length: int = 524288,
                 output_length: int = 4096,
                 final_output_length: int = 4092,
                 num_heads:int = 8,
                 numerical_stabilizer=0.0000001,
                 num_transformer_layers:int = 8,
                 norm=True,
                 max_seq_length:int = 4096,
                 BN_momentum: float = 0.90,
                 use_rot_emb: bool =True,
                 normalize: bool = True,
                 seed: int = 19,
                 filter_list_seq=[512,640,640,768,896,1024],
                 filter_list_atac: list = [32, 64],
                 final_point_scale: int = 6,
                 num_motifs: int = 693,
                 motif_dropout_rate: float = 0.25,
                 motif_units_fc: int = 32,
                 name: str = 'epibert',
                 **kwargs):
        """
        epibert model takes as input sequence, masked ATAC seq, and TF 
        activity and outputs a profile prediction over the masked tokens. 

        Inputs:
            sequence: sequence tensor of shape (batch_size, input_length, 4)
            atac: masked ATAC seq tensor of shape (batch_size, input_length, 1)
            tf_activity: TF activity tensor of shape (batch_size, num_tfs)
        Args: 
             - kernel_transformation: kernel transformation function for Performer attention
             - dropout_rate: dropout rate for Performer layers
             - pointwise_dropout_rate: dropout rate for pointwise convolutions at end
             - input_length: length of input sequence
             - output_length: length of output ATAC tensor before cropping
             - final_output_length: length of final output profile after cropping
             - num_heads: number of heads for Performer attention
             - numerical_stabilizer: numerical stabilizer for Performer attention
             - num_transformer_layers: number of layers for Performer attention
             - norm: whether to use layer normalization in Performer attention
             - max_seq_length: maximum sequence length for Performer attention
             - BN_momentum: momentum for batch normalization
             - use_rot_emb: whether to use rotational embeddings for Performer attention
             - normalize: whether to normalize the output profile
             - seed: random seed to construct projection matrix for Performer attention
             - filter_list_seq: list of filter sizes for sequence convolutions
             - filter_list_atac: list of filter sizes for ATAC convolutions
             - final_point_scale: scale for final pointwise convolution (how to reduce # of channels)
             - num_motifs: number of motif inputs
             - motif_dropout_rate: dropout rate for motif activity fc layer
             - motif_units_fc: number of units for motif fc layer
             - model name for saving

          name: model name
        """

        super(epibert_atac_pretrain, self).__init__(name=name,**kwargs)
        self.kernel_transformation = kernel_transformation
        self.dropout_rate = dropout_rate
        self.pointwise_dropout_rate = pointwise_dropout_rate
        self.num_heads = num_heads
        self.input_length = input_length
        self.numerical_stabilizer = numerical_stabilizer
        self.num_transformer_layers = num_transformer_layers
        self.output_length = output_length
        self.final_output_length = final_output_length
        self.norm = norm
        self.max_seq_length = max_seq_length + 1
        self.use_rot_emb = use_rot_emb
        self.normalize = normalize
        self.seed = seed
        self.filter_list_seq = filter_list_seq
        self.filter_list_atac=filter_list_atac
        self.BN_momentum = BN_momentum
        self.final_point_scale = final_point_scale
        self.num_motifs = num_motifs
        self.motif_units_fc = motif_units_fc
        self.motif_dropout_rate= motif_dropout_rate

        self.hidden_size=self.filter_list_seq[-1]
        self.d_model = self.filter_list_seq[-1]

        self.dim = self.hidden_size  // self.num_heads

        # convolutional stem for sequence input 
        self.stem_conv = tf.keras.layers.Conv1D(
            filters= int(self.filter_list_seq[0]), 
            kernel_size=15,
            kernel_initializer='lecun_normal',
            bias_initializer='zeros',
            strides=1,
            padding='same')
        self.stem_res_conv=Residual(conv_block(int(self.filter_list_seq[0]), 1, 
                                                    BN_momentum=self.BN_momentum,
                                                    name='pointwise_conv_block'))
        self.stem_pool = SoftmaxPooling1D(name='stem_pool')

        # convolutional stem for ATAC profile
        self.stem_conv_atac = tf.keras.layers.Conv1D(
            filters=32,
            kernel_size=50,
            kernel_initializer='lecun_normal',
            bias_initializer='zeros',
            strides=1,
            dilation_rate=1,
            padding='same')
        self.stem_res_conv_atac =Residual(conv_block(32, 1,
                                                    BN_momentum=self.BN_momentum,
                                                    name='pointwise_conv_block_atac'))
        self.stem_pool_atac = tf.keras.layers.MaxPooling1D(pool_size=2)

        # convolutional tower for sequence input
        self.conv_tower = tf.keras.Sequential([
            tf.keras.Sequential([
                conv_block(filters=num_filters,
                               width=5,
                               stride=1,
                               BN_momentum=self.BN_momentum,
                               padding='same'),
                #tf.keras.layers.MaxPooling1D(pool_size=2)],   
                SoftmaxPooling1D(name=f'soft_max_pool_{i}')],
                       name=f'conv_tower_block_{i}')
            for i, num_filters in enumerate(self.filter_list_seq)], name='conv_tower')

        # convolutional tower for ATAC profile
        self.conv_tower_atac = tf.keras.Sequential([
            tf.keras.Sequential([
                conv_block(filters=num_filters,
                               width=5,
                               dilation_rate=1,
                               stride=1,
                               BN_momentum=self.BN_momentum,
                               padding='same'),
                tf.keras.layers.MaxPooling1D(pool_size=4)],
                       name=f'conv_tower_block_atac_{i}')
            for i, num_filters in enumerate(self.filter_list_atac)], name='conv_tower_atac')

        # dropout for TF activity
        self.motif_dropout1=kl.Dropout(rate=self.motif_dropout_rate, **kwargs)
        self.motif_dropout2=kl.Dropout(rate=self.motif_dropout_rate/4, **kwargs)
        # dense layer for motif activity
        self.motif_activity_fc1 = kl.Dense(
            self.motif_units_fc,
            activation='gelu',
            kernel_initializer='lecun_normal',
            bias_initializer='zeros',
            use_bias=True)

        self.motif_activity_fc2 = kl.Dense(
            self.motif_units_fc//4,
            activation=None,
            kernel_initializer='lecun_normal',
            bias_initializer='zeros',
            use_bias=True)
        
        self.pre_transformer_projection = kl.Dense(self.hidden_size,
                                                   activation=None,
                                                    kernel_initializer='lecun_normal',
                                                    use_bias=False)
        
        # Performer attention
        self.performer = Performer_Encoder(
            num_layers=self.num_transformer_layers,
            num_heads=self.num_heads,
            dim = self.dim,
            d_model=self.d_model,
            norm=self.norm,
            max_seq_length=self.max_seq_length,
            hidden_size=self.hidden_size,
            numerical_stabilizer=self.numerical_stabilizer,
            dropout_rate=self.dropout_rate,
            use_rot_emb=self.use_rot_emb,
            kernel_transformation=self.kernel_transformation,
            normalize=self.normalize,
            seed = self.seed,
            load_init=False,
            inits=None,
            name = 'shared_transformer',
            **kwargs)

        # cropping layer
        self.crop_final = TargetLengthCrop1D(uncropped_length=self.output_length,
                                             target_length=self.final_output_length,
                                             name='target_input')

        self.final_pointwise_conv = conv_block(filters=self.filter_list_seq[-1] // self.final_point_scale,
                                               BN_momentum=self.BN_momentum,
                                               **kwargs,
                                               name = 'final_pointwise')

        self.final_dense_profile = kl.Dense(1,
                                            activation='softplus',
                                            kernel_initializer='lecun_normal',
                                            bias_initializer='zeros',
                                            use_bias=True)

        self.dropout = kl.Dropout(rate=self.pointwise_dropout_rate,
                                  **kwargs)
        self.gelu = tf.keras.layers.Activation('gelu')


    def call(self, inputs, training:bool=True):
        """Forward pass.

        Parameters
        ----------
        inputs : Tuple[tf.Tensor, tf.Tensor, tf.Tensor]
            ``(sequence, atac, motif_activity)`` where
            * `sequence`        – one-hot encoded DNA of shape *(B, L, 4)*
            * `atac`            – masked ATAC profile of shape *(B, L, 1)*
            * `motif_activity`  – TF activity of shape *(B, n_motifs)*

        training : bool
            Standard Keras training flag.
        """

        seq_enc, atac_enc, motif_enc = self._encode_inputs(*inputs, training=training)

        x = tf.concat([seq_enc, atac_enc, motif_enc], axis=2)
        x = self.pre_transformer_projection(x)

        # Transformer encoder --------------------------------------------------
        x, _ = self.performer(x, training=training)

        # Point-wise head ------------------------------------------------------
        x = self.final_pointwise_conv(x, training=training)
        x = self.dropout(x, training=training)
        x = self.gelu(x)
        x = self.final_dense_profile(x, training=training)

        return self.crop_final(x)

    # ---------------------------------------------------------------------
    # helper functions
    # ---------------------------------------------------------------------

    def _encode_inputs(self,
                       sequence: tf.Tensor,
                       atac: tf.Tensor,
                       motif_activity: tf.Tensor,
                       training: bool = True):
        """Encode raw inputs into feature maps consumed by the Transformer."""
        # sequence branch
        seq_x = self.stem_conv(sequence, training=training)
        seq_x = self.stem_res_conv(seq_x, training=training)
        seq_x = self.stem_pool(seq_x, training=training)
        seq_x = self.conv_tower(seq_x, training=training)

        # atac branch
        atac_x = self.stem_conv_atac(atac, training=training)
        atac_x = self.stem_res_conv_atac(atac_x, training=training)
        atac_x = self.stem_pool_atac(atac_x, training=training)
        atac_x = self.conv_tower_atac(atac_x, training=training)

        # motif branch
        motif_x = self.motif_activity_fc1(motif_activity)
        motif_x = self.motif_dropout1(motif_x, training=training)
        motif_x = self.motif_activity_fc2(motif_x)
        motif_x = self.motif_dropout2(motif_x, training=training)
        motif_x = tf.tile(motif_x, [1, self.output_length, 1])

        return seq_x, atac_x, motif_x

    def get_config(self):
        config = {
            "kernel_transformation": self.kernel_transformation,
            "dropout_rate": self.dropout_rate,
            "pointwise_dropout_rate": self.pointwise_dropout_rate,
            "num_heads": self.num_heads,
            "input_length": self.input_length,
            "numerical_stabilizer": self.numerical_stabilizer,
            "num_transformer_layers": self.num_transformer_layers,
            "output_length": self.output_length,
            "final_output_length": self.final_output_length,
            "norm": self.norm,
            "max_seq_length": self.max_seq_length,
            "use_rot_emb": self.use_rot_emb,
            "normalize": self.normalize,
            "seed": self.seed,
            "filter_list_seq": self.filter_list_seq,
            "filter_list_atac": self.filter_list_atac,
            "BN_momentum": self.BN_momentum,
            "final_point_scale": self.final_point_scale,
            "num_motifs": self.num_motifs,
            "motif_units_fc": self.motif_units_fc,
            "motif_dropout_rate": self.motif_dropout_rate,
            "hidden_size": self.hidden_size,
            "d_model": self.d_model,
            "dim": self.dim
        }

        base_config = super().get_config()
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def predict_on_batch(self, inputs, training:bool=False):

        sequence,atac,motif_activity = inputs

        # sequence input processing
        sequence = self.stem_conv(sequence, training=training)
        sequence = self.stem_res_conv(sequence, training=training)
        sequence = self.stem_pool(sequence, training=training)
        sequence = self.conv_tower(sequence, training=training)

        # atac input processsing
        atac_x = self.stem_conv_atac(atac, training=training)
        atac_x = self.stem_res_conv_atac(atac_x, training=training)
        atac_x = self.stem_pool_atac(atac_x, training=training)
        atac_x = self.conv_tower_atac(atac_x,training=training)

        ### motif activity processing w/ MLP
        motif_activity = self.motif_activity_fc1(motif_activity)
        motif_activity = self.motif_dropout1(motif_activity,training=training)
        motif_activity = self.motif_activity_fc2(motif_activity)
        motif_activity = tf.tile(motif_activity, [1, self.output_length, 1])

        transformer_input = tf.concat([sequence,atac_x, motif_activity], axis=2) # append processed seq,atac,motif inputs in channel dim.
        transformer_input = self.pre_transformer_projection(transformer_input)
        out_performer,att_matrices = self.performer(transformer_input, training=training)


        out = self.final_pointwise_conv(out_performer, training=training)
        out = self.dropout(out, training=training)
        out = self.gelu(out)
        out = self.final_dense_profile(out, training=training)
        out_profile = self.crop_final(out)
        return out_profile, att_matrices
