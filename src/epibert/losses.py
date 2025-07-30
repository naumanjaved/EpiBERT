import tensorflow as tf
import tensorflow.experimental.numpy as tnp

'''genformer trial losses'''

def poisson_multinomial(y_true, y_pred, total_weight=0.15, epsilon=1e-6, rescale=True):
    ## copied from the basenji suite
  seq_len = tf.cast(tf.shape(y_true)[1],dtype=tf.float32)

  # add epsilon to protect against tiny values
  y_true += epsilon
  y_pred += epsilon

  # sum across lengths
  s_true = tf.math.reduce_sum(y_true, axis=-2, keepdims=True)
  s_pred = tf.math.reduce_sum(y_pred, axis=-2, keepdims=True)

  # normalize to sum to one
  p_pred = y_pred / s_pred

  # total count poisson loss
  poisson_term = tf.keras.losses.poisson(s_true, s_pred) # B x T
  poisson_term /= seq_len

  # multinomial loss
  pl_pred = tf.math.log(p_pred) # B x L x T
  multinomial_dot = -tf.math.multiply(y_true, pl_pred) # B x L x T
  multinomial_term = tf.math.reduce_sum(multinomial_dot, axis=-2) # B x T
  multinomial_term /= seq_len

  # normalize to scale of 1:1 term ratio
  loss_raw = multinomial_term + total_weight * poisson_term
  if rescale:
    loss_rescale = loss_raw*2.0/(1.0 + total_weight)
  else:
    loss_rescale = loss_raw

  return loss_rescale

@tf.function
def regular_mse(y_pred,
                y_true):
    mse = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)
    loss = mse(y_true, y_pred)
    return loss

@tf.function
def poisson(y_pred,
                y_true):
    poisson = tf.keras.losses.Poisson(reduction=tf.keras.losses.Reduction.NONE)
    return poisson(y_true, y_pred)


@tf.function
def log_mse(y_pred,
                y_true):
    mse = tf.keras.losses.MeanSquaredLogarithmicError(reduction=tf.keras.losses.Reduction.NONE)
    return mse(y_true, y_pred)

@tf.function
def abs_mse(y_pred,
                y_true):
    mse = tf.keras.losses.MeanAbsoluteError(reduction=tf.keras.losses.Reduction.NONE)
    return mse(y_true, y_pred)


@tf.function
def tweedie_loss(y_pred,
                 y_true,
                 tss_tokens,
                 bce_weight,
                 crop_size,
                 out_length):

    tss_tokens_sub = crop_tensor(tss_tokens, crop_size, out_length)

    y_true_sub = subset_tensor(crop_tensor(y_true,
                                               crop_size,
                                               out_length),
                                   tss_tokens_sub)
    y_pred_sub = subset_tensor(crop_tensor(y_pred,
                                               crop_size,
                                               out_length),
                                   tss_tokens_sub)

    msle_batch = tweedie_dev_log_loss(tf.math.log(1.0 + tf.math.maximum(y_true_sub_tpm,
                                                                        1e-09)) / tf.math.log(2.0),
                     tf.math.log(1.0 + tf.math.maximum(y_pred_sub_tpm,
                                                       1e-09)) / tf.math.log(2.0))

    tweedie_batch = tweedie_dev_log_loss(tf.math.maximum(tf.math.log(1.0 + y_true) / tf.math.log(2.0), 1e-09),
                                         tf.math.maximum(tf.math.log(1.0 + y_pred) / tf.math.log(2.0), 1e-09))

    return tweedie_batch

'''
@tf.function
def weighted_bce(y_pred,
                 y_true,
                 positive_weight=1.):
    y_pred = tfp.math.clip_by_value_preserve_gradient(tf.cast(y_pred, dtype = tf.float32),
                                                      clip_value_min=tf.keras.backend.epsilon(),
                                                      clip_value_max=1.0-tf.keras.backend.epsilon())
    weights = tf.ones_like(y_pred, dtype=tf.float32)
    weights = tf.where(tf.equal(y_true, 1), positive_weight * weights, weights)
    y_true = tf.cast(y_true, dtype = tf.float32)
    loss = (y_true * tf.math.log(y_pred + tf.keras.backend.epsilon())) + \
                        ((1.0-y_true + tf.keras.backend.epsilon()) * tf.math.log(1.0-y_pred))
    loss = loss * weights
    loss = tf.reduce_mean(loss, axis=1)
    return -1. * loss

@tf.function
def masked_mse(y_pred,
                 y_true,
                 mask,
                 positive_weight=1.):

    zeros = tf.zeros_like(mask, dtype = tf.bfloat16)
    ones = tf.ones_like(mask, dtype = tf.bfloat16)
    weights = tf.where(tf.equal(mask, 1), ones, zeros)
    #SE =
    #SE = tf.pow(tf.math.log(1.0 + y_pred) - tf.math.log(1.0 + y_true), 2)
    SE = SE * weights
    MSE = tf.reduce_mean(SE, axis=1)
    return MSE



@tf.function
def combined_MSE_BCE(y_pred,
                     y_true,
                     tss_tokens,
                     loss_scale,
                     crop_size,
                     out_length):


    tss_tokens_sub = crop_tensor(tss_tokens, crop_size, out_length)

    y_true_sub_tpm = crop_tensor(y_true,
                                 crop_size,
                                 out_length)

    y_pred_sub_tpm = crop_tensor(y_pred['relu'],
                                 crop_size,
                                 out_length)

    y_pred_sub_sig = crop_tensor(y_pred['sigmoid'],
                                 crop_size,
                                 out_length)


    mse_batch = masked_mse(y_pred_sub_tpm,
                             y_true_sub_tpm,
                             tss_tokens_sub)
    bce_batch = weighted_bce(y_pred_sub_sig,
                             tss_tokens_sub)

    total_loss = mse_batch * (1. / loss_scale) + bce_batch
    loss_ratio = mse_batch / bce_batch

    return total_loss, tss_tokens_sub, y_pred_sub_sig, loss_ratio, mse_batch, bce_batch
'''



'''
@tf.function
def zero_agreement(y_pred, y_true):
### ignore left and right 20% of intervals

    zero = tf.constant(0, dtype=tf.float32)

    input_shape = y_pred.shape

    binary_true = tf.where(tf.equal(tf.cast(y_true,
                                            dtype = tf.float32), zero),
                   tf.zeros(input_shape, tf.float32),
                   tf.ones(input_shape, tf.float32))

    loss = tf.nn.weighted_cross_entropy_with_logits(labels=binary_true,
                                                    logits=y_pred,
                                                    pos_weight=tf.constant(1.0, dtype=tf.float32))
    return tf.reduce_mean(loss, 1)

@tf.function
def combined_poisson_binary_CE_loss(y_pred, y_true):

    pred_logits, pred_counts = y_pred

    pred_logits_sub = subset_tensor(tf.cast(pred_logits,
                                           dtype = tf.float32))
    pred_counts_sub = subset_tensor(tf.cast(pred_counts,
                                           dtype = tf.float32))
    y_true_sub = subset_tensor(tf.cast(y_true,
                                      dtype = tf.float32))


    zero_loss = zero_agreement(pred_logits_sub,
                               y_true_sub)
    poisson = tf.keras.losses.poisson(pred_counts_sub,
                                    y_true_sub) #+ zero_loss

    return tf.reduce_mean(zero_loss + poisson, axis=0)


    mse = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.SUM)

@tf.function
TSS_weighted_poisson_loss(y_pred, y_true,
                         gamma_1, gamma_2, gamma_3):
    ### crop 10% of both sides when computing loss

    crop = tf.multiply(y_pred.shape[1], 0.10)
    out_length = tf.multiply(y_pred.shape[1], 0.80)

    TSS_bins = subset_tensor(y_true['exons_binned'],
                             crop, out_length)
    y_true_sub = subset_tensor(y_true['rna'],
                            crop, out_length)

    y_pred_sub = subset_tensor(y_pred,
                                crop, out_length)

    ### compute a binary cross entropy loss on all bins
    ### have to take mean over input batch
    zero_loss = tf.reduce_sum(zero_agreement(y_pred_sub,
                               y_true_sub), axis=1)



    ### compute an MSE loss on log counts in TSS bins
    total_binned_loss = mse(rna_cpm,
                            y_pred_sub,
                            sample_weight=TSS_bins)

    ### compute an MSE loss on log of total # reads in bin
    total_sum_pred = tf.reduce_sum(y_true_sub, axis=1)
    total_sum_true = tf.reduce_sum(y_pred_sub, axis=1)
    total_sum_loss = mse(total_sum_pred,
                         total_sum_true)


    return gamma_1 * zero_loss + \
            gamma_2 * total_binned_loss + \
                gamm_3 * total_sum_loss


@tf.function
def tweedie_dev_log_loss(y_true, y_pred, p=1.5):
    returns tweedie deviance assuming poisson/gamma mixture

    p_tf=tf.constant(p, dtype = tf.float32)
    zero=tf.constant(0.0, dtype=tf.float32)

    #a = tf.math.pow(tf.math.maximum(y_true, zero), 2.0-p) / ((1.0 - p) * (2.0 - p))

    b = y_true * tf.math.exp((1.0 - p) * tf.math.log(y_pred)) / (1.0 - p_tf)
    c = tf.math.exp((2.0 - p_tf) * tf.math.log(y_pred)) / (2.0 - p_tf)

    loss = c - b
    return tf.reduce_sum(loss)

    ### negative binomial loss with truncated ends
    ### taken from https://stackoverflow.com/questions/55782674/how-should-i-write-the-loss-function-with-keras-and-tensorflow-for-negative-bi
    ### neg binomial PMF is ((k+n-1) C (n-1)) * p^n * (1-p)^k
    #@tf.function
    #def nbinom_pmf_tf(x,n,p):
    #    coeff = tf.lgamma(n + x) - tf.lgamma(x + 1) - tf.lgamma(n)
    #    return tf.cast(tf.exp(coeff + n * tf.log(p) + x * tf.log(1 - p)),dtype=tf.float32)

    #@tf.function
    #def loss_neg_bin_tf_differentiable(y_pred, y_true):
    #    result = tf.map_fn(lambda x: -nbinom_pmf_tf(x[1],
    #                                                x[0][0],
    #                                                tf.minimum(tf.constant(0.99,dtype=tf.float32),x[0][1])),
     #                      (y_pred,y_true),
    #                       dtype=tf.float32)
    #    result = tf.reduce_sum(result,axis=0)
    #    return result


@tf.function
def masked_tweedie_dev_loss(y_true,
                            y_pred,
                            mask,
                            p=1.0):

    returns tweedie deviance assuming poisson/gamma mixture

    y_true = tf.cast(tf.math.log(1.0 + y_true), dtype = tf.float32)
    y_pred = tf.cast(tf.math.log(1.0 + y_pred), dtype = tf.float32)

    crop = tf.cast(y_pred.shape[1] / tf.constant(10),
                   dtype = tf.int32)
    out_length = tf.cast(y_pred.shape[1] - 2 * crop,
                         dtype = tf.int32)
    y_true_sub = subset_tensor(y_true,
                               crop, out_length)
    y_pred_sub = subset_tensor(y_pred,
                               crop, out_length)
    mask_sub = subset_tensor(mask,
                             crop,
                             out_length)


    p=tf.constant(1.1, dtype = tf.float32)
    zero=tf.constant(0.0, dtype=tf.float32)

    a = tf.math.pow(tf.math.maximum(y_true_sub, zero), 2-p) / ((1 - p) * (2 - p))

    b = y_true_sub * tf.math.pow(y_pred_sub, 1-p) / (1-p)

    c = tf.math.pow(y_pred_sub, 2-p) / (2-p)
    dev = a + b + c
    NLL = -dev / 2.0

    return NLL

@tf.function
def masked_poisson_loss(y_pred, y_true, mask):

    p = tf.keras.losses.Poisson(reduction=tf.keras.losses.Reduction.NONE)

    crop = tf.cast(y_pred.shape[1] / tf.constant(10),
                   dtype = tf.int32)
    out_length = tf.cast(y_pred.shape[1] - 2 * crop,
                         dtype = tf.int32)
    y_true_sub = subset_tensor(y_true,
                               crop, out_length)
    y_pred_sub = subset_tensor(y_pred,
                               crop, out_length)
    mask_sub = subset_tensor(mask,
                             crop, out_length)

    batch_sum = tf.reduce_sum(p(y_true_sub * mask_sub, y_pred_sub * mask_sub))

    return batch_sum * (1. / y_pred.shape[0])

'''
