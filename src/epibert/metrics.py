import tensorflow as tf
import tensorflow.experimental.numpy as tnp

def pearsons(y_true, y_pred):
    '''
    Helper function to compute r2 for tensors of shape (batch, length)
    '''
    y_true.shape.assert_is_compatible_with(y_pred.shape)
    y_true = tf.cast(y_true, 'float32')
    y_pred = tf.cast(y_pred, 'float32')

    true_sum = tf.reduce_sum(y_true,axis=0)
    pred_sum = tf.reduce_sum(y_pred,axis=0)

    count = tf.reduce_sum(tf.ones_like(y_true),axis=0)

    true_mean = true_sum / count
    true_sq_sum = tf.reduce_sum(tf.math.square(y_true),axis=0)
    pred_mean = pred_sum / count
    pred_sq_sum = tf.reduce_sum(tf.math.square(y_pred),axis=0)
    product_sum = tf.reduce_sum(y_true*y_pred, axis=0)

    covariance = (product_sum - true_mean * pred_sum
                      - pred_mean * true_sum
                      + count * true_mean * pred_mean)

    true_var = true_sq_sum - count * tf.math.square(true_mean)
    pred_var = pred_sq_sum - count * tf.math.square(pred_mean)
    tp_var = tf.math.sqrt(true_var) * tf.math.sqrt(pred_var)
    pearsons = covariance / tp_var

    return pearsons



def r2(y_true, y_pred):
    '''
    Helper function to compute r2 for tensors of shape (batch, length)
    to do: check descrepancy w/ tensorflow implementation
    '''
    y_true.shape.assert_is_compatible_with(y_pred.shape)
    y_true = tf.cast(y_true, 'float32')
    y_pred = tf.cast(y_pred, 'float32')
    residual = tf.reduce_sum(tf.square(y_true - y_pred),axis=0,keepdims=True)
    total = tf.reduce_sum(tf.square(y_true -  tf.reduce_mean(y_true)),axis=0,keepdims=True)
    r2 = tf.constant(1.0,dtype=tf.float32) - residual / total
    return r2#[:,0]

#def plot_att(



def _reduced_shape(shape, axis):
    if axis is None:
        return tf.TensorShape([])
    return tf.TensorShape([d for i, d in enumerate(shape) if i not in axis])


class CorrelationStats(tf.keras.metrics.Metric):
    """Contains shared code for PearsonR and R2."""

    def __init__(self, reduce_axis=None, name='pearsonr'):
        """Pearson correlation coefficient.

        Args: reduce_axis: Specifies over which axis to compute the correlation (say
        (0, 1). If not specified, it will compute the correlation across the
        whole tensor.
        name: Metric name.

        """
        super(CorrelationStats, self).__init__(name=name)
        self._reduce_axis = reduce_axis
        self._shape = None  # Specified in _initialize.

    def _initialize(self, input_shape):
        # Remaining dimensions after reducing over self._reduce_axis.
        self._shape = _reduced_shape(input_shape, self._reduce_axis)

        weight_kwargs = dict(shape=self._shape, initializer='zeros')
        self._count = self.add_weight(name='count', **weight_kwargs)
        self._product_sum = self.add_weight(name='product_sum', **weight_kwargs)
        self._true_sum = self.add_weight(name='true_sum', **weight_kwargs)
        self._true_squared_sum = self.add_weight(name='true_squared_sum',
                                             **weight_kwargs)
        self._pred_sum = self.add_weight(name='pred_sum', **weight_kwargs)
        self._pred_squared_sum = self.add_weight(name='pred_squared_sum',
                                             **weight_kwargs)
    @tf.function
    def update_state(self, y_true, y_pred, sample_weight=None):
        """Update the metric state.

        Args:
            y_true: Multi-dimensional float tensor [batch, ...] containing the ground
            truth values.
            y_pred: float tensor with the same shape as y_true containing predicted
            values.
            sample_weight: 1D tensor aligned with y_true batch dimension specifying
            the weight of individual observations.
        """
        if self._shape is None:
            # Explicit initialization check.
            self._initialize(y_true.shape)
        y_true.shape.assert_is_compatible_with(y_pred.shape)
        y_true = tf.cast(y_true, 'float32')
        y_pred = tf.cast(y_pred, 'float32')

        self._product_sum.assign_add(
            tf.reduce_sum(y_true * y_pred, axis=self._reduce_axis))

        self._true_sum.assign_add(
            tf.reduce_sum(y_true, axis=self._reduce_axis))

        self._true_squared_sum.assign_add(
            tf.reduce_sum(tf.math.square(y_true), axis=self._reduce_axis))

        self._pred_sum.assign_add(
            tf.reduce_sum(y_pred, axis=self._reduce_axis))

        self._pred_squared_sum.assign_add(
            tf.reduce_sum(tf.math.square(y_pred), axis=self._reduce_axis))

        self._count.assign_add(
            tf.reduce_sum(tf.ones_like(y_true), axis=self._reduce_axis))

    def result(self):
        raise NotImplementedError('Must be implemented in subclasses.')
    #@tf.function
    def reset_state(self):
        if self._shape is not None:
            tf.keras.backend.batch_set_value([(v, tnp.zeros(self._shape))
                                        for v in self.variables])
class PearsonR(CorrelationStats):
    """Pearson correlation coefficient.
        Computed as:
            ((x - x_avg) * (y - y_avg) / sqrt(Var[x] * Var[y])
    """

    def __init__(self, reduce_axis=(0,), name='pearsonr'):
        """Pearson correlation coefficient.

        Args:
          reduce_axis: Specifies over which axis to compute the correlation.
          name: Metric name.
    """
        super(PearsonR, self).__init__(reduce_axis=reduce_axis, name=name)

    @tf.function
    def result(self):
        true_mean = self._true_sum / self._count
        pred_mean = self._pred_sum / self._count

        covariance = (self._product_sum
                      - true_mean * self._pred_sum
                      - pred_mean * self._true_sum
                      + self._count * true_mean * pred_mean)

        true_var = self._true_squared_sum - self._count * tf.math.square(true_mean)
        pred_var = self._pred_squared_sum - self._count * tf.math.square(pred_mean)
        tp_var = tf.math.sqrt(true_var) * tf.math.sqrt(pred_var)
        correlation = covariance / tp_var
        return correlation
class R2(CorrelationStats):
    """R-squared  (fraction of explained variance)."""

    def __init__(self, reduce_axis=None, name='R2'):
        """R-squared metric.

    Args:
        reduce_axis: Specifies over which axis to compute the correlation.
        name: Metric name.
    """
        super(R2, self).__init__(reduce_axis=reduce_axis, name=name)
    @tf.function
    def result(self):
        true_mean = self._true_sum / self._count
        total = self._true_squared_sum - self._count * tf.math.square(true_mean)
        residuals = (self._pred_squared_sum - 2 * self._product_sum
                 + self._true_squared_sum)
        return tf.ones_like(residuals) - residuals / total

class MetricDict:
    def __init__(self, metrics):
        self._metrics = metrics

    def update_state(self, y_true, y_pred):
        for k, metric in self._metrics.items():
            metric.update_state(y_true, y_pred)
    def reset_state(self):
        for k, metric in self._metrics.items():
            metric.reset_state()

    def result(self):
        return {k: metric.result() for k, metric in self._metrics.items()}





def log10_1p(x):
    numerator = tf.math.log(x + 1.0)
    denominator = tf.math.log(tf.constant(10, dtype=numerator.dtype))
    return numerator / denominator



class correlation_stats_gene_centered(tf.keras.metrics.Metric):
    def __init__(self, reduce_axis=None, name='correlation_stats_gene_centered'):
        """contains code for computing
            R2, pearsons correlation, and MSE over TSS sites provided
            in the tss input to the call function
        """
        super(correlation_stats_gene_centered, self).__init__(name=name)
        self._init = None

    def _initialize(self):
        self._y_trues = tf.Variable([], shape=(None,), validate_shape=False)
        self._y_preds = tf.Variable([], shape=(None,), validate_shape=False)
        self._cell_types = tf.Variable([], shape=(None,), validate_shape=False,dtype=tf.int32)
        self._gene_map = tf.Variable([], shape=(None,), validate_shape=False,dtype=tf.int32)

        """
        originally wanted to compute over each actual val step but having
        trouble w/ keeping track of values in tensorarray within tf keras metrics subclass
        """

    def update_state(self, y_true, y_pred, cell_type,gene_map):
        if self._init is None:
            # initialization check.
            self._initialize()
        self._init = 1.0

        y_true.shape.assert_is_compatible_with(y_pred.shape)

        ## ensure no batch dimension, this will preserve order of tensors
        y_true = tf.reshape(y_true, [-1])
        y_pred = tf.reshape(y_pred, [-1])
        cell_type=tf.reshape(cell_type,[-1])
        gene_map=tf.reshape(gene_map,[-1])

        self._y_trues.assign(y_true)
        self._y_preds.assign(y_pred)

        self._cell_types.assign(cell_type)
        self._gene_map.assign(gene_map)


    def result(self):
        return {'pearsonR': pearsons(self._y_trues,
                                     self._y_preds),
                'R2': r2(self._y_trues,
                         self._y_preds),
                'y_trues': self._y_trues,
                'y_preds': self._y_preds,
                'cell_types': self._cell_types,
                'gene_map': self._gene_map
               }
    #@tf.function
    def reset_state(self):
        tf.keras.backend.batch_set_value([(v, 0) for v in self.variables])



class correlation_stats(tf.keras.metrics.Metric):
    def __init__(self, reduce_axis=None, name='correlation_stats'):
        """contains code for storing
           values through a training loop
           and computing an overall pearsons
        """
        super(correlation_stats_gene_centered, self).__init__(name=name)
        self._init = None

    def _initialize(self):
        self._y_trues = tf.Variable([], shape=(None,), validate_shape=False)
        self._y_preds = tf.Variable([], shape=(None,), validate_shape=False)

        """
        originally wanted to compute over each actual val step but having
        trouble w/ keeping track of values in tensorarray within tf keras metrics subclass
        """

    def update_state(self, y_true, y_pred):
        if self._init is None:
            # initialization check.
            self._initialize()
        self._init = 1.0

        y_true.shape.assert_is_compatible_with(y_pred.shape)

        ## ensure no batch dimension, this will preserve order of tensors
        y_true = tf.reshape(y_true, [-1])
        y_pred = tf.reshape(y_pred, [-1])

        self._y_trues.assign(y_true)
        self._y_preds.assign(y_pred)

    def result(self):
        return {'pearsonR': pearsons(self._y_trues,
                                     self._y_preds),
                'R2': r2(self._y_trues,
                         self._y_preds),
                'y_trues': self._y_trues,
                'y_preds': self._y_preds
               }
    #@tf.function
    def reset_state(self):
        tf.keras.backend.batch_set_value([(v, 0) for v in self.variables])
