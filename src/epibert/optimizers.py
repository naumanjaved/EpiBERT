"""Original implementation from https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/utils/adafactor.py """
import re

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.python.ops import math_ops

# from https://github.com/Smokrow/addons/blob/dev/adafactore/tensorflow_addons/optimizers/adafactor.py


def reduce_rms(x):
    return tf.math.sqrt(tf.reduce_mean(tf.square(x)))


class AdafactorOptimizer(tf.keras.optimizers.Optimizer):
    """Optimizer that implements the Adafactor algorithm.
    Adafactor is described in https://arxiv.org/abs/1804.04235.
    Adafactor is most similar to Adam (Kingma and Ba), the major differences are:
    1. For a two-dimensional AxB weight matrix, Adafactor uses only A+B auxiliary
         parameters to maintain the second-moment estimator, instead of AB.
         This is advantageous on memory-limited systems.    In addition, beta1
         (momentum) is set to zero by default, saving an additional auxiliary
         parameter per weight.    Variables with >=3 dimensions are treated as
         collections of two-dimensional matrices - factorization is over the final
         two dimensions.
    2. Adafactor incorporates "update-clipping" - a scale-invariant analog of
         gradient clipping.  This adds stability
    3. Adafactor does not require an external "learning rate".    By default, it
         incorporates a relative-update-scale schedule, corresponding to
         inverse-square-root learning-rate-decay in ADAM.  We hope this works well
         for most applications.
    ALGORITHM:
    parameter -= absolute_update_scale * clip(grad / grad_scale)
    where:
        absolute_update_scale := relative_update_scale * parameter_scale
        relative_update_scale := min((step_num + 1)**-0.5, 1e-2)
        parameter_scale := max(rms(var)), epsilon2)
        clip(x) := x / max(1.0, rms(x))
        grad_scale := tf.sqrt(v)     (v is the second-moment estimator)
    The second-moment estimator v is maintained in a manner similar to Adam:
    We initialize
    ```
    if var is 2-dimensional:
        v_r <- zeros([num_rows])
        v_c <- zeros([num_cols])
    if var is 0-dimensional or 1-dimensional:
        v <- zeros(shape(var))
    ```
    The update rule is as follows:
    ```
    decay_rate = 1 - (step_num + 1) ^ -0.8
    grad_squared = tf.square(grad) + epsilon1
    if var is 2-dimensional:
        v_r <- decay_rate * v_r + (1 - decay_rate) * \
                                   reduce_mean(grad_squared, 1)
        v_c <- decay_rate * v_c + (1 - decay_rate) * \
                                   reduce_mean(grad_squared, 0)
        v = outer_prod(v_r, v_c) / reduce_mean(v_r)
    if var is 0-dimensional or 1-dimensional:
        v <- decay_rate * v + (1 - decay_rate) * grad_squared
    ```
    For variables with >=3 dimensions, we factorize the second-moment accumulator
    over the final 2 dimensions.    See the code for details.
    Several parts of this algorithm are configurable from the initializer.
        multiply_by_parameter_scale:    If True, then compute absolute_update_scale
            as described above.  If False, let absolute_update_scale be the externally
            supplied learning_rate.
        learning_rate: represents relative_update_scale if
            multiply_by_parameter_scale==True, or absolute_update_scale if
            multiply_by_parameter_scale==False.
        decay_rate: Decay rate of the second moment estimator (varies by step_num).
            This should be set to a function such that:
            1-1/(step_num + 1) <= decay_rate(step_num) < 1.0
        beta1: enables momentum, as in Adam.    Uses extra memory if nonzero.
        clipping_threshold: should be >=1.0 or None for no update clipping
        factored: whether to factor the second-moment estimator.    True means
            less memory usage.
    """

    def __init__(self,
                 multiply_by_parameter_scale=False,
                 learning_rate=None,
                 learning_rate_fn=None,
                 decay_rate=None,
                 beta1=0.0,
                 clipping_threshold=1.0,
                 factored=True,
                 use_locking=False,
                 name="Adafactor",
                 epsilon1=1e-30,
                 epsilon2=1e-3,
                 **kwargs):
        """Construct a new Adafactor optimizer.
        See class comment.
        Args:
            multiply_by_parameter_scale: a boolean
            learning_rate: an optional Scalar.
            decay_rate: an optional Scalar.
            beta1: a float value between 0 and 1
            clipping_threshold: an optional float >= 1
            factored: a boolean - whether to use factored second-moment estimator
                for 2d variables
            use_locking: If True use locks for update operations.
            name: Optional name for the operations created when applying gradients.
                Defaults to "AdafactorOptimizer".
            epsilon1: Regularization constant for squared gradient.
            epsilon2: Regularization constant for parameter scale.
        Raises:
            ValueError: if absolute_update_scale and relative_update_scale_fn are both
                present or both absent.
        """
        super(AdafactorOptimizer, self).__init__(name=name, **kwargs)

        # Set Flags
        self.multiply_by_parameter_scale = multiply_by_parameter_scale
        self.factored = factored
        self.use_locking = use_locking
        self.has_beta_1 = (beta1 != 0.0)

        # Set defaults
        if learning_rate is None:
            learning_rate = self._learning_rate_default(
                multiply_by_parameter_scale)
        if learning_rate_fn is not None:
            learning_rate = self._learning_rate_from_fn(
                                learning_rate_fn)

        if decay_rate is None:
            decay_rate = self._decay_rate_default()

        # Set Hypers
        self._set_hyper("decay_rate", decay_rate)
        self._set_hyper("learning_rate", learning_rate)
        self._set_hyper("beta1", beta1)
        self._set_hyper("clipping_threshold", clipping_threshold)
        self._set_hyper("factored", factored)
        self._set_hyper("epsilon1", epsilon1)
        self._set_hyper("epsilon2", epsilon2)

    def _prepare_local(self, var_device, var_dtype, apply_state):
        super()._prepare_local(var_device, var_dtype, apply_state)

        local_step = math_ops.cast(self.iterations + 1, var_dtype)
        learning_rate_t = tf.identity(
            self._get_hyper("learning_rate", var_dtype))
        decay_rate_t = tf.identity(self._get_hyper("decay_rate", var_dtype))
        beta_1_t = tf.identity(self._get_hyper("beta1", var_dtype))
        clipping_threshold_t = tf.identity(
            self._get_hyper("clipping_threshold", var_dtype))
        epsilon1_t = tf.identity(self._get_hyper("epsilon1", var_dtype))
        epsilon2_t = tf.identity(self._get_hyper("epsilon2", var_dtype))

        apply_state[(var_device, var_dtype)].update(
            dict(
                learning_rate=learning_rate_t,
                decay_rate=decay_rate_t,
                beta1=beta_1_t,
                clipping_threshold=clipping_threshold_t,
                epsilon1=epsilon1_t,
                epsilon2=epsilon2_t,
            )
        )

    def get_config(self):
        config = {
            "learning_rate": self._serialize_hyperparameter("learning_rate"),
            "decay_rate": self._serialize_hyperparameter("decay_rate"),
            "beta1": self._serialize_hyperparameter("beta1"),
            "clipping_threshold": self._serialize_hyperparameter("clipping_threshold"),
            "epsilon1": self._serialize_hyperparameter("epsilon1"),
            "epsilon2": self._serialize_hyperparameter("epsilon2")
        }
        base_config = super(AdafactorOptimizer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def _should_use_factored_second_moment_estimate(self, shape):
        """Should we use a factored second moment estimator.
        Based on the shape of the variable.
        Args:
            shape: a list of integers
        Returns:
            a boolean
        """
        return self.factored and len(shape) >= 2

    def _create_slots(self, var_list):
        for var in var_list:
            shape = var.get_shape().as_list()
            if self.has_beta_1:
                self.add_slot(var, "m")
            if self._should_use_factored_second_moment_estimate(shape):
                r_val = tf.zeros(shape[:-1], dtype=tf.float32)
                c_val = tf.zeros(shape[:-2] + shape[-1:], dtype=tf.float32)
                self.add_slot(var, "vr", initializer=r_val)
                self.add_slot(var, "vc", initializer=c_val)
            else:
                v_val = tf.zeros(shape, dtype=tf.float32)
                self.add_slot(var, "v", initializer=v_val)

    def _apply_dense(self, grad, var):
        return self._resource_apply_dense(grad, var)

    def _apply_sparse(self, grad, var):
        return self._apply_dense(tf.convert_to_tensor(grad), var)

    def _resource_apply_sparse(self, grad, handle, indices):
        return self._resource_apply_dense(
            tf.convert_to_tensor(tf.IndexedSlices(
                grad, indices, tf.shape(handle))),
            handle)

    def _parameter_scale(self, var):
        """Estimate the scale of the parameters from the current values.
        We include a minimum value of 0.001 to give it a chance to escape 0
        if it was zero-initialized.
        Instead of using the value, we could impute the scale from the shape,
        as initializers do.
        Args:
            var: a variable or Tensor.
        Returns:
            a Scalar
        """
        tf.cast(var, tf.float32)
        testy = tf.maximum(reduce_rms(var), self._get_hyper("epsilon2"))
        tf.cast(testy, tf.float32)
        return tf.maximum(reduce_rms(var), self._get_hyper("epsilon2"))

    def _resource_apply_dense(self, grad, var, apply_state=None):
        var_device, var_dtype = var.device, var.dtype.base_dtype
        coefficients = (apply_state or {}).get(
            (var_device, var_dtype)
        ) or self._fallback_apply_state(var_device, var_dtype)

        grad = tf.cast(grad, tf.float32)
        grad_squared = tf.square(grad) + coefficients["epsilon1"]
        grad_squared_mean = tf.reduce_mean(grad_squared)
        decay_rate = coefficients["decay_rate"]
        update_scale = coefficients["learning_rate"]
        old_val = var
        if self.multiply_by_parameter_scale:
            scale_factor = self._parameter_scale(old_val)
            update_scale *= tf.cast(scale_factor, tf.float32)
        # HACK: Make things dependent on grad.
        # This confounds the XLA rewriter and keeps it from fusing computations
        # across different variables.  This fusion is a bad for HBM usage, since
        # it causes the gradients to persist in memory.
        decay_rate += grad_squared_mean * 1e-30
        update_scale += grad_squared_mean * 1e-30
        # END HACK
        mixing_rate = 1.0 - decay_rate
        shape = var.get_shape().as_list()
        updates = []
        if self._should_use_factored_second_moment_estimate(shape):
            grad_squared_row_mean = tf.reduce_mean(grad_squared, -1)
            vr = self.get_slot(var, "vr")
            new_vr = (decay_rate * vr + mixing_rate * grad_squared_row_mean)
            vr_update = vr.assign(new_vr, use_locking=self.use_locking)
            updates.append(vr_update)

            grad_squared_col_mean = tf.reduce_mean(grad_squared, -2)
            vc = self.get_slot(var, "vc")
            new_vc = (decay_rate * vc + mixing_rate * grad_squared_col_mean)
            vc_update = vc.assign(new_vc, use_locking=self.use_locking)
            updates.append(vc_update)

            long_term_mean = tf.reduce_mean(new_vr, -1, keepdims=True)
            r_factor = tf.math.rsqrt(new_vr / long_term_mean)
            c_factor = tf.math.rsqrt(new_vc)
            x = grad * tf.expand_dims(r_factor, -1) * \
                tf.expand_dims(c_factor, -2)
        else:
            v = self.get_slot(var, "v")
            new_v = decay_rate * v + mixing_rate * grad_squared
            v_update = v.assign(new_v, use_locking=self.use_locking)
            updates = [v_update]
            x = grad * tf.math.rsqrt(new_v)

        if coefficients["clipping_threshold"] is not None:
            clipping_denom = tf.maximum(1.0, reduce_rms(
                x) / coefficients["clipping_threshold"])
            x /= clipping_denom
        subtrahend = update_scale * x

        if self.has_beta_1:
            m = self.get_slot(var, "m")
            new_m = coefficients["beta1"] * \
                tf.cast(m, tf.float32) + \
                (1.0 - coefficients["beta1"]) * subtrahend
            subtrahend = new_m
            new_m = self._cast_like(new_m, var)
            m_update_value = m.assign(new_m, use_locking=self.use_locking)
            updates.append(m_update_value)

        new_val = tf.cast(old_val, tf.float32) - subtrahend
        new_val = var.assign(new_val, use_locking=self.use_locking)
        updates = [new_val] + updates
        return tf.group(*updates)

    def _cast_like(self, x, y):
        """Cast x to y's dtype, if necessary. Grabbed from tensor2tensor/layers/common_layers"""
        x = tf.convert_to_tensor(x)
        y = tf.convert_to_tensor(y)

        if x.dtype.base_dtype == y.dtype.base_dtype:
            return x

        cast_x = tf.cast(x, y.dtype)
        if cast_x.device != x.device:
            x_name = "(eager Tensor)"
            try:
                x_name = x.name
            except AttributeError:
                pass
            tf.logging.warning("Cast for %s may induce copy from '%s' to '%s'", x_name,
                               x.device, cast_x.device)
            return cast_x

    def _decay_rate_default(self):
        return self._adafactor_decay_rate_pow(0.8)

    def _learning_rate_default(self, multiply_by_parameter_scale):
        learning_rate = tf.minimum(tf.math.rsqrt(self.step_num() + 1.0), 0.01)
        if not multiply_by_parameter_scale:
            learning_rate *= 0.05
        return learning_rate
    
    def _learning_rate_from_fn(self, learning_rate_fn):
        learning_rate = tf.cast(learning_rate_fn(self.iterations), tf.float32)
        return learning_rate

    def _adafactor_decay_rate_adam(self, beta2):
        """Second-moment decay rate like Adam, subsuming the correction factor.
        Args:
            beta2: a float between 0 and 1
        Returns:
            a scalar
        """
        t = tf.cast(self.iterations, tf.float32) + 1.0
        decay = beta2 * (1.0 - tf.pow(beta2, t - 1.0)) / \
            (1.0 - tf.pow(beta2, t))
        # decay = tf.cond(tf.equal(t, 1.0), lambda: beta2, lambda: decay)
        return decay

    def _adafactor_decay_rate_pow(self, exponent):
        """Second moment decay rate where memory-length grows as step_num^exponent.
        Args:
            exponent: a float between 0 and 1
        Returns:
            a scalar
        """
        return 1.0 - tf.pow((self.step_num() + 1.0), -exponent)

    def step_num(self):
        return tf.cast(self.iterations, tf.float32)

# from: https://github.com/huggingface/transformers/blob/master/src/transformers/optimization_tf.py


class WarmUp(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Applys a warmup schedule on a given learning rate decay schedule."""

    def __init__(self, initial_learning_rate, decay_schedule_fn, warmup_steps, power=1.0, name=None):
        super().__init__()
        self.initial_learning_rate = initial_learning_rate
        self.warmup_steps = warmup_steps
        self.power = power
        self.decay_schedule_fn = decay_schedule_fn
        self.name = name

    def __call__(self, step):
        with tf.name_scope(self.name or "WarmUp") as name:
            # Implements polynomial warmup. i.e., if global_step < warmup_steps, the
            # learning rate will be `global_step/num_warmup_steps * init_lr`.
            global_step_float = tf.cast(step, tf.float32)
            warmup_steps_float = tf.cast(self.warmup_steps, tf.float32)
            warmup_percent_done = global_step_float / warmup_steps_float
            warmup_learning_rate = self.initial_learning_rate * \
                tf.math.pow(warmup_percent_done, self.power)
            return tf.cond(
                global_step_float < warmup_steps_float,
                lambda: warmup_learning_rate,
                lambda: self.decay_schedule_fn(tf.cast(step,dtype=tf.float32) - warmup_steps_float),
                name=name,
            )

    def get_config(self):
        return {
            "initial_learning_rate": self.initial_learning_rate,
            "decay_schedule_fn": self.decay_schedule_fn,
            "warmup_steps": self.warmup_steps,
            "power": self.power,
            "name": self.name,
        }


class WarmUpLinearDecayScheduler(keras.callbacks.Callback):
    """Cosine decay with warmup learning rate scheduler
    """

    def __init__(self,
                 learning_rate_base,
                 total_steps,
                 global_step_init=0,
                 warmup_learning_rate=0.0,
                 warmup_steps=0,
                 hold_base_rate_steps=0,
                 verbose=0):
        """Constructor for cosine decay with warmup learning rate scheduler.
    Arguments:
        learning_rate_base {float} -- base learning rate.
        total_steps {int} -- total number of training steps.
    Keyword Arguments:
        global_step_init {int} -- initial global step, e.g. from previous checkpoint.
        warmup_learning_rate {float} -- initial learning rate for warm up. (default: {0.0})
        warmup_steps {int} -- number of warmup steps. (default: {0})
        hold_base_rate_steps {int} -- Optional number of steps to hold base learning rate
                                    before decaying. (default: {0})
        verbose {int} -- 0: quiet, 1: update messages. (default: {0})
        """

        super(WarmUpLinearDecayScheduler, self).__init__()
        self.learning_rate_base = learning_rate_base
        self.total_steps = total_steps
        self.global_step = global_step_init
        self.warmup_learning_rate = warmup_learning_rate
        self.warmup_steps = warmup_steps
        self.hold_base_rate_steps = hold_base_rate_steps
        self.verbose = verbose
        self.learning_rates = []

        learning_rate_fn = tf.keras.optimizers.schedules.PolynomialDecay(
            initial_learning_rate=learning_rate_base, decay_steps=total_steps, end_learning_rate=0.0
        )

        self.sched = WarmUp(learning_rate_base,
                            learning_rate_fn, warmup_steps=warmup_steps)

    def on_batch_end(self, batch, logs=None):
        self.global_step = self.global_step + 1
        lr = K.get_value(self.model.optimizer.lr)
        self.learning_rates.append(lr)

    def on_batch_begin(self, batch, logs=None):

        # lr = cosine_decay_with_warmup(global_step=self.global_step,
        #                               learning_rate_base=self.learning_rate_base,
        #                               total_steps=self.total_steps,
        #                               warmup_learning_rate=self.warmup_learning_rate,
        #                               warmup_steps=self.warmup_steps,
        #                               hold_base_rate_steps=self.hold_base_rate_steps)

        lr = self.sched(self.global_step)

        K.set_value(self.model.optimizer.lr, lr)
        if self.verbose > 0:
            print('\nBatch %05d: setting learning '
                  'rate to %s.' % (self.global_step + 1, lr))


class AdamWeightDecay(tf.keras.optimizers.Adam):
    """Adam enables L2 weight decay and clip_by_global_norm on gradients.
  Just adding the square of the weights to the loss function is *not* the
  correct way of using L2 regularization/weight decay with Adam, since that will
  interact with the m and v parameters in strange ways.
  Instead we want ot decay the weights in a manner that doesn't interact with
  the m/v parameters. This is equivalent to adding the square of the weights to
  the loss with plain (non-momentum) SGD.
  """

    def __init__(
        self,
        learning_rate=0.001,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-7,
        amsgrad=False,
        weight_decay_rate=0.0,
        include_in_weight_decay=None,
        exclude_from_weight_decay=["layer_norm", "bias"],
        name="AdamWeightDecay",
        **kwargs
    ):
        super().__init__(learning_rate, beta_1, beta_2, epsilon, amsgrad, name, **kwargs)
        self.weight_decay_rate = weight_decay_rate
        self._include_in_weight_decay = include_in_weight_decay
        self._exclude_from_weight_decay = exclude_from_weight_decay

    @classmethod
    def from_config(cls, config):
        """Creates an optimizer from its config with WarmUp custom object."""
        custom_objects = {"WarmUp": WarmUp}
        return super().from_config(config, custom_objects=custom_objects)

    def _prepare_local(self, var_device, var_dtype, apply_state):
        super()._prepare_local(var_device, var_dtype, apply_state)
        apply_state["weight_decay_rate"] = tf.constant(
            self.weight_decay_rate, name="adam_weight_decay_rate")

    def _decay_weights_op(self, var, learning_rate, apply_state):
        do_decay = self._do_use_weight_decay(var.name)
        if do_decay:
            return var.assign_sub(
                learning_rate * var * apply_state["weight_decay_rate"], use_locking=self._use_locking
            )
        return tf.no_op()

    def apply_gradients(self, grads_and_vars, clip_norm=1.0, name=None, **kwargs):
        grads, tvars = list(zip(*grads_and_vars))
        (grads, _) = tf.clip_by_global_norm(grads, clip_norm=clip_norm)
        return super().apply_gradients(zip(grads, tvars), **kwargs)

    def _get_lr(self, var_device, var_dtype, apply_state):
        """Retrieves the learning rate with the given state."""
        if apply_state is None:
            return self._decayed_lr_t[var_dtype], {}

        apply_state = apply_state or {}
        coefficients = apply_state.get((var_device, var_dtype))
        if coefficients is None:
            coefficients = self._fallback_apply_state(var_device, var_dtype)
            apply_state[(var_device, var_dtype)] = coefficients

        return coefficients["lr_t"], dict(apply_state=apply_state)

    def _resource_apply_dense(self, grad, var, apply_state=None):
        lr_t, kwargs = self._get_lr(
            var.device, var.dtype.base_dtype, apply_state)
        decay = self._decay_weights_op(var, lr_t, apply_state)
        with tf.control_dependencies([decay]):
            return super()._resource_apply_dense(grad, var, **kwargs)

    def _resource_apply_sparse(self, grad, var, indices, apply_state=None):
        lr_t, kwargs = self._get_lr(
            var.device, var.dtype.base_dtype, apply_state)
        decay = self._decay_weights_op(var, lr_t, apply_state)
        with tf.control_dependencies([decay]):
            return super()._resource_apply_sparse(grad, var, indices, **kwargs)

    def get_config(self):
        config = super().get_config()
        config.update({"weight_decay_rate": self.weight_decay_rate})
        return config

    def _do_use_weight_decay(self, param_name):
        """Whether to use L2 weight decay for `param_name`."""
        if self.weight_decay_rate == 0:
            return False

        if self._include_in_weight_decay:
            for r in self._include_in_weight_decay:
                if re.search(r, param_name) is not None:
                    return True

        if self._exclude_from_weight_decay:
            for r in self._exclude_from_weight_decay:
                if re.search(r, param_name) is not None:
                    return False
        return True