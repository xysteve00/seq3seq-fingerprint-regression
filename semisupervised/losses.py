"""Loss function collections."""

import tensorflow as tf

class BaseLoss(object):
    def __init__(self, hparams, is_training=True):
        self._hparams = hparams
        self._is_training = is_training

    def __call__(self, input_tensor, label_tensor):
        raise NotImplementedError


class SparseXentLoss(BaseLoss):
    def __call__(self, input_tensor, label_tensor):
        """
        Arguments:
            input_tensor: input logits, shape: [batch_size, num_classes],
            label_tensor: label numbers, shape: [batch_size].
        """
        with tf.name_scope("XentLoss"):
            per_example_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=tf.to_int64(label_tensor), logits=input_tensor, name="SparseXentLoss")
            loss = tf.reduce_mean(per_example_loss)

        return loss

class SmoothMAELoss(BaseLoss):
    def __call__(self, input_tensor, label_tensor, delta=0.5):
        """
        Arguments:
            input_tensor: input logits, shape: [batch_size].
            label_tensor: label numbers, shape: [batch_size].
        """
        with tf.name_scope("SmoothMAELoss"):
            col_pred = (input_tensor.get_shape().as_list()[1])
            if col_pred != 1:
                raise AssertionError
            true = tf.cast(label_tensor, tf.float32)
            pred = tf.cast(input_tensor, tf.float32)
            loss = tf.where(tf.abs(true-pred) < delta, 0.5*((true-pred)**2),
                        delta*tf.abs(true - pred) - 0.5*(delta**2))
        return tf.reduce_sum(loss)
        

class LogCoshLoss(BaseLoss):
    def __call__(self, input_tensor, label_tensor):
        """
        Arguments:
            input_tensor: input logits, shape: [batch_size],
            label_tensor: label numbers, shape: [batch_size].
        """
        with tf.name_scope("LogCoshLoss"):
            col_pred = (input_tensor.get_shape().as_list()[1])
            if col_pred != 1:
                raise AssertionError
            input_tensor = tf.squeeze(input_tensor, 1)
            loss = tf.log(tf.cosh(tf.cast(label_tensor, tf.float32) -
                                tf.cast(input_tensor, tf.float32)))
        return tf.reduce_sum(loss)

class MSELoss(BaseLoss):
    def __call__(self, input_tensor, label_tensor):
        """
        Arguments:
            input_tensor: input logits, shape: [batch_size],
            label_tensor: label numbers, shape: [batch_size].
        """
        with tf.name_scope("LogCoshLoss"):
            col_pred = (input_tensor.get_shape().as_list()[1])
            if col_pred != 1:
                raise AssertionError
            input_tensor = tf.squeeze(input_tensor, 1)
            loss = tf.losses.mean_squared_error(tf.cast(label_tensor, tf.float64), \
                       tf.cast(input_tensor, tf.float64), weights=1.0, scope="MSELoss")
        return loss
