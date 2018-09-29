"""Metric collections."""

from __future__ import division
import tensorflow as tf


class BaseMetric(object):
    def __init__(self, hparams, is_training=True):
        self._hparams = hparams
        self._is_training = is_training

    def __call__(self, input_tensor, label_tensor):
        raise NotImplementedError


class BCMetric(BaseMetric):
    """Binary Classification Metric."""

    def __call__(self, input_tensor, label_tensor):
        """
        Arguments:
            input_tensor: input logits, shape: [batch_size, num_classes],
            label_tensor: label numbers, shape: [batch_size].
        """
        with tf.name_scope("BCMetric"):
            # Correct Predictions: TP + TN.
            pred = tf.argmax(input_tensor, 1)  # [batch_size]
            label = tf.to_int64(label_tensor)
            correct_predictions = tf.equal(pred, label)
            # tp, fp, fn
            tp = tf.count_nonzero(pred * label, dtype=tf.float64)
            fp = tf.count_nonzero(pred * (label - 1), dtype=tf.float64)
            fn = tf.count_nonzero((pred - 1) * label, dtype=tf.float64)
            # Accuracy.
            pred_acc = tf.reduce_mean(tf.cast(correct_predictions, tf.float64))
            pred_prec = tp / (tp + fp)
            pred_rec = tp / (tp + fn)
            pred_f1 = (2 * pred_prec * pred_rec) / (pred_prec + pred_rec)

        return {
            "accuracy": pred_acc,
            "precision": pred_prec,
            "recall": pred_rec,
            "f1_score": pred_f1}

class RGMetric(BaseMetric):
    """Regression Metric."""

    def __call__(self, input_tensor, label_tensor):
        """
        Arguments:
            input_tensor: input logits, shape: [batch_size],
            label_tensor: label numbers, shape: [batch_size].
        """
        with tf.name_scope("REMetric"):
            col_pred = (input_tensor.get_shape().as_list()[1])
            if col_pred != 1:
                raise AssertionError
            input_tensor = tf.squeeze(input_tensor, 1)
            diff_pred = tf.subtract(tf.cast(label_tensor, tf.float32), tf.cast(input_tensor,tf.float32))

            # neg_root_mean_square_error.
            pred_rmse = tf.sqrt(tf.reduce_mean(tf.square(diff_pred))) 
                
            # explained_variance. Best possible score is 1.0, lower values are worse.
            label_mean, label_var = tf.nn.moments(label_tensor, axes=[0])
            diff_mean, diff_var = tf.nn.moments(diff_pred, axes=[0])
            label_var = tf.add(label_var, 1e-6)            
            expl_var = tf.subtract(tf.ones(diff_var.get_shape()), tf.div(diff_var, label_var))

            # neg_mean_absolute_error.The best value is 0.0.
            pred_mae = tf.reduce_mean(tf.abs(diff_pred))
          
            # neg_median_absolute_error.
            pred_tmp_shape = tf.reshape(tf.abs(diff_pred), [-1])
            pred_dim = self._hparams.batch_size // 2
            pred_median = tf.nn.top_k(pred_tmp_shape, pred_dim, sorted=True).values[pred_dim-1]
            
            # R^2 score. Best possible score is 1.0 and it can be negative.
            sdiff = tf.reduce_sum(tf.square(diff_pred))
            diff_label = tf.subtract(label_tensor, tf.reduce_mean(label_tensor))
            sdiffm = tf.reduce_sum(tf.square(diff_label))
            sdiffm = tf.add(sdiffm, 1e-6)
            r2comp = tf.div(sdiff, sdiffm)
            r2 = 1.0 - r2comp


        return {"RMSE": pred_rmse, "EVar": expl_var ,"R2": r2, "MAE": pred_mae, "MedianAE": pred_median}
