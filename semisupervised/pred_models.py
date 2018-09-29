"""Prediction Network Zoo."""

import tensorflow as tf


class BasePredictionNet(object):
    def __init__(self, hparams, is_training=True, **kwargs):
        self._hparams = hparams
        self._is_training = is_training

    def __call__(self, input_tensor, reuse=False):
        raise NotImplementedError


class MLP(BasePredictionNet):
    """Multi-layer Peceptron."""

    def __call__(self, input_tensor, reuse=False):
        logits = input_tensor
        for idx, output_num in enumerate(self._hparams.mlp_output_nums):
            scope_name = "pred"
            scope_name += ("_%d" % idx) if idx > 0 else ""
            logits = tf.contrib.layers.fully_connected(
                logits,
                output_num,
                activation_fn=tf.nn.relu,
                weights_initializer=tf.contrib.layers.xavier_initializer(),
                biases_initializer=tf.zeros_initializer(),
                reuse=reuse,
                trainable=self._is_training,
                scope=scope_name)
        return logits
