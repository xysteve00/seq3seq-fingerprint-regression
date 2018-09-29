""" Hyper parameters"""
import tensorflow as tf


def build_base_hparams():
    """build hyper-parameters"""
    hparams = tf.contrib.training.HParams(
        dropout_rate=0.5,
        num_layers=3,
        size=256,
        learning_rate=0.5,
        learning_rate_decay_factor=0.99,
        buckets=[[30, 30], [60, 60], [90, 90]],
        target_vocab_size=40,
        batch_size=20,
        source_vocab_size=40,
        max_gradient_norm=5.0,
        alpha=0.01,
        reg=True,
        # If we have the prediction (with label) task.
        label_states=True,
        # If we use a recovery task as regularization.
        use_recovery=True,
        # RNN Cell type: GRU, LSTM.
        # https://www.tensorflow.org/api_docs/python/tf/contrib/rnn.
        rnn_cell="GRU",
        # Model Parameters.
        # Prediction network type, e.g., MLP, etc.
        pred_net_type="MLP",
        #
        # MLP Model Parameters.
        #
        # Default MLP output number.
        mlp_output_nums=[1],
        # Loss Parameters.
        # Loss type, e.g., SparseXentLoss, etc.
        loss_type="MSELoss",
        # Metric Parameters.
        # Metric type, e.g., BCMetric for binary classification, etc.
        metric_type="RGMetric")
    return hparams
