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
            if not self._hparams.reg:
                label_tensor = label_tensor[0]

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
        with tf.name_scope("MSELoss"):
            if self._hparams.num_prop == 1: 
                col_pred = (input_tensor.get_shape().as_list()[1])
                if col_pred != 1:
                    raise AssertionError
#                print("loss label_tensor shape before "+"\n", label_tensor.get_shape().as_list())
                label_tensor = label_tensor[0]
                input_tensor = tf.squeeze(input_tensor, 1)

                loss = tf.losses.mean_squared_error(tf.cast(label_tensor, tf.float64), \
                           tf.cast(input_tensor, tf.float64), weights=1.0, scope="MSELoss")
            if (self._hparams.num_prop > 1):
                loss = 0.
                #input_tensor = tf.Print(input_tensor,[input_tensor], message="input_tensor")
                #label_tensor = tf.Print(label_tensor,[label_tensor], message="label_tensor")

                #print("label_tensor shape after", label_tensor.get_shape().as_list())
                #print("input_tensor shape after", input_tensor.get_shape().as_list())
                #print("label_tensor type", type(label_tensor))
                #print("input_tensor type", type(input_tensor))
                input_tensor = tf.transpose(input_tensor,[1,0])
                #print("input_tensor shape after", input_tensor.get_shape().as_list())
                #input_tensor = tf.Print(input_tensor,[input_tensor], message="input_tensori")
                #label_tensor = tf.Print(label_tensor,[label_tensor], message="label_tensori")
                for i in range(self._hparams.num_prop):
                    #input_tensor2 = tf.Print(input_tensor[i],[input_tensor[i]], message="input_tensori")
                    #label_tensor2 = tf.Print(label_tensor[i],[label_tensor[i]], message="label_tensori")
                    #print("label_tensor i", label_tensor[i].get_shape().as_list())
                    #print("input_tensor i", input_tensor[i].get_shape().as_list())


                    loss += tf.losses.mean_squared_error(tf.cast(label_tensor[i], tf.float64), \
                        tf.cast(input_tensor[i], tf.float64), weights=1.0, scope="MSELoss")


        return loss
