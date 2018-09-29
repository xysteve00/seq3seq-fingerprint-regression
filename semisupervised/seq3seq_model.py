"""Seq3seq Model Extension in Seq2seq-fingerprint."""
# pylint: disable=invalid-name

from __future__ import absolute_import, division, print_function

import copy
import json
import os
import random

import numpy as np
import tensorflow as tf
from six.moves import range
from smile import logging

from . import losses, metrics, pred_models
from .base_hparams import build_base_hparams
from .utils import (EOS_ID, GO_ID, PAD_ID, initialize_vocabulary,
                    sentence_to_token_ids, smile_tokenizer)


def embedding_attention_seq2seq(encoder_inputs,
                                decoder_inputs,
                                cell,
                                num_encoder_symbols,
                                num_decoder_symbols,
                                embedding_size,
                                num_heads=1,
                                output_projection=None,
                                feed_previous=False,
                                dtype=None,
                                scope=None,
                                initial_state_attention=False):
    """A customized version of embedding_attention_seq2seq.
    https://github.com/tensorflow/tensorflow/blob/r1.8/tensorflow/contrib/legacy_seq2seq/python/ops/seq2seq.py#L791
    """
    with tf.variable_scope(
            scope or "embedding_attention_seq2seq", dtype=dtype) as scope:
        dtype = scope.dtype
        # Encoder.
        encoder_cell = copy.deepcopy(cell)
        encoder_cell = tf.contrib.rnn.EmbeddingWrapper(
            encoder_cell,
            embedding_classes=num_encoder_symbols,
            embedding_size=embedding_size)
        encoder_outputs, encoder_state = tf.nn.static_rnn(
            encoder_cell, encoder_inputs, dtype=dtype)

        # First calculate a concatenation of encoder outputs to put attention on.
        top_states = [
            tf.reshape(e, [-1, 1, cell.output_size]) for e in encoder_outputs
        ]
        attention_states = tf.concat(top_states, 1)

        # Decoder.
        output_size = None
        if output_projection is None:
            cell = tf.contrib.rnn.OutputProjectionWrapper(
                cell, num_decoder_symbols)
            output_size = num_decoder_symbols

        if isinstance(feed_previous, bool):
            outputs, decoder_state = (
                tf.contrib.legacy_seq2seq.embedding_attention_decoder(
                    decoder_inputs,
                    encoder_state,
                    attention_states,
                    cell,
                    num_decoder_symbols,
                    embedding_size,
                    num_heads=num_heads,
                    output_size=output_size,
                    output_projection=output_projection,
                    feed_previous=feed_previous,
                    initial_state_attention=initial_state_attention))
            return outputs, encoder_state, decoder_state


class Seq3SeqModel(object):  # pylint: disable=too-many-instance-attributes
    """Customized seq3seq model for fingerprint method."""
    MODEL_PARAMETER_FIELDS = [
        # Feedforward parameters.
        "source_vocab_size",
        "target_vocab_size",
        "buckets",
        "size",
        "num_layers",
        "dropout_rate",
        # Training parameters.
        "max_gradient_norm",
        "batch_size",
        "learning_rate",
        "learning_rate_decay_factor",
        "label_states",
        "alpha",
        "reg"
    ]

    def __init__(  # pylint: disable=too-many-locals, too-many-arguments, too-many-branches, super-init-not-called, too-many-statements
            self,
            hparams,
            forward_only=False,
            num_samples=512,
            dtype=tf.float32):
        """Create the model.
        Args:
            hparams: Hyperparameters used to contruct the nerual network.
            num_samples: number of samples for sampled softmax.
            forward_only: if set, we do not construct the backward pass in the model.
            dtype: the data type to use to store internal variables.
        """
        self.hparams = hparams
        self.source_vocab_size = hparams.source_vocab_size
        self.target_vocab_size = hparams.target_vocab_size
        self.buckets = hparams.buckets
        self.size = hparams.size
        self.num_layers = hparams.num_layers
        self.max_gradient_norm = hparams.max_gradient_norm
        self.batch_size = hparams.batch_size
        self.learning_rate = hparams.learning_rate
        self.learning_rate_decay_factor = hparams.learning_rate_decay_factor
        self.learning_rate_op = tf.Variable(
            float(self.learning_rate), trainable=False, dtype=dtype)
        self.learning_rate_decay_op = self.learning_rate_op.assign(
            self.learning_rate_op * hparams.learning_rate_decay_factor)
        self.dropout_rate = hparams.dropout_rate
        self.label_states = hparams.label_states
        self.alpha = hparams.alpha  # Get coefficient for combined loss function
        self.global_step = tf.Variable(0, trainable=False)
        self.reg = hparams.reg

        logging.info(
            "Initializing model with hparams: %s" % str(self.hparams.to_json()))

        size = hparams.size
        buckets = hparams.buckets
        dropout_rate = hparams.dropout_rate
        num_layers = hparams.num_layers

        # If we use sampled softmax, we need an output projection.
        output_projection = None
        softmax_loss_function = None
        # Sampled softmax only makes sense if we sample less than vocabulary size.
        if num_samples > 0 and num_samples < self.target_vocab_size:
            w_t = tf.get_variable(
                "proj_w", [self.target_vocab_size, hparams.size], dtype=dtype)
            w = tf.transpose(w_t)
            b = tf.get_variable("proj_b", [self.target_vocab_size], dtype=dtype)
            output_projection = (w, b)

            def sampled_loss(labels, logits):
                """Sampleed loss function."""
                labels = tf.reshape(labels, [-1, 1])
                # We need to compute the sampled_softmax_loss using 32bit floats to
                # avoid numerical instabilities.
                local_w_t = tf.cast(w_t, tf.float32)
                local_b = tf.cast(b, tf.float32)
                local_inputs = tf.cast(logits, tf.float32)
                return tf.cast(
                    tf.nn.sampled_softmax_loss(local_w_t, local_b, labels,
                                               local_inputs, num_samples,
                                               self.target_vocab_size), dtype)

            softmax_loss_function = sampled_loss

        # Create the internal multi-layer cell for our RNN.
        def single_cell():
            """internal single cell for RNN"""
            cell_cls_name = "%sCell" % hparams.rnn_cell
            cell_cls = getattr(tf.contrib.rnn, cell_cls_name)
            ret = cell_cls(hparams.size)
            ret = tf.nn.rnn_cell.DropoutWrapper(
                ret,
                input_keep_prob=dropout_rate,
                output_keep_prob=dropout_rate)
            return ret

        self._fp_tensors = []

        # The seq2seq function: we use embedding for the input and attention.
        def seq2seq_f(encoder_inputs, decoder_inputs, do_decode):
            """Sequence to sequence function."""
            cell = single_cell()
            if num_layers > 1:
                cell = tf.contrib.rnn.MultiRNNCell(
                    [single_cell() for _ in range(num_layers)])
            outputs, encoder_state, decoder_state = embedding_attention_seq2seq(
                encoder_inputs,
                decoder_inputs,
                cell,
                num_encoder_symbols=hparams.source_vocab_size,
                num_decoder_symbols=hparams.target_vocab_size,
                embedding_size=size,
                output_projection=output_projection,
                feed_previous=do_decode,
                dtype=dtype)
            self._fp_tensors.append(encoder_state)
            return outputs, decoder_state

        def pred_net(bucket_id, encoder_labels):
            """Build prediction network."""

            fp_tensor = self.get_fingerprint_tensor(bucket_id)

            # Prediction network definition.
            pred_net_cls = getattr(pred_models, hparams.pred_net_type)
            pred = pred_net_cls(hparams)(fp_tensor, reuse=(bucket_id > 0))

            # Prediction loss.
            loss_cls = getattr(losses, hparams.loss_type)
            loss_sup = loss_cls(hparams)(
                input_tensor=pred, label_tensor=encoder_labels)

            # Metrics.
            metric_cls = getattr(metrics, hparams.metric_type)
            metric_ops = metric_cls(hparams)(
                input_tensor=pred, label_tensor=encoder_labels)

            return pred, loss_sup, metric_ops

        # Feeds for inputs.
        self.encoder_inputs = []
        self.decoder_inputs = []
        self.target_weights = []
        self.encoder_labels = []
        for i in range(buckets[-1][0]):  # Last bucket is the biggest one.
            self.encoder_inputs.append(
                tf.placeholder(
                    tf.int32, shape=[None], name="encoder{0}".format(i)))
        if self.label_states:
            self.encoder_labels.append(
                tf.placeholder(
                    tf.float32 if self.reg else tf.int32, shape=[None], name="label{0}".format(0)))
        for i in range(buckets[-1][1] + 1):
            self.decoder_inputs.append(
                tf.placeholder(
                    tf.int32, shape=[None], name="decoder{0}".format(i)))
            self.target_weights.append(
                tf.placeholder(dtype, shape=[None], name="weight{0}".format(i)))
        # Our targets are decoder inputs shifted by one.
        targets = [
            self.decoder_inputs[i + 1]
            for i in range(len(self.decoder_inputs) - 1)
        ]

        # Training outputs and losses.
        if forward_only:
            self.outputs, self.losses = tf.contrib.legacy_seq2seq.model_with_buckets(
                self.encoder_inputs,
                self.decoder_inputs,
                targets,
                self.target_weights,
                buckets,
                lambda x, y: seq2seq_f(x, y, True),
                softmax_loss_function=softmax_loss_function)
            # If we use output projection, we need to project outputs for decoding.
            if output_projection is not None:
                for b in range(len(buckets)):
                    self.outputs[b] = [
                        tf.matmul(output, output_projection[0]) +
                        output_projection[1] for output in self.outputs[b]
                    ]

        else:
            self.outputs, self.losses = tf.contrib.legacy_seq2seq.model_with_buckets(
                self.encoder_inputs,
                self.decoder_inputs,
                targets,
                self.target_weights,
                buckets,
                lambda x, y: seq2seq_f(x, y, False),
                softmax_loss_function=softmax_loss_function)

        if self.label_states:
            self.loss_supervised = [None] * len(buckets)
            self.pred = [None] * len(buckets)
            self.sup_metrics = [None] * len(buckets)
            for bucket_id in range(len(buckets)):
                self.pred[bucket_id], self.loss_supervised[bucket_id],\
                self.sup_metrics[bucket_id] = (
                    pred_net(bucket_id, self.encoder_labels[0]))

        # Gradients and SGD update operation for training the model.
        params = tf.trainable_variables()
        self.summary_ops = []
        self.test_summary_ops = []
        # TODO(zhengxu): This is a workaround to avoid test summary initialization 
        # from train script.
        # Append test summaries.
        self.test_summary_ops = [
            self.get_em_acc_op(bucket_id) for bucket_id in range(len(buckets))
        ]
        if not forward_only:
            self.gradient_norms = []
            self.updates = []
            lr_summary_op = tf.summary.scalar("learning_rate",
                                              self.learning_rate_op)
            opt = tf.train.GradientDescentOptimizer(self.learning_rate_op)
            for b in range(len(buckets)):
                loss = self.losses[b] if hparams.use_recovery else 0.
                if self.label_states:
                    loss += self.alpha * self.loss_supervised[b]
                gradients = tf.gradients(loss, params)
                clipped_gradients, norm = tf.clip_by_global_norm(
                    gradients, hparams.max_gradient_norm)
                self.gradient_norms.append(norm)
                self.updates.append(
                    opt.apply_gradients(
                        zip(clipped_gradients, params),
                        global_step=self.global_step))
                bucket_summary_ops = [
                    # Global norm in each buckets.
                    tf.summary.scalar("global_norm_%d" % b, norm),
                    # Unsupervised (Recovery) Loss in each buckets.
                    tf.summary.scalar("loss_unsup_%d" % b, self.losses[b]),
                    # Learning rate summary op.
                    lr_summary_op
                ]
                if self.label_states:
                    bucket_summary_ops.append([
                        # Supervised (Classification) Loss.
                        tf.summary.scalar("loss_sup_%d" % b,
                                          self.loss_supervised[b]),
                        # Total loss (Multi-task loss).
                        tf.summary.scalar("total_loss_%d" % b, loss)
                    ] + [
                        # Supervised task evaluation metric.
                        tf.summary.scalar("%s_%d" % (k, b), v)
                        for k, v in self.sup_metrics[b].items()
                    ])
                self.summary_ops.append(tf.summary.merge(bucket_summary_ops))

        variables_to_restore = [
            v for v in tf.global_variables() if v.name.split('/')[0] != 'pred'
        ]
        self.saver_sup = tf.train.Saver(
            tf.global_variables(), save_relative_paths=True)
        self.saver_unsup = tf.train.Saver(
            variables_to_restore, save_relative_paths=True)

#
#   Model load and save.
#

    @classmethod
    def load_model_from_files(  # pylint: disable=too-many-arguments
            cls,
            model_file,
            checkpoint_dir,
            forward_only,
            restore_all_vars=True,
            pretrain_model_path="",
            hparams_dict={},
            sess=None):
        """Load model from file."""
        hparams = build_base_hparams()
        if os.path.exists(model_file):
            logging.info(
                "Loading seq2seq model definition from %s..." % model_file)
            with open(model_file, "r") as fobj:
                model_dict = json.load(fobj)
            model_dict["buckets"] = [
                tuple(_bucket) for _bucket in model_dict["buckets"]
            ]
            hparams.set_from_map(model_dict)
        else:
            logging.info("Initializing a fresh training...")
        hparams.set_from_map(hparams_dict)
        model = cls(hparams, forward_only)
        # Load model weights.
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        sess = sess or tf.get_default_session()
        if pretrain_model_path:
            if tf.gfile.IsDirectory(pretrain_model_path):
                pretrain_model_path = os.path.join(pretrain_model_path,
                                                   "weights")
                pretrain_ckpt = tf.train.get_checkpoint_state(
                    pretrain_model_path)
                pretrain_model_path = pretrain_ckpt.model_checkpoint_path
            logging.info("Loading pretrained model weights from checkpoint: %s"
                         % pretrain_model_path)
            if restore_all_vars:
                model.saver_sup.restore(sess, pretrain_model_path)
            else:
                # This is an ugly workaround to load pretrain model for part of
                # the models.
                sess.run(tf.global_variables_initializer())
                model.saver_unsup.restore(sess, pretrain_model_path)
        elif ckpt:
            logging.info("Loading model weights from checkpoint_dir: %s" %
                         checkpoint_dir)
            if restore_all_vars:
                model.saver_sup.restore(sess, ckpt.model_checkpoint_path)
            else:
                sess.run(tf.global_variables_initializer())
                model.saver_unsup.restore(sess, ckpt.model_checkpoint_path)
        else:
            logging.info("Initialize fresh parameters...")
            sess.run(tf.global_variables_initializer())
        return model

    @classmethod
    def load_model_from_dir(  # pylint: disable=too-many-arguments
            cls,
            train_dir,
            forward_only,
            restore_all_vars=True,
            pretrain_model_path="",
            hparams_dict={},
            sess=None):
        """Load model definition from train_dir/model.json and train_dir/weights."""
        model_file = os.path.join(train_dir, "model.json")
        checkpoint_dir = os.path.join(train_dir, "weights/")
        return cls.load_model_from_files(
            model_file,
            checkpoint_dir,
            forward_only,
            restore_all_vars,
            pretrain_model_path=pretrain_model_path,
            hparams_dict=hparams_dict,
            sess=sess)

    def save_model_to_files(  # pylint: disable=too-many-arguments
            self,
            model_file,
            checkpoint_file,
            save_all_vars,
            sess=None,
            verbose=False):
        """Save all the model hyper-parameters to a json file."""
        if verbose:
            logging.info("Save model defintion to %s..." % model_file)
        model_dict = {
            key: getattr(self, key)
            for key in self.MODEL_PARAMETER_FIELDS
        }
        with open(model_file, "w") as fobj:
            json.dump(model_dict, fobj)
        checkpoint_dir = os.path.dirname(checkpoint_file)
        if os.path.exists(checkpoint_dir):
            if verbose:
                logging.info("Save weights to %s..." % checkpoint_file)
            sess = sess or tf.get_default_session()
            if save_all_vars:
                self.saver_sup.save(
                    sess, checkpoint_file, global_step=self.global_step)
            else:
                self.saver_unsup.save(
                    sess, checkpoint_file, global_step=self.global_step)
        elif verbose:
            logging.info("Skip save weights to %s since the dir does not exist."
                         % checkpoint_dir)

    def save_model_to_dir(self,
                          train_dir,
                          save_all_vars,
                          sess=None,
                          verbose=False):
        """Save model definition and weights to train_dir/model.json and train_dir/checkpoints/"""
        model_file = os.path.join(train_dir, "model.json")
        checkpoint_dir = os.path.join(train_dir, "weights")
        checkpoint_file = os.path.join(checkpoint_dir, "weights-ckpt")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        self.save_model_to_files(
            model_file,
            checkpoint_file,
            save_all_vars,
            sess=sess,
            verbose=verbose)

    def get_fingerprint_tensor(self, bucket_id):
        """Get concatenated fingerprint tensors."""
        fp_tensors = self._fp_tensors[bucket_id]

        def single_cell_process(cell_states):
            # If it is a LSTM cell, we concat both cell and hidden state.
            if isinstance(cell_states, tf.contrib.rnn.LSTMStateTuple):
                cell_states = tf.concat([cell_states.c, cell_states.h], axis=1)
            # Otherwise, just return the cell states directly.
            return cell_states

        if isinstance(fp_tensors, (list, tuple)):
            # Multi-layer RNN.
            fp_tensors = [single_cell_process(f) for f in fp_tensors]
            fp_tensor = tf.concat(fp_tensors, axis=1)
        else:
            # Single-layer RNN.
            fp_tensor = single_cell_process(fp_tensors)
        return fp_tensor

    def get_batch(self,
                  data,
                  bucket_id,
                  label_states=False,
                  sample_with_replacement=True):
        """Get a random batch of data from the specified bucket, prepare for step.

        To feed data in step(..) it must be a list of batch-major vectors, while
        data here contains single length-major cases. So the main logic of this
        function is to re-index data cases to be in the proper format for feeding.

        Args:
            data: a tuple of size len(self.buckets) in which each element contains
                lists of pairs of input and output data that we use to create a batch.
            bucket_id: integer, which bucket to get the batch for.
            label_states: if label is provided.
            sample_with_replacement: if examples are sampled with replacement.
            
        Note:
            It is suggested to use `sample_with_replacement` as True in training 
            but False for evaluation.

        Returns:
            The triple (encoder_inputs, decoder_inputs, target_weights) for
            the constructed batch that has the proper format to call step(...) later.
        """
        encoder_size, decoder_size = self.buckets[bucket_id]
        encoder_inputs, decoder_inputs = [], []
        if label_states:
            labels = []

        if sample_with_replacement:
            raw_sampled_batch = [
                random.choice(data[bucket_id]) for _ in range(self.batch_size)
            ]
        else:
            sample_size = min(len(data[bucket_id]), self.batch_size)
            raw_sampled_batch = random.sample(data[bucket_id], sample_size)
        real_batch_size = len(raw_sampled_batch)

        # Get a random batch of encoder and decoder inputs from data,
        # pad them if needed, reverse encoder inputs and add GO to decoder.
        for raw_sample in raw_sampled_batch:
            if label_states:
                encoder_input, decoder_input, label = raw_sample
            else:
                encoder_input, decoder_input = raw_sample
            # Encoder inputs are padded and then reversed.
            encoder_pad = [PAD_ID] * (encoder_size - len(encoder_input))
            encoder_inputs.append(list(reversed(encoder_input + encoder_pad)))
            # Decoder inputs get an extra "GO" symbol, and are padded then.
            decoder_pad_size = decoder_size - len(decoder_input) - 1
            decoder_inputs.append([GO_ID] + decoder_input +
                                  [PAD_ID] * decoder_pad_size)
            if label_states:
                labels.append(label)
        # Now we create batch-major vectors from the data selected above.
        batch_encoder_inputs, batch_decoder_inputs, batch_weights = [], [], []
        if label_states:
            batch_labels = []

        # Batch encoder inputs are just re-indexed encoder_inputs.
        for length_idx in range(encoder_size):
            batch_encoder_inputs.append(
                np.array(
                    [
                        encoder_inputs[batch_idx][length_idx]
                        for batch_idx in range(real_batch_size)
                    ],
                    dtype=np.int32))
        # Batch decoder inputs are re-indexed decoder_inputs, we create weights.
        for length_idx in range(decoder_size):
            batch_decoder_inputs.append(
                np.array(
                    [
                        decoder_inputs[batch_idx][length_idx]
                        for batch_idx in range(real_batch_size)
                    ],
                    dtype=np.int32))
            # Create target_weights to be 0 for targets that are padding.
            batch_weight = np.ones(real_batch_size, dtype=np.float32)
            for batch_idx in range(real_batch_size):
                # We set weight to 0 if the corresponding target is a PAD symbol.
                # The corresponding target is decoder_input shifted by 1 forward.
                if length_idx < decoder_size - 1:
                    target = decoder_inputs[batch_idx][length_idx + 1]
                if length_idx == decoder_size - 1 or target == PAD_ID:
                    batch_weight[batch_idx] = 0.0
            batch_weights.append(batch_weight)
        if label_states:
            batch_labels.append(np.array(labels, dtype=np.int32))
            return batch_encoder_inputs, batch_decoder_inputs, batch_labels, batch_weights
        return batch_encoder_inputs, batch_decoder_inputs, None, batch_weights

    def step(  # pylint: disable=too-many-locals, too-many-arguments, too-many-branches, arguments-differ
            self,
            session,
            encoder_inputs,
            decoder_inputs,
            target_weights,
            bucket_id,
            summary_writer,
            forward_only,
            output_encoder_states=False,
            encoder_labels=None):
        """Run a step of the model feeding the given inputs.

        Args:
            session: tensorflow session to use.
            encoder_inputs: list of numpy int vectors to feed as encoder inputs.
            encoder_labels: list of numpy int vectors to feed as encoder labels.
            decoder_inputs: list of numpy int vectors to feed as decoder inputs.
            target_weights: list of numpy float vectors to feed as target weights.
            bucket_id: which bucket of the model to use.
            forward_only: whether to do the backward step or only forward.

        Returns:
            A triple consisting of gradient norm (or None if we did not do backward),
            average perplexity, and the outputs.

        Raises:
            ValueError: if length of encoder_inputs, decoder_inputs, or
                target_weights disagrees with bucket size for the specified bucket_id.
        """
        # Check if the sizes match.
        encoder_size, decoder_size = self.buckets[bucket_id]
        if len(encoder_inputs) != encoder_size:
            raise ValueError(
                "Encoder length must be equal to the one in bucket,"
                " %d != %d." % (len(encoder_inputs), encoder_size))
        if len(decoder_inputs) != decoder_size:
            raise ValueError(
                "Decoder length must be equal to the one in bucket,"
                " %d != %d." % (len(decoder_inputs), decoder_size))
        if len(target_weights) != decoder_size:
            raise ValueError(
                "Weights length must be equal to the one in bucket,"
                " %d != %d." % (len(target_weights), decoder_size))

        # Input feed: encoder inputs, decoder inputs, target_weights, as provided.
        input_feed = {}
        for l in range(encoder_size):
            input_feed[self.encoder_inputs[l].name] = encoder_inputs[l]
        for l in range(decoder_size):
            input_feed[self.decoder_inputs[l].name] = decoder_inputs[l]
            input_feed[self.target_weights[l].name] = target_weights[l]
        if encoder_labels is not None:
            input_feed[self.encoder_labels[0].name] = encoder_labels[0]

        batch_size = encoder_inputs[0].shape[0]
        # Since our targets are decoder inputs shifted by one, we need one more.
        last_target = self.decoder_inputs[decoder_size].name
        input_feed[last_target] = np.zeros([batch_size], dtype=np.int32)

        # Output feed: depends on whether we do a backward step or not.
        if not forward_only:
            output_feed = [
                self.updates[bucket_id],  # Update Op that does SGD.
                self.gradient_norms[bucket_id],  # Gradient norm.
                self.losses[bucket_id]
            ] + ([self.loss_supervised[bucket_id]]
                 if encoder_labels is not None else [])  #Loss for this batch.
        else:
            output_feed = [self.losses[bucket_id]]
            if encoder_labels:
                output_feed.append(self.sup_metrics[bucket_id])
                if output_encoder_states:
                    output_feed.append(self.pred[bucket_id])
            # Loss for this batch, supervised accuracy for this batch, supervised training predictions for this batch.
            for l in range(decoder_size):  # Output logits.
                output_feed.append(self.outputs[bucket_id][l])
            if output_encoder_states:
                output_feed.extend(self._fp_tensors[bucket_id])
        if not forward_only:
            try:
                outputs, summary = session.run(
                    [output_feed, self.summary_ops[bucket_id]], input_feed)
            except:
                print(input_feed)
                raise
            if summary_writer is not None:
                summary_writer.add_summary(summary, self.global_step.eval())
            if encoder_labels is None:
                # gradient norm, loss, no outputs.
                return outputs[1], outputs[2], 0.
            # gradient norm, loss, loss_sup, no outputs.
            return outputs[1], outputs[2], outputs[3]
        # TODO(xiaoyu): This is an ugly workaround! The issue is because the 
        # summary op only works with train `network` and test `step`.
        # So when performing prediction only, we need to skip train network.
        if encoder_labels and len(self.summary_ops):
            outputs, test_summary = session.run(
                [output_feed, self.summary_ops[bucket_id]], input_feed)
        else:
            outputs = session.run(output_feed, input_feed)
        if summary_writer is not None and encoder_labels:
            summary_writer.add_summary(test_summary, self.global_step.eval())
        if output_encoder_states:
            if encoder_labels is not None:
                # No gradient norm, loss, acc, pred, outputs, fingerprint
                return None, outputs[0], outputs[1], outputs[2], outputs[3:(
                    3 + decoder_size)], outputs[(3 + decoder_size):]
            # No gradient norm, loss, outputs, encoder fixed vector.
            return None, outputs[0], outputs[1:1 + decoder_size], outputs[
                1 + decoder_size:]
        if encoder_labels is None:
            return None, outputs[0], None, outputs[1:1 + decoder_size]
        # No gradient norm, loss, outputs.
        return None, outputs[0], outputs[1], outputs[2:2 + decoder_size]

    def get_em_acc_op(self, bucket_id):
        """Create a em_acc_op."""
        with tf.name_scope("EMAcc_%d" % bucket_id):
            # [sequence_length, batch_size]
            input_ph = tf.placeholder(tf.int64, shape=(None, None))
            # [sequence_length, batch_size, vocab_size]
            output_ph = tf.placeholder(
                tf.float32, shape=(None, None, self.target_vocab_size))
            input_op = tf.reverse_v2(input_ph, axis=[0])
            output_op = tf.argmax(output_ph, axis=2)

            def replace_eos_with_pad(in_seq):
                """Replace all tokens after EOS in sequence with PAD."""
                out_seq = in_seq.copy()
                for idx in range(in_seq.shape[-1]):
                    eos_list = in_seq[:, idx].reshape(in_seq.shape[0]).tolist()
                    eos_idx = eos_list.index(
                        EOS_ID) if EOS_ID in eos_list else -1
                    out_seq[eos_idx:, idx] = PAD_ID
                return out_seq

            eos_op = tf.py_func(replace_eos_with_pad, [output_op], tf.int64)
            equal_op = tf.equal(
                tf.reduce_sum(tf.abs(input_op - eos_op), axis=0), 0)
            em_acc_op = tf.reduce_mean(tf.cast(equal_op, tf.float32), axis=0)
            summary_op = tf.summary.scalar("EMAccSummary", em_acc_op)
        return input_ph, output_ph, em_acc_op, summary_op


class FingerprintFetcher(object):
    """Seq2seq fingerprint fetcher for the seq2seq fingerprint."""

    def __init__(self, model_dir, vocab_path, sess=None):
        """Initialize a fingerprint fetcher for the seq2seq-fingerprint."""
        self.model_dir = model_dir
        self.vocab_path = vocab_path

        # Load tensorflow model
        self.model = Seq3SeqModel.load_model_from_dir(
            self.model_dir, True, sess=sess)
        self.model.batch_size = 1

        # Load vocabulary.
        self.vocab, self.rev_vocab = initialize_vocabulary(self.vocab_path)

    def get_bucket_id(self, token_ids):
        """Determine which bucket should the smile string be placed in."""
        _buckets = self.model.buckets
        bucket_id = len(_buckets) - 1
        for i, bucket in enumerate(_buckets):
            if bucket[0] >= len(token_ids):
                bucket_id = i
                break
        return bucket_id

    def decode(self, smile_string, labels, sess=None):  # pylint: disable=too-many-locals
        """Input a smile string and will output the fingerprint and predicted output."""
        token_ids = sentence_to_token_ids(
            tf.compat.as_bytes(smile_string),
            self.vocab,
            tokenizer=smile_tokenizer,
            normalize_digits=False)
        bucket_id = self.get_bucket_id(token_ids)
        # Get a 1-element batch to feed the sentence to the model.
        encoder_inputs, decoder_inputs, label, target_weights = self.model.get_batch(
            {
                bucket_id: [(token_ids, [], int(labels))]
            }, bucket_id, True)
        # Get output logits for the sentence.
        sess = sess or tf.get_default_session()
        # No gradient norm, loss, acc, pred, outputs, fingerprint
        _, _, acc, pred, output_logits, fps = self.model.step(
            sess,
            encoder_inputs,
            decoder_inputs,
            target_weights,
            bucket_id,
            None,
            True,
            True,
            encoder_labels=label)
        # This is a greedy decoder - outputs are just argmaxes of output_logits.
        outputs = [int(np.argmax(logit, axis=1)) for logit in output_logits]
        # If there is an EOS symbol in outputs, cut them at that point.
        if EOS_ID in outputs:
            outputs = outputs[:outputs.index(EOS_ID)]
        output_smile = "".join(
            [tf.compat.as_str(self.rev_vocab[output]) for output in outputs])
        seq2seq_fp = np.concatenate(tuple([fp.flatten() for fp in fps]))
        # return the fingerprint and predicted smile.
        return seq2seq_fp, output_smile, acc
