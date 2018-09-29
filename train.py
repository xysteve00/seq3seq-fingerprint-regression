"""Train fingerprint."""

from __future__ import absolute_import, division, print_function

import json
import math
import os
import sys
import time

import numpy as np
import smile as sm
import tensorflow as tf
from six.moves import range
from smile import logging

from semisupervised import seq3seq_model
from semisupervised.base_hparams import build_base_hparams
from semisupervised.eval_utils import (AccumulatorWithBuckets, AvgAccumulator,
                                       add_eval_summary)
from semisupervised.utils import EOS_ID

sm.app.flags.DEFINE_string(
    "model_dir", "", "model path of the seq3seq fingerprint.", required=True)
sm.app.flags.DEFINE_string(
    "train_data", "", "train_data for seq3seq fp train.", required=True)
sm.app.flags.DEFINE_string(
    "test_data", "", "test data path of the seq3seq fp eval.", required=True)
sm.app.flags.DEFINE_string("train_labels", "",
                           "train labels for seq3seq fp train.")
sm.app.flags.DEFINE_string("test_labels", "",
                           "test labels for seq3seq fp train.")
sm.app.flags.DEFINE_bool("reset_global_step", False,
                         "If set, reset global step to 0.")
sm.app.flags.DEFINE_float("reset_lr", 0.,
                          "Reset learning rate if specified larger than 0.")
sm.app.flags.DEFINE_integer("max_step", 0,
                            "Max steps to train, 0 for no limit.")
sm.app.flags.DEFINE_float("min_lr_threshold", 1e-7,
                          "minimum lr threshhold to stop training")
sm.app.flags.DEFINE_string(
    "hparams", "",
    "A JSON string which will override the original hparams for the model.")
sm.app.flags.DEFINE_string(
    "pretrain_model_path", "",
    "If specified, load the pretrain model from specific path.")
sm.app.flags.DEFINE_bool("restore_all_vars", False,
                         "restore from unsupervised model: False")
sm.app.flags.DEFINE_bool("save_all_vars", True,
                         "save unsupervised model: False else True")
sm.app.flags.DEFINE_integer("batch_size", 128,
                            "Batch size to use during training.")
sm.app.flags.DEFINE_integer("gpu", 0, "GPU device to use, default: 0.")
sm.app.flags.DEFINE_integer("steps_per_checkpoint", 200,
                            "How many training steps to do per checkpoint.")
sm.app.flags.DEFINE_string("summary_dir", "", "Summary dir.")

FLAGS = sm.app.flags.FLAGS


def read_data(source_path, buckets, max_size=None):
    """Read data from source and target files and put into buckets.

    Args:
        source_path: path to the files with token-ids for the source language.
        max_size: maximum number of lines to read, all other will be ignored;
            if 0 or None, data files will be read completely (no limit).

    Returns:
        data_set: a list of length len(_buckets); data_set[n] contains a list of
            (source, target) pairs read from the provided data files that fit
            into the n-th bucket, i.e., such that len(source) < _buckets[n][0] and
            len(target) < _buckets[n][1]; source and target are lists of token-ids.
    """
    data_set = [[] for _ in buckets]
    with tf.gfile.GFile(source_path, mode="r") as source_file:
        source = source_file.readline()
        counter = 0
        while source and (not max_size or counter < max_size):
            counter += 1
            if counter % 100000 == 0:
                logging.info("  reading data line %d" % counter)
                sys.stdout.flush()
            source_ids = [int(x) for x in source.split()]
            target_ids = [int(x) for x in source.split()]
            target_ids.append(EOS_ID)
            for bucket_id, (source_size, target_size) in enumerate(buckets):
                if len(source_ids) < source_size and len(
                        target_ids) < target_size:
                    data_set[bucket_id].append([source_ids, target_ids])
                    break
            source = source_file.readline()
    return data_set


def read_data_labels(source_path, label_path, reg_flag,buckets, max_size=None):  # pylint: disable=too-many-locals
    """Read data from source and target files and put into buckets.

    Args:
        source_path: path to the files with token-ids for the source language.
        label_path: path to the labels
        max_size: maximum number of lines to read, all other will be ignored;
            if 0 or None, data files will be read completely (no limit).

    Returns:
        data_set: a list of length len(_buckets); data_set[n] contains a list of
            (source, target) pairs read from the provided data files that fit
            into the n-th bucket, i.e., such that len(source) < _buckets[n][0] and
            len(target) < _buckets[n][1]; source and target are lists of token-ids.
    """
    data_set = [[] for _ in buckets]
    with tf.gfile.GFile(source_path) as source_file, tf.gfile.GFile(
            label_path) as label_file:  # pylint: disable=bad-continuation
        source = source_file.readline().strip()
        label = label_file.readline().strip()
        counter = 0
        while (source and label) and (not max_size or counter < max_size):
            counter += 1
            if counter % 100000 == 0:
                logging.info("  reading data line %d" % counter)
                sys.stdout.flush()
            source_ids = [int(x) for x in source.split()]
            target_ids = [int(x) for x in source.split()]
            target_ids.append(EOS_ID)
            for bucket_id, (source_size, target_size) in enumerate(buckets):
                if len(source_ids) < source_size and len(
                        target_ids) < target_size:
                    data_set[bucket_id].append(
                        [source_ids, target_ids,
                         float(label) if reg_flag else int(label)])
                    break
            source = source_file.readline().strip()
            label = label_file.readline().strip()
    return data_set


def eval_dataset(test_label_set,
                 model,
                 label_states,
                 test_writer=None,
                 sess=None):
    """Perform an evaluation on the test dataset."""

    sess = sess or tf.get_default_session()
    acms = AccumulatorWithBuckets()
    for bucket_id in range(len(test_label_set)):
        length_test_set = len(test_label_set[bucket_id])
        if length_test_set == 0:
            logging.info("  eval: empty bucket %d" % (bucket_id))
            continue

        batch_size = model.batch_size
        # Iterate all the data inside the bucket.
        for start_idx in range(0, length_test_set, batch_size):
            # TODO(zhengxu): Provide an option to eval a subset of each bucket for speed.
            tmp_data = [None] * len(test_label_set)
            actual_data_len = (
                min(length_test_set, start_idx + batch_size) - start_idx)
            tmp_data[bucket_id] = test_label_set[bucket_id][
                start_idx:start_idx + actual_data_len]
            encoder_inputs, decoder_inputs, eval_labels, target_weights = (
                model.get_batch(tmp_data, bucket_id, label_states))
            _, eval_loss, eval_acc_sup, output_logits = model.step(
                sess,
                encoder_inputs,
                decoder_inputs,
                target_weights,
                bucket_id,
                test_writer,
                forward_only=True,
                output_encoder_states=False,
                encoder_labels=eval_labels)
            if eval_acc_sup is not None:
                for idx in eval_acc_sup:
                    acms.get(idx, bucket_id).accumulate(actual_data_len,
                                                        eval_acc_sup[idx])
            if eval_loss is not None:
                acms.get("eval_loss", bucket_id).accumulate(
                    actual_data_len, eval_loss)
            input_ph, output_ph, em_acc_op, summary_op = model.test_summary_ops[
                bucket_id]
            em_acc, summary = sess.run(
                [em_acc_op, summary_op],
                feed_dict={
                    input_ph: np.array(encoder_inputs),
                    output_ph: np.array(output_logits)
                })
            if em_acc is not None:
                acms.get("em_acc", bucket_id).accumulate(
                    actual_data_len, em_acc)

        eval_ppx = math.exp(float(acms.get("eval_loss", bucket_id)
                                  .value)) if eval_loss < 300 else float("inf")
        logging.info(
            "  eval: bucket %d perplexity %.6f" % (bucket_id, eval_ppx))

        logging.info("  eval: " + ",".join([
            "%s %.6e " % (key, val[bucket_id].value)
            for key, val in acms.acumulators.items()
        ]))

    # Add summary and calculate the overall evaluation metrics.
    overall_acms = add_eval_summary(test_writer, model.global_step.eval(),
                                    acms.acumulators)
    logging.info("  eval: overall " + ", ".join(
        ["%s %.4e" % (k, v.value) for k, v in overall_acms.items()]))


def train(  # pylint: disable=too-many-locals,too-many-statements,too-many-arguments
        train_data, test_data, train_labels, test_labels, restore_all_vars,
        save_all_vars):
    """Train script."""
    model_dir = FLAGS.model_dir
    batch_size = FLAGS.batch_size

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True

    hparams_override = json.loads(FLAGS.hparams) if FLAGS.hparams else dict()
    # Override some hparams results.
    hparams_override["label_states"] = (bool(train_labels) and
                                        bool(test_labels))
    hparams_override["batch_size"] = hparams_override.get(
        "batch_size", batch_size)

    with tf.Session(config=config) as sess:
        with tf.device("/gpu:%d" % FLAGS.gpu):
            # Create model.
            model = seq3seq_model.Seq3SeqModel.load_model_from_dir(
                model_dir,
                False,
                restore_all_vars,
                pretrain_model_path=FLAGS.pretrain_model_path,
                hparams_dict=hparams_override,
                sess=sess)
        if FLAGS.reset_lr > 0.:
            logging.info("Resetting LR to %.10f..." % FLAGS.reset_lr)
            sess.run(model.learning_rate_op.assign(FLAGS.reset_lr))
        if FLAGS.reset_global_step:
            logging.info("Reset global step to 0.")
            sess.run(model.global_step.assign(0))

        buckets = model.buckets
        reg = model.reg
        alpha = model.alpha  # Get coefficient for combined loss function
        label_states = model.hparams.label_states

        # Read data into buckets and compute their sizes.
        if model.hparams.label_states:
            logging.info("Reading train data from %s..." % train_data)
            train_label_set = read_data_labels(train_data, train_labels, reg,
                                               buckets)
            logging.info("Reading test data from %s..." % test_data)
            test_label_set = read_data_labels(test_data, test_labels, reg, buckets)
        else:
            logging.info("Reading train data from %s..." % train_data)
            train_label_set = read_data(train_data, buckets)
            logging.info("Reading test data from %s..." % test_data)
            test_label_set = read_data(test_data, buckets)

        train_bucket_sizes = [
            len(train_label_set[b]) for b in range(len(buckets))
        ]
        train_total_size = float(sum(train_bucket_sizes))
        train_bucket_prob = [
            size / train_total_size for size in train_bucket_sizes
        ]

        # This is the training loop.
        step_time, loss = 0.0, 0.0
        current_step = 0
        previous_losses = []
        if FLAGS.summary_dir:
            train_writer = tf.summary.FileWriter(
                os.path.join(FLAGS.summary_dir, "train"), sess.graph)
            test_writer = tf.summary.FileWriter(
                os.path.join(FLAGS.summary_dir, "test"), sess.graph)
        else:
            logging.warning(
                "You do not specify any summary directory. Reliance on log file"
                " might be unstable and dangerous.")
            train_writer = None
            test_writer = None

        test_summary_ops = model.test_summary_ops

        while model.learning_rate_op.eval() > FLAGS.min_lr_threshold:
            # Choose a bucket according to data distribution. We pick a random number
            # in [0, 1] and use the corresponding interval in train_buckets_scale.
            bucket_id = np.random.choice(
                len(train_bucket_prob), p=train_bucket_prob)

            # Get a batch and make a step.
            start_time = time.time()

            encoder_inputs, decoder_inputs, labels, target_weights = model.get_batch(
                train_label_set, bucket_id, label_states)
            _, step_loss, step_loss_sup = model.step(  # pylint: disable=unused-variable
                sess, encoder_inputs, decoder_inputs, target_weights, bucket_id,
                train_writer, False, False, labels)
            step_time += (time.time() - start_time) / FLAGS.steps_per_checkpoint
            loss += (
                step_loss + alpha * step_loss_sup) / FLAGS.steps_per_checkpoint
            current_step += 1
            # Once in a while, we save checkpoint, print statistics, and run evals.
            if current_step % FLAGS.steps_per_checkpoint == 0:
                # Print statistics for the previous epoch.
                perplexity = math.exp(
                    float(loss)) if loss < 300 else float("inf")

                logging.info(
                    "global step %d learning rate %.6f step-time %.6f perplexity"
                    "%.6f" % (model.global_step.eval(),
                              model.learning_rate_op.eval(), step_time,
                              perplexity))
                logging.info("  loss_unsupervised: %s   loss_supervised: %s" %
                             (str(step_loss), str(step_loss_sup)))
                # Decrease learning rate if no improvement was seen over last 3 times.
                if len(previous_losses) > 2 and loss > max(
                        previous_losses[-3:]):
                    sess.run(model.learning_rate_decay_op)
                previous_losses.append(loss)
                # Save checkpoint and zero timer and loss.
                model.save_model_to_dir(model_dir, save_all_vars, sess=sess)
                step_time, loss = 0.0, 0.0
                # Run a full evaluation on the test dataset.
                eval_dataset(
                    test_label_set,
                    model,
                    label_states,
                    test_writer=test_writer,
                    sess=sess)
                sys.stdout.flush()
            if FLAGS.max_step and current_step >= FLAGS.max_step:
                break


def main(_):
    """Entry function for the script."""
    train(FLAGS.train_data, FLAGS.test_data, FLAGS.train_labels,
          FLAGS.test_labels, FLAGS.restore_all_vars, FLAGS.save_all_vars)


if __name__ == "__main__":
    sm.app.run()
