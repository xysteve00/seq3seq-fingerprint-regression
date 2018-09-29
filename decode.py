"""Decode fingerprint for the format."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
from tempfile import NamedTemporaryFile

import pandas as pd
import tensorflow as tf
import smile as sm
from smile import logging
from semisupervised.seq3seq_model import FingerprintFetcher

with sm.app.flags.Subcommand("sample", dest="action"):
    sm.app.flags.DEFINE_string(
        "model_dir",
        "",
        "model path of the seq3seq fingerprint.",
        required=True)
    sm.app.flags.DEFINE_string(
        "vocab_path",
        "",
        "Vocabulary path of the seq3seq fingerprint.",
        required=True)
    sm.app.flags.DEFINE_string(
        "data_path", "", "Data path of the sample.", required=True)
    sm.app.flags.DEFINE_integer("sample_size", 100,
                                "Sample size from the data file.")

with sm.app.flags.Subcommand("fp", dest="action"):
    sm.app.flags.DEFINE_string(
        "model_dir",
        "",
        "model path of the seq3seq fingerprint.",
        required=True)
    sm.app.flags.DEFINE_string(
        "vocab_path",
        "",
        "Vocabulary path of the seq3seq fingerprint.",
        required=True)
    sm.app.flags.DEFINE_string(
        "data_path", "", "Required data path.", required=True)
    sm.app.flags.DEFINE_string("output_path", "", "Output path of the sample.")
    sm.app.flags.DEFINE_string("dataset_headers", "smile",
                               "Comma separated headers for dataset.")
    # A know issue is when expected cls_thres is exactly 0..
    sm.app.flags.DEFINE_float("cls_thres", 0., "Classification threshold.")
    sm.app.flags.DEFINE_integer("repeat_num", 1, "Repeat times.")

FLAGS = sm.app.flags.FLAGS


def sample_smiles(data_path, sample_size):
    """Sample several sentences."""
    samples = set()
    with open(data_path) as fobj:
        lines = [_line for _line in fobj.readlines() if len(_line.strip())]
    data_size = len(lines)
    if data_size < sample_size:
        sample_size_ori = sample_size
        sample_size = data_size
        logging.warning("sample size (%d) is too large, "
                        "data size (%d) is used instead as the sample size" %
                        (sample_size_ori, data_size))
    while len(samples) < sample_size:
        samples.add(random.randrange(len(lines)))
    return [lines[index].strip() for index in list(samples)]


# TODO(xiaoyu): No longer working in supervised case.
def sample_decode():
    """Sample some samples from data file and print out the recovered string."""
    with tf.Session() as sess:
        sampled_smiles = sample_smiles(FLAGS.data_path, FLAGS.sample_size)
        fetcher = FingerprintFetcher(FLAGS.model_dir, FLAGS.vocab_path, sess)
        exact_match_num = 0
        for smile in sampled_smiles:
            _, output_smile = fetcher.decode(smile, 1)
            if output_smile == smile:
                exact_match_num += 1
            print(": %s\n> %s\n" % (smile, output_smile))
        print("Exact match count: %d/%d" % (exact_match_num,
                                            len(sampled_smiles)))


class SMISingleTaskReader(object):
    def __init__(self, dataset_cols=None, cls_thres=None, sep="\s+"):
        self._dataset_cols = dataset_cols
        self._cls_thres = cls_thres
        self._sep = sep

    def _read(self, file_path):
        return pd.read_csv(file_path, sep=self._sep, names=self._dataset_cols)

    def _postprocess(self, df):
        if self._cls_thres is not None:
            # Here we assume the last column is the value.
            pidxs = (df.iloc[:, -1] >= self._cls_thres)
            df.loc[pidxs, df.columns[-1]] = 1
            df.loc[pidxs != True, df.columns[-1]] = 0
        return df

    def read(self, file_path):
        df = self._read(file_path)
        return self._postprocess(df)

    def sample(self, file_path, n):
        df = self._read(file_path)
        return self._postprocess(df.sample(n))


def read_smiles(data_file, label_path):
    """Read all smile from a line-splitted file."""
    with open(data_file) as fobj, open(label_path) as lfobj:
        out_smiles = [_line.strip() for _line in fobj if _line.strip()]
        out_labels = [_line.strip() for _line in lfobj if _line.strip()]
        assert len(out_smiles) == len(out_smiles), "Data/label file mismatch."
    return zip(out_smiles, out_labels)


def fp_decode():
    """Decode ALL samples from the given data file and output to file."""
    # TODO(zhengxu): An ugly workaround to ensure the output path is optional.
    output_path = FLAGS.output_path or NamedTemporaryFile(delete=False).name
    with tf.Session() as sess, open(output_path, "w") as fout:
        all_smiles = SMISingleTaskReader(
            dataset_cols=FLAGS.dataset_headers.split(","),
            cls_thres=FLAGS.cls_thres).read(FLAGS.data_path)
        fetcher = FingerprintFetcher(FLAGS.model_dir, FLAGS.vocab_path, sess)
        exact_match_num = 0
        acc_count = 0
        # Note here the idx is the row index in the dataset.
        # So it might not be robust to dataset shuffle.
        for idx, (smile, label) in all_smiles.iterrows():
            seq2seq_fp, output_smile, acc = fetcher.decode(smile, label)
            acc_count += acc["accuracy"]
            if output_smile == smile:
                exact_match_num += 1
            if FLAGS.output_path:
                fout.write(" ".join([str(fp_bit)
                                     for fp_bit in seq2seq_fp]) + "\n")
            if idx % 200 == 0 and idx:
                logging.info("Progress: %d/%d" % (idx, len(all_smiles)))
        final_em_acc = float(exact_match_num) / len(all_smiles)
        final_acc = float(acc_count) / len(all_smiles)
        logging.info("Exact match count: %d/%d, %.4f%%" %
                     (exact_match_num, len(all_smiles), 100. * final_em_acc))
        logging.info("Accuracy: %d/%d, %.4f%%" % (acc_count, len(all_smiles),
                                                  100. * final_acc))
    return final_em_acc, final_acc


def main(_):
    """Entry function for the script."""
    if FLAGS.action == "sample":
        raise NotImplementedError
    elif FLAGS.action == "fp":
        result = []
        for _ in range(FLAGS.repeat_num):
            tf.reset_default_graph()
            result.append(fp_decode())
        em_acc, acc = zip(*result)
        logging.info("EM Acc: %s" % ", ".join(["%.8f" % x for x in em_acc]))
        logging.info("Acc: %s" % ", ".join(["%.8f" % x for x in acc]))
    else:
        print("Unsupported action: %s" % FLAGS.action)


if __name__ == "__main__":
    sm.app.run()
