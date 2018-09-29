"""Evaluation utilities."""


import numpy as np
import tensorflow as tf


def make_summary(name, value):
  """Creates a tf.Summary proto with the given name and value."""
  summary = tf.Summary()
  val = summary.value.add()
  val.tag = str(name)
  val.simple_value = float(value)
  return summary


class BaseAccumulator(object):
    def __init__(self):
        self._num = 0
        self._val = np.float64(0)
        
    @property
    def value(self):
        return self._val

    @property
    def sample_num(self):
        return self._num
    
    @property
    def args(self):
        raise NotImplementedError


class AvgAccumulator(BaseAccumulator):
    def accumulate(self, sample_num, val):
        self._val = self._val * self._num + val * sample_num
        self._num += sample_num
        self._val /= self._num
        
    @property
    def args(self):
        return self.sample_num, self.value

class AccumulatorWithBuckets(object):
    def __init__(self):
        self._acms = {}
     
    def get(self, key, bucket_id):
        acm = self._acms.get(key, [])
        if len(acm) < bucket_id + 1:
            acm.append(AvgAccumulator())
            self._acms[key] = acm
        return acm[bucket_id]

    @property
    def acumulators(self):
        return self._acms


def add_eval_summary(
    summary_writer,
    global_step,
    accumulator_dict,
    summary_scope="eval"):
    """Add eval summary for a specific global step.
    
    Args:
        accumulator_dict: a key-accumulator_list mapping.
    """
    
    # Add overall summary.
    overall_acms = {}
    for k in accumulator_dict:
        overall_acms[k] = accumulator_dict[k][0].__class__()
        for bucket_id, acm in enumerate(accumulator_dict[k]):
            overall_acms[k].accumulate(*acm.args)
            # Add bucket-wise summary.
            summary_writer.add_summary(
                make_summary(
                    "%s/bucket_%d/%s" % (summary_scope, bucket_id, k), acm.value),
                global_step)
        summary_writer.add_summary(
            make_summary(
                "%s/%s" % (summary_scope, k), overall_acms[k].value),
            global_step)
    return overall_acms
    

