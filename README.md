# Seq3seq-fingerprint

This code implements seq3seq fingerprint method, which will be published soon in ACM BCB 2018.

## Installation

We use docker/nvidia-docker for both development and production running, mostly with docker image [utasmile/tensorflow:1.4.1-gpu](https://hub.docker.com/r/utasmile/tensorflow/).

For detailed instructions, you might want to refer to https://github.com/NVIDIA/nvidia-docker#quickstart。

## Dataset

ZINC is used for our experiments, which is a free database of commercially-available compounds for virtual screening. You can download ZINC datasets from [http://zinc.docking.org/](http://zinc.docking.org/)

## Get Started

### Very First Taste

To train a seq2seq model from a debug data with only 5 SMILE representations in [tests/data/debug.smi](tests/data/debug.smi).

Assume you clone the code under your `$HOME` dir and your current work directory is the code folder root.
Example usage:
```bash
nvidia-docker run --rm -it -v $HOME:$HOME -w $PWD
 utasmile/tensorflow:1.4.1-gpu smilectl examples/train_debug.sml runlocal
```

Example output:
```
I0615 21:19:51.832336 140140613834496 seq3seq_model.py:313] Initializing a fresh training...
I0615 21:19:51.835680 140140613834496 seq3seq_model.py:88] Initializing model with hparams: {"loss_ty
pe": "SparseXentLoss", "dropout_rate": 0.5, "learning_rate_decay_factor": 0.99, "pred_net_type": "MLP
", "buckets": [[30, 30], [60, 60], [90, 90]], "label_states": false, "target_vocab_size": 40, "batch_
size": 256, "source_vocab_size": 40, "use_recovery": true, "mlp_output_nums": [2], "num_layers": 3, "
metric_type": "BCMetric", "alpha": 1.0, "max_gradient_norm": 5.0, "learning_rate": 0.5, "size": 128}
I0615 21:21:09.036127 140140613834496 seq3seq_model.py:344] Initialize fresh parameters...
I0615 21:21:14.931850 140140613834496 train.py:178] Reading train data from tests/data/debug.tokens..
.
I0615 21:21:14.932178 140140613834496 train.py:180] Reading test data from tests/data/debug.tokens...
I0615 21:22:24.285237 140140613834496 train.py:240] global step 200 learning rate 0.500000 step-time 
0.143644 perplexity6.374212
I0615 21:22:24.285351 140140613834496 train.py:242]   loss_unsupervised: 1.7795321   loss_supervised:
 0.0
I0615 21:22:36.758291 140140613834496 train.py:279]   eval: bucket 0 perplexity 4.509512, em_acc 0.00
0000, acc_sup None
I0615 21:22:39.320552 140140613834496 train.py:279]   eval: bucket 1 perplexity 5.055138, em_acc 0.00
0000, acc_sup None
I0615 21:22:39.320717 140140613834496 train.py:255]   eval: empty bucket 2
I0615 21:22:58.310951 140140613834496 train.py:240] global step 400 learning rate 0.500000 step-time 
0.094621 perplexity3.357505
I0615 21:22:58.311058 140140613834496 train.py:242]   loss_unsupervised: 0.83335733   loss_supervised
: 0.0
I0615 21:23:08.053030 140140613834496 train.py:279]   eval: bucket 0 perplexity 3.724646, em_acc 0.00
0000, acc_sup None
I0615 21:23:08.102361 140140613834496 train.py:279]   eval: bucket 1 perplexity 3.564833, em_acc 0.00
0000, acc_sup None
I0615 21:23:08.102560 140140613834496 train.py:255]   eval: empty bucket 2
```

### More Examples

We includes few training examples using `smilectl` script in [examples/](examples/) folder.

- [train_debug_sup.sml](examples/train_debug_sup.sml): Example for training with a supervised task.

## References:
If our work is helpful for your research, please consider citing:
```bash
@inproceedings{zhang2018seq3seq,
  title={Seq3seq fingerprint: towards end-to-end semi-supervised deep drug discovery},
  author={Zhang, Xiaoyu and Wang, Sheng and Zhu, Feiyun and Xu, Zheng and Wang, Yuhong and Huang, Junzhou},
  booktitle={Proceedings of the 2018 ACM International Conference on Bioinformatics, Computational Biology, and Health Informatics},
  pages={404--413},
  year={2018},
  organization={ACM}
}

@article{xu2017seq2seqfingerprint,
  title={Seq2seq Fingerprint: An Unsupervised Deep Molecular Embedding for Drug Discovery},
  author={Zheng Xu, Sheng Wang, Feiyun Zhu, and Junzhou Huang},
  journal={BCB’17, Aug 2017, Boston, Massachusetts USA},
  year={2017}
}
```
