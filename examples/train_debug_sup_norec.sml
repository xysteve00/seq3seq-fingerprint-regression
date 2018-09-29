local job_tmpl = import "job.libsm";
local train_job = job_tmpl {
  name: "seq3seq-fp-gru-3-128-logp-cls-batch-20",
  params: {
    train_dir: "/smile/gfs-nb/home/xiaoyu/pretrain/%s/" % $.name,
    train_data_file: "/smile/gfs-nb/home/xiaoyu/data/tokens/logp_train.tokens",
    eval_data_file: "/smile/gfs-nb/home/xiaoyu/data/tokens/logp_test.tokens",
    train_label_file: "/smile/gfs-nb/home/xiaoyu/data/classification/logp/logp_train_label.smi",
    test_label_file: "/smile/gfs-nb/home/xiaoyu/data/classification/logp/logp_test_label.smi",
    batch_size: 20,
  },
  hparams: {
    use_recovery: true,
    label_states: true,
    num_layers: 3,
    size: 128,
    alpha: 0.01,
  },
  args: {
    hparams: "'%s'" % std.toString($.hparams),
    batch_size: $.params.batch_size,
    train_labels: $.params.train_label_file,
    test_labels: $.params.test_label_file,
    pretrain_model_path: "/smile/gfs-nb/home/xiaoyu/pretrain/seq3seq-gru-3-128-logp-new-data",
    summary_dir: "/smile/gfs-nb/home/xiaoyu/pretrain/tbs/%s/" % $.name
  },
  binary: "python /smile/gfs-nb/home/xiaoyu/seq3seq-fingerprint/train.py",
  args_extra: "%s %s %s" % [
      self.params.train_dir, self.params.train_data_file,
      self.params.eval_data_file],
  cmd: "%s %s %s" % [self.binary, self.args_extra, self.flag_string]
};
train_job
