local job_tmpl = import "job.libsm";
local train_job = job_tmpl {
  name: "~/pm2/regression/test/gru-3-256",
  params: {
    train_dir: "/home/xiaoyu/pretrain/%s" % $.name,
    train_data_file: "./tests/data/debug.tokens",
    eval_data_file: "./tests/data/debug.tokens",
    train_label_file:"./tests/data/debug.label",
    test_label_file:"./tests/data/debug.label",
    batch_size: 20,
    reset_lr: 0.5,
    reset_global_step: true,
  },
  hparams: {
    use_recovery: true,
    label_states: true,
    mlp_output_nums: [1],
    //target_vocab_size: 42,
    //source_vocab_size: 42,    
    num_layers: 3,
    size: 256,
    loss_type: "MSELoss",
    metric_type: "RGMetric",
    //learning_rate: 0.5,
  },
  args: {
    hparams: "'%s'" % std.toString($.hparams),
    batch_size: $.params.batch_size,
    reset_lr: $.params.reset_lr,
    reset_global_step: $.params.reset_global_step,
    train_labels: $.params.train_label_file,
    test_labels: $.params.test_label_file,
    //pretrain_model_path: "/smile/gfs-nb/home/xiaoyu/pretrain/debug-gru-3-128-suprec-logp-regression-reset-lr-1e-5",
    summary_dir: "/home/xiaoyu/pretrain/tbs/%s/" % $.name
  },
  binary: "python /home/xiaoyu/seq3seq-fingerprint-pm2/train.py",
  args_extra: "%s %s %s" % [
      self.params.train_dir, self.params.train_data_file,
      self.params.eval_data_file],
  cmd: "%s %s %s" % [self.binary, self.args_extra, self.flag_string]
};
train_job
