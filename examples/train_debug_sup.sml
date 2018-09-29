local job_tmpl = import "job.libsm";
local train_job = job_tmpl {
  params: {
    train_dir: "/tmp/debug-gru-2-128-sup/",
    train_data_file: "tests/data/debug.tokens",
    eval_data_file: "tests/data/debug.tokens",
    label_file: "tests/data/debug.label",
    batch_size: 5
  },
  args: {
    batch_size: $.params.batch_size,
    train_labels: $.params.label_file,
    test_labels: $.params.label_file,
    // pretrain_model_path: "/tmp/debug-gru-2-128/",
    summary_dir: "/tmp/debug-gru-2-128-sup-tbs/"
  },
  binary: "python train.py",
  args_extra: "%s %s %s" % [
      self.params.train_dir, self.params.train_data_file,
      self.params.eval_data_file],
  cmd: "%s %s %s" % [self.binary, self.args_extra, self.flag_string]
};
train_job
