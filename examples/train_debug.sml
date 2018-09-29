local job_tmpl = import "job.libsm";
local train_job = job_tmpl {
  params: {
    train_dir: "/tmp/debug-gru-2-128/",
    train_data_file: "tests/data/debug.tokens",
    eval_data_file: "tests/data/debug.tokens",
    batch_size: 5
  },
  hparams: {
    buckets: [[30, 30], [60, 60]]
  },
  args: {
    hparams: "'%s'" % std.toString($.hparams),
    batch_size: $.params.batch_size,
    max_step: 200,
    summary_dir: "/tmp/debug-gru-2-128-tbs/"
  },
  binary: "python train.py",
  args_extra: "%s %s %s" % [
      self.params.train_dir, self.params.train_data_file,
      self.params.eval_data_file],
  cmd: "%s %s %s" % [self.binary, self.args_extra, self.flag_string]
};
train_job
