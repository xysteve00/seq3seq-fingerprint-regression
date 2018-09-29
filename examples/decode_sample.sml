local job_tmpl = import "job.libsm";
local decode_job = job_tmpl {
  params: {
    model_dir: "/tmp/debug-gru-2-128-sup/",
    vocab_path: "tests/data/zinc.vocab",
    data_path: "tests/data/debug.smi"
  },
  script_command: "fp",
  args: {
    dataset_headers: "smile,logp",
    cls_thres: 1.88
  },
  binary: "python decode.py",
  args_extra: "%s %s %s %s" % [
      self.script_command, self.params.model_dir, 
      self.params.vocab_path, self.params.data_path],
  cmd: "%s %s %s" % [self.binary, self.args_extra, self.flag_string]
};
decode_job
