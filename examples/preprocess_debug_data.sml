local job_tmpl = import "job.libsm";
local data_preprocess_job = job_tmpl {
  args: {
    smi_path: "tests/data/debug.smi",
    vocab_path: "tests/data/zinc.vocab",
    out_path: "tests/data/debug.tokens"
  },
  binary: "python data.py",
};
data_preprocess_job
