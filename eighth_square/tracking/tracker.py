import json
import os
import time


class MetricTracker:
    def __init__(self, run_dir):
        self.run_dir = run_dir
        os.makedirs(run_dir, exist_ok=True)
        self.path = os.path.join(run_dir, "metrics.jsonl")
        self._f = open(self.path, "w")
        self._start = time.time()

    def log(self, ret=None, **kwargs):
        row = kwargs
        if ret is not None:
            row["return"] = ret
        row.setdefault("t", round(time.time() - self._start, 3))
        self._f.write(json.dumps(row) + "\n")
        self._f.flush()

    def save_config(self, config, algo, env):
        with open(os.path.join(self.run_dir, "config.json"), "w") as f:
            json.dump({"algo": algo, "env": env, **config}, f, indent=2, default=str)

    def close(self):
        self._f.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
