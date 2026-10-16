"""Shared helpers for research scripts: import train_cifar, open a log under logs/."""
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.chdir(REPO)

import train_cifar as t  # noqa: E402


def open_log(name, args, script_path):
    """logs/research_<name>_<timestamp>_<pid>.log with the git commit, args and script source."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    path = os.path.join(REPO, "logs", f"research_{name}_{stamp}_{os.getpid()}.log")
    log = open(path, "x", encoding="utf-8", buffering=1)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "train_cifar.py", "research"],
                           capture_output=True, text=True).stdout.strip()
    log.write(f"# commit {commit}{' (dirty: ' + dirty.replace(chr(10), '; ') + ')' if dirty else ''}\n")
    log.write(f"# args {json.dumps(vars(args))}\n")
    log.write(f"# worker {os.environ.get('WORKER_NAME', '')} gpu {os.environ.get('CUDA_VISIBLE_DEVICES', '')}\n")
    with open(script_path, encoding="utf-8") as source:
        log.write("# source\n" + "".join("# | " + line for line in source) + "\n")
    print(f"log: {path}", flush=True)
    return log


def emit(log, **fields):
    fields.setdefault("time", round(time.time(), 1))
    log.write(json.dumps(fields) + "\n")


def new_model():
    import torch
    return t.CifarNet().cuda().to(memory_format=torch.channels_last)


def base_config(batch_size, name=None):
    return t.normalize_config(t.RUN_CONFIGS[name or f"baseline_bs{batch_size}"], key=None)
