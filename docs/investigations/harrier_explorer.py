"""Noise-controlled config explorer for QNN EP on the harrier INT8 HTP context binary.

For each candidate configuration:
  - 3 fresh process invocations (so JIT/session-init effects don't mix across configs)
  - Each invocation does WARMUP iters discarded + N timed iters
  - Returns trimmed mean (drop top 10% and bottom 5%), median, p05, p25, p50, p75, p95, p99, min, max
  - Aggregates per-config across the 3 invocations: median-of-medians, IQR

Outputs a JSON list of result dicts to --out.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


HARNESS_RUNNER = r"""
import argparse, json, time, hashlib, sys
import numpy as np
import onnxruntime as ort
import onnxruntime_qnn as qnn_ep

p = argparse.ArgumentParser()
p.add_argument('--model-dir', required=True)
p.add_argument('--iters', type=int, default=200)
p.add_argument('--warmup', type=int, default=20)
p.add_argument('--label', required=True)
p.add_argument('--out', required=True)
p.add_argument('--seq-len', type=int, default=512)
p.add_argument('--sess-opt', action='append', default=[],
               help='session-level EP option, key=value')
p.add_argument('--run-opt', action='append', default=[],
               help='run-time option (RunOptions add_run_config_entry), key=value')
p.add_argument('--so-entry', action='append', default=[],
               help='session option add_session_config_entry, key=value')
args = p.parse_args()

import os
from pathlib import Path
model_dir = Path(args.model_dir)
model_path = model_dir / 'model.onnx'

reg = 'QNNExecutionProvider'
try:
    ort.register_execution_provider_library(reg, qnn_ep.get_library_path())
except Exception:
    pass
devs = [d for d in ort.get_ep_devices() if d.ep_name == reg]

ep_options = {'backend_path': qnn_ep.get_qnn_htp_path()}
for kv in args.sess_opt:
    k, v = kv.split('=', 1)
    ep_options[k] = v

so = ort.SessionOptions()
for kv in args.so_entry:
    k, v = kv.split('=', 1)
    so.add_session_config_entry(k, v)
so.add_provider_for_devices(devs, ep_options)

t_init0 = time.perf_counter_ns()
sess = ort.InferenceSession(str(model_path), sess_options=so)
init_ms = (time.perf_counter_ns() - t_init0) / 1e6

# Tokenize
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
enc = tok('benchmark prompt for harrier embedding model snapdragon x elite',
          padding='max_length', max_length=args.seq_len,
          truncation=True, return_tensors='np')
inputs_meta = {i.name: i for i in sess.get_inputs()}
feed = {}
for name, meta in inputs_meta.items():
    arr = enc[name] if name in enc else (
        enc['attention_mask'] if name == 'input_mask' else np.zeros_like(enc['input_ids']))
    if 'int32' in meta.type:
        arr = arr.astype(np.int32)
    elif 'int64' in meta.type:
        arr = arr.astype(np.int64)
    feed[name] = arr

run_options = ort.RunOptions()
for kv in args.run_opt:
    k, v = kv.split('=', 1)
    run_options.add_run_config_entry(k, v)

# Warmup
last = None
for _ in range(args.warmup):
    last = sess.run(None, feed, run_options=run_options)

# Timed iters
samples_ns = []
for _ in range(args.iters):
    t = time.perf_counter_ns()
    last = sess.run(None, feed, run_options=run_options)
    samples_ns.append(time.perf_counter_ns() - t)

h = hashlib.blake2b(digest_size=16)
for o in last:
    a = np.round(o.astype(np.float64), 3)
    h.update(repr(a.shape).encode())
    h.update(a.tobytes())

arr = np.asarray(samples_ns, np.int64)
# Trimmed mean: drop bottom 5% and top 10%
lo, hi = np.percentile(arr, [5, 90])
trimmed = arr[(arr >= lo) & (arr <= hi)]
res = {
    'label': args.label,
    'iters': args.iters,
    'warmup': args.warmup,
    'init_ms': init_ms,
    'mean_ms': float(arr.mean())/1e6,
    'trimmed_mean_ms': float(trimmed.mean())/1e6,
    'stdev_ms': float(arr.std(ddof=1))/1e6 if len(arr)>1 else 0,
    'min_ms': float(arr.min())/1e6,
    'p05_ms': float(np.percentile(arr,5))/1e6,
    'p25_ms': float(np.percentile(arr,25))/1e6,
    'median_ms': float(np.median(arr))/1e6,
    'p75_ms': float(np.percentile(arr,75))/1e6,
    'p95_ms': float(np.percentile(arr,95))/1e6,
    'p99_ms': float(np.percentile(arr,99))/1e6,
    'max_ms': float(arr.max())/1e6,
    'out_hash': h.hexdigest(),
    'ep_options': ep_options,
    'so_entries': args.so_entry,
    'run_options': args.run_opt,
}
existing = []
out = Path(args.out)
if out.exists():
    try: existing = json.loads(out.read_text())
    except Exception: pass
existing.append(res)
out.write_text(json.dumps(existing, indent=2))
print(json.dumps({k: res[k] for k in ['label','init_ms','trimmed_mean_ms','median_ms','min_ms','p05_ms','p95_ms','out_hash']}))
"""


def write_runner(path: Path) -> None:
    path.write_text(HARNESS_RUNNER)


def run_config(py: str, runner: Path, model_dir: Path, out: Path,
               label: str, sess_opts: list[str], run_opts: list[str],
               so_entries: list[str], iters: int, warmup: int) -> dict:
    cmd = [py, str(runner),
           "--model-dir", str(model_dir),
           "--iters", str(iters),
           "--warmup", str(warmup),
           "--label", label,
           "--out", str(out)]
    for kv in sess_opts:
        cmd += ["--sess-opt", kv]
    for kv in run_opts:
        cmd += ["--run-opt", kv]
    for kv in so_entries:
        cmd += ["--so-entry", kv]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if r.returncode != 0:
        return {"label": label, "error": r.stderr[-400:].strip(), "ep_options": sess_opts}
    try:
        return json.loads((r.stdout.strip().splitlines() or [""])[-1])
    except Exception:
        return {"label": label, "error": "could not parse stdout: " + r.stdout[-300:], "ep_options": sess_opts}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--repeats", type=int, default=3,
                   help="Fresh process invocations per config (each measures iters samples)")
    p.add_argument("--iters", type=int, default=200)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--cooldown-sec", type=int, default=2)
    p.add_argument("--include-baseline", action="store_true")
    p.add_argument("--configs-only", type=str, default=None,
                   help="comma-separated config names to run; default = all")
    args = p.parse_args()

    PY = sys.executable
    runner_path = args.out.parent / "_qnn_runner.py"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_runner(runner_path)
    if args.out.exists():
        args.out.unlink()

    # Config matrix
    # Each entry: (name, [sess_opts], [run_opts], [so_entries])
    CONFIGS: list[tuple[str, list[str], list[str], list[str]]] = []

    # Baseline (the README-recommended config)
    CONFIGS.append(("baseline_burst_finalize3",
                    ["htp_performance_mode=burst", "htp_graph_finalization_optimization_mode=3"],
                    [], []))
    CONFIGS.append(("baseline_burst_finalize3_rpc100",
                    ["htp_performance_mode=burst", "htp_graph_finalization_optimization_mode=3"],
                    ["qnn.rpc_control_latency=100"], []))

    # Per-run perf modes
    for mode in ["burst", "sustained_high_performance", "high_performance",
                 "balanced", "low_balanced", "default"]:
        CONFIGS.append((f"run_perf_{mode}",
                        ["htp_performance_mode=burst", "htp_graph_finalization_optimization_mode=3"],
                        [f"qnn.perf_mode={mode}"], []))

    # RPC control latency sweep (per-run)
    for lat in [10, 50, 100, 200, 500, 1000]:
        CONFIGS.append((f"rpc_lat_{lat}",
                        ["htp_performance_mode=burst", "htp_graph_finalization_optimization_mode=3"],
                        [f"qnn.rpc_control_latency={lat}"], []))

    # RPC polling time (per-run)
    for poll in [0, 1, 10, 100, 1000, 9999]:
        CONFIGS.append((f"rpc_poll_{poll}",
                        ["htp_performance_mode=burst", "htp_graph_finalization_optimization_mode=3"],
                        [f"qnn.rpc_polling_time={poll}"], []))

    # Both rpc_lat + polling
    CONFIGS.append(("rpc_lat100_poll0",
                    ["htp_performance_mode=burst", "htp_graph_finalization_optimization_mode=3"],
                    ["qnn.rpc_control_latency=100", "qnn.rpc_polling_time=0"], []))
    CONFIGS.append(("rpc_lat100_poll1000",
                    ["htp_performance_mode=burst", "htp_graph_finalization_optimization_mode=3"],
                    ["qnn.rpc_control_latency=100", "qnn.rpc_polling_time=1000"], []))
    CONFIGS.append(("rpc_lat10_poll1000",
                    ["htp_performance_mode=burst", "htp_graph_finalization_optimization_mode=3"],
                    ["qnn.rpc_control_latency=10", "qnn.rpc_polling_time=1000"], []))

    # Context priority
    for pri in ["high", "normal", "low", "normal_high"]:
        CONFIGS.append((f"ctx_pri_{pri}",
                        ["htp_performance_mode=burst", "htp_graph_finalization_optimization_mode=3",
                         f"qnn_context_priority={pri}"], [], []))

    # vtcm_mb sweep
    for v in [0, 1, 2, 4, 8]:
        CONFIGS.append((f"vtcm_{v}mb",
                        ["htp_performance_mode=burst", "htp_graph_finalization_optimization_mode=3",
                         f"vtcm_mb={v}"], [], []))

    # Spill fill buffer
    CONFIGS.append(("spill_fill_on",
                    ["htp_performance_mode=burst", "htp_graph_finalization_optimization_mode=3",
                     "enable_htp_spill_fill_buffer=1"], [], []))

    # share_resource_optimization
    for v in ["0", "1"]:
        CONFIGS.append((f"share_res_opt_{v}",
                        ["htp_performance_mode=burst", "htp_graph_finalization_optimization_mode=3",
                         f"htp_share_resource_optimization={v}"], [], []))

    # offload_graph_io_quantization off (default true)
    CONFIGS.append(("no_offload_io_quant",
                    ["htp_performance_mode=burst", "htp_graph_finalization_optimization_mode=3",
                     "offload_graph_io_quantization=0"], [], []))

    # rpc_control_latency at SESSION level (vs per-run)
    for lat in [100, 1000]:
        CONFIGS.append((f"sess_rpc_lat_{lat}",
                        ["htp_performance_mode=burst", "htp_graph_finalization_optimization_mode=3",
                         f"rpc_control_latency={lat}"], [], []))

    # Combined "everything reasonable" candidates
    CONFIGS.append(("combo_burst_rpc100_pri_high",
                    ["htp_performance_mode=burst", "htp_graph_finalization_optimization_mode=3",
                     "qnn_context_priority=high"],
                    ["qnn.rpc_control_latency=100", "qnn.perf_mode=burst"], []))
    CONFIGS.append(("combo_burst_rpc10_pri_high",
                    ["htp_performance_mode=burst", "htp_graph_finalization_optimization_mode=3",
                     "qnn_context_priority=high"],
                    ["qnn.rpc_control_latency=10", "qnn.perf_mode=burst"], []))

    # Filter by --configs-only
    if args.configs_only:
        keep = set(args.configs_only.split(","))
        CONFIGS = [c for c in CONFIGS if c[0] in keep]

    print(f"Running {len(CONFIGS)} configs x {args.repeats} repeats x {args.iters} iters")
    for round_i in range(args.repeats):
        print(f"\n=== round {round_i+1}/{args.repeats} ===", flush=True)
        for cfg_name, sess_opts, run_opts, so_entries in CONFIGS:
            label = f"{cfg_name}_r{round_i+1}"
            r = run_config(PY, runner_path, args.model_dir, args.out,
                           label, sess_opts, run_opts, so_entries,
                           args.iters, args.warmup)
            if "error" in r:
                print(f"  {label}: FAIL {r['error'][:120]}", flush=True)
            else:
                print(f"  {label}: trimmed={r.get('trimmed_mean_ms',0):.3f} ms  "
                      f"median={r.get('median_ms',0):.3f}  min={r.get('min_ms',0):.3f}  "
                      f"hash={r.get('out_hash','?')[:8]}", flush=True)
            time.sleep(args.cooldown_sec)

    print("done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
