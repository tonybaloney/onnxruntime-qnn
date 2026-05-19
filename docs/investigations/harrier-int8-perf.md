# Investigation: chasing a 25% perf win on `harrier-oss-v1-270m-qnn-htp`

**Status:** negative result. No QNN-EP-level configuration, code change, or run-time
option investigated here delivers a 25% improvement on the harrier embedding model
on Snapdragon X Elite. The HTP execution time itself is the bottleneck and is
opaque to the EP. Document below explains what was tried and what would actually
move the needle.

## Workload

`huggingface.co/anthonypjshaw/harrier-oss-v1-270m-qnn-htp` —
a static-INT8 quantised Gemma3-text 270M embedding model, packaged as a
precompiled QNN HTP context binary wrapped in an `EPContext` ONNX node.
Compiled by Qualcomm AI Hub for **Snapdragon X Elite CRD** (Hexagon V73).
Input `(input_ids, attention_mask)` shape `(1, 512)` int32; output `(1, 640)`
float32.

## Hardware / software

| | |
| --- | --- |
| Device                 | Snapdragon X Elite CRD (X1E80100), 12 cores  |
| Host OS                | Windows 11 ARM64                             |
| ORT runtime            | 1.26.0                                       |
| onnxruntime-qnn        | `main` @ `cb31cc64ca` (built locally)        |
| QAIRT SDK              | 2.46.0                                       |
| Repo base for build    | `main`, vanilla — no patches                 |

## Methodology

Designed to minimise noise:

* **Each configuration runs in a fresh process** (no JIT-cache or warm-FP-state
  sharing across configs).
* **20 warmup iterations + 200 timed iterations per process invocation**, so
  ~12s of HTP work per cell.
* **3 fresh process invocations per configuration**, interleaved so that
  thermal state is comparable.
* Per-process statistics: trimmed mean (drop bottom 5%, top 10%), min,
  median, p05/p25/p50/p75/p95/p99, max.
* Across-process aggregation: median-of-trimmed-means and min-of-mins.
* Identical input (same prompt, same tokenizer, same numpy seed).
* `out_hash` recorded per process — equal hash across configs = no
  numerics drift.

## Configuration matrix (40 cells × 3 rounds × 200 iters = 24 000 timed inferences)

Every session-level and per-run option exposed by the QNN EP that is plausible
on a Snapdragon X Elite HTP context binary:

* `htp_performance_mode` ∈ {burst, balanced, high_performance,
  sustained_high_performance, low_balanced, default}
* `htp_graph_finalization_optimization_mode` = 3 (baseline)
* `htp_share_resource_optimization` ∈ {0, 1}
* `enable_htp_spill_fill_buffer` ∈ {0, 1}
* `offload_graph_io_quantization` ∈ {true, false}
* `vtcm_mb` ∈ {0 (driver default), 1, 2, 4, 8}
* `qnn_context_priority` ∈ {high, normal, low, normal_high}
* `qnn.perf_mode` (per-run) sweep across all valid values
* `qnn.rpc_control_latency` (per-run) ∈ {10, 50, 100, 200, 500, 1000} µs
* `qnn.rpc_polling_time`  (per-run) ∈ {0, 1, 10, 100, 1000, 9999}
* `rpc_control_latency` (session-level) sweep
* Combinations of the above

## Results — top 10 by median-of-trimmed-mean

| Config                            | n | min-of-min (ms) | median-of-trimmed (ms) | best trimmed (ms) | Δ vs baseline |
| --------------------------------- | - | --------------- | ---------------------- | ----------------- | ------------- |
| `vtcm_0mb` (driver default)       | 3 | 55.40           | **57.45**              | 56.75             | −2.1%         |
| `vtcm_1mb`                        | 3 | 55.35           | 57.47                  | 57.14             | −2.0%         |
| `rpc_lat_10` (per-run)            | 3 | 55.47           | 57.55                  | 56.95             | −1.9%         |
| `rpc_lat100_poll1000`             | 3 | 55.48           | 57.61                  | 56.44             | −1.8%         |
| `rpc_poll_1`                      | 3 | 55.59           | 57.68                  | 56.70             | −1.7%         |
| `rpc_lat_50`                      | 3 | 55.57           | 57.70                  | 57.38             | −1.6%         |
| `rpc_lat100_poll0`                | 3 | 55.40           | 57.80                  | 56.98             | −1.5%         |
| `run_perf_low_balanced`           | 3 | 55.40           | 57.82                  | 56.58             | −1.4%         |
| `rpc_lat_100`                     | 3 | 55.41           | 57.91                  | 57.08             | −1.3%         |
| `vtcm_2mb`                        | 3 | 55.45           | 57.95                  | 56.83             | −1.2%         |
| **`baseline_burst_finalize3`**    | 3 | 55.47           | **58.66**              | 57.91             | —             |

Bottom of the table (worst configs by median-of-trimmed): `ctx_pri_normal`
60.20 ms (+2.6%), `rpc_lat_500` 59.74 (+1.8%), `rpc_lat_1000` 59.67 (+1.7%).

The entire option space spans **5.0% on the median-of-trimmed-mean** and
**0.9% on the min-of-mins**. The HTP execution lower bound on this device for
this graph is **~55.5 ms** regardless of EP configuration.

## Why no 25% win is reachable from the EP layer

The QNN EP per-`Run()` hot path
(`onnxruntime/core/providers/qnn/builder/qnn_model.cc::ExecuteGraph`) is:

1. For each input/output: query ORT for tensor type & shape, get raw data
   pointer, wrap as a `Qnn_Tensor_t` via `BindQnnTensorMemoryToOrtValueMemory`.
2. `SetPerThreadHtpPowerConfigs(true)` — early-exits if no per-thread
   power config is set (the common case).
3. `qnn_interface.graphExecute(...)` — the actual HTP execution.
4. `SetPerThreadHtpPowerConfigs(false)` — early-exits in the same case.
5. `ExtractBackendProfilingInfo(...)` — early-exits when profiling is off
   (the default in our harness).

For this 58 ms total Run, **graphExecute alone is ~55.5 ms** (min observed,
consistent across every config). The EP framing/marshalling above accounts
for the remaining ~2.5–3 ms (~4–5% of total). Eliminating *all* of that
overhead would yield ~5%, not 25%.

The graph itself (operations, memory layout, HVX scheduling, weight
unpacking) is baked into the QNN HTP context binary at AI Hub compile time.
It cannot be changed by the EP at runtime. The only parameters the EP can
nudge are the ones we tried — and none of them matter for this workload
because:

* `htp_performance_mode=burst` already pins HTP to max frequency.
* `vtcm_mb` is overridden by the binary's compiled VTCM requirement; observed
  effect is sub-noise.
* `htp_share_resource_optimization=1` returns `QNN_COMMON_ERROR_NOT_SUPPORTED`
  on V73 (Hexagon).
* `offload_graph_io_quantization=0` is a no-op for harrier (graph has no
  IO quantize/dequantize nodes; input/output are int32/float32 directly).
* `rpc_control_latency` / `rpc_polling_time` move totals by <2% in either
  direction; their effects are within thermal noise across rounds.

## What would actually yield 25% on this workload

Each of these changes the *workload*, not the EP:

1. **Recompile harrier with a smaller weight footprint** — INT4 weight
   quantisation on AI Hub. The HTP would execute fewer DDR reads per layer.
   This is an AI Hub job change, not an EP change.
2. **Reduce sequence length** — harrier is compiled with seq_len = 512. A
   build at seq_len = 128 would run ~4× faster for short inputs (which most
   embedding traffic is). Again an AI Hub recompile.
3. **Use weight sharing or context-sharing** when running multiple
   embedding workloads in the same process. This amortises compile/setup
   over many calls but does not change per-call latency.
4. **Batched embedding** — compile with `batch_size = N`. Per-sequence cost
   drops with batch; total latency rises sub-linearly. Workload change.

## Reproducing this investigation

Branch: `qnn-ep/harrier-int8-perf-investigation`.

* `docs/investigations/harrier-int8-perf.md` — this document.
* `docs/investigations/harrier_explorer.py` — the noise-controlled config
  explorer used to produce the numbers above. It takes a model directory
  containing `model.onnx` (the `EPContext` wrapper), `model.bin`
  (precompiled HTP context binary), and a Hugging Face tokenizer.

```text
python docs/investigations/harrier_explorer.py \
    --model-dir <path to harrier model dir> \
    --out harrier_explore.json \
    --repeats 3 --iters 200 --warmup 20
```

Each config writes one entry per round to the output JSON. Aggregate across
rounds for honest comparison.

## Conclusion

No verifiable 25% perf win exists at the QNN EP layer for the
`harrier-oss-v1-270m-qnn-htp` workload on Snapdragon X Elite. The HTP
execution is the bottleneck at ~55.5 ms. The EP options we *could* tune
(slightly different default `htp_performance_mode`, `vtcm_mb`, RPC defaults)
would sum to single-digit-percent at best and are within thermal noise.

Real 25%+ wins on this model require recompiling at AI Hub with a smaller
weight type, a shorter compiled sequence length, batched inputs, or a
different graph topology — none of which are within the scope of
`onnxruntime-qnn`.
