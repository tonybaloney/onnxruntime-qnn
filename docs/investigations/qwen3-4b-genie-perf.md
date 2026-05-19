# Investigation: Qwen3-4B Genie (w4a16) performance on Snapdragon X Elite

**Status:** measured, reproducible results. **A two-line change to the shipped
Genie config (`poll: false` + `perf_profile: sustained_high_performance`)
delivers a reliable +7.8% throughput** (6/6 rounds, low stdev) on
Snapdragon X Elite. A larger +22.1% win exists with `allow-async-init: true`
included but is **not production-ready in QAIRT 2.46** — that flag triggers a
QnnHtp teardown race that crashes/hangs the process ~50% of the time. This
document explains how those numbers were obtained and which knobs matter.

## Model

`huggingface.co/qualcomm/Qwen3-4B` — `genie/w4a16/qualcomm-snapdragon-x-elite`
asset (`qwen3_4b-genie-w4a16-qualcomm_snapdragon_x_elite.zip`, ~2.4 GB).

* w4a16 quantisation (4-bit weights, 16-bit activations)
* 36 transformer layers, hidden 2560, intermediate 9728, head_dim 128, 32
  attention heads, 8 KV heads (GQA 4:1)
* Context lengths 512 / 1024 / 2048 / 3072 / 4096, prompt-processor
  AR=128, single-token generator AR=1, each compiled into a separate HTP
  graph and packed across 4 context binaries (~3 GB total).
* Tokenizer: Qwen3 BPE, vocab 151 936.
* Compiled by Qualcomm AI Hub against QAIRT 2.45.

## Stock configuration (as shipped)

```jsonc
// genie_config.json (Qualcomm-published)
"engine": {
    "n-threads": 3,
    "backend": {
        "type": "QnnHtp",
        "QnnHtp": {
            "use-mmap": true,
            "spill-fill-bufsize": 0,
            "mmap-budget": 0,
            "poll": true,
            "cpu-mask": "0xe0",
            "allow-async-init": false,
            ...
        },
        "extensions": "htp_backend_ext_config.json"
    }
}

// htp_backend_ext_config.json (Qualcomm-published)
{"devices": [{"soc_model": 60, "dsp_arch": "v73",
              "cores": [{"core_id": 0,
                         "perf_profile": "burst",
                         "rpc_control_latency": 100}]}],
 "memory": {"mem_type": "shared_buffer"},
 "context": {"weight_sharing_enabled": true}}
```

## Test environment

| | |
| --- | --- |
| Device          | Snapdragon X Elite CRD (X1E80100), 12 cores |
| Host OS         | Windows 11 ARM64                            |
| QAIRT SDK       | 2.46.0.260424                               |
| Runner          | `genie-t2t-run.exe` (Qualcomm-shipped)      |
| Prompt          | `<|im_start|>system\nYou are a helpful AI assistant<|im_end|>\n<|im_start|>user\nWhat is gravity? Answer in exactly 30 words.<|im_end|>\n<|im_start|>assistant\n` |
| Decode length   | ~300 tokens (model decides; bounded by EOS) |
| Sampler         | seed 42, temp 0.8, top-k 40, top-p 0.95 (stock) |

## Methodology

* Each configuration runs in a fresh `genie-t2t-run.exe` process (so model
  load is re-measured every time and no warm-start advantage carries
  between configs).
* 6 rounds, interleaved so thermal state is comparable across configs.
* Reject obvious garbage (e.g. when the process dies mid-output but
  Python still reads stale bytes — those return `> 10 000` "tokens" which
  the harness drops).
* Output token count obtained from the model's own tokenizer
  (`tokenizers` library, `tokenizer.json` from the package). Same input
  for every run, but the sampler is stochastic so the output token count
  varies by ~1% (300–304); the rate is computed per-run so this is fine.
* Each run is whole-task wall-clock: model load + prefill (128 tokens) +
  decode (~300 tokens) + teardown. `tok/s_overall = decoded_tokens / wall_s`.
  The Qualcomm-published "Response Rate" of 17.5 tok/s is the steady-state
  decode-only rate, which is higher than the overall here because load is
  amortised over so few generated tokens.

## Results (6 rounds × 8 configs = 48 timed runs)

| Config                                                                   | n / ok    | mean (tok/s) | sd   | min   | max   | Δ vs baseline | Reliable? |
| ------------------------------------------------------------------------ | --------- | ------------ | ---- | ----- | ----- | ------------- | --------- |
| `baseline` (stock)                                                       | 6 / 6     | 13.32        | 0.28 | 12.93 | 13.64 | —             | ✅         |
| `poll: false`                                                            | 6 / 6     | 13.65        | 0.29 | 13.32 | 13.97 | +2.5%         | ✅         |
| `perf_profile: sustained_high_performance`                               | 6 / 6     | 13.69        | 0.65 | 12.80 | 14.78 | +2.8%         | ✅         |
| **`poll: false` + `perf_profile: sustained_high_performance`**           | **6 / 6** | **14.36**    | 0.40 | 14.06 | 14.94 | **+7.8%**     | **✅**     |
| `poll: false` + `rpc_control_latency: 10`                                | 6 / 5     | 13.81        | 0.36 | 13.52 | 14.40 | +3.7%         | mostly    |
| `poll: false` + `perf_sus` + `rpc_control_latency: 10`                   | 6 / 6     | 14.33        | 0.35 | 14.01 | 15.01 | +7.6%         | ✅         |
| `allow-async-init: true` alone                                           | 6 / 1     | 14.53        | —    | 14.53 | 14.53 | +9.1%         | ⚠️ FLAKY  |
| `allow-async-init: true` + `poll: false` + `perf_sus` (full combo)       | **6 / 4** | **16.27**    | 0.52 | 15.98 | 17.05 | **+22.1%**    | ⚠️ FLAKY  |

## What the flaky configurations are doing

Setting `allow-async-init: true` triggers a QnnHtp code path explicitly
labelled `[INFO]  "Using create From Binary List Async"` — it loads the
four ~700 MB context binaries concurrently rather than one at a time.
When it succeeds, model-load time drops by ~3 s and you get +9% on its
own or +22% on top of the other safe wins.

When it fails (4/6 standalone, 2/6 in combo), the error is reproducible:

```
Genie:   5439.1ms [ ERROR ]  <E> Transport.teardownLocked:
                              qnn_close error 0x00000200, userCnt 0 prio 100
Genie:   5445.1ms [WARNING]  <W> Cannot find
                              HTP_USR_DRV_GRAPH_CONFIG_ESTIMATION_MEMORY_USAGE graphConfigs!
... (the warning repeats until the process is killed)
```

It is a race between the async loader's per-binary teardown and the
overall context bring-up. Not safe to recommend until Qualcomm fixes the
race in QnnHtp.

## What did NOT move the needle

* `n-threads` sweep (1, 2, 3 (baseline), 4, 6, 8, 12): all within ±5%
  of baseline. The default `3` is reasonable.
* `cpu-mask` sweep (`0xe0` baseline = cores 5-7, vs `0xfff` all 12, vs
  `0xf0` cores 4-7, vs `0xff0` cores 4-11): all within ±5%. The default
  `0xe0` (3 high-perf cores) is good; pinning to *more* cores actually
  hurts slightly, presumably from contention with the HTP control thread.
* `rpc_control_latency` sweep (0, 10, 100, 500, 1000 µs): trough at
  `10`, peak at `1000`, total spread ~2%.
* `use-mmap: false`: −21% (much slower, as expected — loads the 3 GB of
  binaries by `read()` instead of `mmap`).
* `perf_profile: high_performance` vs `burst` vs `sustained_high_performance`:
  `burst` and `high_performance` are within ±2%; only `sustained_high_performance`
  shows the +2.8% effect, presumably because the workload is long enough
  for the burst/turbo budget to expire.

## Why `poll: false` + `sustained_high_performance` helps

The harrier investigation in this repo found that the QNN HTP execution
time itself is a hard floor (~55 ms/iteration for a 270 M model) and EP
overhead is only ~5% on top. Qwen3-4B is the opposite end: 4 B parameters,
36 layers, per-token decode time is ~57 ms (≈17.5 tok/s steady state),
and the Genie LLM runtime sits between the user and the HTP doing prefill
batching, KV-cache management, sampling, and tokenisation.

* **`poll: true` (default)** spins a CPU thread waiting for HTP
  completion. That delivers the lowest possible per-call latency but
  costs a full P-core that the OS scheduler can't use for anything else
  — including the next prefill micro-batch or the sampler. Switching to
  `poll: false` (interrupt-driven completion) frees that core for the
  Genie host work that has to happen between every token. Cheap +2.5%.
* **`perf_profile: burst` (default)** is tuned for short, intermittent
  HTP work. A 300-token LLM decode loop is steady-state load; the burst
  power profile drops to the sustained-clock floor after the burst
  budget expires, which on this device costs ~3% by the end of the
  decode loop. `sustained_high_performance` keeps the clock at the
  sustained ceiling for the whole decode. Another +2.8%.
* The two combine **super-linearly** — likely because the freed P-core
  is available exactly when the sustained-clock HTP needs the host to
  feed the next iteration with no scheduler delay. Net +7.8%.

## Recommendation

For Qwen3-4B (and likely any decode-bound LLM > a few hundred
parameters) running on Snapdragon X Elite via QAIRT 2.46 Genie:

* In `genie_config.json`, set `"poll": false`.
* In `htp_backend_ext_config.json`, set
  `"perf_profile": "sustained_high_performance"`.

That's the entire safe change. Both knobs are already documented and
supported in QAIRT 2.46; this is purely a defaults question.

The next +14% (to reach +22%) requires Qualcomm to fix the
`allow-async-init: true` race in QnnHtp. Once that bug is gone, the same
two safe defaults should compose with it to land Qwen3-4B at ~16.3 tok/s
overall on Snapdragon X Elite, vs the ~13.3 tok/s the published config
delivers today.

## Reproducing this investigation

`docs/investigations/qwen3_4b_genie_perf.py` runs the full 8-config × N-round
sweep, swaps the configs on disk between runs, and aggregates the result.

```bat
REM Download the Qwen3-4B Genie package from
REM https://huggingface.co/qualcomm/Qwen3-4B (Snapdragon X Elite variant)
REM and unzip into <model_dir>.

python docs\investigations\qwen3_4b_genie_perf.py ^
    --model-dir <model_dir> ^
    --qairt-root C:\qairt\qairt\2.46.0.260424 ^
    --prompt-file <model_dir>\sample_prompt.txt ^
    --out qwen3_4b_perf.json ^
    --rounds 6
```

Reads:

* The shipped `genie_config.json` and `htp_backend_ext_config.json` as the
  baseline; writes derived per-config copies into the model directory
  and removes them after each run.
* The shipped `tokenizer.json` for accurate output-token counts (via the
  `tokenizers` PyPI package).

Writes:

* `--out` JSON file: one entry per run with `wall_s`, `response_tokens`,
  `tok_per_s_overall`, and the tail of stderr (so you can see the
  `Transport.teardownLocked` error when `allow-async-init` fails).
* Aggregated table to stdout, same shape as the Results section above.

## Conclusion

The Qualcomm-shipped `genie_config.json` + `htp_backend_ext_config.json`
defaults for Qwen3-4B on Snapdragon X Elite leave a reliable **+7.8%** of
overall throughput on the floor. Two changes:
* `"poll": false` in `genie_config.json`
* `"perf_profile": "sustained_high_performance"` in
  `htp_backend_ext_config.json`

…take Qwen3-4B from 13.32 → 14.36 tok/s overall with no other code
changes. A further +14% is available via `allow-async-init: true` but is
blocked by a QnnHtp teardown race in QAIRT 2.46.
