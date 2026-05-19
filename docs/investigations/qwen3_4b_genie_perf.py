"""Verify the safe (non-async-init) winning Genie configurations for Qwen3-4B
across multiple rounds and report aggregated statistics.

Usage:
    python qwen3_4b_genie_perf.py \
        --model-dir <path to qwen3_4b genie package> \
        --prompt-file <path to a prompt file> \
        --out result.json \
        --rounds 6

This script wraps `genie-t2t-run.exe` and times it externally.  It edits the
`genie_config.json` shipped in the package to apply specific HTP backend
configurations.  All generated config files are written into the model
directory and cleaned up afterwards.
"""
from __future__ import annotations
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


def find_genie(qairt_root: Path) -> tuple[Path, Path]:
    """Locate the genie-t2t-run.exe and the runtime library directory."""
    candidates = [
        ("aarch64-windows-msvc",),
        ("arm64-windows-msvc",),
        ("x86_64-windows-msvc",),
    ]
    for arch in candidates:
        bin_dir = qairt_root / "bin" / arch[0]
        lib_dir = qairt_root / "lib" / arch[0]
        exe = bin_dir / "genie-t2t-run.exe"
        if exe.exists() and lib_dir.exists():
            return exe, lib_dir
    raise SystemExit(f"genie-t2t-run.exe not found under {qairt_root}/bin/*-windows-msvc")


def write_config(model_dir: Path, base_cfg: dict,
                 qnnhtp_overrides: dict | None = None,
                 engine_overrides: dict | None = None,
                 ext_overrides: dict | None = None,
                 ext_filename: str | None = None) -> tuple[Path, list[Path]]:
    """Materialise a derived genie_config (and optional htp_backend_ext) file
    in model_dir and return (config_path, [paths_to_cleanup])."""
    cfg = json.loads(json.dumps(base_cfg))
    if qnnhtp_overrides:
        cfg["dialog"]["engine"]["backend"]["QnnHtp"].update(qnnhtp_overrides)
    if engine_overrides:
        cfg["dialog"]["engine"].update(engine_overrides)

    cleanup: list[Path] = []
    if ext_overrides is not None and ext_filename is not None:
        ext = json.loads((model_dir / "htp_backend_ext_config.json").read_text())
        for k, v in ext_overrides.items():
            if k in ("perf_profile", "rpc_control_latency", "core_id"):
                ext["devices"][0]["cores"][0][k] = v
            elif k in ("soc_model", "dsp_arch"):
                ext["devices"][0][k] = v
            else:
                ext[k] = v
        (model_dir / ext_filename).write_text(json.dumps(ext))
        cleanup.append(model_dir / ext_filename)
        cfg["dialog"]["engine"]["backend"]["extensions"] = ext_filename

    out_name = f"_genie_perf_{int(time.time()*1000) % 10**9}.json"
    out_path = model_dir / out_name
    out_path.write_text(json.dumps(cfg, indent=2))
    cleanup.append(out_path)
    return out_path, cleanup


def run_genie(genie_exe: Path, lib_dir: Path, model_dir: Path,
              config_path: Path, prompt_file: Path, label: str,
              max_seconds: int = 120) -> dict:
    env = os.environ.copy()
    env["PATH"] = f"{lib_dir}{os.pathsep}{genie_exe.parent}{os.pathsep}{env.get('PATH','')}"
    cmd = [str(genie_exe), "-c", config_path.name,
           "--prompt_file", str(prompt_file), "--log", "warn"]
    t0 = time.perf_counter()
    try:
        r = subprocess.run(cmd, cwd=str(model_dir), env=env,
                           capture_output=True, text=True, timeout=max_seconds)
        wall = time.perf_counter() - t0
        ok = r.returncode == 0
        stdout, stderr = r.stdout, r.stderr
    except subprocess.TimeoutExpired as e:
        wall = time.perf_counter() - t0
        ok = False
        stdout = (e.stdout.decode() if e.stdout else "")
        stderr = (e.stderr.decode() if e.stderr else "") + "\n[TIMED OUT]"

    response = stdout
    if "[BEGIN]:" in stdout and "[END]" in stdout:
        try:
            response = stdout.split("[BEGIN]:", 1)[1].split("[END]", 1)[0]
        except Exception:
            pass

    n_tokens = -1
    try:
        from tokenizers import Tokenizer
        tok = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        n_tokens = len(tok.encode(response).ids)
    except Exception:
        n_tokens = max(1, len(response) // 4)
    if n_tokens > 10_000:
        ok = False
        n_tokens = -1

    return {
        "label": label,
        "ok": ok,
        "wall_s": wall,
        "response_tokens": n_tokens,
        "tok_per_s_overall": (n_tokens / wall) if (ok and wall > 0 and n_tokens > 0) else 0.0,
        "stderr_tail": stderr[-400:] if stderr else "",
    }


def safe_configs():
    return [
        ("00_baseline",                  None,                                              None, None,                                                None),
        ("01_poll_false",                {"poll": False},                                   None, None,                                                None),
        ("02_perf_sustained",            None,                                              None, {"perf_profile": "sustained_high_performance"},      "ext_sus.json"),
        ("03_poll_false_perf_sus",       {"poll": False},                                   None, {"perf_profile": "sustained_high_performance"},      "ext_sus.json"),
        ("04_poll_false_rpc_10",         {"poll": False},                                   None, {"rpc_control_latency": 10},                         "ext_rpc10.json"),
        ("05_poll_false_perf_sus_rpc10", {"poll": False},                                   None, {"perf_profile": "sustained_high_performance", "rpc_control_latency": 10}, "ext_combo.json"),
        ("06_async_init_only",           {"allow-async-init": True},                        None, None,                                                None),
        ("07_full_combo_flaky",          {"allow-async-init": True, "poll": False},         None, {"perf_profile": "sustained_high_performance"},      "ext_sus.json"),
    ]


def aggregate(out: Path) -> None:
    rows = json.loads(out.read_text())
    by: dict[str, list[dict]] = {}
    for r in rows:
        key = r["label"].rsplit("_r", 1)[0]
        by.setdefault(key, []).append(r)
    base_oks = [r["tok_per_s_overall"] for r in by.get("00_baseline", []) if r.get("ok")]
    base_mean = sum(base_oks) / len(base_oks) if base_oks else 1.0
    print(f"\nBASELINE: n={len(base_oks)} mean={base_mean:.3f} tok/s")
    for k in sorted(by.keys()):
        g = by[k]
        oks = [r for r in g if r.get("ok")]
        if not oks:
            print(f"  {k:40s} n={len(g)} ok=0  ALL FAILED")
            continue
        rates = [r["tok_per_s_overall"] for r in oks]
        mn = sum(rates) / len(rates)
        sd = (sum((x - mn) ** 2 for x in rates) / (len(rates) - 1)) ** 0.5 if len(rates) > 1 else 0
        delta = (mn - base_mean) / base_mean * 100
        print(f"  {k:40s} n={len(g)} ok={len(oks)} "
              f"mean={mn:6.3f} sd={sd:.3f} min={min(rates):.2f} max={max(rates):.2f} "
              f"delta_pct={delta:+5.1f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--qairt-root", type=Path,
                   default=Path(r"C:\qairt\qairt\2.46.0.260424"))
    p.add_argument("--prompt-file", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--rounds", type=int, default=6)
    p.add_argument("--max-seconds", type=int, default=90)
    args = p.parse_args()

    genie_exe, lib_dir = find_genie(args.qairt_root)
    if args.out.exists():
        args.out.unlink()
    base_cfg = json.loads((args.model_dir / "genie_config.json").read_text())

    configs = safe_configs()
    print(f"Running {len(configs)} configs x {args.rounds} rounds")
    for round_i in range(args.rounds):
        print(f"\n=== round {round_i+1}/{args.rounds} ===", flush=True)
        for label, qnn_o, eng_o, ext_o, ext_fname in configs:
            cfg_path, cleanup = write_config(args.model_dir, base_cfg, qnn_o, eng_o, ext_o, ext_fname)
            full_label = f"{label}_r{round_i+1}"
            try:
                r = run_genie(genie_exe, lib_dir, args.model_dir, cfg_path,
                              args.prompt_file, full_label, args.max_seconds)
            except Exception as e:
                r = {"label": full_label, "ok": False, "error": str(e)[:300]}
            flag = "OK " if r.get("ok") else "FAIL"
            print(f"  {full_label:40s} {flag} wall={r.get('wall_s',0):6.2f}s "
                  f"tokens={r.get('response_tokens',0):4d} tok/s={r.get('tok_per_s_overall',0):5.2f}",
                  flush=True)
            existing = []
            if args.out.exists():
                try: existing = json.loads(args.out.read_text())
                except Exception: pass
            existing.append(r)
            args.out.write_text(json.dumps(existing, indent=2))
            for c in cleanup:
                try: c.unlink()
                except Exception: pass

    aggregate(args.out)


if __name__ == "__main__":
    main()
