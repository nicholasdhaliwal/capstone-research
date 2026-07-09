#!/usr/bin/env python3
"""Generate 01_baseline.ipynb (same generator pattern as legal-rag's
_build_notebooks.py). Edit this file, then run `python3 _build_01_baseline.py`
to regenerate the notebook. No nbformat dependency - plain JSON."""
import json
from pathlib import Path

CELLS = []


def md(src):
    CELLS.append({"cell_type": "markdown", "metadata": {}, "source": src})


def code(src):
    CELLS.append({
        "cell_type": "code",
        "metadata": {},
        "source": src,
        "outputs": [],
        "execution_count": None,
    })


# ------------------------------------------------------------------ 0. header

md(r'''# 01 - CaseHOLD Baseline (vLLM, Vertex AI Workbench)

Baselines every model in the panel on the CaseHOLD test set (no retrieval):
**{zero_shot, few_shot} x {letter_emit, logprob}** per model, plus a Claude
Haiku 4.5 API row (letter-emit only). Produces per-model metrics and the
questions x models correctness grid that feeds IRT.

How to run:
1. Fill in `GCS_BUCKET` / `GCP_PROJECT` in the config cell.
2. Run top-to-bottom with `SMOKE_TEST = True` (first 50 questions, 2 small models).
3. If the smoke table and sanity checks pass, set `SMOKE_TEST = False`, restart, run all.

Everything resumes: results already in GCS are skipped; interrupted runs
continue from their `.items.jsonl` sidecar. Logic lives in `eval_core.py`;
each model runs in a `run_eval.py` subprocess (vLLM does not reliably release
GPU memory in-kernel).''')

# ------------------------------------------------------------------ 1. config

md(r'''## 1. Config + GPU detection''')

code(r'''# ========================= CONFIG (edit here only) =========================
SEED = 42
N_CTX = 8192
TEMPERATURE = 0.0

SMOKE_TEST = True        # True: first 50 questions + 2 small models. False: full run (3,600 x panel).
N_SMOKE = 50

# --- GCP: fill these in on the Workbench instance before running ---
GCS_BUCKET = ""          # results/artifacts bucket, no gs:// prefix
GCP_PROJECT = ""         # project id (also used by the AnthropicVertex client)
GCP_REGION = "global"    # Claude endpoint; "global" recommended (regional adds a 10% premium)

LOCAL_OUT = "outputs"    # local mirror; everything here is also uploaded to GCS
ARTIFACTS_DIR = f"{LOCAL_OUT}/artifacts"
RESULTS_DIR = f"{LOCAL_OUT}/results"

CONDITIONS = ["zero_shot", "few_shot"]
SCORINGS = ["letter_emit", "logprob"]   # logprob is PRIMARY; acc_norm is the headline
GRID_CONDITION = "zero_shot"            # condition used for the IRT correctness grid

import os
from pathlib import Path
for _d in (ARTIFACTS_DIR, RESULTS_DIR):
    Path(_d).mkdir(parents=True, exist_ok=True)

# eval_core's module-level imports are stdlib + numpy only, so this import is
# safe before the pip cell on a stock Workbench image.
import eval_core as ec
assert ec.SEED == SEED and ec.N_CTX == N_CTX and ec.TEMPERATURE == TEMPERATURE
MODEL_PANEL = ec.MODEL_PANEL                   # exact HF checkpoint ids + kind/role
SMOKE_MODELS = ["mistralai/Mistral-7B-Instruct-v0.3", "Qwen/Qwen2.5-7B-Instruct"]
API_MODEL = ec.API_MODEL                       # claude-haiku-4-5@20251001 (Vertex id)
API_PRICE_IN, API_PRICE_OUT = 1.00, 5.00       # $/MTok, Haiku 4.5, global endpoint

os.environ["ARTIFACTS_DIR"] = str(Path(ARTIFACTS_DIR).resolve())
os.environ["EVAL_N_CTX"] = str(N_CTX)
os.environ["GCS_BUCKET"] = GCS_BUCKET

for _m in MODEL_PANEL:
    print(f"{_m['role']:8s} {_m['kind']:9s} {_m['id']}")''')

code(r'''# GPU detection WITHOUT importing torch (so the pip cell below cannot strand a
# stale torch in this kernel). run_eval.py re-detects in-process via torch.
import subprocess
_q = subprocess.run(
    ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
    capture_output=True, text=True)
assert _q.returncode == 0, "nvidia-smi failed - is a GPU attached to this instance?"
GPU_NAME, _mem_mib = [s.strip() for s in _q.stdout.strip().splitlines()[0].split(",")]
GPU_MEM_GB = round(int(_mem_mib) / 1024, 1)

if GPU_MEM_GB >= 70:      # A100-80GB
    GPU_MEM_UTIL, CHUNK_QUESTIONS = 0.92, 500
elif GPU_MEM_GB >= 35:    # A100-40GB
    GPU_MEM_UTIL, CHUNK_QUESTIONS = 0.90, 250
else:                     # L4 24GB
    GPU_MEM_UTIL, CHUNK_QUESTIONS = 0.85, 100

os.environ["GPU_MEMORY_UTILIZATION"] = str(GPU_MEM_UTIL)
os.environ["CHUNK_QUESTIONS"] = str(CHUNK_QUESTIONS)
print(f"{GPU_NAME}  {GPU_MEM_GB} GB  -> gpu_memory_utilization={GPU_MEM_UTIL}, "
      f"chunk={CHUNK_QUESTIONS} questions per generate() call")
if GPU_MEM_GB < 30:
    print("WARNING: microsoft/phi-4 (14B) will OOM on this GPU; its rows will FAIL and be skipped.")''')

# ------------------------------------------------------------------ 2. env

md(r'''## 2. Environment setup

One pinned install cell. Re-running it is a no-op once satisfied. If pip
upgrades torch/vllm on the first run, **restart the kernel** and re-run from
the top.''')

code(r'''%pip install -q "vllm==0.24.0" "transformers>=4.51" "datasets>=3.0" scikit-learn google-cloud-storage "anthropic[vertex]" pandas numpy pyarrow

import importlib.metadata as _im
for _pkg in ("vllm", "torch", "transformers", "datasets", "scikit-learn",
             "google-cloud-storage", "anthropic", "pandas", "numpy", "pyarrow"):
    try:
        print(f"{_pkg:22s} {_im.version(_pkg)}")
    except _im.PackageNotFoundError:
        print(f"{_pkg:22s} MISSING")
print("\nIf pip just upgraded torch/vllm, RESTART THE KERNEL and re-run from the top.")''')

code(r'''import importlib, json, subprocess, sys, time
from datetime import datetime, timezone
import numpy as np
import pandas as pd
importlib.reload(ec)   # pick up eval_core edits without a kernel restart

assert GCS_BUCKET and GCP_PROJECT, "Fill in GCS_BUCKET and GCP_PROJECT in the config cell."
# Verify WRITE access now, not after hours of compute.
_probe = Path(LOCAL_OUT) / "_probe.txt"
_probe.write_text(datetime.now(timezone.utc).isoformat())
print("GCS write OK:", ec.upload_to_gcs(_probe, GCS_BUCKET, "artifacts/_write_probe.txt"))''')

# ------------------------------------------------------------------ 3. data

md(r'''## 3. Data: CaseHOLD + frozen DEV/CONFIRM split + fixed 3-shot examples

Source: `coastalcph/lex_glue` config `case_hold` (parquet; works on
`datasets>=3.0`). The `casehold/casehold` repo is loading-script-only and its
raw CSV test partition has 5,314 rows, not the standard 3,600 - `eval_core`
asserts the split sizes so a wrong source stops the run.

Artifacts are create-once: the first run writes them to GCS, every later run
loads them. Never regenerate.''')

code(r'''data = ec.load_casehold()      # asserts standard split sizes + canonical columns
train, test = data["train"], data["test"]
print(f"train={len(train)}  test={len(test)}")

# Frozen DEV (500, pipeline selection) / CONFIRM (3,100) partition of test.
split = ec.get_split(test, GCS_BUCKET, ARTIFACTS_DIR)
assert split["n_dev"] == 500 and split["n_confirm"] == 3100
print(f"split: dev={split['n_dev']} confirm={split['n_confirm']} (seed={split['seed']})")

# Fixed 3-shot demonstrations from TRAIN (never evaluated on), same for all models.
fewshot_art = ec.get_fewshot(train, GCS_BUCKET, ARTIFACTS_DIR)
print("few-shot train indices:", fewshot_art["train_indices"],
      "| gold letters:", [ec.LETTERS[e["gold_pos"]] for e in fewshot_art["examples"]])''')

# ------------------------------------------------------------------ 4. shuffle

md(r'''## 4. Shuffle: frozen per-question permutations

Mandatory in every condition (Q1 documented an A-bias: 80-90% "A" picks on
unshuffled prompts). The table is written to GCS once so every future notebook
uses the identical shuffle; the cell also proves determinism by recomputing.''')

code(r'''perm_art = ec.get_permutations(test, GCS_BUCKET, ARTIFACTS_DIR)
assert len(perm_art["table"]) == len(test)
# Determinism check: recomputing must reproduce the frozen artifact exactly.
assert ec.build_permutation_table(test) == perm_art["table"], \
    "permutation drift - SEED or RNG derivation changed; do NOT proceed"
_p0 = perm_art["table"][str(test[0]["example_id"])]
print(f"{len(perm_art['table'])} permutations frozen; "
      f"q0 -> perm={_p0['permutation']}, gold={_p0['shuffled_gold']}")''')

# ------------------------------------------------------------------ 5. smoke

md(r'''## 5. Smoke test (only runs when `SMOKE_TEST = True`)

2 smallest models x both scoring methods x zero-shot on the first 50
questions. Sanity checks:
- letter-emit parses 100% (guided decoding guarantees it),
- `sum_logp` differs across the 5 options (all-identical = the historical
  uniform-logit bug -> raise),
- accuracy above 0.20 chance.''')

code(r'''def run_one(model_id, condition, scoring, n):
    """Subprocess one (model, condition, scoring). Resume-safe: skips runs
    already completed in GCS; uploads the JSON right after the run."""
    name = ec.result_name(model_id, condition, scoring, f"n{n}")
    local = Path(RESULTS_DIR) / name
    blob = f"results/{name}"
    if not local.exists():
        ec.download_from_gcs(GCS_BUCKET, blob, local)
    if local.exists():
        res = json.loads(local.read_text())
        if res.get("completed"):
            return res, "cached"
    cmd = [sys.executable, "run_eval.py",
           "--model_id", model_id, "--condition", condition,
           "--scoring", scoring, "--n", str(n), "--output_path", str(local)]
    print(">>>", " ".join(cmd), flush=True)
    _t0 = time.time()
    proc = subprocess.run(cmd)          # vLLM/progress logs stream into this cell
    if proc.returncode != 0:
        print(f"FAILED (exit {proc.returncode}) after {time.time() - _t0:.0f}s", flush=True)
        return None, "failed"
    ec.upload_to_gcs(local, GCS_BUCKET, blob)
    return json.loads(local.read_text()), f"ran {time.time() - _t0:.0f}s"''')

code(r'''if SMOKE_TEST:
    _rows = []
    for _mid in SMOKE_MODELS:
        for _scoring in SCORINGS:
            _res, _status = run_one(_mid, "zero_shot", _scoring, N_SMOKE)
            assert _res is not None, f"smoke run failed: {_mid} {_scoring}"
            _m = _res["metrics"]
            _valid = [r for r in _res["per_item"] if not r.get("error")]
            if _scoring == "letter_emit":
                assert _m["n_parse_fail"] == 0, \
                    f"{_mid}: {_m['n_parse_fail']} parse failures under guided decoding"
            if _scoring == "logprob":
                for _r in _valid:
                    assert len({round(v, 6) for v in _r["sum_logp"].values()}) > 1, \
                        f"{_mid} q{_r['question_id']}: all 5 sum_logp identical - uniform-logit symptom"
            assert _m["accuracy"] > 0.20, \
                f"{_mid} {_scoring}: accuracy {_m['accuracy']:.3f} not above chance"
            _rows.append({"model": _mid.split("/")[-1], "scoring": _scoring, "status": _status,
                          "acc": round(_m["accuracy"], 3), "macro_f1": round(_m["macro_f1"], 3),
                          "ci": f"[{_m['ci_low']:.3f}, {_m['ci_high']:.3f}]",
                          "n": len(_valid), "parse_fail": _m["n_parse_fail"]})
    display(pd.DataFrame(_rows))
    print("\nSmoke checks passed: 100% parseable letters, non-degenerate logprobs, above-chance accuracy.")
else:
    print("SMOKE_TEST=False - smoke cell skipped")''')

# ------------------------------------------------------------------ 6. full run

md(r'''## 6. Full run loop

Model x condition x scoring, one `run_eval.py` subprocess per combination.
Results already completed in GCS are skipped (resume-safe), each JSON uploads
immediately after its run, and the running summary table redraws after every
row. In smoke mode this loop covers the 2 smoke models at n=50 (cheap; the
zero-shot rows are already cached from section 5).''')

code(r'''from IPython.display import display, clear_output

RUN_MODELS = SMOKE_MODELS if SMOKE_TEST else [m["id"] for m in MODEL_PANEL]
N_RUN = N_SMOKE if SMOKE_TEST else len(test)
print(f"running {len(RUN_MODELS)} models x {len(CONDITIONS)} conditions x {len(SCORINGS)} scorings at n={N_RUN}")

summary_rows = []
for _mid in RUN_MODELS:
    for _condition in CONDITIONS:
        for _scoring in SCORINGS:
            _res, _status = run_one(_mid, _condition, _scoring, N_RUN)
            _row = {"model": _mid, "condition": _condition, "scoring": _scoring, "status": _status}
            if _res is not None:
                _m = _res["metrics"]
                _row.update(acc=round(_m["accuracy"], 3), macro_f1=round(_m["macro_f1"], 3),
                            ci_low=round(_m["ci_low"], 3), ci_high=round(_m["ci_high"], 3),
                            parse_fail=_m["n_parse_fail"], n_error=_m.get("n_error", 0))
                if _scoring == "logprob":
                    _row["acc_sum_logp"] = round(_m["accuracy_acc"], 3)   # "acc"; headline stays acc_norm
            summary_rows.append(_row)
            clear_output(wait=True)
            display(pd.DataFrame(summary_rows))
run_table = pd.DataFrame(summary_rows)''')

# ------------------------------------------------------------------ 7. API row

md(r'''## 7. Claude Haiku 4.5 (Vertex AI Model Garden)

Same shuffle, same prompts, letter-emit only (no logprobs over the API).
Rows carry `scoring="letter_emit_api"` so they are never averaged with logprob
rows. Resume-safe via the `.items.jsonl` sidecar; retries with backoff live in
`eval_core.run_api_letter_emit`. Prints a token/cost counter.''')

code(r'''def run_api(condition, n):
    name = ec.result_name(API_MODEL, condition, "letter_emit_api", f"n{n}")
    local = Path(RESULTS_DIR) / name
    blob = f"results/{name}"
    if not local.exists():
        ec.download_from_gcs(GCS_BUCKET, blob, local)
    if local.exists():
        _res = json.loads(local.read_text())
        if _res.get("completed"):
            return _res, "cached"
    jsonl = local.with_suffix(".items.jsonl")
    done, done_ids = [], set()
    if jsonl.exists():
        for _line in jsonl.read_text().splitlines():
            if _line.strip():
                _rec = json.loads(_line)
                if _rec["question_id"] not in done_ids:
                    done_ids.add(_rec["question_id"])
                    done.append(_rec)
    _fs = fewshot_art if condition == "few_shot" else None
    todo = [q for q in ec.prepare_questions(test.select(range(n)), condition, _fs)
            if q["question_id"] not in done_ids]
    print(f"{condition}: {len(done)} resumed, {len(todo)} to run", flush=True)
    usage = {}
    new = ec.run_api_letter_emit(todo, project_id=GCP_PROJECT, region=GCP_REGION,
                                 jsonl_path=jsonl, usage=usage)
    per_item = done + new
    per_item.sort(key=lambda r: (len(r["question_id"]), r["question_id"]))
    cost = (usage.get("input_tokens", 0) / 1e6 * API_PRICE_IN
            + usage.get("output_tokens", 0) / 1e6 * API_PRICE_OUT)
    print(f"tokens this session: in={usage.get('input_tokens', 0):,} "
          f"out={usage.get('output_tokens', 0):,}  cost=${cost:.2f}")
    _valid = [r for r in per_item if not r.get("error")]
    _errors = [r for r in per_item if r.get("error")]
    _metrics = ec.compute_metrics(_valid)
    _metrics["n_error"] = len(_errors)
    result = {
        "config": {"model_id": API_MODEL, "condition": condition,
                   "scoring": "letter_emit_api", "seed": SEED, "n": len(per_item),
                   "region": GCP_REGION,
                   "timestamp": datetime.now(timezone.utc).isoformat()},
        "metrics": _metrics,
        "per_item": _valid + _errors,
        "usage_last_session": usage,
        "completed": True,
    }
    ec.save_json(result, local)
    ec.upload_to_gcs(local, GCS_BUCKET, blob)
    return result, "ran"

for _condition in CONDITIONS:
    _res, _status = run_api(_condition, N_RUN)
    _m = _res["metrics"]
    print(f"{API_MODEL} {_condition}: acc={_m['accuracy']:.3f} f1={_m['macro_f1']:.3f} "
          f"[{_m['ci_low']:.3f}, {_m['ci_high']:.3f}] parse_fail={_m['n_parse_fail']} ({_status})")''')

# ------------------------------------------------------------------ 8. summary

md(r'''## 8. Summary table + per-item correctness grid

The pivot below is saved as CSV to GCS (plus a long-form CSV for downstream
code).''')

code(r'''_rows = []
for _f in sorted(Path(RESULTS_DIR).glob(f"*__n{N_RUN}.json")):
    _r = json.loads(_f.read_text())
    if not _r.get("completed"):
        continue
    _c, _m = _r["config"], _r["metrics"]
    _rows.append({"model": _c["model_id"], "condition": _c["condition"], "scoring": _c["scoring"],
                  "accuracy": _m["accuracy"], "macro_f1": _m["macro_f1"],
                  "ci_low": _m["ci_low"], "ci_high": _m["ci_high"]})
long_df = pd.DataFrame(_rows)
long_df["cell"] = long_df.apply(
    lambda r: f"{r.accuracy:.3f} (F1 {r.macro_f1:.3f}) [{r.ci_low:.3f}, {r.ci_high:.3f}]", axis=1)
summary = long_df.pivot(index="model", columns=["condition", "scoring"], values="cell")
display(summary)

_csv = Path(LOCAL_OUT) / f"summary_baseline_n{N_RUN}.csv"
summary.to_csv(_csv)
_long_csv = Path(LOCAL_OUT) / f"summary_baseline_long_n{N_RUN}.csv"
long_df.drop(columns="cell").to_csv(_long_csv, index=False)
print(ec.upload_to_gcs(_csv, GCS_BUCKET, f"summary/{_csv.name}"))
print(ec.upload_to_gcs(_long_csv, GCS_BUCKET, f"summary/{_long_csv.name}"))''')

md(r'''### IRT input: `grid_baseline.parquet`

Questions x models 0/1 correctness from the **primary scoring (logprob,
acc_norm prediction)** in `GRID_CONDITION`. Context-overflow questions are
`NaN`, never 0 - the completeness of this grid matters more than the summary
metrics. This file is the input to the IRT stage.''')

code(r'''_grid = {}
for _f in sorted(Path(RESULTS_DIR).glob(f"*__{GRID_CONDITION}__logprob__n{N_RUN}.json")):
    _r = json.loads(_f.read_text())
    if not _r.get("completed"):
        continue
    _col = {}
    for _it in _r["per_item"]:
        _col[_it["question_id"]] = np.nan if _it.get("error") else float(_it["correct"])
    _grid[_r["config"]["model_id"]] = _col
grid_df = pd.DataFrame(_grid)
grid_df.index.name = "question_id"
grid_df = grid_df.loc[sorted(grid_df.index, key=lambda s: (len(s), s))]
_pq = Path(LOCAL_OUT) / "grid_baseline.parquet"
grid_df.to_parquet(_pq)
print(ec.upload_to_gcs(_pq, GCS_BUCKET, "grid_baseline.parquet"))
print(f"IRT input: {grid_df.shape[0]} questions x {grid_df.shape[1]} models "
      f"({int(grid_df.isna().sum().sum())} NaN cells)")
display(grid_df.mean().rename("acc_norm").round(3))''')

md(r'''### Consolidated per-question log

Every scored question, across every model x condition x scoring, in one long
table - the audit answer to "how did every question do" without opening
dozens of per-model JSONs. Raw `sum_logp`/`mean_logp` dicts stay out (kept
lean); full logprob detail is still in the per-model result JSONs.''')

code(r'''_panel_meta = {m["id"]: m for m in MODEL_PANEL}
_long_rows = []
for _f in sorted(Path(RESULTS_DIR).glob("*.json")):
    _r = json.loads(_f.read_text())
    if not _r.get("completed"):
        continue
    _c = _r["config"]
    _meta = _panel_meta.get(_c["model_id"], {"kind": "api", "role": "api"})
    for _it in _r["per_item"]:
        _long_rows.append({
            "question_id": _it["question_id"], "model_id": _c["model_id"],
            "kind": _meta["kind"], "role": _meta["role"],
            "condition": _c["condition"], "scoring": _c["scoring"],
            "shuffled_gold": _it.get("shuffled_gold"), "prediction": _it.get("prediction"),
            "correct": _it.get("correct"), "error": bool(_it.get("error", False)),
            "error_type": _it.get("error_type"),
        })
all_items_long = pd.DataFrame(_long_rows)
_all_csv = Path(LOCAL_OUT) / "all_items_long.csv"
all_items_long.to_csv(_all_csv, index=False)
print(ec.upload_to_gcs(_all_csv, GCS_BUCKET, "summary/all_items_long.csv"))
print(f"{len(all_items_long)} rows across {all_items_long['model_id'].nunique()} models")''')

md(r'''### DEV-slice exports (contract with `02_retrieval`)

Stage A of the retrieval notebook compares against notebook-1 bare rows named
`*__dev.json`. Those are a pure subset of the full-test results (same frozen
shuffle, per-item records are independent), so they are derived here with no
extra GPU time.''')

code(r'''if not SMOKE_TEST:
    _dev_ids = set(split["dev_ids"])
    _n_exports = 0
    for _f in sorted(Path(RESULTS_DIR).glob(f"*__n{N_RUN}.json")):
        _r = json.loads(_f.read_text())
        if not _r.get("completed"):
            continue
        _c = _r["config"]
        _sub = [it for it in _r["per_item"] if it["question_id"] in _dev_ids]
        _valid = [it for it in _sub if not it.get("error")]
        if not _valid:
            continue
        _out = {"config": {**_c, "n": len(_valid), "slice": "dev", "derived_from": _f.name},
            "metrics": {**ec.compute_metrics(_valid), "n_error": len(_sub) - len(_valid)},
            "per_item": _sub, "completed": True}
        _name = ec.result_name(_c["model_id"], _c["condition"], _c["scoring"], "dev")
        _p = ec.save_json(_out, Path(RESULTS_DIR) / _name)
        ec.upload_to_gcs(_p, GCS_BUCKET, f"results/{_name}")
        _n_exports += 1
    print(f"{_n_exports} dev-slice result files derived and uploaded")
else:
    print("SMOKE_TEST=True - dev-slice exports skipped (full-test runs required)")''')

# ------------------------------------------------------------------ 9. manifest

md(r'''## 9. Manifest

Every config value, package version, and artifact path from this run. In smoke
mode the manifest records `smoke_test: true`; the full run overwrites it.''')

code(r'''import importlib.metadata as _im

def _v(p):
    try:
        return _im.version(p)
    except _im.PackageNotFoundError:
        return None

manifest = {
    "created": datetime.now(timezone.utc).isoformat(),
    "notebook": "01 - baseline.ipynb",
    "seed": SEED, "n_ctx": N_CTX, "temperature": TEMPERATURE,
    "smoke_test": SMOKE_TEST, "n_run": N_RUN,
    "gpu": {"name": GPU_NAME, "mem_gb": GPU_MEM_GB,
            "gpu_memory_utilization": GPU_MEM_UTIL, "chunk_questions": CHUNK_QUESTIONS},
    "dataset": {"hf_id": "coastalcph/lex_glue", "config": "case_hold",
                "n_train": len(train), "n_test": len(test)},
    "conditions": CONDITIONS, "scorings": SCORINGS + ["letter_emit_api"],
    "model_panel": MODEL_PANEL, "smoke_models": SMOKE_MODELS, "api_model": API_MODEL,
    "fewshot_train_indices": fewshot_art["train_indices"],
    "grid_condition": GRID_CONDITION,
    "versions": {p: _v(p) for p in ("vllm", "torch", "transformers", "datasets",
                                    "scikit-learn", "google-cloud-storage",
                                    "anthropic", "pandas", "numpy", "pyarrow")},
    "artifacts": {
        "split": f"gs://{GCS_BUCKET}/{ec.SPLIT_BLOB}",
        "fewshot": f"gs://{GCS_BUCKET}/{ec.FEWSHOT_BLOB}",
        "permutations": f"gs://{GCS_BUCKET}/{ec.PERM_BLOB}",
        "summary_csv": f"gs://{GCS_BUCKET}/summary/summary_baseline_n{N_RUN}.csv",
        "grid": f"gs://{GCS_BUCKET}/grid_baseline.parquet",
    },
    "results_in_gcs": ec.gcs_list(GCS_BUCKET, "results/"),
}
_mf = Path(LOCAL_OUT) / "manifest.json"
ec.save_json(manifest, _mf)
print(ec.upload_to_gcs(_mf, GCS_BUCKET, "manifest.json"))
print(json.dumps({k: manifest[k] for k in ("created", "n_run", "smoke_test", "gpu", "versions")}, indent=2))''')


nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10"},
    },
    "cells": CELLS,
}

out = Path(__file__).resolve().parent / "01 - baseline.ipynb"
out.write_text(json.dumps(nb, indent=1))
print(f"wrote {out} ({len(CELLS)} cells)")
