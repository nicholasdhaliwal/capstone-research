#!/usr/bin/env python3
"""Generate "02 - retrieval.ipynb" (same generator pattern as
_build_01_baseline.py). Edit this file, then run `python3 _build_02_retrieval.py`
to regenerate the notebook. No nbformat dependency - plain JSON.

One-time bootstrap: this file was produced by converting the pre-existing,
hand-authored "02 - retrieval.ipynb" into generator form (cell ids preserved
verbatim so the first regeneration is byte-identical to the hand-authored
file - see SESSION_HANDOFF.md item 0). Every edit from here on goes through
this file; the notebook itself is never hand-edited again."""
import json
from pathlib import Path

CELLS = []


def md(cell_id, src):
    CELLS.append({
        "cell_type": "markdown",
        "id": cell_id,
        "metadata": {},
        "source": src.splitlines(keepends=True),
    })


def code(cell_id, src):
    CELLS.append({
        "cell_type": "code",
        "id": cell_id,
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": src.splitlines(keepends=True),
    })


# ------------------------------------------------------------------ 0. 02 - Retrieval Bake-off (CaseHOLD, Q2)
md('8c36b3a8', r'''# 02 - Retrieval Bake-off (CaseHOLD, Q2)

Staged bake-off of four retrievers over the full train-split corpus, scored with the
Notebook 1 protocol (`eval_core`). Retrieved cases are prepended to the prompt
**with their correct holdings**.

**DESIGN REVERSAL - INTENTIONAL, DO NOT "FIX":** Q1 retrieved excerpts *without*
holdings. This notebook deliberately attaches each retrieved case's gold holding to
the retrieved text. The leakage risk is handled by a TF-IDF dedup filter
(threshold in `retrieval_core.LEAKAGE_SIM_THRESHOLD`), not by dropping holdings.

Stages (each gated - run in order, inspect before proceeding):

- **A** - 4 retrievers x Saul-7B-Instruct x DEV (500 q), logprob scoring; CIs + McNemar vs bare
- **B** - cross-encoder rerank (retrieve 50 -> rerank -> top 3) on the top-2 Stage-A retrievers
- **C** - HyDE (optional, `STAGE_C_HYDE=False` by default)
- **D** - single winner x ALL panel models x full test set; `winner.json` is written **before** D runs

All GPU work (encoding, reranking, HyDE, vLLM scoring) runs in subprocesses -
one model per process, per the shared protocol. Every run uploads JSON to GCS
and is resume-safe (existing results are skipped).''')
code('8ba98f80', r'''# ---- config (edit here only) -------------------------------------------------
SMOKE_TEST = True   # True: first 50 DEV questions + 2 small models + 2,000-doc corpus
STAGE_C_HYDE = False  # optional stage; flip deliberately

import json
import os
import subprocess
import sys
from pathlib import Path

import eval_core as ec
import retrieval_core as rc
from eval_core import SEED, N_CTX, TEMPERATURE  # single source of truth

GCS_BUCKET = os.environ.get("GCS_BUCKET", "REPLACE_ME-casehold-eval")  # same bucket as Notebook 1
assert GCS_BUCKET and GCS_BUCKET != "REPLACE_ME-casehold-eval", \
    "Set the GCS_BUCKET env var (same bucket Notebook 1 wrote to) before running."
WORK_DIR = Path("work"); WORK_DIR.mkdir(exist_ok=True)
OUT_DIR = Path("results"); OUT_DIR.mkdir(exist_ok=True)

# SONAR runs from its own venv (fairseq2 must ABI-match torch; vLLM pins torch).
# Point this at the venv python created in the setup cell.
SONAR_PYTHON = os.environ.get("SONAR_PYTHON", sys.executable)

SUBJECT = "Equall/Saul-7B-Instruct-v1"
RETRIEVER_NAMES = ["bm25", "splade", "bge", "sonar"]
STAGE_A_CONDITIONS = [rc.CONDITION_OF[r] for r in RETRIEVER_NAMES]
COND_TO_RETRIEVER = {v: k for k, v in rc.CONDITION_OF.items()}

SMOKE_N = 50 if SMOKE_TEST else None
SMOKE_FLAGS = ["--smoke", "--smoke-n", "50"] if SMOKE_TEST else []
SMOKE_MODELS = ["mistralai/Mistral-7B-Instruct-v0.3", "Qwen/Qwen2.5-7B-Instruct"]
STAGE_D_MODELS = SMOKE_MODELS if SMOKE_TEST else [m["id"] for m in ec.MODEL_PANEL]

def rslice(slice_name):
    """Result-file slice tag (matches retrieval_core.run_rag_eval)."""
    return f"{slice_name}_smoke{SMOKE_N}" if SMOKE_TEST else slice_name

print(f"SEED={SEED}  N_CTX={N_CTX}  TEMPERATURE={TEMPERATURE}")
print(f"bucket=gs://{GCS_BUCKET}  smoke={SMOKE_TEST}  hyde={STAGE_C_HYDE}")
print(f"leakage threshold={rc.LEAKAGE_SIM_THRESHOLD}  K={rc.K_FINAL} of {rc.K_CANDIDATES} candidates")''')
code('ff753e9e', r'''# ---- one-time environment setup (uncomment on a fresh instance) ---------------
# Main env: vLLM scoring + BM25/SPLADE/BGE retrievers + reranker.
# %pip install -q vllm "sentence-transformers>=5.0" rank-bm25 scipy scikit-learn \
#     datasets pyarrow pandas google-cloud-storage
#
# SONAR: fairseq2 wheels are built per exact torch+CUDA variant and vLLM pins its
# own torch, so SONAR lives in a SEPARATE venv (fairseq2 0.5.x has no build for
# the torch recent vLLM pins). Create once, then set SONAR_PYTHON:
# !python -m venv ~/venvs/sonar
# !~/venvs/sonar/bin/pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
# !~/venvs/sonar/bin/pip install "fairseq2~=0.5.2" --extra-index-url https://fair.pkg.atmeta.com/fairseq2/whl/pt2.8.0/cu128
# !~/venvs/sonar/bin/pip install sonar-space==0.5.0 numpy pandas pyarrow datasets scikit-learn scipy google-cloud-storage
# os.environ["SONAR_PYTHON"] = str(Path.home() / "venvs/sonar/bin/python")
#
# Gated repos (accept terms once, HF_TOKEN env var assumed set, as for Llama/Gemma):
# naver/splade-v3 (ungated alternative: naver/splade-v3-distilbert).
print("environment setup cell - uncomment on a fresh instance")''')
code('4a5cb47f', r'''# ---- GPU detection: batch sizes / gpu_memory_utilization adapt to the card ----
gpu = ec.detect_gpu()
gpu''')
code('c5114539', r'''# ---- subprocess driver ---------------------------------------------------------
def run(*args, py=None):
    cmd = [py or sys.executable, "retrieval_core.py", *map(str, args),
           "--bucket", GCS_BUCKET, "--work-dir", str(WORK_DIR),
           "--out-dir", str(OUT_DIR), *SMOKE_FLAGS]
    print(">>", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)

def pyfor(retriever):
    """SONAR subcommands run from the dedicated venv; everything else main env."""
    return SONAR_PYTHON if retriever == "sonar" else None''')
code('e93bd9d4', r'''# ---- data, frozen split, corpus -------------------------------------------------
data = ec.load_casehold()
split = ec.get_split(data["test"], GCS_BUCKET, WORK_DIR)   # loads the frozen artifact; creates only on first-ever run
corpus = rc.get_corpus(GCS_BUCKET, WORK_DIR, smoke=SMOKE_TEST)
print(f"test={len(data['test'])}  dev={split['n_dev']}  confirm={split['n_confirm']}  corpus={len(corpus)} docs")''')

# ------------------------------------------------------------------ 1. Stage A - four retrievers on DEV
md('95ad42e3', r'''## Stage A - four retrievers on DEV

**STAGE GATE:** run the three cells below in order (encode -> contexts -> score),
then inspect the table before touching Stage B. Corpus encoding is the slow step;
it runs once per retriever and is cached to GCS - reruns load the cache.''')
code('ce7f06da', r'''# encode corpora (cached; ~minutes from cache, longer on first build)
for r in RETRIEVER_NAMES:
    run("encode", "--retriever", r, py=pyfor(r))''')
code('8b4aef43', r'''# retrieve + leakage-filter DEV contexts for all four retrievers
for r in RETRIEVER_NAMES:
    run("contexts", "--retriever", r, "--slice", "dev", py=pyfor(r))''')
code('fac25797', r'''# score: 4 conditions x Saul-7B-Instruct x DEV, logprob (primary)
for cond in STAGE_A_CONDITIONS:
    run("score", "--model-id", SUBJECT, "--condition", cond, "--scoring", "logprob", "--slice", "dev")''')
code('2c631e24', r'''# ---- Stage A table: acc_norm + CI + trigger rate + McNemar vs bare -------------
import pandas as pd

def load_dev(cond):
    return rc.load_result(SUBJECT, cond, "logprob", rslice("dev"), GCS_BUCKET, OUT_DIR)

try:  # bare baseline per-item records from Notebook 1's GCS output (same 500 dev questions)
    bare = load_dev("zero_shot")
except AssertionError:
    bare = rc.load_result(SUBJECT, "zero_shot", "logprob", "dev", GCS_BUCKET, OUT_DIR)

dev_results = {"bare (NB1)": bare}
rows = [{"condition": "bare (NB1)", **{k: bare["metrics"][k] for k in ("accuracy", "ci_low", "ci_high", "macro_f1")},
         "trigger_rate": None, "mcnemar_p_vs_bare": None}]
for cond in STAGE_A_CONDITIONS:
    res = load_dev(cond)
    dev_results[cond] = res
    mn = rc.mcnemar(res["per_item"], bare["per_item"])
    rows.append({"condition": cond,
                 **{k: res["metrics"][k] for k in ("accuracy", "ci_low", "ci_high", "macro_f1")},
                 "trigger_rate": res["retrieval_summary"]["trigger_rate_questions"],
                 "mcnemar_p_vs_bare": mn["p_value"]})
stage_a_table = pd.DataFrame(rows).rename(columns={"accuracy": "acc_norm"})
stage_a_table''')
md('a1b2c3d4', r'''### Consolidated per-question log (`retrieval_items_long.csv`)

Same idea as Notebook 1's `all_items_long.csv`: every scored question, across
every RAG result JSON in `results/` so far, in one long table. Re-written
after Stage D with the full-test rows added.''')
code('e5f6a7b8', r'''_panel_meta = {m["id"]: m for m in ec.MODEL_PANEL}
_retr_rows = []
for _f in sorted(OUT_DIR.glob("*.json")):
    _r = json.loads(_f.read_text())
    if not _r.get("completed"):
        continue
    _c = _r["config"]
    _meta = _panel_meta.get(_c["model_id"], {"kind": "api", "role": "api"})
    _summary = _r.get("retrieval_summary", {})
    for _it in _r["per_item"]:
        _retr_rows.append({
            "question_id": _it["question_id"], "model_id": _c["model_id"],
            "kind": _meta["kind"], "role": _meta["role"],
            "condition": _c["condition"], "scoring": _c["scoring"],
            "shuffled_gold": _it.get("shuffled_gold"), "prediction": _it.get("prediction"),
            "correct": _it.get("correct"), "error": bool(_it.get("error", False)),
            "error_type": _it.get("error_type"),
            "retriever": _c.get("retriever"), "rerank": bool(_c.get("reranker")),
            "hyde": bool(_c.get("hyde")), "trigger_rate": _summary.get("trigger_rate_questions"),
        })
retrieval_items_long = pd.DataFrame(_retr_rows)
_retr_csv = Path("outputs") / "retrieval_items_long.csv"
_retr_csv.parent.mkdir(exist_ok=True)
retrieval_items_long.to_csv(_retr_csv, index=False)
print(ec.upload_to_gcs(_retr_csv, GCS_BUCKET, "summary/retrieval_items_long.csv"))
print(f"{len(retrieval_items_long)} rows across {retrieval_items_long['model_id'].nunique()} models")''')

# ------------------------------------------------------------------ 2. Stage B - cross-encoder rerank on the top-2 Stage-A retrievers
md('6a67bd8c', r'''## Stage B - cross-encoder rerank on the top-2 Stage-A retrievers

Pipeline: retrieve 50 candidates -> leakage filter -> rerank with
`BAAI/bge-reranker-v2-m3` -> top 3. Same DEV protocol.

**STAGE GATE:** confirm the Stage A table above looks sane (trigger rate is a
reportable statistic - eyeball it) before running.''')
code('cb541c81', r'''acc_of = lambda cond: dev_results[cond]["metrics"]["accuracy"]
top2 = sorted(STAGE_A_CONDITIONS, key=acc_of, reverse=True)[:2]
print("top-2 Stage-A retrievers:", top2)

for cond in top2:
    r = COND_TO_RETRIEVER[cond]
    run("contexts", "--retriever", r, "--slice", "dev", "--k-store", str(rc.K_CANDIDATES), py=pyfor(r))
    run("rerank", "--condition", cond, "--slice", "dev")
    run("score", "--model-id", SUBJECT, "--condition", f"{cond}_rerank", "--scoring", "logprob", "--slice", "dev")''')
code('6f5809b6', r'''for cond in [f"{c}_rerank" for c in top2]:
    res = load_dev(cond)
    dev_results[cond] = res
    mn = rc.mcnemar(res["per_item"], bare["per_item"])
    stage_a_table.loc[len(stage_a_table)] = {
        "condition": cond,
        "acc_norm": res["metrics"]["accuracy"], "ci_low": res["metrics"]["ci_low"],
        "ci_high": res["metrics"]["ci_high"], "macro_f1": res["metrics"]["macro_f1"],
        "trigger_rate": res["retrieval_summary"]["trigger_rate_questions"],
        "mcnemar_p_vs_bare": mn["p_value"]}
stage_a_table''')

# ------------------------------------------------------------------ 3. Stage C (optional, OFF by default) - HyDE
md('aac591a1', r'''## Stage C (optional, OFF by default) - HyDE

Saul-Instruct writes a hypothetical holding per query excerpt (temperature 0,
max 128 tokens); the hypothesis is embedded instead of the raw excerpt, using
the best dense retriever from Stage A. Enable via `STAGE_C_HYDE = True` in the
config cell.''')
code('c58201df', r'''if STAGE_C_HYDE:
    best_dense_cond = max(["rag_bge", "rag_sonar"], key=acc_of)
    r = COND_TO_RETRIEVER[best_dense_cond]
    hyde_cond = f"rag_hyde_{r}"
    run("hyde", "--slice", "dev")
    hyde_file = WORK_DIR / "hyde_dev_v1.json"
    run("contexts", "--retriever", r, "--slice", "dev",
        "--query-file", str(hyde_file), "--condition", hyde_cond, py=pyfor(r))
    run("score", "--model-id", SUBJECT, "--condition", hyde_cond, "--scoring", "logprob", "--slice", "dev")
    res = load_dev(hyde_cond)
    dev_results[hyde_cond] = res
    mn = rc.mcnemar(res["per_item"], bare["per_item"])
    stage_a_table.loc[len(stage_a_table)] = {
        "condition": hyde_cond,
        "acc_norm": res["metrics"]["accuracy"], "ci_low": res["metrics"]["ci_low"],
        "ci_high": res["metrics"]["ci_high"], "macro_f1": res["metrics"]["macro_f1"],
        "trigger_rate": res["retrieval_summary"]["trigger_rate_questions"],
        "mcnemar_p_vs_bare": mn["p_value"]}
    display(stage_a_table)
else:
    print("Stage C skipped (STAGE_C_HYDE=False)")''')

# ------------------------------------------------------------------ 4. Winner selection - written to GCS BEFORE Stage D runs
md('2acb2174', r'''## Winner selection - written to GCS BEFORE Stage D runs

Highest DEV `acc_norm`; ties break toward the simpler pipeline
(bare retriever < rerank < HyDE; BM25 simplest among bases).''')
code('298d394f', r'''def complexity(cond):
    c = 0
    if cond.endswith("_rerank"): c += 1
    if "hyde" in cond: c += 2
    if not cond.startswith("rag_bm25"): c += 1   # neural base costs more than BM25
    return c

candidates = {c: r for c, r in dev_results.items() if c.startswith("rag_")}
winner_cond = sorted(candidates, key=lambda c: (-acc_of(c), complexity(c), c))[0]
choice = {
    "condition": winner_cond,
    "retriever": COND_TO_RETRIEVER.get(winner_cond.replace("_rerank", "").replace("rag_hyde_", "rag_")),
    "rerank": winner_cond.endswith("_rerank"),
    "hyde": "hyde" in winner_cond,
    "chosen_by": "highest DEV acc_norm; tie -> simpler pipeline",
    "dev_acc_norm": {c: acc_of(c) for c in candidates},
}
print(json.dumps(choice, indent=2))
print(rc.write_winner(choice, GCS_BUCKET, WORK_DIR))   # <- artifact exists before Stage D''')

# ------------------------------------------------------------------ 5. Stage D - winner x ALL panel models x full test set
md('65a67ab9', r'''## Stage D - winner x ALL panel models x full test set

**STAGE GATE:** `winner.json` must exist in GCS (previous cell) before anything
below runs. Each model runs in its own subprocess; already-finished results are
skipped, so this cell is safe to re-run after an interruption.''')
code('ab01714d', r'''winner = json.loads((WORK_DIR / Path(rc.WINNER_BLOB).name).read_text())
cond = winner["condition"]
base_cond = cond.replace("_rerank", "")
r = COND_TO_RETRIEVER.get(base_cond, winner.get("retriever"))

# build full-test contexts for the winning pipeline
if winner.get("hyde"):
    run("hyde", "--slice", "test")
    run("contexts", "--retriever", r, "--slice", "test",
        "--query-file", str(WORK_DIR / "hyde_test_v1.json"), "--condition", cond, py=pyfor(r))
elif winner.get("rerank"):
    run("contexts", "--retriever", r, "--slice", "test", "--k-store", str(rc.K_CANDIDATES), py=pyfor(r))
    run("rerank", "--condition", base_cond, "--slice", "test")
else:
    run("contexts", "--retriever", r, "--slice", "test", py=pyfor(r))

for model_id in STAGE_D_MODELS:   # one subprocess per model; resume-safe
    run("score", "--model-id", model_id, "--condition", cond, "--scoring", "logprob", "--slice", "test")''')

# ------------------------------------------------------------------ 6. Summary
md('2162f2d4', r'''## Summary''')
code('5a7708b1', r'''# DEV summary: bare (Notebook 1) vs Stage A / B / C
stage_a_table.sort_values("acc_norm", ascending=False)''')
code('2eaa4354', r'''# Full-test summary: Stage D across the panel
rows = []
for model_id in STAGE_D_MODELS:
    res = rc.load_result(model_id, cond, "logprob", rslice("test"), GCS_BUCKET, OUT_DIR)
    rows.append({"model": model_id, "acc_norm": res["metrics"]["accuracy"],
                 "acc": res["metrics"].get("accuracy_acc"),
                 "ci_low": res["metrics"]["ci_low"], "ci_high": res["metrics"]["ci_high"],
                 "macro_f1": res["metrics"]["macro_f1"], "n": res["config"]["n"]})
stage_d_table = pd.DataFrame(rows).sort_values("acc_norm", ascending=False)
stage_d_table''')
md('c9d0e1f2', r'''### Consolidated per-question log, updated (`retrieval_items_long.csv`)

Re-run of the Stage-A cell above now that Stage D's full-test results exist
in `results/` too - same file, now comprehensive.''')
code('f3a4b5c6', r'''_panel_meta = {m["id"]: m for m in ec.MODEL_PANEL}
_retr_rows = []
for _f in sorted(OUT_DIR.glob("*.json")):
    _r = json.loads(_f.read_text())
    if not _r.get("completed"):
        continue
    _c = _r["config"]
    _meta = _panel_meta.get(_c["model_id"], {"kind": "api", "role": "api"})
    _summary = _r.get("retrieval_summary", {})
    for _it in _r["per_item"]:
        _retr_rows.append({
            "question_id": _it["question_id"], "model_id": _c["model_id"],
            "kind": _meta["kind"], "role": _meta["role"],
            "condition": _c["condition"], "scoring": _c["scoring"],
            "shuffled_gold": _it.get("shuffled_gold"), "prediction": _it.get("prediction"),
            "correct": _it.get("correct"), "error": bool(_it.get("error", False)),
            "error_type": _it.get("error_type"),
            "retriever": _c.get("retriever"), "rerank": bool(_c.get("reranker")),
            "hyde": bool(_c.get("hyde")), "trigger_rate": _summary.get("trigger_rate_questions"),
        })
retrieval_items_long = pd.DataFrame(_retr_rows)
_retr_csv = Path("outputs") / "retrieval_items_long.csv"
_retr_csv.parent.mkdir(exist_ok=True)
retrieval_items_long.to_csv(_retr_csv, index=False)
print(ec.upload_to_gcs(_retr_csv, GCS_BUCKET, "summary/retrieval_items_long.csv"))
print(f"{len(retrieval_items_long)} rows across {retrieval_items_long['model_id'].nunique()} models")''')

# ------------------------------------------------------------------ 7. Outputs
md('74940f70', r'''### Outputs

- `gs://{bucket}/results/*.json` - one file per model x condition x scoring, Notebook 1
  schema plus `retrieved` (doc ids + scores), `filter_triggers`, `truncations` per item,
  and a `retrieval_summary` block (leakage-filter trigger rate is a reportable statistic).
  Difficulty-band analysis in Notebook 3 consumes `per_item` directly.
- `gs://{bucket}/artifacts/retrieval/` - corpus parquet, cached indexes/embeddings
  (+ ID-order files), contexts artifacts, `winner_v1.json`.''')


nb = {
    "cells": CELLS,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out = Path(__file__).resolve().parent / "02 - retrieval.ipynb"
out.write_text(json.dumps(nb, indent=1))
print(f"wrote {out} ({len(CELLS)} cells)")
