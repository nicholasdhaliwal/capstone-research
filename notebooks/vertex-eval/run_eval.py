#!/usr/bin/env python
"""run_eval.py - evaluate ONE model x condition x scoring in an isolated process.

The notebook shells out to this script once per run because vLLM does not
reliably release GPU memory in-kernel; process exit returns it. All reusable
logic lives in eval_core - this file is CLI plumbing, chunking, resume, and
the final result JSON.

Usage:
    python run_eval.py --model_id mistralai/Mistral-7B-Instruct-v0.3 \
        --condition zero_shot --scoring logprob --n 3600 --output_path out.json

Env:
    GPU_MEMORY_UTILIZATION  overrides detect_gpu()'s value (set by the notebook)
    CHUNK_QUESTIONS         questions per generate() call / incremental-save unit (default 500)
    EVAL_N_CTX              context window (default eval_core.N_CTX)
    ARTIFACTS_DIR           local dir holding the frozen few-shot artifact
    GCS_BUCKET              only consulted if an artifact is missing locally
    HF_TOKEN                gated checkpoints (Llama, Gemma)

dtype note: dtype follows the checkpoint ("auto") rather than a per-GPU rule.
gemma-2 must stay bf16 (fp16 breaks its logit soft-capping) and L4 is Ada, so
bf16 is fully supported there. The dtype actually used is recorded in the
result config.

Resume: scored items stream to <output_path>.items.jsonl after every chunk; a
crash loses at most one chunk. Re-running skips completed question ids, and a
finished output JSON (completed=true) exits immediately.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_core as ec


def log(msg):
    # stderr + flush: notebooks buffer stdout while a cell is busy (Q1 lesson)
    print(msg, file=sys.stderr, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", required=True)
    ap.add_argument("--condition", required=True, choices=["zero_shot", "few_shot"])
    ap.add_argument("--scoring", required=True, choices=["letter_emit", "logprob"])
    ap.add_argument("--n", type=int, required=True, help="first n test questions")
    ap.add_argument("--output_path", required=True)
    args = ap.parse_args()

    out_path = Path(args.output_path)
    if out_path.exists():
        prev = json.loads(out_path.read_text())
        cfg = prev.get("config", {})
        expected = {"model_id": args.model_id, "condition": args.condition, "scoring": args.scoring}
        mismatched = {k: v for k, v in expected.items() if cfg.get(k) != v}
        if mismatched:
            raise SystemExit(f"{out_path} holds a different run ({mismatched}) - refusing to mix")
        if prev.get("completed"):
            log(f"already completed: {out_path}")
            return
    jsonl_path = out_path.with_suffix(".items.jsonl")

    kind = ec.panel_entry(args.model_id)["kind"]
    n_ctx = int(os.environ.get("EVAL_N_CTX", ec.N_CTX))
    chunk_size = int(os.environ.get("CHUNK_QUESTIONS", "500"))
    artifacts_dir = os.environ.get("ARTIFACTS_DIR", str(out_path.parent.parent / "artifacts"))
    bucket = os.environ.get("GCS_BUCKET", "")

    log(f"=== {args.model_id} | {args.condition} | {args.scoring} | n={args.n} | kind={kind} ===")

    data = ec.load_casehold()
    test = data["test"]
    slice_ds = ec.select_slice(test, None, "test", smoke_n=args.n if args.n < len(test) else None)
    fewshot_art = None
    if args.condition == "few_shot":
        fewshot_art = ec.get_fewshot(data["train"], bucket, artifacts_dir)
    questions = ec.prepare_questions(slice_ds, args.condition, fewshot_art)

    done_items, done_ids = ec.load_done(jsonl_path)
    todo = [q for q in questions if q["question_id"] not in done_ids]
    log(f"{len(questions)} questions | {len(done_items)} resumed from jsonl | {len(todo)} to run")

    gpu_cfg = ec.detect_gpu()
    if os.environ.get("GPU_MEMORY_UTILIZATION"):
        gpu_cfg["gpu_memory_utilization"] = float(os.environ["GPU_MEMORY_UTILIZATION"])
    log(f"GPU: {gpu_cfg}")

    llm = ec.make_llm(args.model_id, gpu_cfg, n_ctx=n_ctx)
    tokenizer = llm.get_tokenizer()

    todo, error_items = ec.preflight(todo, tokenizer, args.scoring, n_ctx)
    for e in error_items:
        log(f"context overflow, skipping q{e['question_id']} (~{e['n_tokens_estimate']} tokens)")

    new_items = []
    for start in range(0, len(todo), chunk_size):
        chunk = todo[start:start + chunk_size]
        if args.scoring == "letter_emit":
            items = ec.run_letter_emit(llm, chunk, kind, jsonl_path=jsonl_path)
        else:
            items = ec.run_logprob(llm, chunk, jsonl_path=jsonl_path)
            if start == 0 and items:
                # historical uniform-logit symptom: every question scoring all
                # 5 options identically means the logprob path is broken
                degenerate = [
                    len({round(v, 6) for v in it["sum_logp"].values()}) == 1 for it in items
                ]
                if all(degenerate):
                    raise RuntimeError(
                        "all 5 options scored identically for every question in the first "
                        "chunk - uniform-logit symptom; aborting instead of writing garbage"
                    )
        new_items.extend(items)
        log(f"{len(done_items) + len(new_items)}/{len(questions)} scored")

    per_item = done_items + new_items
    per_item.sort(key=lambda r: (len(r["question_id"]), r["question_id"]))  # numeric order for digit ids
    valid = [r for r in per_item if not r.get("error")]

    result = ec.build_result(
        args.model_id, args.condition, args.scoring, valid,
        extra_config={
            "n_requested": int(args.n),
            "n_ctx": int(n_ctx),
            "gpu_memory_utilization": float(gpu_cfg["gpu_memory_utilization"]),
            "kind": kind,
            "chunk_questions": int(chunk_size),
        },
    )
    result["metrics"]["n_error"] = len(error_items)
    result["per_item"] = valid + error_items
    result["completed"] = True
    ec.save_json(result, out_path)
    log(f"saved {out_path}")
    log(f"metrics: {json.dumps(result['metrics'])}")

    del llm
    gc.collect()


if __name__ == "__main__":
    main()
