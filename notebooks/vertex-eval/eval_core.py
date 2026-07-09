"""eval_core.py - shared logic for the Q2 CaseHOLD evaluation stack (Vertex AI Workbench).

Imported by run_eval.py (subprocess worker) and notebooks 02/03. No copy-pasted
logic in notebooks: data loading, split management, shuffling, prompt building,
vLLM scoring, metrics, JSON IO, and GCS upload all live here.

Engine: vLLM only. No llama-cpp, no Ollama, no GGUF, no lm-eval-harness.
Retrieval similarity (future notebooks): numpy L2-normalize + matmul, never FAISS.
"""

import hashlib
import json
import random
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------- constants

SEED = 42  # single seed constant; always pass as int() - some libs reject numpy int64 on py3.12
N_CTX = 8192
TEMPERATURE = 0.0
LETTERS = "ABCDE"
N_OPTIONS = 5
N_DEV = 500          # dev slice for pipeline selection
N_TEST_EXPECTED = 3600
N_FEWSHOT = 3
N_BOOTSTRAP = 10_000

SPLIT_BLOB = "artifacts/dev_confirm_split_v1.json"   # GCS path of the frozen split
FEWSHOT_BLOB = "artifacts/fewshot_v1.json"           # GCS path of the frozen few-shot set

# Prompt wording adapted from the Q1 protocol - keep close to it.
SYSTEM_LETTER = (
    "You are a legal expert. Read the case excerpt and respond with exactly one "
    "letter (A, B, C, D, or E) corresponding to the correct holding."
)
LOGPROB_STEM = "The correct holding is:"

# ------------------------------------------------------------- model panel
# kind: "base" -> raw string prompts (never send chat-template markup)
#       "instruct" -> model's own chat template via vLLM chat API
MODEL_PANEL = [
    # subjects
    {"id": "Equall/Saul-7B-Base",                        "kind": "base",     "role": "subject"},
    {"id": "Equall/Saul-7B-Instruct-v1",                 "kind": "instruct", "role": "subject"},
    {"id": "mistralai/Mistral-7B-Instruct-v0.3",         "kind": "instruct", "role": "subject"},
    # panel
    {"id": "meta-llama/Llama-3.1-8B-Instruct",           "kind": "instruct", "role": "panel"},
    {"id": "meta-llama/Llama-2-7b-hf",                   "kind": "base",     "role": "panel"},
    {"id": "Qwen/Qwen3-8B",                              "kind": "instruct", "role": "panel"},
    {"id": "Qwen/Qwen2.5-7B-Instruct",                   "kind": "instruct", "role": "panel"},
    {"id": "google/gemma-2-9b-it",                       "kind": "instruct", "role": "panel"},
    # substitution: spec said microsoft/Phi-4 but the canonical HF id is lowercase
    # microsoft/phi-4 (capital-P 307-redirects); fallback microsoft/Phi-3.5-mini-instruct
    {"id": "microsoft/phi-4",                            "kind": "instruct", "role": "panel"},
    # caveat: R1-distill chat template force-opens <think>, so constrained letter-emit
    # grades a no-reasoning completion; logprob scoring (primary) is unaffected
    {"id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",    "kind": "instruct", "role": "panel"},
]

def panel_entry(model_id):
    for m in MODEL_PANEL:
        if m["id"] == model_id:
            return m
    raise KeyError(f"{model_id} not in MODEL_PANEL")


# ------------------------------------------------------------ JSON helpers

def json_default(o):
    """Cast numpy scalars/arrays to Python types. float32 is not JSON serializable."""
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not JSON serializable: {type(o)}")


def save_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, default=json_default)
    tmp.replace(path)
    return path


def append_jsonl(record, path):
    """Incremental per-item persistence so a crash at hour N loses nothing."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(record, default=json_default) + "\n")


def load_done(jsonl_path):
    """Items already scored per an append_jsonl sidecar; resume reads this
    before scoring more. Returns (items, seen_question_ids)."""
    items, seen = [], set()
    if Path(jsonl_path).exists():
        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if rec["question_id"] not in seen:
                    seen.add(rec["question_id"])
                    items.append(rec)
    return items, seen


# --------------------------------------------------------------------- GCS

def upload_to_gcs(local_path, bucket_name, blob_path):
    from google.cloud import storage
    client = storage.Client()
    blob = client.bucket(bucket_name).blob(blob_path)
    blob.upload_from_filename(str(local_path))
    return f"gs://{bucket_name}/{blob_path}"


def download_from_gcs(bucket_name, blob_path, local_path):
    from google.cloud import storage
    client = storage.Client()
    blob = client.bucket(bucket_name).blob(blob_path)
    if not blob.exists():
        return None
    Path(local_path).parent.mkdir(parents=True, exist_ok=True)
    blob.download_to_filename(str(local_path))
    return Path(local_path)


def gcs_list(bucket_name, prefix):
    from google.cloud import storage
    client = storage.Client()
    return sorted(b.name for b in client.list_blobs(bucket_name, prefix=prefix))


# ------------------------------------------------------------ data loading

def _normalize(ds):
    """Map any CaseHOLD variant to canonical columns:
    example_id (str), citing_prompt (str), holdings (list[5] of str), label (int 0-4)."""
    cols = set(ds.column_names)
    if {"citing_prompt", "holding_0"}.issubset(cols):          # casehold/casehold
        def f(row, idx):
            return {
                "example_id": str(row.get("example_id", idx)),
                "citing_prompt": row["citing_prompt"],
                "holdings": [row[f"holding_{i}"] for i in range(5)],
                "label": int(row["label"]),
            }
    elif {"context", "endings"}.issubset(cols):                # lex_glue case_hold
        def f(row, idx):
            return {
                "example_id": str(idx),
                "citing_prompt": row["context"],
                "holdings": list(row["endings"]),
                "label": int(row["label"]),
            }
    else:
        raise ValueError(f"unrecognized CaseHOLD columns: {sorted(cols)}")
    out = ds.map(f, with_indices=True, remove_columns=ds.column_names)
    assert set(out.column_names) == {"example_id", "citing_prompt", "holdings", "label"}
    return out


def load_casehold_csv():
    """Historical fallback: raw casehold CSVs with positional column names.

    WARNING (verified 2026-07): this source is a DIFFERENT partition of the
    underlying data (test=5,314, not the standard 3,600) - the split-size
    assert in load_casehold() will stop the run if this path is taken.
    Columns 6-10 are float similarity scores, dropped by the original script.
    """
    from datasets import load_dataset
    base = "https://huggingface.co/datasets/casehold/casehold/resolve/main/data/all/"
    ds = load_dataset("csv", data_files={
        "train": base + "train.csv", "validation": base + "val.csv", "test": base + "test.csv",
    })
    rename = {"Unnamed: 0": "example_id", "0": "citing_prompt",
              "1": "holding_0", "2": "holding_1", "3": "holding_2",
              "4": "holding_3", "5": "holding_4", "11": "label"}
    for split in ds:
        assert all(c in ds[split].column_names for c in rename), \
            f"{split}: unexpected CSV columns {ds[split].column_names}"
    return ds.rename_columns(rename).remove_columns(["6", "7", "8", "9", "10"])


def load_casehold():
    """Returns dict with 'train' and 'test' datasets in canonical schema.

    Primary (verified working on datasets>=3): coastalcph/lex_glue 'case_hold',
    which is script-free parquet with the standard splits (45,000 / 3,900 /
    3,600). 'casehold/casehold' is loader-script only and raises
    "Dataset scripts are no longer supported" on modern datasets, so it is
    not attempted; the raw-CSV fallback exists for emergencies only.
    """
    from datasets import load_dataset
    try:
        ds = load_dataset("coastalcph/lex_glue", "case_hold")
    except Exception as e:
        print(f"lex_glue load failed ({e}); falling back to raw casehold CSVs")
        ds = load_casehold_csv()
    out = {k: _normalize(ds[k]) for k in ("train", "test")}
    n_test = len(out["test"])
    assert n_test == N_TEST_EXPECTED, (
        f"expected {N_TEST_EXPECTED} test questions, got {n_test} - wrong source/partition; "
        "do not evaluate on this split without a protocol decision"
    )
    sample = out["test"][0]
    assert len(sample["holdings"]) == 5 and 0 <= sample["label"] <= 4
    return out


# --------------------------------------------------- dev / confirm split

def get_split(test_ds, bucket_name, local_dir):
    """Load the frozen DEV/CONFIRM partition; create it exactly once.

    The artifact is stored in GCS the first time it is created. Every later
    run loads that artifact - never regenerate it.
    """
    local_path = Path(local_dir) / Path(SPLIT_BLOB).name
    if not local_path.exists():
        download_from_gcs(bucket_name, SPLIT_BLOB, local_path)
    all_ids = [str(x) for x in test_ds["example_id"]]
    if local_path.exists():
        art = json.loads(local_path.read_text())
        assert art["seed"] == SEED
        assert set(art["dev_ids"]) | set(art["confirm_ids"]) == set(all_ids), \
            "split artifact does not match the loaded test set"
        return art
    # first-ever creation
    rng = random.Random(int(SEED))
    dev_ids = sorted(rng.sample(all_ids, N_DEV))
    dev_set = set(dev_ids)
    confirm_ids = [i for i in all_ids if i not in dev_set]
    art = {
        "seed": int(SEED),
        "n_dev": len(dev_ids),
        "n_confirm": len(confirm_ids),
        "created": datetime.now(timezone.utc).isoformat(),
        "dev_ids": dev_ids,
        "confirm_ids": confirm_ids,
    }
    save_json(art, local_path)
    upload_to_gcs(local_path, bucket_name, SPLIT_BLOB)
    return art


def select_slice(test_ds, split_art, which, smoke_n=None):
    """which in {'dev','confirm','test'} ('test' = full test set);
    smoke_n truncates (SMOKE_TEST mode)."""
    if which == "test":
        idxs = list(range(len(test_ds)))
    else:
        ids = set(split_art[f"{which}_ids"])
        idxs = [i for i, eid in enumerate(test_ds["example_id"]) if str(eid) in ids]
    if smoke_n:
        idxs = idxs[: int(smoke_n)]
    return test_ds.select(idxs)


# ------------------------------------------------------ shuffling / prompts

def shuffle_options(holdings, label, example_id):
    """Per-question permutation from an RNG derived from SEED + question id.

    Mandatory in every condition (documented A-bias: 80-90% 'A' picks on
    unshuffled retrieval prompts). Returns (shuffled_holdings, permutation,
    gold_pos) where permutation[i] = original option index shown at position i.
    """
    q = int(example_id) if str(example_id).isdigit() else \
        int(hashlib.sha1(str(example_id).encode()).hexdigest(), 16) % (2**31)
    rng = random.Random(int(SEED) + q)
    perm = list(range(N_OPTIONS))
    rng.shuffle(perm)
    shuffled = [holdings[p] for p in perm]
    gold_pos = perm.index(int(label))
    return shuffled, perm, gold_pos


def format_options(shuffled_holdings):
    return "\n".join(f"{LETTERS[i]}. {h}" for i, h in enumerate(shuffled_holdings))


def build_letter_user(excerpt, shuffled_holdings):
    return f"{excerpt}\n\n{format_options(shuffled_holdings)}\n\nAnswer:"


def get_fewshot(train_ds, bucket_name, local_dir):
    """Fixed few-shot set drawn from TRAIN (never evaluated on), chosen by seed,
    same for all models. Frozen as a GCS artifact like the split."""
    local_path = Path(local_dir) / Path(FEWSHOT_BLOB).name
    if not local_path.exists():
        download_from_gcs(bucket_name, FEWSHOT_BLOB, local_path)
    if local_path.exists():
        return json.loads(local_path.read_text())
    rng = random.Random(int(SEED) + 1)  # offset so it never collides with the split draw
    idxs = rng.sample(range(len(train_ds)), N_FEWSHOT)
    examples = []
    for i in idxs:
        row = train_ds[i]
        shuffled, perm, gold_pos = shuffle_options(row["holdings"], row["label"], row["example_id"])
        examples.append({
            "example_id": str(row["example_id"]),
            "citing_prompt": row["citing_prompt"],
            "shuffled_holdings": shuffled,
            "permutation": perm,
            "gold_pos": gold_pos,
        })
    art = {"seed": int(SEED), "train_indices": idxs, "examples": examples}
    save_json(art, local_path)
    upload_to_gcs(local_path, bucket_name, FEWSHOT_BLOB)
    return art


PERM_BLOB = "artifacts/permutations_v1.json"  # frozen shuffle, shared by every notebook


def build_permutation_table(test_ds):
    """Permutation + shuffled gold letter for every question, keyed by
    question_id (str). Pure function of SEED and the question ids/labels, so
    recomputing must reproduce the frozen artifact exactly."""
    table = {}
    for row in test_ds:
        _, perm, gold_pos = shuffle_options(row["holdings"], row["label"], row["example_id"])
        table[str(row["example_id"])] = {
            "permutation": perm,
            "shuffled_gold": LETTERS[gold_pos],
        }
    return table


def get_permutations(test_ds, bucket_name, local_dir):
    """Frozen per-question permutation table (same create-once pattern as the
    split artifact). Every later notebook loads this identical shuffle."""
    local_path = Path(local_dir) / Path(PERM_BLOB).name
    if not local_path.exists():
        download_from_gcs(bucket_name, PERM_BLOB, local_path)
    if local_path.exists():
        art = json.loads(local_path.read_text())
        assert art["seed"] == SEED
        return art
    art = {
        "seed": int(SEED),
        "created": datetime.now(timezone.utc).isoformat(),
        "table": build_permutation_table(test_ds),
    }
    save_json(art, local_path)
    upload_to_gcs(local_path, bucket_name, PERM_BLOB)
    return art


def fewshot_letter_prefix(fewshot_art):
    blocks = []
    for ex in fewshot_art["examples"]:
        blocks.append(
            f"{ex['citing_prompt']}\n\n{format_options(ex['shuffled_holdings'])}\n\n"
            f"Answer: {LETTERS[ex['gold_pos']]}"
        )
    return "\n\n---\n\n".join(blocks) + "\n\n---\n\n"


def fewshot_logprob_prefix(fewshot_art):
    blocks = []
    for ex in fewshot_art["examples"]:
        gold_text = ex["shuffled_holdings"][ex["gold_pos"]]
        blocks.append(f"{ex['citing_prompt']}\n\n{LOGPROB_STEM} {gold_text}")
    return "\n\n---\n\n".join(blocks) + "\n\n---\n\n"


def prepare_questions(slice_ds, condition, fewshot_art=None):
    """Materialize everything needed to score one question, per scoring method."""
    letter_prefix = fewshot_letter_prefix(fewshot_art) if condition == "few_shot" else ""
    logprob_prefix = fewshot_logprob_prefix(fewshot_art) if condition == "few_shot" else ""
    out = []
    for row in slice_ds:
        shuffled, perm, gold_pos = shuffle_options(row["holdings"], row["label"], row["example_id"])
        user = letter_prefix + build_letter_user(row["citing_prompt"], shuffled)
        stem = logprob_prefix + f"{row['citing_prompt']}\n\n{LOGPROB_STEM}"
        out.append({
            "question_id": str(row["example_id"]),
            "shuffled_holdings": shuffled,
            "permutation": perm,
            "gold_pos": int(gold_pos),
            "letter_user": user,
            "logprob_stem": stem,  # option text is appended per candidate
        })
    return out


# ---------------------------------------------------------- GPU detection

def detect_gpu():
    """Instance-adaptive settings; we don't know yet if this is A100-80GB,
    A100-40GB, or L4."""
    import torch
    assert torch.cuda.is_available(), "no CUDA device visible"
    name = torch.cuda.get_device_name(0)
    mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    if mem_gb >= 70:      # A100-80GB
        cfg = {"gpu_memory_utilization": 0.92, "max_num_seqs": 64, "enforce_eager": False}
    elif mem_gb >= 35:    # A100-40GB
        cfg = {"gpu_memory_utilization": 0.90, "max_num_seqs": 32, "enforce_eager": False}
    else:                 # L4 24GB: 7-9B bf16 weights ~16GB; small batch, skip CUDA
        cfg = {"gpu_memory_utilization": 0.85, "max_num_seqs": 8, "enforce_eager": True}
    # dtype "auto" follows the checkpoint (Gemma-2 must stay bf16 - fp16 breaks
    # its logit soft-capping; L4 is Ada, bf16 fully supported)
    cfg.update({"name": name, "mem_gb": round(mem_gb, 1), "dtype": "auto"})
    return cfg


# ------------------------------------------------------------ vLLM scoring
# vllm imports are lazy: eval_core is also imported by the API notebook,
# which runs without vLLM in the kernel.

def make_llm(model_id, gpu_cfg=None, n_ctx=N_CTX):
    from vllm import LLM
    gpu_cfg = gpu_cfg or detect_gpu()
    kwargs = dict(
        model=model_id,
        max_model_len=int(n_ctx),
        gpu_memory_utilization=float(gpu_cfg["gpu_memory_utilization"]),
        max_num_seqs=int(gpu_cfg["max_num_seqs"]),
        dtype=gpu_cfg["dtype"],
        seed=int(SEED),  # explicit int cast - never numpy int64
        enforce_eager=bool(gpu_cfg["enforce_eager"]),
        # CRITICAL: vLLM V1 does not reliably support prompt_logprobs with
        # prefix caching (open RFC #13414), and the 5-options-per-question
        # logprob layout shares long prefixes - the worst case. Silent wrong
        # logprobs are exactly the Q1 uniform-logit failure class, so caching
        # stays off for every run.
        enable_prefix_caching=False,
    )
    # Pin the xgrammar backend explicitly (default 'auto' is version-dependent).
    # vLLM >=0.12 uses structured_outputs_config; older used guided_decoding_backend.
    try:
        return LLM(structured_outputs_config={"backend": "xgrammar"}, **kwargs)
    except TypeError:
        pass
    try:
        return LLM(guided_decoding_backend="xgrammar", **kwargs)
    except TypeError:
        return LLM(**kwargs)


def letter_sampling_params():
    """Generation constrained to a single letter A-E via structured outputs
    (xgrammar backend is the vLLM default). Compat shim across the
    guided_decoding -> structured_outputs rename."""
    from vllm import SamplingParams
    common = dict(temperature=float(TEMPERATURE), max_tokens=4, seed=int(SEED))
    try:
        from vllm.sampling_params import StructuredOutputsParams
        return SamplingParams(
            structured_outputs=StructuredOutputsParams(choice=list(LETTERS)), **common
        )
    except ImportError:
        from vllm.sampling_params import GuidedDecodingParams
        return SamplingParams(
            guided_decoding=GuidedDecodingParams(choice=list(LETTERS)), **common
        )


def parse_letter(text):
    m = re.search(r"[ABCDE]", text.strip().upper())
    return m.group(0) if m else None


def preflight(questions, tokenizer, scoring, n_ctx):
    """Q1's qi=45 had a ~23k-token citing_prompt; catch overflow before the
    engine does. Overlong questions become error records (never fabricated
    predictions). Returns (ok_questions, error_items)."""
    ok, errors = [], []
    for q in questions:
        if scoring == "letter_emit":
            # +64: headroom for chat-template wrapper tokens and generation
            n_tok = len(tokenizer(q["letter_user"])["input_ids"]) + 64
        else:
            longest = max(q["shuffled_holdings"], key=len)
            n_tok = len(tokenizer(f"{q['logprob_stem']} {longest}")["input_ids"]) + 8
        if n_tok > n_ctx:
            errors.append({
                "question_id": q["question_id"],
                "error": True,
                "error_type": "context_overflow",
                "n_tokens_estimate": int(n_tok),
                "shuffled_gold": LETTERS[q["gold_pos"]],
                "permutation": q["permutation"],
            })
        else:
            ok.append(q)
    return ok, errors


def run_letter_emit(llm, questions, kind, jsonl_path=None):
    """Scoring method 1: constrained single-letter generation."""
    sp = letter_sampling_params()
    if kind == "instruct":
        def _convs(with_system):
            if with_system:
                return [
                    [
                        {"role": "system", "content": SYSTEM_LETTER},
                        {"role": "user", "content": q["letter_user"]},
                    ]
                    for q in questions
                ]
            return [
                [{"role": "user", "content": f"{SYSTEM_LETTER}\n\n{q['letter_user']}"}]
                for q in questions
            ]

        def _chat(conversations):
            try:
                return llm.chat(conversations, sp, chat_template_kwargs={"enable_thinking": False})
            except (TypeError, ValueError):
                # models whose template has no enable_thinking kwarg
                return llm.chat(conversations, sp)

        import jinja2

        try:
            outs = _chat(_convs(True))
        except jinja2.exceptions.TemplateError:
            # gemma-2's chat template raises on a system role; the template
            # error fires before any generation, so the retry is cheap.
            outs = _chat(_convs(False))
    else:
        prompts = [f"{SYSTEM_LETTER}\n\n{q['letter_user']}" for q in questions]
        outs = llm.generate(prompts, sp)

    per_item = []
    for q, o in zip(questions, outs):
        raw = o.outputs[0].text
        pred = parse_letter(raw)
        rec = {
            "question_id": q["question_id"],
            "shuffled_gold": LETTERS[q["gold_pos"]],
            "prediction": pred,
            "correct": bool(pred == LETTERS[q["gold_pos"]]),
            "permutation": q["permutation"],
            "raw_output": raw,
        }
        per_item.append(rec)
        if jsonl_path:
            append_jsonl(rec, jsonl_path)
    return per_item


def _option_token_span(tokenizer, stem, full):
    """Token indices of the option inside the full prompt. Uses the common-prefix
    length of the two tokenizations, which is robust to BPE boundary merges."""
    full_ids = tokenizer(full, add_special_tokens=True)["input_ids"]
    stem_ids = tokenizer(stem, add_special_tokens=True)["input_ids"]
    k = 0
    for a, b in zip(stem_ids, full_ids):
        if a != b:
            break
        k += 1
    return k, len(full_ids)


def run_logprob(llm, questions, jsonl_path=None):
    """Scoring method 2 (PRIMARY): no generation. Score each candidate holding
    appended to the excerpt via prompt_logprobs; sum over option tokens only."""
    from vllm import SamplingParams
    tokenizer = llm.get_tokenizer()
    # prompt_logprobs=0: each position's dict holds ONLY the actual prompt
    # token, so the max() in the loop below reads exactly its logprob. Do NOT
    # raise this above 0 - with top-k alternatives present, max() would
    # silently read the wrong token's logprob.
    sp = SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=0)

    prompts, spans = [], []
    for q in questions:
        for opt in q["shuffled_holdings"]:
            full = f"{q['logprob_stem']} {opt}"
            prompts.append(full)
            spans.append(_option_token_span(tokenizer, q["logprob_stem"], full))

    outs = llm.generate(prompts, sp)

    per_item = []
    for qi, q in enumerate(questions):
        sum_logp, mean_logp = {}, {}
        for oi in range(N_OPTIONS):
            out = outs[qi * N_OPTIONS + oi]
            start, end = spans[qi * N_OPTIONS + oi]
            plps = out.prompt_logprobs  # aligned with prompt tokens; index 0 is None
            vals = []
            for pos in range(start, end):
                d = plps[pos]
                if d is None:
                    continue
                # dict {token_id: Logprob}; the actual prompt token is present
                vals.append(max(v.logprob for v in d.values()))
            n_tok = max(len(vals), 1)
            s = float(sum(vals))
            sum_logp[LETTERS[oi]] = s
            mean_logp[LETTERS[oi]] = s / n_tok
        pred_sum = max(sum_logp, key=sum_logp.get)       # -> acc
        pred_norm = max(mean_logp, key=mean_logp.get)    # -> acc_norm (headline)
        gold = LETTERS[q["gold_pos"]]
        rec = {
            "question_id": q["question_id"],
            "shuffled_gold": gold,
            "prediction": pred_norm,
            "prediction_sum": pred_sum,
            "correct": bool(pred_norm == gold),
            "correct_sum": bool(pred_sum == gold),
            "permutation": q["permutation"],
            "sum_logp": sum_logp,
            "mean_logp": mean_logp,
        }
        per_item.append(rec)
        if jsonl_path:
            append_jsonl(rec, jsonl_path)
    return per_item


# ------------------------------------------------------------------ metrics

def bootstrap_ci(correct, n_boot=N_BOOTSTRAP):
    rng = np.random.default_rng(int(SEED))
    arr = np.asarray(correct, dtype=float)
    if len(arr) == 0:
        return None, None
    means = rng.choice(arr, size=(n_boot, len(arr)), replace=True).mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def compute_metrics(per_item):
    from sklearn.metrics import f1_score
    gold = [r["shuffled_gold"] for r in per_item]
    # prediction can be None (parse failure / API retries exhausted) - map to "?"
    pred = [
        r["prediction"] if isinstance(r["prediction"], str) and r["prediction"] in LETTERS else "?"
        for r in per_item
    ]
    correct = [r["correct"] for r in per_item]
    ci_low, ci_high = bootstrap_ci(correct)
    metrics = {
        "accuracy": float(np.mean(correct)) if correct else None,
        "macro_f1": float(f1_score(gold, pred, labels=list(LETTERS), average="macro", zero_division=0)),
        "ci_low": ci_low,
        "ci_high": ci_high,
        "n_parse_fail": int(sum(1 for r in per_item if r["prediction"] is None)),
    }
    if per_item and "correct_sum" in per_item[0]:  # logprob scoring: report both
        metrics["accuracy_acc"] = float(np.mean([r["correct_sum"] for r in per_item]))
        metrics["accuracy_acc_norm"] = metrics["accuracy"]  # headline
    return metrics


# ------------------------------------------------------------- result files

def result_name(model_id, condition, scoring, slice_name):
    safe = model_id.replace("/", "__")
    return f"{safe}__{condition}__{scoring}__{slice_name}.json"


def build_result(model_id, condition, scoring, per_item, extra_config=None):
    import torch
    try:
        import vllm
        vllm_version = vllm.__version__
    except ImportError:
        vllm_version = None
    gpu = None
    dtype = None
    if torch.cuda.is_available():
        g = detect_gpu()
        gpu, dtype = g["name"], g["dtype"]
    config = {
        "model_id": model_id,
        "condition": condition,
        "scoring": scoring,
        "seed": int(SEED),
        "n": len(per_item),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "vllm_version": vllm_version,
        "torch_version": torch.__version__,
        "gpu": gpu,
        "dtype": dtype,
    }
    if extra_config:
        config.update(extra_config)
    return {"config": config, "metrics": compute_metrics(per_item), "per_item": per_item}


def save_and_upload(result, out_dir, bucket_name, slice_name):
    c = result["config"]
    name = result_name(c["model_id"], c["condition"], c["scoring"], slice_name)
    local = save_json(result, Path(out_dir) / name)
    uri = upload_to_gcs(local, bucket_name, f"results/{name}")
    return local, uri


# ------------------------------------------------- API row (Claude on Vertex)

API_MODEL = "claude-haiku-4-5@20251001"  # Vertex uses @ separators for dated snapshots

def run_api_letter_emit(questions, project_id, region="global",
                        model=API_MODEL, out_json_path=None, jsonl_path=None,
                        max_retries=5, usage=None):
    """Claude Haiku 4.5 via Vertex AI Model Garden. Letter-emit only (no
    logprobs available). scoring='letter_emit_api' so these rows are never
    averaged with logprob rows. GCP creds are ambient on Workbench.

    region: 'global' endpoint recommended - regional endpoints carry a 10%
    pricing premium. usage: optional dict accumulated in place with
    input_tokens / output_tokens for cost tracking."""
    import anthropic
    from anthropic import AnthropicVertex

    client = AnthropicVertex(project_id=project_id, region=region)
    per_item = []
    for i, q in enumerate(questions):
        resp = None
        for attempt in range(max_retries):
            try:
                resp = client.messages.create(
                    model=model,
                    max_tokens=8,
                    temperature=float(TEMPERATURE),
                    system=SYSTEM_LETTER,
                    messages=[{"role": "user", "content": q["letter_user"]}],
                )
                break
            except (anthropic.RateLimitError, anthropic.InternalServerError,
                    anthropic.APIConnectionError):
                time.sleep(min(2 ** attempt * 2, 60))
        if resp is None:
            # retries exhausted on a retryable error - an error record, not a
            # fabricated "model failed to emit a letter" parse failure
            rec = {
                "question_id": q["question_id"],
                "error": True,
                "error_type": "api_failed",
                "shuffled_gold": LETTERS[q["gold_pos"]],
                "permutation": q["permutation"],
            }
        else:
            if usage is not None:
                usage["input_tokens"] = usage.get("input_tokens", 0) + resp.usage.input_tokens
                usage["output_tokens"] = usage.get("output_tokens", 0) + resp.usage.output_tokens
            raw = "".join(b.text for b in resp.content if b.type == "text")
            pred = parse_letter(raw)
            rec = {
                "question_id": q["question_id"],
                "shuffled_gold": LETTERS[q["gold_pos"]],
                "prediction": pred,
                "correct": bool(pred == LETTERS[q["gold_pos"]]),
                "permutation": q["permutation"],
                "raw_output": raw,
            }
        per_item.append(rec)
        if jsonl_path:
            append_jsonl(rec, jsonl_path)
        if (i + 1) % 25 == 0:
            import sys
            print(f"api: {i + 1}/{len(questions)} scored", file=sys.stderr, flush=True)
            if out_json_path:  # periodic full snapshot
                save_json({"partial": True, "n_done": i + 1, "per_item": per_item}, out_json_path)
    return per_item
