"""retrieval_core.py - staged retrieval bake-off for CaseHOLD (Notebook 2).

Imports eval_core for all shared logic (data, split, shuffling, prompts,
vLLM scoring, metrics, JSON IO, GCS). This module adds only retrieval.

DESIGN NOTE - INTENTIONAL REVERSAL FROM Q1 (do not "fix" back): retrieved
cases are prepended to the prompt WITH their gold holdings attached. The
leakage risk this creates is handled by the TF-IDF dedup filter below
(threshold LEAKAGE_SIM_THRESHOLD), NOT by dropping holdings.

All GPU work (encoding, reranking, HyDE generation, vLLM scoring) runs in
subprocesses via the CLI at the bottom; the notebook is a thin staged driver.
Every corpus index/embedding is computed once, uploaded to GCS, and loaded
from cache thereafter - never recomputed silently.
"""

import argparse
import json
import pickle
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import eval_core as ec

# ---------------------------------------------------------------- constants

K_FINAL = 3            # docs prepended to the prompt
K_CANDIDATES = 50      # retrieve depth before filter (and rerank in Stage B)
LEAKAGE_SIM_THRESHOLD = 0.85   # TF-IDF cosine between doc holding and any option
RERANK_MAX_LENGTH = 512  # bge-reranker-v2-m3 tokenizer default; card-recommended
TRUNC_MARGIN_TOKENS = 512      # headroom for system prompt + template + generation
HYDE_MAX_TOKENS = 128

SPLADE_ID = "naver/splade-v3"                      # fallback: naver/splade-cocondenser-ensembledistil
SPLADE_FALLBACK_ID = "naver/splade-cocondenser-ensembledistil"
DENSE_ID = "BAAI/bge-m3"
RERANKER_ID = "BAAI/bge-reranker-v2-m3"
SONAR_ENCODER = "text_sonar_basic_encoder"
HYDE_MODEL_ID = "Equall/Saul-7B-Instruct-v1"

RAG_DOC_TPL = "Related case {i}: {excerpt}\nIts holding: {holding}\n\n"
HYDE_SYSTEM = (
    "You are a legal expert. Read the case excerpt and write the single holding "
    "statement that the court most likely adopted. Respond with the holding text "
    "only - no preamble, no explanation."
)

# GCS layout (versioned; smoke runs use a separate corpus tag so caches never mix)
def corpus_tag(smoke):
    return "smoke2000" if smoke else "full_v1"

def corpus_blob(tag):
    return f"artifacts/retrieval/corpus_{tag}.parquet"

def index_blob(retriever, tag, ext):
    return f"artifacts/retrieval/index_{retriever}_{tag}.{ext}"

def contexts_blob(condition, slice_name, tag):
    return f"artifacts/retrieval/contexts/{condition}__{slice_name}__{tag}.json"

WINNER_BLOB = "artifacts/retrieval/winner_v1.json"


# -------------------------------------------------------------- corpus

def get_corpus(bucket, work_dir, smoke=False):
    """Corpus = ALL CaseHOLD train questions as documents (deliberately scaled
    up from Q1's first-2000 subset; smoke mode keeps the old 2000 for speed).
    Each document = citing excerpt + its gold holding text.
    Parquet columns: [train_id, excerpt, holding]."""
    import pandas as pd
    tag = corpus_tag(smoke)
    local = Path(work_dir) / f"corpus_{tag}.parquet"
    if not local.exists():
        ec.download_from_gcs(bucket, corpus_blob(tag), local)
    if local.exists():
        df = pd.read_parquet(local)
    else:
        train = ec.load_casehold()["train"]
        if smoke:
            train = train.select(range(2000))
        df = pd.DataFrame({
            "train_id": [str(x) for x in train["example_id"]],
            "excerpt": train["citing_prompt"],
            "holding": [h[l] for h, l in zip(train["holdings"], train["label"])],
        })
        local.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(local, index=False)
        ec.upload_to_gcs(local, bucket, corpus_blob(tag))
    assert list(df.columns) == ["train_id", "excerpt", "holding"]
    return df


def doc_texts(corpus_df):
    """Text indexed by every retriever: excerpt + holding."""
    return (corpus_df["excerpt"] + "\n" + corpus_df["holding"]).tolist()


# ------------------------------------------------------------- retrievers
# Common interface: index(corpus_df) then retrieve(query, k) -> (doc_ids, scores).
# doc_ids are corpus train_ids (strings). Neural retrievers cache their corpus
# index to GCS; loading the cache is the default path.

class BaseRetriever:
    name = None
    ext = None

    def __init__(self, bucket, work_dir, tag, device="cuda"):
        self.bucket = bucket
        self.work_dir = Path(work_dir)
        self.tag = tag
        self.device = device
        self.ids = None

    @property
    def cache_path(self):
        return self.work_dir / f"index_{self.name}_{self.tag}.{self.ext}"

    @property
    def ids_path(self):
        return self.work_dir / f"corpus_ids_{self.name}_{self.tag}.json"

    def cache_exists(self):
        for p, blob in [(self.cache_path, index_blob(self.name, self.tag, self.ext)),
                        (self.ids_path, index_blob(self.name, self.tag, "ids.json"))]:
            if not p.exists() and ec.download_from_gcs(self.bucket, blob, p) is None:
                return False
        return True

    def save_cache(self):
        ec.upload_to_gcs(self.cache_path, self.bucket, index_blob(self.name, self.tag, self.ext))
        ec.upload_to_gcs(self.ids_path, self.bucket, index_blob(self.name, self.tag, "ids.json"))

    def index(self, corpus_df, force=False):
        """Load cached index if present (default path); encode only if absent."""
        if not force and self.cache_exists():
            self.ids = json.loads(self.ids_path.read_text())
            self._load()
            assert len(self.ids) == len(corpus_df), \
                f"{self.name} cache built on a different corpus ({len(self.ids)} vs {len(corpus_df)})"
            return self
        print(f"[{self.name}] no cache - encoding {len(corpus_df)} docs (one-time)")
        self.ids = corpus_df["train_id"].tolist()
        self.ids_path.parent.mkdir(parents=True, exist_ok=True)
        self.ids_path.write_text(json.dumps(self.ids))
        self._build(doc_texts(corpus_df))
        self.save_cache()
        return self

    def encode_queries(self, texts):
        raise NotImplementedError

    def retrieve_batch(self, query_texts, k):
        """Returns list of (ids, scores) per query."""
        Q = self.encode_queries(query_texts)
        out = []
        for qi in range(len(query_texts)):
            sims = self._scores(Q, qi)
            order = np.argsort(-sims)[:k]
            out.append(([self.ids[i] for i in order], sims[order].astype(float).tolist()))
        return out


class BM25Retriever(BaseRetriever):
    """Appendix baseline. Whitespace tokenization, matching the Q1 setup."""
    name, ext = "bm25", "pkl"

    def _tok(self, t):
        return t.lower().split()

    def _build(self, texts):
        from rank_bm25 import BM25Okapi
        self.bm25 = BM25Okapi([self._tok(t) for t in texts])
        with open(self.cache_path, "wb") as f:
            pickle.dump(self.bm25, f)

    def _load(self):
        with open(self.cache_path, "rb") as f:
            self.bm25 = pickle.load(f)

    def encode_queries(self, texts):
        return texts  # scored lazily

    def _scores(self, Q, qi):
        return np.asarray(self.bm25.get_scores(self._tok(Q[qi])))


class SpladeRetriever(BaseRetriever):
    """Learned sparse. Encode on GPU in batches; sparse dot-product in scipy.

    naver/splade-v3 is gated (auto-approve): accept terms on the model page once
    and have HF_TOKEN set (already required for Llama/Gemma). The except-branch
    falls back to the ungated cocondenser checkpoint; naver/splade-v3-distilbert
    is another ungated option. Recipe verified against the official naver/splade
    repo: max over sequence of log1p(relu(logits)), attention-masked."""
    name, ext = "splade", "npz"
    batch_size = 32
    max_length = 512

    def _model(self):
        import torch
        from transformers import AutoModelForMaskedLM, AutoTokenizer
        if not hasattr(self, "model"):
            mid = SPLADE_ID
            try:
                self.tok = AutoTokenizer.from_pretrained(mid)
                self.model = AutoModelForMaskedLM.from_pretrained(mid, torch_dtype=torch.float16)
            except Exception:
                mid = SPLADE_FALLBACK_ID
                self.tok = AutoTokenizer.from_pretrained(mid)
                self.model = AutoModelForMaskedLM.from_pretrained(mid, torch_dtype=torch.float16)
            self.model_id = mid
            self.model.eval().to(self.device)
        return self.model

    def _encode(self, texts):
        import torch
        from scipy import sparse
        model = self._model()
        rows = []
        with torch.no_grad():
            for i in range(0, len(texts), self.batch_size):
                batch = self.tok(texts[i:i + self.batch_size], padding=True, truncation=True,
                                 max_length=self.max_length, return_tensors="pt").to(self.device)
                logits = model(**batch).logits
                # SPLADE rep: max over sequence of log(1 + relu(logits)), masked
                w = torch.log1p(torch.relu(logits)) * batch["attention_mask"].unsqueeze(-1)
                rep = w.max(dim=1).values.float().cpu().numpy()
                rows.append(sparse.csr_matrix(rep))
        return sparse.vstack(rows)

    def _build(self, texts):
        from scipy import sparse
        self.M = self._encode(texts)
        sparse.save_npz(self.cache_path, self.M)

    def _load(self):
        from scipy import sparse
        self.M = sparse.load_npz(self.cache_path)

    def encode_queries(self, texts):
        return self._encode(texts)

    def _scores(self, Q, qi):
        return np.asarray((self.M @ Q[qi].T).todense()).ravel()


class DenseRetriever(BaseRetriever):
    """BAAI/bge-m3 dense vectors via sentence-transformers (verified current
    install; FlagEmbedding only needed for its sparse/ColBERT outputs). The
    checkpoint L2-normalizes by default. Numpy cosine (matmul), no FAISS."""
    name, ext = "bge", "npy"
    batch_size = 64
    max_length = 1024

    def _model(self):
        if not hasattr(self, "model"):
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(DENSE_ID, device=self.device)
            self.model.max_seq_length = self.max_length
        return self.model

    def _encode(self, texts):
        emb = self._model().encode(texts, batch_size=self.batch_size,
                                   normalize_embeddings=True, show_progress_bar=True)
        return np.asarray(emb, dtype=np.float32)

    def _build(self, texts):
        self.M = self._encode(texts)
        np.save(self.cache_path, self.M)

    def _load(self):
        self.M = np.load(self.cache_path)

    def encode_queries(self, texts):
        return self._encode(texts)

    def _scores(self, Q, qi):
        return self.M @ Q[qi]


class SonarRetriever(BaseRetriever):
    """SONAR sentence embeddings (1024-d), fairseq2 backend. Installs cleanly on
    Linux - the historical libsndfile pain was macOS-only. BUT fairseq2 must
    ABI-match torch exactly and vLLM pins its own torch, so SONAR runs from a
    SEPARATE venv (see the notebook setup cell; sonar subcommands are invoked
    with that venv's python). Corpus encoded once on GPU, cached to GCS as .npy
    with an ID-order file; loading the cache is the default path."""
    name, ext = "sonar", "npy"
    batch_size = 32
    chunk = 1024  # outer chunking only for progress prints

    def _pipeline(self):
        if not hasattr(self, "t2vec"):
            import torch
            from sonar.inference_pipelines.text import TextToEmbeddingModelPipeline
            self.t2vec = TextToEmbeddingModelPipeline(
                encoder=SONAR_ENCODER, tokenizer=SONAR_ENCODER, device=torch.device(self.device))
        return self.t2vec

    def _encode(self, texts):
        t2vec = self._pipeline()
        chunks = []
        for i in range(0, len(texts), self.chunk):
            v = t2vec.predict(texts[i:i + self.chunk], source_lang="eng_Latn",
                              batch_size=self.batch_size)
            chunks.append(np.asarray(v.cpu().numpy(), dtype=np.float32))
            print(f"[sonar] encoded {min(i + self.chunk, len(texts))}/{len(texts)}", flush=True)
        M = np.vstack(chunks)
        assert M.shape[1] == 1024
        return M / np.linalg.norm(M, axis=1, keepdims=True)

    def _build(self, texts):
        self.M = self._encode(texts)
        np.save(self.cache_path, self.M)

    def _load(self):
        self.M = np.load(self.cache_path)

    def encode_queries(self, texts):
        return self._encode(texts)

    def _scores(self, Q, qi):
        return self.M @ Q[qi]


RETRIEVERS = {c.name: c for c in (BM25Retriever, SpladeRetriever, DenseRetriever, SonarRetriever)}
CONDITION_OF = {"bm25": "rag_bm25", "splade": "rag_splade", "bge": "rag_bge", "sonar": "rag_sonar"}


# --------------------------------------------------------- leakage filter

class LeakageFilter:
    """Guards the answer key: drops any retrieved document whose holding has
    TF-IDF cosine > threshold against ANY of the question's 5 options.
    Every trigger is logged - the trigger rate is a reportable statistic."""

    def __init__(self, corpus_df, threshold=LEAKAGE_SIM_THRESHOLD):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.threshold = float(threshold)
        self.vec = TfidfVectorizer().fit(corpus_df["holding"].tolist())
        self.H = self.vec.transform(corpus_df["holding"].tolist())  # rows l2-normalized
        self.row_of = {tid: i for i, tid in enumerate(corpus_df["train_id"].astype(str))}

    def filter(self, question_id, options, ranked_ids, ranked_scores, k):
        """Walk the ranked list, drop leaked docs, return k survivors + trigger log."""
        V = self.vec.transform(options)  # 5 x vocab, l2-normalized
        survivors, triggers = [], []
        for doc_id, score in zip(ranked_ids, ranked_scores):
            sims = np.asarray((self.H[self.row_of[doc_id]] @ V.T).todense()).ravel()
            worst = int(np.argmax(sims))
            if float(sims[worst]) > self.threshold:
                triggers.append({
                    "question_id": str(question_id),
                    "doc_id": str(doc_id),
                    "similarity": float(sims[worst]),
                    "option": ec.LETTERS[worst],
                })
                continue
            survivors.append({"train_id": str(doc_id), "score": float(score)})
            if len(survivors) >= k:
                break
        return survivors, triggers


# ------------------------------------------------------- context building

def build_contexts(retriever_name, slice_name, bucket, work_dir, smoke=False,
                   smoke_n=None, k_store=K_FINAL, query_file=None, condition=None,
                   force=False):
    """Retrieve K_CANDIDATES, apply the leakage filter, keep k_store survivors.
    Writes a contexts artifact (question_id -> docs + triggers) to GCS.
    query_file (HyDE): JSON {question_id: hypothesis_text} replacing raw excerpts."""
    condition = condition or CONDITION_OF[retriever_name]
    tag = corpus_tag(smoke)
    blob = contexts_blob(condition, slice_name, tag)
    local = Path(work_dir) / Path(blob).name
    if not force and (local.exists() or ec.download_from_gcs(bucket, blob, local)):
        return json.loads(local.read_text())

    data = ec.load_casehold()
    split = ec.get_split(data["test"], bucket, work_dir)
    slice_ds = ec.select_slice(data["test"], split, slice_name, smoke_n=smoke_n)
    corpus = get_corpus(bucket, work_dir, smoke=smoke)
    flt = LeakageFilter(corpus)
    retr = RETRIEVERS[retriever_name](bucket, work_dir, tag).index(corpus)

    hyde = json.loads(Path(query_file).read_text()) if query_file else None
    questions = ec.prepare_questions(slice_ds, "zero_shot")
    queries = [hyde[q["question_id"]] if hyde else row["citing_prompt"]
               for q, row in zip(questions, slice_ds)]

    ranked = retr.retrieve_batch(queries, K_CANDIDATES)
    doc_of = corpus.set_index("train_id")[["excerpt", "holding"]]

    items, all_triggers = {}, []
    for q, (ids, scores) in zip(questions, ranked):
        survivors, triggers = flt.filter(q["question_id"], q["shuffled_holdings"], ids, scores, k_store)
        for s in survivors:
            row = doc_of.loc[s["train_id"]]
            s["excerpt"], s["holding"] = str(row["excerpt"]), str(row["holding"])
        items[q["question_id"]] = {"docs": survivors, "triggers": triggers}
        all_triggers.extend(triggers)

    n_q = len(items)
    art = {
        "condition": condition,
        "retriever": retriever_name,
        "slice": slice_name,
        "corpus_tag": tag,
        "k_candidates": K_CANDIDATES,
        "k_store": k_store,
        "leakage_threshold": LEAKAGE_SIM_THRESHOLD,
        "hyde_query_file": bool(query_file),
        "created": datetime.now(timezone.utc).isoformat(),
        "summary": {
            "n_questions": n_q,
            "n_triggers": len(all_triggers),
            "trigger_rate_questions": sum(1 for v in items.values() if v["triggers"]) / max(n_q, 1),
            "triggers_per_question": len(all_triggers) / max(n_q, 1),
        },
        "items": items,
    }
    ec.save_json(art, local)
    ec.upload_to_gcs(local, bucket, blob)
    return art


def rerank_contexts(base_condition, slice_name, bucket, work_dir, smoke=False, force=False):
    """Stage B: cross-encoder rerank. Consumes a k_store=K_CANDIDATES artifact,
    reranks each question's surviving docs, keeps top K_FINAL."""
    tag = corpus_tag(smoke)
    condition = f"{base_condition}_rerank"
    blob = contexts_blob(condition, slice_name, tag)
    local = Path(work_dir) / Path(blob).name
    if not force and (local.exists() or ec.download_from_gcs(bucket, blob, local)):
        return json.loads(local.read_text())

    src_blob = contexts_blob(f"{base_condition}__cand", slice_name, tag)
    src_local = Path(work_dir) / Path(src_blob).name
    if not src_local.exists():
        assert ec.download_from_gcs(bucket, src_blob, src_local), \
            f"candidate artifact missing: run contexts with --k-store {K_CANDIDATES} first ({src_blob})"
    src = json.loads(src_local.read_text())

    data = ec.load_casehold()
    qtext = {str(r["example_id"]): r["citing_prompt"] for r in data["test"]}

    from sentence_transformers import CrossEncoder
    ce = CrossEncoder(RERANKER_ID, max_length=RERANK_MAX_LENGTH)

    items = {}
    for qid, item in src["items"].items():
        docs = item["docs"]
        if docs:
            pairs = [(qtext[qid], f"{d['excerpt']}\n{d['holding']}") for d in docs]
            scores = ce.predict(pairs)
            order = np.argsort(-np.asarray(scores))[:K_FINAL]
            docs = [{**docs[i], "rerank_score": float(scores[i])} for i in order]
        items[qid] = {"docs": docs, "triggers": item["triggers"]}

    art = {**{k: v for k, v in src.items() if k != "items"},
           "condition": condition, "reranker": RERANKER_ID, "k_store": K_FINAL,
           "created": datetime.now(timezone.utc).isoformat(), "items": items}
    ec.save_json(art, local)
    ec.upload_to_gcs(local, bucket, blob)
    return art


# ------------------------------------------------------------------- HyDE

def hyde_generate(slice_name, bucket, work_dir, smoke_n=None, force=False):
    """Stage C: Saul-Instruct writes a hypothetical holding per query excerpt
    (temperature 0, max 128 tokens); the hypothesis is embedded instead of the
    raw excerpt. Runs vLLM - subprocess only."""
    blob = f"artifacts/retrieval/hyde_{slice_name}_v1.json"
    local = Path(work_dir) / Path(blob).name
    if not force and (local.exists() or ec.download_from_gcs(bucket, blob, local)):
        return local

    data = ec.load_casehold()
    split = ec.get_split(data["test"], bucket, work_dir)
    slice_ds = ec.select_slice(data["test"], split, slice_name, smoke_n=smoke_n)

    from vllm import SamplingParams
    llm = ec.make_llm(HYDE_MODEL_ID)
    sp = SamplingParams(temperature=float(ec.TEMPERATURE), max_tokens=HYDE_MAX_TOKENS,
                        seed=int(ec.SEED))
    convs = [[{"role": "system", "content": HYDE_SYSTEM},
              {"role": "user", "content": r["citing_prompt"]}] for r in slice_ds]
    outs = llm.chat(convs, sp)
    hyp = {str(r["example_id"]): o.outputs[0].text.strip() for r, o in zip(slice_ds, outs)}
    ec.save_json(hyp, local)
    ec.upload_to_gcs(local, bucket, blob)
    return local


# --------------------------------------------------- prompt assembly / scoring

def fit_docs_to_budget(tokenizer, docs, fixed_texts, n_ctx=ec.N_CTX, margin=TRUNC_MARGIN_TOKENS):
    """Truncate retrieved excerpts (never the question) longest-first until the
    full prompt fits N_CTX. Returns (docs, truncation_log)."""
    def ntok(t):
        return len(tokenizer(t, add_special_tokens=False)["input_ids"])

    fixed = sum(ntok(t) for t in fixed_texts) + margin
    budget = int(n_ctx) - fixed
    blocks = [RAG_DOC_TPL.format(i=i + 1, excerpt=d["excerpt"], holding=d["holding"])
              for i, d in enumerate(docs)]
    lens = [ntok(b) for b in blocks]
    log = []
    docs = [dict(d) for d in docs]
    while sum(lens) > budget and any(l > 32 for l in lens):
        j = int(np.argmax(lens))
        excess = sum(lens) - budget
        ids = tokenizer(docs[j]["excerpt"], add_special_tokens=False)["input_ids"]
        keep = max(len(ids) - excess - 8, 32)
        if keep >= len(ids):
            break
        docs[j]["excerpt"] = tokenizer.decode(ids[:keep], skip_special_tokens=True)
        log.append({"doc_id": docs[j]["train_id"], "orig_tokens": len(ids), "kept_tokens": keep})
        blocks[j] = RAG_DOC_TPL.format(i=j + 1, excerpt=docs[j]["excerpt"], holding=docs[j]["holding"])
        lens[j] = ntok(blocks[j])
    return docs, log


def rag_prefix(docs):
    return "".join(RAG_DOC_TPL.format(i=i + 1, excerpt=d["excerpt"], holding=d["holding"])
                   for i, d in enumerate(docs))


def run_rag_eval(model_id, condition, scoring, slice_name, bucket, work_dir, out_dir,
                 smoke=False, smoke_n=None, force=False):
    """One model x one RAG condition x one scoring method, in this process
    (invoke via the CLI so each model gets a fresh subprocess). Resume-safe:
    skips if the result JSON already exists locally or in GCS."""
    result_slice = f"{slice_name}_smoke{smoke_n}" if smoke_n else slice_name
    name = ec.result_name(model_id, condition, scoring, result_slice)
    local_result = Path(out_dir) / name
    if not force and (local_result.exists()
                      or ec.download_from_gcs(bucket, f"results/{name}", local_result)):
        print(f"[skip] {name} already exists")
        return json.loads(local_result.read_text())

    tag = corpus_tag(smoke)
    ctx_blob = contexts_blob(condition, slice_name, tag)
    ctx_local = Path(work_dir) / Path(ctx_blob).name
    if not ctx_local.exists():
        assert ec.download_from_gcs(bucket, ctx_blob, ctx_local), f"contexts artifact missing: {ctx_blob}"
    art = json.loads(ctx_local.read_text())

    data = ec.load_casehold()
    split = ec.get_split(data["test"], bucket, work_dir)
    slice_ds = ec.select_slice(data["test"], split, slice_name, smoke_n=smoke_n)
    questions = ec.prepare_questions(slice_ds, "zero_shot")

    kind = ec.panel_entry(model_id)["kind"]
    llm = ec.make_llm(model_id)
    tokenizer = llm.get_tokenizer()

    meta = {}
    for q in questions:
        item = art["items"].get(q["question_id"], {"docs": [], "triggers": []})
        fixed = [ec.SYSTEM_LETTER, q["letter_user"], max(q["shuffled_holdings"], key=len)]
        docs, trunc_log = fit_docs_to_budget(tokenizer, item["docs"][:K_FINAL], fixed)
        prefix = rag_prefix(docs)
        q["letter_user"] = prefix + q["letter_user"]
        q["logprob_stem"] = prefix + q["logprob_stem"]
        meta[q["question_id"]] = {
            "retrieved": [{"train_id": d["train_id"], "score": d["score"],
                           **({"rerank_score": d["rerank_score"]} if "rerank_score" in d else {})}
                          for d in docs],
            "filter_triggers": item["triggers"],
            "truncations": trunc_log,
        }

    # RAG prefix can itself push a question over N_CTX even when the bare
    # question fits (Q1's qi=45 class, now with retrieved docs prepended).
    questions, error_items = ec.preflight(questions, tokenizer, scoring, ec.N_CTX)

    # Per-item resume (mirrors run_eval.py): a mid-run crash on Stage D's
    # panel-wide loop only loses the in-flight chunk, not the whole model.
    jsonl = local_result.with_suffix(".partial.jsonl")
    done_items, done_ids = ec.load_done(jsonl)
    todo = [q for q in questions if q["question_id"] not in done_ids]
    if scoring == "logprob":
        new_items = ec.run_logprob(llm, todo, jsonl_path=jsonl)
    else:
        new_items = ec.run_letter_emit(llm, todo, kind, jsonl_path=jsonl)
    per_item = done_items + new_items
    for rec in per_item:  # same schema as Notebook 1 + retrieval fields
        rec.update(meta[rec["question_id"]])

    result = ec.build_result(model_id, condition, scoring, per_item, extra_config={
        "retriever": art.get("retriever"),
        "reranker": art.get("reranker"),
        "k": K_FINAL,
        "k_candidates": art.get("k_candidates"),
        "leakage_threshold": art.get("leakage_threshold"),
        "corpus_tag": art.get("corpus_tag"),
        "hyde": art.get("hyde_query_file", False),
    })
    result["metrics"]["n_error"] = len(error_items)
    result["per_item"] = per_item + error_items
    result["retrieval_summary"] = art["summary"]
    local, uri = ec.save_and_upload(result, out_dir, bucket, result_slice)
    print(f"[done] {uri}  acc_norm={result['metrics']['accuracy']:.4f}")
    return result


# ------------------------------------------------------------------ stats

def mcnemar(per_item_a, per_item_b):
    """Exact McNemar (binomial on discordant pairs), matched by question_id.

    Records without a "correct" key (context-overflow / API-failure error
    records - see eval_core.preflight) are skipped on both sides: they were
    never scored, so they can't be discordant. Records whose "permutation"
    disagrees between the two runs means eval_core.shuffle_options drifted
    between the Notebook 1 and Notebook 2 evaluations - a protocol violation,
    not a stat to quietly average over, so it raises instead of comparing
    mismatched shuffles as if they were the same question."""
    from scipy.stats import binomtest
    b_map = {r["question_id"]: r["correct"] for r in per_item_b if "correct" in r}
    b_perm = {r["question_id"]: r["permutation"] for r in per_item_b if "correct" in r}
    n01 = n10 = n_matched = 0
    mismatched = []
    for r in per_item_a:
        qid = r["question_id"]
        if "correct" not in r or qid not in b_map:
            continue
        if r["permutation"] != b_perm[qid]:
            mismatched.append(qid)
            continue
        n_matched += 1
        a, b = bool(r["correct"]), bool(b_map[qid])
        n01 += (not a) and b
        n10 += a and (not b)
    if mismatched:
        raise ValueError(
            f"permutation mismatch between run A and run B for {len(mismatched)} "
            f"question id(s) - eval_core.shuffle_options drifted between runs: "
            f"{mismatched}"
        )
    p = binomtest(min(n01, n10), n01 + n10, 0.5).pvalue if (n01 + n10) else 1.0
    return {"n_matched": n_matched, "a_only_correct": n10, "b_only_correct": n01,
            "p_value": float(p)}


def load_result(model_id, condition, scoring, slice_name, bucket, out_dir):
    name = ec.result_name(model_id, condition, scoring, slice_name)
    local = Path(out_dir) / name
    if not local.exists():
        assert ec.download_from_gcs(bucket, f"results/{name}", local), f"missing result {name}"
    return json.loads(local.read_text())


def write_winner(choice, bucket, work_dir):
    """Winner artifact written BEFORE Stage D runs. Highest DEV acc_norm;
    tie broken toward the simpler pipeline."""
    local = Path(work_dir) / Path(WINNER_BLOB).name
    ec.save_json({**choice, "created": datetime.now(timezone.utc).isoformat()}, local)
    return ec.upload_to_gcs(local, bucket, WINNER_BLOB)


# --------------------------------------------------------------------- CLI

def main(argv=None):
    p = argparse.ArgumentParser(description="CaseHOLD retrieval bake-off worker")
    p.add_argument("cmd", choices=["encode", "contexts", "rerank", "hyde", "score"])
    p.add_argument("--retriever", choices=list(RETRIEVERS))
    p.add_argument("--condition")            # for rerank (base condition) / score
    p.add_argument("--model-id")
    p.add_argument("--scoring", choices=["logprob", "letter_emit"], default="logprob")
    p.add_argument("--slice", dest="slice_name", choices=["dev", "confirm", "test"], default="dev")
    p.add_argument("--bucket", required=True)
    p.add_argument("--work-dir", default="work")
    p.add_argument("--out-dir", default="results")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--smoke-n", type=int, default=None)
    p.add_argument("--k-store", type=int, default=K_FINAL)
    p.add_argument("--query-file", default=None)
    p.add_argument("--force", action="store_true")
    a = p.parse_args(argv)

    common = dict(bucket=a.bucket, work_dir=a.work_dir)
    if a.cmd == "encode":
        corpus = get_corpus(a.bucket, a.work_dir, smoke=a.smoke)
        RETRIEVERS[a.retriever](a.bucket, a.work_dir, corpus_tag(a.smoke)).index(corpus, force=a.force)
    elif a.cmd == "contexts":
        cond = a.condition or CONDITION_OF[a.retriever]
        if a.k_store != K_FINAL:  # candidate artifact for reranking gets its own name
            cond = f"{cond}__cand"
        art = build_contexts(a.retriever, a.slice_name, smoke=a.smoke, smoke_n=a.smoke_n,
                             k_store=a.k_store, query_file=a.query_file,
                             condition=cond, force=a.force, **common)
        print(json.dumps(art["summary"], indent=2))
    elif a.cmd == "rerank":
        art = rerank_contexts(a.condition, a.slice_name, smoke=a.smoke, force=a.force, **common)
        print(json.dumps(art["summary"], indent=2))
    elif a.cmd == "hyde":
        print(hyde_generate(a.slice_name, a.bucket, a.work_dir, smoke_n=a.smoke_n, force=a.force))
    elif a.cmd == "score":
        run_rag_eval(a.model_id, a.condition, a.scoring, a.slice_name, a.bucket,
                     a.work_dir, a.out_dir, smoke=a.smoke, smoke_n=a.smoke_n, force=a.force)


if __name__ == "__main__":
    main()
