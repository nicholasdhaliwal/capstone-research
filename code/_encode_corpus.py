"""One-time pre-encoding of the CaseHOLD corpus into sentence-level SONAR vectors.
Runnable standalone so the notebook doesn't have to do it on first run."""
import sys, time
from pathlib import Path

import numpy as np
import faiss
from datasets import load_dataset
from nltk.tokenize import sent_tokenize
from sonar.inference_pipelines.text import TextToEmbeddingModelPipeline

CORPUS_SIZE = 2000

print("[1/4] loading CaseHOLD...", flush=True)
ds = load_dataset("casehold/casehold", "all")
train = ds["train"].select(range(CORPUS_SIZE))
corpus_texts = [r["citing_prompt"] for r in train]
print(f"  loaded {len(corpus_texts)} excerpts", flush=True)

print("[2/4] segmenting into sentences...", flush=True)
corpus_sentences = []
sentence_to_parent = []
for i, text in enumerate(corpus_texts):
    for s in sent_tokenize(text):
        s = s.strip()
        if len(s) > 10:
            corpus_sentences.append(s)
            sentence_to_parent.append(i)
print(f"  {len(corpus_sentences)} sentences (avg {len(corpus_sentences)/len(corpus_texts):.1f}/excerpt)", flush=True)

cache = Path(f"sonar_sentences_{CORPUS_SIZE}.npy")
if cache.exists():
    print(f"[3/4] cache exists at {cache}, skipping encode", flush=True)
else:
    print("[3/4] loading SONAR encoder...", flush=True)
    sonar = TextToEmbeddingModelPipeline(
        encoder="text_sonar_basic_encoder",
        tokenizer="text_sonar_basic_encoder",
    )
    print(f"[3/4] encoding {len(corpus_sentences)} sentences (batch=16)...", flush=True)
    t0 = time.time()
    chunk = 500
    all_vecs = []
    for i in range(0, len(corpus_sentences), chunk):
        batch = corpus_sentences[i:i+chunk]
        v = sonar.predict(batch, source_lang="eng_Latn", batch_size=16).numpy().astype("float32")
        all_vecs.append(v)
        print(f"  encoded {i+len(batch)}/{len(corpus_sentences)} ({(time.time()-t0):.0f}s elapsed)", flush=True)
    vecs = np.concatenate(all_vecs, axis=0)
    np.save(cache, vecs)
    print(f"  saved {cache} ({cache.stat().st_size//1024//1024} MB) in {time.time()-t0:.0f}s total", flush=True)

print("[4/4] verifying FAISS index...", flush=True)
vecs = np.load(cache)
faiss.normalize_L2(vecs)
idx = faiss.IndexFlatIP(vecs.shape[1])
idx.add(vecs)
print(f"  index: {idx.ntotal} vectors, dim={vecs.shape[1]}", flush=True)
print("done.", flush=True)
