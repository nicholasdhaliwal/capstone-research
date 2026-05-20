# Legal-RAG — CaseHOLD retrieval comparison

This folder contains the experimental infrastructure for the LCM capstone's
retrieval-evaluation paper. The central question: **does concept-level
retrieval (SONAR sentence embeddings) outperform keyword retrieval (BM25)
when augmenting SaulLM-7B-Instruct on the CaseHOLD multiple-choice legal
QA benchmark?**

## What's here

```
notebooks/legal-rag/
├── README.md                          this file
├── saul_base.ipynb                    Saul, no retrieval (bare baseline)
├── saul_bm25.ipynb                    Saul + BM25 keyword retrieval
├── saul_sonar.ipynb                   Saul + SONAR concept retrieval
├── colab/
│   └── colab_logprob_scoring.ipynb    Colab GPU notebook for logprob baseline
├── data_for_colab/                    Precomputed SONAR embeddings (no fairseq2 needed)
│   ├── sonar_corpus_2000.npy
│   ├── sonar_corpus_2000_meta.json
│   ├── sonar_test_queries_50.npy
│   └── manifest.json
├── precompute_for_colab.py            Regenerate the embeddings locally (needs fairseq2)
└── results/                           Per-notebook results land here (gitignored)
```

## Quick start (any of the three team notebooks)

You need:

1. **Python 3.12 venv** with the deps each notebook installs in cell 1
   (`datasets`, `rank-bm25`, `requests`, `pandas`, `matplotlib`, `numpy`).
   The `legal-rag` Jupyter kernel exists on the project owner's machine; on
   a fresh machine, create one:

   ```bash
   python3.12 -m venv .venv
   source .venv/bin/activate
   pip install ipykernel datasets rank-bm25 requests pandas matplotlib numpy nltk tqdm nbformat
   python -m ipykernel install --user --name=legal-rag --display-name="legal-rag"
   ```

2. **Ollama** with `saul-7b` pulled and aliased:

   ```bash
   brew install ollama && ollama serve &     # macOS
   ollama pull hf.co/mradermacher/Saul-7B-Instruct-v1-GGUF:Q4_K_M
   ollama cp  hf.co/mradermacher/Saul-7B-Instruct-v1-GGUF:Q4_K_M saul-7b
   ```

3. Open any of `saul_base.ipynb`, `saul_bm25.ipynb`, `saul_sonar.ipynb` in
   Jupyter, select the `legal-rag` kernel, **Run All**.

Each notebook is independent. Each saves its own results under
`results/<model>_<retriever>_results.json` plus a prediction-distribution
chart at `results/<model>_<retriever>_predictions.png`.

## The A-bias finding (read before interpreting numbers)

Across three letter-emit probes — original prompt, K=2/truncated prompt,
prompt-restructured, both Q4_K_M and Q5_K_M quants — we found that
**SaulLM-7B-Instruct collapses to predicting option `A` on ~80–90% of
retrieval-augmented prompts**, regardless of which retriever produced the
context. BM25 and SONAR conditions produced identical predictions in the
original probe (both 100% A).

The bare condition does **not** exhibit this collapse (it uses all five
letters with some bias toward A), so the model can read options when the
prompt is short and clean. Long retrieval-augmented prompts push the
quantized 7B into a default-A response.

This is consistent with the SaulLM paper's own admission (Colombo 2024
§6.1): *"SaulLM-7B-Instruct is outperformed by Mistral-Instruct on the
conclusion tasks, which necessitate more deductive reasoning."*

**Implication for the team notebooks:** the letter-emit accuracy numbers
in `saul_bm25.ipynb` and `saul_sonar.ipynb` mostly measure the A-bias,
not retrieval quality. Use them as infrastructure / sanity checks, not as
evidence for H1 (SONAR > BM25).

For the paper-grade comparison, use the Colab logprob notebook (next
section), which bypasses letter emission by scoring each option's
conditional log-likelihood directly.

## Two eval methods, two notebook tiers

| Tier | Where | What | When to use |
|---|---|---|---|
| **Letter-emit (Saul says "A"/"B"/...)** | Local: `saul_base/bm25/sonar.ipynb` | Asks Saul to emit the answer letter, parses the first character | Fast iteration, qualitative inspection of retrieved docs, demo for non-technical audiences |
| **Logprob (score each option's NLL)** | Colab: `colab/colab_logprob_scoring.ipynb` | For each option, computes `log p(option \| prompt)` under Saul; argmax wins. Matches Zheng 2021 CaseHOLD eval protocol | Paper-grade accuracy numbers. Bypasses the A-bias. |

After running the Colab notebook, drop the downloaded
`logprob_results.json` into `notebooks/legal-rag/results/`. Section 9 of
each local notebook will pick it up and display a side-by-side comparison.

## SONAR notebook: why no fairseq2 install?

SONAR's text encoder lives in the `fairseq2`/`sonar-space` packages, which
were the source of significant setup friction (libsndfile, CONDA_PREFIX
quirks, transformers/torchvision version chains on Colab). To insulate
the team from this:

- The project owner ran `precompute_for_colab.py` once on a working venv
  to encode the 14,889 corpus sentences and 50 test queries.
- The resulting `.npy` files are committed to `data_for_colab/`.
- The SONAR notebook downloads them at startup. Cosine retrieval is just
  a normalized matmul + argsort — pure numpy.

Teammates don't need fairseq2 / sonar-space / libsndfile / Metal /
anything from that dependency tree.

**To regenerate the embeddings** (only needed if `CORPUS_SIZE` or
`N_TEST` changes): set up the heavy venv per `precompute_for_colab.py`'s
imports, then run it. Expect ~30–60 minutes on CPU.

## Locked decisions (see `docs/HANDOFF.md`)

- **No model training.** SaulLM-7B-Instruct off the shelf, Q4_K_M
  quantization via Ollama.
- **CaseHOLD as the only benchmark.** Programmatic scoring, no graders,
  published baselines for context.
- **Single experiment.** No fine-tuning, no DLCM, no Legal WordNet
  construction in this paper. Those are future work.
- **N_TEST=50, CORPUS_SIZE=2000** are the locked sanity-run defaults.
  Scale to full CaseHOLD test (3,600) and full training corpus (~45,000)
  for the final paper-grade numbers.

## Published reference points (for context)

| Source | Reports |
|---|---|
| Zheng et al. 2021 (CaseHOLD paper, arXiv:2104.08671) | BERT-base, BERT-large, RoBERTa, LegalBERT on CaseHOLD |
| Colombo et al. 2024 (SaulLM-7B, arXiv:2403.03883) | SaulLM vs Mistral on LegalBench |
| arXiv:2505.02172 (May 2025) | GPT-4o ≈ 0.74, AmazonNovaPro ≈ 0.72 F1 on CaseHOLD |
| Kim et al. 2025 (LRAGE, arXiv:2504.01840) | BM25 vs dense retriever on legal benchmarks |

## Troubleshooting

| Symptom | Fix |
|---|---|
| `OSError: fairseq2 requires libsndfile` | You hit this only if you're running `precompute_for_colab.py`. `brew install libsndfile`, then `ln -sf $(brew --prefix)/lib/libsndfile.1.dylib .venv/lib/libsndfile.1.dylib`. |
| `kernel died` mid-notebook | macOS sandbox issue. Make sure no other Metal-using process (Ollama runner, llama.cpp, another Jupyter) is loading models concurrently. |
| `OSError: ... CERTIFICATE_VERIFY_FAILED` (NLTK punkt) | Already handled — the notebooks only use `sent_tokenize` which works from the cached download. If you hit it on a fresh machine: `python -m nltk.downloader punkt punkt_tab`. |
| `ollama is not running` | Run `ollama serve &` in a terminal. The notebooks check before starting. |
| All predictions are `A` | Expected. See "The A-bias finding" above. Run the Colab logprob notebook for the authoritative numbers. |
