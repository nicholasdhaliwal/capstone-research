# LCM Capstone — Session Handoff

**Date:** 2026-05-13
**Author of this session's work:** Nick Dhaliwal (project lead)
**Project:** UChicago MS Applied Data Science capstone thesis
**Working title:** *Large Concept Modeling for Structured Reasoning*
**Showcase target:** 2026-08-15 (locked, no internship break)
**Repo:** `~/Documents/GitHub/lcm-research/` (github.com/nicholasdhaliwal/lcm-research)
**KB:** `~/Documents/GitHub/knowledge-base/` (github.com/nicholasdhaliwal/knowledge-base)

---

## 1 — Where things stand right now

The project has **pivoted from a graph-construction thesis to a retrieval-evaluation thesis** within the same conceptual frame. The original Legal-WordNet-centric plan locked on 2026-05-06 has been functionally superseded as of this session. The new direction is a **Lego-assembly empirical paper using only pre-existing data and models**, motivated by three concerns:

1. Legal WordNet construction (60–100 expert-validated nodes) requires labor and an external legal grader that has not been recruited and is the project's highest-risk dependency.
2. The "build a graph" deliverable does not match what the concept-level NLP field is publishing (which is retrieval, embeddings, and architectures — not handcrafted ontologies).
3. With 14 weeks to showcase and a team of three (one full-time), the original two-experiment plan is at the edge of feasibility; the new direction is comfortably inside it.

The replacement is a focused six-week paper testing whether **concept-level retrieval (SONAR sentence embeddings) outperforms keyword retrieval (BM25)** when augmenting **SaulLM-7B-Instruct** on the **CaseHOLD** legal benchmark. All components are pre-existing and on Hugging Face. The deliverable is a reproducibility notebook plus an eight-page paper.

**Active blocker:** the notebook is written but fails on cell 3 import because `fairseq2` (a SONAR dependency) cannot locate `libsndfile` on disk. Fix in progress (Section 7).

---

## 2 — Project identity

| Field | Value |
|---|---|
| Program | UChicago MS Applied Data Science |
| Cohort | 2025–26 |
| Capstone team | Nick Dhaliwal (lead, full-time), Nicholas Mikhail (full-time), Jaysen Jensen (part-time) |
| Advisor | TBD |
| Submission window | Spring 2026 → Showcase Saturday Aug 15 2026 |
| Tied venture | Marshall Research LLC (legal AI, IRAC/CREAC reasoning) |
| Domain expertise | Nick — UC Berkeley Economics + Legal Studies |

---

## 3 — The pivot: chronological progression of this session

### 3.1 Starting state (entering this session)

The KB project pages (`wiki/projects/lcm-capstone/`) reflected the scope locked on **2026-05-06**:

- **Deliverable:** Legal WordNet for one bounded doctrine (60–100 nodes, every node and edge traceable to a primary source) + Experiment 1 (DLCM cosine boundary detector vs Legal WordNet boundaries on annotated legal corpus) + Experiment 2 (GPT-4 with vs without Legal WordNet subgraph as structured context, blind expert grading on 20–30 fact patterns) + paper + showcase.
- **Doctrine pick:** Contract Formation recommended (LegalNet already had Restatement-cited stub); McDonnell Douglas / Title VII as runner-up. Pending Nick's final decision.
- **Out of scope:** training a concept-LM from scratch (the `concept-lm` codebase preserved as future work); extending Legal WordNet beyond one doctrine; "inventing graph-augmented prompting" (GraphRAG predates it).
- **Existing assets:** Full Paper outline at 8 sections / ~640 lines (`paper/Capstone Research Paper - 2026-05-06 (latest).docx`); 16-paper lit review fully indexed in KB sources; LegalNet repo with 38 seeded nodes; `concept-lm` codebase with 5,700 LoC of working pipeline (CourtListener scraper, SONAR encoder, boundary detector, training infra).
- **Highest project risk:** expert grader recruitment for Exp 2 (cold-start 2–4 weeks).

### 3.2 SaulLM setup and qualitative probes (early session)

The user installed SaulLM-7B-Instruct-v1 locally via Ollama with Q4_K_M quantization (4.4 GB GGUF from `mradermacher/Saul-7B-Instruct-v1-GGUF`). The model was aliased to `saul-7b` for terminal convenience. A first demo notebook (`~/Documents/GitHub/lcm-research/saul-7b/saul_chat.ipynb`) was created to serve as the basic chat interface.

Eight diagnostic probes were then run against SaulLM with seed=42, temperature=0.2, to assess concrete failure modes relevant to the capstone thesis. **Results saved to `~/Documents/GitHub/lcm-research/saul-7b/probe_results.json`.** Key findings used as evidence in subsequent paper planning:

| Probe | Finding |
|---|---|
| 1 — Concept-boundary marking on a Title VII passage | SaulLM marked only sentence-level boundaries; missed the four sub-sentence concept transitions packed into the prima facie case sentence |
| 2 — Restatement section citations for contract elements | Cited §27 (offer), §30 (acceptance), §75 (consideration) — **all wrong** (real values §24, §50, §71). Fluent, confident fabrication |
| 3 — IRAC on a fact pattern (Alice/Bob/Charlie car offer) | Produced structurally-correct IRAC; wrote *"Bob accepted the offer by saying 'I'll think about it.'"*; did not mention Charlie; reached wrong conclusion. **Form preserved, substance hollow.** Flagship exhibit. |
| 4 — Fake authority (*Brennan v. Whitfield*, 412 U.S. 891) | Fabricated a coherent two-sentence summary of reasoning for a non-existent case. No hedge. Cleanest hallucination exhibit. |
| 5 — Explicit hypothetical (*Smith v. Jones*, declared hypothetical in prompt) | Correctly refused. **Failure mode in probe 4 is calibration, not capability.** This is the empirical bridge to the Exp 2 argument (structure substitutes for memorized authority) |
| 6 — McDonnell Douglas burden shifting | Steps 1–3 correct; **invented a 4th step** that isn't part of the framework. Subtle structural hallucination |
| 7 — Atomic concept segmentation (promissory estoppel sentence) | Reasonable coarse decomposition; missed distinctions between "promisor" / "promisee" / "promise" as separate roles |
| 8 — Define consideration + name two theories | Cited §71 correctly (only correct citation in 8 probes); but invented "past consideration theory" as one of two theories of consideration (it is not a recognized theory of consideration) |

### 3.3 Technical deep-dive on SaulLM (parallel agent research)

A research agent retrieved the SaulLM-7B paper (arXiv:2403.03883) and 54B/141B follow-up (arXiv:2407.19584). Findings load-bearing for the paper:

- **Architecture:** continued pretraining from Mistral-7B-v0.1. No architectural changes. Tokenizer inherited unchanged from Mistral (32k SentencePiece BPE, no legal-vocabulary extension, no embedding resize). Pure decoder-only transformer.
- **Pretrain corpus:** 94B raw → 30B post-filter legal tokens (FreeLaw Pile, GovInfo, Open Australian Legal Corpus, USPTO, MultiLegal Pile, EuroParl, EU/UK legislation, EDGAR, CourtListener audio transcripts, Law StackExchange) + 2% SlimPajama replay against catastrophic forgetting.
- **Instruction tuning:** SFT only. No DPO/RLHF in v1 (authors said preference alignment "did not show meaningful improvement"). The 54B/141B follow-up adds DPO.
- **Benchmarks:** SaulLM-7B-Instruct = 0.61 on LegalBench-Instruct vs Mistral-Instruct 0.55. Saul-54B/141B beats GPT-4 on LegalBench by ~2 points (75.34% vs 73.35%).
- **Critical self-admission (§6.1):** *"SaulLM-7B-Instruct largely outperforms generic Instruct models on tasks that most require legal-specific knowledge, but is outperformed by Mistral-Instruct on the conclusion tasks, which necessitates more deductive reasoning."* — direct external concession that 30B legal tokens does not fix deductive reasoning. **Strongest single external citation for the capstone thesis.**
- **No prior concept-level work on SaulLM** — downstream literature is LoRA/QLoRA fine-tunes and generic RAG; no concept-grounding. Clean research gap.

### 3.4 File consolidation

All LCM/capstone artifacts previously scattered across Desktop and Drive-export folders were consolidated into the `lcm-research` repo with structured subfolders. The Desktop is now empty of LCM/capstone files. Specifics:

- `paper/` — two versions of the Capstone Research Paper draft (Apr 27 + May 6)
- `papers/` — 16 numbered reference PDFs + 1 unique arxiv paper
- `notes/` — 7 research-notes docx + archived Desktop CLAUDE.md/.mcp.json
- `presentations/` — 6 decks (team brief, capstone scope, full scope, LLM-vs-LCM, in-class intro)
- `notebooks/concept-lm-staged-notebooks/` — 8 publication-ready notebooks
- `team/` — full team Drive export (admin, meeting notes, deliverables)
- `prior-work/marshall-lcm-0.1/` — earlier solo Marshall iteration (Dec 2024–Mar 2025)
- `reference/meta-lcm-repo/` + `reference/concept-lm-snapshot/`
- `saul-7b/` — first SaulLM notebook
- `legal-rag/` — **the new paper's deliverable notebook (Section 6)**

Two commits pushed to GitHub:
- `lcm-research` main → `f4e6931 project: consolidate capstone artifacts into repo`
- `knowledge-base` main → `5133c8c wiki: add LCM capstone lit review and project pages` and `4942eb2 infra: track MCP server and world-state reference doc`

The `concept-lm` working tree moved from `~/Desktop/LCM Research/concept-lm/` to `~/Documents/GitHub/concept-lm 2/` (macOS Finder appended " 2" automatically due to a pre-existing empty stub clone). Commit `9ca4d19 initial commit: concept-lm architecture and pipeline` preserved.

KB path references updated across 18 `wiki/sources/*.md` files plus the two project pages. KB log entry appended at `[2026-05-13] decision | consolidate capstone artifacts into lcm-research repo`.

### 3.5 Pushback on the partner's "Legal Workflow" document

A team member (likely Nicholas Mikhail or Jaysen Jensen) circulated `Legal Workflow.docx` proposing a 6-task evaluation suite for SaulLM:

1. Legal Text Classification (LEDGAR, LexGLUE)
2. Named Entity Recognition (parties, courts, statutes)
3A. Legal Concept Identification (doctrinal concepts)
3B. Legal Concept Relation Mapping (graph-style)
4. Legal Precedent Retrieval
5. Legal Question Answering
6. Legal Document Generation

The handoff position taken in this session: the proposal is well-organized and uses pre-existing benchmarks, but **has no central thesis** ("we benchmarked SaulLM on 6 tasks" is a result, not a claim); **duplicates LegalBench** (which the SaulLM paper itself reports on, Fig 5); is **~5× too large for 14 weeks**; and **severed the connection to the LCM thesis**. The recommended response: keep the workflow as a §2 framing, but converge on one task (tasks 4/5 fused) with a specific intervention.

### 3.6 Convergence on Paper A (Concept-Level Legal RAG)

After explicit ruling-out of several alternatives (training a small concept-LM, building Legal WordNet, full 6-task benchmark, adding DLCM as a generative comparison), the team landed on a focused experimental paper:

**Test:** does SONAR sentence-embedding retrieval beat BM25 retrieval when augmenting SaulLM on CaseHOLD multiple-choice legal QA?

This connects directly to the LCM thesis (the concept-level claim made empirical on retrieval) while requiring zero novel data collection.

### 3.7 Web-search confirmation of novelty (the gap)

A web search confirmed:

- **Done extensively:** RAG on legal benchmarks (LRAGE, April 2025, has pre-built BM25 + FAISS on LegalBench/KBL/LawBench); SaulLM on CaseHOLD-adjacent tasks; LegalBERT/Legal-SBERT vs BM25 comparisons; the published finding that **BM25 often beats dense retrieval in legal** (Korean Bar Exam study).
- **Not done:** SONAR specifically as a legal retriever (SONAR is the multilingual-concept embedding used by Meta LCM and ByteDance DLCM; nobody has tested it on English legal text); the "does concept-level embedding transfer to legal retrieval" question; SaulLM + retrieval evaluations.

**Risk acknowledged:** SONAR may underperform BM25 (consistent with the legal-RAG literature). This is treated as a feature, not a bug — the strongest version of the paper is honest about the negative result if it occurs, and frames it as evidence that general-domain concept embeddings do not transfer to legal language.

### 3.8 Notebook design iterations (this session)

The notebook went through several configurations as scope was tightened:

| Version | Models | Retrievers | Note |
|---|---|---|---|
| v1 (proposed) | Saul-7B-Instruct + Mistral-7B-Instruct + Llama-3-8B-Instruct | bare, BM25, SONAR, LegalBERT-DPR | Too wide; runtime concerns |
| v2 (recommended) | Saul-7B-Instruct + Mistral-7B-Instruct | bare, BM25, SONAR | Standard ablation table |
| v3 (considered) | + Saul-7B-Base + Mistral-7B-Base | + DLCM-chunked retriever | DLCM not feasible without weights; Base hardly performs on multiple choice |
| **v4 (final, locked)** | **Saul-7B-Instruct only** | **bare, BM25, SONAR** | **Tightest scope, 1 row × 3 cells** |

The final notebook lives at `~/Documents/GitHub/lcm-research/legal-rag/legal_rag_demo.ipynb`.

### 3.9 Installation friction (currently active)

The notebook's install cell ran into compound issues in the user's anaconda base environment:

1. User added invalid version pins (`torch==2.11.0`, `torchvision==0.26.0`, `fsspec==2025.12.0`) that don't exist on PyPI → pip fell back to source builds → `setuptools.build_meta` import error. **Fixed by removing pins.**
2. `fsspec` version conflict: `datasets` requires ≤2026.2.0; `sonar-space` bumped to 2026.4.0. **Fixed by explicit `fsspec==2026.2.0` re-pin at end of install cell.**
3. `fairseq2` import requires `libsndfile` C library; not present on the system. `conda install -c conda-forge libsndfile==1.0.31` stalled because the anaconda base env is from 2022 and the solver enters a long flexible-solve loop trying to reconcile holoviews/spyder/scikit-learn/anaconda pins.
4. **Current state:** recommended fix is `brew install libsndfile` followed by symlinking `/opt/homebrew/lib/libsndfile.dylib` to `/usr/local/lib/libsndfile.dylib` (the latter is on macOS's default dyld search path). Fallback if symlink fails: create a fresh venv outside conda.

---

## 4 — Final research direction

### 4.1 Thesis (one sentence)

Domain-specialized legal LLMs (SaulLM) inherit the same token-level retrieval bottleneck as generic LLMs; retrieval augmentation using concept-space sentence embeddings (SONAR) may close part of this gap, with the magnitude of effect dependent on whether general-domain concept embeddings transfer to specialized legal language.

### 4.2 Two hypotheses

**H1 — The retrieval-method hypothesis.**
Concept-level retrieval (SONAR) outperforms token-level retrieval (BM25) on CaseHOLD when both augment SaulLM, because legal language uses many surface forms for the same underlying doctrine.

| Outcome | Interpretation |
|---|---|
| SONAR > BM25 | Concept embeddings transfer to legal retrieval. Supports the LCM thesis. |
| SONAR ≈ BM25 | General-domain concept embeddings do not pick up legal-specific meaning. Concept-level retrieval requires domain training. |
| SONAR < BM25 | General-domain concept embedding space actively misaligns with legal text. Strong negative result. Consistent with the published Korean Bar Exam finding. |

**H2 — The retrieval-augmentation hypothesis (implicit).**
Adding retrieval improves SaulLM's accuracy over the bare condition on CaseHOLD, regardless of retrieval method.

### 4.3 No-boring-result property

All three H1 outcomes are publishable. The paper is structured to make whichever outcome occurs intellectually coherent with the broader concept-modeling literature.

### 4.4 Pre-existing baselines for citation

| Source | Reports |
|---|---|
| Zheng et al. 2021 (CaseHOLD paper, arXiv:2104.08671) | BERT-base, BERT-large, RoBERTa, LegalBERT scores on CaseHOLD |
| Colombo et al. 2024 (SaulLM-7B, arXiv:2403.03883) | SaulLM-7B-Instruct vs Mistral-Instruct on LegalBench |
| arXiv:2505.02172 ("Identifying Legal Holdings with LLMs", May 2025) | GPT-4o, AmazonNovaPro on CaseHOLD F1 ≈ 0.72–0.74 |
| Kim et al. 2025 (LRAGE, arXiv:2504.01840) | BM25 vs dense retriever performance on legal benchmarks |

---

## 5 — Paper structure

### 5.1 Existing 8-section outline (from `paper/Capstone Research Paper - 2026-05-06 (latest).docx`)

1. **Introduction**
2. **Why LLMs Fail at High-Stakes Strategic Domains**
   - 2.1 What Strategic Reasoning in Law Actually Requires
   - 2.2 Empirical Failure Demonstration
   - 2.3 Why Scale Does Not Fix This
3. **Concept-Level Tokenization as the Right Direction**
   - 3.1 The Token-Level Modeling Problem
   - 3.2 What Concept-Level Modeling Provides
   - 3.3 Novel Technical Claim: Why Concept-Level Modeling Is Necessary, Not Just Efficient
   - 3.4 Legal Text as a Stress Test for Concept-Level Modeling
4. **The Boundary Detection Problem**
   - 4.1 The Problem Stated Precisely
   - 4.2 Survey of Existing Boundary Detection Methods
   - 4.3 The Novel Technical Claim: Proxy Signals vs. Semantic Ground Truth
   - 4.4 What Ground Truth for Concept Boundaries Would Look Like
5. **Legal WordNet as Semantic Ground Truth**
   - 5.1 The Compound Ground Truth Argument
   - 5.2 Scope and Domain Selection
6. **Legal WordNet Construction Methodology**
   - 6.1 Node Design
   - 6.2 Edge Design and Relationship Taxonomy
   - 6.3 Source Hierarchy
   - 6.4 Validation Protocol
   - 6.5 Graph Properties
   - 6.6 Replication and Extension
7. **Experiments**
   - 7.1 Experiment 1: Quantifying the Supervision Gap
   - 7.2 Experiment 2: Graph-Augmented Reasoning
   - 7.3 Joint Interpretation
8. **Conclusion**

### 5.2 Section-by-section mapping under the new direction

| Section | Action under new direction |
|---|---|
| 1 Intro | **Keep.** Reframe one-paragraph thesis as retrieval-evaluation rather than graph-construction. |
| 2 Why LLMs Fail | **Keep.** §2.2 (empirical failure demo) populated with two Saul probes as exhibits: *Brennan v. Whitfield* fabrication + broken-IRAC fact pattern. §2.3 anchored by Colombo §6.1 quote on conclusion-task failure. |
| 3 Concept-Level Tokenization | **Keep with reframing.** Argument shifts from "we should train a concept-LM" to "concept space matters for retrieval, even when the LM stays token-level." DLCM/LCM cited as motivation; SONAR introduced here as the off-the-shelf concept embedding we use. |
| 4 Boundary Detection Problem | **Reframe as "The Token-Retrieval Gap."** Same proxy-vs-semantic critique applied to retrieval rather than to within-document chunking. §4.2 surveys BM25 / DPR / LegalBERT-DPR / SONAR retrieval. Probe 1 (Saul's sentence-level boundary defaults) cited here. |
| 5 Legal WordNet as Ground Truth | **Cut to one paragraph or drop.** Position as future-work motivation for source-grounded retrieval. |
| 6 Legal WordNet Construction | **Drop entirely.** |
| 7 Experiments | **Single experiment (renamed):** "Retrieval Augmentation on CaseHOLD." §7.1 setup, §7.2 results table (1 row × 3 cells: bare/BM25/SONAR), §7.3 error analysis (qualitative breakdown of which question types each retriever helps or hurts on). |
| 8 Conclusion | **Keep.** Future work includes Legal WordNet construction (positioned as the "next paper"), domain-pretrained concept embeddings, DLCM-style dynamic chunking. |

### 5.3 Headline figure (Figure 1 of §7)

A bar chart with three bars: bare SaulLM, +BM25, +SONAR. Y-axis accuracy on CaseHOLD. Generated automatically by cell 8 of the notebook.

---

## 6 — The notebook deliverable

**Path:** `~/Documents/GitHub/lcm-research/legal-rag/legal_rag_demo.ipynb`

### 6.1 Design

Ten cells, top-to-bottom runnable. Every output renders inline. Optional JSON export of full Q&A transcripts.

| Cell | Type | Content |
|---|---|---|
| 0 | markdown | Title, prereqs, runtime expectations |
| 1 | code | `!pip install` for all Python deps + fsspec pin |
| 2 | code | `!ollama pull` and `!ollama cp` for Saul model |
| 3 | code | Imports + module-level config (SEED=42, N_TEST=50, K_RETRIEVE=5, CORPUS_SIZE=2000, MODELS=['saul-7b'], RETRIEVER_NAMES=['bare','bm25','sonar'], SAVE_JSON=False) |
| 4 | code | Load CaseHOLD from HF, slice corpus to 2000 + test to 50 |
| 5 | code | Build BM25 (rank_bm25) + SONAR retrievers (cached embeddings to `sonar_corpus_2000.npy`); FAISS-IP index |
| 6 | code | `build_prompt`, `ask_model` (Ollama HTTP), `parse_letter` (regex on A–E) |
| 7 | code | Nested loop over models × retrievers × test set; records full transcripts |
| 8 | code | pandas groupby + matplotlib bar chart |
| 9 | code | JSON export gated on `SAVE_JSON` |

### 6.2 JSON output schema (when SAVE_JSON=True)

```json
[
  {
    "question_id": 0,
    "excerpt": "<case paragraph with [BLANK]>",
    "options": {"A": "...", "B": "...", "C": "...", "D": "...", "E": "..."},
    "correct": "B",
    "results": [
      {
        "model": "saul-7b",
        "retriever": "bare",
        "retrieved": null,
        "answer_raw": "<saul's raw output>",
        "predicted": "B",
        "correct": true,
        "latency_s": 2.3
      },
      {"model": "saul-7b", "retriever": "bm25", "retrieved": ["...", "..."], ...},
      {"model": "saul-7b", "retriever": "sonar", "retrieved": ["...", "..."], ...}
    ]
  }
]
```

One object per question, all retrieval conditions nested. Allows error analysis (§7.3 of paper) without rerunning the model.

### 6.3 Expected runtime (Apple Silicon, M3 Pro 18 GB)

- First run: ~15 minutes total (≈10 min one-time SONAR encoding of 2000 sentences on CPU + ≈5 min for 150 Ollama calls).
- Subsequent runs: ~5 minutes (SONAR embeddings cached to disk).
- Scaling to full CaseHOLD test (N_TEST=3600) and full training-set retrieval corpus (CORPUS_SIZE=45000): overnight run for first SONAR encoding pass; few hours per inference run after that.

### 6.4 Current blocker

Cell 3 raises:
```
OSError: fairseq2 requires libsndfile. Since you are in a Conda environment,
use `conda install -c conda-forge libsndfile==1.0.31` to install it.
```

Root cause: `fairseq2._load_sndfile()` calls `ctypes.util.find_library("sndfile")`, gets None, detects `CONDA_PREFIX` is set, raises conda-flavored error.

The conda install of libsndfile stalled (2022-era anaconda base env solver loop). Two fix paths available:

**Path A (in-place fix, faster):**
```bash
brew install libsndfile
mkdir -p /usr/local/lib
ln -sf $(brew --prefix)/lib/libsndfile.dylib /usr/local/lib/libsndfile.dylib
python3 -c "import ctypes.util; print(ctypes.util.find_library('sndfile'))"
# Should print a path. Restart Jupyter kernel. Re-run cell 3.
```

**Path B (clean venv, more robust long-term):**
```bash
cd ~/Documents/GitHub/lcm-research/legal-rag
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install ipykernel datasets rank-bm25 faiss-cpu requests tqdm pandas matplotlib numpy fairseq2 sonar-space "fsspec==2026.2.0"
python -m ipykernel install --user --name=legal-rag --display-name="legal-rag"
# In notebook: Kernel → Change Kernel → legal-rag → run from cell 3
```

---

## 7 — Locked decisions (do not re-litigate)

| Decision | Set on | Rationale |
|---|---|---|
| **No model training** | 2026-05-06 (KB) | concept-lm preserved as future work; no GPU budget; no time |
| **No Legal WordNet construction in this paper** | 2026-05-13 (this session) | high-labor, needs unrecruited expert grader, dilutes connection to current research |
| **No DLCM as a comparison condition** | 2026-05-13 (this session) | no released weights; reimplementation from paper expands scope past 6-week target |
| **Saul-Instruct only (no Mistral, no Base, no Llama)** | 2026-05-13 (this session) | tightest scope; the retrieval comparison is the central claim, not the model-pretraining ablation |
| **CaseHOLD as the only benchmark** | 2026-05-13 (this session) | programmatic scoring; no graders; published baselines for comparison; LegalBench as a follow-up if time permits |
| **No novel data collection** | 2026-05-13 (this session) | reduces failure modes; tightens timeline; preserves "Lego assembly" property |
| **Deliverable artifact = reproducibility notebook + paper** | 2026-05-13 (this session) | the notebook IS the experimental engine; the paper is the argument that wraps it |

---

## 8 — Files and locations

### 8.1 Project tree (`~/Documents/GitHub/lcm-research/`)

```
lcm-research/
├── HANDOFF.md                  ← this file
├── README.md
├── CLAUDE.md                   ← project-level Claude instructions
├── .mcp.json
├── .gitignore                  ← excludes binaries (papers/, presentations/, etc.)
├── paper/
│   ├── Capstone Research Paper - 2026-04-27 (drive version).docx
│   └── Capstone Research Paper - 2026-05-06 (latest).docx
├── papers/                     ← 17 reference PDFs (gitignored)
├── notes/                      ← 7 research-notes docx
├── presentations/              ← 6 decks (gitignored)
├── notebooks/concept-lm-staged-notebooks/  ← 8 publication-ready notebooks
├── team/                       ← Drive export (gitignored)
├── prior-work/marshall-lcm-0.1/  ← earlier solo iteration (gitignored)
├── reference/
│   ├── meta-lcm-repo/
│   └── concept-lm-snapshot/
├── saul-7b/
│   ├── saul_chat.ipynb         ← first SaulLM notebook (multi-turn chat demo)
│   ├── requirements.txt
│   └── probe_results.json      ← 8-probe transcript (qualitative paper exhibits)
└── legal-rag/
    └── legal_rag_demo.ipynb    ← the deliverable artifact (currently failing on cell 3)
```

### 8.2 Adjacent repos

| Repo | Path | Purpose |
|---|---|---|
| `knowledge-base` | `~/Documents/GitHub/knowledge-base/` | KB schema, lit review (16 papers), project pages, decision log |
| `concept-lm 2` | `~/Documents/GitHub/concept-lm 2/` | Full concept-LM working tree (5,700 LoC), preserved as future work. Hosts the CourtListener scraper + SONAR encoder + boundary detector + training pipeline. Note: macOS Finder appended " 2" to the folder name during the move on 2026-05-13. |
| `LegalNet` | `github.com/nicholasdhaliwal/LegalNet` (PUBLIC, no local clone in lcm-research) | Legal WordNet repo with 38 seeded nodes; frontend at nicholasdhaliwal.github.io/legalaxy. Preserved for future-work use. |

### 8.3 KB references

| KB path | Contents |
|---|---|
| `wiki/projects/lcm-capstone/index.md` | Master project page (still reflects 2026-05-06 locked plan, needs update for the 2026-05-13 pivot) |
| `wiki/projects/lcm-capstone/team-brief.md` | Team alignment doc |
| `wiki/projects/lcm-capstone/elevator.md` | Elevator pitch |
| `wiki/projects/lcm-capstone/doctrine-decision.md` | Contract Formation pick rationale |
| `wiki/sources/{16-papers}.md` | Full lit review |
| `log.md` | Decision log with `[2026-05-13]` consolidation entry |

**KB state pending update:** the project pages still describe the pre-pivot scope. They should be updated to reflect Paper A as the new direction once the notebook runs successfully and the pivot is firm.

---

## 9 — Open issues / next session priorities

1. **Resolve the libsndfile import error** (Section 6.4). Without this, cell 3 fails and the whole notebook is blocked.
2. **First end-to-end run of the notebook with N_TEST=50.** Verify all three retrievers produce sensible accuracy numbers (anything above chance 20% on bare; non-zero for retrieval conditions). Save the resulting `results.json`.
3. **First error analysis pass.** Manually inspect 10–20 questions where SONAR retrieval changed the answer from wrong to right (or right to wrong). Build the qualitative taxonomy for §7.3 of the paper.
4. **Update KB project pages** (`wiki/projects/lcm-capstone/index.md` and `team-brief.md`) to reflect the new direction: drop Legal WordNet from in-scope, add CaseHOLD/SONAR/BM25 paper as the new deliverable.
5. **Communicate the pivot to the team.** The partner who wrote `Legal Workflow.docx` is owed a clear message explaining what was kept (their workflow taxonomy as §2 framing) and what was changed (one focused experiment, not six). Suggested wording is in this session's chat.
6. **Decide on N_TEST and CORPUS_SIZE for the paper-grade run.** N_TEST=500 minimum recommended for statistical credibility; CORPUS_SIZE=45000 (full training set) for the paper run. Overnight on first encoding pass.
7. **Future-work paragraph in §8.** Connect to Legal WordNet (positioned as the "next paper"), DLCM-style dynamic chunking, domain-pretrained concept embeddings (e.g., legal-trained SONAR variants).

---

## 10 — What NOT to do (locked-out paths)

Reiterating Section 7 in the negative:

- Do not train, fine-tune, or LoRA-adapt SaulLM, Mistral, or any other model. The 6-week budget cannot absorb training. The team-brief explicitly says no.
- Do not begin Legal WordNet node construction. It is now positioned as future work; starting it pulls the project back to the pre-pivot scope.
- Do not reintroduce expert-grader recruitment. CaseHOLD is auto-scored; no graders needed.
- Do not add Mistral or Llama as comparison models in this notebook. The retrieval comparison is the central claim.
- Do not implement DLCM from the paper. No released weights; reimplementing from scratch expands scope.
- Do not collect new data from CourtListener, Caselaw Access Project, or Pile-of-Law. The training-set excerpts that ship with CaseHOLD ARE the retrieval corpus.
- Do not invent new tasks beyond CaseHOLD in this paper. LegalBench subtasks can be a robustness check in a stretch goal, not the centerpiece.

---

## 11 — Reproducibility checklist

When the notebook runs end-to-end and a `results.json` is produced, the artifact should be reproducible by any reader as follows:

1. Install Ollama. Install Python 3.10+. Have ~10 GB free disk.
2. `git clone github.com/nicholasdhaliwal/lcm-research && cd lcm-research/legal-rag`
3. Open `legal_rag_demo.ipynb` in Jupyter or VS Code.
4. Run cells top-to-bottom. First-time SONAR encoding adds ~10 minutes; cached for subsequent runs.
5. Set `SAVE_JSON = True` in the config cell to write `results.json`.
6. Verify the printed accuracy table and the bar chart figure match the paper's Figure 1.

The notebook's only external dependency beyond Hugging Face is Ollama (locally installed). All Python deps are PyPI-installable. All model weights are publicly hosted. All data is publicly licensed (CaseHOLD is MIT, drawn from public-domain U.S. court opinions).

---

## 12 — Closing notes

This handoff captures the project state as of 2026-05-13 ~20:00 PT, immediately after the install-debugging session for `legal_rag_demo.ipynb`. The pivot from Legal-WordNet-construction to retrieval-evaluation is **functionally final** but **not yet ratified by the team or reflected in KB project pages.** The notebook deliverable exists and is one install-fix away from producing real numbers.

The strongest single move available before the next team meeting is **getting the notebook to run end-to-end**, taking a screenshot of the resulting bar chart, and walking the team through it. Numbers on a chart are more convincing than scope-discussion. Once the first run is in hand, KB updates and team communication follow naturally.

The qualitative SaulLM probes (probe_results.json) constitute paper-ready exhibits for §2 already, independent of the notebook's status. Even if the retrieval pipeline takes another day to debug, the §2 motivation can be drafted on Brennan v. Whitfield and broken-IRAC alone.
