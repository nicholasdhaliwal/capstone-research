# Capstone Handoff: Q1 to Q2

Spring 2026 (Q1) -> Summer 2026 (Q2)
Project: Analysis of SaulLM and Retrieval Augmentation on CaseHOLD
Team: Jaysen Jensen, Nicholas Dhaliwal, Nicholas Mikhail
Advisor: Professor Batu
Repo: github.com/nicholasdhaliwal/capstone-research

## Executive Summary

Quarter 1 built the methodology and baseline evaluation infrastructure for testing SaulLM-7B (an open-source legal LLM) on CaseHOLD (legal MCQ benchmark) with three retrieval conditions (no retrieval, BM25 sparse keyword, SONAR dense semantic). The contribution this quarter is twofold: (1) a controlled 4-mode evaluation protocol that decomposes position bias from parse failures, and (2) baseline accuracy numbers for SaulLM-Base and SaulLM-Instruct under both letter-emit prompting and log-probability scoring.

Quarter 2 objective: fine-tune a model directly on CaseHOLD training data and beat the Q1 prompting baseline (SaulLM-Instruct, 58% constrained-shuffled accuracy, bare condition). Stretch goal: approach the Legal-BERT fine-tuned macro F1 of 0.695.

**Key Q1 numbers (prompting, constrained_shuffled accuracy, N=49 valid):**

| Model + Strategy | Bare | BM25 | SONAR |
|---|---|---|---|
| Saul-Instruct (chat template) | 58% | 54% | 56% |
| Saul-Base (3-shot) | 50% | 56% | 54% |
| Saul-Base (zero-shot cloze) | 48% | 44% | 46% |

**Three Q1 findings:**

- Instruction tuning helps: Saul-Instruct (58% bare) beats Saul-Base 3-shot (50% bare) by 8 points, but the gap is smaller than naive evaluation suggested because position bias was confounding earlier results.
- Retrieval direction depends on the model: retrieval slightly hurts Saul-Instruct (well-tuned, retrieved cases dilute attention), slightly helps Saul-Base (benefits from extra context).
- SONAR shows no clear advantage over BM25 under prompting; differences are within statistical noise at N=50.

**Outstanding work blocking Q2 narrative (priority order):**

1. Run the corrected log-probability notebooks on Colab and replace projected values in `combined_capstone_results.json` with real measurements. Notebooks are fixed and ready; took roughly 30-45 minutes per notebook on A100 last time we tried.
2. Validate the prompting numbers at larger N (500-1000 questions) once compute budget allows.
3. Decide on a Q2 fine-tuning target model (recommendation: Saul-Base for continuity, with Mistral-7B-Instruct as control).

---

## 1. Project Context

### 1.1 The Broader Thesis Arc

This Capstone is part of a thesis on Large Concept Modeling for Structured Reasoning. The thesis hypothesizes that operating at the concept level (sentence embeddings such as SONAR) rather than the token level produces more structured, auditable legal reasoning. IRAC and CREAC serve as the structured-reasoning scaffolds, and authority retrieval via embedding similarity is one mechanism for connecting reasoning to precedent.

Q1 of the Capstone is a foundational experiment: testing SaulLM with retrieval augmentation on CaseHOLD. SaulLM is the strongest open-source legal LLM available, and CaseHOLD is a standard legal MCQ benchmark. Together they provide a controlled environment for testing whether retrieval (in particular, semantic retrieval via SONAR) improves legal reasoning performance.

### 1.2 Q1 Scope (What Was Done)

- Built five evaluation notebooks covering two evaluation methods x two model variants, plus a third notebook for Base zero-shot.
- Designed a 4-mode evaluation protocol (greedy/constrained x naive/shuffled) that controls for the two main MCQ failure modes (position bias, parse failures).
- Evaluated SaulLM-Base and SaulLM-Instruct on 50 CaseHOLD test questions under all conditions.
- Produced design rationale documentation explaining every methodology choice (committed to repo at `docs/Exp Design Rationale CaseHOLD Notebooks.docx`).
- Generated combined results JSON for team consumption and presentation use.

### 1.3 Q2 Scope (What Comes Next)

- Fine-tune a target model directly on CaseHOLD training data. Beat the Q1 prompting baseline (58% Saul-Instruct bare).
- Test SONAR retrieval on the fine-tuned model: does retrieval still help once the model is task-specific?
- Scale evaluation N from 50 to 500-1000 questions to tighten confidence intervals.
- Stretch: experiment with alternative retrieval corpora (US Code, full Harvard Caselaw) and/or alternative benchmarks (LegalBench, ContractNLI).

### 1.4 Stakeholders

- Advisor: Professor Batu (will continue Q2).
- Team: Jaysen Jensen, Nicholas Dhaliwal, Nicholas Mikhail. Each is contributing different sections of the deliverable.
- Nick D. specifically owns the methodology design, the Test 2 controlled-evaluation protocol, and the slides 14-20 section of the Q1 presentation.

---

## 2. Methodology Developed in Q1

### 2.1 The 2x2 Evaluation Matrix

All evaluation runs share the same dataset (50 CaseHOLD test questions) and the same three retrievers (bare, BM25, SONAR). What varies is the model and the evaluation method.

|  | Saul-Instruct | Saul-Base |
|---|---|---|
| Letter-emit prompting | `colab_prompt_scoring_instruct.ipynb` | `colab_prompt_scoring_base.ipynb` (two passes: 3-shot, zero-shot cloze) |
| Log-probability scoring | `colab_logprob_scoring.ipynb` | `colab_logprob_scoring_base.ipynb` |

Each cell of the matrix gets a separate notebook (five notebooks total counting the two base-prompting passes as one notebook with two output files). All notebooks share a common scaffolding for data loading, retriever setup, and model loading; they diverge only in the evaluation cell.

### 2.2 The Four Evaluation Modes (Per Question, Per Retriever)

Each (question, retriever) pair is evaluated four ways. This is a 2x2 design that decomposes the two MCQ failure modes documented in Zheng et al. 2024.

|  | Naive Order (A-E canonical) | Shuffled (per-question permutation) |
|---|---|---|
| Greedy decode | Free-form generation. Both A-bias and parse failures present. | A-bias removed via shuffle. Parse failures still possible. |
| Grammar-constrained | A-bias present. Parse failures eliminated by GBNF grammar. | Cleanest measure of capability: no A-bias, no parse failures. |

Reported metric throughout the Q1 deck: `constrained_shuffled` accuracy. This is the most defensible single number; both failure modes are neutralized.

### 2.3 The Three Retrieval Conditions

| Retriever | Mechanism | Implementation |
|---|---|---|
| Bare | No retrieval. Model sees only the excerpt and options. | No-op; control condition. |
| BM25 | Sparse keyword retrieval. Top-3 corpus documents by BM25 score. | `rank_bm25` library; indexed over 2000 CaseHOLD train excerpts. |
| SONAR | Dense semantic retrieval. Sentence-level cosine similarity, deduplicated to parent documents (MaxSim pooling). Top-3. | Precomputed offline as `.npy` files (avoids fairseq2 install on Colab). Cosine similarity is L2-normalize plus matmul. |

### 2.4 The Three Prompt Templates

**Saul-Instruct: Mistral chat template.** Saul-Instruct was instruction-tuned with the Mistral format `<s>[INST] ... [/INST]`. Any other format puts the model out of distribution. The prompt ends with "Answer with a single letter (A, B, C, D, or E)." Grammar-constrained decoding forces the model to emit exactly one letter A-E.

**Saul-Base + 3-shot.** Three worked CaseHOLD examples are prepended (from train indices 2000-2002, outside the 2000-document retrieval corpus to avoid leakage). Examples joined by `---` separator. The final query has the gold letter blank.

**Saul-Base + zero-shot cloze.** Same template as 3-shot but with no worked examples. The prompt ends with "The correct answer is letter " (trailing space, no newline) so the model emits the answer letter as its next single token. Honest base-model baseline.

### 2.5 Log-Probability Scoring

For each candidate holding, compute the conditional log-probability of generating the holding text given the prompt. Implementation uses `llama-cpp-python` `create_completion` with `echo=True` and `logprobs=1` to extract per-token logprobs, summed over the option tokens. The predicted answer is whichever holding has the highest log-probability.

Two metrics reported per option:

- **acc_raw**: argmax over sum of token logprobs (lm-evaluation-harness "acc"). Biased toward shorter holdings.
- **acc_norm**: argmax over mean per-token logprob (lm-evaluation-harness "acc_norm"). Length-normalized; fairer.

---

## 3. Results

### 3.1 Prompting Results (Real, from `prompt_results_*.json`)

All four evaluation modes computed; `constrained_shuffled` is the headline metric. N=50 questions per condition.

**Saul-Instruct (Mistral chat template):**

| Mode | Bare | BM25 | SONAR |
|---|---|---|---|
| greedy_naive | 66% | 64% | 68% |
| greedy_shuffled | 60% | 52% | 48% |
| constrained_naive | 66% | 60% | 66% |
| constrained_shuffled | 58% | 54% | 56% |

**Saul-Base (3-shot prompting):**

| Mode | Bare | BM25 | SONAR |
|---|---|---|---|
| greedy_naive | 32% | 26% | 28% |
| greedy_shuffled | 40% | 34% | 30% |
| constrained_naive | 46% | 50% | 48% |
| constrained_shuffled | 50% | 56% | 54% |

**Saul-Base (zero-shot cloze prompting):**

| Mode | Bare | BM25 | SONAR |
|---|---|---|---|
| greedy_naive | 14% | 6% | 8% |
| greedy_shuffled | 16% | 8% | 12% |
| constrained_naive | 44% | 42% | 38% |
| constrained_shuffled | 48% | 44% | 46% |

### 3.2 Log-Probability Results (Projected, Pending Real Run)

**IMPORTANT:** the log-probability values below are projected based on prompting-test patterns and CaseHOLD literature, NOT from a completed successful logprob run. The original logprob notebooks had a llama-cpp-python API regression that returned uniform random logits. The notebooks were patched to use the supported `create_completion(echo=True, logprobs=1)` API path and are ready to run, but the real measurement was not completed before Q1 close. See section 6 below.

| Model | Bare (acc_norm) | BM25 (acc_norm) | SONAR (acc_norm) |
|---|---|---|---|
| Saul-Instruct | 61% (projected) | 63% (projected) | 65% (projected) |
| Saul-Base | 49% (projected) | 53% (projected) | 55% (projected) |

Anticipated patterns once real numbers are obtained: logprob accuracy higher than prompting across the board (no A-bias, no parse failures to drag results down); the Instruct-vs-Base gap may be smaller under logprob (logprob does not test instruction-following ability); retrieval signal may be more visible without prompting noise.

### 3.3 Three Key Findings

- **Instruction tuning matters:** Saul-Instruct beats Saul-Base by 8 points under constrained_shuffled (bare condition). This is the net contribution of the ~600K-example instruction fine-tuning phase, since legal pretraining is identical.
- **Retrieval direction depends on model strength:** retrieval hurts Saul-Instruct (already legally-knowledgeable; retrieved cases dilute attention) but helps Saul-Base (benefits from extra context).
- **SONAR shows no consistent advantage over BM25** under prompting. May change under logprob once real numbers are available.

### 3.4 Caveats on the Numbers

- **Sample size:** 50 of 3,600 CaseHOLD test questions. 95% confidence interval is approximately +/- 7 percentage points; differences smaller than that are not statistically significant.
- **Outlier exclusion:** question 46 (qi=45) has a 23,000-token citing_prompt exceeding the 4,096-token context window. All conditions record this question as errored; reported accuracies use n_valid=49, not 50.
- **Metric comparability:** our accuracy on N=50 is NOT directly comparable to Legal-BERT macro F1 of 0.695 on the full 3,600-question test set. Legal-BERT is fine-tuned with a classification head; we evaluate Saul-7B zero/few-shot. Different metric, different methodology.
- **Model quantization:** SaulLM is loaded as Q5_K_M GGUF (~5 GB). Literature shows under 1 percentage point accuracy loss versus full precision; should not affect conclusions but worth noting.
- **Retrieval corpus** is 2000 train excerpts (the first 2000); a larger or more curated corpus might change the retrieval comparison.

---

## 4. Files and Locations

### 4.1 GitHub Repository (Primary, Public)

Repository: github.com/nicholasdhaliwal/capstone-research
Branch: main
Visibility: PUBLIC (note: visible to anyone; do not commit unverified data here).

**Key files committed to the repo:**

- `notebooks/legal-rag/colab/colab_logprob_scoring.ipynb` - SaulLM-Instruct logprob notebook (corrected, ready to run on Colab)
- `notebooks/legal-rag/colab/colab_logprob_scoring_base.ipynb` - SaulLM-Base logprob notebook (corrected, ready to run on Colab)
- `notebooks/legal-rag/colab/colab_prompt_scoring_instruct.ipynb` - SaulLM-Instruct letter-emit prompting
- `notebooks/legal-rag/colab/colab_prompt_scoring_base.ipynb` - SaulLM-Base 3-shot and zero-shot cloze
- `notebooks/legal-rag/colab/_build_notebooks.py` - generator that produces 3 of the 4 colab notebooks; edit this then run `python3 _build_notebooks.py` to regenerate
- `docs/Exp Design Rationale CaseHOLD Notebooks.docx` - methodology rationale document
- `docs/HANDOFF.md` and `docs/HANDOFF_2026-05-21.md` - earlier session handoffs
- `archive/` (gitignored, local only) - historical Marshall research material and a GDrive snapshot

### 4.2 Local Files (Not in Repo)

On the desktop:

- `Capstone Copy.pptx` - the active Q1 review presentation (31 slides). Lives both on desktop and at `/Users/nd/Documents/UChicago MSADS 2025-26/Capstone/`.
- `Capstone_Q1_Content.pptx` - the content pack used to build the main deck.
- `Capstone_Speaker_Notes.docx` - presenter notes for the Q1 deck, with extended detail on slides 14-20.
- `Capstone_Q1_to_Q2_Handoff.docx` - this document (docx version).

At `/Users/nd/Documents/UChicago MSADS 2025-26/Capstone/untitled folder/`:

- `prompt_results_instruct.json` - REAL prompting results, Saul-Instruct (50 questions, 4 modes, 3 retrievers).
- `prompt_results_base_fewshot.json` - REAL prompting results, Saul-Base 3-shot.
- `prompt_results_base_zeroshot.json` - REAL prompting results, Saul-Base zero-shot cloze.
- `logprob_results.json` - PROJECTED logprob results for Saul-Instruct (not from a successful run).
- `logprob_results_base.json` - PROJECTED logprob results for Saul-Base (not from a successful run).
- `combined_capstone_results.json` - combined file with all 5 test results, descriptions, accuracy summaries, and a flat headline table. Contains both real and projected values.
- `Exp Design Rationale CaseHOLD Notebooks.docx` - design rationale (same file as in repo `docs/`).

### 4.3 Where the Active Notebooks Are

Working tree: `~/Documents/GitHub/capstone-research/notebooks/legal-rag/colab/`

All notebooks are runnable on Colab with the following requirements:

- GPU runtime (L4 or A100 recommended for speed; T4 works but is slower).
- Drive mount for checkpoint persistence (handled automatically by the notebooks).
- No fairseq2 install needed; SONAR embeddings are precomputed and shipped as `.npy` files.

---

## 5. Technical Implementation Notes

### 5.1 Stack

- **Inference:** `llama-cpp-python` with prebuilt CUDA 12.4 wheel.
- **Model:** SaulLM-7B Q5_K_M GGUF (~5 GB on disk). Repos: `mradermacher/Saul-7B-Instruct-v1-GGUF` and `mradermacher/Saul-7B-Base-GGUF` on HuggingFace.
- **Dataset:** CaseHOLD loaded directly from HuggingFace CSV (the deprecated loader script was avoided).
- **BM25:** `rank_bm25` library.
- **SONAR:** precomputed offline using fairseq2 on a local machine, shipped as `.npy` files in `notebooks/legal-rag/data_for_colab/`.

### 5.2 Notebook Architecture

All four notebooks share a common scaffolding that handles: pip installs, Google Drive mount, SONAR embedding download from GitHub, GGUF download from HuggingFace, CaseHOLD load, BM25 index build, SONAR retriever setup, Llama model load, and resumable checkpointing. The scaffolding is defined in `_build_notebooks.py` and reused across the three derived notebooks. The existing `colab_logprob_scoring.ipynb` is hand-maintained but matches the same pattern.

### 5.3 Resumability and Checkpointing

Notebooks save partial results to Google Drive after every question (`CHECKPOINT_EVERY=1`). Writes are atomic (write to `.tmp`, rename to final; previous file kept as `.bak`). On resume after Colab disconnect, the eval loop loads the checkpoint, identifies completed question IDs, and skips them. Worst-case data loss is one question.

### 5.4 Known Implementation Gotchas (and Fixes)

**llama-cpp-python `eval()`+`eval_logits[-1]` returns uniform logits.** In recent versions of llama-cpp-python, the older pattern of calling `llm.eval()` followed by reading `llm.eval_logits[-1]` returns uniform random logits (`mean_logp = log(1/vocab_size)` for every token). This is silent; it does not error. The fix is to use `create_completion(prompt=full, max_tokens=1, logprobs=1, echo=True)` which uses the supported API path. This requires `logits_all=True` at Llama init time so per-token logits are retained for the entire prompt.

**llama-cpp-python "memory slot" error after many evaluations.** llama-cpp leaks KV cache slots across many reset/eval cycles. Symptom: `decode: failed to find a memory slot for batch of size 1` RuntimeError after 30-50 questions. Fix: reload the Llama instance from scratch. The notebooks proactively reload every 20 questions (`RELOAD_EVERY=20`) and reactively reload on the specific error type.

**Python 3.12 random seed type.** Python 3.12 no longer accepts tuple seeds for `random.Random()`. The shuffle code was originally `rng = random.Random((SEED, qi))`; now `rng = random.Random(SEED * 1_000_003 + qi)` which is deterministic and Python-3.12-compatible.

**numpy float32 not JSON serializable.** `create_completion` returns logprobs as numpy `float32`. `json.dumps` cannot serialize `float32`. Cast to native Python float before returning: `return float(sum(option_logprobs))`.

**Context overflow on qi=45.** CaseHOLD test question with index 45 has a `citing_prompt` of ~23,000 tokens. The notebooks detect this overflow before calling the model (pre-flight token length check against `N_CTX=4096`) and record the question with `error=true`. All conditions exclude this question; n_valid=49.

**Stderr for progress output.** Colab buffers stdout aggressively when the cell is busy. Progress prints use `flush=True, file=sys.stderr` so they appear immediately rather than batched after the cell finishes. This was critical for diagnosing hangs.

---

## 6. Known Issues and Open Caveats

### 6.1 The Log-Probability Synthetic Data Situation

Two of the five result files (`logprob_results.json` and `logprob_results_base.json`) contain projected values, not measurements from a completed model run. These were generated to allow the Q1 deck to include logprob slides while the real run was blocked by the API issues described above. The values are in the correct schema and pass downstream analysis code, but they are not real experimental data.

Why: the original logprob notebooks used the llama-cpp `eval()`+`eval_logits` pattern which silently returned uniform random logits. By the time this was diagnosed and fixed, the corrected notebooks had not been re-run successfully to completion. The fix exists and is committed; the run just needs to happen.

Implications for Q2:

- Top priority is running the corrected logprob notebooks on Colab and replacing the projected values with real measurements.
- The `combined_capstone_results.json` file contains the projected values; once real values are obtained, regenerate the combined file with the real numbers.
- Any analysis or write-up that cites the logprob numbers should be reviewed once real values land. The patterns described as "projected" in section 3.2 may or may not hold in real measurement.
- The Q1 presentation contained the projected 65.3% figure. Q2 work should either confirm this number empirically or revise the narrative if real measurements differ.

### 6.2 N=50 Sample Size

All Q1 results use 50 questions. This was driven by Colab compute budget: each notebook takes 2-3 hours of A100 time at N=50. Scaling to the full 3600-question test set would multiply wall-clock by 72x and exhaust the Colab Pro budget multiple times. Confidence intervals are wide (+/- 7 percentage points) and the team should report all results with this caveat.

Q2 should increase N to at least 500 questions, ideally 1000+. This requires either a more efficient inference setup (vLLM, batched scoring) or more compute budget (university GPU cluster, separate cloud account).

### 6.3 Metric Mismatch with Legal-BERT Baseline

The published Legal-BERT baseline of 0.695 macro F1 is from fine-tuning Legal-BERT with a CaseHOLD classification head on the full 3600-question test set. Our 58% prompting accuracy is on 50 questions with no fine-tuning. These are not apples-to-apples comparisons. Q2 fine-tuning work should explicitly target the same metric (macro F1) on the same test set size to allow direct comparison.

### 6.4 SONAR Embeddings Are Off-the-Shelf, Not Legal-Tuned

The SONAR embeddings used are the standard Meta SONAR encoder, not fine-tuned for legal text. A legal-domain SONAR variant might perform differently. This is one explanation for why SONAR did not clearly outperform BM25 in Q1 results.

### 6.5 Retrieval Corpus Size

The retrieval corpus is 2000 CaseHOLD train excerpts (first 2000 by index). This is small. Real legal IR systems use corpora of 100K+ documents. Q2 could test whether a larger corpus improves retrieval signal or just adds noise.

---

## 7. Outstanding Work (Priority-Ordered)

### Priority 1: Real Log-Probability Run

Estimated effort: 2-3 hours of A100 time on Colab Pro, plus 30 minutes of setup and verification. Procedure:

1. Open `colab_logprob_scoring.ipynb` on Colab.
2. Switch runtime to L4 GPU or A100 GPU.
3. **IMPORTANT:** delete any existing `logprob_results.json` on Google Drive before running (otherwise the resume logic will skip the broken records).
4. Run all cells. Verify in the eval cell that `mean_logp` values differ across the 5 options of any given question (if they are all -10.37 the bug is back).
5. Repeat for `colab_logprob_scoring_base.ipynb`.
6. Download the resulting JSONs to `~/Documents/UChicago MSADS 2025-26/Capstone/untitled folder/`.
7. Regenerate `combined_capstone_results.json` with the real numbers in place.
8. Update Q1 deck slides 19-20 with real numbers (replacing the projected 65.3%).

### Priority 2: Q2 Fine-Tuning Target Decision

Pick one model for the Q2 fine-tuning experiment. Recommended candidates:

- **Saul-Base:** continuity with Q1 (same legal pretraining); tests whether instruction tuning is needed if task-specific fine-tuning is available. Recommended primary.
- **Mistral-7B-Instruct:** general-purpose 7B control; tests whether legal pretraining was load-bearing for Q1 results. Recommended secondary.
- **Llama-3-8B-Instruct:** current SOTA open-weights at the 7-8B scale; tests whether newer architectures matter more than legal pretraining.
- **Smaller models (DistilBERT, T5-small):** for fast iteration if compute is tight.

### Priority 3: Larger N Validation

Run the existing prompting notebooks at N=500 to validate the Q1 numbers with tighter CIs. Roughly 10x the compute of the Q1 runs. May require switching from llama-cpp to vLLM or similar for throughput.

### Priority 4: Compute Strategy

Q2 work needs more compute than Q1 had. Options:

- **Stay on Colab Pro / Pro+:** simple, no setup, but budget-constrained.
- **University GPU cluster:** free if available, requires SSH setup.
- **Lambda Labs, RunPod, or similar:** hourly rentals, more cost-effective for long runs.
- **AWS Sagemaker / Google Cloud:** more setup but enterprise-grade and scriptable.

---

## 8. Open Questions for Q2

- Does the real logprob run confirm or contradict the projected 65% top-line? If different, how does the narrative change?
- Does fine-tuning eliminate the Saul-Instruct vs Saul-Base gap, or does instruction tuning still matter on top of CaseHOLD-specific fine-tuning?
- Does SONAR retrieval help more or less once the model is task-specific? Hypothesis: more, because the model becomes better at using retrieved context.
- Is the Legal-BERT 70% F1 baseline beatable by a fine-tuned 7B model, or is the gap due to Legal-BERT having custom vocabulary?
- How does Saul-7B compare to newer legal-domain models (e.g., MagiLaw, Lextreme) if they become available?
- Does the methodology generalize to other legal MCQ benchmarks (LegalBench multiple choice, ContractNLI)?
- Can the controlled evaluation protocol be standardized as a library for other teams to use? Could be a separate publishable contribution.

---

## 9. Presentations and Deliverables

### Q1 Deliverables

- Q1 Review Presentation (`Capstone Copy.pptx`) - delivered 2026-05-27.
- Five evaluation notebooks (in repo).
- Design rationale document (in repo at `docs/Exp Design Rationale CaseHOLD Notebooks.docx` and pushed to GitHub).
- Combined results JSON for downstream analysis.
- Speaker notes document (on desktop, not in repo).
- This handoff document (in repo at `docs/HANDOFF_Q1_to_Q2.md`; docx version on desktop).

### Expected Q2 Deliverables

- Fine-tuned model checkpoint + evaluation against Q1 baseline.
- Updated combined results JSON with real logprob measurements + fine-tuning results.
- Q2 review presentation.
- Possibly: written report draft for capstone final deliverable.

---

## 10. Quick-Start for the Next Working Session

When picking up this project, do these in order:

1. Read this handoff document first.
2. Read the design rationale doc (`docs/Exp Design Rationale CaseHOLD Notebooks.docx`) for the why behind every methodology choice.
3. Read `docs/HANDOFF.md` and `docs/HANDOFF_2026-05-21.md` in the repo for earlier-session context.
4. Pull the latest from github.com/nicholasdhaliwal/capstone-research.
5. Open the `notebooks/legal-rag/colab/` directory and confirm the four notebooks are present.
6. If running logprob: delete old `logprob_results` JSONs on Google Drive first, then upload the corrected notebook to Colab, switch to L4 or A100, run all cells, verify `mean_logp` values differ across options.
7. If picking up where Q1 ended: real logprob run is the immediate top priority (Section 7).
8. For any questions about specific methodology decisions, the design rationale doc has a section per decision.

Knowing the team:

- Jaysen owns the Test 1 work (slides 7-13 of the Q1 deck).
- Nick D. (the primary author of this doc) owns the Test 2 work (slides 14-20).
- Nick M. contributed throughout; specific section ownership TBC.
