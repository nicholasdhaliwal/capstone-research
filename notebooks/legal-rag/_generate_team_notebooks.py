"""Generate the three team-facing notebooks (saul_base, saul_bm25, saul_sonar)
from one source of truth. Each notebook is standalone; teammates run the one
they want without needing the others to exist.

Design choices:
  - Shared setup section (pip + ollama check + casehold load + prompt helpers)
  - Per-notebook retriever section (none / bm25 / cached-sonar)
  - Shared eval section (letter-emit via ollama chat)
  - Shared results section (accuracy table + bar chart + sample dump)
  - Optional final cell: load `results/logprob_results.json` (from the Colab
    notebook) and show it side-by-side as the authoritative baseline
"""
import json
from pathlib import Path

import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

REPO_RAW = "https://raw.githubusercontent.com/nicholasdhaliwal/capstone-research/main/notebooks/legal-rag/data_for_colab"
OUT_DIR  = Path(".")


# ---------- shared cells ----------

def header(title: str, retriever_label: str, retriever_desc: str) -> str:
    return f"""# {title}

Run Saul-7B-Instruct on CaseHOLD with **{retriever_label}**. {retriever_desc}

**Known finding (do not be surprised by the numbers):** Saul collapses to
predicting option `A` on ~80–90% of retrieval-augmented prompts, regardless
of retrieval method or prompt structure. This is documented in
`README.md` § A-bias finding. The numbers in this notebook are letter-emit
predictions; for the authoritative scoring see
`results/logprob_results.json` produced by the Colab notebook.

**Setup:** see `README.md` in this folder. You need a Python venv with the
deps listed in cell 1, plus Ollama running with `saul-7b` pulled.
"""

SETUP_CODE = """!pip install -q datasets rank-bm25 requests pandas matplotlib numpy
print('deps installed')"""

OLLAMA_CHECK_CODE = """import urllib.request, subprocess, time, requests

def _ollama_up():
    try:
        urllib.request.urlopen('http://localhost:11434/api/tags', timeout=2); return True
    except Exception:
        return False

if not _ollama_up():
    print('ollama is not running — start it from a terminal: `ollama serve`')
    raise SystemExit(1)

tags = requests.get('http://localhost:11434/api/tags').json()
names = {m['name'] for m in tags.get('models', [])}
if 'saul-7b:latest' not in names:
    print('saul-7b model missing — pull it from a terminal:')
    print('  ollama pull hf.co/mradermacher/Saul-7B-Instruct-v1-GGUF:Q4_K_M')
    print('  ollama cp hf.co/mradermacher/Saul-7B-Instruct-v1-GGUF:Q4_K_M saul-7b')
    raise SystemExit(1)
print('ollama up, saul-7b ready')"""

CONFIG_CODE_TEMPLATE = """import json, re, time
from pathlib import Path
import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt
from datasets import load_dataset

SEED         = 42
N_TEST       = 50
K_RETRIEVE   = 5
CORPUS_SIZE  = 2000
MODEL        = 'saul-7b'
RETRIEVER    = '{retriever}'          # 'bare' | 'bm25' | 'sonar'
OLLAMA_URL   = 'http://localhost:11434/api/chat'
RESULTS_DIR  = Path('results'); RESULTS_DIR.mkdir(exist_ok=True)
RESULTS_TAG  = f'{{MODEL}}_{{RETRIEVER}}'

np.random.seed(SEED)
print(f'config ready: model={{MODEL}}  retriever={{RETRIEVER}}  N_TEST={{N_TEST}}')"""

DATASET_CODE = """# Load CaseHOLD directly from HF CSV files instead of through the deprecated
# loader script (`casehold.py`). Works on any `datasets` version, no special
# pinning needed.
CASEHOLD_BASE = 'https://huggingface.co/datasets/casehold/casehold/resolve/main/data/all'
ds = load_dataset('csv', data_files={
    'train': f'{CASEHOLD_BASE}/train.csv',
    'test':  f'{CASEHOLD_BASE}/test.csv',
})

# CSV columns are positional. Header is 'Unnamed: 0,0,1,2,...,11'.
# Per the original casehold.py: col 0=example_id, col 1=citing_prompt,
# cols 2..6=holding_0..4, col 12=label.
def _rename(ex):
    return {
        'example_id':    int(ex['Unnamed: 0']),
        'citing_prompt': ex['0'],
        'holding_0':     ex['1'],
        'holding_1':     ex['2'],
        'holding_2':     ex['3'],
        'holding_3':     ex['4'],
        'holding_4':     ex['5'],
        'label':         int(ex['11']),
    }
ds = ds.map(_rename, remove_columns=ds['train'].column_names)
test = ds['test'].select(range(N_TEST))
print(f'test questions: {len(test)}')"""

# Retriever-specific cells (each notebook gets exactly one of these)

RETRIEVER_BARE_CODE = """# Bare condition — no retrieval. retrieve() returns None for every question.
def retrieve(qi, q_text):
    return None
train = ds['train'].select(range(CORPUS_SIZE)) if False else None  # not used in bare
print('bare retriever (no-op) ready')"""

RETRIEVER_BM25_CODE = """train = ds['train'].select(range(CORPUS_SIZE))
corpus_texts = [r['citing_prompt'] for r in train]
print(f'corpus: {len(corpus_texts)} excerpts')

from rank_bm25 import BM25Okapi
bm25 = BM25Okapi([t.lower().split() for t in corpus_texts])

def retrieve(qi, q_text, k=K_RETRIEVE):
    scores = bm25.get_scores(q_text.lower().split())
    return [corpus_texts[i] for i in np.argsort(scores)[-k:][::-1]]

print('bm25 retriever ready')"""

RETRIEVER_SONAR_CODE = f"""train = ds['train'].select(range(CORPUS_SIZE))
corpus_texts = [r['citing_prompt'] for r in train]
print(f'corpus: {{len(corpus_texts)}} excerpts')

# Download precomputed SONAR embeddings (avoids fairseq2 / sonar-space install).
# These were generated locally on the project owner's venv by precompute_for_colab.py
# and shipped in the repo at notebooks/legal-rag/data_for_colab/.
import urllib.request
DATA_DIR = Path('data_for_colab'); DATA_DIR.mkdir(exist_ok=True)
files = [f'sonar_corpus_{{CORPUS_SIZE}}.npy',
         f'sonar_corpus_{{CORPUS_SIZE}}_meta.json',
         f'sonar_test_queries_{{N_TEST}}.npy']
for f in files:
    dst = DATA_DIR / f
    if dst.exists() and dst.stat().st_size > 0: continue
    url = '{REPO_RAW}/' + f
    print(f'  downloading {{f}} ...')
    urllib.request.urlretrieve(url, dst)
print('embeddings ready')

sent_vec = np.load(DATA_DIR / f'sonar_corpus_{{CORPUS_SIZE}}.npy')
sent_vec = sent_vec / np.linalg.norm(sent_vec, axis=1, keepdims=True)
meta = json.loads((DATA_DIR / f'sonar_corpus_{{CORPUS_SIZE}}_meta.json').read_text())
sentence_to_parent = meta['sentence_to_parent']
assert len(sentence_to_parent) == sent_vec.shape[0]

query_vec = np.load(DATA_DIR / f'sonar_test_queries_{{N_TEST}}.npy')
query_vec = query_vec / np.linalg.norm(query_vec, axis=1, keepdims=True)

def retrieve(qi, q_text, k=K_RETRIEVE):
    sims = (query_vec[qi:qi+1] @ sent_vec.T)[0]
    order = np.argsort(-sims)
    seen, result = set(), []
    for i in order:
        parent = sentence_to_parent[int(i)]
        if parent not in seen:
            seen.add(parent); result.append(corpus_texts[parent])
            if len(result) == k: break
    return result

print(f'sonar retriever ready (cached embeddings, {{sent_vec.shape[0]}} sentence vectors)')"""

PROMPT_CODE = """SYSTEM_PROMPT = 'You are a legal expert. Read the case excerpt and respond with exactly one letter (A, B, C, D, or E). No explanation. No prose. Just the letter.'

def build_prompt(row, retrieved):
    ctx = ''
    if retrieved:
        ctx = 'Similar cases:\\n' + '\\n\\n'.join(f'{i+1}. {d}' for i, d in enumerate(retrieved)) + '\\n\\n'
    opts = '\\n'.join(f'{l}. {row[f"holding_{i}"]}' for i, l in enumerate('ABCDE'))
    return (f'{ctx}Excerpt: {row["citing_prompt"]}\\n\\nOptions:\\n{opts}\\n\\n'
            f'Answer with a single letter (A, B, C, D, or E).')

def ask_model(prompt, model=MODEL):
    t0 = time.time()
    r = requests.post(OLLAMA_URL, json={
        'model': model, 'stream': False,
        'messages': [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user',   'content': prompt},
        ],
        'options': {'temperature': 0.0, 'seed': SEED},
    }, timeout=120).json()
    return r['message']['content'].strip(), round(time.time()-t0, 2)

def parse_letter(text):
    text = text.strip().upper()
    if text and text[0] in 'ABCDE': return 'ABCDE'.index(text[0])
    m = re.search(r'(?:ANSWER|FINAL|CORRECT)\\b[^A-E]{0,20}\\b([A-E])\\b', text)
    if m: return 'ABCDE'.index(m.group(1))
    m = re.findall(r'\\b([A-E])\\b', text)
    return 'ABCDE'.index(m[-1]) if m else -1

print('prompt helpers ready')"""

EVAL_CODE = """from tqdm.auto import tqdm
records = []
for qi, row in enumerate(tqdm(test, desc='questions')):
    q = row['citing_prompt']
    correct = int(row['label'])
    retrieved = retrieve(qi, q)
    ans, lat = ask_model(build_prompt(row, retrieved))
    pred = parse_letter(ans)
    records.append({
        'question_id': qi, 'excerpt': q,
        'options': {l: row[f'holding_{i}'] for i, l in enumerate('ABCDE')},
        'correct': 'ABCDE'[correct],
        'retriever': RETRIEVER,
        'retrieved': retrieved,
        'answer_raw': ans,
        'predicted': 'ABCDE'[pred] if pred >= 0 else None,
        'correct_pred': pred == correct,
        'latency_s': lat,
    })
print(f'done: {len(records)} predictions')"""

RESULTS_CODE = """acc = sum(r['correct_pred'] for r in records) / len(records)
print(f'accuracy ({{RETRIEVER}}): {{acc:.3f}}  ({{sum(r["correct_pred"] for r in records)}}/{{len(records)}})')

from collections import Counter
pred_dist = Counter(r['predicted'] for r in records)
print(f'prediction distribution: {{dict(pred_dist)}}')

lat_mean = np.mean([r['latency_s'] for r in records])
print(f'mean latency: {{lat_mean:.1f}}s per question')

# Bar of predicted-letter distribution
fig, ax = plt.subplots(figsize=(6, 3.5))
letters = list('ABCDE')
ax.bar(letters, [pred_dist.get(L, 0) for L in letters], color='steelblue')
ax.set_title(f'{{MODEL}} + {{RETRIEVER}}  (acc={{acc:.2f}}, N={{N_TEST}})')
ax.set_ylabel('count')
ax.set_xlabel('predicted letter')
plt.tight_layout()
plt.show()

# Save
fig.savefig(RESULTS_DIR / f'{{RESULTS_TAG}}_predictions.png', dpi=150, bbox_inches='tight')
Path(RESULTS_DIR / f'{{RESULTS_TAG}}_results.json').write_text(json.dumps(records, indent=2))
print(f'saved {{RESULTS_DIR}}/{{RESULTS_TAG}}_*')"""

LOGPROB_COMPARISON_CODE = """# Optional: side-by-side with the Colab logprob baseline if it has been
# downloaded into results/logprob_results.json
logprob_path = RESULTS_DIR / 'logprob_results.json'
if not logprob_path.exists():
    print('no logprob_results.json found in results/ — skip')
    print('to populate it, run colab/colab_logprob_scoring.ipynb and drop the')
    print('downloaded file into this folder')
else:
    payload = json.loads(logprob_path.read_text())
    lp_records = payload['records']
    lp_my_retriever = [
        x for rec in lp_records for x in rec['results']
        if x['retriever'] == RETRIEVER
    ]
    if not lp_my_retriever:
        print(f'logprob_results.json has no records for retriever={{RETRIEVER}}')
    else:
        acc_raw  = sum(x['correct_raw']  for x in lp_my_retriever) / len(lp_my_retriever)
        acc_norm = sum(x['correct_norm'] for x in lp_my_retriever) / len(lp_my_retriever)
        print(f'\\n=== eval method comparison ({{RETRIEVER}}) ===')
        print(f'letter-emit (this notebook):           acc = {{acc:.3f}}')
        print(f'logprob raw       (Colab baseline):    acc = {{acc_raw:.3f}}')
        print(f'logprob len-norm  (Colab baseline):    acc = {{acc_norm:.3f}}')
        print('\\nletter-emit numbers above reflect the A-bias documented in README.')
        print('logprob numbers are the authoritative eval; see Zheng 2021.')"""


# ---------- builder ----------

def build(retriever: str, title: str, retriever_label: str, retriever_desc: str,
          retriever_code: str) -> nbformat.NotebookNode:
    cells = [
        new_markdown_cell(header(title, retriever_label, retriever_desc)),
        new_markdown_cell("## 1. Install Python deps"),
        new_code_cell(SETUP_CODE),
        new_markdown_cell("## 2. Verify Ollama + saul-7b model"),
        new_code_cell(OLLAMA_CHECK_CODE),
        new_markdown_cell("## 3. Config + imports"),
        new_code_cell(CONFIG_CODE_TEMPLATE.format(retriever=retriever)),
        new_markdown_cell("## 4. Load CaseHOLD"),
        new_code_cell(DATASET_CODE),
        new_markdown_cell(f"## 5. Retriever ({retriever_label})"),
        new_code_cell(retriever_code),
        new_markdown_cell("## 6. Prompt + scoring helpers"),
        new_code_cell(PROMPT_CODE),
        new_markdown_cell("## 7. Eval loop"),
        new_code_cell(EVAL_CODE),
        new_markdown_cell("## 8. Results"),
        new_code_cell(RESULTS_CODE),
        new_markdown_cell("## 9. Compare with logprob baseline (optional)"),
        new_code_cell(LOGPROB_COMPARISON_CODE),
    ]
    nb = new_notebook(cells=cells)
    nb['metadata'] = {
        'kernelspec': {'display_name': 'legal-rag', 'language': 'python', 'name': 'legal-rag'},
        'language_info': {'name': 'python'},
    }
    return nb


configs = [
    ("bare",  "saul_base.ipynb",  "Saul-7B-Instruct — no retrieval (bare baseline)",
     "bare baseline (no retrieved context)",
     "No retrieved context is added. The model answers from its own parametric knowledge only.",
     RETRIEVER_BARE_CODE),
    ("bm25",  "saul_bm25.ipynb",  "Saul-7B-Instruct + BM25 retrieval",
     "BM25 keyword retrieval over the CaseHOLD training corpus",
     "Each query retrieves the top-5 most lexically similar excerpts from the corpus (rank_bm25, lowercase whitespace tokenization). The retrieved excerpts are prepended to the prompt as 'Similar cases:'.",
     RETRIEVER_BM25_CODE),
    ("sonar", "saul_sonar.ipynb", "Saul-7B-Instruct + SONAR (concept-level) retrieval",
     "SONAR sentence-embedding retrieval over the CaseHOLD training corpus",
     "Corpus excerpts are pre-segmented into sentences and SONAR-encoded (Meta's multilingual concept embedding). Each query is encoded too. Cosine top-K, deduped to parent excerpts. Uses precomputed embeddings from the repo to avoid the fairseq2/sonar-space install (see README).",
     RETRIEVER_SONAR_CODE),
]

for retriever, fname, title, ret_label, ret_desc, ret_code in configs:
    nb = build(retriever, title, ret_label, ret_desc, ret_code)
    out = OUT_DIR / fname
    nbformat.write(nb, out)
    print(f'wrote {out}')
