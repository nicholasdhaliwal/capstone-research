# capstone-research

UChicago MS Applied Data Science capstone thesis on legal-domain large-language-model reasoning. Showcase target: **2026-08-15**.

## Layout

| Path | Contents |
|---|---|
| `paper/` | Paper drafts (latest + prior versions) |
| `notebooks/` | Working Jupyter notebooks |
| `notebooks/legal-rag/` | Main paper artifact: SaulLM + retrieval-augmentation study on CaseHOLD |
| `notebooks/saul-7b/` | Local SaulLM-7B-Instruct chat demo + 8 failure-mode probes |
| `notebooks/concept-lm-staged-notebooks/` | 8 publication-ready notebooks from earlier concept-LM work |
| `code/` | Standalone Python scripts (e.g. SONAR corpus pre-encoder) |
| `data/` | Small tracked datasets (heavy data is gitignored) |
| `outputs/` | Generated figures, tables, result JSONs for the paper |
| `references/` | Cited papers (PDFs) + external repo clones (gitignored) |
| `notes/` | Research notes, lit synthesis, Saul probe transcripts |
| `team/` | Team admin, meeting notes, presentations |
| `docs/` | HANDOFF, design notes, decision records |

Live `concept-lm` working tree (preserved as future work): `~/Documents/GitHub/concept-lm/` (separate repo, github.com/nicholasdhaliwal/concept-lm).

KB project pages: `~/Documents/GitHub/knowledge-base/wiki/projects/lcm-capstone/`.

Earlier solo iteration archived locally at `~/Documents/marshall-lcm-archive/` (Dec 2024 – Mar 2025), outside this repo.

## Scope (active direction, 2026-05-13)

In: SaulLM-7B-Instruct + retrieval-augmentation ablation on CaseHOLD (bare / BM25 / SONAR). Reproducibility notebook + 8-page paper. No novel data collection; all from Hugging Face.

Out: training a concept-LM from scratch; Legal WordNet construction (future-work positioning).

See `docs/HANDOFF.md` and the KB project page for the full thesis framing.
