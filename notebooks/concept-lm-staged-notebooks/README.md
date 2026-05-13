# Concept-Level Language Model Chatbot

A file-aware transformer that reasons over dynamic semantic concept tokens instead of word tokens.

**Research thesis:** Fixed tokenization boundaries are a design choice, not a law of nature. This system learns where concepts begin and end from the data itself, compresses them into a reasoning space, and generates responses that are semantically grounded at a level above the surface token.

---

## Architecture

```
Input Files (PDF, .py, .md, .docx, .ipynb)
        │
        ▼
[Stage 1] File Parser          → raw text segments per domain
        │
        ▼
[Stage 2] Token Encoder (Mamba-2 / Transformer)
        │  H = E(x)  — contextual token representations
        ▼
[Stage 3] Dynamic Boundary Detector
        │  p_t = (1 - cos(q_{t-1}, k_t)) / 2
        │  Bernoulli sampling (train) / hard threshold (inference)
        │  Global Load Balancing @ target ratio R
        ▼
  Concept Pooling: c_k = mean(h_{t in S_k}) @ W_up
        │
        ▼
[Stage 4] Concept Transformer  M(C) = Z
        │  Deep reasoning on compressed concept sequence
        ▼
[Stage 5] Cross-Attention Decoder
        │  Token t attends to concepts C_1..C_{j(t)}
        ▼
Token Predictions
```

---

## Notebooks (in order)

| # | Notebook | Purpose |
|---|----------|---------|
| 01 | `tokenization_baselines.ipynb` | BPE/WordPiece entropy analysis; subword failure modes |
| 02 | `sonar_embedding_exploration.ipynb` | SONAR encode/decode; fragility analysis; cosine distributions |
| 03 | `boundary_detection.ipynb` | Cosine dissimilarity scoring; learned vs. rule-based comparison |
| 04 | `concept_pooling_compression.ipynb` | Mean pooling; compression ratio R experiments; Global Load Balancing |
| 05 | `scaling_laws_analysis.ipynb` | DLCM compression-aware scaling law L(N,D,R,P); FLOP analysis |
| 06 | `file_parsing_pipeline.ipynb` | Multi-format file ingestion; domain-aware segmentation |
| 07 | `concept_lm_training.ipynb` | End-to-end training loop with joint NTP + concept + boundary losses |
| 08 | `concept_chatbot_inference.ipynb` | Interactive chatbot demo with file upload |

---

## Setup

```bash
pip install torch transformers sonar-space mamba-ssm pymupdf python-docx nbformat sentencepiece
pip install jupyter notebook
```

For SONAR:
```bash
pip install sonar-space
```

For Mamba (requires CUDA):
```bash
pip install mamba-ssm causal-conv1d
```

---

## Key Papers

1. [LCM (Meta FAIR, 2024)](https://arxiv.org/abs/2412.08821)
2. [DLCM (ByteDance, 2025)](https://arxiv.org/abs/2512.24617)
3. [H-Net (CMU/Cartesia, 2025)](https://arxiv.org/abs/2507.07955)
4. [BLT (Meta FAIR, 2024)](https://arxiv.org/abs/2412.09871)
5. [SONAR (Meta FAIR, 2023)](https://arxiv.org/abs/2308.11466)
6. [COCONUT (Meta FAIR, 2024)](https://arxiv.org/abs/2412.06769)
7. [CoCoMix (Meta FAIR, 2025)](https://arxiv.org/abs/2502.08524)
8. [Mamba (CMU, 2023)](https://arxiv.org/abs/2312.00752)
9. [Attention Is All You Need (Google, 2017)](https://arxiv.org/abs/1706.03762)
10. [Scaling Laws (OpenAI, 2020)](https://arxiv.org/abs/2001.08361)
11. [muP (Microsoft, 2022)](https://arxiv.org/abs/2203.03466)
12. [Sparse Autoencoders (OpenAI, 2024)](https://arxiv.org/abs/2406.04093)
13. [SONAR-LLM (2025)](https://arxiv.org/abs/2508.05305)
14. [CAFT (2025)](https://arxiv.org/abs/2506.07833)
15. [Tokenization Survey (2021)](https://arxiv.org/abs/2112.10508)

---

## License

MIT
