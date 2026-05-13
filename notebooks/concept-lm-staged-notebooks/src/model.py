"""
concept_lm/model.py

Core architecture for the file-aware concept language model.

Architecture:
    x (tokens) -> Encoder E -> H (token reps)
               -> BoundaryDetector -> segments S
               -> ConceptPooler -> C (concept reps)
               -> ConceptTransformer M -> Z (reasoned concepts)
               -> CrossAttentionDecoder D -> logits

Reference: DLCM (arXiv:2512.24617), H-Net (arXiv:2507.07955),
           LCM (arXiv:2412.08821)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional


@dataclass
class ConceptLMConfig:
    # Token-level encoder/decoder dimensions
    d_token: int = 512
    n_token_layers: int = 4
    n_token_heads: int = 8

    # Concept-level backbone dimensions
    d_concept: int = 1024
    n_concept_layers: int = 12
    n_concept_heads: int = 16

    # Boundary detection
    d_scan: int = 256          # query-key projection dim for boundary scoring
    boundary_threshold: float = 0.5   # hard threshold at inference
    train_temperature: float = 0.1    # sharpening temperature for Bernoulli sampling

    # Compression
    target_ratio: int = 4      # R: average tokens per concept
    aux_loss_weight: float = 0.01

    # Vocabulary
    vocab_size: int = 50257    # GPT-2 default; swap for your tokenizer
    max_seq_len: int = 4096

    # Cross-attention decoder
    d_head: int = 64           # head dim for cross-attention


class TokenEncoder(nn.Module):
    """
    Lightweight causal Transformer encoder.
    Produces contextual token representations H from input token ids.

    For production, swap nn.TransformerEncoder with Mamba-2 layers
    following H-Net's validated design (SSMs have stronger compression bias).
    """
    def __init__(self, cfg: ConceptLMConfig):
        super().__init__()
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_token)
        self.pos_embed = nn.Embedding(cfg.max_seq_len, cfg.d_token)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_token,
            nhead=cfg.n_token_heads,
            dim_feedforward=cfg.d_token * 4,
            dropout=0.0,
            batch_first=True,
            norm_first=True,  # pre-norm for stability
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=cfg.n_token_layers)
        self.norm = nn.LayerNorm(cfg.d_token)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: (B, L) integer token ids
        Returns:
            H: (B, L, d_token) contextual representations
        """
        B, L = input_ids.shape
        positions = torch.arange(L, device=input_ids.device).unsqueeze(0)
        x = self.embed(input_ids) + self.pos_embed(positions)

        # Causal mask
        mask = nn.Transformer.generate_square_subsequent_mask(L, device=input_ids.device)
        H = self.transformer(x, mask=mask, is_causal=True)
        return self.norm(H)


class BoundaryDetector(nn.Module):
    """
    Learns semantic boundary probabilities from token representations.

    Cosine dissimilarity between adjacent positions in a projected space:
        p_t = (1 - cos(q_{t-1}, k_t)) / 2

    Training: sharpen with temperature, sample Bernoulli(p_t^sharp)
    Inference: hard threshold p_t >= boundary_threshold

    Reference: DLCM Section 3.3.1 (arXiv:2512.24617), Equations 5-6
    """
    def __init__(self, cfg: ConceptLMConfig):
        super().__init__()
        self.Wq = nn.Linear(cfg.d_token, cfg.d_scan, bias=False)
        self.Wk = nn.Linear(cfg.d_token, cfg.d_scan, bias=False)
        self.threshold = cfg.boundary_threshold
        self.temperature = cfg.train_temperature

    def forward(self, H: torch.Tensor, training: bool = True):
        """
        Args:
            H: (B, L, d_token) token representations
            training: if True, sample boundaries; if False, apply hard threshold
        Returns:
            b: (B, L) binary boundary indicators (1 = new concept starts here)
            p: (B, L) boundary probabilities (for loss computation)
        """
        B, L, _ = H.shape
        Q = F.normalize(self.Wq(H), dim=-1)   # (B, L, d_scan)
        K = F.normalize(self.Wk(H), dim=-1)   # (B, L, d_scan)

        # Cosine dissimilarity between adjacent positions
        cos_sim = (Q[:, :-1] * K[:, 1:]).sum(-1)          # (B, L-1)
        p_inner = (1.0 - cos_sim) / 2.0                   # in [0, 1]

        # First token always starts a new concept (p_1 = 1)
        p_first = torch.ones(B, 1, device=H.device)
        p = torch.cat([p_first, p_inner], dim=1)          # (B, L)

        if training:
            # Sharpen: p_sharp = sigmoid((p - 0.5) / tau)
            p_sharp = torch.sigmoid((p - 0.5) / self.temperature)
            b = torch.bernoulli(p_sharp).detach()
        else:
            b = (p >= self.threshold).float()

        return b, p


class GlobalLoadBalancer(nn.Module):
    """
    Auxiliary loss that guides the global compression rate toward target ratio R.

    Computes expected (G) and actual (F) boundary rates across the batch,
    then penalizes deviation from the target rate 1/R.

    Reference: DLCM Section 3.3.3, Equation 10 (arXiv:2512.24617)
    Note: In distributed training, G and F require AllReduce across ranks.
    """
    def __init__(self, target_ratio: int):
        super().__init__()
        self.R = target_ratio

    def forward(self, b: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        """
        Args:
            b: (B, L) binary boundary indicators
            p: (B, L) boundary probabilities
        Returns:
            aux_loss: scalar auxiliary loss
        """
        G = p.mean()   # expected boundary rate
        F = b.mean()   # actual boundary rate

        R = self.R
        aux_loss = (R / (R - 1)) * ((R - 1) * F * G + (1 - F) * (1 - G)) - 1.0
        return aux_loss


class ConceptPooler(nn.Module):
    """
    Segments token representations into concept vectors via mean pooling + projection.

    Given boundary indicators b, partitions H into segments and compresses each
    segment into a single concept vector.

    Reference: DLCM Section 3.3.2, Equation 7 (arXiv:2512.24617)
    """
    def __init__(self, cfg: ConceptLMConfig):
        super().__init__()
        self.W_up = nn.Linear(cfg.d_token, cfg.d_concept, bias=False)

    def forward(self, H: torch.Tensor, b: torch.Tensor):
        """
        Args:
            H: (B, L, d_token)
            b: (B, L) binary, 1 where a new concept starts
        Returns:
            C: list of tensors, one per batch item, each (M_i, d_concept)
            segment_maps: list of (L,) tensors mapping each token to concept index
        """
        B, L, _ = H.shape
        C_list, seg_maps = [], []

        for i in range(B):
            boundary_positions = b[i].nonzero(as_tuple=True)[0]
            if len(boundary_positions) == 0:
                boundary_positions = torch.tensor([0], device=H.device)

            starts = boundary_positions.tolist()
            ends = starts[1:] + [L]

            seg_map = torch.zeros(L, dtype=torch.long, device=H.device)
            concepts = []
            for k, (s, e) in enumerate(zip(starts, ends)):
                seg_map[s:e] = k
                c_raw = H[i, s:e].mean(0)              # mean pool
                concepts.append(c_raw)

            C_i = torch.stack(concepts, dim=0)          # (M_i, d_token)
            C_i = self.W_up(C_i)                        # (M_i, d_concept)
            C_list.append(C_i)
            seg_maps.append(seg_map)

        return C_list, seg_maps


class ConceptTransformer(nn.Module):
    """
    High-capacity causal Transformer that performs deep reasoning on concept sequences.

    This is the primary site of compute in the system. Operating on compressed
    sequences of length M << L allows allocating far more parameters here than
    in the token encoder/decoder.

    Reference: DLCM Section 3.4 (arXiv:2512.24617)
    """
    def __init__(self, cfg: ConceptLMConfig):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_concept,
            nhead=cfg.n_concept_heads,
            dim_feedforward=cfg.d_concept * 4,
            dropout=0.0,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=cfg.n_concept_layers)
        self.norm = nn.LayerNorm(cfg.d_concept)

    def forward(self, C: torch.Tensor) -> torch.Tensor:
        """
        Args:
            C: (M, d_concept) concept sequence for one example, or (B, M, d_concept) batched
        Returns:
            Z: same shape as C, enriched concept representations
        """
        if C.dim() == 2:
            C = C.unsqueeze(0)
            M = C.shape[1]
            mask = nn.Transformer.generate_square_subsequent_mask(M, device=C.device)
            Z = self.transformer(C, mask=mask, is_causal=True)
            return self.norm(Z).squeeze(0)
        else:
            M = C.shape[1]
            mask = nn.Transformer.generate_square_subsequent_mask(M, device=C.device)
            Z = self.transformer(C, mask=mask, is_causal=True)
            return self.norm(Z)


class ConceptSmoother(nn.Module):
    """
    Lightweight smoothing of concept representations to reduce discretization artifacts.
    Integrates adjacent concepts before cross-attention decoding.

    Reference: DLCM Section 3.5.1 (arXiv:2512.24617)
    """
    def __init__(self, cfg: ConceptLMConfig):
        super().__init__()
        self.conv = nn.Conv1d(cfg.d_concept, cfg.d_concept, kernel_size=3, padding=1, groups=cfg.d_concept)
        self.norm = nn.LayerNorm(cfg.d_concept)

    def forward(self, Z: torch.Tensor) -> torch.Tensor:
        # Z: (M, d_concept)
        Z = Z.unsqueeze(0).transpose(1, 2)     # (1, d_concept, M)
        Z = self.conv(Z).transpose(1, 2).squeeze(0)  # (M, d_concept)
        return self.norm(Z)


class CrossAttentionDecoder(nn.Module):
    """
    Reconstructs token-level predictions by attending to reasoned concept vectors.

    Token t can only attend to concepts formed from positions <= t (causal).
    Handles heterogeneous dimensions: Q from d_token, K/V from d_concept.

    Reference: DLCM Section 3.5.2, Equations 12-14 (arXiv:2512.24617)
    """
    def __init__(self, cfg: ConceptLMConfig):
        super().__init__()
        self.n_heads = cfg.n_token_heads
        self.d_head = cfg.d_head
        self.WQ = nn.Linear(cfg.d_token, self.n_heads * self.d_head, bias=False)
        self.WK = nn.Linear(cfg.d_concept, self.n_heads * self.d_head, bias=False)
        self.WV = nn.Linear(cfg.d_concept, self.n_heads * self.d_head, bias=False)
        self.WO = nn.Linear(self.n_heads * self.d_head, cfg.d_token, bias=False)
        self.norm = nn.LayerNorm(cfg.d_token)
        self.lm_head = nn.Linear(cfg.d_token, cfg.vocab_size, bias=False)

    def forward(self, H: torch.Tensor, Z_smoothed: torch.Tensor, seg_map: torch.Tensor) -> torch.Tensor:
        """
        Args:
            H: (L, d_token) encoder hidden states (residual)
            Z_smoothed: (M, d_concept) smoothed concept representations
            seg_map: (L,) mapping each token position to its concept index
        Returns:
            logits: (L, vocab_size)
        """
        L, _ = H.shape
        M, _ = Z_smoothed.shape

        Q = self.WQ(H).view(L, self.n_heads, self.d_head)           # (L, h, d_head)
        K = self.WK(Z_smoothed).view(M, self.n_heads, self.d_head)  # (M, h, d_head)
        V = self.WV(Z_smoothed).view(M, self.n_heads, self.d_head)  # (M, h, d_head)

        # Build causal mask: token t can attend to concept k only if k <= seg_map[t]
        concept_indices = torch.arange(M, device=H.device).unsqueeze(0)  # (1, M)
        token_concept = seg_map.unsqueeze(1)                               # (L, 1)
        causal_mask = (concept_indices > token_concept).float() * -1e9    # (L, M)

        # Multi-head attention
        scale = math.sqrt(self.d_head)
        # Q: (L, h, dh) -> (h, L, dh); K: (M, h, dh) -> (h, dh, M)
        Q = Q.transpose(0, 1)           # (h, L, dh)
        K = K.permute(1, 2, 0)         # (h, dh, M)
        V = V.transpose(0, 1)          # (h, M, dh)

        attn = torch.bmm(Q, K) / scale                                # (h, L, M)
        attn = attn + causal_mask.unsqueeze(0)                        # broadcast over heads
        attn = F.softmax(attn, dim=-1)

        out = torch.bmm(attn, V)                                      # (h, L, dh)
        out = out.transpose(0, 1).contiguous().view(L, -1)            # (L, h*dh)
        out = self.WO(out)                                            # (L, d_token)

        # Residual connection + norm
        out = self.norm(H + out)
        logits = self.lm_head(out)                                    # (L, vocab_size)
        return logits


class ConceptLM(nn.Module):
    """
    Full concept-level language model for file-aware chat.

    Takes token ids as input, produces next-token logits as output.
    Internally performs dynamic concept segmentation and concept-space reasoning.
    """
    def __init__(self, cfg: ConceptLMConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = TokenEncoder(cfg)
        self.boundary = BoundaryDetector(cfg)
        self.load_balancer = GlobalLoadBalancer(cfg.target_ratio)
        self.pooler = ConceptPooler(cfg)
        self.concept_transformer = ConceptTransformer(cfg)
        self.smoother = ConceptSmoother(cfg)
        self.decoder = CrossAttentionDecoder(cfg)

    def forward(self, input_ids: torch.Tensor, labels: Optional[torch.Tensor] = None):
        """
        Args:
            input_ids: (B, L)
            labels: (B, L) optional, for computing NTP loss
        Returns:
            logits: (B, L, vocab_size)
            loss: optional scalar if labels provided
            aux_loss: boundary load balancing loss
        """
        B, L = input_ids.shape

        # Stage 1: Token encoding
        H = self.encoder(input_ids)                # (B, L, d_token)

        # Stage 2: Boundary detection
        training = self.training
        b, p = self.boundary(H, training=training)

        # Stage 3: Load balancing auxiliary loss
        aux_loss = self.load_balancer(b, p) * self.cfg.aux_loss_weight

        # Stage 4: Concept pooling and reasoning (process each batch item)
        C_list, seg_maps = self.pooler(H, b)

        all_logits = []
        for i in range(B):
            C_i = C_list[i]                        # (M_i, d_concept)
            seg_map_i = seg_maps[i]                # (L,)
            H_i = H[i]                             # (L, d_token)

            Z_i = self.concept_transformer(C_i)    # (M_i, d_concept)
            Z_smooth_i = self.smoother(Z_i)        # (M_i, d_concept)
            logits_i = self.decoder(H_i, Z_smooth_i, seg_map_i)  # (L, vocab)
            all_logits.append(logits_i)

        logits = torch.stack(all_logits, dim=0)    # (B, L, vocab)

        loss = None
        if labels is not None:
            # Shift: predict next token
            shift_logits = logits[:, :-1].contiguous().view(-1, self.cfg.vocab_size)
            shift_labels = labels[:, 1:].contiguous().view(-1)
            ntp_loss = F.cross_entropy(shift_logits, shift_labels, ignore_index=-100)
            loss = ntp_loss + aux_loss

        return logits, loss, aux_loss

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int = 256, temperature: float = 1.0):
        """
        Autoregressive generation.
        Processes the full context through the concept pipeline at each step.
        For efficiency, cache concept representations for prefill steps.
        """
        self.eval()
        generated = input_ids.clone()

        for _ in range(max_new_tokens):
            logits, _, _ = self.forward(generated)
            next_token_logits = logits[:, -1, :] / temperature
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token], dim=1)

            # Stop at EOS
            if (next_token == 50256).any():
                break

        return generated
