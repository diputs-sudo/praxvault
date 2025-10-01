# Transformers

Transformers model sequences using **self-attention** to capture long-range dependencies without recurrence.  
They use **multi-head attention**, **positional encodings**, **layer normalization**, **residual connections**, and **feed-forward networks**, enabling efficient parallel training and strong performance in language, vision, and multimodal tasks.

---

## 1) Scaled Dot-Product Attention

Given queries `Q ∈ ℝ^{N×d}`, keys `K ∈ ℝ^{N×d}`, values `V ∈ ℝ^{N×d_v}`:

`Attention(Q, K, V) = softmax(Q Kᵀ / √d) V`

With mask `M` (additive, `-∞` on disallowed positions):  
`softmax((Q Kᵀ / √d) + M) V`

---

## 2) Multi-Head Attention

Split `d_model` into `h` heads of size `d_head = d_model / h`:

- Project inputs to `Q_i, K_i, V_i` for each head `i`.
- Compute attention per head; concatenate heads; apply output projection.

---

## 3) Positional Encodings

**Sinusoidal** (fixed):  
`PE[pos, 2i]   = sin(pos / 10000^{2i/d_model})`  
`PE[pos, 2i+1] = cos(pos / 10000^{2i/d_model})`

**Learned**: trainable embeddings by position index.

---

## 4) Encoder & Decoder (Seq2Seq)

- **Encoder layer**: Self-Attn → Add&Norm → FFN → Add&Norm  
- **Decoder layer**: Masked Self-Attn → Add&Norm → Cross-Attn → Add&Norm → FFN → Add&Norm

---

## 5) Minimal Components (PyTorch)

'''python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=10000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        # x: (B, T, d_model)
        T = x.size(1)
        return x + self.pe[:, :T]

def causal_mask(T):
    # returns (1, 1, T, T) with -inf above diagonal
    mask = torch.full((T, T), float("-inf"))
    mask = torch.triu(mask, diagonal=1)
    return mask.unsqueeze(0).unsqueeze(0)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.0):
        super().__init__()
        assert d_model % num_heads == 0
        self.h = num_heads
        self.d_head = d_model // num_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x_q, x_kv, mask=None):
        B, Tq, D = x_q.shape
        Tk = x_kv.shape[1]
        # projections
        Q = self.q_proj(x_q).view(B, Tq, self.h, self.d_head).transpose(1, 2)  # (B,h,Tq,dh)
        K = self.k_proj(x_kv).view(B, Tk, self.h, self.d_head).transpose(1, 2)  # (B,h,Tk,dh)
        V = self.v_proj(x_kv).view(B, Tk, self.h, self.d_head).transpose(1, 2)  # (B,h,Tk,dh)
        # attention scores
        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(self.d_head)  # (B,h,Tq,Tk)
        if mask is not None:
            scores = scores + mask  # broadcastable mask: (1,1,Tq,Tk)
        A = F.softmax(scores, dim=-1)
        A = self.drop(A)
        # aggregate
        H = A @ V  # (B,h,Tq,dh)
        H = H.transpose(1, 2).contiguous().view(B, Tq, D)
        return self.o_proj(H)

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
    def forward(self, x):
        return self.net(x)

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn  = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, src_mask=None):
        # Self-attention
        a = self.attn(self.norm1(x), self.norm1(x), mask=src_mask)
        x = x + self.drop(a)
        # FFN
        f = self.ffn(self.norm2(x))
        x = x + self.drop(f)
        return x

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn   = FeedForward(d_model, d_ff, dropout)
        self.n1 = nn.LayerNorm(d_model)
        self.n2 = nn.LayerNorm(d_model)
        self.n3 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, y, enc, tgt_mask=None, mem_mask=None):
        s = self.self_attn(self.n1(y), self.n1(y), mask=tgt_mask)
        y = y + self.drop(s)
        c = self.cross_attn(self.n2(y), enc, mask=mem_mask)
        y = y + self.drop(c)
        f = self.ffn(self.n3(y))
        y = y + self.drop(f)
        return y
'''

---

## 6) Encoder, Decoder, Full Transformer

'''python
class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, d_model=512, num_layers=6, num_heads=8, d_ff=2048, dropout=0.1, max_len=10000):
        super().__init__()
        self.tok = nn.Embedding(vocab_size, d_model)
        self.pos = PositionalEncoding(d_model, max_len=max_len)
        self.layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, src_ids, src_mask=None):
        x = self.pos(self.tok(src_ids))  # (B,T,d)
        for l in self.layers:
            x = l(x, src_mask=src_mask)
        return self.norm(x)

class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, d_model=512, num_layers=6, num_heads=8, d_ff=2048, dropout=0.1, max_len=10000):
        super().__init__()
        self.tok = nn.Embedding(vocab_size, d_model)
        self.pos = PositionalEncoding(d_model, max_len=max_len)
        self.layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, tgt_ids, enc_out, tgt_mask=None, mem_mask=None):
        y = self.pos(self.tok(tgt_ids))
        for l in self.layers:
            y = l(y, enc_out, tgt_mask=tgt_mask, mem_mask=mem_mask)
        y = self.norm(y)
        logits = self.lm_head(y)
        return logits

class TransformerSeq2Seq(nn.Module):
    def __init__(self, src_vocab, tgt_vocab, **kw):
        super().__init__()
        self.encoder = TransformerEncoder(src_vocab, **kw)
        self.decoder = TransformerDecoder(tgt_vocab, **kw)

    def forward(self, src_ids, tgt_ids, src_pad_idx=0, tgt_pad_idx=0):
        # build masks
        B, Ts = src_ids.size()
        B, Tt = tgt_ids.size()
        # padding masks to -inf on pad positions for attention logits
        src_mask = (src_ids != src_pad_idx).unsqueeze(1).unsqueeze(2)  # (B,1,1,Ts)
        tgt_mask_pad = (tgt_ids != tgt_pad_idx).unsqueeze(1).unsqueeze(2)  # (B,1,1,Tt)
        causal = causal_mask(Tt).to(src_ids.device)                       # (1,1,Tt,Tt)
        tgt_mask = torch.where(tgt_mask_pad, torch.zeros_like(causal), torch.full_like(causal, float("-inf"))) + causal
        src_mask = torch.where(src_mask, torch.zeros_like(src_mask, dtype=torch.float), torch.full_like(src_mask, float("-inf")))
        # encode-decode
        enc = self.encoder(src_ids, src_mask=src_mask)
        logits = self.decoder(tgt_ids, enc, tgt_mask=tgt_mask, mem_mask=None)
        return logits
'''

---

## 7) Language Modeling (Decoder-only, Causal)

'''python
class GPTLike(nn.Module):
    def __init__(self, vocab_size, d_model=512, num_layers=6, num_heads=8, d_ff=2048, max_len=4096, dropout=0.1):
        super().__init__()
        self.tok = nn.Embedding(vocab_size, d_model)
        self.pos = PositionalEncoding(d_model, max_len=max_len)
        self.blocks = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, ids, pad_idx=None):
        B, T = ids.shape
        y = self.pos(self.tok(ids))
        cmask = causal_mask(T).to(ids.device)  # (1,1,T,T)
        # optional pad mask: (B,1,1,T) with -inf at pads
        if pad_idx is not None:
            pad = (ids != pad_idx).unsqueeze(1).unsqueeze(2)
            pad = torch.where(pad, torch.zeros_like(pad, dtype=torch.float), torch.full_like(pad, float("-inf")))
            cmask = cmask + pad
        for blk in self.blocks:
            y = blk(y, enc=None, tgt_mask=cmask, mem_mask=None)  # DecoderLayer ignores cross-attn if enc=None
        y = self.norm(y)
        return self.lm_head(y)

    @torch.no_grad()
    def generate(self, ids, max_new_tokens=50, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            logits = self.forward(ids)[:, -1, :] / max(temperature, 1e-6)
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            ids = torch.cat([ids, next_id], dim=1)
        return ids
'''

---

## 8) Training Loop (LM example)

'''python
import torch
from torch.utils.data import Dataset, DataLoader

class ToyText(Dataset):
    def __init__(self, text_ids, block_size=128):
        self.ids = text_ids
        self.block = block_size
    def __len__(self): return max(1, len(self.ids) - self.block)
    def __getitem__(self, i):
        x = torch.tensor(self.ids[i:i+self.block], dtype=torch.long)
        y = torch.tensor(self.ids[i+1:i+self.block+1], dtype=torch.long)
        return x, y

# toy vocab and data
vocab_size = 100
ids = torch.randint(0, vocab_size, (10000,))
ds = ToyText(ids.tolist(), block_size=64)
dl = DataLoader(ds, batch_size=64, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GPTLike(vocab_size=vocab_size, d_model=256, num_layers=4, num_heads=4, d_ff=1024, max_len=1024, dropout=0.1).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=5)

for epoch in range(5):
    model.train()
    total_loss = 0.0
    for x, y in dl:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        total_loss += loss.item()
    sched.step()
    print(f"epoch {epoch+1} loss={total_loss/len(dl):.4f}")
'''

---

## 9) Attention Complexity & Tweaks

- **Complexity**: O(T²·d) for vanilla attention (T = sequence length).  
- **Memory/Speed tricks**: FlashAttention, chunking, gradient checkpointing, low-rank/linear attention (Performer, FAVOR+), local/sparse patterns (Longformer), sliding windows.  
- **Stabilization**: Pre-norm (LayerNorm before sublayers), residual scaling, RMSNorm, careful init, scaled RoPE/ALiBi positional strategies.

"""""js_plotly
{
  "data": [
    {"x": [128, 256, 512, 1024, 2048], "y": [1, 4, 16, 64, 256], "mode": "lines+markers", "name": "O(T^2)"},
    {"x": [128, 256, 512, 1024, 2048], "y": [1, 2, 4, 8, 16], "mode": "lines+markers", "name": "O(T)"}
  ],
  "layout": {"title": "Attention Complexity (Illustrative)", "xaxis": {"title": "Sequence Length T"}, "yaxis": {"title": "Relative Cost"}}
}
"""""

---

## 10) Masks & Padding

- **Causal mask**: prevent attending to future tokens (strictly upper-triangular `-∞`).  
- **Padding mask**: block attention to `<pad>` positions in variable-length batches.  
- Combine by **adding masks** to logits before softmax.

---

## 11) Classification Head (Encoder-only)

'''python
class TransformerForClassification(nn.Module):
    def __init__(self, vocab_size, num_labels, d_model=512, num_layers=6, num_heads=8, d_ff=2048, max_len=4096, dropout=0.1):
        super().__init__()
        self.encoder = TransformerEncoder(vocab_size, d_model, num_layers, num_heads, d_ff, dropout, max_len)
        self.cls = nn.Linear(d_model, num_labels)

    def forward(self, ids, pad_idx=0):
        mask = (ids != pad_idx).unsqueeze(1).unsqueeze(2)
        mask = torch.where(mask, torch.zeros_like(mask, dtype=torch.float), torch.full_like(mask, float("-inf")))
        h = self.encoder(ids, src_mask=mask)  # (B,T,d)
        h_cls = h[:, 0]  # assume first token is [CLS]
        return self.cls(h_cls)
'''

---

## 12) Practical Notes

- Tokenize consistently; manage special tokens (`<pad>`, `<bos>`, `<eos>`, `[CLS]`, `[SEP]`).  
- Use **AdamW**, cosine decay, warmup steps; gradient clipping.  
- Mixed precision & FlashAttention for throughput; scale batch with gradient accumulation.  
- Regularize with dropout, label smoothing; monitor perplexity/accuracy and validation loss.  
- Save/checkpoint with EMA for stable evaluation.

---

## 13) Troubleshooting

- **Diverging loss** → lower LR, increase warmup, check tokenization/padding masks.  
- **Instability at long T** → pre-norm variants, clip grads, reduce batch, use FlashAttention/rope-scaling.  
- **Overfitting** → more data, dropout, label smoothing, weight decay.  
- **Slow throughput** → fused kernels, AMP, sequence bucketing, packed samples.

---

## 14) Summary

Transformers replace recurrence with **self-attention**, enabling parallelism and strong global context modeling.  
With multi-head attention, positional encodings, and deep stacks of pre-norm residual blocks, they scale effectively across language, vision, and multimodal tasks.  
Variants (encoder-only, decoder-only, encoder–decoder) cover classification, generation, and sequence transduction.
