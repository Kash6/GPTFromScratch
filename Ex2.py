import torch
import torch.nn as nn
from torch.nn import functional as F
import random

# Hyperparameters
DIGITS = 3
TRAIN_SIZE = 50000
VAL_SIZE = 5000
BLOCK_SIZE = DIGITS * 2 + 2 + 4  # a+b= and up to 4 digits of output
BATCH_SIZE = 64
EVAL_INTERVAL = 500
EVAL_ITERS = 200
MAX_ITERS = 2000
LEARNING_RATE = 3e-4
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
N_EMBD = 128
N_HEAD = 4
N_LAYER = 4
DROPOUT = 0.2

# Create vocabulary
chars = list("0123456789+=")
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for ch, i in stoi.items()}
vocab_size = len(stoi)

def encode(s):
    return [stoi[c] for c in s]

def decode(l):
    return ''.join([itos[i] for i in l])

# Generate data
def generate_data(size):
    data = []
    for _ in range(size):
        a = random.randint(0, 10**DIGITS - 1)
        b = random.randint(0, 10**DIGITS - 1)
        x = f"{a:0{DIGITS}d}+{b:0{DIGITS}d}="  # e.g., 123+456=
        y = str(a + b).rjust(DIGITS + 1, '0')  # pad left with zeros
        data.append(encode(x + y[::-1]))  # reversed target
    return data

train_data = generate_data(TRAIN_SIZE)
val_data = generate_data(VAL_SIZE)

def get_batch(split):
    data = train_data if split == 'train' else val_data
    batch = random.sample(data, BATCH_SIZE)
    x = torch.tensor([item[:-1] for item in batch], dtype=torch.long, device=DEVICE)
    y = torch.tensor([item[1:] for item in batch], dtype=torch.long, device=DEVICE)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(EVAL_ITERS)
        for k in range(EVAL_ITERS):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out

class MultiHeadAttention(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        assert n_embd % n_head == 0
        self.head_size = n_embd // n_head
        self.n_head = n_head
        self.key = nn.Linear(n_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(DROPOUT)
        self.register_buffer("tril", torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE)))

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x).view(B, T, self.n_head, self.head_size).transpose(1, 2)
        q = self.query(x).view(B, T, self.n_head, self.head_size).transpose(1, 2)
        v = self.value(x).view(B, T, self.n_head, self.head_size).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (self.head_size ** -0.5)
        att = att.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)
        out = att @ v
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(out)

class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(DROPOUT),
        )
    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.sa = MultiHeadAttention(n_embd, n_head)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class GPTLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, N_EMBD)
        self.position_embedding_table = nn.Embedding(BLOCK_SIZE, N_EMBD)
        self.blocks = nn.Sequential(*[Block(N_EMBD, N_HEAD) for _ in range(N_LAYER)])
        self.ln_f = nn.LayerNorm(N_EMBD)
        self.lm_head = nn.Linear(N_EMBD, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=DEVICE))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -BLOCK_SIZE:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            next_idx = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_idx), dim=1)
        return idx

model = GPTLanguageModel().to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

# Training
for iter in range(MAX_ITERS):
    if iter % EVAL_INTERVAL == 0 or iter == MAX_ITERS - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.6f}, val loss {losses['val']:.6f}")
    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Inference
def predict(a, b):
    expr = f"{a:03d}+{b:03d}="
    context = torch.tensor([encode(expr)], dtype=torch.long, device=DEVICE)
    out = model.generate(context, max_new_tokens=4)
    result = decode(out[0].tolist())
    generated = result[len(expr):]
    print("Generated output during prediction:", repr(generated))
    try:
        reversed_pred = generated[::-1].lstrip("0")
        pred_int = int(reversed_pred) if reversed_pred else 0
        correct = pred_int == a + b
        print(f"{a} + {b} = {pred_int} (correct? {correct})")
    except:
        print(f"{a} + {b} = ??? (conversion failed)")

predict(123, 456)
for _ in range(5):
    a = random.randint(0, 999)
    b = random.randint(0, 999)
    predict(a, b)
