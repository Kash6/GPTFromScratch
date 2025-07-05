# 🧠 GPT From Scratch – Transformer Experiments

This repository follows the *nanoGPT-style* implementation of a GPT model from scratch in PyTorch, with progressively advanced modifications across exercises. It’s based on the educational series by Andrej Karpathy and enhanced with state-of-the-art techniques in later stages.

---

## 📁 Exercise Summary

### ✅ `Ex1.py`: Vanilla GPT with Head and MultiHeadAttention combined into one class 

- **Goal**: Implement a basic GPT model trained on TinyShakespeare.
- **Features**:
  - Token + Positional Embeddings
  - Causal Masking with Triangular Matrix
  - Multi-Head Attention
  - FeedForward block
  - CrossEntropy Loss with teacher forcing

---

### ✅ `Ex2.py`: Addition GPT

- **Goal**: Train a GPT model to perform integer addition (e.g., `123+456=579`).
- **Modifications**:
  - Custom dataset generator for random digit-based addition problems.
  - Target sequence is the **reversed sum digits**, mimicking human addition.
  - Used `y = -1` masking to ignore loss on prompt (input) tokens.
  - Inference via `.generate()` + postprocessing (`[::-1]`).
- **Bug Fixes & Learnings**:
  - Fixed shape mismatch issues in position embeddings and masking.
  - Switched to deterministic sampling during generation.
  - Verified learning using integer prediction accuracy.

---

### ✅ `Ex3.py`:  Pretraining on Large Dataset and Finetuning on TinY Shakespeare

- Chose large dataset like openwebtext
- Tokenize the data using the same vocabulary/tokenizer as Shakespeare
- Train for many steps and save model checkpoint with:
```python
learning_rate = 3e-4
max_iters = 100_000 or more
eval_interval = 1000
torch.save(model.state_dict(), 'pretrained_gpt.pt')
```
- load thepretrained weights and finetune on tiny shakespeare
- Lower validation loss after pretraining 
---

### ✅ `Ex4.py`: Transformer Improvements

- **Goal**: Enhance the vanilla GPT with modern architectural improvements.
- **Implemented Features**:
  #### 1. Multi-Query Attention (MQA)
  - **What**: Shared key and value projections across all heads.
  - **Why**: Reduces memory usage and compute time without degrading performance.
  - **Where**: Replaced `MultiHeadAttention` with `MultiQueryAttention`.

  #### 2. SwiGLU Activation in FeedForward
  - **What**: Replaces ReLU with SwiGLU in the feed-forward layer.
  - **Why**: Empirically shown to improve training dynamics and downstream performance.
  - **Implementation**:
    ```python
    nn.Sequential(
        nn.Linear(n_embd, 2 * n_embd),
        nn.SiLU(),  # Swish
        nn.Linear(n_embd, n_embd),
        nn.Dropout(dropout)
    )
    ```



---

##  Ideas for Future Experiments

- Implement RoPE (Rotary Positional Embeddings)
- Use FlashAttention
- Add residual scaling (`rescale_layer`)
- Switch to GELU approximation (QuickGELU)
- Add LoRA for low-rank adaptation
- Integrate Chain-of-Thought tracing

---

##  Dependencies

- Python 3.8+
- PyTorch 2.x
- CUDA-enabled GPU (optional but recommended)

---

##  Acknowledgements

Inspired by [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT) and research from recent transformer architecture papers (PaLM, LLaMA, FlashAttention, etc.).

---

