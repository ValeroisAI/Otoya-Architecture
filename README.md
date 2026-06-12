
# Failed — Otoya Architecture

**Status:** *Failed Project*  
**Reason:** The AI refused to learn.
---

## What Was This?

**Otoya** was an attempt to build a **zero‑dependency**, **self‑evolving** large language model training framework.  
No PyTorch, no Tinygrad, no CUDA – just NumPy, OpenCL, and a lot of hacker spirit.

The system consisted of:

- A **pure‑Python tensor library** (`QuantumTensor`) with an autograd engine
- A **Quantum‑Inspired Attention** layer (FFT‑based, mostly hype)
- A **Fractal Memory Network** (inspired by Z‑order curves, but entirely broken gradients)
- A **Multi‑Phase Training System** (learning rate changes, phase transitions)
- An **OpenCL backend** (never actually used because the CPU was faster)
- A **GUI** (the only part that worked reliably)

---

## Architecture Overview

```
otoya/
├── quantum_tensor.py      # Our own tensor library (3000+ lines of despair)
├── ultimate_otoya.py      # Main model definition (attention, memory, training)
├── opencl_backend.py      # GPU kernels (theoretical)
├── ultimate_gui.py        # The "Start" button that never stopped crashing
└── README.md              # This document
```

### Core components

**QuantumTensor**  
A NumPy‑backed tensor class with a homemade autograd engine. It supported basic ops, broadcasting, and a topological backward pass. It also included a `MemoryManager` that warned you when you were about to destroy your RAM – which you always ignored.

**QuantumAttentionLayer**  
A multi‑head attention layer that applied `cos()` to a few of its heads and claimed it was "quantum superposition". The gradients sometimes flowed, sometimes not.

**FractalMemoryLayer**  
A recurrent memory module that updated itself with momentum. The parameters were tracked by the optimizer but updated manually, which made autograd cry.

**AdaptiveTrainingSystem**  
A training loop that calculated cross‑entropy loss, clipped gradients, and adjusted the learning rate with cosine annealing. It occasionally produced NaN values just to keep things exciting.

---

## Why It Failed

 **Performance was a lie** – Benchmarks claimed 250k tokens/second. Real‑world test on a 50M parameter model: 0.25 tokens/second on a CPU because the GPU backend never materialised.
 **Documentation** – This README is the entire documentation.

---

## How to Run (if you dare)

```bash
pip install numpy
python ultimate_otoya.py --test   # might work, might segfault
```

---

**Project officially abandoned.**  
