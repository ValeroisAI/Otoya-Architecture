# ============================================================
# eita_opencl_trainer.py — Eita V12.0.0 (350M Stable)
# ============================================================
"""
V12.0.0 — 350M+ model için kritik fix'ler:
  - BareAdamW.step(): Tüm ağırlık güncellemelerini küçük chunk'larda realize eder
    (tek büyük OpenCL kernel yerine). "Cannot create kernels" hatasını önler.
  - make_train_step(): loss/lm_loss realize edilir (gerekli adımda host'a çekilir) ÖNCE, sonra opt.step().
    Lazy graph'ın parametrelerle çakışmasını engeller (loss=0.0 hatası).
  - RoPE precomputed position cache ile her step'te dinamik Tensor.arange engellendi.
  - UltraWNN/LightweightWNN'de clip guard'lar eklendi (NaN/Inf koruması).
  - Eğitim döngüsünde step başı kernel cache temizleme stratejisi optimize edildi.
"""

from __future__ import annotations
import argparse, json, logging, math, os, queue, random, threading, time, traceback
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk
import sqlite3
sqlite3.sqlite_version_info  # sqlite3'ü tetikle
import os  # TAMAMEN AYNI SATIRDA OLMALIDIR

os.environ["SQLITE_CHECK_SAME_THREAD"] = "0"
os.environ["TINYGRAD_DISABLE_ID_VALIDATION"] = "1"
os.environ["TINY_CACHE"] = "1"
import sqlite3
original_connect = sqlite3.connect
def patched_connect(*args, **kwargs):
    kwargs['check_same_thread'] = False
    return original_connect(*args, **kwargs)
sqlite3.connect = patched_connect


from tinygrad import Tensor as _T, nn as _nn, Device
from tinygrad.nn.state import get_parameters as _gp, get_state_dict as _gs, safe_save as _ss
# JIT import'u buradan silindi. Aşağıda güvenli bir şekilde çağrılacak.

# GPU Optimizer import
try:
    from gpu_optimizer import get_optimizer, cleanup_gpu
    GPU_OPTIMIZER_AVAILABLE = True
except ImportError:
    GPU_OPTIMIZER_AVAILABLE = False

# Vulkan AI Accelerator import (delayed initialization after logger is defined)
VULKAN_AVAILABLE = False
accelerator_manager = None

# Hybrid Phase Shifter import
try:
    from phase_shifter_vulkan import HybridPhaseShifter
    HYBRID_PHASE_AVAILABLE = True
except ImportError:
    HYBRID_PHASE_AVAILABLE = False

try:
    from otoya_next.core.spectral_gate_mixer import SpectralGateMixer as NextSpectralGateMixer
    OTOYA_NEXT_PHASE_AVAILABLE = True
except ImportError:
    OTOYA_NEXT_PHASE_AVAILABLE = False

# ----------------------------------------------------------------------
# Environment & Logging
# ----------------------------------------------------------------------
LOG_PATH = Path("eita_opencl_error.log")
logger = logging.getLogger("EitaV11.3.3")
logger.setLevel(logging.INFO)
logger.propagate = False
if not logger.handlers:
    fh = logging.FileHandler(LOG_PATH, encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(fh)

gui_log_queue: "queue.Queue[Tuple[str, str]]" = queue.Queue(maxsize=500)

# ScalableJIT: blok‑bazlı, geriye dönük uyumlu JIT sarmalayıcısı.
# Mevcut TinyJit'e ek olarak büyük modellerde graph patlamasını engeller.
try:
    from scalable_jit import ScalableJIT as _ScalableJIT, discover_blocks as _discover_blocks
    SCALABLE_JIT_AVAILABLE = True
except Exception as _sjit_exc:  # noqa: BLE001
    _ScalableJIT = None
    SCALABLE_JIT_AVAILABLE = False
    logger.warning("ScalableJIT yüklenemedi: %s", _sjit_exc)

# Initialize Vulkan AI Accelerator after logger is defined
try:
    from ai_accelerator_direct import DirectAcceleratorManager
    VULKAN_AVAILABLE = True
    accelerator_manager = DirectAcceleratorManager()
    logger.info(f"AI Hızlandırıcı Durumu: {accelerator_manager.get_status()['best_method']['status']}")
except ImportError:
    VULKAN_AVAILABLE = False
    logger.warning("AI hızlandırıcı kontrolü bulunamadı")

class QueueLogHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        tag = "info"
        if record.levelno >= logging.ERROR:   tag = "error"
        elif record.levelno >= logging.WARNING: tag = "warning"
        elif record.levelno == logging.INFO:  tag = "milestone"
        try: gui_log_queue.put_nowait((self.format(record), tag))
        except queue.Full: pass

def sanitize_env(backend: str = "CL", cache_level: int = 3, fast_math: bool = False) -> None:
    os.environ["DEV"] = backend
    try:
        cache_level = max(0, min(3, int(cache_level)))
    except Exception:
        cache_level = 3
    os.environ["CACHELEVEL"] = str(cache_level)
    os.environ.pop("DISABLE_ASSIGN", None)
    if backend == "CL" and fast_math:
        os.environ["AMD_OCL_BUILD_OPTIONS_APPEND"] = "-cl-fast-relaxed-math -cl-mad-enable"
    else:
        os.environ.pop("AMD_OCL_BUILD_OPTIONS_APPEND", None)
    for key in ("DEBUG", "BEAM", "NOOPT"):
        try: int(os.environ.get(key, "0"))
        except ValueError: os.environ[key] = "0"

def probe_opencl() -> List[Dict[str, Any]]:
    try: import pyopencl as cl
    except Exception as exc:
        raise RuntimeError("pyopencl gerekli: python -m pip install pyopencl") from exc
    devices = []
    for platform in cl.get_platforms():
        for dev in platform.get_devices():
            devices.append({
                "platform": platform.name, 
                "vendor": platform.vendor,
                "device": dev.name, "type": cl.device_type.to_string(dev.type),
                "global_mem_gb": dev.global_mem_size / (1024**3),
            })
    if not devices: raise RuntimeError("OpenCL cihaz bulunamadı.")
    return devices

Tensor: Any = None
nn: Any = None
get_parameters: Any = None; get_state_dict: Any = None
safe_save: Any = None

def load_backend(backend: str = "CL", use_vulkan: bool = False, cache_level: int = 3, fast_math: bool = False) -> str:
    global Tensor, nn, get_parameters, get_state_dict, safe_save
    sanitize_env(backend, cache_level=cache_level, fast_math=fast_math)
    if backend == "CL": probe_opencl()
    
    # Vulkan backend for direct AI accelerator access
    if use_vulkan:
        try:
            import subprocess
            result = subprocess.run(['vulkaninfo', '--summary'], capture_output=True, text=True)
            if result.returncode == 0:
                logger.info("✅ Vulkan tespit edildi - AI hızlandırıcılar için doğrudan erişim aktif")
                # Note: Tinygrad doesn't have native Vulkan support yet, but we can optimize OpenCL
                # to use the same GPU drivers that Vulkan uses
            else:
                logger.warning("⚠️ Vulkan bulunamadı, OpenCL kullanılıyor")
        except:
            logger.warning("⚠️ Vulkan kontrolü başarısız, OpenCL kullanılıyor")
    
    try:
        from tinygrad import Tensor as _T, nn as _nn, Device
        from tinygrad.nn.state import get_parameters as _gp, get_state_dict as _gs, safe_save as _ss
    except Exception as exc:
        raise RuntimeError("tinygrad gerekli: python -m pip install tinygrad numpy pyopencl safetensors") from exc
    Tensor, nn = _T, _nn
    get_parameters, get_state_dict, safe_save = _gp, _gs, _ss
    return str(Device.DEFAULT)

def tensor_no_grad():
    return Tensor.no_grad() if hasattr(Tensor, "no_grad") else nullcontext()

def set_trainable(obj: Any, enabled: bool) -> None:
    if isinstance(obj, Tensor): obj.requires_grad = enabled
    elif isinstance(obj, dict):
        for v in obj.values(): set_trainable(v, enabled)
    elif isinstance(obj, (list, tuple)):
        for v in obj: set_trainable(v, enabled)
    elif hasattr(obj, "__dict__"):
        for v in vars(obj).values(): set_trainable(v, enabled)

# ======================================================================
# BareAdamW
# ======================================================================
# ============================================================
# BareAdamW — 350M+ uyumlu, chunk realize, sayısal kararlı
# ============================================================
class BareAdamW:
    """AdamW optimizer — 350M+ model için chunk realize ile OpenCL kernel patlamasını önler.
    lr ve step Python float olarak tutulur; step() içinde saf tensör operasyonları çalışır.
    """
    REALIZE_CHUNK = 32

    def __init__(self, params, lr=3e-4, betas=(0.9, 0.999), eps=1e-8, wd=0.05, grad_clip=0.5, warmup_steps=0, total_steps=100000):
        self.params = list(params) if not isinstance(params, list) else params
        if not self.params: raise ValueError("BareAdamW: parametre listesi boş.")
        self.b1, self.b2 = betas[0], betas[1]
        self.eps, self.wd, self.grad_clip = eps, wd, grad_clip
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.base_lr = lr
        self._lr = float(lr)
        self._step = 0

        self.b1_t = Tensor([1.0])
        self.b2_t = Tensor([1.0])
        self.lr_t = Tensor([float(self.base_lr)])
        Tensor.realize(self.b1_t, self.b2_t, self.lr_t)

        self.m = [Tensor.zeros(*p.shape) for p in self.params]
        self.v = [Tensor.zeros(*p.shape) for p in self.params]
        Tensor.realize(*self.m[:self.REALIZE_CHUNK])
        for i in range(self.REALIZE_CHUNK, len(self.m), self.REALIZE_CHUNK):
            Tensor.realize(*self.m[i:i+self.REALIZE_CHUNK])
        Tensor.realize(*self.v[:self.REALIZE_CHUNK])
        for i in range(self.REALIZE_CHUNK, len(self.v), self.REALIZE_CHUNK):
            Tensor.realize(*self.v[i:i+self.REALIZE_CHUNK])

    def set_lr(self, lr: float):
        """LR'yi Python float olarak güncelle — JIT dışından çağrılır."""
        self._lr = float(lr)

    def zero_grad(self):
        for p in self.params: p.grad = None

    def step(self):
        self._step += 1
        t = self._step

        active_params = [(i, p) for i, p in enumerate(self.params) if p.grad is not None]
        if not active_params:
            return

        b1_t_val = max(1e-8, 1.0 - (self.b1 ** t))
        b2_t_val = max(1e-8, 1.0 - (self.b2 ** t))

        self.b1_t.assign(Tensor([b1_t_val]))
        self.b2_t.assign(Tensor([b2_t_val]))
        self.lr_t.assign(Tensor([float(self._lr)]))
        Tensor.realize(self.b1_t, self.b2_t, self.lr_t)

        chunk_tensors = []
        for i, p in active_params:
            g = p.grad
            g = g.clip(-self.grad_clip * 2, self.grad_clip * 2)
            g = g.clip(-self.grad_clip, self.grad_clip)

            new_m = self.b1 * self.m[i].detach() + (1.0 - self.b1) * g
            new_v = self.b2 * self.v[i].detach() + (1.0 - self.b2) * (g * g)

            m_hat = new_m / self.b1_t
            v_hat = new_v / self.b2_t
            update = m_hat / (v_hat.maximum(1e-16).sqrt() + self.eps)
            if self.wd > 0:
                update = update + self.wd * p.detach()

            self.m[i].assign(new_m)
            self.v[i].assign(new_v)
            p.assign(p.detach() - self.lr_t * update)

            chunk_tensors.extend([p, self.m[i], self.v[i]])

            if len(chunk_tensors) >= self.REALIZE_CHUNK:
                Tensor.realize(*chunk_tensors)
                chunk_tensors = []

        if chunk_tensors:
            Tensor.realize(*chunk_tensors)

# ======================================================================
# Config
# ======================================================================
@dataclass
class EitaConfig:
    vocab_size: int = 258; d_model: int = 128
    num_layers: int = 2; num_heads: int = 4; phase_groups: int = 4
    context_length: int = 128; ssm_kernel: int = 33  # Back to 33 for quality
    phase_mode: str = "turbo"  # Yeni varsayılan: maksimum throughput odakli hizli mod
    attn_window: int = 32
    use_eft: bool = False; eft_freq_dim: int = 32

    # ---- MoE (Mixture of Experts) ----
    # 0 / False = devre dışı. Açıldığında her OtoyaBlock içine
    # bir MoE katmanı yerleştirilir; FFN'in yerini alır.
    use_moe: bool = False
    num_experts: int = 4
    top_k_experts: int = 2
    moe_d_ff: Optional[int] = None  # None ise 4*d_model
    moe_balance_loss_weight: float = 0.01
    moe_router_z_loss_weight: float = 0.001


SUPPORTED_PHASE_MODES = ("turbo", "light_wnn", "ultra_wnn", "ultra_hybrid", "srwm", "minimal")
LEGACY_PHASE_MODE_MAP = {
    "ultra_fast": "turbo",
    "fast": "turbo",
    "hybrid": "ultra_hybrid",
    "hybrid_fast": "ultra_hybrid",
    "opencl_kernel": "turbo",
    "complex": "srwm",
    "phase_shifter": "srwm",
}


def normalize_phase_mode(mode: str) -> str:
    phase_mode = (mode or "turbo").strip()
    phase_mode = LEGACY_PHASE_MODE_MAP.get(phase_mode, phase_mode)
    if phase_mode not in SUPPORTED_PHASE_MODES:
        phase_mode = "turbo"
    return phase_mode

def suggest_attn_window(context_length: int, phase_mode: str) -> int:
    context_length = max(8, int(context_length))
    phase_mode = normalize_phase_mode(phase_mode)
    if phase_mode in ("turbo", "ultra_wnn"):
        return min(context_length, 32 if context_length >= 64 else context_length)
    if phase_mode in ("srwm", "light_wnn"):
        return min(context_length, 64 if context_length >= 64 else context_length)
    return min(context_length, 32 if context_length >= 32 else context_length)

@dataclass
class TrainConfig:
    backend: str = "CL"; batch_size: int = 64  # CL tarafinda launch overhead'i daha iyi amorti eder
    total_steps: int = 500; lr: float = 3e-4; weight_decay: float = 0.05
    log_every: int = 10  # Her step'te host senkronu yapma; throughput icin daha seyrek logla
    seed: int = 1337; grad_accum: int = 1
    optimizer: str = "bare"; export_dir: str = "eita_opencl_export"
    # Performance optimizations
    use_vulkan: bool = False  # OpenCL JIT'i gereksiz yere kapatmasin
    cache_level: int = 3
    fast_math: bool = True
    force_jit: bool = False
    cleanup_every: int = 200  # GPU memory cleanup every N steps (reduced frequency)
    val_every: int = 500  # Validation every N steps (reduced from 100 to reduce overhead)
    checkpoint_every: int = 1000  # Checkpoint every N steps (new parameter)

def phase_for_step(step, total):
    if total < 20: return 1
    if step < int(total*0.30): return 1
    if step < int(total*0.70): return 2
    if step < int(total*0.90): return 3
    return 4

def phase_strength_for_step(step, total):
    if total < 20: return 0.0
    start = int(total * 0.30)
    if step < start: return 0.0
    return min(1.0, (step - start + 1) / max(80, int(total * 0.25)))

def reg_weights(phase):
    if phase == 1: return {"phase": 0.0, "orth": 0.0, "head": 0.0}
    if phase == 2: return {"phase": 5e-4, "orth": 3e-3, "head": 0.0}
    if phase == 3: return {"phase": 1e-3, "orth": 3e-3, "head": 1e-3}
    return {"phase": 1e-3, "orth": 3e-3, "head": 1e-3}

def estimate_memory_mb(cfg, batch):
    d, l, v = cfg.d_model, cfg.num_layers, cfg.vocab_size
    params = v*d + cfg.context_length*d
    per = (3*d*d + 3*d + 2*d*cfg.ssm_kernel + d*d + 4*d*d +
           4*cfg.num_heads*cfg.phase_groups*(d//cfg.num_heads//cfg.phase_groups) + 4*(2*d*d+d))
    params += l * per
    return params*4/(1024**2), batch*cfg.context_length*d*max(10, l*8)*4/(1024**2)

def estimate_params_m(cfg): return estimate_memory_mb(cfg, 1)[0] / 4.0

def auto_config_for_target(target_m, vram_gb, vram_frac=0.80):
    target_m = max(0.2, target_m)
    if target_m <= 3:
        ctxs = [128, 192]
        d_range = range(64, 513, 32)
        layer_range = range(1, 13)
        head_choices = (4,)
    elif target_m <= 12:
        ctxs = [256, 192, 128]
        d_range = range(64, 769, 32)
        layer_range = range(1, 17)
        head_choices = (4, 8)
    elif target_m <= 50:
        ctxs = [384, 256, 512]
        d_range = range(128, 1025, 32)
        layer_range = range(4, 25)
        head_choices = (8,)
    elif target_m <= 150:
        ctxs = [512, 384, 256]
        d_range = range(384, 1537, 64)
        layer_range = range(8, 33)
        head_choices = (8, 16)
    else:
        ctxs = [512, 768, 384]
        d_range = range(640, 2049, 64)
        layer_range = range(12, 49)
        head_choices = (16,)
    cand = []
    for ctx in ctxs:
        for d in d_range:
            if d % 4: continue
            for heads in head_choices:
                if d % heads:
                    continue
                phase_groups = 4 if heads <= 8 else 8
                for l in layer_range:
                    cfg = EitaConfig(d_model=d, num_layers=l, num_heads=heads, phase_groups=phase_groups, context_length=ctx)
                    score = abs(estimate_params_m(cfg)-target_m)/max(1,target_m) + 0.012*max(0,l-12) + 0.006*max(0,d-1024)/1024
                    cand.append((score, cfg))
    _, best = min(cand, key=lambda x: x[0])
    budget_mb = max(512, vram_gb*1024*vram_frac) * 0.9
    best_b = 1
    for b in range(1, 65):
        pm, am = estimate_memory_mb(best, b)
        if pm*3 + am*2 < budget_mb: best_b = b
    # Performans için batch size'ı en az 32 yap (eğer bellek izin veriyorsa)
    if target_m <= 50 and best_b < 32 and budget_mb > estimate_memory_mb(best, 32)[0] * 3 + estimate_memory_mb(best, 32)[1] * 2:
        best_b = 32
    return best, best_b, estimate_params_m(best), budget_mb

# ======================================================================
# Tokenizer & Dataset
# ======================================================================
class ByteTokenizer:
    pad_token_id = 0; eos_token_id = 1; vocab_size = 258
    def encode(self, text, add_special=True):
        ids = [b+2 for b in text.encode("utf-8","replace")]
        if add_special: ids.append(1)
        return ids
    def decode(self, ids):
        return bytes(max(0, min(255, i-2)) for i in ids if i>=2).decode("utf-8","replace")
    def save(self, out):
        (Path(out)/"tokenizer.json").write_text(
            json.dumps({"type":"byte","vocab_size":258,"pad_token_id":0,"eos_token_id":1}, indent=2))

class StreamingDataset:
    def __init__(self, tokenizer, seq_len, dataset, batch_size=32, custom_path=None, prefetch_size=8, use_eft=False):
        self.seq_len = seq_len; self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.prefetch_size = prefetch_size; self.use_eft = use_eft
        bin_path = self._get_bin_path(dataset, custom_path)
        if not bin_path.exists(): self._create_binary_cache(bin_path, dataset, custom_path)
        self.tokens = np.memmap(str(bin_path), dtype=np.int32, mode='r')
        self.length = len(self.tokens)
        self.max_start = max(1, self.length - self.seq_len - 1)
        self.prefetch_queue = queue.Queue(maxsize=prefetch_size)
        self.prefetch_thread = threading.Thread(target=self._prefetch_worker, daemon=True)
        self.prefetch_thread.start()
        logger.info("Streaming dataset hazır: %s (%d token, prefetch=%d, EFT=%s, batch_size=%d)", bin_path, self.length, prefetch_size, use_eft, batch_size)
        
    def _get_bin_path(self, dataset, custom_path):
        if dataset == "Custom" and custom_path: 
            if custom_path.endswith('.jsonl'):
                return Path(custom_path.replace('.jsonl', '.chatml.tokens.bin'))
            return Path(custom_path + ".tokens.bin")
        if dataset == "TinyStories": return Path("tiny_stories.tokens.bin")
        if dataset == "FineWeb-Edu": return Path("fineweb_edu.tokens.bin")
        return Path("custom_data.tokens.bin")
        
    def _render_chatml_messages(self, messages):
        rendered = []
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            role = str(msg.get("role", "user")).strip().lower() or "user"
            if role not in ("system", "user", "assistant"):
                role = "user"
            content = str(msg.get("content", "")).strip()
            if content:
                rendered.append(f"<|im_start|>{role}\n{content}<|im_end|>")
        return "\n".join(rendered)

    def _create_binary_cache(self, bin_path, dataset, custom_path):
        paths = []
        if dataset == "Custom" and custom_path: paths.append(custom_path)
        if dataset == "TinyStories": paths.append("tiny_stories.txt")
        if dataset == "FineWeb-Edu": paths.append("fineweb_edu.txt")
        paths.append("custom_data.txt")
        
        text = None
        for p in paths:
            if p and Path(p).exists():
                path = Path(p)
                if path.suffix == '.jsonl':
                    import json
                    texts = []
                    skipped = 0
                    chat_records = 0
                    with open(path, 'r', encoding='utf-8', errors='replace') as f:
                        for line in f:
                            try:
                                data = json.loads(line.strip())
                                if isinstance(data, dict):
                                    if 'text' in data and data['text']:
                                        texts.append(data['text'])
                                    elif 'messages' in data:
                                        rendered = self._render_chatml_messages(data['messages'])
                                        if rendered:
                                            texts.append(rendered)
                                            chat_records += 1
                                elif isinstance(data, str) and data:
                                    texts.append(data)
                            except:
                                skipped += 1
                                continue
                    text = '\n\n'.join(texts)
                    logger.info("JSONL dosyası yüklendi: %d kayıt, %d chat kaydı, %d atlandı", len(texts), chat_records, skipped)
                    if not texts:
                        raise ValueError("JSONL dosyasında geçerli veri bulunamadı.")
                else:
                    text = path.read_text(encoding="utf-8", errors="replace")
                break
        
        if text is None:
            logger.warning("Veri dosyası bulunamadı: %s. Dummy metin kullanılıyor.", paths)
            text = "The quick brown fox jumps over the lazy dog. " * 1000
            
        ids = self.tokenizer.encode(text, add_special=True)
        if len(ids) < self.seq_len + 2:
            ids = ids * ((self.seq_len + 2) // max(1, len(ids)) + 8)
        np.asarray(ids, dtype=np.int32).tofile(str(bin_path))
    
    def _prefetch_worker(self):
        """Async prefetch worker - arkada sürekli batch hazırlar"""
        while True:
            try:
                starts = np.random.randint(0, self.max_start, size=self.batch_size)
                x = np.stack([self.tokens[s:s+self.seq_len] for s in starts], 0)
                y = np.stack([self.tokens[s+1:s+self.seq_len+1] for s in starts], 0)
                x_data = x.astype(np.int32)
                y_data = y.astype(np.int32)
                
                if self.use_eft:
                    freq_dim = 32
                    high_feat = np.random.randn(self.batch_size, self.seq_len, freq_dim).astype(np.float32) * 0.1
                    med_feat = np.random.randn(self.batch_size, self.seq_len, freq_dim).astype(np.float32) * 0.1
                    low_feat = np.random.randn(self.batch_size, self.seq_len, freq_dim).astype(np.float32) * 0.1
                    self.prefetch_queue.put((x_data, y_data, high_feat, med_feat, low_feat))
                else:
                    self.prefetch_queue.put((x_data, y_data))
            except Exception as e:
                logger.error(f"Prefetch error: {e}")
                time.sleep(0.1)
    
    def batch(self, batch_size):
        """Prefetch queue'dan batch al - CPU-GPU overlap"""
        if batch_size == self.batch_size and not self.prefetch_queue.empty():
            return self.prefetch_queue.get()
        else:
            starts = np.random.randint(0, self.max_start, size=batch_size)
            x = np.stack([self.tokens[s:s+self.seq_len] for s in starts], 0)
            y = np.stack([self.tokens[s+1:s+self.seq_len+1] for s in starts], 0)
            x, y = x.astype(np.int32), y.astype(np.int32)
            if self.use_eft:
                freq_dim = 32
                high_feat = np.random.randn(batch_size, self.seq_len, freq_dim).astype(np.float32) * 0.1
                med_feat = np.random.randn(batch_size, self.seq_len, freq_dim).astype(np.float32) * 0.1
                low_feat = np.random.randn(batch_size, self.seq_len, freq_dim).astype(np.float32) * 0.1
                return x, y, high_feat, med_feat, low_feat
            return x, y

# ======================================================================
# Math helpers
# ======================================================================
def silu(x): return x * x.sigmoid()
def softplus(x): return (1.0 + x.exp()).log()
def finite_clip(x, limit=20): return x.clip(-limit, limit)
def is_bad_number(x): return math.isnan(x) or math.isinf(x)
def tensor_scalar(x):
    try:
        return float(x.item())
    except Exception:
        return float(x.numpy().flat[0])
def tensor_var(x): m = x.mean(); return ((x-m)*(x-m)).mean()
def cosine_flat(a, b):
    af, bf = a.reshape(a.shape[0],-1), b.reshape(b.shape[0],-1)
    a_norm = (af*af).sum(1).maximum(1e-16).sqrt()
    b_norm = (bf*bf).sum(1).maximum(1e-16).sqrt()
    return ((af*bf).sum(1) / (a_norm*b_norm + 1e-8)).mean()
def stack(items, dim=0): return items[0].stack(*items[1:], dim=dim) if len(items)>1 else items[0].unsqueeze(dim)
def cat(items, dim=0): return items[0].cat(*items[1:], dim=dim) if len(items)>1 else items[0]

class MinimalPhaseShifter:
    """
    ABSOLUTE MINIMAL phase shifter - zero operations for maximum speed.
    Just passes through with minimal modification.
    """
    def __init__(self, d_model, num_heads, groups):
        self.d_model = d_model
        self.alpha = Tensor.zeros(1)
    
    def __call__(self, y_ssm, y_attn, strength, enable_memory=False):
        # Zero operations - just pass through
        extra = Tensor.zeros_like(y_ssm) * strength * 0.01
        return extra, y_ssm, y_attn, {"delta": Tensor([0.0]), "mean_r": Tensor([0.0]), "alpha": Tensor([0.0])}


class LightweightWNN:
    """
    Eski LightWNN çizgisine daha yakın, saf tinygrad ve düşük maliyetli hizalama yolu.

    Head/group bazlı küçük öğrenilebilir ağırlıklarla SSM ve attention farkını ölçer.
    Matmul ve host-copy kullanmaz; bu yüzden eski bridge yolundan daha güvenlidir.
    """

    _backend_log_done = False

    def __init__(self, d_model, num_heads, groups):
        assert d_model % num_heads == 0
        hd = d_model // num_heads
        assert hd % groups == 0
        self.d_model = d_model
        self.h = num_heads
        self.g = groups
        self.gd = hd // groups

        sh = (1, 1, num_heads, groups, self.gd)
        self.w_ssm = Tensor.randn(*sh) * 0.02
        self.w_attn = Tensor.randn(*sh) * 0.02
        self.delta_scale = Tensor.ones(1, 1, num_heads, groups, 1) * 0.35
        self.alpha = Tensor.zeros(1)

        if not LightweightWNN._backend_log_done:
            logger.info("✅ LightWNN tekrar aktif (tinygrad, dusuk maliyetli)")
            LightweightWNN._backend_log_done = True

    def __call__(self, y_ssm, y_attn, strength, enable_memory=False):
        if not isinstance(strength, Tensor):
            strength = Tensor([float(strength)])

        bsz, seq, _ = y_ssm.shape
        s = finite_clip(y_ssm, 8).reshape(bsz, seq, self.h, self.g, self.gd)
        a = finite_clip(y_attn, 8).reshape(bsz, seq, self.h, self.g, self.gd)

        s_proj = (s * self.w_ssm).mean(-1, keepdim=True)
        a_proj = (a * self.w_attn).mean(-1, keepdim=True)
        delta = (s_proj - a_proj) * self.delta_scale * strength.reshape(1, 1, 1, 1, 1)
        delta = delta.clip(-0.5, 0.5)

        s_out = s * (1.0 + delta)
        a_out = a * (1.0 - delta)

        s_out = s_out.reshape(bsz, seq, self.d_model)
        a_out = a_out.reshape(bsz, seq, self.d_model)
        extra = strength.reshape(1, 1, 1) * self.alpha.tanh().reshape(1, 1, 1) * (s_out + a_out) * 0.1

        return extra, s_out, a_out, {
            "delta": delta.reshape(bsz, seq, self.h * self.g),
            "mean_r": delta.abs().mean(),
            "alpha": self.alpha.tanh().mean(),
        }


class UltraWNN:
    """
    En agresif throughput modu.

    LightWNN'deki öğrenilebilir vektörler yerine yalnızca grup ortalamaları ve birkaç
    skaler kapı kullanır; amaç mümkün olan en ucuz hizalama yolunu sağlamaktır.
    """

    _backend_log_done = False

    def __init__(self, d_model, num_heads, groups):
        assert d_model % num_heads == 0
        hd = d_model // num_heads
        assert hd % groups == 0
        self.d_model = d_model
        self.h = num_heads
        self.g = groups
        self.gd = hd // groups

        sh = (1, 1, num_heads, groups, 1)
        self.mix_gain = Tensor.ones(*sh) * 0.22
        self.state_bias = Tensor.zeros(*sh)
        self.attn_bias = Tensor.zeros(*sh)
        self.alpha = Tensor.zeros(1)

        if not UltraWNN._backend_log_done:
            logger.info("✅ UltraWNN tekrar aktif (tinygrad, maksimum hiz modu)")
            UltraWNN._backend_log_done = True

    def __call__(self, y_ssm, y_attn, strength, enable_memory=False):
        if not isinstance(strength, Tensor):
            strength = Tensor([float(strength)])

        bsz, seq, _ = y_ssm.shape
        s = finite_clip(y_ssm, 8).reshape(bsz, seq, self.h, self.g, self.gd)
        a = finite_clip(y_attn, 8).reshape(bsz, seq, self.h, self.g, self.gd)

        s_mean = s.mean(-1, keepdim=True)
        a_mean = a.mean(-1, keepdim=True)
        delta = (self.state_bias + s_mean - self.attn_bias - a_mean) * self.mix_gain
        delta = delta.clip(-0.35, 0.35) * strength.reshape(1, 1, 1, 1, 1)

        mix = 0.5 * (s + a)
        s_out = mix + (s - mix) * (1.0 + delta)
        a_out = mix + (a - mix) * (1.0 - delta)

        s_out = s_out.reshape(bsz, seq, self.d_model)
        a_out = a_out.reshape(bsz, seq, self.d_model)
        extra = strength.reshape(1, 1, 1) * self.alpha.tanh().reshape(1, 1, 1) * mix.reshape(bsz, seq, self.d_model) * 0.08

        return extra, s_out, a_out, {
            "delta": delta.reshape(bsz, seq, self.h * self.g),
            "mean_r": delta.abs().mean(),
            "alpha": self.alpha.tanh().mean(),
        }


class UltraHybridMixer:
    """
    UltraWNN hiz cizgisini koruyup, ileri fazlarda cok ucuz bir global-memory izi ekler.

    Amaç SRWM kadar pahali olmadan, state ve attention karisiminin ustune
    dusuk maliyetli bir sekans ozeti bindirmektir.
    """

    _backend_log_done = False
    _large_batch_guard_logged = False

    def __init__(self, d_model, num_heads, groups, memory_scale=0.06):
        self.core = UltraWNN(d_model, num_heads, groups)
        self.d_model = d_model
        self.h = self.core.h
        self.g = self.core.g
        self.gd = self.core.gd
        self.memory_scale = memory_scale
        sh = (1, 1, self.h, self.g, 1)
        self.memory_gain = Tensor.ones(*sh) * 0.18
        self.memory_bias = Tensor.zeros(*sh)
        self.alpha = Tensor.zeros(1)

        if not UltraHybridMixer._backend_log_done:
            logger.info("✅ UltraHybrid aktif (UltraWNN + hafif global memory izi)")
            UltraHybridMixer._backend_log_done = True

    def __call__(self, y_ssm, y_attn, strength, enable_memory=False):
        if not isinstance(strength, Tensor):
            strength = Tensor([float(strength)])

        extra, s_out, a_out, stats = self.core(y_ssm, y_attn, strength, enable_memory=False)
        if not enable_memory or self.memory_scale <= 0.0:
            return extra, s_out, a_out, {
                "delta": stats["delta"],
                "mean_r": stats["mean_r"],
                "alpha": stats["alpha"],
                "memory_energy": Tensor([0.0]),
                "memory_gate": Tensor([0.0]),
            }

        bsz, seq, _ = s_out.shape
        token_load = bsz * seq
        if token_load >= 16384:
            if not UltraHybridMixer._large_batch_guard_logged:
                logger.info("ℹ️ UltraHybrid memory guard aktif: buyuk token yukunde faz 3/4 memory yolu hafifletildi.")
                UltraHybridMixer._large_batch_guard_logged = True
            return extra, s_out, a_out, {
                "delta": stats["delta"],
                "mean_r": stats["mean_r"],
                "alpha": stats["alpha"],
                "memory_energy": Tensor([0.0]),
                "memory_gate": Tensor([0.0]),
            }

        s = s_out.reshape(bsz, seq, self.h, self.g, self.gd)
        a = a_out.reshape(bsz, seq, self.h, self.g, self.gd)
        base = 0.5 * (s + a)
        # CAUSAL memory:
        # Tum sekansin ortalamasini geri yaymak gelecegi sizdiriyordu.
        # Bunun yerine her token yalnizca kendisine kadar olan prefix'in
        # kume ortalamasini gorur.
        denom = Tensor.arange(1, seq + 1).reshape(1, seq, 1, 1, 1).cast(base.dtype)
        recalled = base.cumsum(1) / denom

        base_score = base.mean(-1, keepdim=True)
        recall_score = recalled.mean(-1, keepdim=True)
        memory_gate = (self.memory_bias + (recall_score - base_score) * self.memory_gain).sigmoid()
        memory_mix = memory_gate * recalled + (1.0 - memory_gate) * base
        gain = self.memory_scale * strength.reshape(1, 1, 1, 1, 1)

        s = s + gain * memory_mix
        a = a + gain * memory_mix
        s_out = s.reshape(bsz, seq, self.d_model)
        a_out = a.reshape(bsz, seq, self.d_model)
        extra = extra + strength.reshape(1, 1, 1) * self.alpha.tanh().reshape(1, 1, 1) * memory_mix.reshape(bsz, seq, self.d_model) * 0.05

        return extra, s_out, a_out, {
            "delta": stats["delta"],
            "mean_r": 0.5 * (stats["mean_r"] + memory_mix.abs().mean()),
            "alpha": 0.5 * (stats["alpha"] + self.alpha.tanh().mean()),
            "memory_energy": memory_mix.abs().mean(),
            "memory_gate": memory_gate.mean(),
        }


class TurboPhaseMixer:
    """
    LightWNN'den daha ucuz hizalama hattı.

    Trig, slot-memory ve buyuk projeksiyonlar yerine head/group bazli cok kucuk
    skaler kapilar kullanir. Amac, dikkat ve SSM akislarini neredeyse "free"
    maliyetle birbirine baglamaktir.
    """

    _backend_log_done = False

    def __init__(self, d_model, num_heads, groups):
        assert d_model % num_heads == 0
        hd = d_model // num_heads
        assert hd % groups == 0
        self.d_model = d_model
        self.h = num_heads
        self.g = groups
        self.gd = hd // groups

        sh = (1, 1, num_heads, groups, 1)
        self.state_gain = Tensor.ones(*sh) * 0.12
        self.context_gain = Tensor.ones(*sh) * 0.12
        self.mix_bias = Tensor.zeros(*sh)
        self.delta_gain = Tensor.ones(*sh) * 0.08
        self.alpha = Tensor.zeros(1)

        if not TurboPhaseMixer._backend_log_done:
            logger.info("✅ TURBO mixer aktif (light/ultra yerine throughput odakli)")
            TurboPhaseMixer._backend_log_done = True

    def __call__(self, y_ssm, y_attn, strength, enable_memory=False):
        if not isinstance(strength, Tensor):
            strength = Tensor([float(strength)])

        bsz, seq, _ = y_ssm.shape
        s = finite_clip(y_ssm, 8).reshape(bsz, seq, self.h, self.g, self.gd)
        a = finite_clip(y_attn, 8).reshape(bsz, seq, self.h, self.g, self.gd)

        # Agir matmul yerine group mean/std ile tek skaler ozet cikar.
        s_mean = s.mean(-1, keepdim=True)
        a_mean = a.mean(-1, keepdim=True)
        diff = a_mean - s_mean

        mix_gate = (self.mix_bias + self.state_gain * s_mean + self.context_gain * a_mean).sigmoid()
        delta = (diff * self.delta_gain).clip(-0.75, 0.75) * strength.reshape(1, 1, 1, 1, 1)

        blend = a - s
        s_out = s + mix_gate * delta * blend
        a_out = a - (1.0 - mix_gate) * delta * blend

        s_out = s_out.reshape(bsz, seq, self.d_model)
        a_out = a_out.reshape(bsz, seq, self.d_model)
        extra = strength.reshape(1, 1, 1) * self.alpha.tanh().reshape(1, 1, 1) * (s_out + a_out) * 0.1

        return extra, s_out, a_out, {
            "delta": delta.reshape(bsz, seq, self.h * self.g),
            "mean_r": delta.abs().mean(),
            "alpha": self.alpha.tanh().mean(),
            "mix_gate": mix_gate.reshape(bsz, seq, self.h * self.g),
        }


class HelixPhaseMixer:
    """
    Yeni varsayılan hizalama katmanı.

    Spectral Gate Mixer'i kısa sekans içi slot hafızasıyla birleştirir.
    Böylece yalnızca state ve context'i karıştırmak yerine, sekans içinde
    sıkıştırılmış bir hafıza izi üretip yeniden token akışına geri verir.
    """

    _backend_log_done = False
    _large_batch_guard_logged = False

    def __init__(self, d_model, num_heads, groups, memory_slots=8, memory_scale=0.18):
        self.d_model = d_model
        self.memory_slots = memory_slots
        self.memory_scale = memory_scale
        if OTOYA_NEXT_PHASE_AVAILABLE:
            self.mixer = NextSpectralGateMixer(d_model, num_heads, groups)
            if not HelixPhaseMixer._backend_log_done:
                logger.info("✅ SRWM zero-copy (tinygrad) mixer aktif")
                HelixPhaseMixer._backend_log_done = True
        else:
            logger.warning("⚠️ Otoya Next mixer bulunamadı, minimal fallback kullanılıyor")
            self.mixer = MinimalPhaseShifter(d_model, num_heads, groups)
        self.router = nn.Linear(d_model, memory_slots, bias=False)
        self.to_memory = nn.Linear(d_model, d_model, bias=False)
        self.merge_gate = nn.Linear(2 * d_model, d_model, bias=True)
        self.memory_alpha = Tensor.zeros(1)

    def __call__(self, y_ssm, y_attn, strength, enable_memory=False):
        if not isinstance(strength, Tensor):
            strength = Tensor([float(strength)])

        extra, s_out, a_out, mixer_stats = self.mixer(y_ssm, y_attn, strength)
        if not enable_memory or self.memory_scale <= 0.0:
            mean_delta = mixer_stats.get("mean_delta", mixer_stats.get("mean_r", Tensor([0.0])))
            alpha = mixer_stats.get("alpha", Tensor([0.0]))
            stats = {
                "delta": mixer_stats["delta"],
                "mean_r": mean_delta,
                "alpha": alpha,
                "memory_energy": Tensor([0.0]),
                "memory_gate": Tensor([0.0]),
            }
            return extra, s_out, a_out, stats

        base = 0.5 * (s_out + a_out)
        bsz, seq, d_model = base.shape
        token_load = bsz * seq

        if token_load >= 8192:
            if not HelixPhaseMixer._large_batch_guard_logged:
                logger.info("ℹ️ SRWM memory guard aktif: buyuk token yukunde faz 3/4 slot memory yolu hafifletildi.")
                HelixPhaseMixer._large_batch_guard_logged = True
            mean_delta = mixer_stats.get("mean_delta", mixer_stats.get("mean_r", Tensor([0.0])))
            alpha = mixer_stats.get("alpha", Tensor([0.0]))
            return extra, s_out, a_out, {
                "delta": mixer_stats["delta"],
                "mean_r": mean_delta,
                "alpha": alpha,
                "memory_energy": Tensor([0.0]),
                "memory_gate": Tensor([0.0]),
            }

        effective_slots = self.memory_slots
        if token_load >= 4096:
            effective_slots = min(self.memory_slots, 2)
        elif token_load >= 1024:
            effective_slots = min(self.memory_slots, 4)

        router_logits = self.router(base)[:, :, :effective_slots]
        slot_weights = router_logits.softmax(-1)
        mem_tokens = self.to_memory(base)

        # CAUSAL slot memory:
        # Eski kod tum seq boyunca slot_state topluyor, sonra her tokene geri
        # dagitiyordu; bu future leakage yapiyordu. Simdi her pozisyonda yalnizca
        # prefix'e kadar olan slot ozetini kullaniyoruz.
        weighted_tokens = slot_weights.unsqueeze(-1) * mem_tokens.unsqueeze(2)  # (B,S,slots,D)
        slot_prefix_sum = weighted_tokens.cumsum(1)
        slot_weight_sum = slot_weights.cumsum(1).unsqueeze(-1) + 1e-6
        slot_state = slot_prefix_sum / slot_weight_sum                          # (B,S,slots,D)
        recalled = (slot_weights.unsqueeze(-1) * slot_state).sum(2)             # (B,S,D)

        merge_gate = self.merge_gate(cat([base, recalled], dim=-1)).sigmoid()
        memory_mix = merge_gate * recalled + (1.0 - merge_gate) * base
        memory_gain = self.memory_scale * (0.5 if token_load >= 4096 else 1.0)

        s_out = s_out + memory_gain * memory_mix
        a_out = a_out + memory_gain * memory_mix
        extra = extra + strength.reshape(1, 1, 1) * self.memory_alpha.tanh().reshape(1, 1, 1) * memory_mix

        mean_delta = mixer_stats.get("mean_delta", mixer_stats.get("mean_r", Tensor([0.0])))
        alpha = mixer_stats.get("alpha", Tensor([0.0]))
        stats = {
            "delta": mixer_stats["delta"],
            "mean_r": 0.5 * (mean_delta + slot_state.abs().mean()),
            "alpha": 0.5 * (alpha + self.memory_alpha.tanh().mean()),
            "memory_energy": slot_state.abs().mean(),
            "memory_gate": merge_gate.mean(),
        }
        return extra, s_out, a_out, stats

# ======================================================================
# Model Components
# ======================================================================
class ConvSelectiveSSM:
    def __init__(self, d_model, kernel_size):
        self.delta = nn.Linear(d_model, d_model)
        self.depthwise = nn.Conv1d(d_model, d_model, kernel_size, groups=d_model, bias=True)
        self.mix = nn.Linear(d_model, d_model); self.k = kernel_size
    def __call__(self, x):
        # JIT-optimized: fused gate computation
        gate = self.delta(x).sigmoid()
        z = x * gate
        # JIT-optimized: fused permute and pad
        z_perm = z.permute(0,2,1).pad(((0,0),(0,0),(self.k-1,0)))
        # JIT-optimized: fused depthwise, silu, permute, and mix
        return self.mix(silu(self.depthwise(z_perm).permute(0,2,1)))

class CausalRouterAttention:
    def __init__(self, d_model, num_heads, context_length=128, attn_window=32):
        assert d_model % num_heads == 0
        self.d, self.h, self.hd = d_model, num_heads, d_model//num_heads
        self.context_length = context_length
        self.attn_window = max(0, min(attn_window, context_length))
        self.qkv = nn.Linear(d_model, 3*d_model, bias=False)
        self.router = nn.Linear(d_model, 1, bias=False); self.out = nn.Linear(d_model, d_model)
        
        # V12.0.0: Sabit causal mask matrisi (Graph patlamasını önler)
        mask_np = np.triu(np.ones((context_length, context_length), dtype=np.float32), k=1) * -1e4
        self.causal_mask = Tensor(mask_np)
        if self.attn_window > 0 and self.attn_window < context_length:
            local_np = np.full((context_length, context_length), -1e4, dtype=np.float32)
            for i in range(context_length):
                start = max(0, i - self.attn_window + 1)
                local_np[i, start:i+1] = 0.0
            self.local_causal_mask = Tensor(local_np)
        else:
            self.local_causal_mask = self.causal_mask
        # Precompute scale
        self.scale = Tensor([1.0 / math.sqrt(self.hd)])
        # RoPE: frekansları ve pozisyon matrisini tamamen precompute et
        # Eğitim boyunca her step'te Tensor.arange() çağrısını engeller (graph bloat)
        half_hd = max(self.hd // 2, 1)
        inv_freq_np = 1.0 / (10000 ** (np.arange(0, half_hd * 2, 2, dtype=np.float32) / self.hd))
        self.inv_freq = Tensor(inv_freq_np)
        # context_length x half_hd pozisyon matrisi
        pos_np = np.arange(context_length, dtype=np.float32).reshape(-1, 1) * inv_freq_np.reshape(1, -1)
        self._rope_cos = Tensor(np.cos(pos_np))  # (ctx, half_hd)
        self._rope_sin = Tensor(np.sin(pos_np))  # (ctx, half_hd)
        
        # KV Cache for fast generation - preallocated for performance
        self.k_cache = None
        self.v_cache = None
        self.router_bias_cache = None  # DÜZELTME: Router bias'ı da cache'le
        self.cache_seq_len = 0
        self.cache_idx = 0

    def reset_cache(self):
        """Reset KV cache for new generation."""
        self.k_cache = None
        self.v_cache = None
        self.router_bias_cache = None  # DÜZELTME: Router bias cache'ini de sıfırla
        self.cache_seq_len = 0
        self.cache_idx = 0
    
    def _apply_rope(self, x, start_pos=0):
        """RoPE — precomputed cos/sin tablolarından dilimle (sıfır overhead).
        Eğitim sırasında her adımda Tensor.arange/cos/sin yaratmaz."""
        bsz, num_heads, seq_len, head_dim = x.shape
        half_dim = head_dim // 2
        actual_start = min(max(int(start_pos), 0), max(0, self.context_length - seq_len))
        end = actual_start + seq_len
        # Precomputed tablolardan dilimle
        emb_cos = self._rope_cos[actual_start:end, :half_dim].reshape(1, 1, seq_len, half_dim)
        emb_sin = self._rope_sin[actual_start:end, :half_dim].reshape(1, 1, seq_len, half_dim)
        x1 = x[..., :half_dim]
        x2 = x[..., half_dim:]
        x_rot1 = x1 * emb_cos - x2 * emb_sin
        x_rot2 = x1 * emb_sin + x2 * emb_cos
        return cat([x_rot1, x_rot2], dim=-1)
    
    def __call__(self, x, use_cache=False, start_pos=0):
        bsz, seq, _ = x.shape; h, hd = self.h, self.hd
        # JIT-optimized: single qkv projection
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        
        # DÜZELTME: Router bias'ı hesapla ve cache'le
        router_bias = self.router(x).reshape(bsz, seq)
        
        if use_cache and self.k_cache is not None:
            # Generation mode: use cached K, V, only compute new Q
            q = q.reshape(bsz, seq, h, hd).permute(0, 2, 1, 3)
            k_new = k.reshape(bsz, seq, h, hd).permute(0, 2, 1, 3)
            v_new = v.reshape(bsz, seq, h, hd).permute(0, 2, 1, 3)
            
            # DÜZELTME: RoPE'yi yeni Q ve K'ye uygula (start_pos ile)
            q = self._apply_rope(q, start_pos)
            k_new = self._apply_rope(k_new, self.cache_seq_len)
            
            # DÜZELTME: Slice assignment kullan (preallocated belleği verimli kullan)
            # .cat() yerine doğrudan atama ile bellek sızıntısını önle
            idx = self.cache_idx
            self.k_cache[:, :, idx:idx+seq, :] = k_new
            self.v_cache[:, :, idx:idx+seq, :] = v_new
            self.router_bias_cache[:, idx:idx+seq] = router_bias
            self.cache_idx += seq
            self.cache_seq_len += seq
            
            # Use cached K, V
            cache_start = max(0, self.cache_seq_len - self.attn_window) if self.attn_window > 0 else 0
            k = self.k_cache[:, :, cache_start:self.cache_seq_len, :]
            v = self.v_cache[:, :, cache_start:self.cache_seq_len, :]
            # DÜZELTME: Cached router bias'ı kullan (full sequence)
            bias = self.router_bias_cache[:, cache_start:self.cache_seq_len]
            total_seq = self.cache_seq_len - cache_start
        else:
            # Training or first token: compute full Q, K, V
            q = q.reshape(bsz, seq, h, hd).permute(0, 2, 1, 3)
            k = k.reshape(bsz, seq, h, hd).permute(0, 2, 1, 3)
            v = v.reshape(bsz, seq, h, hd).permute(0, 2, 1, 3)
            
            # DÜZELTME: RoPE'yi Q ve K'ye uygula
            q = self._apply_rope(q, start_pos)
            k = self._apply_rope(k, start_pos)
            
            total_seq = seq
            bias = router_bias
            
            if use_cache:
                # Initialize preallocated cache
                self.k_cache = Tensor.zeros(bsz, h, self.context_length, hd)
                self.v_cache = Tensor.zeros(bsz, h, self.context_length, hd)
                self.router_bias_cache = Tensor.zeros(bsz, self.context_length)
                self.k_cache[:, :, :seq, :] = k
                self.v_cache[:, :, :seq, :] = v
                self.router_bias_cache[:, :seq] = router_bias
                self.cache_seq_len = seq
                self.cache_idx = seq

        scores = finite_clip(q.matmul(k.permute(0,1,3,2)) * self.scale, 30)
        bias_reshaped = bias.reshape(bsz, 1, 1, total_seq)
        mask_source = self.local_causal_mask if (self.attn_window > 0 and not use_cache) else self.causal_mask
        mask = mask_source[:total_seq, :total_seq].reshape(1,1,total_seq,total_seq)
        if use_cache and self.cache_seq_len > seq:
            mask = mask[:, :, -seq:, :]
        probs = (scores + bias_reshaped + mask).softmax(-1)
        y = probs.matmul(v).permute(0,2,1,3).reshape(bsz,seq,self.d)
        # DÜZELTME: Router entropy için sadece current token'ları kullan
        router_dist = router_bias.softmax(-1)
        ent = -(router_dist*(router_dist+1e-8).log()).sum(-1).mean()
        return self.out(y), ent

class SemanticGate:
    def __init__(self, d_model):
        self.syntax = nn.Linear(2*d_model, d_model); self.memory = nn.Linear(2*d_model, d_model)
        self.logic = nn.Linear(2*d_model, d_model); self.code = nn.Linear(2*d_model, d_model)
        self.importance = Tensor.zeros(4)
        self.gates = [self.syntax, self.memory, self.logic, self.code]
        
    def __call__(self, ys, ya, active):
        x = ys.cat(ya, dim=-1)
        w = self.importance.softmax(0); outs, vals = [], []
        for idx, gate in enumerate(self.gates):
            g = gate(x).sigmoid(); vals.append(g); outs.append((g*ya+(1-g)*ys)*w[idx])
        mixed = stack(outs).sum(0)
        gate_mean = stack(vals).mean(0)
        if not bool(active):
            # Faz 1-2'de gate tamamen kapali kalmasin; taban 50/50 karisima yakin
            # kalirken gate agirliklari sessizce isinabilsin.
            baseline = ys * 0.5 + ya * 0.5
            warm = 0.15
            return baseline * (1.0 - warm) + mixed * warm, (ys * 0 + 0.5) * (1.0 - warm) + gate_mean * warm
        return mixed, gate_mean
        
    def loss(self): 
        p = self.importance.softmax(0)
        return ((p-.25)*(p-.25)).mean()

class OtoyaBlock:
    def __init__(self, cfg: EitaConfig):
        d = cfg.d_model
        phase_mode = normalize_phase_mode(cfg.phase_mode)
        self.norm = nn.RMSNorm(d); self.in_proj = nn.Linear(d, 3*d)
        self.ssm = ConvSelectiveSSM(d, cfg.ssm_kernel)
        self.attn = CausalRouterAttention(d, cfg.num_heads, cfg.context_length, cfg.attn_window)
        self.ssm_norm = nn.RMSNorm(d); self.attn_norm = nn.RMSNorm(d)
        if phase_mode == "turbo":
            self.phase = TurboPhaseMixer(d, cfg.num_heads, cfg.phase_groups)
        elif phase_mode == "light_wnn":
            self.phase = LightweightWNN(d, cfg.num_heads, cfg.phase_groups)
        elif phase_mode == "ultra_wnn":
            self.phase = UltraWNN(d, cfg.num_heads, cfg.phase_groups)
        elif phase_mode == "ultra_hybrid":
            self.phase = UltraHybridMixer(d, cfg.num_heads, cfg.phase_groups)
        elif phase_mode == "srwm":
            self.phase = HelixPhaseMixer(d, cfg.num_heads, cfg.phase_groups)
        else:
            self.phase = MinimalPhaseShifter(d, cfg.num_heads, cfg.phase_groups)
        self.gate = SemanticGate(d); self.out = nn.Linear(d, d)
        # MoE (opsiyonel). use_moe=True ise FFN'in yerini alır.
        # Gate → MoE → out_proj sırası; MoE "karışımı ne yapacağını" öğrenir.
        self.moe = None
        self.use_moe = bool(getattr(cfg, "use_moe", False))
        if self.use_moe:
            try:
                from eita_moe import MoE as _MoE
                self.moe = _MoE(
                    d_model=d,
                    d_ff=getattr(cfg, "moe_d_ff", None),
                    num_experts=getattr(cfg, "num_experts", 4),
                    top_k=getattr(cfg, "top_k_experts", 2),
                    balance_loss_weight=getattr(cfg, "moe_balance_loss_weight", 0.01),
                    router_z_loss_weight=getattr(cfg, "moe_router_z_loss_weight", 0.001),
                )
            except Exception as _moe_err:  # noqa: BLE001
                logger.warning("OtoyaBlock: MoE başlatılamadı (%s); devre dışı.", _moe_err)
                self.moe = None
                self.use_moe = False

    def __call__(self, x, phase, strength, use_cache=False, start_pos=0):
        residual = x; xs, xa, xg = self.in_proj(self.norm(x)).chunk(3, -1)
        ys = finite_clip(self.ssm_norm(self.ssm(xs)), 8)
        # DÜZELTME: start_pos'u attention katmanına geçir
        ya, ent = self.attn(xa, use_cache=use_cache, start_pos=start_pos); ya = finite_clip(self.attn_norm(ya), 8)
        use_memory = phase >= 3 and isinstance(self.phase, (HelixPhaseMixer, UltraHybridMixer))
        extra, ys2, ya2, pst = self.phase(ys, ya, strength, enable_memory=use_memory)
        mix, _ = self.gate(ys2, ya2, phase>=3)
        # FFN / MoE
        if self.use_moe and self.moe is not None:
            moe_out, moe_aux = self.moe(mix)
            mix = mix + moe_out
            # Auxiliary loss'ları istatistik olarak geri döndür
            pst = dict(pst)
            for k, v in moe_aux.items():
                if k != "moe_expert_usage":
                    pst[k] = v
                else:
                    pst["moe_expert_usage"] = v
        out = residual + .2*self.out(finite_clip((mix+extra)*silu(xg), 8))
        cos = cosine_flat(ys, ya)
        return out, {"delta":pst["delta"],"mean_r":pst["mean_r"],"alpha":pst["alpha"],
                     "cos":cos,"orth":cos*cos,"router_entropy":ent,"head":self.gate.loss(),
                     **{k: v for k, v in pst.items() if k not in ("delta","mean_r","alpha")}}

class EitaModel:
    def __init__(self, cfg: EitaConfig):
        self.cfg = cfg
        self.cfg.phase_mode = normalize_phase_mode(self.cfg.phase_mode)
        self.tok = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos = nn.Embedding(cfg.context_length, cfg.d_model)
        self.pos_index_cache = Tensor(np.arange(cfg.context_length, dtype=np.int32).reshape(1, cfg.context_length))
        self.layers = [OtoyaBlock(cfg) for _ in range(cfg.num_layers)]
        self.norm = nn.RMSNorm(cfg.d_model)
        
        # EFT holographic embedding layers (optional)
        if cfg.use_eft:
            self.high_freq_proj = nn.Linear(cfg.eft_freq_dim, cfg.d_model)
            self.med_freq_proj = nn.Linear(cfg.eft_freq_dim, cfg.d_model)
            self.low_freq_proj = nn.Linear(cfg.eft_freq_dim, cfg.d_model)
            self.fusion_weights = Tensor.ones(3) / 3.0
    
    def reset_cache(self):
        """Reset KV cache in all attention layers."""
        for layer in self.layers:
            layer.attn.reset_cache()

    def __call__(self, ids, targets, phase, strength, high_feat=None, med_feat=None, low_feat=None, use_cache=False, start_pos=0):
        bsz, seq = ids.shape
        pos_start = min(max(int(start_pos), 0), max(0, self.cfg.context_length - seq))
        pos = self.pos_index_cache[:, pos_start:pos_start+seq]

        # Base embedding
        x = self.tok(ids) + self.pos(pos)

        # EFT pahali oldugu icin erken fazlarda tamamen kapatilir.
        if self.cfg.use_eft and phase >= 3 and high_feat is not None:
            high_emb = self.high_freq_proj(high_feat)
            med_emb = self.med_freq_proj(med_feat)
            low_emb = self.low_freq_proj(low_feat)
            weights = self.fusion_weights.softmax(0)
            holographic = weights[0] * high_emb + weights[1] * med_emb + weights[2] * low_emb
            x = x + 0.3 * holographic  # Residual connection

        stats = []
        for l in self.layers:
            # DÜZELTME: start_pos'u her katmana geçir (RoPE için gerekli)
            x, st = l(x, phase, strength, use_cache=use_cache, start_pos=start_pos)
            stats.append(st)
        logits = finite_clip(self.norm(x).matmul(self.tok.weight.T), 30)
        loss = None
        if targets is not None:
            loss = logits.reshape(bsz*seq, self.cfg.vocab_size).sparse_categorical_crossentropy(targets.reshape(bsz*seq))
        return logits, loss, aggregate_stats(stats)

    def generate_gpu(self, prompt_ids, max_new=200, temp=0.8, top_k=20, use_cache=True):
        was_training = Tensor.training; Tensor.training = False
        self.reset_cache()  # Cache sıfırla
        try:
            with tensor_no_grad():
                ctx = self.cfg.context_length; ids = list(prompt_ids[-ctx:]); generated = []
                
                # Prompt'u işle ve cache'i doldur (start_pos = 0)
                inp = np.array([ids[-ctx:]], dtype=np.int32); x_t = Tensor(inp)
                logits, _, _ = self(x_t, None, 4, 1.0, use_cache=use_cache, start_pos=0)
                
                # GÜNCEL POZİSYONU KAYDET
                curr_pos = x_t.shape[1]
                
                for _ in range(max_new):
                    # Context limitini aşmamak için güvenlik kilidi:
                    if curr_pos >= self.cfg.context_length: break

                    # Tek token üret (start_pos = curr_pos)
                    inp = np.array([[ids[-1]]], dtype=np.int32); x_t = Tensor(inp)
                    logits, _, _ = self(x_t, None, 4, 1.0, use_cache=use_cache, start_pos=curr_pos)
                    
                    curr_pos += 1  # Her adımda zamanı 1 ileri al
                    
                    last = logits[0, -1, :]
                    if temp > 0: last = last / temp
                    if top_k > 0:
                        vals = last.numpy(); kth = np.partition(vals, -top_k)[-top_k]
                        vals = np.where(vals<kth, -1e9, vals); last = Tensor(vals.astype(np.float32))
                    probs = last.softmax(0).numpy()
                    if np.any(np.isnan(probs)) or np.any(np.isinf(probs)): probs = np.ones(len(probs))/len(probs)
                    probs = np.clip(probs, 0, None); probs /= probs.sum() + 1e-9
                    next_id = int(np.random.choice(len(probs), p=probs))
                    ids.append(next_id); generated.append(next_id)
                    if next_id == 1: break
            return generated
        finally: Tensor.training = was_training

def aggregate_stats(stats):
    delta = cat([s["delta"].reshape(-1) for s in stats])
    return {"delta":delta,"phase_util":tensor_var(delta),"mean_delta":delta.abs().mean(),
            "cos":stack([s["cos"] for s in stats]).mean(),"alpha":stack([s["alpha"] for s in stats]).mean(),
            "mean_r":stack([s["mean_r"] for s in stats]).mean(),"orth":stack([s["orth"] for s in stats]).mean(),
            "router_entropy":stack([s["router_entropy"] for s in stats]).mean(),"head":stack([s["head"] for s in stats]).mean()}

def set_phase_trainable(model, phase):
    for l in model.layers:
        set_trainable(l.phase, phase>=2)
        set_trainable(l.gate, True)

def total_loss(lm_loss, stats, phase):
    w = reg_weights(phase); loss = lm_loss
    if w["phase"]:
        phase_penalty = (1.0 - stats["phase_util"].clip(0, 1)) + 0.02 * (1.0 - stats["router_entropy"].clip(0, 1))
        loss = loss + w["phase"] * phase_penalty
    if w["orth"]: loss = loss + w["orth"]*stats["orth"]
    if w["head"]: loss = loss + w["head"]*stats["head"]
    return loss

def resolve_export_target(raw_target: str, default_name: str = "eita_opencl_export") -> str:
    raw_target = (raw_target or "").strip()
    if not raw_target:
        raw_target = default_name
    if not any(sep in raw_target for sep in ("\\", "/")):
        cleaned = "".join(ch if ch.isalnum() or ch in ("-", "_", ".", " ") else "_" for ch in raw_target)
        raw_target = cleaned.strip(" .") or default_name
    target = Path(raw_target)
    if not target.is_absolute():
        target = Path.cwd() / target
    return str(target.resolve())

def export_model(model, tokenizer, out_dir="eita_opencl_export"):
    def _normalize_key(key: str, prefix: str) -> str:
        if prefix and key.startswith(prefix) and len(key) > len(prefix) and key[len(prefix)] != ".":
            return f"{prefix}.{key[len(prefix):]}"
        return key

    def _collect_state(obj, prefix="", out=None, seen=None):
        if out is None:
            out = {}
        if seen is None:
            seen = set()
        if obj is None:
            return out

        oid = id(obj)
        if oid in seen:
            return out
        seen.add(oid)

        if isinstance(obj, Tensor):
            if prefix:
                out[prefix] = obj
            return out
        if isinstance(obj, dict):
            for k, v in obj.items():
                key = f"{prefix}.{k}" if prefix else str(k)
                _collect_state(v, key, out, seen)
            return out
        if isinstance(obj, (list, tuple)):
            for i, v in enumerate(obj):
                key = f"{prefix}.{i}" if prefix else str(i)
                _collect_state(v, key, out, seen)
            return out

        try:
            local_state = get_state_dict(obj, prefix=prefix)
        except Exception:
            local_state = {}
        for key, value in local_state.items():
            out[_normalize_key(key, prefix)] = value

        names = []
        if hasattr(obj, "__dict__"):
            names.extend(getattr(obj, "__dict__").keys())
        slots = getattr(type(obj), "__slots__", ())
        if isinstance(slots, str):
            slots = (slots,)
        names.extend(slots)

        for name in sorted(set(n for n in names if isinstance(n, str))):
            if not name or name.startswith("_") or name == "grad":
                continue
            try:
                value = getattr(obj, name)
            except Exception:
                continue
            if callable(value) or not isinstance(value, (list, tuple, dict)):
                continue
            key = f"{prefix}.{name}" if prefix else name
            _collect_state(value, key, out, seen)
        return out

    state_dict = _collect_state(model)
    if not state_dict:
        raise RuntimeError("Export state_dict bos geldi; model agirliklari toplanamadi.")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cfg = asdict(model.cfg)
    cfg.update({"architecture":"OtoyaOpenCLV11.3.3","backend":"tinygrad-opencl"})
    (out/"config.json").write_text(json.dumps(cfg, indent=2))
    tokenizer.save(out)
    safe_save(state_dict, str(out/"model.safetensors"))
    return str(out.resolve())

# ======================================================================
# Trainer
# ======================================================================

class Trainer(threading.Thread):
    def __init__(self, cfg, train_cfg, sampler, tokenizer, q, auto_export_dir=None):
        super().__init__(daemon=True)
        self.cfg, self.train_cfg, self.sampler = cfg, train_cfg, sampler
        self.tokenizer, self.q, self.auto_export_dir = tokenizer, q, auto_export_dir
        self.stop_event = threading.Event()
        self.model = None

    def stop(self):
        self.stop_event.set()

    def run(self):
        try:
            self._run()
        except Exception as exc:
            logger.error("Eğitim başarısız: %s\n%s", exc, traceback.format_exc())
            self.q.put(("error", f"{exc}\nBkz: {LOG_PATH.resolve()}"))

    def _run(self):
        
        backend = load_backend(self.train_cfg.backend, self.train_cfg.use_vulkan, self.train_cfg.cache_level, self.train_cfg.fast_math)
        logger.info("Backend hazır: %s", backend)
        logger.info("Backend tuning: CACHELEVEL=%d fast_math=%s", self.train_cfg.cache_level, "on" if self.train_cfg.fast_math else "off")
        
        # AI Hızlandırıcı Durumu
        if VULKAN_AVAILABLE and accelerator_manager is not None:
            status = accelerator_manager.get_status()
            logger.info(f"AI Hızlandırıcılar: {status['best_method']['method']} - {status['best_method']['status']}")
            if status['best_method']['available']:
                logger.info(f"✅ Doğrudan AI hızlandırıcı erişimi aktif ({status['best_method']['accelerator_count']} hızlandırıcı)")
            else:
                logger.info("⚠️ AI hızlandırıcılar dolaylı erişim (OpenCL üzerinden)")

        random.seed(self.train_cfg.seed)
        np.random.seed(self.train_cfg.seed)
        Tensor.manual_seed(self.train_cfg.seed)
        Tensor.training = True
        BATCH_SIZE = self.train_cfg.batch_size

        model = EitaModel(self.cfg)
        self.model = model

        TinyJit = None
        for _jit_path in ["tinygrad.engine.jit", "tinygrad.jit", "tinygrad"]:
            try:
                _mod = __import__(_jit_path, fromlist=["TinyJit"])
                TinyJit = getattr(_mod, "TinyJit")
                break
            except (ImportError, AttributeError):
                continue

        params_m = estimate_params_m(self.cfg)
        jit_blocked = self.train_cfg.use_vulkan or self.train_cfg.backend == "HIP" or self.cfg.use_eft or self.cfg.phase_mode == "srwm"
        force_jit = bool(self.train_cfg.force_jit)
        auto_jit = False
        use_jit = force_jit and TinyJit is not None and not jit_blocked

        if use_jit:
            logger.info("✅ Force JIT aktif (%.1fM) — deneysel JIT yolu kullanılıyor.", params_m)
        else:
            if self.train_cfg.use_vulkan:
                logger.info("ℹ️ Vulkan doğrulama yolu aktif — JIT kapatildi.")
            elif self.train_cfg.backend == "HIP":
                logger.info("ℹ️ HIP backend icin JIT kapatildi.")
            elif self.cfg.use_eft:
                logger.info("ℹ️ EFT acik oldugu icin JIT kapatildi.")
            elif self.cfg.phase_mode == "srwm":
                logger.info("ℹ️ SRWM bellek yolu aktif oldugu icin JIT kapatildi.")
            elif TinyJit is None:
                logger.info("⚠️ TinyJit bulunamadi, eager mod kullaniliyor.")
            else:
                logger.info("ℹ️ JIT kapali — stabil eager yol kullaniliyor (%.1fM). Force JIT ile test edebilirsin.", params_m)

        # ============ EĞİTİM ADIMI FABRİKASI ============
        def make_train_step(mdl, optimizer, phase_val, use_eft):
            def _forward_backward(x_t, y_t, strength_t, high_t=None, med_t=None, low_t=None):
                optimizer.zero_grad()
                if use_eft and high_t is not None:
                    _, lm_loss, stats = mdl(
                        x_t, y_t, phase_val, strength_t,
                        high_feat=high_t, med_feat=med_t, low_feat=low_t,
                    )
                    realize_inputs = [x_t, y_t, strength_t, high_t, med_t, low_t]
                else:
                    _, lm_loss, stats = mdl(x_t, y_t, phase_val, strength_t)
                    realize_inputs = [x_t, y_t, strength_t]

                loss = total_loss(lm_loss, stats, phase_val)
                loss.backward()
                grads = [p.grad for p in optimizer.params if p.grad is not None]
                if grads:
                    Tensor.realize(*(realize_inputs + grads + [loss, lm_loss]))
                else:
                    Tensor.realize(*(realize_inputs + [loss, lm_loss]))
                return loss, lm_loss

            def _measure_metrics(x_np, y_np, strength_val: float, high_np=None, med_np=None, low_np=None):
                prev_training = Tensor.training
                Tensor.training = False
                try:
                    with tensor_no_grad():
                        x_eval = Tensor(np.asarray(x_np, dtype=np.int32))
                        y_eval = Tensor(np.asarray(y_np, dtype=np.int32))
                        strength_eval = Tensor([float(strength_val)])
                        if use_eft and high_np is not None:
                            high_eval = Tensor(np.asarray(high_np, dtype=np.float32))
                            med_eval = Tensor(np.asarray(med_np, dtype=np.float32))
                            low_eval = Tensor(np.asarray(low_np, dtype=np.float32))
                            _, lm_eval, stats_eval = mdl(
                                x_eval, y_eval, phase_val, strength_eval,
                                high_feat=high_eval, med_feat=med_eval, low_feat=low_eval,
                            )
                        else:
                            _, lm_eval, stats_eval = mdl(x_eval, y_eval, phase_val, strength_eval)
                        loss_eval = total_loss(lm_eval, stats_eval, phase_val)
                        Tensor.realize(loss_eval, lm_eval)
                        return tensor_scalar(loss_eval), tensor_scalar(lm_eval)
                finally:
                    Tensor.training = prev_training

            if use_jit and SCALABLE_JIT_AVAILABLE and _ScalableJIT is not None:
                try:
                    jit_fb = _ScalableJIT(
                        _forward_backward,
                        model=mdl,
                        policy="auto",
                        use_tiny_jit=True,
                        token_load_hint=BATCH_SIZE * self.cfg.context_length,
                    )
                    _sjit_info = jit_fb.info
                    if _sjit_info.get("wrapped_blocks", 0) > 0:
                        logger.info(
                            "🧩 ScalableJIT: %d blok sarmalandi (period=%s, tinyjit=%s)",
                            _sjit_info["wrapped_blocks"],
                            _sjit_info["realize_period"],
                            "on" if _sjit_info["tinyjit_active"] else "off",
                        )
                    elif _sjit_info.get("fallback_reason"):
                        logger.info(
                            "ℹ️ ScalableJIT pasif (%s); standart TinyJit yolu kullaniliyor.",
                            _sjit_info["fallback_reason"],
                        )
                except Exception as _sjit_err:  # noqa: BLE001
                    logger.warning("ScalableJIT kurulumu basarisiz (%s); TinyJit'e dusuluyor.", _sjit_err)
                    jit_fb = TinyJit(_forward_backward)
            else:
                jit_fb = TinyJit(_forward_backward) if use_jit and TinyJit is not None else _forward_backward

            def _step(x_np, y_np, strength_val: float, high_np=None, med_np=None, low_np=None, fetch_metrics: bool = False):
                x_t = Tensor(np.asarray(x_np, dtype=np.int32))
                y_t = Tensor(np.asarray(y_np, dtype=np.int32))
                strength_t = Tensor([float(strength_val)])

                if use_eft and high_np is not None:
                    high_t = Tensor(np.asarray(high_np, dtype=np.float32))
                    med_t = Tensor(np.asarray(med_np, dtype=np.float32))
                    low_t = Tensor(np.asarray(low_np, dtype=np.float32))
                    step_loss, step_lm_loss = jit_fb(x_t, y_t, strength_t, high_t, med_t, low_t)
                else:
                    step_loss, step_lm_loss = jit_fb(x_t, y_t, strength_t)

                optimizer.step()

                if fetch_metrics:
                    if use_jit:
                        loss_val = tensor_scalar(step_loss)
                        lm_val = tensor_scalar(step_lm_loss)
                        suspicious = (
                            (not math.isfinite(loss_val)) or
                            (not math.isfinite(lm_val)) or
                            (abs(loss_val) < 1e-8 and abs(lm_val) < 1e-8)
                        )
                        if suspicious:
                            return _measure_metrics(x_np, y_np, float(strength_val), high_np, med_np, low_np)
                        return loss_val, lm_val
                    return tensor_scalar(step_loss), tensor_scalar(step_lm_loss)
                return None, None

            _step.close = getattr(jit_fb, "close", None)
            return _step

        # ============ WARMUP ============
        logger.info("Cekirdek hazirlaniyor...")
        if self.cfg.use_eft:
            logger.info("EFT erken fazlarda pasif: ilk hiz icin faz 3'e kadar holografik yol kapali.")
        dx = Tensor(np.zeros((BATCH_SIZE, self.cfg.context_length), dtype=np.int32))
        dy = Tensor(np.zeros((BATCH_SIZE, self.cfg.context_length), dtype=np.int32))
        _, wl, _ = model(dx, dy, 1, Tensor([0.0]))
        wl.backward()
        Tensor.realize(wl)
        for p in get_parameters(model):
            p.grad = None

        # İlk eğitim adımı fabrikası
        phase = 1
        strength = phase_strength_for_step(0, self.train_cfg.total_steps)
        set_phase_trainable(model, phase)
        all_params = list(get_parameters(model))
        opt = BareAdamW(all_params, lr=self.train_cfg.lr, wd=self.train_cfg.weight_decay, grad_clip=0.5)
        jit_step = make_train_step(model, opt, phase, self.cfg.use_eft)

        # Warmup sonrası gradientleri sıfırla
        for p in get_parameters(model):
            p.grad = None

        logger.info("Warmup tamamlandı.")

        # ============ EĞİTİM DÖNGÜSÜ ============
        current_phase, start, tokens = -1, time.time(), 0
        # Warmup'da oluşturulan jit_step ve opt'ı kullan
        # Faz değişikliğinde yeniden oluşturulacak
        jit_step = jit_step  # Warmup'dan gelen
        opt = opt  # Warmup'dan gelen

        for step in range(self.train_cfg.total_steps):
            if self.stop_event.is_set():
                self.q.put(("stopped", "Durduruldu."))
                return

            # Performance monitoring
            monitor_step = GPU_OPTIMIZER_AVAILABLE and (step == 0 or (step + 1) % max(10, self.train_cfg.log_every) == 0)
            if monitor_step:
                get_optimizer().optimize_training_step()

            phase = phase_for_step(step, self.train_cfg.total_steps)
            lr = self.train_cfg.lr * 0.5 * (1 + math.cos(math.pi * step / max(1, self.train_cfg.total_steps)))
            strength = phase_strength_for_step(step, self.train_cfg.total_steps)

            # Faz geçişi: trainable bayraklarını güncelle, optimizer state'i koru
            if phase != current_phase:
                logger.info(f"🔄 Faz geçişi: {current_phase} -> {phase}")
                
                # KRİTİK: Faz geçişinden ÖNCE GPU belleğini temizle
                # Faz 3 ve 4 geçişlerinde PC kasmasını önlemek için
                if current_phase >= 2:  # Faz 2'den 3'e veya 3'ten 4'e geçerken
                    logger.info(f"🧹 Faz geçişi öncesi GPU belleği temizleniyor...")
                    try:
                        import gc
                        gc.collect()
                        gc.collect()  # Twice to be sure
                        # Clear Tinygrad cache
                        if hasattr(Tensor, '_cache'):
                            Tensor._cache.clear()
                        # Clear JIT cache if available
                        if hasattr(Tensor, '_jit_cache'):
                            Tensor._jit_cache.clear()
                        logger.info("✅ GPU belleği temizlendi")
                    except Exception as e:
                        logger.warning(f"GPU temizliğinde hata: {e}")
                
                current_phase = phase
                set_phase_trainable(model, phase)
                opt.zero_grad()
                close_prev = getattr(jit_step, "close", None)
                if callable(close_prev):
                    close_prev()
                jit_step = make_train_step(model, opt, phase, self.cfg.use_eft)
                phase_flags = []
                phase_flags.append(self.cfg.phase_mode.upper())
                if use_jit: phase_flags.append("JIT")
                if self.cfg.use_eft and phase >= 3: phase_flags.append("EFT")
                if self.cfg.phase_mode == "srwm" and phase >= 3: phase_flags.append("MEM")
                suffix = f" [{' '.join(phase_flags)}]" if phase_flags else ""
                logger.info(f"Faz {phase} — strength={strength:.3f}{suffix}")

            # LR güncelle (optimizer Tensor buffer'ını assign eder)
            opt.set_lr(lr)

            # Veri al ve eğitim adımı
            batch_data = self.sampler.batch(BATCH_SIZE)
            if batch_data is None:
                logger.warning("Sampler bos batch dondu, adim atlandi.")
                continue

            if self.cfg.use_eft:
                if len(batch_data) != 5:
                    logger.warning("EFT batch formati bozuk, adim atlandi.")
                    continue
                xb, yb, high_f, med_f, low_f = batch_data
                should_log = (step == 0) or ((step + 1) % self.train_cfg.log_every == 0)
                loss_val, lm_val = jit_step(xb, yb, strength, high_f, med_f, low_f, fetch_metrics=should_log)
            else:
                if len(batch_data) != 2:
                    logger.warning("Batch formati bozuk, adim atlandi.")
                    continue
                xb, yb = batch_data
                should_log = (step == 0) or ((step + 1) % self.train_cfg.log_every == 0)
                loss_val, lm_val = jit_step(xb, yb, strength, fetch_metrics=should_log)

            tokens += BATCH_SIZE * self.cfg.context_length

            # NaN/Inf koruması — sayısal hata olursa adımı atla
            if loss_val is not None and not math.isfinite(loss_val):
                logger.warning("step=%d — loss sayısal hata: %.6f, adım atlanıyor.", step + 1, loss_val)
                for p in get_parameters(model): p.grad = None
                continue
            if should_log and loss_val is not None and lm_val is not None and step < 100 and abs(loss_val) < 1e-8 and abs(lm_val) < 1e-8:
                y_sample = yb[0, :min(12, yb.shape[1])].tolist()
                logger.warning("step=%d — loss ve lm_loss sifira cok yakin. Ornek target tokenlari: %s", step + 1, y_sample)

            if should_log:
                elapsed = time.time() - start
                tps = tokens / max(1e-3, elapsed)

                # Validation loss
                val_loss = None
                if (step + 1) % self.train_cfg.val_every == 0:
                    val_batches = 3
                    val_loss_sum = 0.0
                    prev_training = Tensor.training
                    Tensor.training = False
                    with tensor_no_grad():
                        for _ in range(val_batches):
                            val_batch = self.sampler.batch(BATCH_SIZE)
                            if self.cfg.use_eft and phase >= 3:
                                vxb, vyb, v_high_f, v_med_f, v_low_f = val_batch
                                vx, vy = Tensor(vxb), Tensor(vyb)
                                v_high_t, v_med_t, v_low_t = Tensor(v_high_f), Tensor(v_med_f), Tensor(v_low_f)
                                _, v_lm_loss, _ = model(vx, vy, phase, Tensor([0.0]), high_feat=v_high_t, med_feat=v_med_t, low_feat=v_low_t)
                            else:
                                vxb, vyb = val_batch
                                vx, vy = Tensor(vxb), Tensor(vyb)
                                _, v_lm_loss, _ = model(vx, vy, phase, Tensor([0.0]))
                            Tensor.realize(v_lm_loss)
                            vl = tensor_scalar(v_lm_loss)
                            if math.isfinite(vl): val_loss_sum += vl
                    val_loss = val_loss_sum / val_batches
                    Tensor.training = prev_training

                metrics = {
                    "step": step + 1,
                    "phase": phase,
                    "loss": loss_val,
                    "lm_loss": lm_val,
                    "val_loss": val_loss,
                    "tok_per_sec": tps,
                    "lr": lr,
                    "elapsed": elapsed
                }
                self.q.put(("metrics", metrics))
                self.q.put(("progress", (step + 1) / self.train_cfg.total_steps))
                logger.info("step=%d loss=%.4f lm=%.4f tps=%.0f", step+1, loss_val, lm_val, tps)
                # Erken steplerde veya loss sifira cok yakin oldugunda ham degeri detayli logla
                if step < 10 or (loss_val is not None and abs(loss_val) < 1e-3):
                    logger.info("  ↳ RAW: loss=%.6f lm_loss=%.6f lr=%.2e phase=%d", loss_val, lm_val, lr, phase)
            
            # Performance monitoring end step
            if monitor_step:
                step_time = get_optimizer().end_training_step()
                if step_time > 10:  # If step takes more than 10 seconds
                    logger.warning(f"⚠️ Step {step+1} çok yavaş: {step_time:.2f}s")
            
            # Periodic GPU memory cleanup to prevent 350+ step slowdown
            if (step + 1) % self.train_cfg.cleanup_every == 0:
                logger.info(f"🧹 GPU belleği temizleniyor (step {step+1})...")
                try:
                    import gc
                    gc.collect()
                    # Clear Tinygrad cache if available
                    if hasattr(Tensor, '_cache'):
                        Tensor._cache.clear()
                    logger.info("✅ GPU belleği temizlendi")
                except Exception as e:
                    logger.warning(f"GPU temizliğinde hata: {e}")

        if self.auto_export_dir:
            export_path = export_model(model, self.tokenizer, self.auto_export_dir)
            self.q.put(("export_ready", export_path))
        self.q.put(("done", {"model": model, "tokenizer": self.tokenizer}))
        
        # GPU Memory Cleanup
        logger.info("GPU belleği temizleniyor...")
        try:
            close_prev = getattr(jit_step, "close", None)
            if callable(close_prev):
                close_prev()
            del model
            del opt
            del jit_step
            
            # Force garbage collection
            import gc
            gc.collect()
            
            # Clear Tinygrad cache
            if hasattr(Tensor, '_cache'):
                Tensor._cache.clear()
            
            logger.info("GPU belleği başarıyla temizlendi.")
        except Exception as e:
            logger.warning(f"GPU temizliğinde hata: {e}")


# ======================================================================
# GUI
# ======================================================================
class GUI:
    def __init__(self):
        self.root = tk.Tk(); self.root.title("Eita OpenCL V12.0.0 — Glass Trainer")
        self.root.geometry("1380x860"); self.root.minsize(1160, 760)
        self.root.configure(bg="#080808"); self.root.resizable(True, True)
        self.theme = {
            "bg": "#080808",
            "bg_2": "#0D0D0D",
            "card": "#141414",
            "panel": "#1A1A1A",
            "panel_2": "#222222",
            "entry": "#1F1F1F",
            "line": "#343434",
            "fg": "#F5F5F5",
            "muted": "#A6A6A6",
            "accent": "#F2F2F2",
            "accent_2": "#CFCFCF",
            "success": "#E8E8E8",
            "warning": "#BDBDBD",
            "danger": "#8F8F8F",
            "shadow": "#050505",
        }
        self.q: "queue.Queue[Tuple[str,object]]" = queue.Queue()
        self.trainer, self.model, self.tokenizer = None, None, None
        self.export_path, self.custom_path = None, None
        self.progress_var = tk.DoubleVar(value=0.0)
        self.progress_target = 0.0
        self.status_var = tk.StringVar(value="Hazır")
        self.substatus_var = tk.StringVar(value="Temiz ortam ve yeni çekirdek yapısı hazır.")
        self.status_mode = "idle"
        self._pulse_phase = 0.0
        has_queue_handler = any(isinstance(h, QueueLogHandler) for h in logger.handlers)
        if not has_queue_handler:
            qh = QueueLogHandler()
            qh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
            logger.addHandler(qh)
        self._style(); self._widgets(); self._start_animations(); self.root.after(100, self._poll)

    def _style(self):
        s = ttk.Style(); s.theme_use("clam")
        t = self.theme
        s.configure("TFrame", background=t["bg"])
        s.configure("App.TFrame", background=t["bg"])
        s.configure("Glass.TFrame", background=t["card"], relief="flat", borderwidth=0)
        s.configure("GlassInset.TFrame", background=t["panel"], relief="flat", borderwidth=0)
        s.configure("TLabel", background=t["bg"], foreground=t["fg"], font=("Segoe UI", 9))
        s.configure("Header.TLabel", background=t["card"], foreground=t["fg"], font=("Segoe UI Semibold", 11))
        s.configure("Section.TLabel", background=t["card"], foreground=t["muted"], font=("Segoe UI Semibold", 10))
        s.configure("Lbl.TLabel", background=t["card"], foreground=t["muted"], font=("Segoe UI", 9))
        s.configure("Hint.TLabel", background=t["card"], foreground=t["muted"], font=("Segoe UI", 9))
        s.configure("Stat.TLabel", background=t["panel"], foreground=t["fg"], font=("Consolas", 9))
        s.configure("Glass.TEntry", fieldbackground=t["entry"], foreground=t["fg"], bordercolor=t["line"], lightcolor=t["line"], darkcolor=t["line"], padding=6)
        s.configure("Glass.TCombobox", fieldbackground=t["entry"], background=t["entry"], foreground=t["fg"], bordercolor=t["line"], lightcolor=t["line"], darkcolor=t["line"], arrowsize=12, padding=6)
        s.map("Glass.TCombobox",
              fieldbackground=[("readonly", t["entry"])],
              background=[("readonly", t["entry"])],
              foreground=[("readonly", t["fg"])])
        s.configure("Glass.TCheckbutton", background=t["card"], foreground=t["fg"])
        s.map("Glass.TCheckbutton", foreground=[("disabled", t["muted"])])

        s.configure("Accent.TButton", background=t["accent"], foreground="#080808", font=("Segoe UI Semibold", 9), padding=8, borderwidth=0)
        s.map("Accent.TButton",
              background=[("active", "#FFFFFF"), ("disabled", "#4D4D4D")],
              foreground=[("disabled", "#A8A8A8")])
        s.configure("Ghost.TButton", background=t["panel"], foreground=t["fg"], font=("Segoe UI Semibold", 9), padding=8, borderwidth=0)
        s.map("Ghost.TButton",
              background=[("active", t["panel_2"]), ("disabled", "#1A2533")],
              foreground=[("disabled", t["muted"])])
        s.configure("Danger.TButton", background=t["danger"], foreground="#080808", font=("Segoe UI Semibold", 9), padding=8, borderwidth=0)
        s.map("Danger.TButton",
              background=[("active", "#BEBEBE"), ("disabled", "#444444")],
              foreground=[("disabled", "#A8A8A8")])

        s.configure("Glass.Horizontal.TProgressbar", troughcolor=t["panel"], background=t["accent"], bordercolor=t["panel"], lightcolor=t["accent"], darkcolor=t["accent"], thickness=10)

    def _le(self, p, l, d, r, c=0):
        ttk.Label(p, text=l, style="Lbl.TLabel").grid(row=r, column=c, sticky="w", padx=(8, 6), pady=4)
        e = ttk.Entry(p, width=12, style="Glass.TEntry")
        e.insert(0, d)
        e.grid(row=r, column=c+1, sticky="ew", pady=4, padx=(0, 8))
        return e

    def _sec(self, parent, t):
        f = ttk.Frame(parent, style="Glass.TFrame", padding=12)
        f.pack(padx=0, pady=6, fill="x")
        for col in range(4):
            f.grid_columnconfigure(col, weight=1 if col in (1, 3) else 0)
        ttk.Label(f, text=t, style="Section.TLabel").grid(row=0, column=0, columnspan=4, sticky="w", pady=(0, 8))
        line = tk.Frame(f, height=1, bg=self.theme["line"])
        line.grid(row=1, column=0, columnspan=4, sticky="ew", pady=(0, 8))
        return f

    def _widgets(self):
        self.main = ttk.Frame(self.root, style="App.TFrame", padding=14)
        self.main.pack(fill="both", expand=True)

        hero = ttk.Frame(self.main, style="Glass.TFrame", padding=16)
        hero.pack(fill="x", pady=(0, 10))
        hero.grid_columnconfigure(0, weight=1)
        hero.grid_columnconfigure(1, weight=0)

        self.hero_title = tk.Label(hero, text="OTOYA / EITA TRAINER", bg=self.theme["card"], fg=self.theme["fg"], font=("Segoe UI Semibold", 20))
        self.hero_title.grid(row=0, column=0, sticky="w")
        self.hero_subtitle = tk.Label(hero, text="Kompakt trainer paneli • ultra_hybrid varsayilan • throughput ve kalite dengesi", bg=self.theme["card"], fg=self.theme["muted"], font=("Segoe UI", 10))
        self.hero_subtitle.grid(row=1, column=0, sticky="w", pady=(4, 0))
        self.status_chip = tk.Label(hero, textvariable=self.status_var, bg="#2A2A2A", fg=self.theme["fg"], font=("Segoe UI Semibold", 9), padx=12, pady=6)
        self.status_chip.grid(row=0, column=1, sticky="e")
        self.status_meta = tk.Label(hero, textvariable=self.substatus_var, bg=self.theme["card"], fg=self.theme["muted"], font=("Segoe UI", 9))
        self.status_meta.grid(row=1, column=1, sticky="e", pady=(4, 0))

        self.toolbar = ttk.Frame(self.main, style="App.TFrame")
        self.toolbar.pack(fill="x", pady=(0, 10))
        self.sb = ttk.Button(self.toolbar, text="Başlat", style="Accent.TButton", command=self.start)
        self.sb.pack(side="left")
        self.stb = ttk.Button(self.toolbar, text="Durdur", style="Danger.TButton", command=self.stop, state="disabled")
        self.stb.pack(side="left", padx=6)
        self.eb = ttk.Button(self.toolbar, text="Dışa Aktar", style="Ghost.TButton", command=self.export, state="disabled")
        self.eb.pack(side="left", padx=6)
        ttk.Button(self.toolbar, text="Logu Ac", style="Ghost.TButton", command=self._open_log_file).pack(side="left", padx=6)
        ttk.Button(self.toolbar, text="Logu Temizle", style="Ghost.TButton", command=self._clear_log_view).pack(side="left")

        progress_card = ttk.Frame(self.main, style="Glass.TFrame", padding=12)
        progress_card.pack(fill="x", pady=(0, 10))
        ttk.Label(progress_card, text="Eğitim İlerlemesi", style="Section.TLabel").pack(anchor="w")
        self.pb = ttk.Progressbar(progress_card, length=676, mode="determinate", style="Glass.Horizontal.TProgressbar", variable=self.progress_var, maximum=100.0)
        self.pb.pack(fill="x", pady=(8, 0))

        self.body = ttk.Frame(self.main, style="App.TFrame")
        self.body.pack(fill="both", expand=True)
        self.body.grid_columnconfigure(0, weight=0)
        self.body.grid_columnconfigure(1, weight=1)
        self.body.grid_rowconfigure(0, weight=1)

        self.left_col = ttk.Frame(self.body, style="App.TFrame", width=430)
        self.left_col.grid(row=0, column=0, sticky="nsw", padx=(0, 10))
        self.left_col.grid_propagate(False)

        self.right_col = ttk.Frame(self.body, style="App.TFrame")
        self.right_col.grid(row=0, column=1, sticky="nsew")
        self.right_col.grid_columnconfigure(0, weight=1)
        self.right_col.grid_rowconfigure(1, weight=1)

        af = self._sec(self.left_col, "OTO YAPILANDIRMA")
        ttk.Label(af, text="Hedef Parametre (M):", style="Lbl.TLabel").grid(row=2, column=0, sticky="w", padx=(8, 6), pady=6)
        self.te = ttk.Entry(af, width=10, style="Glass.TEntry"); self.te.insert(0, "7"); self.te.grid(row=2, column=1, sticky="ew", pady=6)
        ttk.Label(af, text="VRAM (GB):", style="Lbl.TLabel").grid(row=2, column=2, sticky="w", padx=(12, 6), pady=6)
        self.ve = ttk.Entry(af, width=8, style="Glass.TEntry"); self.ve.insert(0, "8"); self.ve.grid(row=2, column=3, sticky="ew", pady=6)
        ttk.Button(af, text="Otomatik Ayarla", style="Ghost.TButton", command=self._ac).grid(row=3, column=0, columnspan=4, sticky="w", padx=8, pady=(4, 2))
        self.as_ = ttk.Label(af, text="Henüz hesaplanmadı.", style="Hint.TLabel"); self.as_.grid(row=4, column=0, columnspan=4, sticky="w", padx=8)

        mf = self._sec(self.left_col, "MODEL")
        self.de = self._le(mf, "d_model:", "128", 2); self.le = self._le(mf, "Layers:", "2", 3); self.ce = self._le(mf, "Context:", "128", 4)
        self.he = self._le(mf, "Heads:", "4", 2, 2); self.ge = self._le(mf, "Groups:", "4", 3, 2); self.ke = self._le(mf, "SSM Kernel:", "33", 4, 2)
        ttk.Label(mf, text="Faz Modu:", style="Lbl.TLabel").grid(row=5, column=0, sticky="w", padx=(8, 6), pady=6)
        self.phase_mode_var = tk.StringVar(value="ultra_hybrid")
        ttk.Combobox(mf, textvariable=self.phase_mode_var, values=["turbo", "light_wnn", "ultra_wnn", "ultra_hybrid", "srwm", "minimal"], state="readonly", width=16, style="Glass.TCombobox").grid(row=5, column=1, sticky="ew", pady=6)
        self.awe = self._le(mf, "Attn Window:", "32", 5, 2)
        ttk.Label(mf, text="EFT:", style="Lbl.TLabel").grid(row=6, column=0, sticky="w", padx=(8, 6), pady=6)
        self.eft_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(mf, variable=self.eft_var, style="Glass.TCheckbutton").grid(row=6, column=1, sticky="w", pady=6)

        tf = self._sec(self.left_col, "EĞİTİM")
        self.be = self._le(tf, "Batch:", "64", 2); self.se = self._le(tf, "Steps:", "500", 3); self.lre = self._le(tf, "LR:", "3e-4", 4)
        self.ae = self._le(tf, "Grad Accum:", "1", 2, 2)
        ttk.Label(tf, text="Backend:", style="Lbl.TLabel").grid(row=4, column=2, sticky="w", padx=(12, 6))
        self.bv = tk.StringVar(value="CL")
        ttk.Combobox(tf, textvariable=self.bv, values=["CL", "HIP", "CPU"], state="readonly", width=12, style="Glass.TCombobox").grid(row=4, column=3, sticky="ew")
        self.cle = self._le(tf, "CacheLevel:", "3", 5)
        ttk.Label(tf, text="Fast Math:", style="Lbl.TLabel").grid(row=5, column=2, sticky="w", padx=(12, 6), pady=6)
        self.fast_math_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(tf, variable=self.fast_math_var, style="Glass.TCheckbutton").grid(row=5, column=3, sticky="w", pady=6)
        ttk.Label(tf, text="Force JIT:", style="Lbl.TLabel").grid(row=6, column=0, sticky="w", padx=(8, 6), pady=6)
        self.force_jit_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(tf, variable=self.force_jit_var, style="Glass.TCheckbutton").grid(row=6, column=1, sticky="w", pady=6)
        self.export_entry = self._le(tf, "Export:", TrainConfig.export_dir, 7)

        df = self._sec(self.left_col, "VERİ SETİ")
        ttk.Label(df, text="Dataset:", style="Lbl.TLabel").grid(row=2, column=0, sticky="w", padx=(8, 6))
        self.dv = tk.StringVar(value="TinyStories")
        ds = ttk.Combobox(df, textvariable=self.dv, values=["TinyStories", "FineWeb-Edu", "Custom"], state="readonly", width=16, style="Glass.TCombobox")
        ds.grid(row=2, column=1, sticky="ew"); ds.bind("<<ComboboxSelected>>", self._dc)
        self.bb = ttk.Button(df, text="Gözat…", style="Ghost.TButton", command=self._br, state="disabled")
        self.bb.grid(row=2, column=2, padx=(8, 0))

        monitor_top = ttk.Frame(self.right_col, style="App.TFrame")
        monitor_top.grid(row=0, column=0, sticky="ew")
        monitor_top.grid_columnconfigure(0, weight=3)
        monitor_top.grid_columnconfigure(1, weight=2)

        mef = ttk.Frame(monitor_top, style="Glass.TFrame", padding=12)
        mef.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        ttk.Label(mef, text="CANLI METRİKLER", style="Section.TLabel").pack(anchor="w")
        self.mt = tk.Text(mef, height=6, width=54, bg=self.theme["panel"], fg=self.theme["fg"], relief="flat", font=("Cascadia Code", 9), state="disabled", highlightthickness=1, highlightbackground=self.theme["line"], bd=0, insertbackground=self.theme["fg"], padx=10, pady=8)
        self.mt.pack(fill="both", expand=True, pady=(8, 0))

        quick = ttk.Frame(monitor_top, style="Glass.TFrame", padding=12)
        quick.grid(row=0, column=1, sticky="nsew")
        ttk.Label(quick, text="AKTİF PROFİL", style="Section.TLabel").pack(anchor="w")
        self.quick_info = tk.Label(quick, justify="left", anchor="nw",
                                   text="Faz: ultra_hybrid\nBackend: CL\nCache: 3\nFast Math: on\nBatch: 64",
                                   bg=self.theme["card"], fg=self.theme["fg"], font=("Cascadia Code", 9))
        self.quick_info.pack(fill="both", expand=True, pady=(8, 0))

        lf = ttk.Frame(self.right_col, style="Glass.TFrame", padding=12)
        lf.grid(row=1, column=0, sticky="nsew", pady=(8, 0))
        ttk.Label(lf, text="SİSTEM LOGU", style="Section.TLabel").pack(anchor="w")
        self.lt = scrolledtext.ScrolledText(lf, height=18, width=84, bg=self.theme["bg_2"], fg=self.theme["fg"], relief="flat", font=("Cascadia Code", 8), state="disabled", highlightthickness=1, highlightbackground=self.theme["line"], bd=0, insertbackground=self.theme["fg"], padx=10, pady=8)
        self.lt.pack(fill="both", expand=True, pady=(8, 0))
        self.lt.tag_config("info", foreground=self.theme["fg"])
        self.lt.tag_config("warning", foreground=self.theme["warning"])
        self.lt.tag_config("error", foreground=self.theme["danger"])
        self.lt.tag_config("milestone", foreground=self.theme["accent"])
        self._log("V12: UltraHybrid varsayilan. Kompakt panel ve ChatML veri yolu hazir.", "milestone")
        self._log(f"Log dosyasi: {LOG_PATH.resolve()}", "info")
        self._set_status("Hazır", "UltraHybrid varsayilan. Hız ve kalite dengesi icin EFT kapali baslat.", mode="idle")

    def _mix_color(self, a, b, t):
        t = max(0.0, min(1.0, float(t)))
        a = a.lstrip("#"); b = b.lstrip("#")
        ar, ag, ab = int(a[0:2], 16), int(a[2:4], 16), int(a[4:6], 16)
        br, bg, bb = int(b[0:2], 16), int(b[2:4], 16), int(b[4:6], 16)
        r = int(ar + (br - ar) * t)
        g = int(ag + (bg - ag) * t)
        bl = int(ab + (bb - ab) * t)
        return f"#{r:02X}{g:02X}{bl:02X}"

    def _start_animations(self):
        self.root.after(16, self._animate_progress)
        self.root.after(40, self._animate_status_chip)

    def _animate_progress(self):
        cur = self.progress_var.get()
        delta = self.progress_target - cur
        if abs(delta) < 0.05:
            self.progress_var.set(self.progress_target)
        else:
            self.progress_var.set(cur + delta * 0.14)
        self.root.after(16, self._animate_progress)

    def _animate_status_chip(self):
        self._pulse_phase = (self._pulse_phase + 0.14) % (2 * math.pi)
        pulse = (math.sin(self._pulse_phase) + 1.0) * 0.5
        if self.status_mode == "running":
            bg = self._mix_color("#2A2A2A", self.theme["accent"], 0.28 + 0.14 * pulse)
            fg = "#080808"
        elif self.status_mode == "success":
            bg = self._mix_color("#2C2C2C", self.theme["success"], 0.30 + 0.10 * pulse)
            fg = "#080808"
        elif self.status_mode == "error":
            bg = self._mix_color("#202020", self.theme["danger"], 0.34 + 0.10 * pulse)
            fg = "#F5F5F5"
        elif self.status_mode == "warning":
            bg = self._mix_color("#242424", self.theme["warning"], 0.28 + 0.10 * pulse)
            fg = "#F5F5F5"
        else:
            bg = self._mix_color("#242424", self.theme["accent_2"], 0.14 + 0.06 * pulse)
            fg = self.theme["fg"]
        self.status_chip.configure(bg=bg, fg=fg)
        self.root.after(40, self._animate_status_chip)

    def _set_status(self, title, subtitle, mode="idle"):
        self.status_var.set(title)
        self.substatus_var.set(subtitle)
        self.status_mode = mode

    def _set_quick_info(self, lines):
        self.quick_info.configure(text="\n".join(lines))

    def _ac(self):
        try:
            tm = float(self.te.get()); vg = float(self.ve.get()); cfg, bat, am, bud = auto_config_for_target(tm, vg)
            self.de.delete(0,"end");self.de.insert(0,str(cfg.d_model)); self.le.delete(0,"end");self.le.insert(0,str(cfg.num_layers))
            self.ce.delete(0,"end");self.ce.insert(0,str(cfg.context_length)); self.he.delete(0,"end");self.he.insert(0,str(cfg.num_heads))
            self.ge.delete(0,"end");self.ge.insert(0,str(cfg.phase_groups)); self.be.delete(0,"end");self.be.insert(0,str(bat))
            self.awe.delete(0, "end"); self.awe.insert(0, str(suggest_attn_window(cfg.context_length, self.phase_mode_var.get())))
            msg = f"✔ {am:.2f}M param · d={cfg.d_model} L={cfg.num_layers} · batch={bat}"
            self.as_.configure(text=msg,foreground="#00FF88")
            self._set_status("Hazır", f"Otomatik yapılandırma seçildi: {am:.2f}M parametre hedefi.", mode="idle")
        except Exception as exc: self.as_.configure(text=f"Hata: {exc}",foreground="#FF4444")
    def _dc(self, _=None): self.bb.configure(state="normal" if self.dv.get()=="Custom" else "disabled")
    def _br(self):
        p = filedialog.askopenfilename(title="Veri seti seç",filetypes=[("Text/JSONL","*.txt;*.jsonl"),("Text","*.txt"),("JSONL","*.jsonl"),("All Files","*.*")])
        if p:
            self.custom_path = p
            self._set_status("Veri Seçildi", f"Özel veri kaynağı hazır: {Path(p).name}", mode="idle")
    def _log(self, m, tag="info"):
        self.lt.configure(state="normal"); self.lt.insert("end",f"[{datetime.now().strftime('%H:%M:%S')}] {m}\n",tag)
        self.lt.see("end"); self.lt.configure(state="disabled")
    def _open_log_file(self):
        try:
            os.startfile(str(LOG_PATH.resolve()))
        except Exception as exc:
            self._log(f"Log dosyasi acilamadi: {exc}", "error")
    def _clear_log_view(self):
        self.lt.configure(state="normal")
        self.lt.delete("1.0", "end")
        self.lt.configure(state="disabled")
        self._log("GUI log gorunumu temizlendi.", "milestone")
    def _poll(self):
        while not gui_log_queue.empty(): m, t = gui_log_queue.get(); self._log(m, t)
        while not self.q.empty(): self._handle(*self.q.get())
        self.root.after(100, self._poll)
    def _handle(self, msg, payload):
        if msg == "metrics":
            d = payload
            text = (f"Adım: {d['step']:>6}   Faz: {d['phase']}\n"
            f"Loss: {d['loss']:.4f}   LM: {d['lm_loss']:.4f}\n")
            
            # Validation loss metriği ekle
            if d.get('val_loss') is not None:
                text += f"Val Loss: {d['val_loss']:.4f}\n"
            
            text += (f"Token/sn: {d.get('tok_per_sec',0):.0f}   Süre: {d.get('elapsed',0):.1f}s\n"
            f"LR: {d['lr']:.2e}")
            
            if "delta_loss" in d:
                text += f"Delta Loss: {d['delta_loss']:.4f}\n"
            
            self.mt.configure(state="normal"); self.mt.delete("1.0","end"); self.mt.insert("1.0",text); self.mt.configure(state="disabled")
            self._set_quick_info([
                f"Faz: {self.phase_mode_var.get()} / {d['phase']}",
                f"Backend: {self.bv.get()}",
                f"Batch: {self.be.get()}",
                f"Cache: {self.cle.get()}",
                f"Fast Math: {'on' if self.fast_math_var.get() else 'off'}",
                f"TPS: {d.get('tok_per_sec',0):.0f}",
            ])
            self._set_status("Eğitim Çalışıyor", f"Adım {d['step']} • {d.get('tok_per_sec',0):.0f} tok/sn • LR {d['lr']:.2e}", mode="running")
        elif msg == "progress": self.progress_target = float(payload)*100
        elif msg == "error" or msg == "stopped":
            self._log(str(payload),"error"); self.sb.configure(state="normal"); self.stb.configure(state="disabled")
            self._set_status("Durdu", str(payload).splitlines()[0], mode="error" if msg == "error" else "warning")
        elif msg == "done":
            if isinstance(payload, dict):
                self.model = payload.get("model", self.model)
                self.tokenizer = payload.get("tokenizer", self.tokenizer)
            elif self.trainer is not None:
                self.model = getattr(self.trainer, "model", self.model)
            self.sb.configure(state="normal"); self.stb.configure(state="disabled"); self.eb.configure(state="normal")
            self._log("Eğitim başarıyla bitti.","milestone")
            self.progress_target = 100.0
            self._set_status("Tamamlandı", "Eğitim başarıyla tamamlandı ve export hazır.", mode="success")
        elif msg == "export_ready":
            self.export_path = str(payload); self._log(f"Export: {self.export_path}","milestone")
            self._set_status("Export Hazır", Path(self.export_path).name, mode="success")
    def start(self):
        try:
            tokenizer = ByteTokenizer(); backend = self.bv.get()
            phase_mode = normalize_phase_mode(self.phase_mode_var.get())
            self.phase_mode_var.set(phase_mode)
            effective_fast_math = bool(self.fast_math_var.get())
            context_length = int(self.ce.get())
            batch_size = int(self.be.get())
            attn_window = int(self.awe.get())
            cache_level = int(self.cle.get())
            if attn_window < 0:
                raise ValueError("Attn Window 0 veya daha buyuk olmali.")
            if attn_window > context_length:
                self._log(f"Attn Window context'i asti; {context_length} olarak kirpildi.", "warning")
                attn_window = context_length
            if cache_level < 0 or cache_level > 3:
                raise ValueError("CacheLevel 0 ile 3 arasinda olmali.")
            cfg = EitaConfig(vocab_size=tokenizer.vocab_size, d_model=int(self.de.get()), num_layers=int(self.le.get()),
                             num_heads=int(self.he.get()), phase_groups=int(self.ge.get()),
                             context_length=context_length, ssm_kernel=int(self.ke.get()),
                             phase_mode=phase_mode, attn_window=attn_window, use_eft=False)
            if cfg.use_eft and phase_mode == "srwm" and batch_size > 16:
                self._log("Uyari: EFT + SRWM + buyuk batch CL tarafinda yavas olabilir. 8-16 batch daha verimli olur.", "warning")
            if phase_mode == "turbo":
                self._log("Turbo mod secildi: light/ultra sinifindan daha ucuz hizalama hattı kullanilacak.", "milestone")
                if cfg.use_eft:
                    self._log("Turbo + EFT acik. Maksimum hiz icin EFT kapali baslatman daha iyi olur.", "warning")
            elif phase_mode == "light_wnn":
                self._log("LightWNN geri acildi: throughput icin eski hizli aile tinygrad uzerinden kullanilacak.", "milestone")
            elif phase_mode == "ultra_wnn":
                self._log("UltraWNN geri acildi: en agresif hiz modu aktif.", "milestone")
            elif phase_mode == "ultra_hybrid":
                self._log("UltraHybrid secildi: UltraWNN hizi korunup ileri fazlarda hafif global memory izi eklenecek.", "milestone")
            if cfg.attn_window == 0 or cfg.attn_window >= cfg.context_length:
                self._log("Full attention acik. En yuksek hiz icin turbo modda 16-32 attention window onerilir.", "warning")
            else:
                self._log(f"Local attention aktif: pencere={cfg.attn_window}/{cfg.context_length}", "milestone")
            if backend == "CL" and batch_size < 64 and not cfg.use_eft:
                self._log("CL backend icin batch 64 genelde daha iyi throughput verir; VRAM uygunsa bunu dene.", "warning")
            if backend == "CL" and effective_fast_math:
                self._log("AMD/OpenCL fast-math aktif: bu mod daha hizli ama sayisal risk tasir.", "warning")
            if self.force_jit_var.get():
                self._log("Force JIT aktif: bu test yolu deneysel, 0 loss veya donma gorursen kapat.", "warning")
            sanitize_env(backend, cache_level=cache_level, fast_math=effective_fast_math)
            sampler = StreamingDataset(tokenizer, cfg.context_length, self.dv.get(), batch_size=batch_size, custom_path=self.custom_path, prefetch_size=8, use_eft=cfg.use_eft)
            export_target = resolve_export_target(self.export_entry.get(), TrainConfig.export_dir)
            # grad_accum değerini GUI'den almak yerine manuel olarak 1 veriyoruz
            train_cfg = TrainConfig(backend=backend, batch_size=batch_size, total_steps=int(self.se.get()), lr=float(self.lre.get()),
                                    grad_accum=1, cache_level=cache_level, fast_math=effective_fast_math,
                                    force_jit=self.force_jit_var.get(), export_dir=export_target)
            self.trainer = Trainer(cfg, train_cfg, sampler, tokenizer, self.q, auto_export_dir=train_cfg.export_dir)
            self.export_path = None
            self._set_quick_info([
                f"Faz: {phase_mode}",
                f"Backend: {backend}",
                f"Batch: {batch_size}",
                f"Cache: {cache_level}",
                f"Fast Math: {'on' if effective_fast_math else 'off'}",
                f"JIT: {'force' if self.force_jit_var.get() else 'auto/off'}",
                f"Context: {context_length}",
            ])
            self.progress_var.set(0.0); self.progress_target = 0.0; self.trainer.start()
            self.sb.configure(state="disabled"); self.stb.configure(state="normal"); self.eb.configure(state="disabled")
            self._set_status("Başlatıldı", f"{backend} backend • {phase_mode} çekirdeği hazırlanıyor.", mode="running")
            self._log(f"Baslatildi • backend={backend} • faz={phase_mode} • dataset={self.dv.get()} • batch={train_cfg.batch_size} • cache={train_cfg.cache_level} • fast_math={'on' if train_cfg.fast_math else 'off'} • jit={'force' if train_cfg.force_jit else 'auto/off'} • export={train_cfg.export_dir}", "milestone")
        except Exception as exc:
            logger.error("GUI baslatma hatasi: %s", exc)
            self._log(f"Baslatma hatasi: {exc}", "error")
            messagebox.showerror("Eita V11.3.3", str(exc))
    def stop(self):
        if self.trainer and self.trainer.is_alive():
            self.trainer.stop()
            logger.info("Eğitim durduruldu, GPU belleği temizleniyor...")
            self._set_status("Durduruluyor", "GPU belleği temizleniyor ve eğitim kapanıyor.", mode="warning")
            try:
                import gc
                gc.collect()
                logger.info("GPU belleği temizlendi.")
            except Exception as e:
                logger.warning(f"GPU temizliğinde hata: {e}")
    def export(self):
        model_ref = self.model or (getattr(self.trainer, "model", None) if self.trainer is not None else None)
        tokenizer_ref = self.tokenizer or (getattr(self.trainer, "tokenizer", None) if self.trainer is not None else None)
        if model_ref is None or tokenizer_ref is None:
            if self.export_path:
                messagebox.showinfo("Export", self.export_path)
            else:
                messagebox.showwarning("Export", "Export icin elde model bulunamadi. Once egitimi tamamla.")
            return
        target = resolve_export_target(self.export_entry.get(), TrainConfig.export_dir)
        try:
            export_path = export_model(model_ref, tokenizer_ref, target)
            self.export_path = export_path
            self._log(f"Export yeniden olusturuldu: {export_path}", "milestone")
            self._set_status("Export Hazır", Path(export_path).name, mode="success")
            messagebox.showinfo("Export", export_path)
        except Exception as exc:
            logger.error("Manuel export hatasi: %s", exc)
            messagebox.showerror("Export Hatasi", str(exc))
    def run(self): self.root.mainloop()

if __name__ == "__main__":
    GUI().run()
