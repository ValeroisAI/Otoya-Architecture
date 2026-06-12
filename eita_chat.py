# ============================================================
# eita_chat.py - Simple local chat/test UI for Eita OpenCL exports
# ============================================================
"""
Loads an Eita/Otoya OpenCL export folder:
  - config.json
  - model.safetensors
  - tokenizer.json

No PyTorch. Uses the model classes from your main trainer file and tinygrad CL.

Run:
  python eita_chat.py
"""

from __future__ import annotations

import sqlite3
# SQLite'ın thread kontrolünü devre dışı bırakıyoruz (Hatanın kesin çözümü)
sqlite3.sqlite_version_info  # sqlite3'ün yüklenmesini tetikle
import os
os.environ["SQLITE_CHECK_SAME_THREAD"] = "0"
os.environ["TINY_CACHE"] = "1" # Cache açık kalabilir

import json
import math
import queue
import threading
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Literal
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk
from dataclasses import dataclass, field, asdict
from datetime import datetime
import pickle

# Dosya adı eita.py mi yoksa eita_opencl_trainer.py mi otomatik bulur:
try:
    from eita import (
        ByteTokenizer, EitaConfig, EitaModel, load_backend, probe_opencl, sanitize_env
    )
except ModuleNotFoundError:
    from eita_opencl_trainer import (
        ByteTokenizer, EitaConfig, EitaModel, load_backend, probe_opencl, sanitize_env
    )

def find_export_dirs(root: Path) -> List[Path]:
    dirs: List[Path] = []
    seen = set()
    search_roots = [root]
    models_archive = root / "models_archive"
    if models_archive.exists() and models_archive.is_dir():
        search_roots.append(models_archive)

    for base in search_roots:
        for p in base.iterdir():
            if p.is_dir() and (p / "config.json").exists() and (p / "model.safetensors").exists():
                key = str(p.resolve()).lower()
                if key not in seen:
                    seen.add(key)
                    dirs.append(p)
    return sorted(dirs, key=lambda x: x.stat().st_mtime, reverse=True)

def load_config(path: Path) -> EitaConfig:
    data = json.loads((path / "config.json").read_text(encoding="utf-8"))
    allowed = {k for k in EitaConfig.__dataclass_fields__.keys()}
    cfg = {k: data[k] for k in allowed if k in data}
    return EitaConfig(**cfg)

def softmax_np(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    e = np.exp(x)
    return e / np.sum(e)

@dataclass
class Message:
    role: Literal["user", "assistant", "system"]
    content: str
    tokens: List[int] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now().strftime("%H:%M:%S"))
    token_count: int = 0
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Message':
        return cls(**data)

def sample_token(logits: np.ndarray, temperature: float, top_k: int, greedy: bool = False,
                 top_p: float = 1.0, repetition_penalty: float = 1.0,
                 recent_tokens=None) -> int:
    """CPU sampler — sample_token_gpu ile aynı parametre seti."""
    logits = logits.astype(np.float64)
    if not np.all(np.isfinite(logits)):
        return 1  # NaN gelirse EOS döndür

    # Gizli Bomba Filtresi: Kontrol karakterlerini baskıla
    for i in range(logits.shape[0]):
        if i < 32 or (i > 126 and i < 250):
            logits[i] -= 10.0

    # Repetition penalty
    if repetition_penalty > 1.0 and recent_tokens:
        seen = set(recent_tokens[-64:])
        for t in seen:
            if 0 <= t < logits.shape[0]:
                if logits[t] > 0:
                    logits[t] /= repetition_penalty
                else:
                    logits[t] *= repetition_penalty

    if greedy:
        return int(np.argmax(logits))

    temperature = max(0.05, float(temperature))
    scaled = logits / temperature

    if top_k > 0 and top_k < scaled.shape[0]:
        kth = np.partition(scaled, -top_k)[-top_k]
        scaled = np.where(scaled < kth, -1e9, scaled)

    if 0.0 < top_p < 1.0:
        sorted_idx = np.argsort(-scaled)
        sorted_logits = scaled[sorted_idx]
        exp_sorted = np.exp(sorted_logits - sorted_logits[0])
        probs_sorted = exp_sorted / exp_sorted.sum()
        cum = np.cumsum(probs_sorted)
        keep = cum <= top_p
        keep[0] = True
        mask = np.zeros_like(scaled, dtype=bool)
        mask[sorted_idx[keep]] = True
        scaled = np.where(mask, scaled, -1e9)

    scaled = scaled - scaled.max()
    e = np.exp(scaled)
    probs = e / e.sum()
    if not np.all(np.isfinite(probs)) or probs.sum() <= 0:
        return 1
    return int(np.random.choice(np.arange(probs.shape[0]), p=probs))

def sample_token_gpu(logits, temperature: float, top_k: int, greedy: bool = False,
                    top_p: float = 1.0, repetition_penalty: float = 1.0,
                    recent_tokens=None) -> int:
    """GPU-based sampling to avoid CPU transfer bottleneck.

    İyileştirmeler (eski sürüme göre):
      - Top‑k + top‑p (nucleus) tek çağrıda birleştirildi.
      - Repetition penalty: son N token'ın logits'i cezalandırılır.
      - Kontrol‑karakter baskılama eklendi.
      - Greedy mode argmax ile tek satır.
    """
    logits_np = logits.numpy().astype(np.float64)
    if not np.all(np.isfinite(logits_np)):
        return 1

    # Gizli Bomba Filtresi: Kontrol karakterlerini baskıla
    # (CPU sampler ile aynı).
    for i in range(logits_np.shape[0]):
        if i < 32 or (i > 126 and i < 250):
            logits_np[i] -= 10.0

    # Repetition penalty: son token'ların logits'ini böl
    # (positive_scale mantığı; çok hafif, aggressive değil)
    if repetition_penalty > 1.0 and recent_tokens:
        seen = set(recent_tokens[-64:])
        for t in seen:
            if 0 <= t < logits_np.shape[0]:
                if logits_np[t] > 0:
                    logits_np[t] /= repetition_penalty
                else:
                    logits_np[t] *= repetition_penalty

    # Greedy mode
    if greedy:
        return int(np.argmax(logits_np))

    temperature = max(0.05, float(temperature))
    scaled = logits_np / temperature

    # Top‑k filtre
    if top_k > 0 and top_k < scaled.shape[0]:
        kth = np.partition(scaled, -top_k)[-top_k]
        scaled = np.where(scaled < kth, -1e9, scaled)

    # Top‑p (nucleus) filtre — kümülatif softmax
    if 0.0 < top_p < 1.0:
        sorted_idx = np.argsort(-scaled)
        sorted_logits = scaled[sorted_idx]
        exp_sorted = np.exp(sorted_logits - sorted_logits[0])
        probs_sorted = exp_sorted / exp_sorted.sum()
        cum = np.cumsum(probs_sorted)
        keep = cum <= top_p
        keep[0] = True  # en azından 1 token
        mask = np.zeros_like(scaled, dtype=bool)
        mask[sorted_idx[keep]] = True
        scaled = np.where(mask, scaled, -1e9)

    # Softmax
    scaled = scaled - scaled.max()
    e = np.exp(scaled)
    probs = e / e.sum()
    if not np.all(np.isfinite(probs)) or probs.sum() <= 0:
        return 1
    return int(np.random.choice(probs.shape[0], p=probs))

class EitaRuntime:
    def __init__(self, model_dir: Path, backend: str = "CL"):
        self.backend = load_backend(backend, cache_level=3, fast_math=False)
        from tinygrad import Tensor
        from tinygrad.nn.state import safe_load, load_state_dict

        Tensor.training = False
        self.Tensor = Tensor
        self.model_dir = model_dir
        self.cfg = load_config(model_dir)
        self.tokenizer = ByteTokenizer()
        self.model = EitaModel(self.cfg)
        self.phase = 4
        self.strength = 1.0

        state = safe_load(str(model_dir / "model.safetensors"))
        # strict=True YAPILDI: Eğer model mimarisi uyuşmuyorsa rastgele ağırlık atamak 
        # yerine hata verecek. Böylece modelin "gg" olup olmadığını anında göreceğiz.
        load_state_dict(self.model, state, strict=True, verbose=False, realize=True)
        
        # Conversation history
        self.conversation: List[Message] = []
        self.system_prompt = "You are a helpful coding and chat assistant."
        self.max_context_tokens = self.cfg.context_length - 32  # Reserve space for generation
        self._last_tokens_per_sec = 0.0

    def _forward_logits(self, token_ids: List[int], *, use_cache: bool, start_pos: int = 0):
        ctx_ids = token_ids[-self.cfg.context_length:]
        x = self.Tensor(np.asarray([ctx_ids], dtype=np.int32))
        logits, _, _ = self.model(
            x, None,
            phase=self.phase,
            strength=self.strength,
            use_cache=use_cache,
            start_pos=start_pos,
        )
        return logits[0, -1, :], len(ctx_ids)

    def build_chat_prompt(self, messages: List[Message]) -> str:
        """Build chat prompt from conversation history with ChatML format (matches training dataset)."""
        prompt_parts = []
        
        # Add system prompt if present
        if self.system_prompt:
            prompt_parts.append(f"<|im_start|>system\n{self.system_prompt}<|im_end|>")
        
        # Add conversation history
        for msg in messages:
            if msg.role == "user":
                prompt_parts.append(f"<|im_start|>user\n{msg.content}<|im_end|>")
            elif msg.role == "assistant":
                prompt_parts.append(f"<|im_start|>assistant\n{msg.content}<|im_end|>")
            elif msg.role == "system":
                prompt_parts.append(f"<|im_start|>system\n{msg.content}<|im_end|>")
        
        # Add assistant prefix for next response
        prompt_parts.append("<|im_start|>assistant\n")
        
        return "\n".join(prompt_parts)
    
    def truncate_to_context(self, messages: List[Message]) -> List[Message]:
        """Truncate conversation history to fit within context window."""
        # Build prompt and check token count
        full_prompt = self.build_chat_prompt(messages)
        tokens = self.tokenizer.encode(full_prompt, add_special=False)
        
        if len(tokens) <= self.max_context_tokens:
            return messages
        
        # Remove oldest messages until it fits
        truncated = messages.copy()
        while len(tokens) > self.max_context_tokens and len(truncated) > 1:
            # Remove the oldest non-system message
            for i, msg in enumerate(truncated):
                if msg.role != "system":
                    truncated.pop(i)
                    break
            full_prompt = self.build_chat_prompt(truncated)
            tokens = self.tokenizer.encode(full_prompt, add_special=False)

        # Tek mesaj bile pencereye sigmiyorsa, son mesaji token tabanli kirp.
        if len(tokens) > self.max_context_tokens and truncated:
            last_msg = truncated[-1]
            probe = Message(role=last_msg.role, content="", timestamp=last_msg.timestamp)
            overhead = len(self.tokenizer.encode(self.build_chat_prompt([probe]), add_special=False))
            budget = max(8, self.max_context_tokens - overhead)
            clipped_tokens = self.tokenizer.encode(last_msg.content, add_special=False)[-budget:]
            truncated[-1] = Message(
                role=last_msg.role,
                content=self.tokenizer.decode(clipped_tokens),
                tokens=clipped_tokens,
                timestamp=last_msg.timestamp,
                token_count=len(clipped_tokens),
            )
        
        return truncated
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 120,
        temperature: float = 0.8,
        top_k: int = 40,
        stop_on_eos: bool = True,
        progress_cb=None,
        use_history: bool = True,
        use_cache: bool = True,
        greedy: bool = False,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
    ) -> str:
        """Generate response with full conversation context awareness and KV cache.

        İyileştirmeler (eski sürüme göre):
          - Hot‑path'ten tüm `print()` debug çağrıları kaldırıldı.
          - `use_cache` artık etkin (varsayılan).
          - `decode()` hot‑path'te yalnızca gerekli yerlerde çağrılır
            (önceki sürümde her adımda decode + repr + f‑string vardı).
          - `stop_on_eos` aktif; boşluk spam filtresi korundu.
          - `top_p` (nucleus) ve `repetition_penalty` opsiyonel parametreler.
        """
        import time

        # Add user message to conversation history
        user_msg = Message(role="user", content=prompt)
        if use_history:
            self.conversation.append(user_msg)

        # Build context from conversation history
        context_messages = self.conversation if use_history else [user_msg]
        context_messages = self.truncate_to_context(context_messages)

        # Build the full prompt with conversation history
        chat_prompt = self.build_chat_prompt(context_messages)

        # Encode the full prompt
        ids = self.tokenizer.encode(chat_prompt, add_special=False)
        if not ids:
            ids = [2]  # Boşsa rastgele bir karakterden başla, 1 (EOS) kullanma.

        generated: List[int] = []
        recent_window: List[int] = []  # repetition penalty için
        start_time = time.time()

        # Reset KV cache for new generation
        if use_cache:
            self.model.reset_cache()
        eos_id = getattr(self.tokenizer, "eos_token_id", 1)
        ctx_len = self.cfg.context_length
        started_output = False

        if use_cache:
            next_logits, curr_pos = self._forward_logits(ids, use_cache=True, start_pos=0)
        else:
            next_logits, _ = self._forward_logits(ids, use_cache=False, start_pos=0)
            curr_pos = min(len(ids), ctx_len)

        # ---------- Autoregressive döngü ----------
        for step in range(max_new_tokens):
            tok = sample_token_gpu(
                next_logits, temperature, top_k,
                greedy=greedy, top_p=top_p,
                repetition_penalty=repetition_penalty,
                recent_tokens=recent_window,
            )
            recent_window.append(tok)
            if len(recent_window) > 128:
                recent_window = recent_window[-128:]

            # EOS kontrolü (aktif)
            if stop_on_eos and tok == eos_id:
                break

            ids.append(tok)
            piece = self.tokenizer.decode([tok])
            if started_output or piece.strip():
                started_output = True
                generated.append(tok)

            # Progress callback: 1, 4, 8, 12, ... veya son adım
            if progress_cb and generated and (step == 0 or (step + 1) % 4 == 0 or step == max_new_tokens - 1):
                progress_cb(self.tokenizer.decode(generated))

            if use_cache:
                if curr_pos >= ctx_len:
                    break
                next_logits, _ = self._forward_logits([tok], use_cache=True, start_pos=curr_pos)
                curr_pos += 1
            else:
                start_pos = max(0, len(ids) - min(len(ids), ctx_len))
                next_logits, curr_pos = self._forward_logits(ids, use_cache=False, start_pos=start_pos)

        elapsed = time.time() - start_time
        tokens_per_sec = len(generated) / elapsed if elapsed > 0 else 0.0
        self._last_tokens_per_sec = tokens_per_sec

        # Tek seferde toplu decode (daha hızlı + utf-8 güvenli)
        response = self.tokenizer.decode(generated)

        if use_history:
            assistant_msg = Message(role="assistant", content=response, tokens=generated)
            self.conversation.append(assistant_msg)

        return response
    
    def clear_conversation(self):
        """Clear conversation history."""
        self.conversation = []
    
    def set_system_prompt(self, prompt: str):
        """Set the system prompt for the conversation."""
        self.system_prompt = prompt
    
    def get_conversation_history(self) -> List[Message]:
        """Get the current conversation history."""
        return self.conversation.copy()
    
    def load_conversation(self, messages: List[Message]):
        """Load a conversation history."""
        self.conversation = messages.copy()
    
    def get_token_info(self, text: str) -> Tuple[List[int], int]:
        """Get token IDs and count for a text string."""
        tokens = self.tokenizer.encode(text, add_special=False)
        return tokens, len(tokens)
    
    def get_context_usage(self) -> Tuple[int, int, float]:
        """Get context window usage: (used, total, percentage)."""
        if not self.conversation:
            return 0, self.cfg.context_length, 0.0
        
        full_prompt = self.build_chat_prompt(self.conversation)
        tokens = self.tokenizer.encode(full_prompt, add_special=False)
        used = len(tokens)
        total = self.cfg.context_length
        percentage = (used / total) * 100 if total > 0 else 0.0
        return used, total, percentage

class ChatGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Eita Chat - Glass Console")
        self.root.geometry("1220x880")
        self.root.minsize(1080, 760)
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
        }
        self.root.configure(bg=self.theme["bg"])
        self.runtime: Optional[EitaRuntime] = None
        
        self.q = queue.Queue()  
        self.ai_queue = queue.Queue()
        self.use_history = tk.BooleanVar(value=True)
        self.use_cache = tk.BooleanVar(value=True)
        self.greedy = tk.BooleanVar(value=False)
        self.status = tk.StringVar(value="Hazır")
        self.substatus = tk.StringVar(value="Model yüklenmedi. Kök dizin ve model arşivi taranıyor.")
        self.context_status = tk.StringVar(value="Context: 0/0 tokens (0%)")
        self.status_mode = "idle"
        self._pulse_phase = 0.0
        
        self.export_dirs = find_export_dirs(Path.cwd())
        self._style()
        self._widgets()
        self._start_animations()
        
        threading.Thread(target=self._ai_worker_loop, daemon=True).start()
        self.root.after(100, self._poll)

    def _ai_worker_loop(self) -> None:
        while True:
            task_type, payload = self.ai_queue.get()
            
            if task_type == "load":
                try:
                    path = payload
                    devices = probe_opencl()
                    rt = EitaRuntime(path, backend="CL")
                    
                    from tinygrad import Tensor
                    import numpy as np
                    dummy = Tensor(np.zeros((1, 1), dtype=np.int32))
                    rt.model(dummy, None, phase=rt.phase, strength=rt.strength)[0].realize()
                    
                    self.runtime = rt
                    self.q.put(("loaded", (rt, devices)))
                except Exception as exc:
                    self.q.put(("error", f"Model Yükleme Hatası (Uyumsuz Mimari Olabilir):\n{exc}"))
                    
            elif task_type == "generate":
                try:
                    chat_prompt, max_tokens, temp, top_k, top_p, repetition_penalty, greedy, use_history, use_cache = payload
                    if self.runtime is None:
                        self.q.put(("error", "Model is not loaded!"))
                        continue
                        
                    def progress(partial: str) -> None:
                        self.q.put(("partial", partial))
                    
                    text = self.runtime.generate(
                        chat_prompt,
                        max_tokens,
                        temp,
                        top_k,
                        progress_cb=progress,
                        use_history=use_history,
                        use_cache=use_cache,
                        greedy=greedy,
                        top_p=top_p,
                        repetition_penalty=repetition_penalty,
                    )
                    self.q.put(("generated", text))
                    # Get speed info from runtime
                    if hasattr(self.runtime, '_last_tokens_per_sec'):
                        self.q.put(("speed", self.runtime._last_tokens_per_sec))
                except Exception as exc:
                    self.q.put(("error", f"{exc}\n{traceback.format_exc()}"))
            
            self.ai_queue.task_done()

    def _style(self) -> None:
        style = ttk.Style()
        style.theme_use("clam")
        t = self.theme
        style.configure("TFrame", background=t["bg"])
        style.configure("App.TFrame", background=t["bg"])
        style.configure("Glass.TFrame", background=t["card"], relief="flat", borderwidth=0)
        style.configure("TLabel", background=t["bg"], foreground=t["fg"], font=("Segoe UI", 10))
        style.configure("Header.TLabel", background=t["card"], foreground=t["fg"], font=("Segoe UI Semibold", 12))
        style.configure("Section.TLabel", background=t["card"], foreground=t["muted"], font=("Segoe UI Semibold", 11))
        style.configure("Lbl.TLabel", background=t["card"], foreground=t["muted"], font=("Segoe UI", 10))
        style.configure("Hint.TLabel", background=t["card"], foreground=t["muted"], font=("Segoe UI", 9))
        style.configure("Glass.TEntry", fieldbackground=t["entry"], foreground=t["fg"], bordercolor=t["line"], lightcolor=t["line"], darkcolor=t["line"], padding=8)
        style.configure("Glass.TCombobox", fieldbackground=t["entry"], background=t["entry"], foreground=t["fg"], bordercolor=t["line"], lightcolor=t["line"], darkcolor=t["line"], arrowsize=14, padding=8)
        style.map("Glass.TCombobox",
                  fieldbackground=[("readonly", t["entry"])],
                  background=[("readonly", t["entry"])],
                  foreground=[("readonly", t["fg"])])
        style.configure("Glass.TCheckbutton", background=t["card"], foreground=t["fg"])
        style.configure("Accent.TButton", background=t["accent"], foreground="#080808", padding=10, borderwidth=0, font=("Segoe UI Semibold", 10))
        style.map("Accent.TButton", background=[("active", "#FFFFFF"), ("disabled", "#4D4D4D")], foreground=[("disabled", "#A8A8A8")])
        style.configure("Ghost.TButton", background=t["panel"], foreground=t["fg"], padding=10, borderwidth=0, font=("Segoe UI Semibold", 10))
        style.map("Ghost.TButton", background=[("active", t["panel_2"]), ("disabled", "#1A2533")], foreground=[("disabled", t["muted"])])
        style.configure("Danger.TButton", background=t["danger"], foreground="#080808", padding=10, borderwidth=0, font=("Segoe UI Semibold", 10))
        style.map("Danger.TButton", background=[("active", "#BEBEBE"), ("disabled", "#444444")], foreground=[("disabled", "#A8A8A8")])

    def _widgets(self) -> None:
        self.main = ttk.Frame(self.root, style="App.TFrame", padding=18)
        self.main.pack(fill="both", expand=True)

        hero = ttk.Frame(self.main, style="Glass.TFrame", padding=20)
        hero.pack(fill="x", pady=(0, 12))
        hero.grid_columnconfigure(0, weight=1)
        hero.grid_columnconfigure(1, weight=0)
        self.hero_title = tk.Label(hero, text="OTOYA CHAT CONSOLE", bg=self.theme["card"], fg=self.theme["fg"], font=("Segoe UI Semibold", 24))
        self.hero_title.grid(row=0, column=0, sticky="w")
        self.hero_subtitle = tk.Label(hero, text="Glass themed export tester • tarihçe • token görünürlüğü • arşiv uyumluluğu", bg=self.theme["card"], fg=self.theme["muted"], font=("Segoe UI", 11))
        self.hero_subtitle.grid(row=1, column=0, sticky="w", pady=(6, 0))
        self.status_chip = tk.Label(hero, textvariable=self.status, bg="#2A2A2A", fg=self.theme["fg"], font=("Segoe UI Semibold", 10), padx=14, pady=7)
        self.status_chip.grid(row=0, column=1, sticky="e")
        self.status_label = tk.Label(hero, textvariable=self.substatus, bg=self.theme["card"], fg=self.theme["muted"], font=("Segoe UI", 10))
        self.status_label.grid(row=1, column=1, sticky="e", pady=(6, 0))

        top = ttk.Frame(self.main, style="Glass.TFrame", padding=16)
        top.pack(fill="x", pady=(0, 10))
        top.grid_columnconfigure(1, weight=1)
        ttk.Label(top, text="Model folder:", style="Lbl.TLabel").grid(row=0, column=0, sticky="w")
        names = [str(p) for p in self.export_dirs]
        self.model_var = tk.StringVar(value=names[0] if names else "")
        self.model_box = ttk.Combobox(top, textvariable=self.model_var, values=names, width=50, style="Glass.TCombobox", state="readonly")
        self.model_box.grid(row=0, column=1, sticky="ew", padx=8)
        ttk.Button(top, text="Gözat", style="Ghost.TButton", command=self.browse_model).grid(row=0, column=2, padx=4)
        ttk.Button(top, text="Yükle", style="Accent.TButton", command=self.load_model).grid(row=0, column=3, padx=4)

        params = ttk.Frame(self.main, style="Glass.TFrame", padding=16)
        params.pack(fill="x", pady=(0, 10))
        ttk.Label(params, text="Max tokens:", style="Lbl.TLabel").grid(row=0, column=0, sticky="w")
        self.max_entry = ttk.Entry(params, width=8, style="Glass.TEntry")
        self.max_entry.insert(0, "120")
        self.max_entry.grid(row=0, column=1, padx=(4, 16))
        ttk.Label(params, text="Temperature:", style="Lbl.TLabel").grid(row=0, column=2, sticky="w")
        self.temp_entry = ttk.Entry(params, width=8, style="Glass.TEntry")
        self.temp_entry.insert(0, "0.8")
        self.temp_entry.grid(row=0, column=3, padx=(4, 16))
        ttk.Label(params, text="Top-k:", style="Lbl.TLabel").grid(row=0, column=4, sticky="w")
        self.topk_entry = ttk.Entry(params, width=8, style="Glass.TEntry")
        self.topk_entry.insert(0, "40")
        self.topk_entry.grid(row=0, column=5, padx=(4, 16))
        ttk.Label(params, text="Top-p:", style="Lbl.TLabel").grid(row=0, column=6, sticky="w")
        self.topp_entry = ttk.Entry(params, width=8, style="Glass.TEntry")
        self.topp_entry.insert(0, "0.95")
        self.topp_entry.grid(row=0, column=7, padx=(4, 16))
        ttk.Label(params, text="Repeat:", style="Lbl.TLabel").grid(row=0, column=8, sticky="w")
        self.repeat_entry = ttk.Entry(params, width=8, style="Glass.TEntry")
        self.repeat_entry.insert(0, "1.10")
        self.repeat_entry.grid(row=0, column=9, padx=(4, 16))

        ttk.Checkbutton(params, text="Use History", variable=self.use_history, style="Glass.TCheckbutton").grid(row=0, column=10, padx=(8, 4))
        ttk.Checkbutton(params, text="Use Cache", variable=self.use_cache, style="Glass.TCheckbutton").grid(row=0, column=11, padx=4)
        ttk.Checkbutton(params, text="Greedy", variable=self.greedy, style="Glass.TCheckbutton").grid(row=0, column=12, padx=4)
        ttk.Button(params, text="Temizle", style="Ghost.TButton", command=self.clear_chat).grid(row=0, column=13, padx=4)
        ttk.Button(params, text="System Prompt", style="Ghost.TButton", command=self.edit_system_prompt).grid(row=0, column=14, padx=4)
        ttk.Button(params, text="Kaydet", style="Ghost.TButton", command=self.save_chat).grid(row=0, column=15, padx=4)
        ttk.Button(params, text="Yükle", style="Ghost.TButton", command=self.load_chat).grid(row=0, column=16, padx=4)
        ttk.Button(params, text="Token Info", style="Ghost.TButton", command=self.show_token_info).grid(row=0, column=17, padx=4)

        context_card = ttk.Frame(self.main, style="Glass.TFrame", padding=14)
        context_card.pack(fill="x", pady=(0, 10))
        ttk.Label(context_card, text="BAĞLAM DURUMU", style="Section.TLabel").pack(anchor="w")
        self.context_bar = tk.Frame(context_card, bg=self.theme["panel"], height=10)
        self.context_bar.pack(fill="x", pady=(10, 0))
        self.context_fill = tk.Frame(self.context_bar, bg=self.theme["accent"], height=10, width=1)
        self.context_fill.pack(side="left", fill="y")
        self.context_label = tk.Label(context_card, textvariable=self.context_status, bg=self.theme["card"], fg=self.theme["muted"], font=("Segoe UI", 10))
        self.context_label.pack(anchor="w", pady=(8, 0))

        self.chat_card = ttk.Frame(self.main, style="Glass.TFrame", padding=16)
        self.chat_card.pack(fill="both", expand=True, pady=(0, 10))
        ttk.Label(self.chat_card, text="KONUŞMA AKIŞI", style="Section.TLabel").pack(anchor="w")
        self.chat = scrolledtext.ScrolledText(self.chat_card, height=28, bg=self.theme["bg_2"], fg=self.theme["fg"], insertbackground="#FFFFFF", relief="flat", font=("Cascadia Code", 10), highlightthickness=1, highlightbackground=self.theme["line"], bd=0, padx=12, pady=10)
        self.chat.pack(fill="both", expand=True, pady=(10, 0))
        self.chat.tag_config("user", foreground="#FFFFFF")
        self.chat.tag_config("bot", foreground="#DADADA")
        self.chat.tag_config("sys", foreground="#BFBFBF")
        self.chat.tag_config("err", foreground="#8F8F8F")
        self.chat.tag_config("system_msg", foreground="#CFCFCF")

        bottom = ttk.Frame(self.main, style="Glass.TFrame", padding=16)
        bottom.pack(fill="x", pady=(0, 0))
        ttk.Label(bottom, text="MESAJ", style="Section.TLabel").pack(anchor="w")
        input_row = ttk.Frame(bottom, style="Glass.TFrame")
        input_row.pack(fill="x", pady=(10, 0))
        self.prompt_entry = tk.Text(input_row, height=4, bg=self.theme["entry"], fg=self.theme["fg"], insertbackground="#FFFFFF", relief="flat", font=("Segoe UI", 11), highlightthickness=1, highlightbackground=self.theme["line"], bd=0, padx=12, pady=10)
        self.prompt_entry.pack(side="left", fill="x", expand=True)
        self.send_btn = ttk.Button(input_row, text="Generate", style="Accent.TButton", command=self.generate, state="disabled")
        self.send_btn.pack(side="left", padx=(8, 0), fill="y")

        if names:
            self._append("sys", f"Bulunan model klasörü: {len(names)}. Yüklemeye hazırsın.\n")
            self._set_status("Hazır", f"Kök dizin ve model arşivi içinde {len(names)} model bulundu.", mode="idle")
        else:
            self._append("sys", "Model klasörü bulunamadı. Gözat ile manuel seçebilirsin.\n")
            self._set_status("Model Yok", "Kök dizinde veya models_archive içinde export bulunamadı.", mode="warning")

    def _mix_color(self, a: str, b: str, t: float) -> str:
        t = max(0.0, min(1.0, float(t)))
        a = a.lstrip("#")
        b = b.lstrip("#")
        ar, ag, ab = int(a[0:2], 16), int(a[2:4], 16), int(a[4:6], 16)
        br, bg, bb = int(b[0:2], 16), int(b[2:4], 16), int(b[4:6], 16)
        r = int(ar + (br - ar) * t)
        g = int(ag + (bg - ag) * t)
        bl = int(ab + (bb - ab) * t)
        return f"#{r:02X}{g:02X}{bl:02X}"

    def _start_animations(self) -> None:
        self.root.after(40, self._animate_status_chip)

    def _animate_status_chip(self) -> None:
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

    def _set_status(self, title: str, subtitle: str, mode: str = "idle") -> None:
        self.status.set(title)
        self.substatus.set(subtitle)
        self.status_mode = mode

    def _refresh_model_list(self) -> None:
        self.export_dirs = find_export_dirs(Path.cwd())
        names = [str(p) for p in self.export_dirs]
        self.model_box.configure(values=names)
        if names and self.model_var.get() not in names:
            self.model_var.set(names[0])

    def _append(self, tag: str, text: str) -> None:
        self.chat.insert("end", text, tag)
        self.chat.see("end")

    def browse_model(self) -> None:
        path = filedialog.askdirectory(title="Select Eita export folder")
        if path:
            self.model_var.set(path)
            self._set_status("Klasör Seçildi", Path(path).name, mode="idle")

    def load_model(self) -> None:
        path = Path(self.model_var.get())
        if not (path / "config.json").exists() or not (path / "model.safetensors").exists():
            messagebox.showerror("Load error", "Folder must contain config.json and model.safetensors.")
            return
        self._set_status("Yükleniyor", f"{path.name} OpenCL üzerinde hazırlanıyor.", mode="running")
        self.send_btn.configure(state="disabled")
        self.ai_queue.put(("load", path))

    def clear_chat(self) -> None:
        """Clear the conversation history."""
        if self.runtime:
            self.runtime.clear_conversation()
            self._append("sys", "\n--- Conversation cleared ---\n")
            self._update_context_status()
            self._set_status("Temizlendi", "Konuşma geçmişi sıfırlandı.", mode="idle")
        else:
            messagebox.showwarning("Warning", "No model loaded.")
    
    def edit_system_prompt(self) -> None:
        """Edit the system prompt."""
        if not self.runtime:
            messagebox.showwarning("Warning", "No model loaded.")
            return
        
        current_prompt = self.runtime.system_prompt
        dialog = tk.Toplevel(self.root)
        dialog.title("System Prompt")
        dialog.geometry("600x300")
        dialog.configure(bg="#1E1E1E")
        
        ttk.Label(dialog, text="System Prompt:", background="#1E1E1E", foreground="#EAEAEA").pack(pady=(10, 5))
        
        text_area = scrolledtext.ScrolledText(dialog, height=12, bg="#2D2D2D", fg="#FFFFFF", insertbackground="#FFFFFF", relief="flat")
        text_area.pack(fill="both", expand=True, padx=16, pady=5)
        text_area.insert("1.0", current_prompt)
        
        def save_prompt():
            new_prompt = text_area.get("1.0", "end").strip()
            self.runtime.set_system_prompt(new_prompt)
            self._append("sys", f"\nSystem prompt updated.\n")
            self._set_status("System Prompt", "Sistem yönlendirmesi güncellendi.", mode="idle")
            dialog.destroy()
        
        btn_frame = ttk.Frame(dialog)
        btn_frame.pack(pady=10)
        ttk.Button(btn_frame, text="Save", command=save_prompt).pack(side="left", padx=5)
        ttk.Button(btn_frame, text="Cancel", command=dialog.destroy).pack(side="left", padx=5)
    
    def save_chat(self) -> None:
        """Save the current conversation to a file."""
        if not self.runtime:
            messagebox.showwarning("Warning", "No model loaded.")
            return
        
        path = filedialog.asksaveasfilename(
            defaultextension=".eita",
            filetypes=[("Eita Chat", "*.eita"), ("All Files", "*.*")],
            title="Save Conversation"
        )
        if not path:
            return
        
        try:
            chat_data = {
                "model_dir": str(self.runtime.model_dir),
                "system_prompt": self.runtime.system_prompt,
                "messages": [msg.to_dict() for msg in self.runtime.conversation],
                "config": asdict(self.runtime.cfg),
                "timestamp": datetime.now().isoformat()
            }
            with open(path, 'wb') as f:
                pickle.dump(chat_data, f)
            self._append("sys", f"\nConversation saved to {path}\n")
            self._set_status("Kaydedildi", Path(path).name, mode="success")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save conversation: {e}")
    
    def load_chat(self) -> None:
        """Load a conversation from a file."""
        path = filedialog.askopenfilename(
            filetypes=[("Eita Chat", "*.eita"), ("All Files", "*.*")],
            title="Load Conversation"
        )
        if not path:
            return
        
        try:
            with open(path, 'rb') as f:
                chat_data = pickle.load(f)
            
            # Check if model matches
            if self.runtime:
                saved_model_dir = chat_data.get("model_dir", "")
                current_model_dir = str(self.runtime.model_dir)
                if saved_model_dir != current_model_dir:
                    response = messagebox.askyesno(
                        "Model Mismatch",
                        f"This conversation was created with a different model:\n{saved_model_dir}\n\nCurrent model:\n{current_model_dir}\n\nLoad anyway?"
                    )
                    if not response:
                        return
            
            # Load conversation
            self.runtime.set_system_prompt(chat_data.get("system_prompt", "You are a helpful AI assistant."))
            messages = [Message.from_dict(msg) for msg in chat_data.get("messages", [])]
            self.runtime.load_conversation(messages)
            
            # Display loaded conversation
            self.chat.delete("1.0", "end")
            self._append("sys", f"\n--- Conversation loaded from {path} ---\n")
            for msg in messages:
                if msg.role == "user":
                    self._append("user", f"User: {msg.content}\n")
                elif msg.role == "assistant":
                    self._append("bot", f"Assistant: {msg.content}\n")
                elif msg.role == "system":
                    self._append("system_msg", f"System: {msg.content}\n")
            
            self._update_context_status()
            self._set_status("Yüklendi", Path(path).name, mode="success")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load conversation: {e}")
    
    def show_token_info(self) -> None:
        """Show token information for the last message."""
        if not self.runtime or not self.runtime.conversation:
            messagebox.showinfo("Token Info", "No conversation history.")
            return
        
        last_msg = self.runtime.conversation[-1]
        tokens, count = self.runtime.get_token_info(last_msg.content)
        
        dialog = tk.Toplevel(self.root)
        dialog.title("Token Information")
        dialog.geometry("500x400")
        dialog.configure(bg="#1E1E1E")
        
        info_frame = ttk.Frame(dialog)
        info_frame.pack(fill="x", padx=16, pady=10)
        
        ttk.Label(info_frame, text=f"Role: {last_msg.role}", background="#1E1E1E", foreground="#EAEAEA").pack(anchor="w")
        ttk.Label(info_frame, text=f"Token count: {count}", background="#1E1E1E", foreground="#EAEAEA").pack(anchor="w")
        ttk.Label(info_frame, text=f"Timestamp: {last_msg.timestamp}", background="#1E1E1E", foreground="#EAEAEA").pack(anchor="w")
        
        ttk.Label(dialog, text="Content:", background="#1E1E1E", foreground="#EAEAEA").pack(anchor="w", padx=16, pady=(5, 0))
        content_text = scrolledtext.ScrolledText(dialog, height=8, bg="#2D2D2D", fg="#FFFFFF", insertbackground="#FFFFFF", relief="flat")
        content_text.pack(fill="both", expand=True, padx=16, pady=5)
        content_text.insert("1.0", last_msg.content)
        content_text.config(state="disabled")
        
        ttk.Label(dialog, text=f"Token IDs (first 50):", background="#1E1E1E", foreground="#EAEAEA").pack(anchor="w", padx=16, pady=(5, 0))
        token_text = scrolledtext.ScrolledText(dialog, height=6, bg="#2D2D2D", fg="#B8F7B8", insertbackground="#FFFFFF", relief="flat", font=("Consolas", 9))
        token_text.pack(fill="both", expand=True, padx=16, pady=5)
        token_text.insert("1.0", str(tokens[:50]))
        token_text.config(state="disabled")
    
    def _update_context_status(self) -> None:
        """Update the context usage status bar."""
        if self.runtime:
            used, total, percentage = self.runtime.get_context_usage()
            self.context_status.set(f"Context: {used}/{total} tokens ({percentage:.1f}%)")
            fill_ratio = min(1.0, max(0.0, percentage / 100.0))
            if self.context_bar.winfo_width() > 1:
                width = max(1, int(self.context_bar.winfo_width() * fill_ratio))
                self.context_fill.configure(width=width)

            if percentage > 90:
                self.context_status.set(f"Context: {used}/{total} tokens ({percentage:.1f}%) - WARNING")
                self.context_fill.configure(bg="#7A7A7A")
            elif percentage > 75:
                self.context_status.set(f"Context: {used}/{total} tokens ({percentage:.1f}%) - High")
                self.context_fill.configure(bg="#A5A5A5")
            else:
                self.context_fill.configure(bg=self.theme["accent"])
    
    def generate(self) -> None:
        prompt = self.prompt_entry.get("1.0", "end").strip()
        if not prompt:
            return
        self.prompt_entry.delete("1.0", "end")
        
        chat_prompt = prompt
        self._append("user", f"\nUser: {prompt}\n")
        self._append("bot", "Assistant: ")
        self._set_status("Üretiyor", "Model yanıt oluşturuyor.", mode="running")
        self.send_btn.configure(state="disabled")

        try:
            max_tokens = int(self.max_entry.get())
            temp = float(self.temp_entry.get())
            top_k = int(self.topk_entry.get())
            top_p = float(self.topp_entry.get())
            repetition_penalty = float(self.repeat_entry.get())
        except ValueError:
            messagebox.showerror("Bad params", "Max tokens, temperature, top-k, top-p ve repeat sayisal olmali.")
            self.send_btn.configure(state="normal")
            return

        self._last_partial = ""
        use_hist = self.use_history.get()
        self.ai_queue.put(("generate", (
            chat_prompt, max_tokens, temp, top_k, top_p, repetition_penalty,
            self.greedy.get(), use_hist, self.use_cache.get(),
        )))

    def _poll(self) -> None:
        while not self.q.empty():
            msg, payload = self.q.get()
            if msg == "loaded":
                rt, devices = payload 
                dev = devices[0]
                self._set_status("Yüklendi", f"{rt.model_dir.name} • {dev['device']} • {dev['global_mem_gb']:.1f} GB", mode="success")
                self._append("sys", f"Loaded: {rt.model_dir}\n")
                self._append("sys", f"Context length: {rt.cfg.context_length}, Vocab size: {rt.cfg.vocab_size}\n")
                self._append("sys", f"Model: d_model={rt.cfg.d_model}, layers={rt.cfg.num_layers}, heads={rt.cfg.num_heads}, phase_mode={rt.cfg.phase_mode}\n")
                self._append("sys", f"Sampling defaults: cache=on, top_p={self.topp_entry.get()}, repeat={self.repeat_entry.get()}, greedy={'on' if self.greedy.get() else 'off'}\n")
                self._append("system_msg", f"System prompt: {rt.system_prompt}\n")
                self._update_context_status()
                self.send_btn.configure(state="normal")
            elif msg == "partial":
                partial = str(payload)
                new = partial[len(getattr(self, "_last_partial", "")) :]
                self._last_partial = partial
                self._append("bot", new)
            elif msg == "generated":
                final = str(payload)
                old = getattr(self, "_last_partial", "")
                if len(final) > len(old):
                    self._append("bot", final[len(old) :])
                self._append("bot", "\n")
                self._set_status("Hazır", "Üretim tamamlandı, yeni mesaj bekleniyor.", mode="idle")
                self._update_context_status()
                self.send_btn.configure(state="normal")
            elif msg == "speed":
                tokens_per_sec = payload
                self._append("sys", f"\nGeneration speed: {tokens_per_sec:.2f} tokens/sec\n")
                self._set_status("Hazır", f"Oluşturma hızı: {tokens_per_sec:.2f} tok/sn", mode="idle")
            elif msg == "error":
                self._set_status("Hata", "İşlem sırasında hata oluştu. Log alanını kontrol et.", mode="error")
                self._append("err", f"\nERROR: {payload}\n")
                self.send_btn.configure(state="normal" if self.runtime else "disabled")
        self.root.after(100, self._poll)

    def run(self) -> None:
        self.root.mainloop()

if __name__ == "__main__":
    ChatGUI().run()
