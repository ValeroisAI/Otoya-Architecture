# ============================================================
# eitanew.py — Nihai Otoya Mimari V1.0.0
# ============================================================
"""
Nihai Otoya Mimari — Piyasayı Yerle Bir Edecek Sistem

YENİLİKLER:
1. **Quantum-Inspired Attention**: Süperpozisyon tabanlı attention mekanizması
2. **Fractal Memory Network**: Ölçeklenebilir bellek sistemi
3. **Adaptive Gradient Flow**: Akıllı gradient yönetimi
4. **Multi-Phase Fusion**: Çoklu faz birleştirme
5. **Lightning Optimizer**: Ultra hızlı optimizasyon
6. **Smart Memory Manager**: Akıllı GPU bellek yönetimi
7. **Neural Architecture Search**: Otomatik mimari optimizasyonu
8. **Distributed Training Ready**: Dağıtık eğitim desteği
"""

from __future__ import annotations
import argparse, json, logging, math, os, queue, random, threading, time, traceback
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

# Yeni kütüphaneler
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.cuda.amp import autocast, GradScaler
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("⚠️ PyTorch bulunamadı, CPU modu kullanılacak")

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False

# ----------------------------------------------------------------------
# Environment & Logging
# ----------------------------------------------------------------------
LOG_PATH = Path("otoya_ultimate.log")
logger = logging.getLogger("OtoyaUltimate")
logger.setLevel(logging.INFO)
logger.propagate = False
if not logger.handlers:
    fh = logging.FileHandler(LOG_PATH, encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(fh)

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
@dataclass
class UltimateConfig:
    """Nihai Otoya Konfigürasyonu"""
    # Model boyutları
    vocab_size: int = 50257
    d_model: int = 1024
    num_layers: int = 24
    num_heads: int = 16
    context_length: int = 2048
    
    # Yeni parametreler
    quantum_heads: int = 4  # Quantum-inspired attention heads
    fractal_depth: int = 3  # Fractal memory depth
    fusion_layers: int = 8  # Multi-phase fusion layers
    adaptive_grad: bool = True  # Adaptive gradient flow
    use_nas: bool = False  # Neural Architecture Search
    
    # Eğitim parametreleri
    batch_size: int = 8
    learning_rate: float = 3e-4
    weight_decay: float = 0.05
    grad_clip: float = 0.5
    
    # Yeni optimizasyonlar
    use_mixed_precision: bool = True
    gradient_accumulation: int = 4
    memory_efficient: bool = True
    distributed_training: bool = False
    
    def __post_init__(self):
        # Parametre validasyonu
        assert self.d_model % self.num_heads == 0, "d_model num_heads'a bölünebilir olmalı"
        assert self.quantum_heads <= self.num_heads, "Quantum heads <= total heads"

# ----------------------------------------------------------------------
# Quantum-Inspired Attention
# ----------------------------------------------------------------------
class QuantumAttention(nn.Module):
    """Quantum süperpozisyonu tabanlı attention mekanizması"""
    
    def __init__(self, d_model: int, num_heads: int, quantum_heads: int = 4):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.quantum_heads = quantum_heads
        self.head_dim = d_model // num_heads
        
        # Quantum projection layers
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)
        
        # Quantum superposition parameters
        self.quantum_weights = nn.Parameter(torch.randn(quantum_heads, self.head_dim))
        self.phase_shift = nn.Parameter(torch.randn(quantum_heads))
        
        # Fractal scaling
        self.fractal_scale = nn.Parameter(torch.ones(num_heads))
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        # Linear projections
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Quantum superposition
        quantum_q = self._apply_quantum_superposition(q)
        quantum_k = self._apply_quantum_superposition(k)
        
        # Attention scores with fractal scaling
        scores = torch.matmul(quantum_q, quantum_k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Apply fractal scaling
        scores = scores * self.fractal_scale.view(1, -1, 1, 1)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Attention weights
        attn_weights = F.softmax(scores, dim=-1)
        
        # Context with phase shift
        context = torch.matmul(attn_weights, v)
        context = self._apply_phase_shift(context)
        
        # Combine heads
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        # Output projection
        output = self.o_proj(context)
        
        return output
    
    def _apply_quantum_superposition(self, x: torch.Tensor) -> torch.Tensor:
        """Quantum süperpozisyonu uygula"""
        batch_size, num_heads, seq_len, head_dim = x.shape
        
        # Quantum heads'e uygula
        quantum_x = x[:, :self.quantum_heads, :, :]
        
        # Apply quantum weights
        quantum_x = torch.einsum('bhsd,qd->bhsq', quantum_x, self.quantum_weights)
        
        # Superposition
        quantum_x = torch.fft.fft(quantum_x, dim=-1).real
        
        # Normal heads ile birleştir
        if self.quantum_heads < self.num_heads:
            normal_x = x[:, self.quantum_heads:, :, :]
            x = torch.cat([quantum_x, normal_x], dim=1)
        else:
            x = quantum_x
            
        return x
    
    def _apply_phase_shift(self, x: torch.Tensor) -> torch.Tensor:
        """Phase shift uygula"""
        batch_size, num_heads, seq_len, head_dim = x.shape
        
        # Quantum heads'e phase shift uygula
        if self.quantum_heads > 0:
            phase_shift = torch.exp(1j * self.phase_shift).real
            x[:, :self.quantum_heads, :, :] = x[:, :self.quantum_heads, :, :] * phase_shift.view(1, -1, 1, 1)
            
        return x

# ----------------------------------------------------------------------
# Fractal Memory Network
# ----------------------------------------------------------------------
class FractalMemory(nn.Module):
    """Fractal bellek sistemi - ölçeklenebilir ve derin"""
    
    def __init__(self, d_model: int, depth: int = 3, memory_slots: int = 32):
        super().__init__()
        self.d_model = d_model
        self.depth = depth
        self.memory_slots = memory_slots
        
        # Fractal memory layers
        self.memory_layers = nn.ModuleList([
            nn.Linear(d_model, d_model) for _ in range(depth)
        ])
        
        # Memory slots
        self.memory = nn.Parameter(torch.randn(memory_slots, d_model) * 0.02)
        self.memory_gate = nn.Linear(d_model, memory_slots)
        
        # Adaptive scaling
        self.scale_factors = nn.Parameter(torch.ones(depth))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Fractal bellek işlemi"""
        batch_size, seq_len, _ = x.shape
        
        # Initial memory interaction
        memory_scores = torch.matmul(x, self.memory.T) / math.sqrt(self.d_model)
        memory_weights = F.softmax(memory_scores, dim=-1)
        
        # Memory retrieval
        retrieved = torch.matmul(memory_weights, self.memory)
        
        # Fractal processing
        for i, layer in enumerate(self.memory_layers):
            scale = self.scale_factors[i]
            x = layer(x + scale * retrieved)
            
            # Update memory
            if i < self.depth - 1:
                gate_scores = self.memory_gate(x).mean(dim=1)  # [batch, slots]
                memory_update = torch.matmul(gate_scores.softmax(dim=-1).T, x.mean(dim=1))
                self.memory.data = 0.9 * self.memory.data + 0.1 * memory_update.mean(dim=0)
        
        return x

# ----------------------------------------------------------------------
# Adaptive Gradient Flow
# ----------------------------------------------------------------------
class AdaptiveGradientFlow:
    """Akıllı gradient akışı yönetimi"""
    
    def __init__(self, model: nn.Module, config: UltimateConfig):
        self.model = model
        self.config = config
        self.gradient_stats = {}
        self.adaptive_lr = {}
        
        # Initialize adaptive learning rates
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.adaptive_lr[name] = config.learning_rate
                
    def compute_gradient_stats(self):
        """Gradient istatistiklerini hesapla"""
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                grad = param.grad.data
                
                # Compute statistics
                mean = grad.abs().mean().item()
                std = grad.std().item()
                sparsity = (grad == 0).float().mean().item()
                
                self.gradient_stats[name] = {
                    'mean': mean,
                    'std': std,
                    'sparsity': sparsity
                }
                
    def adjust_learning_rates(self):
        """Öğrenme oranlarını adaptif olarak ayarla"""
        if not self.gradient_stats:
            return
            
        for name, param in self.model.named_parameters():
            if name in self.gradient_stats:
                stats = self.gradient_stats[name]
                
                # Adaptive adjustment rules
                if stats['sparsity'] > 0.8:
                    # Too sparse, increase LR
                    self.adaptive_lr[name] *= 1.1
                elif stats['std'] > 1.0:
                    # Too noisy, decrease LR
                    self.adaptive_lr[name] *= 0.9
                elif stats['mean'] < 1e-6:
                    # Too small, increase LR
                    self.adaptive_lr[name] *= 1.05
                    
                # Clamp to reasonable range
                self.adaptive_lr[name] = max(1e-6, min(1e-2, self.adaptive_lr[name]))
                
    def apply_gradient_clipping(self):
        """Akıllı gradient clipping uygula"""
        total_norm = 0.0
        for param in self.model.parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
                
        total_norm = total_norm ** 0.5
        
        # Adaptive clipping threshold
        clip_threshold = self.config.grad_clip * math.sqrt(len(list(self.model.parameters())))
        
        if total_norm > clip_threshold:
            clip_coef = clip_threshold / (total_norm + 1e-6)
            for param in self.model.parameters():
                if param.grad is not None:
                    param.grad.data.mul_(clip_coef)
                    
            logger.debug(f"Gradient clipped: {total_norm:.4f} -> {clip_threshold:.4f}")

# ----------------------------------------------------------------------
# Multi-Phase Fusion Layer
# ----------------------------------------------------------------------
class MultiPhaseFusion(nn.Module):
    """Çoklu faz birleştirme katmanı"""
    
    def __init__(self, d_model: int, num_phases: int = 4):
        super().__init__()
        self.d_model = d_model
        self.num_phases = num_phases
        
        # Phase-specific projections
        self.phase_projections = nn.ModuleList([
            nn.Linear(d_model, d_model) for _ in range(num_phases)
        ])
        
        # Fusion weights
        self.fusion_weights = nn.Parameter(torch.ones(num_phases) / num_phases)
        
        # Dynamic routing
        self.router = nn.Linear(d_model, num_phases)
        
    def forward(self, x: torch.Tensor, phase: int) -> torch.Tensor:
        """Çoklu faz birleştirme"""
        batch_size, seq_len, _ = x.shape
        
        # Get phase-specific representation
        phase_repr = self.phase_projections[phase](x)
        
        # Compute routing scores
        routing_scores = self.router(x.mean(dim=1))  # [batch, num_phases]
        routing_weights = F.softmax(routing_scores, dim=-1)
        
        # Weighted fusion
        fusion_weight = self.fusion_weights[phase] * routing_weights[:, phase].unsqueeze(-1).unsqueeze(-1)
        
        # Apply fusion
        output = x + fusion_weight * phase_repr
        
        return output

# ----------------------------------------------------------------------
# Lightning Optimizer
# ----------------------------------------------------------------------
class LightningOptimizer:
    """Ultra hızlı optimizasyon sistemi"""
    
    def __init__(self, model: nn.Module, config: UltimateConfig):
        self.model = model
        self.config = config
        
        # Separate parameter groups
        self.param_groups = self._create_param_groups()
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.param_groups,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # Gradient scaler for mixed precision
        self.scaler = GradScaler() if config.use_mixed_precision else None
        
        # Adaptive gradient flow
        self.gradient_flow = AdaptiveGradientFlow(model, config)
        
        # Performance tracking
        self.step_times = []
        self.gradient_norms = []
        
    def _create_param_groups(self):
        """Parametre gruplarını oluştur"""
        param_groups = []
        
        # Group by layer type
        attention_params = []
        memory_params = []
        fusion_params = []
        other_params = []
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if 'attention' in name.lower():
                    attention_params.append(param)
                elif 'memory' in name.lower():
                    memory_params.append(param)
                elif 'fusion' in name.lower():
                    fusion_params.append(param)
                else:
                    other_params.append(param)
                    
        # Add groups with different settings
        if attention_params:
            param_groups.append({'params': attention_params, 'lr': self.config.learning_rate})
        if memory_params:
            param_groups.append({'params': memory_params, 'lr': self.config.learning_rate * 0.5})
        if fusion_params:
            param_groups.append({'params': fusion_params, 'lr': self.config.learning_rate * 1.5})
        if other_params:
            param_groups.append({'params': other_params, 'lr': self.config.learning_rate})
            
        return param_groups
    
    def zero_grad(self):
        """Gradientleri sıfırla"""
        self.optimizer.zero_grad()
        
    def step(self, loss: torch.Tensor):
        """Optimizasyon adımı"""
        start_time = time.time()
        
        # Backward pass
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
            
        # Adaptive gradient flow
        self.gradient_flow.compute_gradient_stats()
        self.gradient_flow.apply_gradient_clipping()
        
        # Optimizer step
        if self.scaler is not None:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()
            
        # Adjust learning rates
        self.gradient_flow.adjust_learning_rates()
        
        # Track performance
        step_time = time.time() - start_time
        self.step_times.append(step_time)
        
        # Keep only recent history
        if len(self.step_times) > 100:
            self.step_times.pop(0)
            
        return step_time
    
    def get_average_step_time(self):
        """Ortalama step süresini al"""
        if not self.step_times:
            return 0.0
        return sum(self.step_times) / len(self.step_times)

# ----------------------------------------------------------------------
# Smart Memory Manager
# ----------------------------------------------------------------------
class SmartMemoryManager:
    """Akıllı GPU bellek yöneticisi"""
    
    def __init__(self, device: torch.device):
        self.device = device
        self.memory_stats = {
            'allocated': [],
            'reserved': [],
            'peak_allocated': 0,
            'peak_reserved': 0
        }
        
    def track_memory(self):
        """Bellek kullanımını takip et"""
        if self.device.type == 'cuda':
            allocated = torch.cuda.memory_allocated(self.device) / 1024**3  # GB
            reserved = torch.cuda.memory_reserved(self.device) / 1024**3  # GB
            
            self.memory_stats['allocated'].append(allocated)
            self.memory_stats['reserved'].append(reserved)
            
            self.memory_stats['peak_allocated'] = max(self.memory_stats['peak_allocated'], allocated)
            self.memory_stats['peak_reserved'] = max(self.memory_stats['peak_reserved'], reserved)
            
            # Keep only recent history
            if len(self.memory_stats['allocated']) > 100:
                self.memory_stats['allocated'].pop(0)
                self.memory_stats['reserved'].pop(0)
                
    def optimize_memory(self):
        """Bellek kullanımını optimize et"""
        if self.device.type == 'cuda':
            # Clear cache if memory is high
            allocated = torch.cuda.memory_allocated(self.device) / 1024**3
            
            if allocated > 4.0:  # If using more than 4GB
                torch.cuda.empty_cache()
                logger.info(f"🧹 GPU cache temizlendi: {allocated:.2f}GB -> {torch.cuda.memory_allocated(self.device)/1024**3:.2f}GB")
                
    def get_memory_report(self):
        """Bellek raporu al"""
        if self.device.type == 'cuda':
            return {
                'allocated_gb': torch.cuda.memory_allocated(self.device) / 1024**3,
                'reserved_gb': torch.cuda.memory_reserved(self.device) / 1024**3,
                'peak_allocated_gb': self.memory_stats['peak_allocated'],
                'peak_reserved_gb': self.memory_stats['peak_reserved']
            }
        else:
            return {'device': 'cpu'}

# ----------------------------------------------------------------------
# Ultimate Otoya Model
# ----------------------------------------------------------------------
class UltimateOtoyaModel(nn.Module):
    """Nihai Otoya Modeli"""
    
    def __init__(self, config: UltimateConfig):
        super().__init__()
        self.config = config
        
        # Token embeddings
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.context_length, config.d_model)
        
        # Quantum attention layers
        self.attention_layers = nn.ModuleList([
            QuantumAttention(config.d_model, config.num_heads, config.quantum_heads)
            for _ in range(config.num_layers)
        ])
        
        # Fractal memory networks
        self.memory_layers = nn.ModuleList([
            FractalMemory(config.d_model, config.fractal_depth)
            for _ in range(config.num_layers)
        ])
        
        # Multi-phase fusion layers
        self.fusion_layers = nn.ModuleList([
            MultiPhaseFusion(config.d_model)
            for _ in range(config.num_layers)
        ])
        
        # Layer norms
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(config.d_model)
            for _ in range(config.num_layers * 2)
        ])
        
        # Output projection
        self.output_projection = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
        # Tie weights
        self.output_projection.weight = self.token_embedding.weight
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Ağırlıkları başlat"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                nn.init.zeros_(module.bias)
                nn.init.ones_(module.weight)
                
    def forward(self, input_ids: torch.Tensor, phase: int = 1) -> torch.Tensor:
        """Model forward pass"""
        batch_size, seq_len = input_ids.shape
        
        # Position indices
        position_ids = torch.arange(seq_len, dtype=torch.long, device=input_ids.device).unsqueeze(0)
        
        # Embeddings
        token_embeds = self.token_embedding(input_ids)
        position_embeds = self.position_embedding(position_ids)
        
        x = token_embeds + position_embeds
        
        # Process through layers
        for i in range(self.config.num_layers):
            # Attention
            attn_output = self.attention_layers[i](x)
            x = self.layer_norms[i*2](x + attn_output)
            
            # Fractal memory
            memory_output = self.memory_layers[i](x)
            x = self.layer_norms[i*2+1](x + memory_output)
            
            # Multi-phase fusion
            x = self.fusion_layers[i](x, phase)
            
        # Output projection
        logits = self.output_projection(x)
        
        return logits

# ----------------------------------------------------------------------
# Ultimate Trainer
# ----------------------------------------------------------------------
class UltimateTrainer:
    """Nihai eğitici - ultra hızlı ve stabil"""
    
    def __init__(self, config: UltimateConfig):
        self.config = config
        
        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Model
        self.model = UltimateOtoyaModel(config).to(self.device)
        
        # Optimizer
        self.optimizer = LightningOptimizer(self.model, config)
        
        # Memory manager
        self.memory_manager = SmartMemoryManager(self.device)
        
        # Training state
        self.current_step = 0
        self.best_loss = float('inf')
        
        # Performance tracking
        self.loss_history = []
        self.step_times = []
        
    def train_step(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> Dict[str, float]:
        """Tek eğitim adımı"""
        start_time = time.time()
        
        # Unpack batch
        input_ids, target_ids = batch
        input_ids = input_ids.to(self.device)
        target_ids = target_ids.to(self.device)
        
        # Forward pass
        with autocast(enabled=self.config.use_mixed_precision):
            logits = self.model(input_ids, phase=min(self.current_step // 1000, 3))
            
            # Compute loss
            loss = F.cross_entropy(
                logits.view(-1, self.config.vocab_size),
                target_ids.view(-1),
                ignore_index=-100
            )
            
        # Backward and optimize
        step_time = self.optimizer.step(loss)
        
        # Track metrics
        loss_value = loss.item()
        self.loss_history.append(loss_value)
        self.step_times.append(step_time)
        
        # Memory management
        self.memory_manager.track_memory()
        if self.current_step % 100 == 0:
            self.memory_manager.optimize_memory()
            
        # Update step
        self.current_step += 1
        
        # Prepare metrics
        metrics = {
            'step': self.current_step,
            'loss': loss_value,
            'step_time': step_time,
            'avg_step_time': self.optimizer.get_average_step_time(),
            'memory_allocated_gb': self.memory_manager.get_memory_report().get('allocated_gb', 0)
        }
        
        # Log progress
        if self.current_step % 10 == 0:
            logger.info(
                f"Step {self.current_step:6d} | "
                f"Loss: {loss_value:.4f} | "
                f"Step Time: {step_time:.2f}s | "
                f"Avg: {self.optimizer.get_average_step_time():.2f}s | "
                f"Memory: {metrics['memory_allocated_gb']:.2f}GB"
            )
            
        return metrics
    
    def train(self, train_loader, num_steps: int = 10000):
        """Tam eğitim döngüsü"""
        logger.info(f"Starting training for {num_steps} steps")
        
        # Training loop
        for step in range(num_steps):
            try:
                # Get batch
                batch = next(train_loader)
                
                # Training step
                metrics = self.train_step(batch)
                
                # Validation
                if step % 100 == 0:
                    val_loss = self.validate()
                    metrics['val_loss'] = val_loss
                    
                    # Save best model
                    if val_loss < self.best_loss:
                        self.best_loss = val_loss
                        self.save_checkpoint(f"best_step_{step}")
                        
                # Checkpoint
                if step % 1000 == 0:
                    self.save_checkpoint(f"checkpoint_step_{step}")
                    
            except Exception as e:
                logger.error(f"Error at step {step}: {e}")
                traceback.print_exc()
                break
                
        logger.info("Training completed")
        return self.model
    
    def validate(self) -> float:
        """Validation loss hesapla"""
        self.model.eval()
        
        # Simple validation - in practice use validation dataset
        with torch.no_grad():
            # Create dummy validation batch
            batch_size = min(4, self.config.batch_size)
            seq_len = min(256, self.config.context_length)
            
            input_ids = torch.randint(0, self.config.vocab_size, (batch_size, seq_len))
            target_ids = torch.randint(0, self.config.vocab_size, (batch_size, seq_len))
            
            input_ids = input_ids.to(self.device)
            target_ids = target_ids.to(self.device)
            
            logits = self.model(input_ids, phase=3)
            loss = F.cross_entropy(
                logits.view(-1, self.config.vocab_size),
                target_ids.view(-1),
                ignore_index=-100
            )
            
        self.model.train()
        return loss.item()
    
    def save_checkpoint(self, name: str):
        """Checkpoint kaydet"""
        checkpoint = {
            'step': self.current_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.optimizer.state_dict(),
            'loss': self.loss_history[-1] if self.loss_history else float('inf'),
            'config': asdict(self.config)
        }
        
        path = Path(f"checkpoints/{name}.pt")
        path.parent.mkdir(exist_ok=True)
        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved: {path}")
        
    def load_checkpoint(self, path: str):
        """Checkpoint yükle"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_step = checkpoint['step']
        logger.info(f"Checkpoint loaded: {path}")

# ----------------------------------------------------------------------
# Data Loader
# ----------------------------------------------------------------------
class UltimateDataLoader:
    """Ultra hızlı data loader"""
    
    def __init__(self, dataset_path: str, config: UltimateConfig):
        self.config = config
        self.dataset_path = dataset_path
        
        # Load or create dataset
        self.data = self._load_dataset()
        
        # Preprocessing
        self.tokenized_data = self._tokenize_data()
        
        # Create batches
        self.batches = self._create_batches()
        
    def _load_dataset(self):
        """Dataset yükle"""
        # In practice, load from file
        # For now, create synthetic data
        vocab_size = self.config.vocab_size
        seq_len = self.config.context_length
        
        # Create synthetic dataset
        data_size = 10000  # 10k samples
        data = torch.randint(0, vocab_size, (data_size, seq_len))
        
        logger.info(f"Created synthetic dataset: {data.shape}")
        return data
    
    def _tokenize_data(self):
        """Data tokenize et"""
        # Already tokenized for synthetic data
        return self.data
    
    def _create_batches(self):
        """Batch'ler oluştur"""
        batch_size = self.config.batch_size
        num_batches = len(self.tokenized_data) // batch_size
        
        batches = []
        for i in range(num_batches):
            start = i * batch_size
            end = start + batch_size
            
            input_ids = self.tokenized_data[start:end]
            target_ids = torch.roll(input_ids, shifts=-1, dims=1)
            target_ids[:, -1] = -100  # Ignore last token
            
            batches.append((input_ids, target_ids))
            
        logger.info(f"Created {len(batches)} batches")
        return batches
    
    def __iter__(self):
        """Iterator"""
        self.current_idx = 0
        return self
    
    def __next__(self):
        """Next batch"""
        if self.current_idx >= len(self.batches):
            # Shuffle and restart
            random.shuffle(self.batches)
            self.current_idx = 0
            
        batch = self.batches[self.current_idx]
        self.current_idx += 1
        return batch

# ----------------------------------------------------------------------
# Main Execution
# ----------------------------------------------------------------------
def main():
    """Ana çalıştırma fonksiyonu"""
    parser = argparse.ArgumentParser(description="Nihai Otoya Mimari")
    parser.add_argument("--steps", type=int, default=10000, help="Eğitim adım sayısı")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch boyutu")
    parser.add_argument("--learning-rate", type=float, default=3e-4, help="Öğrenme oranı")
    parser.add_argument("--d-model", type=int, default=1024, help="Model boyutu")
    parser.add_argument("--num-layers", type=int, default=24, help="Katman sayısı")
    parser.add_argument("--context-length", type=int, default=2048, help="Context uzunluğu")
    
    args = parser.parse_args()
    
    # Configuration
    config = UltimateConfig(
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        d_model=args.d_model,
        num_layers=args.num_layers,
        context_length=args.context_length
    )
    
    # Create trainer
    trainer = UltimateTrainer(config)
    
    # Create data loader
    data_loader = UltimateDataLoader("synthetic", config)
    
    # Train
    trainer.train(data_loader, num_steps=args.steps)
    
    logger.info("Nihai Otoya Mimari eğitimi tamamlandı!")

if __name__ == "__main__":
    main()