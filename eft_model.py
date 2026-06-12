# ============================================================
# EFT-Enhanced Otoya Model
# ============================================================
"""
Enhanced Otoya model with Holographic Embedding support.
This integrates the EFT tokenizer with the existing architecture.
"""

from tinygrad import Tensor, nn
from tinygrad.nn.state import get_parameters
import numpy as np
from typing import Optional, Tuple
from dataclasses import dataclass

from eita import EitaConfig, EitaModel, OtoyaBlock, finite_clip, cosine_flat, aggregate_stats


@dataclass
class EFTEmbeddingConfig:
    """Configuration for holographic embedding layer"""
    base_vocab_size: int = 258
    freq_dim: int = 32
    use_holographic: bool = True
    learnable_fusion: bool = True


class HolographicEmbeddingLayer:
    """
    Tinygrad-compatible holographic embedding layer.
    Replaces the simple nn.Embedding with frequency-aware embeddings.
    """
    
    def __init__(self, cfg: EFTEmbeddingConfig, d_model: int):
        self.cfg = cfg
        self.d_model = d_model
        
        # Base token embedding (for compatibility)
        self.base_embedding = nn.Embedding(cfg.base_vocab_size, d_model)
        
        # Frequency-specific embeddings
        self.high_freq_proj = nn.Linear(cfg.freq_dim, d_model)
        self.med_freq_proj = nn.Linear(cfg.freq_dim, d_model)
        self.low_freq_proj = nn.Linear(cfg.freq_dim, d_model)
        
        # Learnable fusion weights
        if cfg.learnable_fusion:
            self.fusion_weights = Tensor.ones(3) / 3.0
        else:
            self.fusion_weights = Tensor([0.33, 0.33, 0.34])
    
    def __call__(self, token_ids: Tensor, 
                 high_feat: Optional[Tensor] = None,
                 med_feat: Optional[Tensor] = None,
                 low_feat: Optional[Tensor] = None) -> Tensor:
        """
        Forward pass with holographic features.
        
        Args:
            token_ids: Token IDs [batch, seq]
            high_feat: High-frequency features [batch, seq, freq_dim]
            med_feat: Medium-frequency features [batch, seq, freq_dim]
            low_feat: Low-frequency features [batch, seq, freq_dim]
            
        Returns:
            Combined embeddings [batch, seq, d_model]
        """
        # Base embedding
        base_emb = self.base_embedding(token_ids)
        
        if not self.cfg.use_holographic or high_feat is None:
            return base_emb
        
        # Project frequency features to d_model space
        high_emb = self.high_freq_proj(high_feat)
        med_emb = self.med_freq_proj(med_feat)
        low_emb = self.low_freq_proj(low_feat)
        
        # Normalize fusion weights
        weights = self.fusion_weights.softmax(0)
        
        # Combine embeddings
        holographic_emb = (weights[0] * high_emb + 
                          weights[1] * med_emb + 
                          weights[2] * low_emb)
        
        # Residual connection with base embedding
        combined = base_emb + 0.3 * holographic_emb
        
        return combined


class EFTEitaModel:
    """
    Eita Model enhanced with Holographic Embeddings.
    Compatible with existing training pipeline.
    """
    
    def __init__(self, cfg: EitaConfig, eft_cfg: Optional[EFTEmbeddingConfig] = None):
        self.cfg = cfg
        self.eft_cfg = eft_cfg or EFTEmbeddingConfig()
        
        # Holographic embedding layer
        self.tok = HolographicEmbeddingLayer(self.eft_cfg, cfg.d_model)
        
        # Position embedding (unchanged)
        self.pos = nn.Embedding(cfg.context_length, cfg.d_model)
        
        # Otoya blocks (unchanged)
        self.layers = [OtoyaBlock(cfg) for _ in range(cfg.num_layers)]
        self.norm = nn.RMSNorm(cfg.d_model)
        
        # For generation, we need the base embedding weight
        self._base_embedding = nn.Embedding(cfg.vocab_size, cfg.d_model)
    
    def reset_cache(self):
        """Reset KV cache in all attention layers."""
        for layer in self.layers:
            layer.attn.reset_cache()
    
    def __call__(self, ids, targets, phase, strength, 
                 high_feat=None, med_feat=None, low_feat=None,
                 use_cache=False, start_pos=0):
        """
        Forward pass with optional holographic features.
        
        Args:
            ids: Token IDs [batch, seq]
            targets: Target token IDs [batch, seq]
            phase: Current training phase
            strength: Phase strength
            high_feat: High-frequency features [batch, seq, freq_dim]
            med_feat: Medium-frequency features [batch, seq, freq_dim]
            low_feat: Low-frequency features [batch, seq, freq_dim]
            use_cache: Whether to use KV cache
            start_pos: Starting position for generation
            
        Returns:
            logits, loss, stats
        """
        bsz, seq = ids.shape
        pos = Tensor.arange(seq).reshape(1, seq) + start_pos
        
        # Holographic embedding
        x = self.tok(ids, high_feat, med_feat, low_feat) + self.pos(pos)
        
        stats = []
        for l in self.layers:
            x, st = l(x, phase, strength, use_cache=use_cache, start_pos=start_pos)
            stats.append(st)
        
        # Use base embedding weight for logits (for compatibility)
        logits = finite_clip(self.norm(x).matmul(self._base_embedding.weight.T), 30)
        
        loss = None
        if targets is not None:
            loss = logits.reshape(bsz*seq, self.cfg.vocab_size).sparse_categorical_crossentropy(
                targets.reshape(bsz*seq))
        
        return logits, loss, aggregate_stats(stats)
    
    def generate_gpu(self, prompt_ids, max_new=200, temp=0.8, top_k=20, use_cache=True):
        """
        Generation with holographic embeddings.
        For generation, we fall back to base embeddings (no frequency features).
        """
        was_training = Tensor.training
        Tensor.training = False
        self.reset_cache()
        try:
            with Tensor.no_grad():
                ctx = self.cfg.context_length
                ids = list(prompt_ids[-ctx:])
                generated = []
                
                # Process prompt with base embeddings
                inp = np.array([ids[-ctx:]], dtype=np.int32)
                x_t = Tensor(inp)
                logits, _, _ = self(x_t, None, 4, 1.0, use_cache=use_cache, start_pos=0)
                
                curr_pos = x_t.shape[1]
                
                for _ in range(max_new):
                    if curr_pos >= self.cfg.context_length:
                        break
                    
                    inp = np.array([[ids[-1]]], dtype=np.int32)
                    x_t = Tensor(inp)
                    logits, _, _ = self(x_t, None, 4, 1.0, use_cache=use_cache, start_pos=curr_pos)
                    
                    curr_pos += 1
                    
                    last = logits[0, -1, :]
                    if temp > 0:
                        last = last / temp
                    if top_k > 0:
                        vals = last.numpy()
                        kth = np.partition(vals, -top_k)[-top_k]
                        vals = np.where(vals < kth, -1e9, vals)
                        last = Tensor(vals.astype(np.float32))
                    probs = last.softmax(0).numpy()
                    if np.any(np.isnan(probs)) or np.any(np.isinf(probs)):
                        probs = np.ones(len(probs)) / len(probs)
                    probs = np.clip(probs, 0, None)
                    probs /= probs.sum() + 1e-9
                    next_id = int(np.random.choice(len(probs), p=probs))
                    ids.append(next_id)
                    generated.append(next_id)
                    if next_id == 1:
                        break
            return generated
        finally:
            Tensor.training = was_training


class EFTStreamingDataset:
    """
    Streaming dataset that precomputes holographic features.
    This enhances the existing StreamingDataset with EFT support.
    """
    
    def __init__(self, tokenizer, seq_len, dataset, custom_path=None, 
                 prefetch_size=2, use_eft=True):
        from eita import StreamingDataset
        self.seq_len = seq_len
        self.tokenizer = tokenizer
        self.use_eft = use_eft
        self.prefetch_size = prefetch_size
        
        # Use base dataset for token IDs
        self.base_dataset = StreamingDataset(tokenizer._base_tokenizer if use_eft else tokenizer,
                                            seq_len, dataset, custom_path, prefetch_size)
        
        # EFT feature cache (optional - can be computed on-the-fly)
        self.eft_cache = None
        if use_eft:
            self._build_eft_cache(dataset, custom_path)
    
    def _build_eft_cache(self, dataset, custom_path):
        """Build cache of holographic features."""
        # For now, compute features on-the-fly during training
        # This can be optimized with caching later
        pass
    
    def batch(self, batch_size):
        """Get a batch with optional holographic features."""
        x, y = self.base_dataset.batch(batch_size)
        
        if not self.use_eft:
            return x, y, None, None, None
        
        # Compute holographic features on-the-fly
        # This is a simplified version - in production, precompute and cache
        batch_size, seq_len = x.shape
        freq_dim = self.tokenizer.config.freq_dim
        
        high_feat = np.zeros((batch_size, seq_len, freq_dim), dtype=np.float32)
        med_feat = np.zeros((batch_size, seq_len, freq_dim), dtype=np.float32)
        low_feat = np.zeros((batch_size, seq_len, freq_dim), dtype=np.float32)
        
        # For each sequence in batch
        for i in range(batch_size):
            # Decode token IDs to text
            text = self.tokenizer.decode(x[i].tolist())
            
            # Extract features
            _, features = self.tokenizer.encode_with_features(text, add_special=False)
            
            # Pad/truncate to seq_len
            for j in range(min(seq_len, len(features))):
                high, med, low = features[j]
                high_feat[i, j] = high
                med_feat[i, j] = med
                low_feat[i, j] = low
        
        return x, y, high_feat, med_feat, low_feat


# ============================================================
# Integration Helper
# ============================================================
def create_eft_model(cfg: EitaConfig, eft_cfg: Optional[EFTEmbeddingConfig] = None):
    """
    Create an EFT-enhanced Eita model.
    
    Args:
        cfg: EitaConfig for the base model
        eft_cfg: Optional EFTEmbeddingConfig for holographic embeddings
        
    Returns:
        EFTEitaModel instance
    """
    return EFTEitaModel(cfg, eft_cfg)


def test_eft_integration():
    """Test EFT integration with the model."""
    print("=" * 60)
    print("EFT Integration Test")
    print("=" * 60)
    
    # Create configs
    eita_cfg = EitaConfig(d_model=128, num_layers=2, num_heads=4, 
                         context_length=128, vocab_size=258)
    eft_cfg = EFTEmbeddingConfig(base_vocab_size=258, freq_dim=32, 
                                 use_holographic=True, learnable_fusion=True)
    
    # Create model
    model = create_eft_model(eita_cfg, eft_cfg)
    print("✓ EFT-enhanced model created")
    
    # Test forward pass
    batch_size = 2
    seq_len = 128
    token_ids = Tensor(np.random.randint(0, 258, (batch_size, seq_len), dtype=np.int32))
    targets = Tensor(np.random.randint(0, 258, (batch_size, seq_len), dtype=np.int32))
    
    # Create holographic features
    high_feat = Tensor(np.random.randn(batch_size, seq_len, 32).astype(np.float32))
    med_feat = Tensor(np.random.randn(batch_size, seq_len, 32).astype(np.float32))
    low_feat = Tensor(np.random.randn(batch_size, seq_len, 32).astype(np.float32))
    
    # Forward pass
    logits, loss, stats = model(token_ids, targets, phase=1, strength=0.0,
                                high_feat=high_feat, med_feat=med_feat, low_feat=low_feat)
    
    print(f"✓ Forward pass successful")
    print(f"  Logits shape: {logits.shape}")
    print(f"  Loss: {loss}")
    print(f"  Stats keys: {list(stats.keys())}")
    
    # Test without holographic features (fallback to base embeddings)
    logits2, loss2, stats2 = model(token_ids, targets, phase=1, strength=0.0)
    print(f"✓ Fallback to base embeddings successful")
    print(f"  Logits shape: {logits2.shape}")
    print(f"  Loss: {loss2}")
    
    print("\n" + "=" * 60)
    print("EFT integration test completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    test_eft_integration()
