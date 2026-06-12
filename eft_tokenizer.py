# ============================================================
# EFT (Eita Fourier Tokenizer) - Holographic Tokenizer
# ============================================================
"""
EFT decomposes text into three frequency components:
- High Frequency (Red): Orthographic features (character n-grams, syllables)
- Medium Frequency (Green): Morphological features (roots, stems, affixes)
- Low Frequency (Blue): Contextual features (POS, semantic category)

These three components are combined into a single "holographic vector"
that provides the model with rich, multi-dimensional linguistic information.
"""

import re
import numpy as np
from collections import defaultdict
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import json
from pathlib import Path


@dataclass
class EFTConfig:
    """Configuration for EFT Tokenizer"""
    vocab_size: int = 258  # Base vocabulary size (compatible with ByteTokenizer)
    freq_dim: int = 32  # Dimension for each frequency component
    max_char_ngram: int = 3  # Maximum character n-gram size
    max_syllable_len: int = 5  # Maximum syllable length
    use_morphology: bool = True  # Whether to use morphological analysis
    use_context: bool = True  # Whether to use contextual features
    language: str = "auto"  # Language: 'auto', 'tr', 'en', etc.


class FrequencyPrism:
    """
    The "prism" that splits text into three frequency components.
    Language-agnostic - works with any language.
    """
    
    def __init__(self, config: EFTConfig):
        self.config = config
        self.char_ngram_vocab = self._build_char_ngram_vocab()
        self.syllable_vocab = self._build_syllable_vocab()
        
    def _build_char_ngram_vocab(self) -> Dict[str, int]:
        """Build vocabulary for character n-grams (high frequency)"""
        vocab = {}
        idx = 2  # Reserve 0 for pad, 1 for eos
        
        # Single characters
        for i in range(256):
            vocab[chr(i)] = idx
            idx += 1
            
        # Character n-grams (bigrams, trigrams)
        for n in range(2, self.config.max_char_ngram + 1):
            # Common n-grams will be learned during training
            pass
            
        return vocab
    
    def _build_syllable_vocab(self) -> Dict[str, int]:
        """Build vocabulary for syllable patterns (high frequency)"""
        vocab = {}
        idx = 2
        # Common syllable patterns (CV, CVC, etc.)
        patterns = ['CV', 'CVC', 'VC', 'V', 'CCV', 'CCVC', 'CVCC']
        for pattern in patterns:
            vocab[pattern] = idx
            idx += 1
        return vocab
    
    def extract_high_freq(self, word: str) -> np.ndarray:
        """
        Extract high-frequency component: orthographic features
        - Character n-grams
        - Syllable patterns
        - Case information
        """
        features = np.zeros(self.config.freq_dim, dtype=np.float32)
        
        # Character n-gram features
        for n in range(1, min(self.config.max_char_ngram + 1, len(word) + 1)):
            for i in range(len(word) - n + 1):
                ngram = word[i:i+n]
                # Hash to feature index
                idx = hash(ngram) % self.config.freq_dim
                features[idx] += 1.0
                
        # Normalize
        if features.sum() > 0:
            features = features / features.sum()
            
        return features
    
    def extract_medium_freq(self, word: str, context: List[str]) -> np.ndarray:
        """
        Extract medium-frequency component: morphological features
        - Word roots/stems
        - Affixes (prefixes, suffixes)
        - Morphological patterns
        """
        features = np.zeros(self.config.freq_dim, dtype=np.float32)
        
        if not self.config.use_morphology:
            return features
            
        # Simple morphological analysis (language-agnostic)
        # Extract potential prefixes/suffixes
        if len(word) > 3:
            # Prefix (first 2 chars)
            prefix = word[:2]
            idx = hash(prefix) % self.config.freq_dim
            features[idx] += 0.5
            
            # Suffix (last 2 chars)
            suffix = word[-2:]
            idx = hash(suffix) % self.config.freq_dim
            features[idx] += 0.5
            
            # Stem (remove common suffixes)
            common_suffixes = ['ing', 'ed', 'tion', 'ness', 'ment', 'ity', 
                             'yor', 'iyor', 'mek', 'mak', 'lar', 'ler']
            for suf in common_suffixes:
                if word.endswith(suf):
                    stem = word[:-len(suf)]
                    idx = hash(stem) % self.config.freq_dim
                    features[idx] += 1.0
                    break
                    
        # Normalize
        if features.sum() > 0:
            features = features / features.sum()
            
        return features
    
    def extract_low_freq(self, word: str, context: List[str], position: int) -> np.ndarray:
        """
        Extract low-frequency component: contextual features
        - Part-of-speech hints
        - Semantic category
        - Position in sentence
        """
        features = np.zeros(self.config.freq_dim, dtype=np.float32)
        
        if not self.config.use_context:
            return features
            
        # Position encoding
        features[0] = position / 100.0  # Normalized position
        
        # Capitalization (sentence start)
        if word and word[0].isupper():
            features[1] = 1.0
            
        # Length feature
        features[2] = len(word) / 20.0  # Normalized length
        
        # Punctuation hints
        if word.endswith(('.', '!', '?')):
            features[3] = 1.0
        elif word.endswith((',', ';', ':')):
            features[4] = 1.0
            
        # Normalize
        if features.sum() > 0:
            features = features / (features.sum() + 1e-8)
            
        return features
    
    def decompose(self, word: str, context: List[str], position: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Decompose a word into three frequency components.
        
        Returns:
            (high_freq, medium_freq, low_freq) - three feature vectors
        """
        high = self.extract_high_freq(word)
        medium = self.extract_medium_freq(word, context)
        low = self.extract_low_freq(word, context, position)
        
        return high, medium, low


class HolographicEmbedding:
    """
    Combines three frequency components into a single "holographic vector".
    This replaces the simple embedding lookup with a rich, multi-dimensional representation.
    """
    
    def __init__(self, base_vocab_size: int, freq_dim: int):
        self.base_vocab_size = base_vocab_size
        self.freq_dim = freq_dim
        
        # Initialize frequency embeddings (will be learned)
        # These act as "prism coefficients" for each frequency band
        self.high_freq_embedding = np.random.randn(base_vocab_size, freq_dim) * 0.02
        self.med_freq_embedding = np.random.randn(base_vocab_size, freq_dim) * 0.02
        self.low_freq_embedding = np.random.randn(base_vocab_size, freq_dim) * 0.02
        
        # Fusion weights (learnable)
        self.fusion_weights = np.ones(3) / 3.0  # Equal weighting initially
        
    def combine(self, token_id: int, high_feat: np.ndarray, 
                med_feat: np.ndarray, low_feat: np.ndarray) -> np.ndarray:
        """
        Combine frequency features with learned embeddings to create holographic vector.
        
        Args:
            token_id: Base token ID (from ByteTokenizer)
            high_feat: High-frequency feature vector
            med_feat: Medium-frequency feature vector
            low_feat: Low-frequency feature vector
            
        Returns:
            Combined holographic vector
        """
        # Get learned embeddings for this token
        high_emb = self.high_freq_embedding[token_id]
        med_emb = self.med_freq_embedding[token_id]
        low_emb = self.low_freq_embedding[token_id]
        
        # Element-wise multiplication with features (attention mechanism)
        high_combined = high_emb * high_feat
        med_combined = med_emb * med_feat
        low_combined = low_emb * low_feat
        
        # Weighted sum
        holographic = (self.fusion_weights[0] * high_combined +
                      self.fusion_weights[1] * med_combined +
                      self.fusion_weights[2] * low_combined)
        
        return holographic


class EFTTokenizer:
    """
    Eita Fourier Tokenizer - Holographic Tokenizer
    
    Replaces ByteTokenizer with a frequency-aware tokenizer that provides
    rich linguistic information to the model.
    """
    
    pad_token_id = 0
    eos_token_id = 1
    vocab_size = 258  # Compatible with ByteTokenizer
    
    def __init__(self, config: Optional[EFTConfig] = None):
        self.config = config or EFTConfig()
        self.prism = FrequencyPrism(self.config)
        self.holographic = HolographicEmbedding(self.vocab_size, self.config.freq_dim)
        
        # For compatibility with existing code
        self._base_tokenizer = ByteTokenizer()
        
    def encode(self, text: str, add_special: bool = True) -> List[int]:
        """
        Encode text to token IDs.
        For compatibility, returns the same format as ByteTokenizer.
        The holographic features are computed separately during training.
        """
        return self._base_tokenizer.encode(text, add_special)
    
    def decode(self, ids: List[int]) -> str:
        """Decode token IDs to text."""
        return self._base_tokenizer.decode(ids)
    
    def encode_with_features(self, text: str, add_special: bool = True) -> Tuple[List[int], List[Tuple[np.ndarray, np.ndarray, np.ndarray]]]:
        """
        Encode text and return both token IDs and holographic features.
        
        Returns:
            (token_ids, features_list) where features_list contains
            (high_freq, med_freq, low_freq) tuples for each token
        """
        # Split into words
        words = re.findall(r'\w+|[^\w\s]', text)
        token_ids = []
        features_list = []
        
        for position, word in enumerate(words):
            # Get base token ID (using ByteTokenizer for compatibility)
            word_tokens = self._base_tokenizer.encode(word, add_special=False)
            if word_tokens:
                token_id = word_tokens[0]
                token_ids.append(token_id)
                
                # Extract holographic features
                high, med, low = self.prism.decompose(word, words, position)
                features_list.append((high, med, low))
        
        if add_special:
            token_ids.append(self.eos_token_id)
            features_list.append((np.zeros(self.config.freq_dim),
                                 np.zeros(self.config.freq_dim),
                                 np.zeros(self.config.freq_dim)))
        
        return token_ids, features_list
    
    def get_holographic_embedding(self, token_id: int, 
                                 high_feat: np.ndarray,
                                 med_feat: np.ndarray,
                                 low_feat: np.ndarray) -> np.ndarray:
        """
        Get holographic embedding for a token.
        This replaces the simple embedding lookup in the model.
        """
        return self.holographic.combine(token_id, high_feat, med_feat, low_feat)
    
    def save(self, out_dir):
        """Save tokenizer configuration and learned parameters."""
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        
        # Save config
        config_dict = {
            "type": "eft",
            "vocab_size": self.vocab_size,
            "freq_dim": self.config.freq_dim,
            "max_char_ngram": self.config.max_char_ngram,
            "use_morphology": self.config.use_morphology,
            "use_context": self.config.use_context,
            "language": self.config.language,
            "pad_token_id": self.pad_token_id,
            "eos_token_id": self.eos_token_id
        }
        
        (out_path / "tokenizer.json").write_text(json.dumps(config_dict, indent=2))
        
        # Save learned embeddings
        np.save(out_path / "high_freq_embedding.npy", self.holographic.high_freq_embedding)
        np.save(out_path / "med_freq_embedding.npy", self.holographic.med_freq_embedding)
        np.save(out_path / "low_freq_embedding.npy", self.holographic.low_freq_embedding)
        np.save(out_path / "fusion_weights.npy", self.holographic.fusion_weights)
    
    def load(self, in_dir):
        """Load tokenizer configuration and learned parameters."""
        in_path = Path(in_dir)
        
        # Load config
        config_path = in_path / "tokenizer.json"
        if config_path.exists():
            config_dict = json.loads(config_path.read_text())
            self.config = EFTConfig(**{k: v for k, v in config_dict.items() 
                                      if k in EFTConfig.__dataclass_fields__})
        
        # Load learned embeddings
        if (in_path / "high_freq_embedding.npy").exists():
            self.holographic.high_freq_embedding = np.load(in_path / "high_freq_embedding.npy")
            self.holographic.med_freq_embedding = np.load(in_path / "med_freq_embedding.npy")
            self.holographic.low_freq_embedding = np.load(in_path / "low_freq_embedding.npy")
            self.holographic.fusion_weights = np.load(in_path / "fusion_weights.npy")


class ByteTokenizer:
    """Original ByteTokenizer for compatibility."""
    pad_token_id = 0
    eos_token_id = 1
    vocab_size = 258
    
    def encode(self, text, add_special=True):
        ids = [b+2 for b in text.encode("utf-8", "replace")]
        if add_special:
            ids.append(1)
        return ids
    
    def decode(self, ids):
        return bytes(max(0, min(255, i-2)) for i in ids if i >= 2).decode("utf-8", "replace")
    
    def save(self, out):
        (Path(out) / "tokenizer.json").write_text(
            json.dumps({"type": "byte", "vocab_size": 258, 
                       "pad_token_id": 0, "eos_token_id": 1}, indent=2))


# ============================================================
# Test / Demo
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("EFT (Eita Fourier Tokenizer) - Holographic Tokenizer Demo")
    print("=" * 60)
    
    # Create tokenizer
    config = EFTConfig(freq_dim=32, use_morphology=True, use_context=True)
    tokenizer = EFTTokenizer(config)
    
    # Test text (Turkish and English)
    test_texts = [
        "Merhaba dünya! Geliyorum.",
        "Hello world! I am coming.",
        "The quick brown fox jumps over the lazy dog."
    ]
    
    for text in test_texts:
        print(f"\nText: {text}")
        
        # Encode with features
        token_ids, features = tokenizer.encode_with_features(text)
        
        print(f"Token IDs: {token_ids[:10]}...")
        print(f"Number of tokens: {len(token_ids)}")
        print(f"Features shape: {len(features)} x 3 x {config.freq_dim}")
        
        # Show holographic embedding for first token
        if token_ids and features:
            high, med, low = features[0]
            holographic = tokenizer.get_holographic_embedding(token_ids[0], high, med, low)
            print(f"First token holographic embedding shape: {holographic.shape}")
            print(f"First token holographic embedding (sample): {holographic[:5]}")
    
    print("\n" + "=" * 60)
    print("EFT Tokenizer demo completed successfully!")
    print("=" * 60)
