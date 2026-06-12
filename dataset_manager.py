#!/usr/bin/env python3
"""
Ultimate Otoya Dataset Manager
Multiple dataset selection, mixing strategies, and format support
"""

import os
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union, Any
from dataclasses import dataclass, field
from enum import Enum
import random
from collections import defaultdict, Counter
import time

from quantum_tensor import QuantumTensor

class DatasetFormat(Enum):
    """Supported dataset formats"""
    TXT = "txt"
    JSON = "json"
    JSONL = "jsonl"
    TOKENS_BIN = "tokens.bin"

class DatasetPurpose(Enum):
    """Dataset purposes for mixing strategy"""
    CONVERSATION = "conversation"  # Ayrı konuşma dataset'i
    KNOWLEDGE = "knowledge"        # Neler bilecek dataset'i
    CODE = "code"                  # Kod dataset'i
    GENERAL = "general"            # Genel dataset

@dataclass
class DatasetInfo:
    """Information about a dataset"""
    name: str
    path: str
    format: DatasetFormat
    purpose: DatasetPurpose
    size_bytes: int = 0
    num_samples: int = 0
    avg_length: float = 0.0
    language: str = "tr"  # Default Turkish
    description: str = ""
    tags: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Calculate dataset statistics"""
        if os.path.exists(self.path):
            self.size_bytes = os.path.getsize(self.path)
            self._calculate_stats()
    
    def _calculate_stats(self):
        """Calculate dataset statistics"""
        try:
            if self.format == DatasetFormat.TXT:
                with open(self.path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    self.num_samples = len(lines)
                    if lines:
                        lengths = [len(line.strip().split()) for line in lines]
                        self.avg_length = sum(lengths) / len(lengths)
            
            elif self.format == DatasetFormat.JSONL:
                with open(self.path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    self.num_samples = len(lines)
                    if lines:
                        lengths = []
                        for line in lines:
                            try:
                                data = json.loads(line.strip())
                                text = data.get('text', '') if isinstance(data, dict) else str(data)
                                lengths.append(len(text.split()))
                            except:
                                lengths.append(0)
                        self.avg_length = sum(lengths) / len(lengths) if lengths else 0
            
            elif self.format == DatasetFormat.JSON:
                with open(self.path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        self.num_samples = len(data)
                        lengths = []
                        for item in data:
                            text = item.get('text', '') if isinstance(item, dict) else str(item)
                            lengths.append(len(text.split()))
                        self.avg_length = sum(lengths) / len(lengths) if lengths else 0
                    else:
                        self.num_samples = 1
            
            elif self.format == DatasetFormat.TOKENS_BIN:
                # Binary token files
                self.num_samples = self.size_bytes // 4  # Assuming int32 tokens
                self.avg_length = 128  # Default assumption
        
        except Exception as e:
            print(f"⚠️ Error calculating stats for {self.name}: {e}")
    
    def get_info_dict(self) -> Dict:
        """Get dataset info as dictionary"""
        return {
            'name': self.name,
            'path': self.path,
            'format': self.format.value,
            'purpose': self.purpose.value,
            'size_bytes': self.size_bytes,
            'size_mb': self.size_bytes / (1024**2),
            'num_samples': self.num_samples,
            'avg_length': self.avg_length,
            'language': self.language,
            'description': self.description,
            'tags': self.tags
        }

class DatasetLoader:
    """Load datasets in different formats"""
    
    @staticmethod
    def load_txt(file_path: str, max_samples: Optional[int] = None) -> List[str]:
        """Load text file line by line"""
        texts = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    if max_samples and i >= max_samples:
                        break
                    text = line.strip()
                    if text:  # Skip empty lines
                        texts.append(text)
        except Exception as e:
            print(f"❌ Error loading TXT file {file_path}: {e}")
        
        return texts
    
    @staticmethod
    def load_jsonl(file_path: str, max_samples: Optional[int] = None) -> List[str]:
        """Load JSONL file"""
        texts = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    if max_samples and i >= max_samples:
                        break
                    try:
                        data = json.loads(line.strip())
                        
                        # Handle different JSONL formats
                        if isinstance(data, dict):
                            # Format 1: OpenAI format with 'messages'
                            if 'messages' in data:
                                messages = data['messages']
                                # Combine all messages into one text
                                combined = ""
                                for msg in messages:
                                    if 'content' in msg:
                                        combined += f"{msg['role']}: {msg['content']}\n"
                                if combined:
                                    texts.append(combined.strip())
                            
                            # Format 2: Simple 'text' field
                            elif 'text' in data:
                                text = data['text']
                                if text:
                                    texts.append(text)
                            
                            # Format 3: Try other common keys
                            else:
                                for key in ['content', 'message', 'prompt', 'response', 'question', 'answer']:
                                    if key in data:
                                        text = data[key]
                                        if text:
                                            texts.append(str(text))
                                        break
                        
                        # Format 4: Direct string in JSONL
                        elif isinstance(data, str):
                            if data.strip():
                                texts.append(data.strip())
                        
                        # Format 5: Other types
                        else:
                            text = str(data)
                            if text.strip():
                                texts.append(text.strip())
                                
                    except json.JSONDecodeError:
                        # Try to extract text from malformed JSON
                        line_text = line.strip()
                        if line_text and not line_text.startswith('{'):
                            texts.append(line_text)
                        continue
        except Exception as e:
            print(f"❌ Error loading JSONL file {file_path}: {e}")
        
        return texts
    
    @staticmethod
    def load_json(file_path: str, max_samples: Optional[int] = None) -> List[str]:
        """Load JSON file"""
        texts = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                if isinstance(data, list):
                    for i, item in enumerate(data):
                        if max_samples and i >= max_samples:
                            break
                        if isinstance(item, dict):
                            text = item.get('text', '')
                            if not text:
                                for key in ['content', 'message', 'prompt', 'response']:
                                    if key in item:
                                        text = item[key]
                                        break
                        else:
                            text = str(item)
                        
                        if text:
                            texts.append(text)
                elif isinstance(data, dict):
                    # Single document
                    text = data.get('text', '')
                    if text:
                        texts.append(text)
        except Exception as e:
            print(f"❌ Error loading JSON file {file_path}: {e}")
        
        return texts
    
    @staticmethod
    def load_tokens_bin(file_path: str, max_samples: Optional[int] = None) -> List[np.ndarray]:
        """Load binary token file"""
        tokens_list = []
        try:
            with open(file_path, 'rb') as f:
                # Read all tokens as int32
                all_tokens = np.frombuffer(f.read(), dtype=np.int32)
                
                # Assume each sample is 128 tokens
                seq_length = 128
                num_samples = len(all_tokens) // seq_length
                
                if max_samples:
                    num_samples = min(num_samples, max_samples)
                
                for i in range(num_samples):
                    start_idx = i * seq_length
                    end_idx = start_idx + seq_length
                    tokens_list.append(all_tokens[start_idx:end_idx])
        except Exception as e:
            print(f"❌ Error loading tokens.bin file {file_path}: {e}")
        
        return tokens_list
    
    @staticmethod
    def load_samples(file_path: str, format: DatasetFormat, max_samples: Optional[int] = None) -> List[Union[str, np.ndarray]]:
        """Load samples from a dataset file based on format"""
        if format == DatasetFormat.TXT:
            return DatasetLoader.load_txt(file_path, max_samples)
        elif format == DatasetFormat.JSON:
            return DatasetLoader.load_json(file_path, max_samples)
        elif format == DatasetFormat.JSONL:
            return DatasetLoader.load_jsonl(file_path, max_samples)
        elif format == DatasetFormat.TOKENS_BIN:
            return DatasetLoader.load_tokens_bin(file_path, max_samples)
        else:
            raise ValueError(f"Unsupported format: {format}")

class DatasetMixer:
    """Mix multiple datasets with different strategies"""
    
    def __init__(self, datasets: List[DatasetInfo], vocab_size: int = 50000):
        self.datasets = datasets
        self.vocab_size = vocab_size
        
        # Calculate mixing weights
        self.weights = self._calculate_mixing_weights()
        
        # Load samples from each dataset
        self.samples_by_dataset = self._load_samples()
        
        # Create mixing schedule
        self.mixing_schedule = self._create_mixing_schedule()
        
        print(f"🧩 Dataset Mixer initialized with {len(datasets)} datasets")
        for dataset, weight in zip(datasets, self.weights):
            print(f"   • {dataset.name}: {weight:.1%} ({dataset.num_samples} samples)")
    
    def _calculate_mixing_weights(self) -> List[float]:
        """Calculate mixing weights based on dataset purpose and size"""
        weights = []
        
        for dataset in self.datasets:
            # Base weight based on purpose
            if dataset.purpose == DatasetPurpose.CONVERSATION:
                base_weight = 0.4  # Highest priority for conversation
            elif dataset.purpose == DatasetPurpose.KNOWLEDGE:
                base_weight = 0.3  # High priority for knowledge
            elif dataset.purpose == DatasetPurpose.CODE:
                base_weight = 0.2  # Medium priority for code
            else:  # GENERAL
                base_weight = 0.1  # Lower priority for general
            
            # Adjust based on dataset size (more data = more weight)
            size_factor = min(1.0, dataset.num_samples / 10000)  # Cap at 10k samples
            adjusted_weight = base_weight * (0.7 + 0.3 * size_factor)
            
            weights.append(adjusted_weight)
        
        # Normalize weights to sum to 1
        total = sum(weights)
        if total > 0:
            weights = [w / total for w in weights]
        
        return weights
    
    def _load_samples(self) -> Dict[str, List]:
        """Load samples from each dataset"""
        samples_by_dataset = {}
        
        for dataset in self.datasets:
            print(f"📥 Loading {dataset.name}...")
            
            if dataset.format == DatasetFormat.TXT:
                samples = DatasetLoader.load_txt(dataset.path, max_samples=10000)
            elif dataset.format == DatasetFormat.JSONL:
                samples = DatasetLoader.load_jsonl(dataset.path, max_samples=10000)
            elif dataset.format == DatasetFormat.JSON:
                samples = DatasetLoader.load_json(dataset.path, max_samples=10000)
            elif dataset.format == DatasetFormat.TOKENS_BIN:
                samples = DatasetLoader.load_tokens_bin(dataset.path, max_samples=10000)
            else:
                samples = []
            
            samples_by_dataset[dataset.name] = samples
            print(f"   Loaded {len(samples)} samples")
        
        return samples_by_dataset
    
    def _create_mixing_schedule(self) -> List[Tuple[str, int]]:
        """Create a mixing schedule for training"""
        schedule = []
        
        # Calculate total samples per dataset based on weights
        total_samples = 100000  # Target total samples
        samples_per_dataset = []
        
        for i, dataset in enumerate(self.datasets):
            target_samples = int(total_samples * self.weights[i])
            actual_samples = len(self.samples_by_dataset[dataset.name])
            samples_to_use = min(target_samples, actual_samples)
            samples_per_dataset.append(samples_to_use)
        
        # Create interleaved schedule
        max_samples = max(samples_per_dataset)
        
        for step in range(max_samples):
            for i, dataset in enumerate(self.datasets):
                if step < samples_per_dataset[i]:
                    schedule.append((dataset.name, step))
        
        # Shuffle the schedule
        random.shuffle(schedule)
        
        return schedule
    
    def get_next_sample(self) -> Optional[str]:
        """Get next sample according to mixing schedule"""
        if not self.mixing_schedule:
            return None
        
        dataset_name, sample_idx = self.mixing_schedule.pop(0)
        samples = self.samples_by_dataset[dataset_name]
        
        if sample_idx < len(samples):
            return samples[sample_idx]
        
        return None
    
    def get_batch(self, batch_size: int, seq_length: int) -> Tuple[np.ndarray, np.ndarray]:
        """Get a batch of samples"""
        inputs = []
        targets = []
        
        for _ in range(batch_size):
            sample = self.get_next_sample()
            if sample is None:
                # If no more samples, recycle
                random.shuffle(self.mixing_schedule)
                sample = self.get_next_sample()
                if sample is None:
                    break
            
            # Handle different sample types
            if isinstance(sample, str):
                # Text sample - convert to tokens
                words = sample.split()
                tokens = [hash(word) % self.vocab_size for word in words]
            elif isinstance(sample, np.ndarray):
                # Already tokenized
                tokens = sample.tolist()
            else:
                # Convert to string
                tokens = [hash(str(sample)) % self.vocab_size]
            
            # Create sequences
            if len(tokens) >= seq_length:
                start_idx = random.randint(0, len(tokens) - seq_length)
                seq_tokens = tokens[start_idx:start_idx + seq_length]
                
                # Input is all but last token
                input_seq = seq_tokens[:-1]
                # Target is next token prediction
                target_seq = seq_tokens[1:]
                
                inputs.append(input_seq)
                targets.append(target_seq)
        
        if not inputs:
            return np.array([]), np.array([])
        
        return np.array(inputs, dtype=np.int32), np.array(targets, dtype=np.int32)

class UltimateDatasetManager:
    """Main dataset manager for Ultimate Otoya"""
    
    def __init__(self, data_archive_path: str = "data_archive"):
        self.data_archive_path = Path(data_archive_path)
        self.datasets: Dict[str, DatasetInfo] = {}
        self.vocab_size = 50000
        
        # Discover available datasets
        self._discover_datasets()
        
        # Current mixing strategy
        self.current_mixer: Optional[DatasetMixer] = None
        
        print(f"📚 Ultimate Dataset Manager initialized")
        print(f"   Found {len(self.datasets)} datasets in {self.data_archive_path}")
    
    def _discover_datasets(self):
        """Discover all datasets in data archive"""
        if not self.data_archive_path.exists():
            print(f"⚠️ Data archive path not found: {self.data_archive_path}")
            return
        
        # Common dataset patterns
        dataset_patterns = [
            ("*.txt", DatasetFormat.TXT),
            ("*.json", DatasetFormat.JSON),
            ("*.jsonl", DatasetFormat.JSONL),
            ("*.tokens.bin", DatasetFormat.TOKENS_BIN),
        ]
        
        for pattern, format_type in dataset_patterns:
            for file_path in self.data_archive_path.glob(pattern):
                dataset_name = file_path.stem
                
                # Determine purpose based on filename
                purpose = self._determine_purpose(dataset_name)
                
                # Create dataset info
                dataset_info = DatasetInfo(
                    name=dataset_name,
                    path=str(file_path),
                    format=format_type,
                    purpose=purpose,
                    description=f"{format_type.value.upper()} dataset"
                )
                
                self.datasets[dataset_name] = dataset_info
        
        # Sort datasets by purpose for better organization
        self.datasets = dict(sorted(
            self.datasets.items(),
            key=lambda x: (x[1].purpose.value, x[0])
        ))
    
    def _determine_purpose(self, dataset_name: str) -> DatasetPurpose:
        """Determine dataset purpose based on filename"""
        name_lower = dataset_name.lower()
        
        if any(word in name_lower for word in ['chat', 'conversation', 'dialog', 'soru', 'cevap']):
            return DatasetPurpose.CONVERSATION
        elif any(word in name_lower for word in ['knowledge', 'bilgi', 'wiki', 'science']):
            return DatasetPurpose.KNOWLEDGE
        elif any(word in name_lower for word in ['code', 'programming', 'kod', 'python']):
            return DatasetPurpose.CODE
        else:
            return DatasetPurpose.GENERAL
    
    def list_datasets(self) -> List[Dict]:
        """List all available datasets"""
        return [dataset.get_info_dict() for dataset in self.datasets.values()]
    
    def get_dataset_info(self, dataset_name: str) -> Optional[Dict]:
        """Get information about a specific dataset"""
        if dataset_name in self.datasets:
            return self.datasets[dataset_name].get_info_dict()
        return None
    
    def create_mixer(self, selected_datasets: List[str], 
                    mixing_strategy: str = "balanced") -> DatasetMixer:
        """
        Create a dataset mixer with selected datasets
        
        Args:
            selected_datasets: List of dataset names to include
            mixing_strategy: 'balanced', 'conversation_first', 'knowledge_heavy'
        """
        # Filter selected datasets
        datasets_to_mix = []
        for name in selected_datasets:
            if name in self.datasets:
                datasets_to_mix.append(self.datasets[name])
            else:
                print(f"⚠️ Dataset not found: {name}")
        
        if not datasets_to_mix:
            print("❌ No valid datasets selected")
            return None
        
        # Adjust purposes based on mixing strategy
        if mixing_strategy == "conversation_first":
            for dataset in datasets_to_mix:
                if dataset.purpose == DatasetPurpose.CONVERSATION:
                    # Boost conversation datasets
                    dataset.description += " (conversation priority)"
        
        elif mixing_strategy == "knowledge_heavy":
            for dataset in datasets_to_mix:
                if dataset.purpose == DatasetPurpose.KNOWLEDGE:
                    # Boost knowledge datasets
                    dataset.description += " (knowledge priority)"
        
        print(f"🧩 Creating dataset mixer with {len(datasets_to_mix)} datasets:")
        for dataset in datasets_to_mix:
            print(f"   • {dataset.name} ({dataset.purpose.value})")
        
        self.current_mixer = DatasetMixer(datasets_to_mix, self.vocab_size)
        return self.current_mixer
    
    def get_training_batch(self, batch_size: int, seq_length: int) -> Tuple[QuantumTensor, QuantumTensor]:
        """Get a training batch as QuantumTensors"""
        if self.current_mixer is None:
            print("❌ No dataset mixer created. Call create_mixer() first.")
            return None, None
        
        inputs_np, targets_np = self.current_mixer.get_batch(batch_size, seq_length)
        
        if len(inputs_np) == 0:
            print("⚠️ No samples available in dataset mixer")
            return None, None
        
        # Convert to QuantumTensors
        inputs_qt = QuantumTensor(inputs_np, requires_grad=False)
        targets_qt = QuantumTensor(targets_np, requires_grad=False)
        
        return inputs_qt, targets_qt
    
    def get_sample_preview(self, dataset_name: str, num_samples: int = 3) -> List[str]:
        """Get preview samples from a dataset"""
        if dataset_name not in self.datasets:
            return []
        
        dataset = self.datasets[dataset_name]
        
        # Load limited number of samples
        if dataset.format == DatasetFormat.TXT:
            samples = DatasetLoader.load_txt(dataset.path, max_samples=num_samples)
        elif dataset.format == DatasetFormat.JSONL:
            samples = DatasetLoader.load_jsonl(dataset.path, max_samples=num_samples)
        elif dataset.format == DatasetFormat.JSON:
            samples = DatasetLoader.load_json(dataset.path, max_samples=num_samples)
        elif dataset.format == DatasetFormat.TOKENS_BIN:
            # For token files, return token counts
            token_arrays = DatasetLoader.load_tokens_bin(dataset.path, max_samples=num_samples)
            samples = [f"Token array shape: {arr.shape}" for arr in token_arrays]
        else:
            samples = []
        
        return samples

# Example usage and testing
if __name__ == "__main__":
    print("🧪 Testing Ultimate Dataset Manager")
    print("=" * 70)
    
    # Create manager
    manager = UltimateDatasetManager("data_archive")
    
    # List datasets
    print("\n📚 Available datasets:")
    datasets = manager.list_datasets()
    for dataset in datasets:
        print(f"   • {dataset['name']}: {dataset['num_samples']} samples, "
              f"{dataset['avg_length']:.1f} avg length, {dataset['purpose']}")
    
    # Create mixer with conversation and knowledge datasets
    print("\n🧩 Creating dataset mixer...")
    selected = ["eita_dataset", "eita_chat_data"]  # Example dataset names
    mixer = manager.create_mixer(selected, mixing_strategy="balanced")
    
    if mixer:
        # Get a batch
        print("\n📦 Getting training batch...")
        inputs, targets = manager.get_training_batch(batch_size=2, seq_length=32)
        
        if inputs is not None:
            print(f"   Input shape: {inputs.shape}")
            print(f"   Target shape: {targets.shape}")
            
            # Get sample preview
            print("\n🔍 Sample preview from eita_dataset:")
            preview = manager.get_sample_preview("eita_dataset", num_samples=2)
            for i, sample in enumerate(preview):
                print(f"   Sample {i+1}: {sample[:100]}...")
        
        print("\n✅ Dataset manager test completed successfully!")
    else:
        print("❌ Failed to create dataset mixer")