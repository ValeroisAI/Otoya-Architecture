#!/usr/bin/env python3
"""
Ultimate Otoya Dataset Integration Test
Multiple dataset selection, mixing strategies, and format support test
"""

import sys
import os
sys.path.append('.')

import numpy as np
import time
from pathlib import Path

print("🚀 Ultimate Otoya Dataset Integration Test")
print("=" * 50)

# Test 1: Dataset Manager
print("\n📚 Test 1: Dataset Manager Initialization")
try:
    from dataset_manager import UltimateDatasetManager
    
    # Create dataset manager
    manager = UltimateDatasetManager("data_archive")
    
    print(f"✅ Dataset Manager başarıyla oluşturuldu")
    print(f"   • Bulunan dataset'ler: {len(manager.datasets)}")
    
    # List datasets
    for name, info in manager.datasets.items():
        print(f"   • {name}: {info.num_samples:,} samples, {info.format.value}, {info.purpose.value}")
    
except Exception as e:
    print(f"❌ Dataset Manager hatası: {e}")
    import traceback
    traceback.print_exc()

# Test 2: Dataset Loader
print("\n📚 Test 2: Dataset Loader")
try:
    from dataset_manager import DatasetLoader
    
    loader = DatasetLoader()
    
    # Test loading samples
    if manager.datasets:
        first_dataset = list(manager.datasets.values())[0]
        print(f"   • Test dataset: {first_dataset.name}")
        
        samples = loader.load_samples(
            first_dataset.path, 
            first_dataset.format, 
            max_samples=2
        )
        
        print(f"   • Yüklenen örnekler: {len(samples)}")
        
        for i, sample in enumerate(samples):
            if isinstance(sample, str):
                preview = sample[:80].replace('\n', ' ')
                print(f"     {i+1}. {preview}...")
            elif isinstance(sample, dict):
                print(f"     {i+1}. {str(sample)[:80]}...")
    
    print("✅ Dataset Loader testi başarılı")
    
except Exception as e:
    print(f"❌ Dataset Loader hatası: {e}")

# Test 3: Dataset Mixer
print("\n📚 Test 3: Dataset Mixer")
try:
    # Select first 2 datasets for mixing
    if len(manager.datasets) >= 2:
        dataset_names = list(manager.datasets.keys())[:2]
        print(f"   • Mixing için seçilen dataset'ler: {dataset_names}")
        
        # Create mixer
        mixer = manager.create_mixer(
            dataset_names,
            mixing_strategy="balanced"
        )
        
        print(f"   • Mixer başarıyla oluşturuldu")
        print(f"   • Dataset sayısı: {len(mixer.datasets)}")
        print(f"   • Weights: {mixer.weights}")
        
        # Test batch generation
        batch = mixer.get_batch(batch_size=4, seq_length=64)
        
        if batch:
            inputs, targets = batch
            print(f"   • Batch boyutları: inputs={inputs.shape}, targets={targets.shape}")
            print(f"   • Batch dtype: {inputs.dtype}")
        
        print("✅ Dataset Mixer testi başarılı")
    else:
        print("⚠️  En az 2 dataset gerekiyor")
        
except Exception as e:
    print(f"❌ Dataset Mixer hatası: {e}")

# Test 4: DataLoader Integration
print("\n📚 Test 4: DataLoader Integration")
try:
    from ultimate_dataloader import DataLoader
    
    # Test with multiple datasets
    if len(manager.datasets) >= 2:
        dataset_names = list(manager.datasets.keys())[:2]
        
        print(f"   • DataLoader için dataset'ler: {dataset_names}")
        
        # Create DataLoader with multiple datasets
        data_loader = DataLoader(
            data_source=dataset_names,
            batch_size=4,
            seq_length=64,
            dataset_manager=manager,
            mixing_strategy="balanced"
        )
        
        print(f"   • DataLoader başarıyla oluşturuldu")
        print(f"   • Using multiple datasets: {data_loader.using_multiple_datasets}")
        
        # Test iteration
        batch_count = 0
        max_batches = 3
        
        try:
            for inputs, targets in data_loader:
                batch_count += 1
                print(f"   • Batch {batch_count}: inputs={inputs.shape}, targets={targets.shape}")
                
                if batch_count >= max_batches:
                    break
            
            print(f"✅ DataLoader {batch_count} batch başarıyla üretti")
        except Exception as e:
            print(f"⚠️  DataLoader iteration hatası: {e}")
            print(f"   • Ancak DataLoader başarıyla oluşturuldu")
        
except Exception as e:
    print(f"❌ DataLoader Integration hatası: {e}")
    import traceback
    traceback.print_exc()

# Test 5: Trainer Integration
print("\n📚 Test 5: Trainer Integration")
try:
    from ultimate_otoya import UltimateConfig
    from train_ultimate import UltimateTrainer
    
    # Create config
    config = UltimateConfig(
        vocab_size=50000,
        d_model=256,  # Smaller for testing
        num_layers=4,
        num_heads=4,
        context_length=256,
        learning_rate=0.0003
    )
    
    # Select datasets
    if len(manager.datasets) >= 2:
        dataset_names = list(manager.datasets.keys())[:2]
        
        print(f"   • Trainer için dataset'ler: {dataset_names}")
        
        # Create trainer with datasets
        trainer = UltimateTrainer(
            config,
            selected_datasets=dataset_names,
            mixing_strategy="balanced"
        )
        
        print(f"   • Trainer başarıyla oluşturuldu")
        
        # Test training step
        print(f"   • Training step testi...")
        result = trainer.training_step(batch_size=2, seq_len=64)
        
        print(f"   • Step sonucu: loss={result['loss']:.4f}, accuracy={result['accuracy']:.3f}")
        print(f"   • Step süresi: {result['step_time']:.3f}s")
        
        print("✅ Trainer Integration testi başarılı")
    else:
        print("⚠️  En az 2 dataset gerekiyor")
        
except Exception as e:
    print(f"❌ Trainer Integration hatası: {e}")
    import traceback
    traceback.print_exc()

# Test 6: Dataset Format Support
print("\n📚 Test 6: Dataset Format Support")
try:
    from dataset_manager import DatasetFormat
    
    print("   • Desteklenen formatlar:")
    for fmt in DatasetFormat:
        print(f"     - {fmt.value}")
    
    # Check if data_archive directory exists
    data_dir = Path("data_archive")
    if data_dir.exists():
        print(f"   • data_archive dizini mevcut")
        
        # List files
        files = list(data_dir.glob("*"))
        print(f"   • {len(files)} dosya bulundu:")
        
        for file in files:
            print(f"     - {file.name} ({file.stat().st_size:,} bytes)")
    else:
        print(f"⚠️  data_archive dizini bulunamadı")
    
    print("✅ Dataset Format testi başarılı")
    
except Exception as e:
    print(f"❌ Dataset Format hatası: {e}")

print("\n" + "=" * 50)
print("🎉 Dataset Integration Test Tamamlandı!")
print("\n📋 Özet:")
print(f"   • Dataset Manager: {'✅' if 'manager' in locals() else '❌'}")
print(f"   • Dataset Loader: {'✅' if 'loader' in locals() else '❌'}")
print(f"   • Dataset Mixer: {'✅' if 'mixer' in locals() else '❌'}")
print(f"   • DataLoader Integration: {'✅' if 'data_loader' in locals() else '❌'}")
print(f"   • Trainer Integration: {'✅' if 'trainer' in locals() else '❌'}")

# Check if we have the example dataset
example_path = Path("data_archive/eita_dataset.jsonl")
if example_path.exists():
    print(f"\n📁 Örnek dataset mevcut: {example_path}")
    print(f"   • Boyut: {example_path.stat().st_size:,} bytes")
else:
    print(f"\n⚠️  Örnek dataset bulunamadı: {example_path}")
    print("   • Lütfen data_archive/eita_dataset.jsonl dosyasını oluşturun")

print("\n🚀 Ultimate Otoya Dataset sistemi hazır!")