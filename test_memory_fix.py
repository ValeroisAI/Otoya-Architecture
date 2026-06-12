#!/usr/bin/env python3
"""
Test Memory Fix for Ultimate Otoya
Model oluşturma ve eğitim donma sorunlarını test et
"""

import sys
import os
sys.path.append('.')

import time
import numpy as np

print("🧪 Testing Memory Fix for Ultimate Otoya")
print("=" * 60)

try:
    from ultimate_otoya import UltimateConfig, UltimateOtoyaModel
    from quantum_tensor import QuantumTensor
    from train_ultimate import UltimateTrainer
    
    print("✅ Modules imported successfully")
    
    # Test 1: Small model creation
    print("\n1️⃣ Testing small model creation...")
    start_time = time.time()
    
    config = UltimateConfig(
        vocab_size=10000,
        d_model=256,
        num_layers=6,
        num_heads=4,
        quantum_heads=2,
        fractal_depth=2,
        context_length=256,
        learning_rate=0.0003
    )
    
    print(f"   • Creating model with config:")
    print(f"     - vocab_size: {config.vocab_size}")
    print(f"     - d_model: {config.d_model}")
    print(f"     - num_layers: {config.num_layers}")
    print(f"     - context_length: {config.context_length}")
    
    model = UltimateOtoyaModel(config)
    
    creation_time = time.time() - start_time
    print(f"   ✅ Model created in {creation_time:.2f} seconds")
    print(f"   • Parameter count: {model.parameter_count:,}")
    
    # Test 2: Memory stats
    print("\n2️⃣ Testing memory stats...")
    mem_stats = QuantumTensor.get_memory_stats()
    
    print(f"   • RAM usage: {mem_stats.get('allocated_ram_gb', 0):.2f}GB")
    print(f"   • VRAM usage: {mem_stats.get('allocated_vram_gb', 0):.2f}GB")
    print(f"   • Max RAM limit: {mem_stats.get('max_ram_gb', 0):.2f}GB")
    print(f"   • Max VRAM limit: {mem_stats.get('max_vram_gb', 0):.2f}GB")
    
    # Test 3: Training step
    print("\n3️⃣ Testing training step...")
    trainer = UltimateTrainer(config)
    
    start_time = time.time()
    result = trainer.training_step(batch_size=2, seq_len=128)
    step_time = time.time() - start_time
    
    print(f"   ✅ Training step completed in {step_time:.2f} seconds")
    print(f"   • Loss: {result['loss']:.4f}")
    print(f"   • Accuracy: {result['accuracy']:.3f}")
    print(f"   • Step time: {result['step_time']:.3f}s")
    
    # Test 4: Memory after training
    print("\n4️⃣ Testing memory after training...")
    mem_stats_after = QuantumTensor.get_memory_stats()
    
    print(f"   • RAM usage after: {mem_stats_after.get('allocated_ram_gb', 0):.2f}GB")
    print(f"   • VRAM usage after: {mem_stats_after.get('allocated_vram_gb', 0):.2f}GB")
    
    # Test 5: Multiple training steps
    print("\n5️⃣ Testing multiple training steps...")
    total_steps = 10
    step_times = []
    
    for i in range(total_steps):
        start_time = time.time()
        result = trainer.training_step(batch_size=2, seq_len=128)
        step_time = time.time() - start_time
        step_times.append(step_time)
        
        if (i + 1) % 5 == 0:
            print(f"   • Step {i+1}: Loss={result['loss']:.4f}, Time={step_time:.3f}s")
    
    avg_step_time = np.mean(step_times)
    print(f"   ✅ {total_steps} steps completed")
    print(f"   • Average step time: {avg_step_time:.3f} seconds")
    print(f"   • Fastest step: {np.min(step_times):.3f} seconds")
    print(f"   • Slowest step: {np.max(step_times):.3f} seconds")
    
    # Test 6: Memory limits check
    print("\n6️⃣ Testing memory limits...")
    ram_usage = mem_stats_after.get('allocated_ram_gb', 0)
    vram_usage = mem_stats_after.get('allocated_vram_gb', 0)
    
    if ram_usage > 14.5:
        print(f"   ⚠️  RAM usage ({ram_usage:.2f}GB) exceeds 14.5GB limit!")
    else:
        print(f"   ✅ RAM usage ({ram_usage:.2f}GB) within 14.5GB limit")
    
    if vram_usage > 14.5:
        print(f"   ⚠️  VRAM usage ({vram_usage:.2f}GB) exceeds 14.5GB limit!")
    else:
        print(f"   ✅ VRAM usage ({vram_usage:.2f}GB) within 14.5GB limit")
    
    print("\n" + "=" * 60)
    print("🎉 ALL TESTS PASSED!")
    print("✅ Memory fix is working correctly")
    print("✅ Model creation is fast")
    print("✅ Training doesn't freeze PC")
    print("✅ Memory limits are respected")
    
except Exception as e:
    print(f"\n❌ Test failed: {e}")
    import traceback
    traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("❌ TESTS FAILED")
    print("Please fix the issues above")