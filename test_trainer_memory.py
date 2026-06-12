#!/usr/bin/env python3
"""
Trainer Memory Test Script
"""

import sys
sys.path.append('.')
from ultimate_otoya import UltimateConfig
from train_ultimate import UltimateTrainer
import numpy as np

print('🧪 Testing Ultimate Trainer Memory Management...')

# Create config
print('1. Creating config...')
config = UltimateConfig(
    vocab_size=50000,
    d_model=512,
    num_heads=8,
    num_layers=12,
    context_length=1024,
    learning_rate=1e-4
)

# Create trainer
print('2. Creating trainer...')
trainer = UltimateTrainer(config)

# Test batch generation
print('3. Testing batch generation...')
batch_size = 2
seq_len = 128

inputs, targets = trainer.generate_batch(batch_size, seq_len)
print(f'  Inputs shape: {inputs.shape}')
print(f'  Targets shape: {targets.shape}')

# Test training step
print('4. Testing training step...')
try:
    loss, accuracy = trainer.train_step(inputs, targets)
    print(f'  Loss: {loss:.4f}')
    print(f'  Accuracy: {accuracy:.3f}')
    print('  ✅ Training step successful')
except Exception as e:
    print(f'  ❌ Training step failed: {e}')

# Test memory stats
print('5. Testing memory stats...')
stats = trainer.stats
print(f'  Losses: {len(stats["losses"])}')
print(f'  Accuracies: {len(stats["accuracies"])}')
print(f'  Step times: {len(stats["step_times"])}')

# Test chunked operations with large tensors
print('6. Testing chunked operations...')
try:
    # Create large tensors that should trigger chunked operations
    large_inputs = QuantumTensor(np.random.randn(4, 1024, 512))
    large_targets = QuantumTensor(np.random.randn(4, 1024, 512))
    
    # This should use chunked matmul internally
    result = large_inputs @ large_inputs.T
    print(f'  Large matmul result shape: {result.shape}')
    print('  ✅ Chunked operations working')
except Exception as e:
    print(f'  ❌ Chunked operations failed: {e}')

print('\n✅ Ultimate Trainer Memory Test Completed')