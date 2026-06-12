#!/usr/bin/env python3
"""
Memory Manager Test Script
"""

import sys
sys.path.append('.')
from quantum_tensor import QuantumTensor
import numpy as np

print('🧪 Testing Memory Manager...')

# Create some tensors
print('1. Creating tensors...')
t1 = QuantumTensor(np.random.randn(100, 100))
t2 = QuantumTensor(np.random.randn(100, 100))
t3 = QuantumTensor(np.random.randn(100, 100))

# Get memory stats
print('\n2. Memory Stats:')
stats = QuantumTensor.get_memory_stats()
for key, value in stats.items():
    print(f'  {key}: {value}')

# Test memory estimation
print('\n3. Model Memory Estimation:')
mem_est = QuantumTensor.estimate_model_memory(
    vocab_size=50000,
    d_model=512,
    num_layers=12,
    context_length=1024
)
for key, value in mem_est.items():
    print(f'  {key}: {value:.1f} MB')

# Test batch size recommendation
print('\n4. Batch Size Recommendation:')
recommended = QuantumTensor.recommend_batch_size(
    mem_est['total_training_mb'],
    12 * 1024  # 12GB in MB
)
print(f'  Recommended batch size: {recommended}')

# Test chunked operations
print('\n5. Testing chunked operations...')
large_tensor = QuantumTensor(np.random.randn(5000, 5000))
print(f'  Large tensor created: {large_tensor.shape}')

# Test memory safety
print('\n6. Testing memory safety...')
try:
    # Try to allocate too much memory
    huge_tensor = QuantumTensor(np.random.randn(10000, 10000))
    print(f'  Huge tensor created: {huge_tensor.shape}')
except MemoryError as e:
    print(f'  ✅ Memory safety working: {e}')

print('\n✅ Memory Manager test completed')