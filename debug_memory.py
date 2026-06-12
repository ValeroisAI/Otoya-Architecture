#!/usr/bin/env python3
"""
Debug FractalMemoryLayer
"""

import sys
sys.path.append('.')
import numpy as np

from ultimate_otoya import FractalMemoryLayer
from quantum_tensor import QuantumTensor

print("🧪 Debug FractalMemoryLayer")

# Create memory layer
memory = FractalMemoryLayer(d_model=512, memory_slots=128, depth=3)
print(f"Created memory layer: d_model=512, memory_slots=128, depth=3")

# Create input tensor
x = QuantumTensor(np.random.randn(2, 32, 512), requires_grad=True)
print(f"Input shape: {x.shape}")

# Test forward pass
try:
    print("\nTrying forward pass...")
    output = memory.forward(x)
    print(f"✓ Forward pass successful: {output.shape}")
    print(f"  Memory shape: {memory.memory.shape}")
except Exception as e:
    print(f"✗ Error in forward pass: {e}")
    import traceback
    traceback.print_exc()
    
    # Debug step by step
    print("\n🔍 Debugging step by step:")
    
    # Get batch size, seq_len, d_model
    batch_size, seq_len, d_model = x.shape
    
    # Apply gate
    print("1. Applying gate...")
    try:
        gate_weights = memory._linear(x, memory.gate_weight, memory.gate_bias)
        print(f"   gate_weights shape: {gate_weights.shape}")
        
        # Apply sigmoid
        gate = gate_weights.sigmoid()
        print(f"   gate shape after sigmoid: {gate.shape}")
    except Exception as e2:
        print(f"   ✗ Error in gate application: {e2}")