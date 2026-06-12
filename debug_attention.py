#!/usr/bin/env python3
"""
Debug QuantumAttentionLayer
"""

import sys
sys.path.append('.')
import numpy as np

from ultimate_otoya import QuantumAttentionLayer
from quantum_tensor import QuantumTensor

print("🧪 Debug QuantumAttentionLayer")

# Create attention layer
attention = QuantumAttentionLayer(d_model=512, num_heads=8, quantum_heads=4)
print(f"Created attention layer: d_model=512, num_heads=8, quantum_heads=4")

# Create input tensor
x = QuantumTensor(np.random.randn(2, 32, 512), requires_grad=True)
print(f"Input shape: {x.shape}")

# Test forward pass
try:
    print("\nTrying forward pass...")
    output = attention.forward(x)
    print(f"✓ Forward pass successful: {output.shape}")
except Exception as e:
    print(f"✗ Error in forward pass: {e}")
    import traceback
    traceback.print_exc()
    
    # Debug step by step
    print("\n🔍 Debugging step by step:")
    
    # Get batch size, seq_len
    batch_size, seq_len, _ = x.shape
    
    # Project queries
    print("1. Projecting queries...")
    q = attention._linear(x, attention.w_q, attention.b_q)
    print(f"   q shape after _linear: {q.shape}")
    
    # Reshape for multi-head
    print("2. Reshaping for multi-head...")
    q_reshaped = q.reshape(batch_size, seq_len, attention.num_heads, attention.head_dim)
    print(f"   q shape after reshape: {q_reshaped.shape}")
    
    # Try transpose
    print("3. Trying transpose...")
    try:
        q_transposed = q_reshaped.transpose(1, 2)
        print(f"   q shape after transpose: {q_transposed.shape}")
    except Exception as e2:
        print(f"   ✗ Error in transpose: {e2}")
        print(f"   q_reshaped.ndim: {q_reshaped.ndim}")
        print(f"   q_reshaped.shape: {q_reshaped.shape}")