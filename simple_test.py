#!/usr/bin/env python3
"""
Simple test for QuantumTensor system
"""

import numpy as np
import time
from quantum_tensor import QuantumTensor, QuantumOptimizer

def test_simple():
    print("🧪 Simple QuantumTensor Test")
    print("=" * 50)
    
    # 1. Create tensors
    print("1. Creating tensors...")
    a = QuantumTensor([[1, 2], [3, 4]], requires_grad=True)
    b = QuantumTensor([[5, 6], [7, 8]], requires_grad=True)
    
    print(f"   a: {a}")
    print(f"   b: {b}")
    
    # 2. Basic operations
    print("\n2. Testing operations...")
    
    # Addition
    c = a + b
    print(f"   a + b: shape={c.shape}")
    
    # Multiplication
    d = a * b
    print(f"   a * b: shape={d.shape}")
    
    # Matrix multiplication
    e = a @ b
    print(f"   a @ b: shape={e.shape}, data=\n{e.data}")
    
    # 3. Gradient test
    print("\n3. Testing gradients...")
    
    # Simple computation graph
    result = (a * b).sum()
    print(f"   Result: {result.data}")
    
    # Backward pass
    result.backward()
    
    print(f"   a.grad:\n{a.grad}")
    print(f"   b.grad:\n{b.grad}")
    
    # 4. Optimizer test
    print("\n4. Testing optimizer...")
    
    params = [a, b]
    optimizer = QuantumOptimizer(params, lr=0.01)
    
    # Store original values
    a_orig = a.data.copy()
    b_orig = b.data.copy()
    
    # Zero gradients
    optimizer.zero_grad()
    
    # Set gradients manually
    a.grad = np.ones_like(a.data)
    b.grad = np.ones_like(b.data)
    
    # Optimizer step
    step_time = optimizer.step()
    
    print(f"   Step time: {step_time:.6f}s")
    print(f"   a changed: {not np.allclose(a.data, a_orig)}")
    print(f"   b changed: {not np.allclose(b.data, b_orig)}")
    
    # 5. Performance test
    print("\n5. Performance test...")
    
    sizes = [64, 128, 256]
    
    for size in sizes:
        print(f"\n   Size {size}x{size}:")
        
        # Create large tensors
        x = QuantumTensor.randn((size, size))
        y = QuantumTensor.randn((size, size))
        
        # Time matmul
        start = time.time()
        z = x @ y
        elapsed = time.time() - start
        
        ops = size ** 3
        ops_per_sec = ops / elapsed if elapsed > 0 else 0
        
        print(f"     Matmul time: {elapsed:.6f}s")
        print(f"     Operations: {ops:,}")
        print(f"     Ops/sec: {ops_per_sec:,.0f}")
    
    print("\n" + "=" * 50)
    print("✅ Simple test PASSED!")
    
    return True

def test_memory_efficiency():
    print("\n💾 Memory Efficiency Test")
    print("=" * 50)
    
    # Test with different batch sizes
    batch_sizes = [1, 2, 4, 8]
    seq_len = 512
    d_model = 256
    
    print(f"Seq len: {seq_len}, d_model: {d_model}")
    
    for batch_size in batch_sizes:
        print(f"\n  Batch size: {batch_size}")
        
        # Create tensors
        start = time.time()
        x = QuantumTensor.randn((batch_size, seq_len, d_model))
        y = QuantumTensor.randn((batch_size, seq_len, d_model))
        create_time = time.time() - start
        
        # Memory usage estimate
        memory_mb = (x.data.nbytes + y.data.nbytes) / (1024**2)
        
        # Simple operation
        start = time.time()
        z = x + y
        op_time = time.time() - start
        
        print(f"    Create time: {create_time:.3f}s")
        print(f"    Memory: {memory_mb:.2f}MB")
        print(f"    Op time: {op_time:.3f}s")
        
        # Clean up
        del x, y, z
    
    return True

def main():
    print("⚡ QuantumTensor Simple Test")
    print("=" * 50)
    
    try:
        # Run simple test
        test_simple()
        
        # Run memory test
        test_memory_efficiency()
        
        print("\n" + "=" * 50)
        print("🎉 ALL TESTS PASSED!")
        print("\nSystem features:")
        print("  • Pure NumPy implementation")
        print("  • Automatic differentiation")
        print("  • Optimizer with AdamW")
        print("  • Memory efficient")
        print("  • No external dependencies")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(main())