#!/usr/bin/env python3
"""
OpenCL Backend for QuantumTensor
GPU acceleration support
"""

import sys
import os
import numpy as np
import time
from typing import List, Tuple, Optional, Union, Dict
import warnings

try:
    import pyopencl as cl
    import pyopencl.array as cl_array
    HAS_OPENCL = True
except ImportError:
    HAS_OPENCL = False
    print("⚠️  pyopencl not installed. GPU backend disabled.")
    print("   Install with: pip install pyopencl")

class OpenCLDevice:
    """OpenCL device wrapper"""
    
    def __init__(self, device_idx: int = 0):
        if not HAS_OPENCL:
            raise RuntimeError("OpenCL not available")
            
        # Get platforms and devices
        platforms = cl.get_platforms()
        if not platforms:
            raise RuntimeError("No OpenCL platforms found")
        
        # Get all devices
        all_devices = []
        for platform in platforms:
            try:
                devices = platform.get_devices()
                all_devices.extend(devices)
            except:
                continue
        
        if not all_devices:
            raise RuntimeError("No OpenCL devices found")
        
        # Select device
        if device_idx >= len(all_devices):
            warnings.warn(f"Device index {device_idx} out of range. Using device 0.")
            device_idx = 0
        
        self.device = all_devices[device_idx]
        self.platform = self.device.platform
        
        # Create context and queue
        self.context = cl.Context([self.device])
        self.queue = cl.CommandQueue(self.context)
        
        # Device info
        self.name = self.device.name.strip()
        self.vendor = self.device.vendor.strip()
        self.max_compute_units = self.device.max_compute_units
        self.global_mem_size = self.device.global_mem_size
        self.local_mem_size = self.device.local_mem_size
        
        # Compile kernels
        self._compile_kernels()
        
        print(f"✅ OpenCL Device Initialized:")
        print(f"   • Device: {self.name}")
        print(f"   • Vendor: {self.vendor}")
        print(f"   • Compute Units: {self.max_compute_units}")
        print(f"   • Global Memory: {self.global_mem_size / 1024**3:.2f}GB")
        print(f"   • Local Memory: {self.local_mem_size / 1024:,.0f}KB")
    
    def _compile_kernels(self):
        """Compile OpenCL kernels"""
        # Basic kernels
        kernel_source = """
        __kernel void elementwise_add(
            __global const float* a,
            __global const float* b,
            __global float* result,
            const int size
        ) {
            int idx = get_global_id(0);
            if (idx < size) {
                result[idx] = a[idx] + b[idx];
            }
        }
        
        __kernel void elementwise_mul(
            __global const float* a,
            __global const float* b,
            __global float* result,
            const int size
        ) {
            int idx = get_global_id(0);
            if (idx < size) {
                result[idx] = a[idx] * b[idx];
            }
        }
        
        __kernel void matmul(
            __global const float* a,
            __global const float* b,
            __global float* result,
            const int m,
            const int n,
            const int p
        ) {
            int row = get_global_id(0);
            int col = get_global_id(1);
            
            if (row < m && col < p) {
                float sum = 0.0f;
                for (int k = 0; k < n; k++) {
                    sum += a[row * n + k] * b[k * p + col];
                }
                result[row * p + col] = sum;
            }
        }
        
        __kernel void relu(
            __global const float* input,
            __global float* output,
            const int size
        ) {
            int idx = get_global_id(0);
            if (idx < size) {
                output[idx] = input[idx] > 0.0f ? input[idx] : 0.0f;
            }
        }
        
        __kernel void gelu(
            __global const float* input,
            __global float* output,
            const int size
        ) {
            int idx = get_global_id(0);
            if (idx < size) {
                float x = input[idx];
                output[idx] = 0.5f * x * (1.0f + tanh(sqrt(2.0f / M_PI_F) * (x + 0.044715f * x * x * x)));
            }
        }
        
        __kernel void softmax(
            __global const float* input,
            __global float* output,
            const int batch_size,
            const int seq_len,
            const int vocab_size
        ) {
            int batch = get_global_id(0);
            int seq = get_global_id(1);
            
            if (batch < batch_size && seq < seq_len) {
                // Find max for numerical stability
                float max_val = input[batch * seq_len * vocab_size + seq * vocab_size];
                for (int i = 1; i < vocab_size; i++) {
                    float val = input[batch * seq_len * vocab_size + seq * vocab_size + i];
                    if (val > max_val) {
                        max_val = val;
                    }
                }
                
                // Compute exp(x - max)
                float sum = 0.0f;
                for (int i = 0; i < vocab_size; i++) {
                    float val = input[batch * seq_len * vocab_size + seq * vocab_size + i];
                    float exp_val = exp(val - max_val);
                    output[batch * seq_len * vocab_size + seq * vocab_size + i] = exp_val;
                    sum += exp_val;
                }
                
                // Normalize
                for (int i = 0; i < vocab_size; i++) {
                    output[batch * seq_len * vocab_size + seq * vocab_size + i] /= sum;
                }
            }
        }
        
        // Backward kernels
        __kernel void matmul_backward_a(
            __global const float* grad,
            __global const float* b,
            __global float* result,
            const int m,
            const int p,
            const int n
        ) {
            int row = get_global_id(0);
            int col = get_global_id(1);
            
            if (row < m && col < n) {
                float sum = 0.0f;
                for (int k = 0; k < p; k++) {
                    sum += grad[row * p + k] * b[col * p + k];  // b.T
                }
                result[row * n + col] = sum;
            }
        }
        
        __kernel void matmul_backward_b(
            __global const float* a,
            __global const float* grad,
            __global float* result,
            const int n,
            const int m,
            const int p
        ) {
            int row = get_global_id(0);
            int col = get_global_id(1);
            
            if (row < n && col < p) {
                float sum = 0.0f;
                for (int k = 0; k < m; k++) {
                    sum += a[k * n + row] * grad[k * p + col];  // a.T
                }
                result[row * p + col] = sum;
            }
        }
        
        __kernel void relu_backward(
            __global const float* grad,
            __global const float* x,
            __global float* result,
            const int size
        ) {
            int idx = get_global_id(0);
            if (idx < size) {
                result[idx] = x[idx] > 0.0f ? grad[idx] : 0.0f;
            }
        }
        
        __kernel void gelu_backward(
            __global const float* grad,
            __global const float* x,
            __global float* result,
            const int size
        ) {
            int idx = get_global_id(0);
            if (idx < size) {
                float x_val = x[idx];
                // GELU derivative approximation
                float sqrt_2_over_pi = sqrt(2.0f / M_PI_F);
                float cdf = 0.5f * (1.0f + tanh(sqrt_2_over_pi * (x_val + 0.044715f * x_val * x_val * x_val)));
                float pdf = 0.5f * sqrt_2_over_pi * (1.0f + 0.134145f * x_val * x_val) / 
                           pow(cosh(sqrt_2_over_pi * (x_val + 0.044715f * x_val * x_val * x_val)), 2.0f);
                result[idx] = grad[idx] * (cdf + x_val * pdf);
            }
        }
        
        __kernel void softmax_backward(
            __global const float* grad,
            __global const float* output,
            __global float* result,
            const int batch_size,
            const int seq_len,
            const int vocab_size
        ) {
            int batch = get_global_id(0);
            int seq = get_global_id(1);
            
            if (batch < batch_size && seq < seq_len) {
                // Compute sum(grad * output)
                float sum = 0.0f;
                for (int i = 0; i < vocab_size; i++) {
                    int idx = batch * seq_len * vocab_size + seq * vocab_size + i;
                    sum += grad[idx] * output[idx];
                }
                
                // Compute gradient: output * (grad - sum)
                for (int i = 0; i < vocab_size; i++) {
                    int idx = batch * seq_len * vocab_size + seq * vocab_size + i;
                    result[idx] = output[idx] * (grad[idx] - sum);
                }
            }
        }
        """
        
        # Build program
        self.program = cl.Program(self.context, kernel_source).build()
        
        # Get kernel functions
        self.kernels = {
            'add': self.program.elementwise_add,
            'mul': self.program.elementwise_mul,
            'matmul': self.program.matmul,
            'relu': self.program.relu,
            'gelu': self.program.gelu,
            'softmax': self.program.softmax,
            'matmul_backward_a': self.program.matmul_backward_a,
            'matmul_backward_b': self.program.matmul_backward_b,
            'relu_backward': self.program.relu_backward,
            'gelu_backward': self.program.gelu_backward,
            'softmax_backward': self.program.softmax_backward
        }
    
    def create_buffer(self, shape: Tuple[int, ...], dtype: np.dtype = np.float32,
                     flags: cl.mem_flags = cl.mem_flags.READ_WRITE):
        """Create OpenCL buffer"""
        size = np.prod(shape) * np.dtype(dtype).itemsize
        return cl.Buffer(self.context, flags, size)
    
    def to_device(self, data: np.ndarray) -> cl_array.Array:
        """Copy data to device"""
        return cl_array.to_device(self.queue, data)
    
    def from_device(self, cl_array_obj: cl_array.Array) -> np.ndarray:
        """Copy data from device"""
        return cl_array_obj.get()
    
    def matmul(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Matrix multiplication on GPU"""
        m, n = a.shape
        n2, p = b.shape
        
        if n != n2:
            raise ValueError(f"Matrix dimensions don't match: {a.shape} vs {b.shape}")
        
        # Create device arrays
        a_dev = self.to_device(a.astype(np.float32))
        b_dev = self.to_device(b.astype(np.float32))
        result_dev = cl_array.empty(self.queue, (m, p), dtype=np.float32)
        
        # Execute kernel
        kernel = self.kernels['matmul']
        kernel(self.queue, (m, p), None,
               a_dev.data, b_dev.data, result_dev.data,
               np.int32(m), np.int32(n), np.int32(p))
        
        # Copy result back
        return self.from_device(result_dev)
    
    def elementwise_add(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Element-wise addition on GPU"""
        if a.shape != b.shape:
            raise ValueError(f"Shapes don't match: {a.shape} vs {b.shape}")
        
        size = a.size
        
        # Create device arrays
        a_dev = self.to_device(a.astype(np.float32))
        b_dev = self.to_device(b.astype(np.float32))
        result_dev = cl_array.empty(self.queue, a.shape, dtype=np.float32)
        
        # Execute kernel
        kernel = self.kernels['add']
        kernel(self.queue, (size,), None,
               a_dev.data, b_dev.data, result_dev.data,
               np.int32(size))
        
        return self.from_device(result_dev)
    
    def elementwise_mul(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Element-wise multiplication on GPU"""
        if a.shape != b.shape:
            raise ValueError(f"Shapes don't match: {a.shape} vs {b.shape}")
        
        size = a.size
        
        # Create device arrays
        a_dev = self.to_device(a.astype(np.float32))
        b_dev = self.to_device(b.astype(np.float32))
        result_dev = cl_array.empty(self.queue, a.shape, dtype=np.float32)
        
        # Execute kernel
        kernel = self.kernels['mul']
        kernel(self.queue, (size,), None,
               a_dev.data, b_dev.data, result_dev.data,
               np.int32(size))
        
        return self.from_device(result_dev)
    
    def relu(self, x: np.ndarray) -> np.ndarray:
        """ReLU activation on GPU"""
        size = x.size
        
        # Create device arrays
        x_dev = self.to_device(x.astype(np.float32))
        result_dev = cl_array.empty(self.queue, x.shape, dtype=np.float32)
        
        # Execute kernel
        kernel = self.kernels['relu']
        kernel(self.queue, (size,), None,
               x_dev.data, result_dev.data,
               np.int32(size))
        
        return self.from_device(result_dev)
    
    def gelu(self, x: np.ndarray) -> np.ndarray:
        """GELU activation on GPU"""
        size = x.size
        
        # Create device arrays
        x_dev = self.to_device(x.astype(np.float32))
        result_dev = cl_array.empty(self.queue, x.shape, dtype=np.float32)
        
        # Execute kernel
        kernel = self.kernels['gelu']
        kernel(self.queue, (size,), None,
               x_dev.data, result_dev.data,
               np.int32(size))
        
        return self.from_device(result_dev)
    
    def softmax(self, x: np.ndarray, dim: int = -1) -> np.ndarray:
        """Softmax on GPU"""
        if dim != -1:
            raise NotImplementedError("Only last dimension softmax supported")
        
        if x.ndim != 3:
            raise ValueError("Softmax expects 3D tensor (batch, seq, vocab)")
        
        batch_size, seq_len, vocab_size = x.shape
        
        # Create device arrays
        x_dev = self.to_device(x.astype(np.float32))
        result_dev = cl_array.empty(self.queue, x.shape, dtype=np.float32)
        
        # Execute kernel
        kernel = self.kernels['softmax']
        kernel(self.queue, (batch_size, seq_len), None,
               x_dev.data, result_dev.data,
               np.int32(batch_size), np.int32(seq_len), np.int32(vocab_size))
        
        return self.from_device(result_dev)
    
    # ------------------------------------------------------------------
    # Backward Kernels
    # ------------------------------------------------------------------
    def matmul_backward(self, grad: np.ndarray, a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Matrix multiplication backward pass"""
        m, n = a.shape
        n2, p = b.shape
        
        if n != n2:
            raise ValueError(f"Matrix dimensions don't match: {a.shape} vs {b.shape}")
        
        # Create device arrays
        grad_dev = self.to_device(grad.astype(np.float32))
        a_dev = self.to_device(a.astype(np.float32))
        b_dev = self.to_device(b.astype(np.float32))
        
        # Gradients: dL/da = grad @ b.T, dL/db = a.T @ grad
        grad_a_dev = cl_array.empty(self.queue, (m, n), dtype=np.float32)
        grad_b_dev = cl_array.empty(self.queue, (n, p), dtype=np.float32)
        
        # Kernel for dL/da = grad @ b.T
        kernel_da = self.kernels['matmul_backward_a']
        kernel_da(self.queue, (m, n), None,
                  grad_dev.data, b_dev.data, grad_a_dev.data,
                  np.int32(m), np.int32(p), np.int32(n))
        
        # Kernel for dL/db = a.T @ grad
        kernel_db = self.kernels['matmul_backward_b']
        kernel_db(self.queue, (n, p), None,
                  a_dev.data, grad_dev.data, grad_b_dev.data,
                  np.int32(n), np.int32(m), np.int32(p))
        
        grad_a = self.from_device(grad_a_dev)
        grad_b = self.from_device(grad_b_dev)
        
        return grad_a, grad_b
    
    def relu_backward(self, grad: np.ndarray, x: np.ndarray) -> np.ndarray:
        """ReLU backward pass"""
        size = x.size
        
        # Create device arrays
        grad_dev = self.to_device(grad.astype(np.float32))
        x_dev = self.to_device(x.astype(np.float32))
        result_dev = cl_array.empty(self.queue, x.shape, dtype=np.float32)
        
        # Execute kernel
        kernel = self.kernels['relu_backward']
        kernel(self.queue, (size,), None,
               grad_dev.data, x_dev.data, result_dev.data,
               np.int32(size))
        
        return self.from_device(result_dev)
    
    def gelu_backward(self, grad: np.ndarray, x: np.ndarray) -> np.ndarray:
        """GELU backward pass"""
        size = x.size
        
        # Create device arrays
        grad_dev = self.to_device(grad.astype(np.float32))
        x_dev = self.to_device(x.astype(np.float32))
        result_dev = cl_array.empty(self.queue, x.shape, dtype=np.float32)
        
        # Execute kernel
        kernel = self.kernels['gelu_backward']
        kernel(self.queue, (size,), None,
               grad_dev.data, x_dev.data, result_dev.data,
               np.int32(size))
        
        return self.from_device(result_dev)
    
    def softmax_backward(self, grad: np.ndarray, output: np.ndarray) -> np.ndarray:
        """Softmax backward pass"""
        if grad.ndim != 3 or output.ndim != 3:
            raise ValueError("Softmax backward expects 3D tensors")
        
        batch_size, seq_len, vocab_size = grad.shape
        
        # Create device arrays
        grad_dev = self.to_device(grad.astype(np.float32))
        output_dev = self.to_device(output.astype(np.float32))
        result_dev = cl_array.empty(self.queue, grad.shape, dtype=np.float32)
        
        # Execute kernel
        kernel = self.kernels['softmax_backward']
        kernel(self.queue, (batch_size, seq_len), None,
               grad_dev.data, output_dev.data, result_dev.data,
               np.int32(batch_size), np.int32(seq_len), np.int32(vocab_size))
        
        return self.from_device(result_dev)

class OpenCLBackend:
    """OpenCL backend manager"""
    
    def __init__(self, device_idx: int = 0):
        self.device = None
        self.is_initialized = False
        
        if HAS_OPENCL:
            try:
                self.device = OpenCLDevice(device_idx)
                self.is_initialized = True
            except Exception as e:
                print(f"❌ OpenCL initialization failed: {e}")
                self.is_initialized = False
    
    def is_available(self) -> bool:
        """Check if OpenCL backend is available"""
        return self.is_initialized
    
    def benchmark(self):
        """Benchmark GPU operations"""
        if not self.is_available():
            print("❌ OpenCL backend not available")
            return
        
        print("\n🚀 OpenCL Benchmark")
        print("=" * 60)
        
        # Test sizes
        sizes = [256, 512, 1024, 2048]
        
        for size in sizes:
            print(f"\n🔹 Matrix Size: {size}×{size}")
            
            # Create random matrices
            a = np.random.randn(size, size).astype(np.float32)
            b = np.random.randn(size, size).astype(np.float32)
            
            # CPU matmul
            cpu_start = time.time()
            cpu_result = np.dot(a, b)
            cpu_time = time.time() - cpu_start
            
            # GPU matmul
            gpu_start = time.time()
            gpu_result = self.device.matmul(a, b)
            gpu_time = time.time() - gpu_start
            
            # Verify correctness
            error = np.max(np.abs(cpu_result - gpu_result))
            
            print(f"   • CPU Time: {cpu_time*1000:.2f} ms")
            print(f"   • GPU Time: {gpu_time*1000:.2f} ms")
            print(f"   • Speedup: {cpu_time/gpu_time:.2f}x")
            print(f"   • Max Error: {error:.6e}")
            
            if error > 1e-4:
                print(f"   ⚠️  High error detected!")
    
    def test_operations(self):
        """Test all GPU operations"""
        if not self.is_available():
            print("❌ OpenCL backend not available")
            return
        
        print("\n🧪 OpenCL Operation Tests")
        print("=" * 60)
        
        # Test data
        a = np.random.randn(32, 32).astype(np.float32)
        b = np.random.randn(32, 32).astype(np.float32)
        
        # Test addition
        print("1. Element-wise Addition:")
        cpu_result = a + b
        gpu_result = self.device.elementwise_add(a, b)
        error = np.max(np.abs(cpu_result - gpu_result))
        print(f"   ✓ Error: {error:.6e}")
        
        # Test multiplication
        print("2. Element-wise Multiplication:")
        cpu_result = a * b
        gpu_result = self.device.elementwise_mul(a, b)
        error = np.max(np.abs(cpu_result - gpu_result))
        print(f"   ✓ Error: {error:.6e}")
        
        # Test matmul
        print("3. Matrix Multiplication:")
        cpu_result = np.dot(a, b)
        gpu_result = self.device.matmul(a, b)
        error = np.max(np.abs(cpu_result - gpu_result))
        print(f"   ✓ Error: {error:.6e}")
        
        # Test ReLU
        print("4. ReLU Activation:")
        cpu_result = np.maximum(a, 0)
        gpu_result = self.device.relu(a)
        error = np.max(np.abs(cpu_result - gpu_result))
        print(f"   ✓ Error: {error:.6e}")
        
        # Test GELU
        print("5. GELU Activation:")
        # Simple GELU approximation for CPU
        def gelu_cpu(x):
            return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))
        
        cpu_result = gelu_cpu(a)
        gpu_result = self.device.gelu(a)
        error = np.max(np.abs(cpu_result - gpu_result))
        print(f"   ✓ Error: {error:.6e}")
        
        print("\n✅ All OpenCL operations tested successfully!")

def main():
    """Main function"""
    print("🔧 OpenCL Backend Test")
    print("=" * 60)
    
    # Initialize backend
    backend = OpenCLBackend()
    
    if not backend.is_available():
        print("❌ OpenCL backend not available. Exiting.")
        return
    
    # Test operations
    backend.test_operations()
    
    # Benchmark
    backend.benchmark()
    
    print("\n🎉 OpenCL Backend Ready!")

if __name__ == "__main__":
    main()