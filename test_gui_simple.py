#!/usr/bin/env python3
"""
Test GUI functionality without actually opening the window
"""

import sys
sys.path.append('.')
import tkinter as tk

from ultimate_gui import UltimateOtoyaGUI

print("🧪 Testing GUI functionality...")

# Test 1: Check if all imports work
try:
    from ultimate_otoya import UltimateConfig, UltimateOtoyaModel, LightningOptimizer, AdaptiveGradientFlow
    from quantum_tensor import QuantumTensor
    from train_ultimate import UltimateTrainer
    from auto_optimizer import UltimateAutoOptimizer, OptimizationResult
    print("✅ All imports successful")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

# Test 2: Create a simple tkinter window to test GUI components
try:
    root = tk.Tk()
    root.withdraw()  # Don't show the window
    
    # Test GUI class creation
    gui = UltimateOtoyaGUI(root)
    print("✅ GUI class created successfully")
    
    # Test some GUI methods
    print(f"  • Training status: {gui.training_status}")
    print(f"  • Current step: {gui.current_step}")
    
    # Test auto optimizer
    optimizer = UltimateAutoOptimizer(max_vram_gb=15.5, max_ram_gb=15.5)
    print("✅ Auto optimizer created successfully")
    
    # Test optimization
    result = optimizer.optimize_for_target_params(target_params=100_000_000)
    print(f"✅ Optimization successful for 100M params")
    print(f"  • Estimated params: {result.estimated_params:,}")
    print(f"  • Estimated memory: {result.estimated_memory_gb:.1f}GB")
    print(f"  • Max batch size: {result.max_batch_size}")
    
    root.destroy()
    print("✅ GUI tests completed successfully")
    
except Exception as e:
    print(f"❌ GUI test error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n🎉 All GUI tests passed!")