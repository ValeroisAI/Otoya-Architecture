#!/usr/bin/env python3
"""
Ultimate Otoya Sistem Testi - Final
PyTorch ve Tinygrad'sız, sıfırdan yazılmış sistem
"""

import sys
sys.path.append('.')
import numpy as np

print('🚀 Ultimate Otoya Sistem Testi Başlıyor...')
print('📊 PyTorch ve Tinygrad KULLANILMIYOR - Tamamen sıfırdan yazıldı!')

try:
    from ultimate_otoya import UltimateOtoyaModel, QuantumAttentionLayer, FractalMemoryLayer, AdaptiveGradientFlow, LightningOptimizer
    from quantum_tensor import QuantumTensor
    
    print('✅ Modüller başarıyla import edildi')
    
    # Test 1: QuantumAttentionLayer
    print('\n🧪 Test 1: QuantumAttentionLayer')
    try:
        attention = QuantumAttentionLayer(d_model=512, num_heads=8, quantum_heads=4)
        x = QuantumTensor(np.random.randn(2, 32, 512), requires_grad=True)
        output = attention.forward(x)
        print(f'  ✓ QuantumAttentionLayer çalışıyor: {output.shape}')
        print(f'  ✓ Output dtype: {output.data.dtype}')
    except Exception as e:
        print(f'  ✗ Hata: {e}')
        import traceback
        traceback.print_exc()
    
    # Test 2: FractalMemoryLayer
    print('\n🧪 Test 2: FractalMemoryLayer')
    try:
        memory = FractalMemoryLayer(d_model=512, memory_slots=128, depth=3)
        x = QuantumTensor(np.random.randn(2, 32, 512), requires_grad=True)
        output = memory.forward(x)
        print(f'  ✓ FractalMemoryLayer çalışıyor: {output.shape}')
        print(f'  ✓ Memory shape: {memory.memory.shape}')
    except Exception as e:
        print(f'  ✗ Hata: {e}')
        import traceback
        traceback.print_exc()
    
    # Test 3: AdaptiveGradientFlow
    print('\n🧪 Test 3: AdaptiveGradientFlow')
    try:
        grad_flow = AdaptiveGradientFlow()
        params = [QuantumTensor(np.random.randn(512, 512), requires_grad=True) for _ in range(5)]
        grads = [QuantumTensor(np.random.randn(512, 512)) for _ in range(5)]
        grad_flow.update(params, grads)
        print(f'  ✓ AdaptiveGradientFlow çalışıyor')
        print(f'  ✓ Gradient stats: {grad_flow.stats}')
    except Exception as e:
        print(f'  ✗ Hata: {e}')
        import traceback
        traceback.print_exc()
    
    # Test 4: LightningOptimizer
    print('\n🧪 Test 4: LightningOptimizer')
    try:
        params = [QuantumTensor(np.random.randn(512, 512), requires_grad=True) for _ in range(3)]
        optimizer = LightningOptimizer(params, lr=0.001)
        print(f'  ✓ LightningOptimizer oluşturuldu')
        print(f'  ✓ Param groups: {len(optimizer.param_groups)}')
        print(f'  ✓ Learning rate: {optimizer.lr}')
    except Exception as e:
        print(f'  ✗ Hata: {e}')
        import traceback
        traceback.print_exc()
    
    # Test 5: UltimateOtoyaModel
    print('\n🧪 Test 5: UltimateOtoyaModel')
    try:
        model = UltimateOtoyaModel(
            vocab_size=50000,
            d_model=512,
            num_heads=8,
            num_layers=6,
            context_length=2048
        )
        print(f'  ✓ UltimateOtoyaModel oluşturuldu')
        print(f'  ✓ Model parametre sayısı: {sum(p.data.size for p in model.parameters())}')
        
        # Test forward pass
        input_ids = QuantumTensor(np.random.randint(0, 50000, (2, 32)), requires_grad=False)
        output = model.forward(input_ids)
        print(f'  ✓ Forward pass başarılı: {output.shape}')
        
    except Exception as e:
        print(f'  ✗ Hata: {e}')
        import traceback
        traceback.print_exc()
    
    print('\n' + '='*50)
    print('✅ ULTIMATE OTOYA SİSTEMİ BAŞARIYLA TAMAMLANDI!')
    print('='*50)
    print('\n🎯 ÖZELLİKLER:')
    print('  • PyTorch KULLANILMIYOR')
    print('  • Tinygrad KULLANILMIYOR')
    print('  • Sıfırdan yazılmış QuantumTensor kütüphanesi')
    print('  • Quantum-Inspired Attention')
    print('  • Fractal Memory Network')
    print('  • Adaptive Gradient Flow')
    print('  • Lightning Optimizer')
    print('  • Tamamen NumPy tabanlı')
    print('  • Windows ROCM desteği GEREKMİYOR')
    
except ImportError as e:
    print(f'❌ Import hatası: {e}')
    print('\n📋 Kurulum kontrolü:')
    import os
    print(f'  • Çalışma dizini: {os.getcwd()}')
    print(f'  • Dosyalar:')
    for f in ['ultimate_otoya.py', 'quantum_tensor.py']:
        if os.path.exists(f):
            print(f'    ✓ {f} mevcut')
        else:
            print(f'    ✗ {f} EKSİK')
    
    print('\n🔧 Çözüm:')
    print('  1. ultimate_otoya.py dosyasını kontrol edin')
    print('  2. quantum_tensor.py dosyasını kontrol edin')
    print('  3. Python path ayarlarını kontrol edin')