#!/usr/bin/env python3
from pathlib import Path
from eita import (
    load_backend,
    EitaConfig,
    EitaModel,
    ByteTokenizer,
    StreamingDataset,
    TrainConfig
)
from tinygrad import Tensor
import numpy as np
import random


def main():
    print("=== Eita System Working Test ===")
    print()

    # 1. Backend yükleme
    print("1. Backend yükleniyor...")
    backend = load_backend("CL", use_vulkan=False)
    print(f"   ✅ Backend hazır: {backend}")
    print()

    # 2. Config oluşturma
    print("2. Model config oluşturuluyor...")
    cfg = EitaConfig(
        d_model=64,
        num_layers=1,
        num_heads=2,
        phase_groups=2,
        context_length=32,
        ssm_kernel=17,
        phase_mode="srwm"
    )
    print(f"   ✅ Config hazır (phase_mode: {cfg.phase_mode})")
    print()

    # 3. Model oluşturma
    print("3. Model oluşturuluyor...")
    model = EitaModel(cfg)
    print("   ✅ Model oluşturuldu")
    print()

    # 4. Tokenizer ve veri hazırlama
    print("4. Tokenizer ve veri hazırlanıyor...")
    tokenizer = ByteTokenizer()
    test_text = "Merhaba dünya, bu bir test metnidir."
    test_tokens = tokenizer.encode(test_text)
    print(f"   ✅ Test metni tokenize edildi: {len(test_tokens)} token")
    print()

    # 5. Basit bir forward pass
    print("5. Forward pass çalıştırılıyor...")
    bsz = 1
    seq_len = len(test_tokens)
    # EitaModel'e uygun boyutlandırma: context_length sınırı
    if seq_len > cfg.context_length:
        test_tokens = test_tokens[:cfg.context_length]
        seq_len = cfg.context_length
        print(f"   ⚠️ Test metni context_length'e ({cfg.context_length}) kısaltıldı")
    ids = Tensor(np.array(test_tokens, dtype=np.int32)).unsqueeze(0)
    targets = ids
    Tensor.training = False
    # EitaModel signature: ids, targets, phase, strength, ...
    logits, loss, stats = model(ids, targets, phase=4, strength=0.8)
    Tensor.realize(logits, loss)
    print(f"   ✅ Forward pass başarılı")
    print(f"      Logits shape: {logits.shape}")
    print(f"      Loss scalar: {loss.numpy().item():.4f}")
    print()

    # 6. Basit bir backward pass
    print("6. Backward pass çalıştırılıyor...")
    Tensor.training = True
    # Tekrar forward (training=True)
    logits, loss, stats = model(ids, targets, phase=4, strength=0.8)
    loss.backward()
    print("   ✅ Backward pass başarılı")
    print()

    print("=== Tüm testler başarıyla geçti ===")
    print("✅ Sistem çalışır durumda!")


if __name__ == "__main__":
    main()
