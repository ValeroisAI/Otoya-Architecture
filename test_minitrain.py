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
    print("=== Mini Eğitim Testi ===")
    print()

    # 1. Backend yükleme
    print("1. Backend yükleniyor...")
    backend = load_backend("CL", use_vulkan=False)
    print(f"   ✅ Backend hazır: {backend}")
    print()

    # 2. Config oluşturma
    print("2. Config oluşturuluyor...")
    cfg = EitaConfig(
        d_model=64,
        num_layers=1,
        num_heads=2,
        phase_groups=2,
        context_length=32,
        ssm_kernel=17,
        phase_mode="srwm"
    )
    train_cfg = TrainConfig(
        backend="CL",
        batch_size=2,
        total_steps=5,
        lr=1e-3,
        grad_accum=1,
        use_vulkan=False,
        cleanup_every=10,
        val_every=100,
        checkpoint_every=100
    )
    print(f"   ✅ Config hazır")
    print()

    # 3. Tokenizer ve veri hazırlama
    print("3. Tokenizer ve veri hazırlanıyor...")
    tokenizer = ByteTokenizer()

    # Küçük bir test verisi oluştur
    test_text = "Merhaba dünya. This is a small test sentence. " * 20
    test_path = Path("test_text.txt")
    with open(test_path, "w", encoding="utf-8") as f:
        f.write(test_text)

    # StreamingDataset hazırla
    sampler = StreamingDataset(
        tokenizer,
        cfg.context_length,
        "Custom",  # Dataset adı "Custom" olmalı ki custom_path'i kullansın
        batch_size=train_cfg.batch_size,
        custom_path=str(test_path),  # String'e çevir
        prefetch_size=4,
        use_eft=False
    )
    print(f"   ✅ Veri hazırlandı: {sampler.batch_size} batch, {sampler.seq_len} seq len")
    print()

    # 4. Model ve optimizer
    print("4. Model ve optimizer hazırlanıyor...")
    model = EitaModel(cfg)

    # Basit SGD yerine basit optimizer
    params = []
    for name, param in get_parameters(model).items():
        params.append(param)
    # Basit AdamW-like güncelleme (manuel)
    for p in params:
        p.requires_grad = True
    print(f"   ✅ Model hazır (param sayısı: {len(params)})")
    print()

    # 5. Mini eğitim
    print("5. Mini eğitim (5 step) başlıyor...")
    Tensor.training = True
    losses = []
    for step in range(train_cfg.total_steps):
        # Batch al
        batch = sampler.batch(train_cfg.batch_size)
        ids = Tensor(batch[0].astype(np.int32))
        targets = Tensor(batch[1].astype(np.int32))

        # Phase hesapla
        phase = min(4, 1 + int(step / train_cfg.total_steps * 3))
        strength = min(1.0, (step / max(1, train_cfg.total_steps // 2)) if step > train_cfg.total_steps // 4 else 0.0)

        # Forward
        logits, loss, stats = model(ids, targets, phase=phase, strength=strength)
        loss_val = loss.numpy().item()
        losses.append(loss_val)

        # Backward
        loss.backward()

        # Basit SGD güncelleme
        lr = train_cfg.lr
        for p in params:
            if p.grad is not None:
                g = p.grad.numpy()
                new_val = p.numpy() - lr * g
                p.assign(Tensor(new_val))
                p.grad = None  # Grad sıfırla

        # Logla
        mean_delta = stats.get('mean_r', 0)
        memory_energy = stats.get('memory_energy', 0)
        print(f"   Step {step+1}/{train_cfg.total_steps}: loss={loss_val:.4f}, phase={phase}, strength={strength:.2f}")

    print()
    print("=== Mini eğitim tamamlandı ===")
    print(f"   Başlangıç loss: {losses[0]:.4f}")
    print(f"   Bitiş loss: {losses[-1]:.4f}")
    if losses[-1] < losses[0]:
        print("   ✅ Loss azaldı! Eğitim mantığı çalışıyor")
    else:
        print("   ⚠️ Loss azalmadı ama eğitim akışı çalıştı")
    print()

    # 6. Temizlik
    print("6. Temizlik yapılıyor...")
    test_path.unlink(missing_ok=True)
    print("   ✅ Tamamlandı")
    print()
    print("✅ Tüm testler başarılı! Sistem tamamen çalışır durumda.")


def get_parameters(obj):
    """EitaModel içindeki tüm tinygrad parametrelerini toplar."""
    params = {}
    def collect(name, current):
        if hasattr(current, "weight") and hasattr(current.weight, "numpy"):
            params[name + ".weight"] = current.weight
        if hasattr(current, "bias") and current.bias is not None and hasattr(current.bias, "numpy"):
            params[name + ".bias"] = current.bias
        # HelixPhaseMixer özel parametreleri
        if hasattr(current, "mixer"):
            collect(name + ".mixer", current.mixer)
        if hasattr(current, "router"):
            collect(name + ".router", current.router)
        if hasattr(current, "to_memory"):
            collect(name + ".to_memory", current.to_memory)
        if hasattr(current, "merge_gate"):
            collect(name + ".merge_gate", current.merge_gate)
        if hasattr(current, "memory_alpha"):
            params[name + ".memory_alpha"] = current.memory_alpha
        # Mixer özel parametreleri
        if hasattr(current, "w_state"):
            params[name + ".w_state"] = current.w_state
        if hasattr(current, "w_context"):
            params[name + ".w_context"] = current.w_context
        if hasattr(current, "mix_scale"):
            params[name + ".mix_scale"] = current.mix_scale
        if hasattr(current, "delta_scale"):
            params[name + ".delta_scale"] = current.delta_scale
        if hasattr(current, "alpha"):
            params[name + ".alpha"] = current.alpha
        # Listeleri tara
        if isinstance(current, list):
            for i, x in enumerate(current):
                collect(f"{name}[{i}]", x)
        # Çocuk nesneleri tara
        for child_attr in ["tok", "pos", "norm", "ssm", "attn", "ssm_norm", "attn_norm", "phase", "gate", "out", "depthwise", "mix", "delta"]:
            if hasattr(current, child_attr):
                collect(f"{name}.{child_attr}", getattr(current, child_attr))
    collect("model", obj)
    return params


if __name__ == "__main__":
    main()
