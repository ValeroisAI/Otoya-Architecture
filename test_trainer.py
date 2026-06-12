#!/usr/bin/env python3
import threading
import queue
from eita import Trainer, TrainConfig, EitaConfig

import logging

def main():
    print("=== Trainer JIT Testi ===")
    import eita
    
    # Logger'ı konsola da basalım
    ch = logging.StreamHandler()
    eita.logger.addHandler(ch)
    
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
    
    q = queue.Queue()
    stop_event = threading.Event()
    
    # Custom dataset için bir dosya oluşturalım
    with open("test_trainer_data.txt", "w", encoding="utf-8") as f:
        f.write("Merhaba dünya bu bir trainer testi " * 50)
        
    # Dataset string ismini test_trainer_data.txt olarak ver
    # _run içinde config üzerinden geçiyoruz, eita.py içinde varsayılan dataset ismi tiny_stories.txt.
    # Bu yüzden küçük bir monkeypatch
    import eita
    eita.DATASET_NAME = "test_trainer_data.txt"
    
    trainer = Trainer(cfg, train_cfg, None, q, stop_event)
    trainer.start()
    
    # Kuyruktan mesajları dinle
    while True:
        try:
            msg = q.get(timeout=60)
            if isinstance(msg, tuple):
                print(f"[{msg[0]}] {msg[1]}")
                if msg[0] == "stopped" or msg[0] == "error":
                    break
                if msg[0] == "metrics":
                    # Stop after 5 steps
                    if msg[1].get("step", 0) >= 5:
                        stop_event.set()
        except queue.Empty:
            print("Timeout!")
            break

if __name__ == "__main__":
    main()
