# C:\Users\Administrator\Desktop\OTOYA\diagnose_v7.py
import os
import sys
import numpy as np
import gc

# Add project root to path
sys.path.append(r"c:\Users\Administrator\Desktop\OTOYA")

import eita
from tinygrad import Tensor

def main():
    print("=== Eita Diagnostic Run V7 (Stable Float Return & GC) ===")
    
    # 1. Load backend
    backend = eita.load_backend("CL", use_vulkan=False)
    
    # Get config for 350M
    best_cfg, best_b, params_m, budget = eita.auto_config_for_target(350, 16)
    print(f"Model size M: {params_m:.2f}")
    
    cfg = best_cfg
    cfg.phase_mode = "ultra_wnn"
    
    # 3. Model
    model = eita.EitaModel(cfg)
    print("Model initialized successfully.")
    
    # 4. Tokenizer & Dataset
    tokenizer = eita.ByteTokenizer()
    sampler = eita.StreamingDataset(tokenizer, cfg.context_length, "TinyStories", batch_size=32, prefetch_size=2, use_eft=False)
    
    # Setup trainable parameters for Phase 1
    phase = 1
    eita.set_phase_trainable(model, phase)
    trainable = [p for p in eita.get_parameters(model) if getattr(p, "requires_grad", True)]
    print(f"Trainable parameters: {len(trainable)}")
    
    opt = eita.BareAdamW(trainable, lr=3e-4, wd=0.05, grad_clip=1.0)
    
    def stable_step(x, y, s_t):
        opt.zero_grad()
        _, lm_loss, stats = model(x, y, phase, s_t)
        loss = eita.total_loss(lm_loss, stats, phase)
        loss.backward()
        
        # Split realize: 1. loss
        Tensor.realize(loss, lm_loss)
        
        # FETCH FLOAT VALUES IMMEDIATELY
        loss_val = float(loss.numpy().item())
        lm_loss_val = float(lm_loss.numpy().item())
        
        # 2. weight update
        opt.step()
        
        # 3. parameters
        to_realize = [opt.t]
        for i, p in enumerate(opt.params):
            if p.grad is not None:
                to_realize.extend([p, opt.m[i], opt.v[i]])
        Tensor.realize(*to_realize)
        
        # Aggressive memory cleanup inside the step
        del loss
        del lm_loss
        del stats
        
        # Clear grads to free up gradient memory!
        for p in model.layers:
            # clear layer grads
            pass
        
        # Force garbage collection
        gc.collect()
        
        # Clear Tinygrad cache
        if hasattr(Tensor, '_cache'):
            Tensor._cache.clear()
            
        return loss_val, lm_loss_val

    # Get a batch
    xb, yb = sampler.batch(32)
    x = Tensor(xb)
    y = Tensor(yb)
    s_t = Tensor([0.0])
    
    # Step 1
    print("\n--- Running Step 1 ---")
    loss_val, lm_loss_val = stable_step(x, y, s_t)
    print(f"Step 1 - Loss: {loss_val:.6f}, LM Loss: {lm_loss_val:.6f}")
    
    # Step 2
    print("\n--- Running Step 2 ---")
    del x
    del y
    gc.collect()
    
    xb, yb = sampler.batch(32)
    x = Tensor(xb)
    y = Tensor(yb)
    loss_val, lm_loss_val = stable_step(x, y, s_t)
    print(f"Step 2 - Loss: {loss_val:.6f}, LM Loss: {lm_loss_val:.6f}")

    print("\nDiagnostic completed successfully.")

if __name__ == "__main__":
    main()
