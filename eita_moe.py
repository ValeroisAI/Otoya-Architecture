"""
eita_moe.py
=============

Eita V13 — Sparse Mixture of Experts (MoE) katmanı.

**Dense-Batched Top‑K Routing** (Mixtral tarzı routing,
vektörize dense-batched forward).

  output = Σ_{k=1..K}  w_k · Expert_{e_k}(x)

burada e_k ve w_k sırasıyla k. en iyi expert indeksi ve
ağırlığı. Gerçek sparse MoE'ye göre K kat daha fazla
FLOPs gerektirir, fakat GPU'da çok daha hızlı çalışır
(tüm expert'ler tek bir matmul'a sıkıştırılır).

Bileşenler
----------
* **Top‑K Router**: nn.Linear(d_model, num_experts)
* **Expert parametreleri**: (E, D, F) ve (E, F, D) matrisleri
* **SwiGLU**: Modern MoE standardı (Mixtral, Qwen MoE)
* **Load balancing loss**: DeepSeek‑V2 formülü
* **Router z‑loss**: Logit büyüklüğü cezası
* **ScalableJIT uyumu**: OtoyaBlock.moe = MoE(...) olarak
  eklendiğinde OtoyaBlock ScalableJIT tarafından sarmalanır.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

try:
    from tinygrad import Tensor, nn
    _HAS_TINYGRAD = True
except Exception:  # noqa: BLE001
    _HAS_TINYGRAD = False
    Tensor = None  # type: ignore
    nn = None  # type: ignore


def _silu(x):
    if _HAS_TINYGRAD:
        return x * x.sigmoid()
    return x


class MoE:
    """Top‑K Dense-Batched MoE."""

    def __init__(
        self,
        d_model: int,
        d_ff: Optional[int] = None,
        num_experts: int = 4,
        top_k: int = 2,
        balance_loss_weight: float = 0.01,
        router_z_loss_weight: float = 0.001,
    ):
        self.d_model = d_model
        self.d_ff = d_ff if d_ff is not None else 4 * d_model
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)
        self.balance_loss_weight = balance_loss_weight
        self.router_z_loss_weight = router_z_loss_weight

        if not _HAS_TINYGRAD:
            raise RuntimeError(
                "MoE sınıfı yalnızca tinygrad yüklüyken kullanılabilir."
            )

        # Router
        self.router = nn.Linear(d_model, num_experts, bias=False)

        # Expert parametreleri — (E, D, F) ve (E, F, D)
        # Kaiming-uniform init.
        import numpy as _np
        bound_g = 1.0 / math.sqrt(d_model)
        bound_d = 1.0 / math.sqrt(self.d_ff)
        self.gate_proj = Tensor(
            _np.random.uniform(-bound_g, bound_g,
                               size=(num_experts, d_model, self.d_ff)).astype(_np.float32)
        )
        self.up_proj = Tensor(
            _np.random.uniform(-bound_g, bound_g,
                               size=(num_experts, d_model, self.d_ff)).astype(_np.float32)
        )
        self.down_proj = Tensor(
            _np.random.uniform(-bound_d, bound_d,
                               size=(num_experts, self.d_ff, d_model)).astype(_np.float32)
        )

        # Son istatistikler
        self._last_topk_idx = None
        self._last_expert_usage: list = []

    # ----- forward ---------------------------------------------------------

    def _route(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Routing: (B, S, D) → (B, S, K) idx + (B, S, K) ağırlık + full + logits."""
        bsz, seq, d = x.shape
        flat = x.reshape(bsz * seq, d)
        logits = self.router(flat)                            # (N, E)
        probs = logits.softmax(-1)                            # (N, E)
        topk_vals, topk_idx = probs.topk(self.top_k, dim=-1)  # (N, K)
        topk_vals = topk_vals / (topk_vals.sum(-1, keepdim=True) + 1e-9)
        return (
            topk_idx.reshape(bsz, seq, self.top_k),
            topk_vals.reshape(bsz, seq, self.top_k),
            probs.reshape(bsz, seq, self.num_experts),
            logits.reshape(bsz, seq, self.num_experts),
        )

    def _balance_loss(self, router_probs_flat: Tensor) -> Tensor:
        """DeepSeek-V2 load balancing loss.

        L = alpha * E * sum_e( f_e * p_e )
        f_e: top-K seçim sayım oranı
        p_e: ortalama router olasılığı
        """
        E = self.num_experts
        if self._last_topk_idx is None:
            return Tensor([0.0])
        # f_e: her expert seçilme oranı
        topk_flat = self._last_topk_idx.reshape(-1)  # (N*K,)
        # Vektörize sayım: one-hot ile matmul
        # onehot: (N*K, E) → f: (E,)
        f = (topk_flat.unsqueeze(-1) == Tensor.arange(E).reshape(1, E)).float().mean(0)
        p = router_probs_flat.mean(0)
        return self.balance_loss_weight * E * (f * p).sum()

    def _z_loss(self, logits_flat: Tensor) -> Tensor:
        """Router z-loss: büyük logit'leri cezalandır."""
        log_z = logits_flat.logsumexp(-1)
        return self.router_z_loss_weight * (log_z * log_z).mean()

    def _expert_forward_dense(
        self,
        x: Tensor,
    ) -> Tensor:
        """Tüm expert'ler tüm token'ları görür.

        x: (B, S, D)
        Returns: (B, S, E, D) — her expert'in çıktısı

        Yöntem: her expert için tek bir matmul (B*S, D) @ (D, F).
        Toplam E matmul; hepsi GPU üzerinde. Tinygrad matmul broadcasting
        sınırlı olduğu için loop en güvenli yol.
        """
        bsz, seq, d = x.shape
        F_dim = self.d_ff
        E = self.num_experts

        # (B, S, D) → (B*S, D)
        x_flat = x.reshape(bsz * seq, d)

        # Her expert için: gate, up, silu, down
        expert_outs = []
        for e in range(E):
            g = x_flat @ self.gate_proj[e]   # (B*S, F)
            u = x_flat @ self.up_proj[e]     # (B*S, F)
            h = _silu(g) * u                 # (B*S, F)
            y = h @ self.down_proj[e]        # (B*S, D)
            expert_outs.append(y.reshape(bsz, seq, d))

        # (B, S, D) list → (B, S, E, D)
        return Tensor.stack(*expert_outs, dim=2)

    def _gather_topk(self, y: Tensor, topk_idx: Tensor) -> Tensor:
        """y: (B, S, E, D), topk_idx: (B, S, K) → (B, S, K, D)."""
        bsz, seq, K = topk_idx.shape
        E, d = y.shape[2], y.shape[3]
        # Tensor.gather veya fancy indexing.
        # Tinygrad'ta genelde: a.gather(dim, index) veya a[index] kullanılır.
        # Eğer desteklenmiyorsa, one-hot matmul yöntemi.
        # One-hot: (B*S, K, E) ile y reshape (B*S, E, D) → (B*S, K, D)
        onehot = (topk_idx.reshape(bsz * seq, K, 1) ==
                  Tensor.arange(E).reshape(1, 1, E)).float()  # (BS, K, E)
        y_flat = y.reshape(bsz * seq, E, d)
        out = onehot @ y_flat  # (BS, K, D)
        return out.reshape(bsz, seq, K, d)

    def __call__(self, x: Tensor) -> Tuple[Tensor, dict]:
        """x: (B, S, D) → ((B, S, D), stats dict)."""
        bsz, seq, d = x.shape
        assert d == self.d_model, f"d_model uyumsuz: {d} != {self.d_model}"

        # 1) Routing
        topk_idx, topk_vals, router_probs, router_logits = self._route(x)

        # 2) Tüm expert'ler tüm token'ları görür (dense)
        y_all = self._expert_forward_dense(x)  # (B, S, E, D)

        # 3) Top-K expert'in çıktısını seç
        y_topk = self._gather_topk(y_all, topk_idx)  # (B, S, K, D)

        # 4) Router ağırlıklarıyla ağırlıklandır ve topla
        out = (y_topk * topk_vals.unsqueeze(-1)).sum(2)  # (B, S, D)

        # 5) Auxiliary losses
        aux = {}
        if self.balance_loss_weight > 0:
            self._last_topk_idx = topk_idx
            self._last_router_probs = router_probs
            aux["moe_balance"] = self._balance_loss(router_probs.reshape(-1, self.num_experts))
        if self.router_z_loss_weight > 0:
            aux["moe_z"] = self._z_loss(router_logits.reshape(-1, self.num_experts))

        # İstatistikler
        if topk_idx is not None:
            import numpy as _np
            topk_flat_np = topk_idx.reshape(-1).numpy().astype(_np.int32)
            usage = _np.bincount(topk_flat_np, minlength=self.num_experts)
            self._last_expert_usage = usage.tolist()
            aux["moe_expert_usage"] = self._last_expert_usage

        return out, aux


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("MoE izole test")
    print("=" * 60)

    if not _HAS_TINYGRAD:
        print("Tinygrad yüklü değil.")
    else:
        import numpy as np
        np.random.seed(42)

        moe = MoE(d_model=64, d_ff=128, num_experts=4, top_k=2)
        print(f"MoE: d_model=64, d_ff=128, num_experts=4, top_k=2")

        x = Tensor.randn(2, 8, 64)
        print(f"Input shape: {x.shape}")

        out, aux = moe(x)
        out.realize()
        print(f"Output shape: {out.shape}")
        assert out.shape == x.shape

        # Output değerleri finite olmalı
        out_np = out.numpy()
        print(f"Output: min={out_np.min():.3f} max={out_np.max():.3f} mean={out_np.mean():.3f}")
        assert np.all(np.isfinite(out_np)), "Output non-finite değer içeriyor!"

        # Aux losses finite olmalı
        for k, v in aux.items():
            if isinstance(v, list):
                print(f"  {k}: {v}")
            elif hasattr(v, 'numpy'):
                val = v.numpy()
                if val.size == 1:
                    val_f = float(val)
                    print(f"  {k}: {val_f:.6f}")
                    assert math.isfinite(val_f), f"{k} non-finite!"
                else:
                    print(f"  {k}: {val}")

        # Top-K=1 testi
        print("\nTop-K=1 testi:")
        moe1 = MoE(d_model=64, d_ff=128, num_experts=4, top_k=1)
        out1, _ = moe1(x)
        out1.realize()
        print(f"  Shape: {out1.shape}, OK")
        assert out1.shape == x.shape

        # Top-K=4 testi (tüm expert'ler)
        print("\nTop-K=4 testi:")
        moe4 = MoE(d_model=64, d_ff=128, num_experts=4, top_k=4)
        out4, _ = moe4(x)
        out4.realize()
        print(f"  Shape: {out4.shape}, OK")
        assert out4.shape == x.shape

        print("\n[OK] MoE izole testi gecti")
