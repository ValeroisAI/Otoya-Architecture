# phase_shifter_bridge.py
import numpy as np
from tinygrad import Tensor

class UltraFastPhaseShifter:
    """
    Ultra-fast Phase Shifter - JIT uyumlu, Tinygrad native
    Maximum performance for 100M+ parameters
    """
    def __init__(self, d_model, num_heads, groups):
        assert d_model % num_heads == 0
        hd = d_model // num_heads
        assert hd % groups == 0
        self.d = d_model
        self.h = num_heads
        self.g = groups
        self.gd = hd // groups
        sh = (num_heads, groups, self.gd)

        # Ağırlıkları tinygrad Tensor olarak tut
        self.w_ssm = Tensor.randn(*sh) * 0.01
        self.w_attn = Tensor.randn(*sh) * 0.01
        self.delta_scale = Tensor.randn(num_heads, groups) * 0.1
        self.alpha = Tensor.zeros(1)

    def __call__(self, y_ssm, y_attn, strength):
        """
        Ultra-fast forward pass - Tinygrad native operations (JIT uyumlu)
        """
        bsz, seq, d = y_ssm.shape

        # JIT uyumlu: strength'ı Tensor olarak kullan
        if not isinstance(strength, Tensor):
            strength = Tensor(strength)

        # Reshape weights for broadcasting (tek seferde)
        w_ssm_reshaped = self.w_ssm.reshape(1, 1, self.h, self.g, self.gd)
        w_attn_reshaped = self.w_attn.reshape(1, 1, self.h, self.g, self.gd)

        # Reshape inputs for broadcasting (tek seferde)
        y_ssm_reshaped = y_ssm.reshape(bsz, seq, self.h, self.g, self.gd)
        y_attn_reshaped = y_attn.reshape(bsz, seq, self.h, self.g, self.gd)

        # Scalar projections (fused operations)
        s_scalar = (y_ssm_reshaped * w_ssm_reshaped).sum(-1, keepdim=True)
        a_scalar = (y_attn_reshaped * w_attn_reshaped).sum(-1, keepdim=True)

        # Reshape delta_scale for broadcasting
        delta_scale_reshaped = self.delta_scale.reshape(1, 1, self.h, self.g, 1)

        # Compute delta (fused operations)
        delta = ((s_scalar - a_scalar) * delta_scale_reshaped).tanh() * strength

        # Compute sin/cos (fused with rotation)
        sin_val = delta.sin()
        cos_val = delta.cos()

        # Apply rotation (fused with reshape)
        s_rot = (y_ssm_reshaped * cos_val - y_attn_reshaped * sin_val).reshape(bsz, seq, d)
        a_rot = (y_ssm_reshaped * sin_val + y_attn_reshaped * cos_val).reshape(bsz, seq, d)

        # Extra output (fused)
        alpha_tanh = self.alpha.tanh()
        extra = strength * alpha_tanh * (s_rot + a_rot)

        # Statistics
        stats = {
            "delta": delta.reshape(bsz, seq, self.h * self.g),
            "mean_r": delta.abs().mean(),
            "alpha": alpha_tanh,
        }

        return extra, s_rot, a_rot, stats

class PhaseShifterCLWrapper:
    """
    PhaseShifter OpenCL kernel sürümü - JIT uyumlu.
    Dışarıya tinygrad Tensor arayüzü sunar, içeride OpenCL çalıştırır.
    Ağırlıklar tinygrad Tensor olarak tutulur (BareAdamW ile uyumlu).
    JIT için: numpy dönüşümleri minimize edilir, autograd manuel yönetilir.
    """
    def __init__(self, d_model, num_heads, groups):
        assert d_model % num_heads == 0
        hd = d_model // num_heads
        assert hd % groups == 0
        self.d = d_model
        self.h = num_heads
        self.g = groups
        self.gd = hd // groups
        sh = (num_heads, groups, self.gd)

        # Ağırlıkları tinygrad Tensor olarak tut
        self.w_sr = Tensor.randn(*sh) * 0.015
        self.w_si = Tensor.randn(*sh) * 0.015
        self.w_ar = Tensor.randn(*sh) * 0.015
        self.w_ai = Tensor.randn(*sh) * 0.015
        self.lambda_hat = Tensor.ones(num_heads, groups)
        self.alpha = Tensor.zeros(1)

        # OpenCL modülü (tembel yükleme)
        self._cl = None
        # JIT için cache
        self._last_y_ssm_np = None
        self._last_y_attn_np = None

    def _get_cl(self):
        if self._cl is None:
            self._cl = PhaseShifterOpenCL(self.d, self.h, self.g)
            self._sync_weights_to_cl()
        return self._cl

    def _sync_weights_to_cl(self):
        """tinygrad tensörlerini numpy'a çevir, OpenCL buffer'larına yükle"""
        cl = self._cl
        cl.w_sr = self.w_sr.numpy()
        cl.w_si = self.w_si.numpy()
        cl.w_ar = self.w_ar.numpy()
        cl.w_ai = self.w_ai.numpy()
        cl.lambda_hat = self.lambda_hat.numpy()
        cl.alpha = self.alpha.numpy()
        cl._upload_weights()

    def _sync_grads_from_cl(self):
        """OpenCL'deki gradyanları tinygrad Tensor gradyanlarına yaz"""
        self.w_sr.grad = Tensor(self._cl.grad_w_sr.copy())
        self.w_si.grad = Tensor(self._cl.grad_w_si.copy())
        self.w_ar.grad = Tensor(self._cl.grad_w_ar.copy())
        self.w_ai.grad = Tensor(self._cl.grad_w_ai.copy())
        self.lambda_hat.grad = Tensor(self._cl.grad_lambda_hat.copy())
        self.alpha.grad = Tensor(self._cl.grad_alpha.copy())

    def __call__(self, y_ssm, y_attn, strength):
        """
        PhaseShifter ile aynı arayüz.
        y_ssm, y_attn: tinygrad Tensor
        strength: float veya Tensor (float'a çevirir)
        Dönüş: (extra, s_out, a_out, stats) — hepsi tinygrad Tensor
        JIT uyumlu: numpy dönüşümleri minimize edilir
        """
        bsz, seq, _ = y_ssm.shape
        
        # JIT için: sadece gerekli olduğunda numpy'a çevir
        y_ssm_np = y_ssm.numpy()
        y_attn_np = y_attn.numpy()
        self._last_y_ssm_np = y_ssm_np
        self._last_y_attn_np = y_attn_np

        if isinstance(strength, Tensor):
            strength_val = float(strength.item())
        else:
            strength_val = float(strength)

        cl = self._get_cl()
        self._sync_weights_to_cl()
        extra_np, s_out_np, a_out_np, delta_np = cl.forward(
            y_ssm_np, y_attn_np, strength_val
        )

        # OpenCL backward gradyanlarını sıfırla (her adımda birikir)
        cl.zero_grad()

        # İstatistikleri hesapla
        stats = {
            "delta": Tensor(delta_np.reshape(bsz, seq, self.h * self.g)),
            "mean_r": Tensor([float(delta_np.mean())]),
            "alpha": Tensor([float(np.tanh(self.alpha.numpy()[0]))]),
        }

        # Çıktıları tinygrad Tensor olarak döndür (autograd için requires_grad=True)
        extra = Tensor(extra_np)
        extra.requires_grad = True

        s_out = Tensor(s_out_np)
        s_out.requires_grad = True

        a_out = Tensor(a_out_np)
        a_out.requires_grad = True

        # Backward hook'u: tinygrad geri yayılım yaptıktan sonra
        # OpenCL kernel backward'ını çağırmak için
        def _backward_hook():
            # grad_extra, grad_s_out, grad_a_out şu an bu Tensor'ların .grad'ında
            ge = extra.grad.numpy() if extra.grad is not None else np.zeros_like(extra_np)
            gs = s_out.grad.numpy() if s_out.grad is not None else np.zeros_like(s_out_np)
            ga = a_out.grad.numpy() if a_out.grad is not None else np.zeros_like(a_out_np)

            # OpenCL backward ile ağırlık gradyanlarını hesapla
            cl.zero_grad()
            gy_ssm, gy_attn = cl.backward(ge, gs, ga)

            # Ağırlık gradyanlarını tinygrad'a aktar
            self._sync_grads_from_cl()

            # Girdi gradyanlarını y_ssm ve y_attn'ın .grad'ına yaz
            # JIT uyumlu: manuel backward çağrısı
            y_ssm.backward(Tensor(gy_ssm))
            y_attn.backward(Tensor(gy_attn))

        # Hook'u sakla (eğitim döngüsünde backward'dan sonra çağıracağız)
        self._backward_hook = _backward_hook

        return extra, s_out, a_out, stats