import pyopencl as cl
import numpy as np

class PhaseShifterOpenCL:
    """
    PhaseShifter – özel OpenCL kernel kullanan bağımsız modül.
    Tinygrad bağımlılığı yoktur.
    """

    def __init__(self, d_model, num_heads, groups, ctx=None, queue=None):
        assert d_model % num_heads == 0
        hd = d_model // num_heads
        assert hd % groups == 0
        self.d   = d_model
        self.h   = num_heads
        self.g   = groups
        self.gd  = hd // groups

        # OpenCL bağlam
        if ctx is None:
            platform = cl.get_platforms()[0]
            devices = [platform.get_devices()[0]]  # ilk GPU
            self.ctx = cl.Context(devices)
            self.queue = cl.CommandQueue(self.ctx)
        else:
            self.ctx = ctx
            self.queue = queue

        # Kernel derleme ve ön-hazırlık
        with open("phase_shifter_kernel.cl", "r") as f:
            src = f.read()
        self.prg = cl.Program(self.ctx, src).build()
        self.knl_forward = cl.Kernel(self.prg, "phase_shifter_forward")
        self.knl_backward = cl.Kernel(self.prg, "phase_shifter_backward")

        # Ağırlık başlatma
        sh = (num_heads, groups, self.gd)
        self.w_sr       = np.random.randn(*sh).astype(np.float32) * 0.015
        self.w_si       = np.random.randn(*sh).astype(np.float32) * 0.015
        self.w_ar       = np.random.randn(*sh).astype(np.float32) * 0.015
        self.w_ai       = np.random.randn(*sh).astype(np.float32) * 0.015
        self.lambda_hat = np.ones((num_heads, groups), dtype=np.float32)
        self.alpha      = np.zeros(1, dtype=np.float32)

        # GPU buffer'lar
        self._upload_weights()
        self.zero_grad()

    def _upload_weights(self):
        mf = cl.mem_flags
        self.buf_w_sr       = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.w_sr)
        self.buf_w_si       = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.w_si)
        self.buf_w_ar       = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.w_ar)
        self.buf_w_ai       = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.w_ai)
        self.buf_lambda_hat = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.lambda_hat)

    def zero_grad(self):
        self.grad_w_sr       = np.zeros_like(self.w_sr)
        self.grad_w_si       = np.zeros_like(self.w_si)
        self.grad_w_ar       = np.zeros_like(self.w_ar)
        self.grad_w_ai       = np.zeros_like(self.w_ai)
        self.grad_lambda_hat = np.zeros_like(self.lambda_hat)
        self.grad_alpha      = np.zeros(1, dtype=np.float32)

    def forward(self, y_ssm_np, y_attn_np, strength):
        bsz, seq, d = y_ssm_np.shape
        h, g, gd = self.h, self.g, self.gd
        flat_size = bsz * seq * d
        hg_size = bsz * seq * h * g
        hgd_size = bsz * seq * h * g * gd

        mf = cl.mem_flags

        # Orijinal girdi buffer'ları (backward için sakla)
        self._y_ssm_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=y_ssm_np.ravel())
        self._y_attn_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=y_attn_np.ravel())

        # Çıktı buffer'ları
        buf_s_out = cl.Buffer(self.ctx, mf.WRITE_ONLY, flat_size * 4)
        buf_a_out = cl.Buffer(self.ctx, mf.WRITE_ONLY, flat_size * 4)
        buf_extra = cl.Buffer(self.ctx, mf.WRITE_ONLY, flat_size * 4)

        # Cache buffer'ları
        c = {}
        c["re_s"]      = cl.Buffer(self.ctx, mf.READ_WRITE, hg_size * 4)
        c["im_s"]      = cl.Buffer(self.ctx, mf.READ_WRITE, hg_size * 4)
        c["re_a"]      = cl.Buffer(self.ctx, mf.READ_WRITE, hg_size * 4)
        c["im_a"]      = cl.Buffer(self.ctx, mf.READ_WRITE, hg_size * 4)
        c["rs"]        = cl.Buffer(self.ctx, mf.READ_WRITE, hg_size * 4)
        c["ra"]        = cl.Buffer(self.ctx, mf.READ_WRITE, hg_size * 4)
        c["sin_d"]     = cl.Buffer(self.ctx, mf.READ_WRITE, hg_size * 4)
        c["conf"]      = cl.Buffer(self.ctx, mf.READ_WRITE, hg_size * 4)
        c["delta"]     = cl.Buffer(self.ctx, mf.READ_WRITE, hg_size * 4)
        c["cos_delta"] = cl.Buffer(self.ctx, mf.READ_WRITE, hg_size * 4)
        c["sin_delta"] = cl.Buffer(self.ctx, mf.READ_WRITE, hg_size * 4)
        c["s"]         = cl.Buffer(self.ctx, mf.READ_WRITE, hgd_size * 4)
        c["a"]         = cl.Buffer(self.ctx, mf.READ_WRITE, hgd_size * 4)

        self._cache = c
        self._buf_s_out = buf_s_out
        self._buf_a_out = buf_a_out
        self._bsz, self._seq = bsz, seq
        self._strength = np.float32(strength)
        self._alpha_tanh = np.float32(np.tanh(self.alpha[0]))

        # Kernel başlat
        self.knl_forward(
            self.queue, (hg_size,), None,
            self._y_ssm_buf, self._y_attn_buf,
            self.buf_w_sr, self.buf_w_si, self.buf_w_ar, self.buf_w_ai,
            self.buf_lambda_hat,
            self._alpha_tanh, self._strength,
            buf_s_out, buf_a_out, buf_extra,
            c["re_s"], c["im_s"], c["re_a"], c["im_a"],
            c["rs"], c["ra"], c["sin_d"], c["conf"],
            c["delta"], c["cos_delta"], c["sin_delta"],
            c["s"], c["a"],
            np.int32(bsz), np.int32(seq), np.int32(d),
            np.int32(h), np.int32(g), np.int32(gd)
        )

        # Sonuçları kopyala
        s_out_np = np.empty((bsz, seq, d), dtype=np.float32)
        a_out_np = np.empty((bsz, seq, d), dtype=np.float32)
        extra_np = np.empty((bsz, seq, d), dtype=np.float32)
        delta_np = np.empty(hg_size, dtype=np.float32)

        cl.enqueue_copy(self.queue, s_out_np, buf_s_out)
        cl.enqueue_copy(self.queue, a_out_np, buf_a_out)
        cl.enqueue_copy(self.queue, extra_np, buf_extra)
        cl.enqueue_copy(self.queue, delta_np, c["delta"])
        self.queue.finish()

        return extra_np, s_out_np, a_out_np, delta_np.reshape(bsz, seq, h * g)

    def backward(self, grad_extra_np, grad_s_out_np, grad_a_out_np):
        bsz, seq = self._bsz, self._seq
        d, h, g, gd = self.d, self.h, self.g, self.gd
        flat_size = bsz * seq * d
        hg_size = bsz * seq * h * g

        mf = cl.mem_flags

        # Gradyan girdileri
        buf_ge = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=grad_extra_np.ravel())
        buf_gs = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=grad_s_out_np.ravel())
        buf_ga = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=grad_a_out_np.ravel())

        # Gradyan çıktı buffer'ları
        buf_gy_ssm  = cl.Buffer(self.ctx, mf.WRITE_ONLY, flat_size * 4)
        buf_gy_attn = cl.Buffer(self.ctx, mf.WRITE_ONLY, flat_size * 4)
        buf_gw_sr   = cl.Buffer(self.ctx, mf.READ_WRITE, h * g * gd * 4)
        buf_gw_si   = cl.Buffer(self.ctx, mf.READ_WRITE, h * g * gd * 4)
        buf_gw_ar   = cl.Buffer(self.ctx, mf.READ_WRITE, h * g * gd * 4)
        buf_gw_ai   = cl.Buffer(self.ctx, mf.READ_WRITE, h * g * gd * 4)
        buf_glambda = cl.Buffer(self.ctx, mf.READ_WRITE, h * g * 4)
        buf_galpha  = cl.Buffer(self.ctx, mf.READ_WRITE, 4)

        # Sıfırla
        zero_hgd = np.zeros(h * g * gd, dtype=np.float32)
        zero_hg  = np.zeros(h * g, dtype=np.float32)
        zero_1   = np.zeros(1, dtype=np.float32)
        cl.enqueue_copy(self.queue, buf_gw_sr,   zero_hgd)
        cl.enqueue_copy(self.queue, buf_gw_si,   zero_hgd)
        cl.enqueue_copy(self.queue, buf_gw_ar,   zero_hgd)
        cl.enqueue_copy(self.queue, buf_gw_ai,   zero_hgd)
        cl.enqueue_copy(self.queue, buf_glambda, zero_hg)
        cl.enqueue_copy(self.queue, buf_galpha,  zero_1)

        c = self._cache

        self.knl_backward(
            self.queue, (hg_size,), None,
            c["re_s"], c["im_s"], c["re_a"], c["im_a"],
            c["rs"], c["ra"], c["sin_d"], c["conf"],
            c["delta"], c["cos_delta"], c["sin_delta"],
            c["s"], c["a"],
            self._buf_s_out, self._buf_a_out,
            buf_ge, buf_gs, buf_ga,
            self._y_ssm_buf, self._y_attn_buf,
            self.buf_w_sr, self.buf_w_si, self.buf_w_ar, self.buf_w_ai,
            self.buf_lambda_hat,
            buf_gy_ssm, buf_gy_attn,
            buf_gw_sr, buf_gw_si, buf_gw_ar, buf_gw_ai,
            buf_glambda, buf_galpha,
            self._alpha_tanh, self._strength, np.float32(self.alpha[0]),
            np.int32(bsz), np.int32(seq), np.int32(d),
            np.int32(h), np.int32(g), np.int32(gd)
        )

        # Kopyala ve biriktir
        gy_ssm  = np.empty(flat_size, dtype=np.float32)
        gy_attn = np.empty(flat_size, dtype=np.float32)
        gw_sr   = np.empty(h * g * gd, dtype=np.float32)
        gw_si   = np.empty(h * g * gd, dtype=np.float32)
        gw_ar   = np.empty(h * g * gd, dtype=np.float32)
        gw_ai   = np.empty(h * g * gd, dtype=np.float32)
        glambda = np.empty(h * g, dtype=np.float32)
        galpha  = np.empty(1, dtype=np.float32)

        cl.enqueue_copy(self.queue, gy_ssm,  buf_gy_ssm)
        cl.enqueue_copy(self.queue, gy_attn, buf_gy_attn)
        cl.enqueue_copy(self.queue, gw_sr,   buf_gw_sr)
        cl.enqueue_copy(self.queue, gw_si,   buf_gw_si)
        cl.enqueue_copy(self.queue, gw_ar,   buf_gw_ar)
        cl.enqueue_copy(self.queue, gw_ai,   buf_gw_ai)
        cl.enqueue_copy(self.queue, glambda, buf_glambda)
        cl.enqueue_copy(self.queue, galpha,  buf_galpha)
        self.queue.finish()

        self.grad_w_sr       += gw_sr.reshape(self.h, self.g, self.gd)
        self.grad_w_si       += gw_si.reshape(self.h, self.g, self.gd)
        self.grad_w_ar       += gw_ar.reshape(self.h, self.g, self.gd)
        self.grad_w_ai       += gw_ai.reshape(self.h, self.g, self.gd)
        self.grad_lambda_hat += glambda.reshape(self.h, self.g)
        self.grad_alpha      += galpha

        return gy_ssm.reshape(bsz, seq, d), gy_attn.reshape(bsz, seq, d)

    def update_weights_sgd(self, lr=3e-4):
        self.w_sr       -= lr * self.grad_w_sr
        self.w_si       -= lr * self.grad_w_si
        self.w_ar       -= lr * self.grad_w_ar
        self.w_ai       -= lr * self.grad_w_ai
        self.lambda_hat -= lr * self.grad_lambda_hat
        self.alpha      -= lr * self.grad_alpha
        self._upload_weights()

    # BareAdamW entegrasyon örneği
    def update_weights_adamw(self, lr=3e-4, beta1=0.9, beta2=0.999, eps=1e-8, wd=0.01, t=1):
        if not hasattr(self, 'adam_m'):
            self.adam_m = [np.zeros_like(p) for p in
                           [self.w_sr, self.w_si, self.w_ar, self.w_ai, self.lambda_hat, self.alpha]]
            self.adam_v = [np.zeros_like(p) for p in
                           [self.w_sr, self.w_si, self.w_ar, self.w_ai, self.lambda_hat, self.alpha]]
        grads  = [self.grad_w_sr, self.grad_w_si, self.grad_w_ar, self.grad_w_ai,
                  self.grad_lambda_hat, self.grad_alpha]
        params = [self.w_sr, self.w_si, self.w_ar, self.w_ai, self.lambda_hat, self.alpha]
        for i, (p, g) in enumerate(zip(params, grads)):
            self.adam_m[i] = beta1 * self.adam_m[i] + (1 - beta1) * g
            self.adam_v[i] = beta2 * self.adam_v[i] + (1 - beta2) * (g ** 2)
            m_hat = self.adam_m[i] / (1 - beta1 ** t)
            v_hat = self.adam_v[i] / (1 - beta2 ** t)
            params[i] -= lr * (m_hat / (np.sqrt(v_hat) + eps) + wd * params[i])
        self._upload_weights()