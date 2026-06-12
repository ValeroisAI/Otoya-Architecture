// phase_shifter_kernel.cl
// PhaseShifter forward + backward özel OpenCL implementasyonu
// AMD RDNA 4 (RX 9070 XT) – float32, atomic float toplama destekli.
// OPTIMIZED: Reduced memory operations, better cache utilization

#pragma OPENCL EXTENSION cl_khr_int64_extended_atomics : enable
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

// ------------------- Yardımcı Fonksiyonlar -------------------

// Float atomic add (global bellekte, cmpxchg döngüsü ile)
inline void atomic_add_float(volatile __global float *addr, float val) {
    volatile __global int *iaddr = (volatile __global int *)addr;
    int old, new;
    do {
        old = *iaddr;
        new = as_int(as_float(old) + val);
    } while (atomic_cmpxchg(iaddr, old, new) != old);
}

inline float sigmoid(float x) {
    return 1.0f / (1.0f + exp(-x));
}

inline float softplus(float x) {
    return log(1.0f + exp(x));
}

// Dallanmasız clip
inline float clip(float x, float min_val, float max_val) {
    return fmin(fmax(x, min_val), max_val);
}

// ------------------- Forward Kernel -------------------
__kernel void phase_shifter_forward(
    // Girdiler
    __global const float* y_ssm,
    __global const float* y_attn,
    __global const float* w_sr,
    __global const float* w_si,
    __global const float* w_ar,
    __global const float* w_ai,
    __global const float* lambda_hat,
    float alpha_tanh,
    float strength,
    // Çıktılar
    __global float* s_out,
    __global float* a_out,
    __global float* extra,
    // Cache (backward için saklanan ara değerler)
    __global float* cache_re_s,
    __global float* cache_im_s,
    __global float* cache_re_a,
    __global float* cache_im_a,
    __global float* cache_rs,
    __global float* cache_ra,
    __global float* cache_sin_d,
    __global float* cache_conf,
    __global float* cache_delta,
    __global float* cache_cos_delta,
    __global float* cache_sin_delta,
    __global float* cache_s,
    __global float* cache_a,
    // Boyutlar
    int bsz, int seq, int d, int h, int g, int gd
) {
    int tid = get_global_id(0);
    int total_hg = h * g;
    int total_tiles = bsz * seq * total_hg;
    if (tid >= total_tiles) return;

    int hg_idx = tid % total_hg;
    int bt_idx = tid / total_hg;
    int b = bt_idx / seq;
    int t = bt_idx % seq;
    int hh = hg_idx / g;
    int gg = hg_idx % g;

    int out_base = (b * seq + t) * d + hh * (g * gd) + gg * gd;
    int w_base = (hh * g + gg) * gd;
    int cache_base = (b * seq + t) * total_hg * gd + hg_idx * gd;

    // OPTIMIZATION: Use local memory for better cache utilization
    __local float local_w_sr[128];
    __local float local_w_si[128];
    __local float local_w_ar[128];
    __local float local_w_ai[128];
    
    // Load weights into local memory (coalesced access)
    int lid = get_local_id(0);
    if (lid < gd) {
        local_w_sr[lid] = w_sr[w_base + lid];
        local_w_si[lid] = w_si[w_base + lid];
        local_w_ar[lid] = w_ar[w_base + lid];
        local_w_ai[lid] = w_ai[w_base + lid];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // Maksimum gd = 128 varsayıyoruz (yeterince büyük)
    float s[128], a_vals[128];

    // 1. Girdi kırpma ve cache
    for (int i = 0; i < gd; ++i) {
        s[i]      = clip(y_ssm[out_base + i], -8.0f, 8.0f);
        a_vals[i] = clip(y_attn[out_base + i], -8.0f, 8.0f);
        cache_s[cache_base + i] = s[i];
        cache_a[cache_base + i] = a_vals[i];
    }

    // 2. Kompleks projeksiyon (dot product) - OPTIMIZED: use local memory
    float re_s_raw = 0.0f, im_s_raw = 0.0f;
    float re_a_raw = 0.0f, im_a_raw = 0.0f;
    for (int i = 0; i < gd; ++i) {
        re_s_raw += s[i] * local_w_sr[i];
        im_s_raw += s[i] * local_w_si[i];
        re_a_raw += a_vals[i] * local_w_ar[i];
        im_a_raw += a_vals[i] * local_w_ai[i];
    }

    float re_s = clip(re_s_raw, -12.0f, 12.0f);
    float im_s = clip(im_s_raw, -12.0f, 12.0f);
    float re_a = clip(re_a_raw, -12.0f, 12.0f);
    float im_a = clip(im_a_raw, -12.0f, 12.0f);

    cache_re_s[tid] = re_s;
    cache_im_s[tid] = im_s;
    cache_re_a[tid] = re_a;
    cache_im_a[tid] = im_a;

    // 3. Büyüklükler
    float rs = sqrt(re_s * re_s + im_s * im_s + 1e-4f);
    float ra = sqrt(re_a * re_a + im_a * im_a + 1e-4f);
    float den = rs * ra + 1e-4f;

    cache_rs[tid] = rs;
    cache_ra[tid] = ra;

    // 4. Açı farkı
    float sin_d_raw = (im_s * re_a - re_s * im_a) / den;
    float sin_d = clip(sin_d_raw, -0.95f, 0.95f);
    cache_sin_d[tid] = sin_d;

    // 5. Güven
    float conf = sigmoid((rs + ra) * 0.5f - 0.1f);
    cache_conf[tid] = conf;

    // 6. Sıcaklık
    float temp = softplus(lambda_hat[hh * g + gg]) + 0.5f;

    // 7. Delta (döndürme açısı)
    float delta_raw = sin_d * conf / temp;
    float delta = clip(delta_raw, -1.0f, 1.0f) * strength;
    cache_delta[tid] = delta;

    // 8. Rotasyon trigonometrisi
    float cos_delta = cos(delta);
    float sin_delta_val = sin(delta);
    cache_cos_delta[tid] = cos_delta;
    cache_sin_delta[tid] = sin_delta_val;

    // 9. Rotasyon uygula ve çıktıları yaz
    for (int i = 0; i < gd; ++i) {
        float s_rot = s[i] * cos_delta - a_vals[i] * sin_delta_val;
        float a_rot = s[i] * sin_delta_val + a_vals[i] * cos_delta;
        s_out[out_base + i] = s_rot;
        a_out[out_base + i] = a_rot;
        extra[out_base + i] = strength * alpha_tanh * (s_rot + a_rot);
    }
}

// ------------------- Backward Kernel -------------------
__kernel void phase_shifter_backward(
    // Cache (forward'dan)
    __global const float* cache_re_s,
    __global const float* cache_im_s,
    __global const float* cache_re_a,
    __global const float* cache_im_a,
    __global const float* cache_rs,
    __global const float* cache_ra,
    __global const float* cache_sin_d,
    __global const float* cache_conf,
    __global const float* cache_delta,
    __global const float* cache_cos_delta,
    __global const float* cache_sin_delta,
    __global const float* cache_s,
    __global const float* cache_a,
    __global const float* s_out,
    __global const float* a_out,
    // Loss gradyan girdileri
    __global const float* grad_extra,
    __global const float* grad_s_out,
    __global const float* grad_a_out,
    // Orijinal girdiler (clip maskeleri için)
    __global const float* y_ssm,
    __global const float* y_attn,
    // Ağırlıklar (gradyanları ve projeksiyon geri yayılımı için)
    __global const float* w_sr,
    __global const float* w_si,
    __global const float* w_ar,
    __global const float* w_ai,
    __global const float* lambda_hat,
    // Gradyan çıktıları
    __global float* grad_y_ssm,
    __global float* grad_y_attn,
    __global float* grad_w_sr,
    __global float* grad_w_si,
    __global float* grad_w_ar,
    __global float* grad_w_ai,
    __global float* grad_lambda_hat,
    __global float* grad_alpha,
    // Skaler parametreler
    float alpha_tanh,
    float strength,
    float alpha_raw,
    int bsz, int seq, int d, int h, int g, int gd
) {
    int tid = get_global_id(0);
    int total_hg = h * g;
    int total_tiles = bsz * seq * total_hg;
    if (tid >= total_tiles) return;

    int hg_idx = tid % total_hg;
    int bt_idx = tid / total_hg;
    int b = bt_idx / seq;
    int t = bt_idx % seq;
    int hh = hg_idx / g;
    int gg = hg_idx % g;

    int out_base = (b * seq + t) * d + hh * (g * gd) + gg * gd;
    int w_base = (hh * g + gg) * gd;
    int cache_base = (b * seq + t) * total_hg * gd + hg_idx * gd;

    // Cache'leri oku
    float re_s = cache_re_s[tid];
    float im_s = cache_im_s[tid];
    float re_a = cache_re_a[tid];
    float im_a = cache_im_a[tid];
    float rs   = cache_rs[tid];
    float ra   = cache_ra[tid];
    float sin_d = cache_sin_d[tid];
    float conf = cache_conf[tid];
    float cos_delta = cache_cos_delta[tid];
    float sin_delta_val = cache_sin_delta[tid];

    // Private vektörler
    float s_vec[128], a_vec[128], s_out_vec[128], a_out_vec[128];
    for (int i = 0; i < gd; ++i) {
        s_vec[i] = cache_s[cache_base + i];
        a_vec[i] = cache_a[cache_base + i];
        s_out_vec[i] = s_out[out_base + i];
        a_out_vec[i] = a_out[out_base + i];
    }

    // ---- Adım 11: extra gradyanı ve alpha katkısı ----
    float grad_alpha_sum = 0.0f;
    float one_minus_tanh2 = 1.0f - alpha_tanh * alpha_tanh;
    float grad_s_total[128], grad_a_total[128];

    for (int i = 0; i < gd; ++i) {
        float ge = grad_extra[out_base + i];
        float gs = grad_s_out[out_base + i];
        float ga = grad_a_out[out_base + i];

        grad_s_total[i] = gs + ge * strength * alpha_tanh;
        grad_a_total[i] = ga + ge * strength * alpha_tanh;
        grad_alpha_sum += ge * strength * one_minus_tanh2 * (s_out_vec[i] + a_out_vec[i]);
    }
    atomic_add_float(grad_alpha, grad_alpha_sum);

    // ---- Adım 9: Rotasyon geri yayılımı ----
    float grad_s_rot[128], grad_a_rot[128];
    float grad_delta_sum = 0.0f;

    for (int i = 0; i < gd; ++i) {
        float gs = grad_s_total[i];
        float ga = grad_a_total[i];

        grad_s_rot[i] = gs * cos_delta + ga * sin_delta_val;
        grad_a_rot[i] = gs * (-sin_delta_val) + ga * cos_delta;

        grad_delta_sum += gs * (-s_vec[i] * sin_delta_val - a_vec[i] * cos_delta)
                        + ga * ( s_vec[i] * cos_delta - a_vec[i] * sin_delta_val);
    }

    // ---- Adım 8: delta, clip ve temp ----
    float temp = softplus(lambda_hat[hh * g + gg]) + 0.5f;
    float delta_raw = sin_d * conf / temp;
    float clip_mask_delta = (delta_raw > -1.0f && delta_raw < 1.0f) ? 1.0f : 0.0f;
    float grad_delta_raw = grad_delta_sum * strength * clip_mask_delta;

    float grad_sin_d = grad_delta_raw * conf / temp;
    float grad_conf  = grad_delta_raw * sin_d / temp;
    float grad_temp_contrib = -grad_delta_raw * (sin_d * conf) / (temp * temp);

    // lambda_hat gradyanı
    float sig_lambda = sigmoid(lambda_hat[hh * g + gg]);
    atomic_add_float(&grad_lambda_hat[hh * g + gg], grad_temp_contrib * sig_lambda);

    // ---- Adım 6: conf gradyanı ----
    float grad_rs_plus_ra = grad_conf * conf * (1.0f - conf) * 0.5f;
    float grad_rs_conf = grad_rs_plus_ra;
    float grad_ra_conf = grad_rs_plus_ra;

    // ---- Adım 5: sin_d ve den ----
    float den = rs * ra + 1e-4f;
    float sin_d_raw_val = (im_s * re_a - re_s * im_a) / den;
    float clip_mask_sin = (sin_d_raw_val > -0.95f && sin_d_raw_val < 0.95f) ? 1.0f : 0.0f;
    float grad_sin_d_raw = grad_sin_d * clip_mask_sin;

    float grad_im_s_sin = grad_sin_d_raw * re_a / den;
    float grad_re_a_sin = grad_sin_d_raw * im_s / den;
    float grad_re_s_sin = grad_sin_d_raw * (-im_a / den);
    float grad_im_a_sin = grad_sin_d_raw * (-re_s / den);
    float grad_den_sin   = grad_sin_d_raw * (-sin_d_raw_val / den);

    // ---- Adım 4: rs, ra ve büyüklük gradyanları ----
    float grad_rs_den = grad_den_sin * ra;
    float grad_ra_den = grad_den_sin * rs;

    float grad_rs_total = grad_rs_den + grad_rs_conf;
    float grad_ra_total = grad_ra_den + grad_ra_conf;

    float grad_re_s_rs = grad_rs_total * (re_s / rs);
    float grad_im_s_rs = grad_rs_total * (im_s / rs);
    float grad_re_a_ra = grad_ra_total * (re_a / ra);
    float grad_im_a_ra = grad_ra_total * (im_a / ra);

    float grad_re_s_total = grad_re_s_sin + grad_re_s_rs;
    float grad_im_s_total = grad_im_s_sin + grad_im_s_rs;
    float grad_re_a_total = grad_re_a_sin + grad_re_a_ra;
    float grad_im_a_total = grad_im_a_sin + grad_im_a_ra;

    // ---- Adım 3: Projeksiyon (tekrar raw hesapla, maskeler, ağırlık gradyanları) ----
    float re_s_raw = 0.0f, im_s_raw = 0.0f, re_a_raw = 0.0f, im_a_raw = 0.0f;
    for (int i = 0; i < gd; ++i) {
        re_s_raw += s_vec[i] * w_sr[w_base + i];
        im_s_raw += s_vec[i] * w_si[w_base + i];
        re_a_raw += a_vec[i] * w_ar[w_base + i];
        im_a_raw += a_vec[i] * w_ai[w_base + i];
    }

    float mask_re_s = (re_s_raw > -12.0f && re_s_raw < 12.0f) ? 1.0f : 0.0f;
    float mask_im_s = (im_s_raw > -12.0f && im_s_raw < 12.0f) ? 1.0f : 0.0f;
    float mask_re_a = (re_a_raw > -12.0f && re_a_raw < 12.0f) ? 1.0f : 0.0f;
    float mask_im_a = (im_a_raw > -12.0f && im_a_raw < 12.0f) ? 1.0f : 0.0f;

    float grad_re_s_clip = grad_re_s_total * mask_re_s;
    float grad_im_s_clip = grad_im_s_total * mask_im_s;
    float grad_re_a_clip = grad_re_a_total * mask_re_a;
    float grad_im_a_clip = grad_im_a_total * mask_im_a;

    float grad_s_proj[128], grad_a_proj[128];
    for (int i = 0; i < gd; ++i) {
        atomic_add_float(&grad_w_sr[w_base + i], grad_re_s_clip * s_vec[i]);
        atomic_add_float(&grad_w_si[w_base + i], grad_im_s_clip * s_vec[i]);
        atomic_add_float(&grad_w_ar[w_base + i], grad_re_a_clip * a_vec[i]);
        atomic_add_float(&grad_w_ai[w_base + i], grad_im_a_clip * a_vec[i]);

        grad_s_proj[i] = grad_re_s_clip * w_sr[w_base + i] + grad_im_s_clip * w_si[w_base + i];
        grad_a_proj[i] = grad_re_a_clip * w_ar[w_base + i] + grad_im_a_clip * w_ai[w_base + i];
    }

    // ---- Adım 2: Toplam gradyan ve girdi clip maskesi ----
    for (int i = 0; i < gd; ++i) {
        float gs_final = grad_s_rot[i] + grad_s_proj[i];
        float ga_final = grad_a_rot[i] + grad_a_proj[i];

        float y_ssm_val = y_ssm[out_base + i];
        float y_attn_val = y_attn[out_base + i];
        float mask_s_in = (y_ssm_val > -8.0f && y_ssm_val < 8.0f) ? 1.0f : 0.0f;
        float mask_a_in = (y_attn_val > -8.0f && y_attn_val < 8.0f) ? 1.0f : 0.0f;

        grad_y_ssm[out_base + i]  = gs_final * mask_s_in;
        grad_y_attn[out_base + i] = ga_final * mask_a_in;
    }
}