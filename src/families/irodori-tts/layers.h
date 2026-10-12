#pragma once

#include <cmath>
#include <cstdlib>
#include <string>

#include "graph.h"
#include "model-file.h"

namespace irodori {

/** The building blocks Irodori-TTS's own modules share, on channel-first activations. */
struct Layers {
    ggml_context * ctx;
    const ModelFile & m;
    float eps;

    /** A linear layer with the bias `name.bias`. */
    ggml_tensor * linear(ggml_tensor * x, const std::string & name) const {
        return ggml_add(ctx, mul_mat(ctx, m.tensor(name + ".weight"), x), m.tensor(name + ".bias"));
    }

    /** RMSNorm with the weight `name`. */
    ggml_tensor * rms(ggml_tensor * x, const std::string & name) const {
        return ggml_mul(ctx, ggml_rms_norm(ctx, x, eps), m.tensor(name));
    }

    /** SwiGLU: ffn_down(silu(ffn_gate x) * ffn_up x). */
    ggml_tensor * swiglu(ggml_tensor * x, const std::string & prefix) const {
        ggml_tensor * gate = ggml_silu(ctx, mul_mat(ctx, m.tensor(prefix + "ffn_gate"), x));
        return mul_mat(ctx, m.tensor(prefix + "ffn_down"), ggml_mul(ctx, gate, mul_mat(ctx, m.tensor(prefix + "ffn_up"), x)));
    }

    /**
     * Scaled dot-product attention of q [head_dim, heads, n_q, batch] over k and v [head_dim, heads, n_kv,
     * batch], with an optional additive mask [n_kv, n_q, 1, batch]; the result is [head_dim * heads, n_q,
     * batch].
     */
    ggml_tensor * attention(ggml_tensor * q, ggml_tensor * k, ggml_tensor * v, ggml_tensor * mask) const {
        const int64_t head_dim = q->ne[0], heads = q->ne[1], n_q = q->ne[2], batch = q->ne[3];
        static const bool flash = std::getenv("SPEECH_IRODORI_FLASH") != nullptr;
        if (flash) {
            ggml_tensor * fm = mask ? ggml_cast(ctx, mask, GGML_TYPE_F16) : nullptr;
            ggml_tensor * o = ggml_flash_attn_ext(ctx, ggml_permute(ctx, q, 0, 2, 1, 3), ggml_permute(ctx, k, 0, 2, 1, 3),
                                                  ggml_permute(ctx, v, 0, 2, 1, 3), fm, 1.0f / std::sqrt((float) head_dim), 0.0f, 0.0f);
            ggml_prec_set_acc(o, GGML_PREC_F32);
            return ggml_reshape_3d(ctx, o, head_dim * heads, n_q, batch);
        }
        ggml_tensor * qh = ggml_cont(ctx, ggml_permute(ctx, q, 0, 2, 1, 3));
        ggml_tensor * kh = ggml_cont(ctx, ggml_permute(ctx, k, 0, 2, 1, 3));
        ggml_tensor * vt = ggml_cont(ctx, ggml_permute(ctx, v, 1, 2, 0, 3));
        ggml_tensor * kq = ggml_soft_max_ext(ctx, mul_mat(ctx, kh, qh), mask, 1.0f / std::sqrt((float) head_dim), 0.0f);
        ggml_tensor * o = mul_mat(ctx, vt, kq);
        return ggml_reshape_3d(ctx, ggml_cont(ctx, ggml_permute(ctx, o, 0, 2, 1, 3)), head_dim * heads, n_q, batch);
    }

    /** x * (1 + scale) + shift, in that order of operations. */
    ggml_tensor * modulate(ggml_tensor * x, ggml_tensor * shift, ggml_tensor * scale) const {
        return ggml_add(ctx, ggml_mul(ctx, x, ggml_scale_bias(ctx, scale, 1.0f, 1.0f)), shift);
    }
};

}  // namespace irodori
