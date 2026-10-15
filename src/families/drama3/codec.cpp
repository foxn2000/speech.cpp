#include "codec.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <functional>
#include <string>

#include "error.h"
#include "layout.h"

namespace drama3 {

/*
 * Activations are channel-first ([channels, samples], ne0 = channels), so a convolution of width K runs as K matrix
 * products, one per tap, over views of its input shifted in time, the input padded on the left alone since every
 * convolution is causal. A convolution of width 2s and stride s pads s samples on the left and cuts its input into
 * frames of s samples, which a reshape of the channel-first layout gives for free; output frame n is the first half of
 * the kernel on input frame n - 1 and the second half on frame n. A transposed convolution of width 2s and stride s
 * gives every tap of every input frame in one matrix product; output frame n is the first s taps of input frame n plus
 * the last s taps of frame n - 1, the official module cutting the s samples past the last frame.
 */

namespace {

/** The epsilon of a ConvNeXt block's LayerNorm, which the official module fixes. */
constexpr float kConvNextEps = 1e-6f;

/** `f` rounded to bfloat16, to nearest with ties to even, as torch converts it. */
float bf16_round(float f) {
    uint32_t u;
    std::memcpy(&u, &f, sizeof u);
    if ((u & 0x7fffffffu) > 0x7f800000u) return f;
    u += 0x7fffu + ((u >> 16) & 1u);
    u &= 0xffff0000u;
    std::memcpy(&f, &u, sizeof f);
    return f;
}

struct Layers {
    Graph & g;
    const ModelFile & m;

    ggml_context * ctx() const { return g.ctx(); }

    ggml_tensor * left_pad(ggml_tensor * x, int64_t n) { return n > 0 ? ggml_concat(ctx(), g.zeros(x->ne[0], n), x, 1) : x; }

    ggml_tensor * linear(ggml_tensor * x, const std::string & name) {
        return ggml_add(ctx(), mul_mat(ctx(), m.tensor(name + ".weight"), x), m.tensor(name + ".bias"));
    }

    /** A causal convolution of stride 1; weight ne = [in, out, k]. */
    ggml_tensor * conv(ggml_tensor * x, const std::string & name, int dilation = 1) {
        ggml_tensor * w = m.tensor(name + ".weight");
        const int64_t in = w->ne[0], out = w->ne[1], width = w->ne[2], t = x->ne[1];
        ggml_tensor * y = nullptr;
        if (width == 1) {
            y = mul_mat(ctx(), ggml_reshape_2d(ctx(), w, in, out), x);
        } else {
            ggml_tensor * xp = left_pad(x, (width - 1) * dilation);
            for (int64_t k = 0; k < width; k++) {
                ggml_tensor * wk = ggml_view_2d(ctx(), w, in, out, w->nb[1], k * w->nb[2]);
                ggml_tensor * xk = ggml_view_2d(ctx(), xp, in, t, xp->nb[1], k * dilation * xp->nb[1]);
                ggml_tensor * yk = mul_mat(ctx(), wk, xk);
                y = y ? ggml_add(ctx(), y, yk) : yk;
            }
        }
        return ggml_add(ctx(), y, m.tensor(name + ".bias"));
    }

    /** A causal convolution of width 2s and stride s, its kernel stored in halves [s * in, out]. */
    ggml_tensor * down(ggml_tensor * x, const std::string & name, int s) {
        const int64_t c = x->ne[0], n = x->ne[1] / s;
        ggml_tensor * frames = ggml_reshape_2d(ctx(), left_pad(x, s), c * s, n + 1);
        ggml_tensor * before = ggml_view_2d(ctx(), frames, c * s, n, frames->nb[1], 0);
        ggml_tensor * own = ggml_view_2d(ctx(), frames, c * s, n, frames->nb[1], frames->nb[1]);
        ggml_tensor * y = ggml_add(ctx(), mul_mat(ctx(), m.tensor(name + ".first"), before), mul_mat(ctx(), m.tensor(name + ".second"), own));
        return ggml_add(ctx(), y, m.tensor(name + ".bias"));
    }

    /** A convolution whose width is its stride s, its kernel stored as one matrix [s * in, out]. */
    ggml_tensor * framed(ggml_tensor * x, const std::string & name, int s) {
        ggml_tensor * frames = ggml_reshape_2d(ctx(), ggml_cont(ctx(), x), x->ne[0] * s, x->ne[1] / s);
        return linear(frames, name);
    }

    /** A causal transposed convolution of width 2s and stride s; weight ne = [in, out, 2s]. */
    ggml_tensor * up(ggml_tensor * x, const std::string & name, int s) {
        ggml_tensor * w = m.tensor(name + ".weight");
        const int64_t in = w->ne[0], out = w->ne[1], frames = x->ne[1];
        ggml_tensor * z = ggml_reshape_3d(ctx(), mul_mat(ctx(), ggml_reshape_2d(ctx(), w, in, out * 2 * s), x), out, 2 * s, frames);
        ggml_tensor * own = ggml_cont(ctx(), ggml_view_3d(ctx(), z, out, s, frames, z->nb[1], z->nb[2], 0));
        ggml_tensor * edge = ggml_reshape_3d(ctx(), g.zeros(out, s), out, s, 1);
        ggml_tensor * before = frames > 1
            ? ggml_concat(ctx(), edge, ggml_cont(ctx(), ggml_view_3d(ctx(), z, out, s, frames - 1, z->nb[1], z->nb[2], s * z->nb[1])), 2)
            : edge;
        ggml_tensor * y = ggml_reshape_2d(ctx(), ggml_add(ctx(), own, before), out, s * frames);
        return ggml_add(ctx(), y, m.tensor(name + ".bias"));
    }

    /** A transposed convolution whose width is its stride s; weight ne = [in, out, s]. */
    ggml_tensor * up_framed(ggml_tensor * x, const std::string & name, int s) {
        ggml_tensor * w = m.tensor(name + ".weight");
        const int64_t in = w->ne[0], out = w->ne[1];
        ggml_tensor * z = mul_mat(ctx(), ggml_reshape_2d(ctx(), w, in, out * s), x);
        return ggml_add(ctx(), ggml_reshape_2d(ctx(), z, out, s * x->ne[1]), m.tensor(name + ".bias"));
    }

    /**
     * Snake: x + sin(alpha x)^2 / (alpha + 1e-9). Metal and Vulkan run these five operations as one kernel only in this
     * order and form, x times alpha first and x added last, with alpha and inv_alpha of [channels] and x contiguous.
     */
    ggml_tensor * snake(ggml_tensor * x, const std::string & name) {
        ggml_tensor * s = ggml_sin(ctx(), ggml_mul(ctx(), x, m.tensor(name + ".alpha")));
        return ggml_add(ctx(), x, ggml_mul(ctx(), ggml_sqr(ctx(), s), m.tensor(name + ".inv_alpha")));
    }

    ggml_tensor * residual(ggml_tensor * x, const std::string & name, int dilation) {
        ggml_tensor * h = conv(snake(x, name + ".snake1"), name + ".conv1", dilation);
        h = conv(snake(h, name + ".snake2"), name + ".conv2");
        return ggml_add(ctx(), x, h);
    }

    /** A ConvNeXt block: a causal depthwise convolution of width 7, LayerNorm, two linear layers about GELU, a scale. */
    ggml_tensor * convnext(ggml_tensor * x, const std::string & name) {
        ggml_tensor * w = m.tensor(name + ".dwconv.weight");
        const int64_t c = w->ne[0], width = w->ne[1], t = x->ne[1];
        ggml_tensor * xp = left_pad(x, width - 1);
        ggml_tensor * h = nullptr;
        for (int64_t k = 0; k < width; k++) {
            ggml_tensor * hk = ggml_mul(ctx(), ggml_view_2d(ctx(), xp, c, t, xp->nb[1], k * xp->nb[1]), ggml_view_1d(ctx(), w, c, k * w->nb[1]));
            h = h ? ggml_add(ctx(), h, hk) : hk;
        }
        h = ggml_add(ctx(), h, m.tensor(name + ".dwconv.bias"));
        h = ggml_add(ctx(), ggml_mul(ctx(), ggml_norm(ctx(), h, kConvNextEps), m.tensor(name + ".norm.weight")), m.tensor(name + ".norm.bias"));
        h = linear(ggml_gelu_erf(ctx(), linear(h, name + ".pw1")), name + ".pw2");
        return ggml_add(ctx(), x, ggml_mul(ctx(), h, m.tensor(name + ".gamma")));
    }

    /**
     * The interleaved RoPE of `x`, [head_dim, heads, t]: each pair of channels turned by the angle of its frequency at
     * the frame's position, by the tables `cos` and `sin`, [1, head_dim / 2, 1, t].
     */
    ggml_tensor * rope(ggml_tensor * x, ggml_tensor * cos, ggml_tensor * sin) {
        const int64_t hd = x->ne[0], heads = x->ne[1], t = x->ne[2];
        ggml_tensor * pairs = ggml_reshape_4d(ctx(), ggml_cont(ctx(), x), 2, hd / 2, heads, t);
        ggml_tensor * even = ggml_cont(ctx(), ggml_view_4d(ctx(), pairs, 1, hd / 2, heads, t, pairs->nb[1], pairs->nb[2], pairs->nb[3], 0));
        ggml_tensor * odd = ggml_cont(ctx(), ggml_view_4d(ctx(), pairs, 1, hd / 2, heads, t, pairs->nb[1], pairs->nb[2], pairs->nb[3], pairs->nb[0]));
        ggml_tensor * e = ggml_sub(ctx(), ggml_mul(ctx(), even, cos), ggml_mul(ctx(), odd, sin));
        ggml_tensor * o = ggml_add(ctx(), ggml_mul(ctx(), odd, cos), ggml_mul(ctx(), even, sin));
        return ggml_reshape_3d(ctx(), ggml_concat(ctx(), e, o, 0), hd, heads, t);
    }

    /**
     * Attention of the queries `q`, [head_dim, heads, t], to the keys and values `k` and `v`, [head_dim, heads, c + t],
     * whose last t rows are the queries' own: each row to itself and the `window` - 1 rows before it, in blocks of
     * `window` queries, so that no block holds more than 2 * window - 1 keys however long the input.
     */
    ggml_tensor * windowed_attention(ggml_tensor * q, ggml_tensor * k, ggml_tensor * v, int64_t window) {
        const int64_t hd = q->ne[0], heads = q->ne[1], t = q->ne[2], c = k->ne[2] - t;
        ggml_tensor * out = nullptr;
        for (int64_t a = 0; a < t; a += window) {
            const int64_t qn = std::min(window, t - a), k0 = std::max<int64_t>(0, c + a - window + 1), kn = c + a + qn - k0;
            const auto block = [&](ggml_tensor * x, int64_t first, int64_t n) { return ggml_view_3d(ctx(), x, hd, heads, n, x->nb[1], x->nb[2], first * x->nb[2]); };
            ggml_tensor * qb = ggml_cont(ctx(), ggml_permute(ctx(), block(q, a, qn), 0, 2, 1, 3));
            ggml_tensor * kb = ggml_cont(ctx(), ggml_permute(ctx(), block(k, k0, kn), 0, 2, 1, 3));
            ggml_tensor * vb = ggml_cont(ctx(), ggml_permute(ctx(), block(v, k0, kn), 1, 2, 0, 3));
            std::vector<float> mask((size_t) (kn * qn));
            for (int64_t i = 0; i < qn; i++) {
                for (int64_t j = 0; j < kn; j++) {
                    const int64_t qp = c + a + i, kp = k0 + j;
                    mask[(size_t) (i * kn + j)] = kp <= qp && qp - kp < window ? 0.0f : -INFINITY;
                }
            }
            ggml_tensor * kq = ggml_soft_max_ext(ctx(), mul_mat(ctx(), kb, qb), g.input(mask, kn, qn), 1.0f / std::sqrt((float) hd), 0.0f);
            ggml_tensor * kqv = ggml_cont(ctx(), ggml_permute(ctx(), mul_mat(ctx(), vb, kq), 0, 2, 1, 3));
            ggml_tensor * o = ggml_reshape_2d(ctx(), kqv, hd * heads, qn);
            out = out ? ggml_concat(ctx(), out, o, 1) : o;
        }
        return out;
    }

    /**
     * The RoPE tables of `rows` rows from position `first`, [1, head_dim / 2, 1, rows]. The official module computes its
     * tables once, in bfloat16, so the angles here are rounded to bfloat16 too.
     */
    std::pair<ggml_tensor *, ggml_tensor *> rope_tables(int64_t first, int64_t rows, int hd, float base) {
        std::vector<float> cos((size_t) (hd / 2 * rows)), sin(cos.size());
        for (int64_t r = 0; r < rows; r++) {
            for (int i = 0; i < hd / 2; i++) {
                const float inv = 1.0f / std::pow(base, (float) (2 * i) / (float) hd);
                const float angle = (float) (first + r) * inv;
                cos[(size_t) (r * (hd / 2) + i)] = bf16_round(std::cos(angle));
                sin[(size_t) (r * (hd / 2) + i)] = bf16_round(std::sin(angle));
            }
        }
        return {g.input(cos, 1, hd / 2, 1, rows), g.input(sin, 1, hd / 2, 1, rows)};
    }

    /**
     * The codec's transformer of `prefix` on `x`, [dim, t], the rows from position `first`: pre-norm layers of windowed
     * causal attention with RoPE and a gated feed-forward, each scaled before it is added, then a final norm. `before`,
     * where given, holds each layer's inputs of the rows before these that its attention reaches, [dim, c] or none,
     * and receives its inputs of the last window - 1 rows, for the rows that follow.
     */
    ggml_tensor * transformer(ggml_tensor * x, const std::string & prefix, const std::string & keys, int64_t first = 0,
                              std::vector<ggml_tensor *> * before = nullptr) {
        const int layers = m.count(keys + "layers"), heads = m.size(keys + "heads"), hd = m.size(keys + "head_dim");
        const int64_t window = m.size(keys + "window"), t = x->ne[1];
        const float base = m.f32(keys + "rope_base"), eps = m.f32(keys + "norm_eps");
        // Every layer holds the same rows before these, its inputs at the same positions.
        const int64_t c = before && (*before)[0] ? (*before)[0]->ne[1] : 0;
        const auto [kc, ks] = rope_tables(first - c, c + t, hd, base);
        const auto [qc, qs] = c ? rope_tables(first, t, hd, base) : std::make_pair(kc, ks);
        for (int l = 0; l < layers; l++) {
            const std::string b = prefix + ".blk." + std::to_string(l) + ".";
            ggml_tensor * rows = c ? ggml_concat(ctx(), (*before)[(size_t) l], x, 1) : x;
            if (before) {
                const int64_t kept = std::min(window - 1, c + t);
                (*before)[(size_t) l] = ggml_cont(ctx(), ggml_view_2d(ctx(), rows, rows->ne[0], kept, rows->nb[1], (size_t) (c + t - kept) * rows->nb[1]));
            }
            ggml_tensor * h = ggml_mul(ctx(), ggml_rms_norm(ctx(), rows, eps), m.tensor(b + "attn_norm"));
            ggml_tensor * own = c ? ggml_view_2d(ctx(), h, h->ne[0], t, h->nb[1], (size_t) c * h->nb[1]) : h;
            ggml_tensor * q = rope(ggml_reshape_3d(ctx(), mul_mat(ctx(), m.tensor(b + "attn_q"), own), hd, heads, t), qc, qs);
            ggml_tensor * k = rope(ggml_reshape_3d(ctx(), mul_mat(ctx(), m.tensor(b + "attn_k"), h), hd, heads, c + t), kc, ks);
            ggml_tensor * v = ggml_reshape_3d(ctx(), mul_mat(ctx(), m.tensor(b + "attn_v"), h), hd, heads, c + t);
            ggml_tensor * o = mul_mat(ctx(), m.tensor(b + "attn_o"), windowed_attention(q, k, v, window));
            x = ggml_add(ctx(), x, ggml_mul(ctx(), o, m.tensor(b + "attn_scale")));
            h = ggml_mul(ctx(), ggml_rms_norm(ctx(), x, eps), m.tensor(b + "ffn_norm"));
            ggml_tensor * gate = ggml_silu(ctx(), mul_mat(ctx(), m.tensor(b + "ffn_gate"), h));
            h = mul_mat(ctx(), m.tensor(b + "ffn_down"), ggml_mul(ctx(), gate, mul_mat(ctx(), m.tensor(b + "ffn_up"), h)));
            x = ggml_add(ctx(), x, ggml_mul(ctx(), h, m.tensor(b + "ffn_scale")));
        }
        return ggml_mul(ctx(), ggml_rms_norm(ctx(), x, eps), m.tensor(prefix + ".norm"));
    }

    /**
     * The code of each frame of `z`, [latent, t], under the vector quantizer of `prefix`, which projects a frame to its
     * codebook's width and takes the code nearest to it once both are normalized, by the squared distance the official
     * module computes; and the codebook vector of that code projected back, which `vector` receives.
     */
    ggml_tensor * quantize(ggml_tensor * z, const std::string & prefix, ggml_tensor ** vector) {
        ggml_tensor * e = ggml_l2_norm(ctx(), linear(z, prefix + "in_proj"), 1e-12f);
        ggml_tensor * book = m.tensor(prefix + "codebook");
        ggml_tensor * c = ggml_l2_norm(ctx(), book, 1e-12f);
        ggml_tensor * e2 = ggml_sum_rows(ctx(), ggml_sqr(ctx(), e));
        ggml_tensor * c2 = ggml_cont(ctx(), ggml_transpose(ctx(), ggml_sum_rows(ctx(), ggml_sqr(ctx(), c))));
        ggml_tensor * dist = ggml_add(ctx(), ggml_add(ctx(), ggml_scale(ctx(), mul_mat(ctx(), c, e), -2.0f), e2), c2);
        ggml_tensor * codes = ggml_argmax(ctx(), ggml_scale(ctx(), dist, -1.0f));
        *vector = linear(ggml_get_rows(ctx(), book, codes), prefix + "out_proj");
        return codes;
    }
};

}  // namespace

Codec::Codec(const ModelFile & m, ggml_backend_t backend) : backend_(backend), m_(m) {
    const std::string c = "drama3.codec.";
    sample_rate_ = m.size("speech.sample_rate");
    hop_ = m.size(c + "hop_length");
    latent_ = m.size(c + "latent_dim");
    residual_codebooks_ = m.count(c + "residual_codebooks");
    semantic_size_ = m.size(c + "semantic_codebook_size");
    codebook_size_ = m.size(c + "codebook_size");
    encoder_rates_ = m.i32_array(c + "encoder_rates");
    decoder_rates_ = m.i32_array(c + "decoder_rates");
    factors_ = m.i32_array(c + "downsample_factors");
    allocr_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend_));
}

Codec::~Codec() {
    if (allocr_) ggml_gallocr_free(allocr_);
}

std::vector<int32_t> Codec::encode(const std::vector<float> & audio, CodecStages * stages) {
    if (audio.empty()) throw Error(Fault::OutOfRange, "the codec has no audio to encode");
    const int64_t frames = this->frames(audio.size()), n = frames * hop_;
    std::vector<float> padded = audio;
    padded.resize((size_t) n, 0.0f);
    Graph g;
    Layers l{g, m_};
    std::vector<std::pair<std::string, ggml_tensor *>> kept;
    const auto stage = [&](const std::string & name, ggml_tensor * t) {
        if (!stages) return;
        g.output(t);
        kept.push_back({name, t});
    };
    ggml_tensor * x = l.conv(g.input(padded, 1, n), "codec.enc.in_conv");
    stage("encoder_0", x);
    for (size_t b = 0; b < encoder_rates_.size(); b++) {
        const std::string o = "codec.enc.blk." + std::to_string(b);
        for (int j = 0, dilation = 1; j < codec_residual_units; j++, dilation *= 3) x = l.residual(x, o + ".res." + std::to_string(j), dilation);
        x = l.down(l.snake(x, o + ".snake"), o + ".down", encoder_rates_[b]);
        if (b + 1 == encoder_rates_.size()) x = l.transformer(x, o + ".tf", "drama3.codec.encoder_transformer.");
        stage("encoder_" + std::to_string(b + 1), x);
    }
    x = l.snake(x, "codec.enc.out_snake");
    stage("encoder_" + std::to_string(encoder_rates_.size() + 1), x);
    x = l.conv(x, "codec.enc.out_conv");
    stage("encoder_" + std::to_string(encoder_rates_.size() + 2), x);
    for (size_t i = 0; i < factors_.size(); i++) {
        x = l.framed(x, "codec.q.down." + std::to_string(i) + ".conv", factors_[i]);
        x = l.convnext(x, "codec.q.down." + std::to_string(i) + ".convnext");
    }
    stage("downsample", x);
    x = l.transformer(x, "codec.q.pre", "drama3.codec.quantizer_transformer.");
    stage("pre_module", x);
    g.output(x);
    g.compute(backend_, allocr_);
    if (stages) {
        for (const auto & [name, t] : kept) (*stages)[name] = Graph::read(t);
    }
    return quantize(Graph::read(x), frames);
}

std::vector<int32_t> Codec::quantize(const std::vector<float> & latent, int64_t frames) {
    if (frames < 1 || (int64_t) latent.size() != frames * latent_) {
        throw Error(Fault::OutOfRange, "the codec's quantizer takes a whole number of frames of " + std::to_string(latent_) + " values");
    }
    Graph g;
    Layers l{g, m_};
    ggml_tensor * x = g.input(latent, latent_, frames);
    std::vector<ggml_tensor *> codes;
    ggml_tensor * vector = nullptr;
    codes.push_back(l.quantize(x, "codec.q.semantic.", &vector));
    ggml_tensor * residual = ggml_sub(g.ctx(), x, vector);
    for (int j = 0; j < residual_codebooks_; j++) {
        codes.push_back(l.quantize(residual, "codec.q.residual." + std::to_string(j) + ".", &vector));
        residual = ggml_sub(g.ctx(), residual, vector);
    }
    for (ggml_tensor * c : codes) g.output(c);
    g.compute(backend_, allocr_);
    std::vector<int32_t> out((size_t) (codes.size() * frames));
    for (size_t i = 0; i < codes.size(); i++) {
        ggml_backend_tensor_get(codes[i], out.data() + i * frames, 0, (size_t) frames * sizeof(int32_t));
    }
    return out;
}

ggml_tensor * Codec::build_decoder(Graph & g, const std::vector<int32_t> & codes, int64_t frames, std::vector<ggml_tensor *> * before,
                                   ggml_tensor * held, ggml_tensor ** post, const std::function<void(const std::string &, ggml_tensor *)> & stage) {
    if (frames < 1 || (int64_t) codes.size() != frames * codebooks()) {
        throw Error(Fault::OutOfRange, "the codec's decoder takes a whole number of frames of " + std::to_string(codebooks()) + " codes");
    }
    Layers l{g, m_};
    const auto lookup = [&](int book, const std::string & prefix, int size) {
        std::vector<int32_t> ids(codes.begin() + book * frames, codes.begin() + (book + 1) * frames);
        // The official decoder clamps a code past its codebook to the codebook's last, which the audio decoder may draw
        // for a residual codebook, whose heads have as many codes as the semantic one.
        for (int32_t & id : ids) id = std::clamp<int32_t>(id, 0, size - 1);
        return l.linear(ggml_get_rows(g.ctx(), m_.tensor(prefix + "codebook"), g.input(ids, frames)), prefix + "out_proj");
    };
    ggml_tensor * semantic = lookup(0, "codec.q.semantic.", semantic_size_);
    ggml_tensor * rest = nullptr;
    for (int j = 0; j < residual_codebooks_; j++) {
        ggml_tensor * v = lookup(1 + j, "codec.q.residual." + std::to_string(j) + ".", codebook_size_);
        rest = rest ? ggml_add(g.ctx(), rest, v) : v;
    }
    ggml_tensor * x = ggml_add(g.ctx(), semantic, rest);
    stage("quantized", x);
    x = l.transformer(x, "codec.q.post", "drama3.codec.quantizer_transformer.", position_, before);
    stage("post_module", x);
    *post = x;
    if (held) x = ggml_concat(g.ctx(), held, x, 1);
    for (size_t i = 0; i < factors_.size(); i++) {
        x = l.up_framed(x, "codec.q.up." + std::to_string(i) + ".conv", factors_[factors_.size() - 1 - i]);
        x = l.convnext(x, "codec.q.up." + std::to_string(i) + ".convnext");
    }
    stage("upsample", x);
    x = l.conv(x, "codec.dec.in_conv");
    stage("decoder_0", x);
    for (size_t b = 0; b < decoder_rates_.size(); b++) {
        const std::string o = "codec.dec.blk." + std::to_string(b);
        x = l.up(l.snake(x, o + ".snake"), o + ".up", decoder_rates_[b]);
        for (int j = 0, dilation = 1; j < codec_residual_units; j++, dilation *= 3) x = l.residual(x, o + ".res." + std::to_string(j), dilation);
        stage("decoder_" + std::to_string(b + 1), x);
    }
    x = l.snake(x, "codec.dec.out_snake");
    stage("decoder_" + std::to_string(decoder_rates_.size() + 1), x);
    x = l.conv(x, "codec.dec.out_conv");
    stage("decoder_" + std::to_string(decoder_rates_.size() + 2), x);
    x = ggml_tanh(g.ctx(), x);
    stage("decoder_" + std::to_string(decoder_rates_.size() + 3), x);
    return x;
}

std::vector<float> Codec::decode(const std::vector<int32_t> & codes, int64_t frames, CodecStages * stages) {
    start_decoding();
    Graph g;
    std::vector<std::pair<std::string, ggml_tensor *>> kept;
    ggml_tensor * post = nullptr;
    ggml_tensor * audio = build_decoder(g, codes, frames, nullptr, nullptr, &post, [&](const std::string & name, ggml_tensor * t) {
        if (!stages) return;
        g.output(t);
        kept.push_back({name, t});
    });
    g.output(audio);
    g.compute(backend_, allocr_);
    if (stages) {
        for (const auto & [name, t] : kept) (*stages)[name] = Graph::read(t);
    }
    return Graph::read(audio);
}

void Codec::start_decoding() {
    position_ = 0;
    before_.clear();
    held_.clear();
}

std::vector<float> Codec::decode_more(const std::vector<int32_t> & codes, int64_t frames) {
    Graph g;
    const int layers = m_.count("drama3.codec.quantizer_transformer.layers");
    std::vector<ggml_tensor *> before((size_t) layers, nullptr);
    for (size_t i = 0; i < before_.size(); i++) before[i] = g.input(before_[i], latent_, (int64_t) before_[i].size() / latent_);
    const int64_t held = (int64_t) held_.size() / latent_;
    ggml_tensor * post = nullptr;
    ggml_tensor * audio = build_decoder(g, codes, frames, &before, held ? g.input(held_, latent_, held) : nullptr, &post, [](const std::string &, ggml_tensor *) {});
    // The convolutions of the next frames start from the transformer's outputs of the last kHeldFrames frames.
    ggml_tensor * outputs = held ? ggml_concat(g.ctx(), g.input(held_, latent_, held), post, 1) : post;
    const int64_t keep = std::min<int64_t>(kHeldFrames, held + frames);
    ggml_tensor * next = ggml_cont(g.ctx(), ggml_view_2d(g.ctx(), outputs, latent_, keep, outputs->nb[1], (size_t) (held + frames - keep) * outputs->nb[1]));
    g.output(audio);
    g.output(next);
    for (ggml_tensor * t : before) g.output(t);
    g.compute(backend_, allocr_);
    before_.clear();
    for (ggml_tensor * t : before) before_.push_back(Graph::read(t));
    held_ = Graph::read(next);
    position_ += frames;
    const std::vector<float> all = Graph::read(audio);
    return std::vector<float>(all.begin() + held * hop_, all.end());
}

}  // namespace drama3
