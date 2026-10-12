#include "qwen3-decoder.h"

#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <map>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <utility>

#include "error.h"
#include "log.h"

/*
 * Activations are channel-first ([hidden, rows]). Every row has one position, which in Qwen's multimodal RoPE stands
 * for the same position in each of its three sections, and there it is the ordinary rotate-half RoPE
 * (GGML_ROPE_TYPE_NEOX).
 */

namespace {

constexpr int kGraphSize = 8192;

/**
 * The cache grows in steps of this many positions, starting with room for the first run and as many more (20 s of
 * Qwen3-TTS's speech), and doubling as the sequence grows.
 */
constexpr int64_t kCacheStep = 256;

/**
 * Flash attention reads the keys and values of a whole number of blocks of this many positions, its mask hiding those
 * past the sequence, and a cache holds a whole number of blocks. Metal's flash attention reads keys 32 positions at a
 * time in its kernel for a few rows and 64 in the one for many, and first copies a last block that the positions do not
 * fill into a buffer of its own: the attention of 28 layers at 400 positions, the shape of both Qwen3-ASR decoders,
 * took 1.17 ms on the GPU of an Apple M5, and 0.62 ms read as 416 positions with the mask (2026-10-07). A whole number
 * of blocks also starts every row of the products' transposed values on 32 bytes in half precision, as ggml aligns a
 * tensor: Metal's matrix-vector product reads a row four values at a time whenever its length is a multiple of four,
 * whatever the stride between rows, and a row that does not start on four bytes reads the wrong values, so that a
 * cache of 85 positions put the talker's logits 42% off on Metal.
 */
constexpr int64_t kAttentionBlock = 64;

int64_t round_up(int64_t n, int64_t step) {
    return (n + step - 1) / step * step;
}

/** The positions of `rows` rows after `past` positions. */
std::vector<int32_t> positions_after(int64_t past, int64_t rows) {
    std::vector<int32_t> positions((size_t) rows);
    for (int64_t i = 0; i < rows; i++) positions[(size_t) i] = (int32_t) (past + i);
    return positions;
}

/**
 * The mask of `rows` rows after `past` positions over the first `n_kv` positions, [n_kv, rows]: a row sees the
 * positions before it and its own.
 */
std::vector<float> causal_mask(int64_t past, int64_t rows, int64_t n_kv) {
    std::vector<float> mask((size_t) (n_kv * rows));
    for (int64_t i = 0; i < rows; i++) {
        for (int64_t j = 0; j < n_kv; j++) mask[(size_t) (i * n_kv + j)] = j <= past + i ? 0.0f : -INFINITY;
    }
    return mask;
}

/** Whether `backend` computes flash attention of one row over a block of a cache of `cache_type`, the stack's heads. */
bool computes_flash_attention(ggml_backend_t backend, const Qwen3Shape & s, ggml_type cache_type) {
    ggml_init_params params = {ggml_tensor_overhead() * 8, nullptr, true};
    std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx(ggml_init(params), ggml_free);
    ggml_tensor * q = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, s.head_dim, 1, s.n_head);
    ggml_tensor * k = ggml_new_tensor_3d(ctx.get(), cache_type, s.head_dim, kAttentionBlock, s.n_kv_head);
    ggml_tensor * v = ggml_new_tensor_3d(ctx.get(), cache_type, s.head_dim, kAttentionBlock, s.n_kv_head);
    ggml_tensor * mask = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F16, kAttentionBlock, 1);
    ggml_tensor * attention = ggml_flash_attn_ext(ctx.get(), q, k, v, mask, 1.0f, 0.0f, 0.0f);
    ggml_prec_set_acc(attention, GGML_PREC_F32);
    return ggml_backend_supports_op(backend, attention);
}

}  // namespace

Qwen3Attention qwen3_attention(ggml_backend_t backend, const Qwen3Shape & shape, ggml_type cache_type) {
    const bool cpu = ggml_backend_dev_type(ggml_backend_get_device(backend)) == GGML_BACKEND_DEVICE_TYPE_CPU;
    return !cpu && computes_flash_attention(backend, shape, cache_type) ? Qwen3Attention::Flash : Qwen3Attention::Products;
}

const char * qwen3_attention_name(Qwen3Attention attention) {
    return attention == Qwen3Attention::Flash ? "flash attention" : "two matrix products";
}

Qwen3Shape read_qwen3_shape(const ModelFile & m, const std::string & prefix) {
    Qwen3Shape s;
    s.hidden = m.size(prefix + ".hidden_size");
    s.ffn = m.size(prefix + ".intermediate_size");
    s.n_layer = m.count(prefix + ".num_hidden_layers");
    s.n_head = m.size(prefix + ".num_attention_heads");
    s.n_kv_head = m.size(prefix + ".num_key_value_heads");
    s.head_dim = m.size(prefix + ".head_dim");
    s.rms_eps = m.f32(prefix + ".rms_norm_eps");
    s.rope_theta = m.f32(prefix + ".rope_theta");
    const auto require = [&](bool condition, const std::string & what) {
        if (!condition) throw Error(Fault::File, m.path() + ": " + prefix + what + "; " + m.remedy());
    };
    require(s.n_head % s.n_kv_head == 0, ".num_attention_heads is not a multiple of " + prefix + ".num_key_value_heads");
    require(s.head_dim % 2 == 0, ".head_dim is odd, where RoPE turns pairs of channels");
    return s;
}

void add_qwen3_tensors(std::vector<TensorSpec> & t, const std::string & tensors, const Qwen3Shape & s, const Storage & matrix) {
    static const Storage f32 = float32_storage();
    const int64_t h = s.hidden, q = (int64_t) s.n_head * s.head_dim, kv = (int64_t) s.n_kv_head * s.head_dim;
    add_numbered(t, tensors + ".blk.", s.n_layer,
                 {{"attn_norm", {h}, f32},
                  {"ffn_norm", {h}, f32},
                  {"attn_q", {h, q}, matrix},
                  {"attn_k", {h, kv}, matrix},
                  {"attn_v", {h, kv}, matrix},
                  {"attn_o", {q, h}, matrix},
                  {"attn_q_norm", {s.head_dim}, f32},
                  {"attn_k_norm", {s.head_dim}, f32},
                  {"ffn_gate", {h, s.ffn}, matrix},
                  {"ffn_up", {h, s.ffn}, matrix},
                  {"ffn_down", {s.ffn, h}, matrix}});
    t.push_back({tensors + ".norm", {h}, f32});
}

/**
 * The keys and values of every layer for `capacity` positions: the keys a row per position,
 * [kv heads * head dim, capacity], and the values in the layout the attention reads. Flash attention reads them a row
 * per position, as the keys. The two products read them transposed, a row per channel, [capacity, kv heads * head dim],
 * through a view as [positions, head dim, kv heads], the operand of the second product, so that a step reads the cache
 * once and copies none of it: values kept a row per position need a copy into that layout at every step, with which a
 * step of the 1.7B talker 8192 frames into its speech took 259 ms instead of 40 on the CPU of an Apple M5 (2026-10-06).
 * On that CPU flash attention takes twice as long as the products at 8,000 positions, 51 ms against 22 for 28 layers
 * of the heads of both Qwen3-ASR decoders (2026-10-07), and sums the values in half precision.
 */
struct Qwen3Decoder::Cache {
    ggml_context * ctx = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    std::vector<ggml_tensor *> k, v;
    int64_t capacity = 0;
    /** Whether the values are kept a row per position, for flash attention. */
    bool values_by_position = false;

    /** A cache with room for `positions` positions, room(positions) in all. */
    Cache(ggml_backend_t backend, ggml_type type, const Qwen3Shape & s, int64_t positions, bool by_position)
        : capacity(room(positions)), values_by_position(by_position) {
        const int64_t kv_dim = (int64_t) s.n_kv_head * s.head_dim;
        ggml_init_params params = {ggml_tensor_overhead() * (2 * s.n_layer + 1), nullptr, true};
        ctx = ggml_init(params);
        for (int l = 0; l < s.n_layer; l++) {
            k.push_back(ggml_new_tensor_2d(ctx, type, kv_dim, capacity));
            v.push_back(by_position ? ggml_new_tensor_2d(ctx, type, kv_dim, capacity) : ggml_new_tensor_2d(ctx, type, capacity, kv_dim));
        }
        buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
        if (!buffer) {
            ggml_free(ctx);
            throw Error(Fault::OutOfMemory, "cannot allocate a key/value cache of " + std::to_string(capacity) + " positions");
        }
    }
    ~Cache() {
        ggml_backend_buffer_free(buffer);
        ggml_free(ctx);
    }
    Cache(const Cache &) = delete;
    Cache & operator=(const Cache &) = delete;

    /** The positions a cache made for `positions` holds: `positions` rounded up to a multiple of kAttentionBlock. */
    static int64_t room(int64_t positions) { return round_up(positions, kAttentionBlock); }

    /** The keys of layer `l` at positions [from, from + n), [kv dim, n]. */
    ggml_tensor * keys(ggml_context * c, int l, int64_t from, int64_t n) const {
        return ggml_view_2d(c, k[l], k[l]->ne[0], n, k[l]->nb[1], from * k[l]->nb[1]);
    }
    /** The values of layer `l` at positions [from, from + n): [kv dim, n] by position, [n, kv dim] transposed. */
    ggml_tensor * values(ggml_context * c, int l, int64_t from, int64_t n) const {
        if (values_by_position) return ggml_view_2d(c, v[l], v[l]->ne[0], n, v[l]->nb[1], from * v[l]->nb[1]);
        return ggml_view_2d(c, v[l], n, v[l]->ne[1], v[l]->nb[1], from * v[l]->nb[0]);
    }
};

/**
 * The graph of a step whose input is a row of a table looked up by a token's id, which flash attention computes again
 * for the steps after it while they read the same cache and block of positions, their id, positions and mask given
 * anew: a step of the 0.6B Qwen3-ASR decoder at 350 to 450 positions took 6.5 ms building and allocating its graph and
 * takes 6.3 ms with the kept one on an Apple M5 in Q8_0 on Metal, interleaved in one process (2026-10-07). It has an
 * allocator of its own, so that no other graph moves its tensors.
 */
struct Qwen3Decoder::TokenStep {
    Graph graph{kGraphSize};
    ggml_gallocr_t allocr;
    ggml_tensor * table, * head;
    int64_t n_kv;
    ggml_tensor * id = nullptr, * hidden = nullptr, * logits = nullptr, * best = nullptr;
    RunInputs inputs;

    TokenStep(ggml_backend_t backend, ggml_tensor * table, ggml_tensor * head, int64_t n_kv)
        : allocr(ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend))), table(table), head(head), n_kv(n_kv) {}
    ~TokenStep() { ggml_gallocr_free(allocr); }
    TokenStep(const TokenStep &) = delete;
    TokenStep & operator=(const TokenStep &) = delete;
};

Qwen3Decoder::Qwen3Decoder(const ModelFile & m, ggml_backend_t backend, std::string tensors, const Qwen3Shape & shape,
                           int64_t max_positions, ggml_type cache_type, int64_t block_rows, std::optional<Qwen3Attention> attention)
    : backend_(backend),
      m_(m),
      tensors_(std::move(tensors)),
      shape_(shape),
      max_positions_(max_positions),
      cache_type_(cache_type),
      block_rows_(block_rows),
      flash_(attention.value_or(qwen3_attention(backend, shape, cache_type)) == Qwen3Attention::Flash) {
    if (block_rows_ < 1) throw std::logic_error("a decoder runs at least one row in a graph");
    if (flash_ && !computes_flash_attention(backend, shape, cache_type)) {
        throw Error(Fault::Device, std::string("the device ") + ggml_backend_name(backend) + " cannot compute flash attention over heads of " +
                                       std::to_string(shape.head_dim) + " with a " + ggml_type_name(cache_type) + " cache");
    }
    log_message(LogLevel::Info, "the decoder " + tensors_ + " attends with " + qwen3_attention_name(this->attention()) + " on " +
                                    ggml_backend_name(backend));
    allocr_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend_));
}

Qwen3Decoder::~Qwen3Decoder() {
    if (allocr_) ggml_gallocr_free(allocr_);
}

int64_t Qwen3Decoder::cache_capacity() const {
    return cache_ ? cache_->capacity : 0;
}

size_t Qwen3Decoder::graph_bytes() const {
    return ggml_gallocr_get_buffer_size(allocr_, 0);
}

void Qwen3Decoder::read_cache(int layer, std::vector<float> & keys, std::vector<float> & values) const {
    const int64_t kv_dim = (int64_t) shape_.n_kv_head * shape_.head_dim, capacity = cache_->capacity;
    const auto as_floats = [&](const ggml_tensor * t, int64_t count) {
        std::vector<uint8_t> raw(ggml_row_size(t->type, count));
        ggml_backend_tensor_get(t, raw.data(), 0, raw.size());
        std::vector<float> out((size_t) count);
        if (t->type == GGML_TYPE_F32) std::memcpy(out.data(), raw.data(), raw.size());
        else ggml_get_type_traits(t->type)->to_float(raw.data(), out.data(), count);
        return out;
    };
    keys = as_floats(cache_->k[layer], n_past_ * kv_dim);
    if (cache_->values_by_position) {
        values = as_floats(cache_->v[layer], n_past_ * kv_dim);
        return;
    }
    const std::vector<float> transposed = as_floats(cache_->v[layer], capacity * kv_dim);
    values.resize((size_t) (n_past_ * kv_dim));
    for (int64_t p = 0; p < n_past_; p++) {
        for (int64_t c = 0; c < kv_dim; c++) values[p * kv_dim + c] = transposed[c * capacity + p];
    }
}

void Qwen3Decoder::resize_cache(int64_t positions) {
    // The kept step reads the cache it was built on.
    token_step_.reset();
    chain_.reset();
    auto next = std::make_unique<Cache>(backend_, cache_type_, shape_, positions, flash_);
    // On the device, layer by layer, the positions so far are copied into the new buffer and, for flash attention, the
    // others are set to zero: flash attention reads the positions of the last block past the sequence, which its mask
    // hides, and a key or value there that was never written may be a NaN, which no mask hides. Set from the CPU, the
    // zeros counted every page of the cache in the process's memory footprint: the 1338.42 s input's peak was 6.15 GB
    // with the 0.6B Qwen3-ASR model in Q8_0 on Metal, against 2.52 GB in 0.7.0 (2026-10-07).
    if (n_past_ > 0 || flash_) {
        Graph g(kGraphSize);
        ggml_context * ctx = g.ctx();
        const int64_t rest = next->capacity - n_past_;
        for (int l = 0; l < shape_.n_layer; l++) {
            if (n_past_ > 0) {
                g.copy(cache_->keys(ctx, l, 0, n_past_), next->keys(ctx, l, 0, n_past_));
                g.copy(cache_->values(ctx, l, 0, n_past_), next->values(ctx, l, 0, n_past_));
            }
            if (flash_) {
                g.expand(ggml_fill_inplace(ctx, next->keys(ctx, l, n_past_, rest), 0.0f));
                g.expand(ggml_fill_inplace(ctx, next->values(ctx, l, n_past_, rest), 0.0f));
            }
        }
        g.compute(backend_, allocr_);
    }
    cache_ = std::move(next);
}

void Qwen3Decoder::start(int64_t positions, int64_t first) {
    if (positions > max_positions_ || first < 1 || first > positions) {
        throw std::logic_error("a sequence of " + std::to_string(positions) + " positions whose first run has " + std::to_string(first) +
                               " rows does not fit a decoder of " + std::to_string(max_positions_) + " positions");
    }
    positions_ = positions;
    n_past_ = 0;
    const int64_t room = std::min(positions, round_up(first + kCacheStep, kCacheStep));
    if (cache_capacity() != Cache::room(room)) resize_cache(room);
}

ggml_tensor * Qwen3Decoder::layers(Graph & g, ggml_tensor * x, int64_t rows, bool output, RunInputs & inputs) {
    const ModelFile & m = m_;
    const Qwen3Shape & s = shape_;
    ggml_context * ctx = g.ctx();
    const int64_t kv_dim = (int64_t) s.n_kv_head * s.head_dim;
    // The positions attention reads: flash attention reads whole blocks, which the cache holds.
    const int64_t n_kv = flash_ ? round_up(n_past_ + rows, kAttentionBlock) : n_past_ + rows;

    ggml_tensor * pos = inputs.positions = g.input(positions_after(n_past_, rows), rows);
    ggml_tensor * mask = nullptr;
    if (flash_) mask = g.half_input(causal_mask(n_past_, rows, n_kv), n_kv, rows);
    else if (rows > 1) mask = g.input(causal_mask(n_past_, rows, n_kv), n_kv, rows);
    inputs.mask = mask;

    static const bool cached = std::getenv("SPEECH_QWEN3_CACHED_TENSORS") != nullptr;
    if (cached && layer_tensors_.empty()) {
        static const char * names[] = {"attn_norm", "attn_q", "attn_k", "attn_v", "attn_q_norm", "attn_k_norm", "attn_o", "ffn_norm", "ffn_gate", "ffn_up", "ffn_down"};
        for (int l = 0; l < s.n_layer; l++) {
            std::array<ggml_tensor *, 11> t{};
            for (int i = 0; i < 11; i++) t[i] = m.tensor(tensors_ + ".blk." + std::to_string(l) + "." + names[i]);
            layer_tensors_.push_back(t);
        }
    }
    for (int l = 0; l < s.n_layer; l++) {
        std::string b;
        if (!cached) b = tensors_ + ".blk." + std::to_string(l) + ".";
        auto T = [&](int i, const char * name) { return cached ? layer_tensors_[l][i] : m.tensor(b + name); };
        ggml_tensor * h = ggml_mul(ctx, ggml_rms_norm(ctx, x, s.rms_eps), T(0, "attn_norm"));
        ggml_tensor * q = ggml_mul_mat(ctx, T(1, "attn_q"), h);
        ggml_tensor * k = ggml_mul_mat(ctx, T(2, "attn_k"), h);
        ggml_tensor * v = ggml_mul_mat(ctx, T(3, "attn_v"), h);
        // Nodes that read the same inputs follow one another in the graph: Metal runs a node without waiting for the
        // one before it when neither writes what the other reads. With the feed-forward's SwiGLU in one node, this took
        // a step of the 0.6B Qwen3-ASR decoder from 7.86 to 7.67 ms on an Apple M5 (2026-10-07).
        g.expand(q);
        g.expand(k);
        g.expand(v);
        q = ggml_mul(ctx, ggml_rms_norm(ctx, ggml_reshape_3d(ctx, q, s.head_dim, s.n_head, rows), s.rms_eps), T(4, "attn_q_norm"));
        k = ggml_mul(ctx, ggml_rms_norm(ctx, ggml_reshape_3d(ctx, k, s.head_dim, s.n_kv_head, rows), s.rms_eps), T(5, "attn_k_norm"));
        g.expand(q);
        g.expand(k);
        // On a GPU the rows are written to the cache rows their positions name, so that a kept step's graph writes
        // where its positions say.
        if (flash_) g.expand(ggml_set_rows(ctx, cache_->v[l], v, pos));
        else g.copy(ggml_transpose(ctx, v), cache_->values(ctx, l, n_past_, rows));
        q = ggml_rope_ext(ctx, q, pos, nullptr, s.head_dim, GGML_ROPE_TYPE_NEOX, 0, s.rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        k = ggml_rope_ext(ctx, k, pos, nullptr, s.head_dim, GGML_ROPE_TYPE_NEOX, 0, s.rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        g.expand(q);
        if (flash_) g.expand(ggml_set_rows(ctx, cache_->k[l], ggml_reshape_2d(ctx, k, kv_dim, rows), pos));
        else g.copy(ggml_reshape_2d(ctx, k, kv_dim, rows), cache_->keys(ctx, l, n_past_, rows));
        // A run whose output nobody reads computes only what the cache keeps: its last layer's keys and values.
        if (l + 1 == s.n_layer && !output) return nullptr;

        ggml_tensor * kc = cache_->k[l];
        ggml_tensor * vc = cache_->v[l];
        ggml_tensor * keys = ggml_view_3d(ctx, kc, s.head_dim, s.n_kv_head, n_kv, ggml_row_size(kc->type, s.head_dim), kc->nb[1], 0);
        ggml_tensor * kh = ggml_permute(ctx, keys, 0, 2, 1, 3);
        ggml_tensor * qh = ggml_permute(ctx, q, 0, 2, 1, 3);
        const float scale = 1.0f / std::sqrt((float) s.head_dim);
        ggml_tensor * o;
        if (flash_) {
            ggml_tensor * values = ggml_view_3d(ctx, vc, s.head_dim, s.n_kv_head, n_kv, ggml_row_size(vc->type, s.head_dim), vc->nb[1], 0);
            o = ggml_flash_attn_ext(ctx, qh, kh, ggml_permute(ctx, values, 0, 2, 1, 3), mask, scale, 0.0f, 0.0f);
            ggml_prec_set_acc(o, GGML_PREC_F32);
        } else {
            ggml_tensor * vt = ggml_view_3d(ctx, vc, n_kv, s.head_dim, s.n_kv_head, vc->nb[1], s.head_dim * vc->nb[1], 0);
            ggml_tensor * kq = ggml_soft_max_ext(ctx, ggml_mul_mat(ctx, kh, qh), mask, scale, 0.0f);
            o = ggml_cont(ctx, ggml_permute(ctx, ggml_mul_mat(ctx, vt, kq), 0, 2, 1, 3));
        }
        x = ggml_add(ctx, x, ggml_mul_mat(ctx, T(6, "attn_o"), ggml_reshape_2d(ctx, o, (int64_t) s.n_head * s.head_dim, rows)));

        h = ggml_mul(ctx, ggml_rms_norm(ctx, x, s.rms_eps), T(7, "ffn_norm"));
        ggml_tensor * gate = ggml_mul_mat(ctx, T(8, "ffn_gate"), h);
        ggml_tensor * up = ggml_mul_mat(ctx, T(9, "ffn_up"), h);
        g.expand(gate);
        g.expand(up);
        x = ggml_add(ctx, x, ggml_mul_mat(ctx, T(10, "ffn_down"), ggml_swiglu_split(ctx, gate, up)));
    }
    return x;
}

void Qwen3Decoder::make_room(int64_t n) {
    if (!cache_ || n < 1 || n_past_ + n > positions_) throw std::logic_error("a decoder was fed past the positions of its sequence");
    if (n_past_ + n > cache_->capacity) {
        resize_cache(std::min(positions_, std::max(n_past_ + n, round_up(2 * cache_->capacity, kCacheStep))));
    }
}

void Qwen3Decoder::outputs(Graph & g, ggml_tensor * last, ggml_tensor * head, ggml_tensor *& hidden, ggml_tensor *& logits) const {
    ggml_context * ctx = g.ctx();
    hidden = ggml_mul(ctx, ggml_rms_norm(ctx, last, shape_.rms_eps), m_.tensor(tensors_ + ".norm"));
    logits = ggml_mul_mat(ctx, head, hidden);
    g.output(hidden);
    g.output(logits);
}

void Qwen3Decoder::run(int64_t n, const Qwen3Rows & rows, ggml_tensor * head) {
    make_room(n);
    for (int64_t from = 0; from < n; from += block_rows_) {
        const int64_t count = std::min(block_rows_, n - from);
        const bool output = head && from + count == n;
        using clock = std::chrono::steady_clock;
        const auto t0 = clock::now();
        Graph g(kGraphSize);
        RunInputs inputs;
        ggml_tensor * x = layers(g, rows(g, from, count), count, output, inputs);
        ggml_tensor * hidden = nullptr, * logits = nullptr;
        if (output) outputs(g, ggml_view_2d(g.ctx(), x, shape_.hidden, 1, x->nb[1], (count - 1) * x->nb[1]), head, hidden, logits);
        ggml_tensor * best = nullptr;
        if (output && argmax_) {
            best = ggml_argmax(g.ctx(), logits);
            g.output(best);
        }
        const auto t1 = clock::now();
        if (!ggml_gallocr_alloc_graph(allocr_, g.cgraph())) throw Error(Fault::OutOfMemory, "cannot allocate the memory of a graph on the device");
        const auto t2 = clock::now();
        g.upload();
        const auto t3 = clock::now();
        compute_graph(backend_, g.cgraph());
        const auto t4 = clock::now();
        if (output) {
            if (reads_hidden_) Graph::read(hidden, hidden_);
            if (argmax_) ggml_backend_tensor_get(best, &best_, 0, sizeof(best_));
            else Graph::read(logits, logits_);
        }
        const auto t5 = clock::now();
        if (std::getenv("SPEECH_DECODER_TIMES")) {
            static std::map<std::string, std::array<double, 6>> totals;
            static bool registered = false;
            if (!registered) {
                registered = true;
                std::atexit([] {
                    for (auto & [name, t] : totals) {
                        std::fprintf(stderr, "decoder %s runs %.0f ms per run: build %.3f alloc %.3f upload %.3f compute %.3f read %.3f\n", name.c_str(), t[5],
                                     1e3 * t[0] / t[5], 1e3 * t[1] / t[5], 1e3 * t[2] / t[5], 1e3 * t[3] / t[5], 1e3 * t[4] / t[5]);
                    }
                });
            }
            auto & t = totals[tensors_ + (count == 1 ? " 1 row" : " rows")];
            const auto sec = [](auto a, auto b) { return std::chrono::duration<double>(b - a).count(); };
            t[0] += sec(t0, t1); t[1] += sec(t1, t2); t[2] += sec(t2, t3); t[3] += sec(t3, t4); t[4] += sec(t4, t5); t[5] += 1;
        }
        n_past_ += count;
    }
}

void Qwen3Decoder::step(int32_t id, ggml_tensor * table, ggml_tensor * head) {
    if (!flash_) {
        run(1, [&](Graph & g, int64_t, int64_t) { return ggml_get_rows(g.ctx(), table, g.input(std::vector<int32_t>{id}, 1)); }, head);
        return;
    }
    make_room(1);
    using clock = std::chrono::steady_clock;
    const auto t0 = clock::now();
    const int64_t n_kv = round_up(n_past_ + 1, kAttentionBlock);
    TokenStep * s = token_step_.get();
    if (s && s->table == table && s->head == head && s->n_kv == n_kv) {
        s->graph.set(s->id, std::vector<int32_t>{id});
        s->graph.set(s->inputs.positions, positions_after(n_past_, 1));
        s->graph.set(s->inputs.mask, causal_mask(n_past_, 1, n_kv));
        s->graph.compute_again(backend_);
    } else {
        token_step_ = std::make_unique<TokenStep>(backend_, table, head, n_kv);
        s = token_step_.get();
        s->id = s->graph.input(std::vector<int32_t>{id}, 1);
        ggml_tensor * x = layers(s->graph, ggml_get_rows(s->graph.ctx(), table, s->id), 1, true, s->inputs);
        outputs(s->graph, x, head, s->hidden, s->logits);
        if (argmax_) {
            s->best = ggml_argmax(s->graph.ctx(), s->logits);
            s->graph.output(s->best);
        }
        s->graph.compute(backend_, s->allocr);
    }
    const auto t1 = clock::now();
    if (reads_hidden_) Graph::read(s->hidden, hidden_);
    if (argmax_) ggml_backend_tensor_get(s->best, &best_, 0, sizeof(best_));
    else Graph::read(s->logits, logits_);
    const auto t2 = clock::now();
    if (std::getenv("SPEECH_DECODER_TIMES")) {
        static std::map<std::string, std::array<double, 3>> totals;
        static bool registered = false;
        if (!registered) {
            registered = true;
            std::atexit([] {
                for (auto & [name, t] : totals) {
                    std::fprintf(stderr, "decoder %s token steps %.0f ms per step: compute %.3f read %.3f\n", name.c_str(), t[2], 1e3 * t[0] / t[2], 1e3 * t[1] / t[2]);
                }
            });
        }
        auto & t = totals[tensors_];
        t[0] += std::chrono::duration<double>(t1 - t0).count();
        t[1] += std::chrono::duration<double>(t2 - t1).count();
        t[2] += 1;
    }
    n_past_ += 1;
}

/** A chain's graph, kept to be computed again while the cache it was built on stays. */
struct Qwen3Decoder::Chain {
    Graph graph{kGraphSize * 4};
    ggml_gallocr_t allocr;
    std::vector<ggml_tensor *> ids;
    const void * cache;
    explicit Chain(ggml_backend_t backend, const void * cache) : allocr(ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend))), cache(cache) {}
    ~Chain() { ggml_gallocr_free(allocr); }
};

std::vector<int32_t> Qwen3Decoder::chain(int64_t first_rows, const Qwen3Rows & first, int steps, const std::function<ggml_tensor *(int)> & head,
                                         const std::function<ggml_tensor *(Graph &, int, ggml_tensor *)> & next,
                                         const std::function<void(Graph &)> & refill) {
    make_room(first_rows + steps - 1);
    static const bool keep = std::getenv("SPEECH_CP_CHAIN_KEEP") != nullptr;
    if (keep && refill && chain_ && chain_->cache == cache_.get() && n_past_ == 0) {
        refill(chain_->graph);
        chain_->graph.compute_again(backend_);
        n_past_ += first_rows + steps - 1;
    } else {
        chain_ = std::make_unique<Chain>(backend_, cache_.get());
        Graph & g = chain_->graph;
        ggml_tensor * x = first(g, 0, first_rows);
        int64_t rows = first_rows;
        for (int s = 0; s < steps; s++) {
            RunInputs inputs;
            ggml_tensor * y = layers(g, x, rows, true, inputs);
            ggml_tensor * hidden = nullptr, * logits = nullptr;
            outputs(g, ggml_view_2d(g.ctx(), y, shape_.hidden, 1, y->nb[1], (rows - 1) * y->nb[1]), head(s), hidden, logits);
            ggml_tensor * id = ggml_argmax(g.ctx(), logits);
            g.output(id);
            chain_->ids.push_back(id);
            n_past_ += rows;
            if (s + 1 < steps) {
                x = next(g, s, id);
                rows = 1;
            }
        }
        g.compute(backend_, chain_->allocr);
    }
    std::vector<int32_t> out(chain_->ids.size());
    for (size_t i = 0; i < out.size(); i++) ggml_backend_tensor_get(chain_->ids[i], &out[i], 0, sizeof(int32_t));
    return out;
}
