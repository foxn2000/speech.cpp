#include "text-model.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "error.h"

namespace drama3 {

const ModelFile * g_experts = nullptr;
ggml_backend_t g_cpu = nullptr;

namespace {

/**
 * The rows one graph of a run takes at most: a block's attention scores take rows × positions × heads × 4 bytes, 268 MB
 * for a block of the longest prompt, 8,192 rows of 16 heads.
 */
constexpr int64_t kBlockRows = 512;
/** The epsilon of the official l2norm() of the Gated DeltaNet's queries and keys: x * rsqrt(sum(x^2) + 1e-6). */
constexpr float kL2Eps = 1e-6f;

std::string layer_name(int l) {
    return "blk." + std::to_string(l) + ".";
}

}  // namespace

/**
 * The recurrent state of each Gated DeltaNet layer of the run, [key dim, value dim, value heads] with the key channel
 * fastest as ggml_gated_delta_net() keeps it, and the inputs of its convolution's last rows, [width - 1, channels].
 */
struct TextModel::States {
    ggml_context * ctx = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    std::vector<ggml_tensor *> delta, conv;

    States(ggml_backend_t backend, int layers, int key_dim, int value_dim, int value_heads, int width, int64_t channels) {
        ggml_init_params params = {ggml_tensor_overhead() * (size_t) (2 * layers + 1), nullptr, true};
        ctx = ggml_init(params);
        if (!ctx) throw Error(Fault::OutOfMemory, "cannot create a ggml context");
        for (int l = 0; l < layers; l++) {
            delta.push_back(ggml_new_tensor_3d(ctx, GGML_TYPE_F32, key_dim, value_dim, value_heads));
            conv.push_back(ggml_new_tensor_2d(ctx, GGML_TYPE_F32, width - 1, channels));
        }
        buffer = layers > 0 ? ggml_backend_alloc_ctx_tensors(ctx, backend) : nullptr;
        if (layers > 0 && !buffer) {
            ggml_free(ctx);
            throw Error(Fault::OutOfMemory, "cannot allocate the Gated DeltaNet's states");
        }
    }
    ~States() {
        if (buffer) ggml_backend_buffer_free(buffer);
        ggml_free(ctx);
    }
    States(const States &) = delete;
    States & operator=(const States &) = delete;
};

TextModel::TextModel(const ModelFile & m, ggml_backend_t backend, int first, int last) : backend_(backend), m_(m) {
    const std::string x = "drama3.text.", a = "drama3.audio_decoder.";
    for (const std::string & type : m.str_array(x + "layer_types")) types_.push_back(type == "full_attention");
    first_ = first;
    last_ = last < 0 ? layers() : last;
    if (first_ < 0 || first_ >= last_ || last_ > layers()) throw std::invalid_argument("the text model's layers to run are not among its layers");
    h_ = m.size(x + "hidden_size");
    heads_ = m.size(x + "num_attention_heads");
    kv_heads_ = m.size(x + "num_key_value_heads");
    head_dim_ = m.size(x + "head_dim");
    rotary_dim_ = m.size(x + "rotary_dim");
    key_heads_ = m.size(x + "linear_num_key_heads");
    value_heads_ = m.size(x + "linear_num_value_heads");
    key_dim_ = m.size(x + "linear_key_head_dim");
    value_dim_ = m.size(x + "linear_value_head_dim");
    conv_width_ = m.size(x + "linear_conv_kernel_dim");
    experts_ = m.size(x + "num_experts");
    used_ = m.size(x + "num_experts_per_tok");
    norm_topk_ = m.boolean(x + "norm_topk_prob");
    eps_ = m.f32(x + "rms_norm_eps");
    rope_theta_ = m.f32(x + "rope_theta");
    codebooks_ = m.count(a + "num_codebooks");
    codes_ = m.size(a + "vocab_size");
    seen_ = (int) m.u32(a + "vq_backbone_codebooks");
    semantic_begin_ = (int32_t) m.u32("drama3.semantic_start_token_id");
    semantic_end_ = (int32_t) m.u32("drama3.semantic_end_token_id");
    int attention_layers = 0, delta_layers = 0;
    slot_.assign((size_t) layers(), -1);
    for (int l = first_; l < last_; l++) slot_[(size_t) l] = attention(l) ? attention_layers++ : delta_layers++;
    const int64_t channels = 2 * (int64_t) key_heads_ * key_dim_ + (int64_t) value_heads_ * value_dim_;
    states_ = std::make_unique<States>(backend_, delta_layers, key_dim_, value_dim_, value_heads_, conv_width_, channels);
    allocr_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend_));
}

TextModel::~TextModel() {
    if (allocr_) ggml_gallocr_free(allocr_);
}

void TextModel::start(int64_t rows) {
    const int attention_layers = (int) std::count(types_.begin() + first_, types_.begin() + last_, true);
    if (attention_layers > 0 && (!keys_values_ || keys_values_->capacity() < rows)) {
        keys_values_.reset();
        keys_values_ = std::make_unique<KeysValues>(backend_, attention_layers, head_dim_, kv_heads_, rows);
    }
    if (states_->buffer) ggml_backend_buffer_clear(states_->buffer, 0);
    capacity_ = rows;
    n_past_ = 0;
}

ggml_tensor * TextModel::attention_mixer(Graph & g, int l, ggml_tensor * x, int64_t first, ggml_tensor * pos, ggml_tensor * mask) {
    ggml_context * c = g.ctx();
    const std::string p = layer_name(l);
    const int64_t n = x->ne[1];
    const auto norm = [&](ggml_tensor * t, const std::string & name) { return ggml_mul(c, ggml_rms_norm(c, t, eps_), m_.tensor(p + name)); };
    ggml_tensor * q = ggml_reshape_3d(c, mul_mat(c, m_.tensor(p + "attn_q"), x), head_dim_, heads_, n);
    ggml_tensor * k = ggml_reshape_3d(c, mul_mat(c, m_.tensor(p + "attn_k"), x), head_dim_, kv_heads_, n);
    ggml_tensor * v = ggml_reshape_3d(c, mul_mat(c, m_.tensor(p + "attn_v"), x), head_dim_, kv_heads_, n);
    ggml_tensor * gate = ggml_sigmoid(c, mul_mat(c, m_.tensor(p + "attn_gate"), x));
    // RoPE turns the first rotary_dim channels of each head in two halves, ggml's NeoX mode.
    q = ggml_rope_ext(c, norm(q, "attn_q_norm"), pos, nullptr, rotary_dim_, GGML_ROPE_TYPE_NEOX, 0, rope_theta_, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    k = ggml_rope_ext(c, norm(k, "attn_k_norm"), pos, nullptr, rotary_dim_, GGML_ROPE_TYPE_NEOX, 0, rope_theta_, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    ggml_tensor * o = ggml_mul(c, keys_values_->attend(g, slot_[(size_t) l], q, k, v, first, mask), gate);
    return mul_mat(c, m_.tensor(p + "attn_output"), o);
}

ggml_tensor * TextModel::delta_mixer(Graph & g, int l, ggml_tensor * x) {
    ggml_context * c = g.ctx();
    const std::string p = layer_name(l);
    const int64_t n = x->ne[1], keys = (int64_t) key_heads_ * key_dim_, channels = 2 * keys + (int64_t) value_heads_ * value_dim_;
    ggml_tensor * conv_state = states_->conv[(size_t) slot_[(size_t) l]];
    ggml_tensor * state = states_->delta[(size_t) slot_[(size_t) l]];

    // The causal convolution runs over the rows before these, whose inputs the state keeps, and these.
    ggml_tensor * qkv = mul_mat(c, m_.tensor(p + "ssm_in_qkv"), x);
    ggml_tensor * rows = ggml_concat(c, conv_state, ggml_cont(c, ggml_transpose(c, qkv)), 0);
    g.copy(ggml_view_2d(c, rows, conv_width_ - 1, channels, rows->nb[1], (size_t) n * rows->nb[0]), conv_state);
    ggml_tensor * mixed = ggml_silu(c, ggml_ssm_conv(c, ggml_reshape_3d(c, rows, conv_width_ - 1 + n, channels, 1), m_.tensor(p + "ssm_conv1d")));
    mixed = ggml_reshape_2d(c, mixed, channels, n);
    const auto heads = [&](int64_t offset, int64_t dim, int64_t count) {
        return ggml_cont(c, ggml_view_3d(c, mixed, dim, count, n, (size_t) dim * sizeof(float), mixed->nb[1], (size_t) offset * sizeof(float)));
    };
    // The official l2norm(), x * rsqrt(sum(x^2) + eps), is RMSNorm with eps / dim divided by sqrt(dim).
    const auto l2 = [&](ggml_tensor * t) { return ggml_scale(c, ggml_rms_norm(c, t, kL2Eps / (float) key_dim_), 1.0f / std::sqrt((float) key_dim_)); };
    ggml_tensor * q = l2(heads(0, key_dim_, key_heads_));
    ggml_tensor * k = l2(heads(keys, key_dim_, key_heads_));
    ggml_tensor * v = heads(2 * keys, value_dim_, value_heads_);
    // Each key head serves value_heads / key_heads value heads in a row, as repeat_interleave() gives them, where
    // ggml_gated_delta_net() would take key head i % key_heads for value head i.
    const int64_t groups = value_heads_ / key_heads_;
    if (groups > 1) {
        const auto interleave = [&](ggml_tensor * t) {
            return ggml_reshape_3d(c, ggml_repeat_4d(c, ggml_reshape_4d(c, t, key_dim_, 1, key_heads_, n), key_dim_, groups, key_heads_, n),
                                   key_dim_, value_heads_, n);
        };
        q = interleave(q);
        k = interleave(k);
    }
    ggml_tensor * beta = ggml_reshape_3d(c, ggml_sigmoid(c, mul_mat(c, m_.tensor(p + "ssm_in_b"), x)), 1, value_heads_, n);
    ggml_tensor * a = ggml_softplus(c, ggml_add(c, mul_mat(c, m_.tensor(p + "ssm_in_a"), x), m_.tensor(p + "ssm_dt_bias")));
    ggml_tensor * decay = ggml_mul(c, a, ggml_scale(c, ggml_exp(c, m_.tensor(p + "ssm_a_log")), -1.0f));
    decay = ggml_reshape_3d(c, decay, 1, value_heads_, n);

    // ggml_gated_delta_net() scales the queries by 1/sqrt(value dim), which is the key dim, as the official rule does.
    ggml_tensor * out = ggml_gated_delta_net(c, q, k, v, decay, beta, state, 1);
    const size_t outputs = (size_t) value_dim_ * value_heads_ * n;
    ggml_tensor * core = ggml_view_3d(c, out, value_dim_, value_heads_, n, (size_t) value_dim_ * sizeof(float),
                                      (size_t) value_dim_ * value_heads_ * sizeof(float), 0);
    g.copy(ggml_view_3d(c, out, key_dim_, value_dim_, value_heads_, state->nb[1], state->nb[2], outputs * sizeof(float)), state);
    ggml_tensor * z = ggml_reshape_3d(c, mul_mat(c, m_.tensor(p + "ssm_in_z"), x), value_dim_, value_heads_, n);
    core = ggml_mul(c, ggml_mul(c, ggml_rms_norm(c, ggml_cont(c, core), eps_), m_.tensor(p + "ssm_norm")), ggml_silu(c, z));
    return mul_mat(c, m_.tensor(p + "ssm_out"), ggml_reshape_2d(c, core, (int64_t) value_heads_ * value_dim_, n));
}

ggml_tensor * TextModel::experts(Graph & g, int l, ggml_tensor * x, ggml_tensor ** router, bool routed_only_skip) {
    ggml_context * c = g.ctx();
    const std::string p = layer_name(l);
    const int64_t n = x->ne[1];
    *router = mul_mat(c, m_.tensor(p + "ffn_gate_inp"), x);
    ggml_tensor * probabilities = ggml_soft_max(c, *router);
    ggml_tensor * chosen = ggml_top_k(c, probabilities, used_);
    ggml_tensor * weights = ggml_get_rows(c, ggml_reshape_3d(c, probabilities, 1, experts_, n), chosen);
    if (norm_topk_) weights = ggml_div(c, weights, ggml_reshape_3d(c, ggml_sum_rows(c, ggml_reshape_2d(c, weights, used_, n)), 1, 1, n));
    ggml_tensor * rows = ggml_reshape_3d(c, x, h_, 1, n);
    const ModelFile & ex = g_experts && g_experts->loaded(p + "ffn_gate_exps") ? *g_experts : m_;
    ggml_tensor * gate = mul_mat_id(c, ex.tensor(p + "ffn_gate_exps"), rows, chosen);
    ggml_tensor * up = mul_mat_id(c, ex.tensor(p + "ffn_up_exps"), rows, chosen);
    ggml_tensor * down = mul_mat_id(c, ex.tensor(p + "ffn_down_exps"), ggml_mul(c, ggml_silu(c, gate), up), chosen);
    down = ggml_mul(c, down, weights);
    ggml_tensor * sum = nullptr;
    for (int e = 0; e < used_; e++) {
        ggml_tensor * one = ggml_view_2d(c, down, h_, n, down->nb[2], (size_t) e * down->nb[1]);
        sum = sum ? ggml_add(c, sum, one) : one;
    }
    if (routed_only_skip) return sum;
    ggml_tensor * shared = mul_mat(c, m_.tensor(p + "ffn_down_shexp"),
                                   ggml_mul(c, ggml_silu(c, mul_mat(c, m_.tensor(p + "ffn_gate_shexp"), x)), mul_mat(c, m_.tensor(p + "ffn_up_shexp"), x)));
    shared = ggml_mul(c, shared, ggml_sigmoid(c, mul_mat(c, m_.tensor(p + "ffn_gate_inp_shexp"), x)));
    return ggml_add(c, sum, shared);
}

std::vector<float> TextModel::embed(const std::vector<int32_t> & frames, int64_t n) {
    const int width = 1 + codebooks_;
    if (n < 1 || (int64_t) frames.size() != n * width) throw std::invalid_argument("the text model embeds whole frames");
    ggml_tensor * table = m_.tensor("token_embd");
    std::vector<int32_t> ids((size_t) n), seen;
    std::vector<float> audio((size_t) n), text((size_t) n);
    for (int64_t i = 0; i < n; i++) {
        const int32_t id = frames[(size_t) (i * width)];
        if (id < 0 || id >= table->ne[1]) throw std::invalid_argument("a frame's main token is outside the vocabulary");
        ids[(size_t) i] = id;
        audio[(size_t) i] = id >= semantic_begin_ && id <= semantic_end_ ? 1.0f : 0.0f;
        text[(size_t) i] = 1.0f - audio[(size_t) i];
    }
    // The file holds the embeddings of codebooks 1 to 9, codebook b's from row (b - 1) * vocab_size.
    for (int b = 1; b <= seen_; b++) {
        for (int64_t i = 0; i < n; i++) {
            const int32_t code = frames[(size_t) (i * width + 1 + b)];
            if (code < 0 || code >= codes_) throw std::invalid_argument("a frame's code is outside its codebook");
            seen.push_back((b - 1) * codes_ + code);
        }
    }
    Graph g;
    ggml_context * c = g.ctx();
    ggml_tensor * e = ggml_get_rows(c, table, g.input(ids, n));
    if (seen_ > 0) {
        ggml_tensor * rows = ggml_get_rows(c, m_.tensor("ad.codebook_embd"), g.input(seen, (int64_t) seen.size()));
        ggml_tensor * mixed = e;
        for (int b = 0; b < seen_; b++) mixed = ggml_add(c, mixed, ggml_view_2d(c, rows, h_, n, rows->nb[1], (size_t) (b * n) * rows->nb[1]));
        mixed = ggml_scale(c, mixed, 1.0f / std::sqrt((float) (1 + seen_)));
        e = ggml_add(c, ggml_mul(c, e, g.input(text, 1, n)), ggml_mul(c, mixed, g.input(audio, 1, n)));
    }
    g.output(e);
    g.compute(backend_, allocr_);
    return Graph::read(e);
}

std::vector<float> TextModel::run(const std::vector<float> & input, int64_t n, TextModelStages * stages) {
    if (n < 1 || (int64_t) input.size() != n * h_) throw std::invalid_argument("the text model takes an input of each row");
    if (n_past_ + n > capacity_) throw std::invalid_argument("the text model's rows outgrow the sequence it was started for");
    const int count = last_ - first_;
    if (stages) {
        for (auto * s : {&stages->mixer, &stages->router, &stages->moe, &stages->layers}) s->assign((size_t) count, {});
    }
    std::vector<float> out;
    out.reserve(input.size());
    for (int64_t from = 0; from < n; from += kBlockRows) {
        const int64_t rows = std::min(kBlockRows, n - from);
        const auto tb = std::chrono::steady_clock::now();
        Graph g;
        ggml_context * c = g.ctx();
        ggml_tensor * x = g.input(std::vector<float>(input.begin() + from * h_, input.begin() + (from + rows) * h_), h_, rows);
        // A graph holds only the inputs its nodes read, and only attention reads the positions and the mask.
        const bool attends = std::count(types_.begin() + first_, types_.begin() + last_, true) > 0;
        ggml_tensor * pos = attends ? positions(g, n_past_, rows) : nullptr;
        ggml_tensor * mask = attends ? causal_mask(g, n_past_, rows) : nullptr;
        std::vector<std::vector<ggml_tensor *>> kept((size_t) count);
        for (int l = first_; l < last_; l++) {
            const std::string p = layer_name(l);
            ggml_tensor * h = ggml_mul(c, ggml_rms_norm(c, x, eps_), m_.tensor(p + "attn_norm"));
            static const char * skip = std::getenv("DRAMA3_SKIP");
            const bool skip_mixer = skip && std::string(skip).find("mixer") != std::string::npos && rows == 1;
            const bool skip_moe = skip && std::string(skip).find("moe") != std::string::npos && rows == 1;
            const bool skip_shared = skip && std::string(skip).find("routed") != std::string::npos && rows == 1;
            ggml_tensor * mixer = skip_mixer ? nullptr : attention(l) ? attention_mixer(g, l, h, n_past_, pos, mask) : delta_mixer(g, l, h);
            if (mixer) x = ggml_add(c, x, mixer);
            ggml_tensor * router = nullptr;
            ggml_tensor * moe = skip_moe ? nullptr : experts(g, l, ggml_mul(c, ggml_rms_norm(c, x, eps_), m_.tensor(p + "post_attn_norm")), &router, skip_shared);
            if (moe) x = ggml_add(c, x, moe);
            if (stages) {
                for (ggml_tensor * t : {mixer, router, moe, x}) g.output(t);
                kept[(size_t) (l - first_)] = {mixer, router, moe, x};
            }
        }
        if (last_ == layers()) x = ggml_mul(c, ggml_rms_norm(c, x, eps_), m_.tensor("output_norm"));
        g.output(x);
        const auto tc = std::chrono::steady_clock::now();
        if (g_cpu) {
            if (!sched_) {
                ggml_backend_t backends[2] = {backend_, g_cpu};
                sched_ = ggml_backend_sched_new(backends, nullptr, 2, 65536, false, std::getenv("DRAMA3_NO_OFFLOAD") == nullptr);
            }
            g.compute_scheduled(sched_);
            if (rows == 1 && std::getenv("DRAMA3_TIMING")) {
                static int shown = 0;
                if (shown++ == 0) std::fprintf(stderr, "scheduled row: %d splits, %d copies\n", ggml_backend_sched_get_n_splits(sched_), ggml_backend_sched_get_n_copies(sched_));
            }
        } else {
            g.compute(backend_, allocr_);
        }
        const auto te = std::chrono::steady_clock::now();
        if (rows == 1 && std::getenv("DRAMA3_TIMING")) {
            static double build = 0, compute = 0;
            static int calls = 0;
            build += std::chrono::duration<double, std::milli>(tc - tb).count();
            compute += std::chrono::duration<double, std::milli>(te - tc).count();
            if (++calls % 50 == 0) std::fprintf(stderr, "text row: build %.2f ms, alloc+compute %.2f ms, nodes %d\n", build / calls, compute / calls, ggml_graph_n_nodes(g.cgraph()));
        }
        if (stages) {
            for (int i = 0; i < count; i++) {
                std::vector<float> * where[] = {&stages->mixer[(size_t) i], &stages->router[(size_t) i], &stages->moe[(size_t) i],
                                                &stages->layers[(size_t) i]};
                for (int s = 0; s < 4; s++) {
                    const std::vector<float> v = Graph::read(kept[(size_t) i][(size_t) s]);
                    where[s]->insert(where[s]->end(), v.begin(), v.end());
                }
            }
        }
        const std::vector<float> y = Graph::read(x);
        out.insert(out.end(), y.begin(), y.end());
        n_past_ += rows;
    }
    return out;
}

std::vector<float> TextModel::logits(const std::vector<float> & hidden) {
    if ((int64_t) hidden.size() != h_) throw std::invalid_argument("the head takes the state of one row");
    Graph g;
    ggml_tensor * l = mul_mat(g.ctx(), m_.tensor("output"), g.input(hidden, h_, 1));
    g.output(l);
    g.compute(backend_, allocr_);
    return Graph::read(l);
}

std::vector<float> TextModel::delta_state(int l) const {
    return Graph::read(states_->delta[(size_t) slot_[(size_t) l]]);
}

std::vector<float> TextModel::conv_state(int l) const {
    return Graph::read(states_->conv[(size_t) slot_[(size_t) l]]);
}

}  // namespace drama3
