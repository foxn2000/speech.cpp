#include "kda.h"

#include <cmath>

namespace fastconformer {

namespace {

/** Each layer holds its memory, then its q, k and v convolution histories. */
constexpr size_t kStatesPerLayer = 4;
size_t memory_index(int layer) { return (size_t) layer * kStatesPerLayer; }
size_t convolution_index(int layer, int channel) { return memory_index(layer) + 1 + channel; }

}  // namespace

Kda::Kda(const ModelFile & m)
    : m_(m), layers_(m.count("fastconformer.decoder.prediction_layers")),
      hidden_((int) m.tensor("pred.embed.weight")->ne[0]), heads_(m.size("fastconformer.decoder.kda.num_heads")),
      dim_((int) m.tensor("pred.kda.0.q.weight")->ne[1] / heads_),
      conv_((int) m.tensor("pred.kda.0.q_conv.weight")->ne[0]),
      norm_eps_(m.f32("fastconformer.decoder.kda.norm_eps")),
      gate_norm_eps_(m.f32("fastconformer.decoder.kda.gate_norm_eps")),
      qk_norm_eps_(m.f32("fastconformer.decoder.kda.qk_norm_eps")) {}

PredictionState Kda::initial_state() const {
    PredictionState state((size_t) layers_ * kStatesPerLayer);
    for (int l = 0; l < layers_; l++) {
        state[memory_index(l)].assign((size_t) heads_ * dim_ * dim_, 0.0f);
        for (int i = 0; i < (int) kStatesPerLayer - 1; i++) {
            state[convolution_index(l, i)].assign((size_t) heads_ * dim_ * (conv_ - 1), 0.0f);
        }
    }
    return state;
}

ggml_tensor * Kda::norm(ggml_context * ctx, ggml_tensor * x, const std::string & name) const {
    return ggml_add(ctx, ggml_mul(ctx, ggml_norm(ctx, x, norm_eps_), m_.tensor(name + ".weight")), m_.tensor(name + ".bias"));
}

PredictionNetwork::Step Kda::build(Graph & g, ggml_tensor * label, const std::vector<ggml_tensor *> & state) const {
    ggml_context * ctx = g.ctx();
    const int channels = heads_ * dim_;
    ggml_tensor * x = ggml_get_rows(ctx, m_.tensor("pred.embed.weight"), label);
    Step step{};
    step.state.resize((size_t) layers_ * kStatesPerLayer);
    for (int l = 0; l < layers_; l++) {
        const std::string p = "pred.kda." + std::to_string(l) + ".";
        ggml_tensor * h = norm(ctx, x, p + "norm");
        auto project = [&](ggml_tensor * in, const std::string & name) {
            return mul_mat(ctx, m_.tensor(p + name + ".weight"), in);
        };
        auto convolve = [&](const std::string & name, int i) {
            ggml_tensor * projected = ggml_reshape_2d(ctx, project(h, name), 1, channels);
            ggml_tensor * history = ggml_reshape_2d(ctx, state[convolution_index(l, i)], conv_ - 1, channels);
            ggml_tensor * window = ggml_concat(ctx, history, projected, 0);
            // The cache is oldest first, and each channel keeps only the W-1 inputs the next step needs.
            step.state[convolution_index(l, i)] = ggml_cont(ctx, ggml_view_2d(ctx, window, conv_ - 1, channels, window->nb[1], sizeof(float)));
            return ggml_reshape_2d(ctx, ggml_silu(ctx, ggml_ssm_conv(ctx, window, m_.tensor(p + name + "_conv.weight"))), dim_, heads_);
        };
        ggml_tensor * q = convolve("q", 0);
        ggml_tensor * k = convolve("k", 1);
        ggml_tensor * v = convolve("v", 2);
        // FLA adds epsilon to the squared L2 norm; ggml_l2_norm instead clamps the norm.
        auto l2_norm = [&](ggml_tensor * t) {
            return ggml_scale(ctx, ggml_rms_norm(ctx, t, qk_norm_eps_ / dim_), 1.0f / std::sqrt((float) dim_));
        };
        q = l2_norm(q);
        k = l2_norm(k);
        ggml_tensor * decay = ggml_add(ctx, project(project(h, "f_down"), "f_up"), m_.tensor(p + "dt_bias"));
        decay = ggml_reshape_2d(ctx, ggml_softplus(ctx, decay), dim_, heads_);
        ggml_tensor * a = ggml_reshape_2d(ctx, ggml_scale(ctx, ggml_exp(ctx, m_.tensor(p + "a_log")), -1.0f), 1, heads_);
        decay = ggml_mul(ctx, decay, a);
        ggml_tensor * beta = ggml_reshape_2d(ctx, ggml_sigmoid(ctx, project(h, "beta")), 1, heads_);
        ggml_tensor * memory = ggml_reshape_3d(ctx, state[memory_index(l)], dim_, dim_, heads_);
        // ggml_gated_delta_net returns [output (dim*heads) | next memory] in one tensor.
        ggml_tensor * recurrent = ggml_gated_delta_net(ctx, q, k, v, decay, beta, memory, 1);
        ggml_tensor * out = ggml_view_2d(ctx, recurrent, dim_, heads_, dim_ * sizeof(float), 0);
        step.state[memory_index(l)] = ggml_view_1d(ctx, recurrent, (int64_t) dim_ * dim_ * heads_, channels * sizeof(float));
        ggml_tensor * gate = ggml_add(ctx, project(project(h, "g_down"), "g_up"), m_.tensor(p + "g_up.bias"));
        out = ggml_mul(ctx, ggml_rms_norm(ctx, out, gate_norm_eps_), m_.tensor(p + "out_norm.weight"));
        out = ggml_mul(ctx, out, ggml_reshape_2d(ctx, ggml_sigmoid(ctx, gate), dim_, heads_));
        x = ggml_add(ctx, x, project(ggml_reshape_1d(ctx, out, channels), "out"));
    }
    step.output = norm(ctx, x, "pred.norm");
    return step;
}

}  // namespace fastconformer
