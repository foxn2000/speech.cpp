#include "lstm.h"

namespace fastconformer {

Lstm::Lstm(const ModelFile & m)
    : m_(m), layers_(m.count("fastconformer.decoder.prediction_layers")), hidden_((int) m.tensor("pred.embed.weight")->ne[0]) {}

PredictionState Lstm::initial_state() const {
    const size_t n = (size_t) hidden_ * (size_t) layers_;
    return {std::vector<float>(n, 0.0f), std::vector<float>(n, 0.0f)};
}

/**
 * torch.nn.LSTM's step: gates = W_ih x + W_hh h + b, stacked as input, forget, cell and output;
 * c' = sigmoid(f) c + sigmoid(i) tanh(g) and h' = sigmoid(o) tanh(c'). The converter sums b_ih and b_hh.
 */
PredictionNetwork::Step Lstm::build(Graph & g, ggml_tensor * label, const std::vector<ggml_tensor *> & state) const {
    ggml_tensor * h_in = state[0];
    ggml_tensor * c_in = state[1];
    ggml_context * ctx = g.ctx();
    const size_t width = (size_t) hidden_ * sizeof(float);
    ggml_tensor * x = ggml_get_rows(ctx, m_.tensor("pred.embed.weight"), label);
    ggml_tensor * h_out = nullptr;
    ggml_tensor * c_out = nullptr;
    for (int l = 0; l < layers_; l++) {
        const std::string p = "pred.lstm." + std::to_string(l);
        ggml_tensor * h = ggml_view_1d(ctx, h_in, hidden_, (size_t) l * width);
        ggml_tensor * c = ggml_view_1d(ctx, c_in, hidden_, (size_t) l * width);
        ggml_tensor * gates = ggml_add(ctx, ggml_add(ctx, mul_mat(ctx, m_.tensor(p + ".ih.weight"), x), mul_mat(ctx, m_.tensor(p + ".hh.weight"), h)),
                                       m_.tensor(p + ".bias"));
        auto gate = [&](int i) { return ggml_view_1d(ctx, gates, hidden_, (size_t) i * width); };
        ggml_tensor * next_c = ggml_add(ctx, ggml_mul(ctx, ggml_sigmoid(ctx, gate(1)), c), ggml_mul(ctx, ggml_sigmoid(ctx, gate(0)), ggml_tanh(ctx, gate(2))));
        ggml_tensor * next_h = ggml_mul(ctx, ggml_sigmoid(ctx, gate(3)), ggml_tanh(ctx, next_c));
        h_out = h_out ? ggml_concat(ctx, h_out, next_h, 0) : next_h;
        c_out = c_out ? ggml_concat(ctx, c_out, next_c, 0) : next_c;
        x = next_h;
    }
    return {x, {h_out, c_out}};
}

}  // namespace fastconformer
