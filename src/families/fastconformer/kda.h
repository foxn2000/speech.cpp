#pragma once

#include "transducer.h"

namespace fastconformer {

/** Haas predicts labels through pre-norm residual Kimi Delta Attention layers and a final LayerNorm. */
class Kda final : public PredictionNetwork {
public:
    explicit Kda(const ModelFile & m);
    PredictionState initial_state() const override;
    Step build(Graph & g, ggml_tensor * label, const std::vector<ggml_tensor *> & state) const override;
    int hidden() const override { return hidden_; }

private:
    ggml_tensor * norm(ggml_context * ctx, ggml_tensor * x, const std::string & name) const;

    const ModelFile & m_;
    int layers_, hidden_, heads_, dim_, conv_;
    float norm_eps_, gate_norm_eps_, qk_norm_eps_;
};

}  // namespace fastconformer
