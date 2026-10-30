#pragma once

#include "transducer.h"

namespace fastconformer {

/** NeMo's RNNTDecoder uses an embedding and stacked LSTM, with the blank as the zero padding row. */
class Lstm final : public PredictionNetwork {
public:
    explicit Lstm(const ModelFile & m);
    PredictionState initial_state() const override;
    Step build(Graph & g, ggml_tensor * label, const std::vector<ggml_tensor *> & state) const override;
    int hidden() const override { return hidden_; }

private:
    const ModelFile & m_;
    int layers_, hidden_;
};

}  // namespace fastconformer
