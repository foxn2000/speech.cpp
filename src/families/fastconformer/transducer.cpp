#include "transducer.h"

#include <string>

#include "kda.h"
#include "lstm.h"

namespace fastconformer {

namespace {

ggml_tensor * linear(ggml_context * ctx, const ModelFile & m, ggml_tensor * x, const std::string & name) {
    return ggml_add(ctx, mul_mat(ctx, m.tensor(name + ".weight"), x), m.tensor(name + ".bias"));
}

}  // namespace

bool decoding_tensor(const std::string & name) {
    for (const char * prefix : {"pred.", "joint.pred.", "joint.out."}) {
        if (name.rfind(prefix, 0) == 0) return true;
    }
    return false;
}

bool quantized_decoding(const ModelFile & m) {
    std::vector<std::string> names = {"joint.pred.weight", "joint.out.weight"};
    std::string prefix;
    std::vector<std::string> parts;
    if (m.str("fastconformer.decoder.prediction") == "lstm") {
        prefix = "pred.lstm.";
        parts = {"ih", "hh"};
    } else {
        prefix = "pred.kda.";
        parts = {"q", "k", "v", "f_down", "f_up", "beta", "g_down", "g_up", "out"};
    }
    for (int l = 0; l < m.count("fastconformer.decoder.prediction_layers"); l++) {
        for (const auto & part : parts) names.push_back(prefix + std::to_string(l) + "." + part + ".weight");
    }
    for (const std::string & name : names) {
        if (!ggml_is_quantized(m.stored_type(name))) return false;
    }
    return true;
}

std::unique_ptr<PredictionNetwork> make_prediction(const ModelFile & m) {
    if (m.one_of("fastconformer.decoder.prediction", {"lstm", "kda"}) == "lstm") return std::make_unique<Lstm>(m);
    return std::make_unique<Kda>(m);
}

PredictionNetwork::Step PredictionNetwork::build(Graph & g, int32_t label, const PredictionState & state) const {
    std::vector<ggml_tensor *> inputs;
    for (const auto & buffer : state) inputs.push_back(g.input(buffer, (int64_t) buffer.size()));
    return build(g, g.input(std::vector<int32_t>{label}, 1), inputs);
}

PredictionState PredictionNetwork::read_state(const Step & step) {
    PredictionState state;
    for (const auto * buffer : step.state) state.push_back(Graph::read(buffer));
    return state;
}

Joint::Joint(const ModelFile & m)
    : m_(m), hidden_((int) m.tensor("joint.out.weight")->ne[0]), outputs_((int) m.tensor("joint.out.weight")->ne[1]) {}

ggml_tensor * Joint::project_encoder(ggml_context * ctx, ggml_tensor * encoded) const {
    return linear(ctx, m_, encoded, "joint.enc");
}

ggml_tensor * Joint::project_prediction(ggml_context * ctx, ggml_tensor * prediction) const {
    return linear(ctx, m_, prediction, "joint.pred");
}

ggml_tensor * Joint::build(ggml_context * ctx, ggml_tensor * f, ggml_tensor * g) const {
    return linear(ctx, m_, ggml_relu(ctx, ggml_add(ctx, f, g)), "joint.out");
}

}  // namespace fastconformer
