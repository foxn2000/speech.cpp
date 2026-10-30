#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "graph.h"
#include "model-file.h"

namespace fastconformer {

/** Each prediction network defines the sizes and layouts of its flat state buffers. */
using PredictionState = std::vector<std::vector<float>>;

/**
 * Whether the tensor `name` is one that a decoding reads: the prediction network's and the joint's, but for the joint's
 * projection of the encoder, which the encoder's graph computes.
 */
bool decoding_tensor(const std::string & name);

/**
 * Whether every matrix a decoding multiplies by, the prediction network's layers and the joint's projection of the
 * prediction and output layer, is quantized in the file `m`: Q8_0 or a K-quant rather than F16 or F32.
 */
bool quantized_decoding(const ModelFile & m);

/** fastconformer.decoder.prediction selects the label embedding and recurrent prediction network. */
class PredictionNetwork {
public:
    virtual ~PredictionNetwork() = default;

    /** A step carries one prediction and its next flat state buffers. */
    struct Step {
        ggml_tensor * output;
        std::vector<ggml_tensor *> state;
    };

    /** The initial blank label receives a zero state. */
    virtual PredictionState initial_state() const = 0;
    Step build(Graph & g, int32_t label, const PredictionState & state) const;

    /** Builds a step from reusable label and flat state inputs. */
    virtual Step build(Graph & g, ggml_tensor * label, const std::vector<ggml_tensor *> & state) const = 0;

    /** Reads the flat state buffers after the step's graph computes them. */
    static PredictionState read_state(const Step & step);
    virtual int hidden() const = 0;
};

/** Makes the prediction network named by the model's layout. */
std::unique_ptr<PredictionNetwork> make_prediction(const ModelFile & m);

/**
 * NeMo's RNNTJoint: the encoder's frames and the prediction network's output each projected to the joint's width,
 * summed, a ReLU and a linear layer to the outputs, the tokens and the blank then, for TDT, one per duration.
 * NeMo applies a log-softmax over all outputs on the CPU only, which changes no argmax and is left out.
 */
class Joint {
public:
    explicit Joint(const ModelFile & m);

    /** The encoder's output [d_model, T] projected, [hidden, T]; computed once per utterance. */
    ggml_tensor * project_encoder(ggml_context * ctx, ggml_tensor * encoded) const;
    /** The prediction network's output [prediction hidden, n] projected, [hidden, n]. */
    ggml_tensor * project_prediction(ggml_context * ctx, ggml_tensor * prediction) const;

    /**
     * The logits of projected frames `f` ([hidden, n]) with projected predictions `g`, one for all frames ([hidden])
     * or one each ([hidden, n]), [outputs, n].
     */
    ggml_tensor * build(ggml_context * ctx, ggml_tensor * f, ggml_tensor * g) const;

    int hidden() const { return hidden_; }
    int outputs() const { return outputs_; }

private:
    const ModelFile & m_;
    int hidden_, outputs_;
};

}  // namespace fastconformer
