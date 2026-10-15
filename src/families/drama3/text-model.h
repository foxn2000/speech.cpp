#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "attention.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "graph.h"
#include "model-file.h"

namespace drama3 {

/** Experiment: the experts' file on the CPU and the CPU's backend, when the experts compute there. */
extern const ModelFile * g_experts;
extern ggml_backend_t g_cpu;

/**
 * What the layers of a run computed on the way, by layer, each [width, rows] with the channel fastest: the attention's
 * or the Gated DeltaNet's output, the router's logits, the experts' output with the shared expert's, and the layer's
 * output.
 */
struct TextModelStages {
    std::vector<std::vector<float>> mixer, router, moe, layers;
};

/**
 * Drama 3's text model, a Qwen3.5-MoE backbone, or a run of its layers: in each, RMSNorm and a Gated DeltaNet or gated
 * attention, then RMSNorm and 256 experts of which the router takes 8, with a shared expert whose sigmoid gate scales
 * it; after the last, RMSNorm and the head of the main token's logits. It runs one sequence at a time and keeps, for
 * each attention layer, the keys and values of its rows, and for each Gated DeltaNet layer its recurrent state and the
 * inputs of its convolution's last rows.
 */
class TextModel {
public:
    /**
     * Layers [first, last) of the text model of the model file `m`, which outlives it, and the final norm where `last`
     * is the model's last; -1 takes every layer from `first`. Only their tensors need be loaded.
     */
    TextModel(const ModelFile & m, ggml_backend_t backend, int first = 0, int last = -1);
    ~TextModel();
    TextModel(const TextModel &) = delete;
    TextModel & operator=(const TextModel &) = delete;

    int hidden() const { return h_; }
    /** The layers of the model, whichever this one runs. */
    int layers() const { return (int) types_.size(); }
    bool attention(int l) const { return types_[(size_t) l]; }
    /** The rows run so far. */
    int64_t n_past() const { return n_past_; }

    /** Starts a sequence of at most `rows` rows: the attention layers get room for them, and every state starts at 0. */
    void start(int64_t rows);

    /**
     * The inputs of `n` frames of 1 + codebooks ids, row-major, [hidden, n]: the main token's embedding, and for a frame
     * whose main token is a semantic id, the mean of it and the embeddings of the codes the text model sees. It reads
     * token_embd and the audio decoder's ad.codebook_embd.
     */
    std::vector<float> embed(const std::vector<int32_t> & frames, int64_t n);

    /**
     * Runs `n` rows of input `x`, [hidden, n], after n_past() through the layers, and returns the last layer's output,
     * after the final norm where it is the model's last. `stages`, when given, receives what each layer computed.
     */
    std::vector<float> run(const std::vector<float> & x, int64_t n, TextModelStages * stages = nullptr);

    /** The head's logits of `hidden`, [hidden] after the final norm, over drama3.output.token_ids. */
    std::vector<float> logits(const std::vector<float> & hidden);

    /**
     * Gated DeltaNet layer `l`'s recurrent state, [key dim, value dim, value heads] with the key channel fastest, and the
     * inputs of its convolution's last rows, [width - 1, channels] with the row fastest: what a check compares with the
     * official ones.
     */
    std::vector<float> delta_state(int l) const;
    std::vector<float> conv_state(int l) const;

private:
    struct States;

    ggml_tensor * attention_mixer(Graph & g, int l, ggml_tensor * x, int64_t first, ggml_tensor * pos, ggml_tensor * mask);
    ggml_tensor * delta_mixer(Graph & g, int l, ggml_tensor * x);
    /** The experts and the shared expert on `x`, and the router's logits in `router`. */
    ggml_tensor * experts(Graph & g, int l, ggml_tensor * x, ggml_tensor ** router, bool routed_only_skip = false);

    ggml_backend_t backend_;
    const ModelFile & m_;
    ggml_gallocr_t allocr_ = nullptr;
    ggml_backend_sched_t sched_ = nullptr;
    int first_ = 0, last_ = 0;
    std::vector<bool> types_;
    int h_ = 0, heads_ = 0, kv_heads_ = 0, head_dim_ = 0, rotary_dim_ = 0;
    int key_heads_ = 0, value_heads_ = 0, key_dim_ = 0, value_dim_ = 0, conv_width_ = 0;
    int experts_ = 0, used_ = 0;
    bool norm_topk_ = false;
    float eps_ = 0, rope_theta_ = 0;
    int codebooks_ = 0, codes_ = 0, seen_ = 0;
    int32_t semantic_begin_ = 0, semantic_end_ = 0;
    std::unique_ptr<KeysValues> keys_values_;
    std::unique_ptr<States> states_;
    /** The index of each layer among the attention layers or the Gated DeltaNet layers of the run. */
    std::vector<int> slot_;
    int64_t capacity_ = 0, n_past_ = 0;
};

}  // namespace drama3
