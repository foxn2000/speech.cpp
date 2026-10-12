#pragma once

#include <cstdint>
#include <functional>
#include <vector>

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "model-file.h"
#include "qwen3-decoder.h"

/**
 * The talker and its code predictor, two Qwen3 decoder stacks. The talker reads one embedding per frame and predicts
 * the frame's first code; the code predictor, conditioned on the talker's last hidden state, predicts the other
 * fifteen one by one. Both keep a key/value cache: the talker's spans the utterance and grows with it, the code
 * predictor's starts again at every frame.
 */
class Talker {
public:
    /** The talker of the model file `m`, which outlives it. */
    Talker(const ModelFile & m, ggml_backend_t backend);
    ~Talker();
    Talker(const Talker &) = delete;
    Talker & operator=(const Talker &) = delete;

    const ModelFile & model() const { return m_; }
    int num_code_groups() const { return n_groups_; }
    int hidden() const { return talker_.shape().hidden; }
    int vocab() const { return vocab_; }
    int cp_vocab() const { return cp_vocab_; }

    /** Projected text embeddings of `ids`, row-major [ids.size(), hidden]. */
    std::vector<float> text_embeddings(const std::vector<int32_t> & ids);
    /** Talker codec embeddings of `ids`, row-major [ids.size(), hidden]. */
    std::vector<float> codec_embeddings(const std::vector<int32_t> & ids);

    /**
     * Starts an utterance of at most `positions` positions, the prompt's and the frames': gives the cache room for the
     * prompt and its first frames, and runs `embeds` ([n, hidden], row-major) through the talker.
     */
    void prefill(const std::vector<float> & embeds, int n, int64_t positions);
    /** Feeds one frame: the sum of its 16 code embeddings plus `extra` ([hidden]). */
    void step(const int32_t * codes, const std::vector<float> & extra);

    /** The codec head's logits for the next frame's first code. */
    const std::vector<float> & logits() const { return talker_.logits(); }
    /** The talker's last hidden state after the final norm, which conditions the code predictor. */
    const std::vector<float> & hidden_state() const { return talker_.hidden(); }

    /** Code predictor: starts a frame from the talker's hidden state and the first code, and returns the logits of code 1. */
    const std::vector<float> & cp_begin(int32_t code0);
    /** Feeds code `group` (1..14) and returns the logits of code `group + 1`. */
    const std::vector<float> & cp_next(int group, int32_t code);
    /** Code predictor: the codes 1 to 15 of a frame whose first code is `code0`, each its logits' argmax, in one graph. */
    std::vector<int32_t> cp_greedy_frame(int32_t code0);
    /**
     * The same, each code drawn on the device from its logits under `temperature` and the `top_k` largest, by the
     * uniform numbers `draws`, one a code.
     */
    std::vector<int32_t> cp_sampled_frame(int32_t code0, float temperature, int top_k, const std::vector<float> & draws);
    void print_debug(const std::vector<float> & draws);
    std::vector<std::pair<ggml_tensor *, ggml_tensor *>> debug_;

private:
    const std::vector<float> & run_cp(int32_t code, int group);

    ggml_backend_t backend_;
    const ModelFile & m_;
    int n_groups_ = 0;
    int vocab_ = 0;
    int cp_vocab_ = 0;
    Qwen3Decoder talker_, cp_;
    bool cp_projected_ = false;
    ggml_tensor * chain_code0_ = nullptr, * chain_hidden_ = nullptr, * chain_draws_ = nullptr;
    std::vector<int32_t> cp_frame(int32_t code0, const std::function<Qwen3Decoder::ChainPick(Graph &, int, ggml_tensor *)> & select,
                                  const std::function<void(Graph &)> & refill_more);
    /** The allocator of the embedding graphs; each decoder has its own. */
    ggml_gallocr_t allocr_ = nullptr;
};
