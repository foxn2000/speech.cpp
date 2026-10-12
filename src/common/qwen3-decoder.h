#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "graph.h"
#include "model-file.h"

/** The shape of one Qwen3 decoder stack. */
struct Qwen3Shape {
    int hidden = 0;
    int ffn = 0;
    int n_layer = 0;
    int n_head = 0;
    int n_kv_head = 0;
    int head_dim = 0;
    float rms_eps = 0;
    float rope_theta = 0;
};

/**
 * The shape that the keys `<prefix>.hidden_size`, `.intermediate_size`, `.num_hidden_layers`, `.num_attention_heads`,
 * `.num_key_value_heads`, `.head_dim`, `.rms_norm_eps` and `.rope_theta` of `m` give, named as Qwen3's configuration
 * names them. A size of 0, heads that are no multiple of the key/value heads and an odd head width, which RoPE cannot
 * turn in pairs, throw.
 */
Qwen3Shape read_qwen3_shape(const ModelFile & m, const std::string & prefix);

/**
 * Appends to `t` the tensors of the stack `tensors` of shape `s` that Qwen3Decoder reads, its matrices stored as
 * `matrix` says and its norms in F32.
 */
void add_qwen3_tensors(std::vector<TensorSpec> & t, const std::string & tensors, const Qwen3Shape & s, const Storage & matrix);

/** Builds rows [from, from + rows) of a run's input, [hidden, rows], in the graph `g` of the block that holds them. */
using Qwen3Rows = std::function<ggml_tensor *(Graph & g, int64_t from, int64_t rows)>;

/**
 * The rows one graph of a run computes at most. On the CPU each row of a block holds a score for every position before
 * it, so a block's scores take block rows × positions × heads × 4 bytes, against the square of the positions in one
 * graph: for the longest prompt of Qwen3-TTS's talker, 24,576 rows, 0.8 GB where one graph needs 39 GB. On an Apple
 * M5, blocks of 128 to 1024 rows prefilled 5,137 rows of the 0.6B talker in Q8_0 on the CPU equally fast, in 39 to
 * 42 s, against 86 to 99 s in one graph (2026-10-06). Flash attention, on a GPU, holds no scores: there the blocks
 * bound the other activations, and 3,665 rows of the 0.6B talker in F32 took 1.57 s in blocks of 512 and 1.62 s in one
 * graph on Metal (2026-10-07).
 */
constexpr int64_t kQwen3BlockRows = 512;

/** How a Qwen3 decoder stack attends. */
enum class Qwen3Attention {
    /** ggml_flash_attn_ext over a cache that keeps the values a row per position. */
    Flash,
    /** Two matrix products and a softmax over a cache that keeps the values transposed, a row per channel. */
    Products,
};

/**
 * The attention a stack of shape `shape` with a cache of `cache_type` runs on `backend`: flash attention on a GPU whose
 * backend computes it for those heads and that cache, and the two products on the CPU, where flash attention took 51 ms
 * against 22 ms for 8,000 positions of 28 layers on an Apple M5 (2026-10-07) and sums the values in half precision, and
 * on any other GPU.
 */
Qwen3Attention qwen3_attention(ggml_backend_t backend, const Qwen3Shape & shape, ggml_type cache_type);

/** The attention's name, as a log says it. */
const char * qwen3_attention_name(Qwen3Attention attention);

/**
 * A Qwen3 decoder stack of a model file: in each layer, RMSNorm, attention with grouped key/value heads, RMSNorm of
 * each head's queries and keys and rotate-half RoPE, and RMSNorm and a SwiGLU feed-forward; and RMSNorm at the end.
 * Its tensors are `<tensors>.blk.<layer>.` attn_norm, attn_q, attn_k, attn_v, attn_q_norm, attn_k_norm, attn_o,
 * ffn_norm, ffn_gate, ffn_up and ffn_down, and `<tensors>.norm`. Its attention is chosen once, when it is made, by
 * what its device computes (qwen3_attention()).
 *
 * It runs one sequence at a time and keeps the keys and values of the sequence's positions in a cache that grows with
 * the sequence. The rows of a run go through in blocks, each block attending to the positions before it and causally
 * to its own rows, so that the memory of a long run grows with its rows rather than with their square. The caller
 * builds a run's input rows, embeddings it computed or rows the graph looks up from ids, and names the output matrix
 * whose logits the run's last row gives: a head of the model's own, or the input embeddings where the model ties its
 * head to them.
 */
class Qwen3Decoder {
public:
    /**
     * The stack `tensors` of `m`, which outlives it, of shape `shape`, for sequences of at most `max_positions`
     * positions, with a cache of `cache_type`, running blocks of at most `block_rows` rows, and attending as
     * `attention` says or, without it, as qwen3_attention() chooses; it logs which. Flash attention that the backend
     * cannot compute throws an Error of the device.
     */
    Qwen3Decoder(const ModelFile & m, ggml_backend_t backend, std::string tensors, const Qwen3Shape & shape,
                 int64_t max_positions, ggml_type cache_type, int64_t block_rows = kQwen3BlockRows,
                 std::optional<Qwen3Attention> attention = std::nullopt);
    ~Qwen3Decoder();
    Qwen3Decoder(const Qwen3Decoder &) = delete;
    Qwen3Decoder & operator=(const Qwen3Decoder &) = delete;

    const Qwen3Shape & shape() const { return shape_; }
    Qwen3Attention attention() const { return flash_ ? Qwen3Attention::Flash : Qwen3Attention::Products; }
    int64_t max_positions() const { return max_positions_; }
    /** The positions of the current sequence that the cache holds. */
    int64_t n_past() const { return n_past_; }
    /** The positions the cache holds before it grows. */
    int64_t cache_capacity() const;
    /** The bytes the device holds for computing a graph: what the largest graph so far took. */
    size_t graph_bytes() const;

    /**
     * Starts a sequence of at most `positions` positions whose first run has `first` rows. The cache gets room for
     * those rows and one step of its growth more, up to `positions`, and a cache that a longer sequence grew is given
     * back. More positions than max_positions() throw.
     */
    void start(int64_t positions, int64_t first);

    /**
     * Runs `n` rows, which `rows` builds, through the stack after the positions the sequence has so far, growing the
     * cache when it has no room for them. With a `head`, an output matrix [hidden, vocabulary], the last row's hidden
     * state after the final norm and its logits under `head` are read for hidden() and logits().
     */
    void run(int64_t n, const Qwen3Rows & rows, ggml_tensor * head = nullptr);

    /**
     * Feeds the token `id`, whose row of `table` is its input, after the sequence, and reads its hidden state and its
     * logits under `head` as run() does. On a GPU the graph of such a step is kept and computed again for the steps
     * after it while they read the same block of positions.
     */
    void step(int32_t id, ggml_tensor * table, ggml_tensor * head);

    /**
     * Runs `steps` steps after the sequence in one graph: the first step's `first_rows` rows from `first`, and each later
     * step's one row from `next`, given the id the step before it chose; each step's id is the argmax of its last row's
     * logits under `head(step)`. Returns the ids.
     */
    /** A step's choice in a chain: the candidates' ids, [k] (none for the whole vocabulary), and the pick among them, [1]. */
    struct ChainPick {
        ggml_tensor * candidates = nullptr;
        ggml_tensor * pick = nullptr;
    };
    std::vector<int32_t> chain(int64_t first_rows, const Qwen3Rows & first, int steps, const std::function<ggml_tensor *(int)> & head,
                               const std::function<ChainPick(Graph &, int, ggml_tensor *)> & select,
                               const std::function<ggml_tensor *(Graph &, int, const ChainPick &)> & next,
                               const std::function<void(Graph &)> & refill = {});

    const std::vector<float> & hidden() const { return hidden_; }
    /** Whether run() reads the last row's hidden state back; a caller that never asks for hidden() turns it off. */
    void set_reads_hidden(bool reads) { reads_hidden_ = reads; }
    /** Whether a run or step reads only the logits' argmax, best(), rather than the logits. */
    void set_argmax(bool argmax) { argmax_ = argmax; }
    int32_t best() const { return best_; }
    const std::vector<float> & logits() const { return logits_; }

    /**
     * The keys and the values the cache holds for layer `layer` at the n_past() positions, as float32 a row per
     * position, [n_past, kv heads * head dim] each: what a check compares row by row.
     */
    void read_cache(int layer, std::vector<float> & keys, std::vector<float> & values) const;

private:
    struct Cache;
    struct TokenStep;
    struct Chain;

    /** The inputs of a run's graph that depend on where its rows are in the sequence. */
    struct RunInputs {
        /** The rows' positions, which on a GPU are also the cache rows they are written to. */
        ggml_tensor * positions = nullptr;
        /** What each row sees of the positions attention reads, or none on the CPU for a single row. */
        ggml_tensor * mask = nullptr;
    };

    /**
     * Builds the layers over the `rows` rows of `x` that follow the n_past() positions, writing their keys and values,
     * and returns their output, or with `output` false none, the last layer stopping at its keys and values. `inputs`
     * gets the inputs it made.
     */
    ggml_tensor * layers(Graph & g, ggml_tensor * x, int64_t rows, bool output, RunInputs & inputs);
    /** Builds the hidden state after the final norm of `last`, one row, and its logits under `head`, as outputs of `g`. */
    void outputs(Graph & g, ggml_tensor * last, ggml_tensor * head, ggml_tensor *& hidden, ggml_tensor *& logits) const;
    /** Grows the cache when it has no room for `n` more rows. */
    void make_room(int64_t n);
    /** Moves the cache to one with room for `positions` positions, keeping the n_past() it holds. */
    void resize_cache(int64_t positions);

    ggml_backend_t backend_;
    const ModelFile & m_;
    const std::string tensors_;
    const Qwen3Shape shape_;
    const int64_t max_positions_;
    const ggml_type cache_type_;
    const int64_t block_rows_;
    /** Whether attention runs as ggml_flash_attn_ext rather than as two products. */
    const bool flash_;

    std::unique_ptr<Cache> cache_;
    std::unique_ptr<TokenStep> token_step_;
    std::unique_ptr<Chain> chain_;
    ggml_gallocr_t allocr_ = nullptr;
    int64_t positions_ = 0;
    int64_t n_past_ = 0;
    std::vector<float> hidden_, logits_;
    bool reads_hidden_ = true;
    bool argmax_ = false;
    int32_t best_ = 0;
    std::vector<std::array<ggml_tensor *, 11>> layer_tensors_;
};
