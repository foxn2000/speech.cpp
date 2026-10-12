#include "talker.h"

#include <cstdlib>
#include <string>

#include "graph.h"

namespace {

/** The nodes of an embedding graph at most. */
constexpr int kEmbeddingGraphSize = 32;

}  // namespace

Talker::Talker(const ModelFile & m, ggml_backend_t backend)
    : backend_(backend),
      m_(m),
      n_groups_((int) m.u32("qwen3-tts.talker.num_code_groups")),
      vocab_((int) m.u32("qwen3-tts.talker.vocab_size")),
      cp_vocab_((int) m.u32("qwen3-tts.code_predictor.vocab_size")),
      talker_(m, backend, "talker", read_qwen3_shape(m, "qwen3-tts.talker"), m.u32("qwen3-tts.talker.max_position_embeddings"),
              GGML_TYPE_F16),
      // The code predictor's sequence is the talker's hidden state and a frame's codes.
      cp_(m, backend, "cp", read_qwen3_shape(m, "qwen3-tts.code_predictor"), n_groups_ + 1, GGML_TYPE_F32),
      cp_projected_(cp_.shape().hidden != talker_.shape().hidden),
      allocr_(ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend))) {
    if (std::getenv("SPEECH_CP_NO_HIDDEN")) cp_.set_reads_hidden(false);
}

Talker::~Talker() {
    if (allocr_) ggml_gallocr_free(allocr_);
}

std::vector<float> Talker::text_embeddings(const std::vector<int32_t> & ids) {
    const ModelFile & m = m_;
    Graph g(kEmbeddingGraphSize);
    ggml_context * ctx = g.ctx();
    ggml_tensor * x = ggml_get_rows(ctx, m.tensor("talker.text_embd"), g.input(ids, (int64_t) ids.size()));
    x = ggml_add(ctx, ggml_mul_mat(ctx, m.tensor("talker.text_proj.fc1.weight"), x), m.tensor("talker.text_proj.fc1.bias"));
    x = ggml_silu(ctx, x);
    x = ggml_add(ctx, ggml_mul_mat(ctx, m.tensor("talker.text_proj.fc2.weight"), x), m.tensor("talker.text_proj.fc2.bias"));
    g.output(x);
    g.compute(backend_, allocr_);
    return Graph::read(x);
}

std::vector<float> Talker::codec_embeddings(const std::vector<int32_t> & ids) {
    Graph g(kEmbeddingGraphSize);
    ggml_tensor * x = ggml_get_rows(g.ctx(), m_.tensor("talker.codec_embd"), g.input(ids, (int64_t) ids.size()));
    g.output(x);
    g.compute(backend_, allocr_);
    return Graph::read(x);
}

void Talker::prefill(const std::vector<float> & embeds, int n, int64_t positions) {
    const int h = hidden();
    talker_.start(positions, n);
    talker_.run(
        n,
        [&](Graph & g, int64_t from, int64_t rows) {
            return g.input(std::vector<float>(embeds.begin() + from * h, embeds.begin() + (from + rows) * h), h, rows);
        },
        m_.tensor("talker.codec_head"));
}

void Talker::step(const int32_t * codes, const std::vector<float> & extra) {
    const ModelFile & m = m_;
    talker_.run(
        1,
        [&](Graph & g, int64_t, int64_t) {
            // A frame's input is the sum of its codes' embeddings: the first from the talker's table, the
            // others from the code predictor's, which live in the talker's hidden space.
            ggml_context * ctx = g.ctx();
            static const bool packed = std::getenv("SPEECH_TALKER_PACKED") != nullptr;
            ggml_tensor * all = packed ? g.input(std::vector<int32_t>(codes, codes + n_groups_), n_groups_) : nullptr;
            auto code = [&](int q) {
                return packed ? ggml_view_1d(ctx, all, 1, (size_t) q * sizeof(int32_t)) : g.input(std::vector<int32_t>{codes[q]}, 1);
            };
            ggml_tensor * x = ggml_get_rows(ctx, m.tensor("talker.codec_embd"), code(0));
            for (int q = 1; q < n_groups_; q++) {
                ggml_tensor * id = code(q);
                x = ggml_add(ctx, x, ggml_get_rows(ctx, m.tensor("cp.codec_embd." + std::to_string(q - 1)), id));
            }
            return ggml_add(ctx, x, g.input(extra, hidden()));
        },
        m.tensor("talker.codec_head"));
}

const std::vector<float> & Talker::cp_begin(int32_t code0) {
    cp_.start(n_groups_ + 1, 2);
    return run_cp(code0, 0);
}

const std::vector<float> & Talker::cp_next(int group, int32_t code) {
    return run_cp(code, group);
}

const std::vector<float> & Talker::run_cp(int32_t code, int group) {
    const ModelFile & m = m_;
    // The first step reads two rows: the talker's hidden state and the first code's embedding.
    cp_.run(
        group == 0 ? 2 : 1,
        [&](Graph & g, int64_t, int64_t) {
            ggml_context * ctx = g.ctx();
            ggml_tensor * id = g.input(std::vector<int32_t>{code}, 1);
            ggml_tensor * x;
            if (group == 0) {
                x = ggml_concat(ctx, g.input(talker_.hidden(), hidden(), 1), ggml_get_rows(ctx, m.tensor("talker.codec_embd"), id), 1);
            } else {
                x = ggml_get_rows(ctx, m.tensor("cp.codec_embd." + std::to_string(group - 1)), id);
            }
            if (cp_projected_) x = ggml_add(ctx, ggml_mul_mat(ctx, m.tensor("cp.in_proj.weight"), x), m.tensor("cp.in_proj.bias"));
            return x;
        },
        m.tensor("cp.head." + std::to_string(group)));
    return cp_.logits();
}

std::vector<int32_t> Talker::cp_greedy_frame(int32_t code0) {
    const ModelFile & m = m_;
    cp_.start(n_groups_ + 1, 2);
    auto project = [&](ggml_context * ctx, ggml_tensor * x) {
        return cp_projected_ ? ggml_add(ctx, ggml_mul_mat(ctx, m.tensor("cp.in_proj.weight"), x), m.tensor("cp.in_proj.bias")) : x;
    };
    return cp_.chain(
        2,
        [&](Graph & g, int64_t, int64_t) {
            ggml_context * ctx = g.ctx();
            chain_code0_ = g.input(std::vector<int32_t>{code0}, 1);
            chain_hidden_ = g.input(talker_.hidden(), hidden(), 1);
            return project(ctx, ggml_concat(ctx, chain_hidden_, ggml_get_rows(ctx, m.tensor("talker.codec_embd"), chain_code0_), 1));
        },
        n_groups_ - 1, [&](int s) { return m.tensor("cp.head." + std::to_string(s)); },
        [&](Graph & g, int s, ggml_tensor * id) {
            return project(g.ctx(), ggml_get_rows(g.ctx(), m.tensor("cp.codec_embd." + std::to_string(s)), id));
        },
        [&](Graph & g) {
            g.set(chain_code0_, std::vector<int32_t>{code0});
            g.set(chain_hidden_, talker_.hidden());
        });
}
