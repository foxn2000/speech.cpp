#include "decoder.h"

#include <cstdlib>

#include <algorithm>
#include <stdexcept>
#include <string>

#include "graph.h"

namespace qwen3_asr {

namespace {

/** The nodes of an embedding graph at most. */
constexpr int kEmbeddingGraphSize = 8;

}  // namespace

Decoder::Decoder(const ModelFile & m, ggml_backend_t backend, ggml_type cache_type, std::optional<Qwen3Attention> attention)
    : m_(m),
      backend_(backend),
      stack_(m, backend, "dec", read_qwen3_shape(m, "qwen3-asr.decoder"), m.u32("qwen3-asr.decoder.max_position_embeddings"), cache_type,
             kQwen3BlockRows, attention),
      max_new_tokens_(m.u32("qwen3-asr.generation.max_new_tokens")),
      eos_(m.i32_array("qwen3-asr.generation.eos_ids")),
      allocr_(ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend))) {
    if (std::getenv("SPEECH_ASR_NO_HIDDEN")) stack_.set_reads_hidden(false);
    if (std::getenv("SPEECH_ASR_ARGMAX")) stack_.set_argmax(true);
}

Decoder::~Decoder() {
    if (allocr_) ggml_gallocr_free(allocr_);
}

std::vector<float> Decoder::embeddings(const PromptIds & prompt, const std::vector<float> & audio) {
    const int64_t h = hidden(), n = (int64_t) prompt.ids.size(), audio_tokens = (int64_t) audio.size() / h;
    if (prompt.audio_start + audio_tokens > n) throw std::logic_error("the audio does not fit the prompt's audio tokens");
    std::vector<int32_t> text;
    for (int64_t i = 0; i < n; i++) {
        if (i < prompt.audio_start || i >= prompt.audio_start + audio_tokens) text.push_back(prompt.ids[(size_t) i]);
    }
    Graph g(kEmbeddingGraphSize);
    ggml_tensor * rows = ggml_get_rows(g.ctx(), m_.tensor("dec.token_embd"), g.input(text, (int64_t) text.size()));
    g.output(rows);
    g.compute(backend_, allocr_);
    const std::vector<float> text_rows = Graph::read(rows);
    std::vector<float> out((size_t) (n * h));
    const auto before = text_rows.begin() + (std::ptrdiff_t) (prompt.audio_start * h);
    std::copy(text_rows.begin(), before, out.begin());
    std::copy(audio.begin(), audio.end(), out.begin() + (std::ptrdiff_t) (prompt.audio_start * h));
    std::copy(before, text_rows.end(), out.begin() + (std::ptrdiff_t) ((prompt.audio_start + audio_tokens) * h));
    return out;
}

bool Decoder::prefill(const std::vector<float> & embeds, int64_t n, const std::function<bool(int64_t rows)> & keep_going) {
    const int64_t h = hidden();
    stack_.start(positions(n), n);
    for (int64_t from = 0; from < n; from += kQwen3BlockRows) {
        const int64_t count = std::min(kQwen3BlockRows, n - from);
        stack_.run(
            count,
            [&](Graph & g, int64_t, int64_t rows) {
                const auto first = embeds.begin() + (std::ptrdiff_t) (from * h);
                return g.input(std::vector<float>(first, first + (std::ptrdiff_t) (rows * h)), h, rows);
            },
            from + count == n ? m_.tensor("dec.token_embd") : nullptr);
        if (!keep_going(from + count)) return false;
    }
    return true;
}

void Decoder::step(int32_t id) {
    stack_.step(id, m_.tensor("dec.token_embd"), m_.tensor("dec.token_embd"));
}

std::optional<Generation> Decoder::generate(const std::function<bool(size_t tokens)> & keep_going) {
    Generation out;
    for (;;) {
        static const bool argmax = std::getenv("SPEECH_ASR_ARGMAX") != nullptr;
        int32_t id;
        if (argmax) {
            id = stack_.best();
        } else {
            const std::vector<float> & l = logits();
            id = (int32_t) (std::max_element(l.begin(), l.end()) - l.begin());
        }
        out.ids.push_back(id);
        if (std::find(eos_.begin(), eos_.end(), id) != eos_.end()) return out;
        if ((int64_t) out.ids.size() == max_new_tokens_) {
            out.limited = true;
            return out;
        }
        if (!keep_going(out.ids.size())) return std::nullopt;
        step(id);
    }
}

}  // namespace qwen3_asr
