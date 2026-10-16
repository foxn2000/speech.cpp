// Checks the Qwen3 decoder's runs in blocks against a run of the same rows in one graph, on the talker of a Qwen3-TTS
// model file. The rows are the talker's text embeddings of a text, repeated whole until the rows of the prefill fill
// more than two blocks, so that a block follows a whole one. All but the last 16 go in one run, once in a single
// block, as the decoder computed every run before it had blocks, and once in blocks of the given rows. The keys and
// values the two wrote into their caches are compared position by position in every layer, where a wrong position or
// mask shows at the rows it touches; then the last 16 rows go one at a time, reading those caches, and their logits
// and hidden states are compared. It prints the largest relative error of a row of each, whether the logits' argmax
// agrees, and the time and the device memory of each prefill, and fails when the two part further than the order of
// summation explains.
//
// It takes an F32 file. With Q8_0 weights the CPU's numbers for the same rows change with how a graph groups them: a
// prefill of 5,137 rows of the 0.6B talker in blocks and one in a single block each part from the F32 file's logits by
// 6e-2 and from each other by 5e-2 (2026-10-06).
//
// The two decoders attend as they choose on the device, or as the last argument forces them; the block rows, which
// default to the decoder's, may be left out before it.
//
// usage: qwen3-decoder-check <qwen3-tts model F32.gguf> <text file> [gpu|cpu] [block rows] [flash|products]

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <optional>
#include <string>
#include <vector>

#include "args.h"
#include "backend.h"
#include "qwen2-tokenizer.h"
#include "qwen3-decoder.h"
#include "qwen3-tts/layout.h"
#include "qwen3-tts/synthesizer.h"
#include "qwen3-tts/talker.h"
#include "weight-types.h"

namespace {

/** The rows that follow the prefill one at a time. */
constexpr int kSteps = 16;

/**
 * The largest relative error of a row that the order of summation explains on the CPU, and on a GPU. With 5,137 rows in
 * blocks of 512 against one block, the 0.6B and the 1.7B talker in F32 wrote the same caches and gave the same logits
 * on the CPU of an Apple M5; on its Metal, attending with the two products, whose matrix kernel rounds its inputs to
 * half precision, a cached row parted by 1.9e-2 at most, growing through the middle layers from 0 in the first two, and
 * the logits by 2.0e-3. Hiding from each block the position before it moved a cached row by 5.1e-1 or more on both, and
 * counting the positions after the first block from one too many by 3.1e-1 or more (2026-10-06). With flash attention
 * on that Metal, which runs a graph of many rows on the tensor API's kernel and a block of fewer on its simdgroup kernel,
 * 1,064 rows of the 0.6B talker in blocks of 512 wrote cached rows 1.9e-3 at most from one block's and gave logits
 * 1.3e-3 from them (2026-10-17).
 */
constexpr double kCpuTolerance = 1e-4;
constexpr double kGpuTolerance = 1e-1;

double relative_error(const float * got, const float * want, size_t n) {
    double num = 0, den = 0;
    for (size_t i = 0; i < n; i++) {
        num += (double) (got[i] - want[i]) * (got[i] - want[i]);
        den += (double) want[i] * want[i];
    }
    return std::sqrt(num / std::max(den, 1e-30));
}

/** The largest relative error of a row, and where it is. */
struct Worst {
    double error = 0;
    int layer = 0;
    int64_t position = 0;

    void add(double e, int l, int64_t p) {
        if (e > error) *this = {e, l, p};
    }
};

}  // namespace

int main(int argc, char ** argv) {
    const std::vector<std::string> args = utf8_args(argc, argv);
    const bool forced = args.size() > 4 && (args.back() == "flash" || args.back() == "products");
    const size_t positional = args.size() - (forced ? 1 : 0);
    if (args.size() < 3 || positional > 5) {
        std::fprintf(stderr, "usage: %s <qwen3-tts model F32.gguf> <text file> [gpu|cpu] [block rows] [flash|products]\n", args[0].c_str());
        return 2;
    }
    if (!takes_weights(args[1], qwen3_tts_layout, {"F32"},
                       "with quantized weights a row's numbers change with how a graph groups the rows")) {
        return 2;
    }
    ggml_backend_t backend = init_backend(args.size() > 3 ? args[3] : "");
    const int64_t block_rows = positional > 4 ? std::stoll(args[4]) : kQwen3BlockRows;
    std::optional<Qwen3Attention> attention;
    if (forced) attention = args.back() == "flash" ? Qwen3Attention::Flash : Qwen3Attention::Products;
    const ModelFile model(args[1], backend, qwen3_tts_layout);
    Talker talker(model, backend);
    std::ifstream f(std::filesystem::u8path(args[2]));
    const std::string text((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    const std::vector<int32_t> text_ids = Qwen2Tokenizer(model, kTextTokenizer).encode(text);
    if (text_ids.empty()) {
        std::fprintf(stderr, "%s holds no text\n", args[2].c_str());
        return 2;
    }
    std::vector<int32_t> ids;
    while ((int64_t) ids.size() - kSteps <= 2 * block_rows) ids.insert(ids.end(), text_ids.begin(), text_ids.end());
    const int64_t n = (int64_t) ids.size(), prefill = n - kSteps;
    const std::vector<float> embeds = talker.text_embeddings(ids);

    const Qwen3Shape shape = read_qwen3_shape(model, "qwen3-tts.talker");
    const int64_t h = shape.hidden, kv_dim = (int64_t) shape.n_kv_head * shape.head_dim;
    const auto rows = [&](int64_t first) {
        return [&embeds, h, first](Graph & g, int64_t from, int64_t count) {
            const auto begin = embeds.begin() + (first + from) * h;
            return g.input(std::vector<float>(begin, begin + count * h), h, count);
        };
    };
    ggml_tensor * head = model.tensor("talker.codec_head");
    const int64_t max_positions = model.u32("qwen3-tts.talker.max_position_embeddings");
    Qwen3Decoder whole(model, backend, "talker", shape, max_positions, GGML_TYPE_F16, Qwen3Readback::LogitsAndHidden, prefill, attention);
    Qwen3Decoder blocks(model, backend, "talker", shape, max_positions, GGML_TYPE_F16, Qwen3Readback::LogitsAndHidden, block_rows,
                        attention);
    std::printf("backend: %s, %s\n", ggml_backend_name(backend), qwen3_attention_name(blocks.attention()));

    double seconds[2];
    Qwen3Decoder * decoders[2] = {&whole, &blocks};
    for (int d = 0; d < 2; d++) {
        decoders[d]->start(n, prefill);
        const auto start = std::chrono::steady_clock::now();
        decoders[d]->run(prefill, rows(0), head);
        seconds[d] = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    }

    Worst keys, values;
    std::vector<float> k_whole, v_whole, k_blocks, v_blocks;
    for (int l = 0; l < shape.n_layer; l++) {
        whole.read_cache(l, k_whole, v_whole);
        blocks.read_cache(l, k_blocks, v_blocks);
        for (int64_t p = 0; p < prefill; p++) {
            keys.add(relative_error(&k_blocks[p * kv_dim], &k_whole[p * kv_dim], kv_dim), l, p);
            values.add(relative_error(&v_blocks[p * kv_dim], &v_whole[p * kv_dim], kv_dim), l, p);
        }
    }

    Worst logits, hidden;
    int argmax_differs = 0;
    const auto argmax = [](const std::vector<float> & v) { return std::max_element(v.begin(), v.end()) - v.begin(); };
    for (int64_t i = prefill - 1; i < n; i++) {
        if (i >= prefill) {
            whole.run(1, rows(i), head);
            blocks.run(1, rows(i), head);
        }
        logits.add(relative_error(blocks.logits().data(), whole.logits().data(), whole.logits().size()), 0, i);
        hidden.add(relative_error(blocks.hidden().data(), whole.hidden().data(), whole.hidden().size()), 0, i);
        if (argmax(blocks.logits()) != argmax(whole.logits())) argmax_differs++;
    }

    std::printf("%lld rows prefilled, then %d one at a time\n", (long long) prefill, kSteps);
    std::printf("one block:      %7.2f s, graphs %8.1f MiB\n", seconds[0], whole.graph_bytes() / 1048576.0);
    std::printf("blocks of %-4lld %7.2f s, graphs %8.1f MiB\n", (long long) block_rows, seconds[1], blocks.graph_bytes() / 1048576.0);
    std::printf("cached keys   worst relative error of a row %.2e (layer %d, position %lld)\n", keys.error, keys.layer, (long long) keys.position);
    std::printf("cached values worst relative error of a row %.2e (layer %d, position %lld)\n", values.error, values.layer,
                (long long) values.position);
    std::printf("logits        worst relative error %.2e, hidden state %.2e, argmax differs %d of %d\n", logits.error, hidden.error,
                argmax_differs, kSteps + 1);
    const bool cpu = ggml_backend_dev_type(ggml_backend_get_device(backend)) == GGML_BACKEND_DEVICE_TYPE_CPU;
    const bool ok = std::max({keys.error, values.error, logits.error, hidden.error}) <= (cpu ? kCpuTolerance : kGpuTolerance) &&
                    argmax_differs == 0;
    ggml_backend_free(backend);
    return ok ? 0 : 1;
}
