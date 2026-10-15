#include "synthesizer.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>

#include <algorithm>
#include <deque>
#include <optional>

#include "chunking.h"
#include "error.h"
#include "sampling.h"

namespace drama3 {

namespace {

/** The frames of window `i` of a chunk's audio: 4, 8, 16, 32, 64 and then 128. */
int64_t window_frames(int i) {
    return int64_t(4) << std::min(i, 5);
}

}  // namespace

/** What a chunk spoke: its codes, [codebooks, frames] with the frame fastest, and how it ended. */
struct Synthesizer::Chunk {
    std::vector<int32_t> codes;
    int64_t frames = 0;
    bool ended = false, stopped = false;
};

Synthesizer::Synthesizer(const ModelFile & m, ggml_backend_t backend)
    : m_(m), prompt_(m), text_(m, backend), decoder_(m, backend), codec_(m, backend) {
    chunk_bytes_ = m.size("drama3.generation.chunk_bytes");
    chunk_speakers_ = m.size("drama3.generation.chunk_speakers");
    recent_turns_ = m.u32("drama3.generation.recent_turns");
    eos_ = (int32_t) m.u32("drama3.eos_token_id");
}

Synthesizer::Chunk Synthesizer::speak(const std::vector<int32_t> & rows, const Request & r, const FrameSampler & sampler, std::mt19937_64 & rng,
                                      const AudioSink & sink) {
    const int width = prompt_.width(), codebooks = width - 1, h = text_.hidden();
    const int64_t n = (int64_t) rows.size() / width;
    text_.start(n + r.max_frames);
    decoder_.start(n + r.max_frames);
    using clock = std::chrono::steady_clock;
    const auto since = [](clock::time_point t) { return std::chrono::duration<double, std::milli>(clock::now() - t).count(); };
    double t_prefill = 0, t_logits = 0, t_decoder = 0, t_embed = 0, t_run = 0, t_codec = 0;
    auto t0 = clock::now();
    const std::vector<float> hidden = text_.run(text_.embed(rows, n), n);
    // The audio decoder runs each prompt row with the frame after it, which the last row has yet to draw.
    decoder_.prefill(std::vector<float>(hidden.begin(), hidden.end() - h), rows, n - 1);
    t_prefill = since(t0);
    std::vector<float> last(hidden.end() - h, hidden.end());
    std::vector<int32_t> frame(rows.end() - width, rows.end()), mains;
    for (int64_t i = 0; i < n; i++) mains.push_back(rows[(size_t) (i * width)]);

    Chunk chunk;
    std::vector<std::vector<int32_t>> drawn;
    int64_t sent = 0;
    int windows = 0;
    codec_.start_decoding();
    const auto pass = [&](int64_t until) {
        if (until <= sent) return true;
        const int64_t count = until - sent;
        std::vector<int32_t> codes((size_t) (codebooks * count));
        for (int64_t f = 0; f < count; f++)
            for (int b = 0; b < codebooks; b++) codes[(size_t) (b * count + f)] = drawn[(size_t) (sent + f)][(size_t) b];
        const auto tc = clock::now();
        const std::vector<float> audio = codec_.decode_more(codes, count);
        t_codec += since(tc);
        if (std::getenv("DRAMA3_TIMING")) std::fprintf(stderr, "codec window of %lld frames: %.1f ms\n", (long long) count, since(tc));
        sent = until;
        windows++;
        return sink(audio.data(), audio.size());
    };
    for (int64_t step = 0; step < r.max_frames; step++) {
        // A cancel stops the chunk between frames, rather than at the next window of audio.
        if (r.stopped && r.stopped()) {
            chunk.stopped = true;
            return chunk;
        }
        t0 = clock::now();
        const int32_t main = sampler.main(text_.logits(last), mains, uniform(rng), uniform(rng));
        t_logits += since(t0);
        if (main == eos_) {
            chunk.ended = true;
            break;
        }
        t0 = clock::now();
        const std::vector<int32_t> codes = decoder_.step(last, frame, main, [&](int, const std::vector<float> & logits) {
            return sampler.code(logits, uniform(rng));
        });
        t_decoder += since(t0);
        frame.assign(1, main);
        frame.insert(frame.end(), codes.begin(), codes.end());
        mains.push_back(main);
        drawn.push_back(codes);
        if ((int64_t) drawn.size() - sent >= window_frames(windows) && !pass((int64_t) drawn.size())) {
            chunk.stopped = true;
            return chunk;
        }
        if (step + 1 < r.max_frames) {
            t0 = clock::now();
            const std::vector<float> e = text_.embed(frame, 1);
            t_embed += since(t0);
            t0 = clock::now();
            last = text_.run(e, 1);
            t_run += since(t0);
        }
    }
    if (!pass((int64_t) drawn.size())) chunk.stopped = true;
    chunk.frames = (int64_t) drawn.size();
    if (std::getenv("DRAMA3_TIMING")) {
        const double f = (double) std::max<int64_t>(1, chunk.frames);
        std::fprintf(stderr, "timing: %lld prompt rows, prefill %.1f ms; %lld frames, per frame: logits+main %.2f, decoder %.2f, embed %.2f, run %.2f, codec %.2f ms\n",
                     (long long) n, t_prefill, (long long) chunk.frames, t_logits / f, t_decoder / f, t_embed / f, t_run / f, t_codec / f);
    }
    chunk.codes.resize((size_t) (codebooks * chunk.frames));
    for (int64_t f = 0; f < chunk.frames; f++)
        for (int b = 0; b < codebooks; b++) chunk.codes[(size_t) (b * chunk.frames + f)] = drawn[(size_t) f][(size_t) b];
    return chunk;
}

Ending Synthesizer::synthesize(const Request & r, const Voice & voice, const AudioSink & sink) {
    if (r.max_frames < 1) throw std::invalid_argument("a synthesis takes at least a frame of each chunk");
    const size_t tokens = count_tokens(r.text), most = (size_t) m_.size("drama3.generation.max_prompt_rows");
    if (tokens > most) {
        throw Error(Fault::OutOfRange,
                    "the text takes " + std::to_string(tokens) + " tokens, more than the model's " + std::to_string(most) + "; split it into shorter texts",
                    "text");
    }
    std::vector<std::string> chunks = split_text(r.text, chunk_bytes_, chunk_speakers_);
    // A text of no words has no chunk, and the prompt refuses it as empty.
    if (chunks.empty()) chunks.push_back(r.text);
    for (const std::string & text : chunks) prompt_.rows(text, voice.references, {});

    const FrameSampler sampler(m_, r.sampling);
    std::mt19937_64 rng(r.seed);
    std::optional<Utterance> first;
    std::deque<Utterance> recent;
    Ending ending = Ending::Complete;
    for (const std::string & text : chunks) {
        // The turns before this chunk, the longest first: the first chunk and the recent ones, then fewer recent ones,
        // then none.
        std::vector<std::vector<Utterance>> histories;
        for (size_t from = 0; from <= recent.size(); from++) {
            std::vector<Utterance> h;
            if (first) h.push_back(*first);
            h.insert(h.end(), recent.begin() + (std::ptrdiff_t) from, recent.end());
            histories.push_back(h);
        }
        if (first) histories.push_back({});
        std::vector<int32_t> rows;
        for (size_t i = 0; i < histories.size() && rows.empty(); i++) {
            try {
                rows = prompt_.rows(text, voice.references, histories[i]);
            } catch (const Error & e) {
                if (e.fault() != Fault::OutOfRange || i + 1 == histories.size()) throw;
            }
        }
        const Chunk chunk = speak(rows, r, sampler, rng, sink);
        if (chunk.stopped) return Ending::Stopped;
        if (!chunk.ended) ending = Ending::MaxFrames;
        if (chunk.frames == 0) continue;
        Utterance spoken{text, chunk.codes, chunk.frames};
        if (!first) {
            first = spoken;
        } else {
            recent.push_back(spoken);
            if (recent.size() > recent_turns_) recent.pop_front();
        }
    }
    return ending;
}

}  // namespace drama3
