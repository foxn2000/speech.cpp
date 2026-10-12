#include "synthesizer.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>

#include "error.h"
#include "layout.h"
#include "text-normalizer.h"

namespace irodori {

namespace {

/** Adds the seconds from its construction to its destruction to `total`. */
struct Timer {
    double & total;
    std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
    ~Timer() { total += std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count(); }
};

bool starts_with_riff(const std::string & path) {
    std::ifstream f(std::filesystem::u8path(path), std::ios::binary);
    if (!f) throw Error(Fault::Io, "cannot open " + path + "; check the path and that the file can be read");
    char magic[4] = {};
    f.read(magic, 4);
    return std::memcmp(magic, "RIFF", 4) == 0;
}

}  // namespace

TailCut::TailCut(const ModelFile & m)
    : window((int) m.u32("irodori-tts.tail.window")),
      std_threshold(m.f32("irodori-tts.tail.std_threshold")),
      mean_threshold(m.f32("irodori-tts.tail.mean_threshold")) {}

void TailCut::check(const TailCut & asked) const {
    if (!asked.keep) return;
    const char * idle = asked.window != window                 ? "tail_window_size"
                        : asked.std_threshold != std_threshold   ? "tail_std_threshold"
                        : asked.mean_threshold != mean_threshold ? "tail_mean_threshold"
                                                                 : nullptr;
    if (idle) throw Error(Fault::InvalidArgument, std::string("keep_tail leaves the tail uncut, so ") + idle + " has no effect; leave out one of them", idle);
}

int TailCut::flattening_point(const std::vector<float> & latent, int frames, int latent_dim) const {
    for (int i = 0; i < frames; i++) {
        double sum = 0, squares = 0;
        for (int f = i; f < i + window; f++) {
            for (int d = 0; d < latent_dim; d++) {
                const double v = f < frames ? latent[(size_t) f * latent_dim + d] : 0.0;
                sum += v;
                squares += v * v;
            }
        }
        const double n = (double) window * latent_dim, mean = sum / n;
        const double deviation = std::sqrt(std::max(0.0, squares / n - mean * mean));
        if (deviation < std_threshold && std::fabs(mean) < mean_threshold) return i;
    }
    return frames;
}

/*
 * The decoder's windows after the first run from 24 to 48 frames, sized as it measures its speed so that the next one
 * arrives while the listener still has 0.1 s of the audio sent, a fifth of the first window, for the error of an
 * estimate made from a window or two and the time the caller takes to queue the audio. On an Apple M5 (2026-10-07,
 * 0.7.1 through the worker on speech-bench's 20 sentences) the 48-frame second window came 0.288 s after the first
 * window's 0.48 s, and on an RTX 2080 0.32 s, so a machine half as fast as the M5 ran dry between them.
 *
 * Each window decodes 10 frames on either side of its own (Codec::kDecoderMargin): 12 frames decode 32, 24 decode 44
 * and 48 decode 68. A floor of 24 halves the wait for a window against 48 for 29% more decoding per second of audio,
 * where 12 would cost 88% more, which a machine that decodes 48-frame windows barely faster than real time could not
 * afford. The ceiling is 48, 0.7.1's window, so that a cancel and the silence between a worker's chunks never wait
 * longer than they did.
 *
 * The sizes may follow the clock because the audio does not depend on them: the decoder is convolutions without a
 * cache, and a margin of 10 frames covers its receptive field of 7.7, so decoding in windows of any sizes gives the
 * samples of decoding at once, bit for bit on the CPU and on Metal (irodori-codec-check), and a seed repeats its
 * samples. Qwen3-TTS's chunked codec differs by up to 7e-4 between chunkings and keeps a fixed schedule.
 */
WindowRule Synthesizer::window_rule(int hop, int sample_rate) {
    return {12, 24, 48, Codec::kDecoderMargin, (double) hop / sample_rate, 0.1};
}

Synthesizer::Synthesizer(const std::string & model_path, ggml_backend_t backend)
    : backend_(backend),
      model_(std::make_unique<ModelFile>(model_path, backend, model_layout)),
      codec_(*model_, backend),
      tokenizer_(*model_),
      text_(*model_),
      speaker_(*model_),
      duration_(*model_),
      dit_(*model_),
      sampler_(dit_, *model_, backend),
      reference_(*model_),
      tail_(*model_) {
    allocr_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend_));
}

Synthesizer::~Synthesizer() {
    if (allocr_) ggml_gallocr_free(allocr_);
}

Voice Synthesizer::voice_from_latent(std::vector<float> latent) {
    Voice v;
    v.frames = (int) (latent.size() / codec_.latent_dim());
    v.latent = std::move(latent);
    Graph g;
    ggml_tensor * state = speaker_.build(g, v.latent);
    g.output(state);
    g.compute(backend_, allocr_);
    v.speaker = Graph::read(state);
    v.speaker_tokens = (int) state->ne[1];
    return v;
}

Voice Synthesizer::load_voice(const std::string & path) {
    if (starts_with_riff(path)) {
        return voice_from_latent(encode_reference(codec_, path, reference_).latent);
    }
    const ModelFile file(path, backend_, voice_layout(*model_));
    const bool embedding = file.str("irodori-tts-voice.source") == "embedding";
    ggml_tensor * t = file.tensor(embedding ? "speaker" : "latent");
    std::vector<float> values(ggml_nelements(t));
    ggml_backend_tensor_get(t, values.data(), 0, ggml_nbytes(t));
    if (!embedding) return voice_from_latent(std::move(values));
    // The runtime attends to an embedding as it is, in place of the speaker condition a reference would give.
    Voice v;
    v.speaker_tokens = (int) t->ne[1];
    v.speaker = std::move(values);
    return v;
}

size_t Synthesizer::synthesize(const Request & r, const Voice & voice, const AudioSink & sink, Stats * stats) {
    if (r.guidance && dit_.meanflow()) throw std::logic_error("a MeanFlow model takes no guidance, and the options of its file offer none");
    const bool reference = voice.speaker_tokens > 0;
    if (!reference && !has_null_speaker()) throw std::logic_error("a voice without a reference reached a file without the null speaker");
    const std::string caption = strip_caption(r.caption);
    if (!caption.empty() && !has_caption()) throw std::logic_error("a caption reached a file without the caption's encoder");
    const TailCut & tail = r.tail ? *r.tail : tail_;
    const int steps = r.steps > 0 ? r.steps : sampler_.default_steps();
    duration_.check(r.length);
    const Guidance guidance = sampler_.check(r.guidance ? *r.guidance : sampler_.guidance(), steps, reference, !caption.empty());
    tail_.check(tail);
    Stats local;
    Stats & st = stats ? *stats : local;
    const auto start = std::chrono::steady_clock::now();
    std::vector<float> text_state, caption_state;
    float predicted = 0;
    int tokens = 0, caption_tokens = 0;
    {
        Timer t{st.text};
        const std::string text = normalize_text(r.text);
        if (text.empty()) throw Error(Fault::InvalidArgument, "the text is empty after normalization; give a text to speak", "text");
        const std::vector<int32_t> ids = tokenizer_.encode(text);
        tokens = (int) ids.size();
        if (tokens > text_.max_tokens()) {
            throw Error(Fault::OutOfRange, "the text is " + std::to_string(tokens) + " tokens long and Irodori-TTS takes at most " +
                                               std::to_string(text_.max_tokens()) + "; split it into sentences",
                        "text");
        }
        // The caption is tokenized as it is, after <s>, as the runtime's caption tokenizer takes it.
        std::vector<int32_t> caption_ids;
        if (!caption.empty()) {
            caption_ids = tokenizer_.encode(caption);
            caption_tokens = (int) caption_ids.size();
            const int most = (int) model_->u32("irodori-tts.caption.max_tokens");
            if (caption_tokens > most) {
                throw Error(Fault::OutOfRange, "the instructions are " + std::to_string(caption_tokens) + " tokens long and Irodori-TTS takes at most " +
                                                   std::to_string(most) + "; shorten them",
                            "instructions");
            }
        }
        Graph g;
        ggml_tensor * state = text_.build(g, ids);
        g.output(state);
        ggml_tensor * caption_condition = caption.empty() ? nullptr : text_.build(g, caption_ids, Condition::Caption);
        if (caption_condition) g.output(caption_condition);
        // The runtime runs the duration predictor only when no length is fixed.
        ggml_tensor * sum = nullptr;
        if (!r.length.fixed()) {
            // The speaker's summary is its condition's first token, or the learned null speaker without a reference.
            ggml_tensor * summary = reference ? g.input(std::vector<float>(voice.speaker.begin(), voice.speaker.begin() + speaker_.dim()), speaker_.dim())
                                              : model_->tensor("duration.null_speaker");
            sum = duration_.build(g, state, summary, caption_condition);
            g.output(sum);
        }
        g.compute(backend_, allocr_);
        text_state = Graph::read(state);
        if (caption_condition) caption_state = Graph::read(caption_condition);
        if (sum) predicted = Graph::read(sum)[0];
    }
    const Length length = duration_.length(r.length, predicted);
    const int frames = length.frames;
    std::vector<float> x;
    {
        Timer t{st.sampling};
        Conditions c{text_state, tokens, voice.speaker, voice.speaker_tokens, {}, caption_state, caption_tokens};
        const size_t n = (size_t) frames * codec_.latent_dim(), m = guidance.speaker_noise ? voice.speaker.size() : 0;
        std::vector<float> noise = r.noise, speaker_draw = r.speaker_noise;
        if (noise.empty()) {
            // The runtime draws the speaker's noise after the latent's from the same generator.
            noise = gaussian_noise(r.seed, n + m);
            speaker_draw.assign(noise.begin() + (std::ptrdiff_t) n, noise.end());
            noise.resize(n);
        }
        if (noise.size() != n || speaker_draw.size() != m) {
            throw std::runtime_error("the given noise has " + std::to_string(noise.size() / codec_.latent_dim()) + " frames and the speech " +
                                     std::to_string(frames) + ", or its speaker noise " + std::to_string(speaker_draw.size()) + " values where " +
                                     std::to_string(m) + " are wanted");
        }
        if (guidance.speaker_noise) c.speaker_noise = speaker_noise(speaker_draw, voice.speaker);
        x = sampler_.sample(c, std::move(noise), frames, steps, guidance, r.progress);
    }
    if (x.empty()) return 0;
    int64_t samples = length.samples;
    if (!tail.keep) {
        const int flat = tail.flattening_point(x, frames, codec_.latent_dim());
        if (flat > 0) samples = std::min(samples, (int64_t) flat * codec_.hop());
    }

    size_t emitted = 0;
    {
        Timer t{st.codec};
        const WindowRule rule = window_rule(codec_.hop(), codec_.sample_rate());
        codec_.decode(x, samples, [&](const DecodeProgress & p) { return next_window(rule, p); }, [&](const float * s, size_t n) {
            if (emitted == 0) st.first_audio = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
            emitted += n;
            return sink(s, n);
        });
    }
    st.tokens = tokens;
    st.frames = frames;
    st.samples = emitted;
    if (std::getenv("SPEECH_TTS_STATS")) {
        std::fprintf(stderr, "irodori stats: frames %d steps %d text %.4f sampling %.4f codec %.4f first_audio %.4f samples %zu\n", frames, steps,
                     st.text, st.sampling, st.codec, st.first_audio, emitted);
    }
    return emitted;
}

}  // namespace irodori
