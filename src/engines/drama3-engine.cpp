#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <memory>
#include <string>

#include "drama3/layout.h"
#include "drama3/synthesizer.h"
#include "drama3/voice-file.h"
#include "backend.h"
#include "engine.h"
#include "error.h"
#include "json.h"
#include "wav.h"

namespace {

/** The voice that speaks without a reference, every speaker of the text in a voice of the model's choosing. */
constexpr const char * kNoReference = "none";

/** The frames of each chunk a warm-up draws at most, enough to run every graph of a frame and a window of the codec. */
constexpr int64_t kWarmUpFrames = 4;

/**
 * The shortest decimal that reads back as `v` in float32, so that a default the model file holds in float32 shows as
 * the model card writes it (0.9, not 0.8999999761581421); the request rounds it back to `v`.
 */
double shortest(float v) {
    char text[32];
    for (int digits = 1;; digits++) {
        std::snprintf(text, sizeof text, "%.*g", digits, (double) v);
        if ((float) std::strtod(text, nullptr) == v) return std::strtod(text, nullptr);
    }
}

/** The seconds of `frames` frames of the model file `m`. */
double seconds_of(const ModelFile & m, int64_t frames) {
    return (double) frames * m.u32("drama3.codec.hop_length") / m.u32("speech.sample_rate");
}

class Drama3Engine : public Engine {
public:
    Drama3Engine(const std::string & path, ggml_backend_t backend)
        : model_(path, backend, drama3::model_layout,
                 [](const std::string & name) { return !drama3::g_experts || name.find("_exps") == std::string::npos; }),
          synth_(model_, backend) {
        voices_.emplace(kNoReference, drama3::Voice{});
    }

    speech_stop speak(const std::string & text, const RequestValues & values, Run & run) override {
        drama3::Request r;
        r.text = text;
        r.seed = (uint64_t) values.integer(SPEECH_OPT_SEED);
        r.sampling.temperature = (float) values.number(SPEECH_OPT_TEMPERATURE);
        // The sampler divides the logits by the temperature as a float, which must hold it.
        if (r.sampling.temperature == 0 || std::isinf(r.sampling.temperature)) {
            throw Error(Fault::OutOfRange, "the option temperature is " + json_number(values.number(SPEECH_OPT_TEMPERATURE)) +
                                               ", beyond what the sampler's float holds; give a value nearer 1",
                        "temperature");
        }
        r.sampling.top_k = (int) values.integer(SPEECH_OPT_TOP_K);
        r.sampling.top_p = (float) values.number(SPEECH_OPT_TOP_P);
        const double seconds = values.number(SPEECH_OPT_MAX_SECONDS);
        r.max_frames = (int64_t) std::floor(seconds * synth_.sample_rate()) / synth_.samples_per_frame();
        if (r.max_frames < 1) {
            throw Error(Fault::OutOfRange, "max_seconds is shorter than a frame of the model's speech; give at least " +
                                               std::to_string(seconds_of(model_, 1)) + " seconds",
                        "max_seconds");
        }
        r.stopped = [&] { return run.stopped(); };
        const drama3::Ending ending = synth_.synthesize(r, voices_.at(values.string(SPEECH_OPT_VOICE)),
                                                        [&](const float * samples, size_t n) { return run.audio(samples, n); });
        // A synthesis the sink stopped is cancelled, which the request reports whatever this returns.
        return ending == drama3::Ending::MaxFrames ? SPEECH_STOP_MAX_SECONDS : SPEECH_STOP_COMPLETE;
    }

    std::string add_voice(const std::string & name, const std::string & path) override {
        // A recording alone lacks the words Drama 3 reads beside it.
        if (starts_with_riff(path)) {
            throw Error(Fault::InvalidArgument,
                        path + " is a recording, and a Drama 3 voice needs its transcript too; make a voice file of it with speech voice and --transcript",
                        "path");
        }
        const std::vector<drama3::VoiceReference> references = drama3::read_voice_file(path, model_);
        voices_.emplace(name, drama3::voice_of(references));
        return drama3::describe_speakers(references);
    }

    void warm_up() override {
        // The first synthesis compiles the GPU kernels, which on Vulkan takes seconds: a prompt, a few frames and a
        // window of the codec run every graph.
        drama3::Request warmup;
        warmup.text = "こんにちは。";
        warmup.sampling = drama3::file_sampling(model_);
        warmup.max_frames = kWarmUpFrames;
        synth_.synthesize(warmup, voices_.at(kNoReference), [](const float *, size_t) { return true; });
    }

private:
    ModelFile model_;
    drama3::Synthesizer synth_;
    std::map<std::string, drama3::Voice> voices_;
};

/** The tokens of a text as the synthesis counts them against max_text_tokens: those the prompt gives it. */
struct TokenCounter {
    explicit TokenCounter(const ModelFile & m) : prompt(m) {}
    size_t count(const std::string & text) const { return prompt.tokenize(text).size(); }
    drama3::Prompt prompt;
};

}  // namespace

/**
 * The table of the options Drama 3 takes. Its one voice of its own is none, which speaks every speaker of the text in a
 * voice of the model's choosing; its languages are checked and not used, since the model reads the language from the
 * text, where the speakers and the cues are written too. It samples with the model card's temperature, top-k and top-p
 * by default, every draw of a frame alike, and max_seconds bounds each chunk of a long text, by default the frames
 * fish-speech's server gives a chunk, at most the most it speaks in one pass. A text may have as many tokens as one
 * prompt takes rows.
 */
FamilyInfo describe_drama3(const std::shared_ptr<const ModelFile> & file) {
    const ModelFile & m = *file;
    FamilyInfo info;
    info.incremental = true;
    info.voices.push_back({kNoReference, "", "", "speaks every speaker of the text in a voice of the model's choosing"});
    info.voice_codec = m.str("drama3.codec.sha256");
    info.max_text_tokens = m.size("drama3.generation.max_prompt_rows");
    const drama3::Sampling card = drama3::file_sampling(m);
    info.options = {
        {SPEECH_OPT_VOICE, true, true},
        {SPEECH_OPT_LANGUAGE, false, false, std::string("auto")},
        {SPEECH_OPT_SEED, false, true, std::nullopt, 0, (double) kMaxSeed},
        {SPEECH_OPT_MAX_SECONDS, false, true, seconds_of(m, m.size("drama3.generation.default_frames")), 0,
         seconds_of(m, m.size("drama3.generation.max_frames")), true},
        {SPEECH_OPT_TOP_K, false, true, (int64_t) card.top_k, 1, (double) m.size("drama3.audio_decoder.vocab_size") + 1},
        {SPEECH_OPT_TOP_P, false, true, shortest(card.top_p), 0, 1, true},
        {SPEECH_OPT_TEMPERATURE, false, true, shortest(card.temperature), 0, INFINITY, true},
    };
    const auto counter = std::make_shared<Lazy<TokenCounter>>(file);
    info.count_tokens = [counter](const std::string & text) { return counter->get().count(text); };
    return info;
}

std::unique_ptr<Engine> load_drama3(const std::string & path, ggml_backend_t backend, int threads) {
    if (std::getenv("DRAMA3_CPU_EXPERTS")) {
        drama3::g_cpu = start_device(find_device("cpu"), threads > 0 ? threads : default_threads());
        drama3::g_experts = new ModelFile(path, drama3::g_cpu, drama3::model_layout, [](const std::string & name) { return name.find("_exps") != std::string::npos; });
    }
    return std::make_unique<Drama3Engine>(path, backend);
}

void make_drama3_voice(const std::string & model_path, const VoiceRecipe & recipe, const std::string & voice_path, ggml_backend_t backend) {
    if (!recipe.embedding.empty() || !recipe.normalize || recipe.lufs) {
        throw Error(Fault::InvalidArgument, "Drama 3 takes its recordings as they are, without an embedding or a loudness; leave them out",
                    !recipe.embedding.empty() ? "embedding" : "loudness");
    }
    std::vector<drama3::VoiceSource> sources;
    for (const VoicePart & part : recipe.parts) sources.push_back({part.voice_file, part.path, part.transcript, part.speaker});
    drama3::make_voice_file(model_path, sources, voice_path, backend);
}
