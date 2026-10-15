#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iterator>
#include <limits>
#include <map>
#include <string>
#include <vector>

#include "engine.h"
#include "error.h"
#include "irodori-tts/synthesizer.h"
#include "irodori-tts/text-normalizer.h"
#include "irodori-tts/voice-file.h"

namespace {

/** The voice of a file with the null speaker, which speaks without a reference as the runtime's no_ref. */
constexpr const char * kNoReference = "none";

/** The frames of the latent a warm-up speaks from, a second of silence, since a model may have no voice when it loads. */
constexpr int kWarmUpFrames = 25;

/**
 * The sampler's steps of a warm-up. Two run both of its graphs: RF guides its first step, at t 0.999, and not its
 * second, at 0.4995, below the irodori-tts.sampler.cfg_min_t of 0.5 that the RF file gives; MeanFlow has one graph.
 */
constexpr int kWarmUpSteps = 2;

class IrodoriTtsEngine : public Engine {
public:
    IrodoriTtsEngine(const std::string & path, ggml_backend_t backend) : synth_(path, backend) {
        if (synth_.has_null_speaker()) voices_.emplace(kNoReference, irodori::Voice{});
    }

    speech_stop speak(const std::string & text, const RequestValues & values, Run & run) override {
        irodori::Request r;
        r.text = text;
        if (values.has(SPEECH_OPT_INSTRUCTIONS)) r.caption = values.string(SPEECH_OPT_INSTRUCTIONS);
        r.seed = (uint64_t) values.integer(SPEECH_OPT_SEED);
        r.steps = (int) values.integer(SPEECH_OPT_STEPS);
        r.length.seconds = values.has(SPEECH_OPT_SECONDS) ? values.number(SPEECH_OPT_SECONDS) : 0;
        r.length.duration_scale = values.number(SPEECH_OPT_DURATION_SCALE);
        r.length.speed = values.number(SPEECH_OPT_SPEED);
        // An RF file alone declares the guidance, and with it a default of every option of it that has one.
        if (values.has(SPEECH_OPT_CFG_SCALE_TEXT)) r.guidance = guidance(values);
        r.tail.emplace();
        r.tail->keep = values.boolean(SPEECH_OPT_KEEP_TAIL);
        r.tail->window = (int) values.integer(SPEECH_OPT_TAIL_WINDOW_SIZE);
        r.tail->std_threshold = (float) values.number(SPEECH_OPT_TAIL_STD_THRESHOLD);
        r.tail->mean_threshold = (float) values.number(SPEECH_OPT_TAIL_MEAN_THRESHOLD);
        r.progress = [&](double done) { return run.progress(done); };
        synth_.synthesize(r, voices_.at(values.string(SPEECH_OPT_VOICE)), [&](const float * samples, size_t n) { return run.audio(samples, n); });
        return SPEECH_STOP_COMPLETE;
    }

    std::string add_voice(const std::string & name, const std::string & path) override {
        voices_.emplace(name, synth_.load_voice(path));
        return "";
    }

    void warm_up() override {
        // The first synthesis compiles the GPU kernels, which on Vulkan takes seconds; a short and a longer text run
        // through every stage and both sizes of the decoder's windows.
        const irodori::Voice silence = synth_.voice_from_latent(std::vector<float>((size_t) kWarmUpFrames * synth_.codec().latent_dim(), 0.0f));
        for (const char * text : {"あ。", "明日の東京は晴れで、最高気温は二十四度の予報です。"}) {
            irodori::Request warmup;
            warmup.text = text;
            warmup.steps = kWarmUpSteps;
            synth_.synthesize(warmup, silence, [](const float *, size_t) { return true; });
        }
    }

private:
    /** The RF sampler's settings of a request, which the scales, the noise's factor and Sway's coefficient give in float32 as the runtime's tensors take them. */
    static irodori::Guidance guidance(const RequestValues & values) {
        irodori::Guidance g;
        g.text = (float) values.number(SPEECH_OPT_CFG_SCALE_TEXT);
        g.speaker = (float) values.number(SPEECH_OPT_CFG_SCALE_SPEAKER);
        const std::string & mode = values.string(SPEECH_OPT_CFG_GUIDANCE_MODE);
        for (size_t i = 0; i < std::size(irodori::kGuidanceModes); i++) {
            if (mode == irodori::kGuidanceModes[i]) g.mode = (irodori::GuidanceMode) i;
        }
        g.min_t = values.number(SPEECH_OPT_CFG_MIN_T);
        g.max_t = values.number(SPEECH_OPT_CFG_MAX_T);
        g.speaker_noise = values.string(SPEECH_OPT_SPEAKER_UNCOND_MODE) == irodori::kSpeakerNoise;
        if (values.has(SPEECH_OPT_TRUNCATION_FACTOR)) g.truncation = (float) values.number(SPEECH_OPT_TRUNCATION_FACTOR);
        if (values.has(SPEECH_OPT_RESCALE_K)) g.rescale_k = values.number(SPEECH_OPT_RESCALE_K);
        if (values.has(SPEECH_OPT_RESCALE_SIGMA)) g.rescale_sigma = values.number(SPEECH_OPT_RESCALE_SIGMA);
        g.sway = (float) values.number(SPEECH_OPT_SWAY_COEFF);
        g.speaker_kv_scale = (float) values.number(SPEECH_OPT_SPEAKER_KV_SCALE);
        g.speaker_kv_min_t = (float) values.number(SPEECH_OPT_SPEAKER_KV_MIN_T);
        g.speaker_kv_layers = (int) values.integer(SPEECH_OPT_SPEAKER_KV_MAX_LAYERS);
        if (values.has(SPEECH_OPT_CFG_SCALE_INSTRUCTIONS)) g.caption = (float) values.number(SPEECH_OPT_CFG_SCALE_INSTRUCTIONS);
        return g;
    }

    irodori::Synthesizer synth_;
    std::map<std::string, irodori::Voice> voices_;
};

/**
 * The shortest decimal that reads back as `v` in float32, so that a default the model file holds in float32 shows as
 * the official code writes it (0.05, not 0.05000000074505806); the request rounds it back to `v`.
 */
double shortest(float v) {
    char text[32];
    for (int digits = 1;; digits++) {
        std::snprintf(text, sizeof text, "%.*g", digits, (double) v);
        if ((float) std::strtod(text, nullptr) == v) return std::strtod(text, nullptr);
    }
}

/**
 * The declaration of an option the family computes with as a float32, as the runtime's tensors take it: its range held
 * to the floats, from the smallest normal float where the option is above 0, since Metal's fast math may flush a
 * subnormal one to 0, to the largest, so that a value that would turn into 0 or infinity is out_of_range when the
 * request sets it rather than run.
 */
OptionSpec float32(OptionSpec s) {
    constexpr double largest = std::numeric_limits<float>::max(), smallest = std::numeric_limits<float>::min();
    s.minimum = std::max(s.minimum, -largest);
    s.maximum = std::min(s.maximum, largest);
    if (s.minimum == 0 && s.minimum_exclusive) {
        s.minimum = smallest;
        s.minimum_exclusive = false;
    }
    return s;
}

/** The tokens of a text as the synthesis counts them: those of its normalized form, <s> included. */
struct TokenCounter {
    explicit TokenCounter(const ModelFile & m) : tokenizer(m) {}
    size_t count(const std::string & text) const { return tokenizer.encode(irodori::normalize_text(text)).size(); }
    irodori::Tokenizer tokenizer;
};

}  // namespace

/**
 * The table of the options Irodori-TTS takes, with the bounds of the length and the speed, the sampler's steps and
 * guidance and the cut at the tail its file gives. Its one voice of its own is none, which speaks without a reference,
 * where the file holds the null speaker; a request otherwise speaks in a voice added since loading. Its one language
 * is checked and not used. It fixes the length before it makes the speech, so it takes seconds and a scale of the
 * predicted length rather than max_seconds. An RF model takes the official runtime's guidance, schedule and scaling of
 * the speaker, which a MeanFlow model folded into its training or ignores and does not take. A file with the caption's
 * encoder takes instructions, the runtime's caption, and an RF one the caption's scale.
 */
FamilyInfo describe_irodori_tts(const std::shared_ptr<const ModelFile> & file) {
    const ModelFile & m = *file;
    FamilyInfo info;
    if (m.boolean("irodori-tts.duration.null_speaker")) {
        info.voices.push_back({kNoReference, "", "", "speaks without a reference recording"});
    } else {
        info.lacks.push_back({SPEECH_OPT_VOICE, kNoReference,
                              m.path() + " has layout " + std::to_string(m.layout_version()) +
                                  ", which lacks the null speaker that the voice none speaks with without a reference; " + m.remedy()});
    }
    info.voice_codec = m.str("irodori-tts.codec.sha256");
    info.voice_recordings = true;
    info.max_text_tokens = m.u32("irodori-tts.text.max_tokens");
    info.options = {
        {SPEECH_OPT_VOICE, true, true},
        {SPEECH_OPT_LANGUAGE, false, false, std::string("auto")},
        {SPEECH_OPT_SEED, false, true, std::nullopt, 0, (double) kMaxSeed},
        {SPEECH_OPT_SPEED, false, true, 1.0, m.f32("irodori-tts.length.min_speed"), m.f32("irodori-tts.length.max_speed")},
        {SPEECH_OPT_SECONDS, false, true, std::nullopt, m.f32("irodori-tts.length.min_seconds"), m.f32("irodori-tts.length.max_seconds")},
        {SPEECH_OPT_DURATION_SCALE, false, true, 1.0, 0, INFINITY, true},
        {SPEECH_OPT_STEPS, false, true, (int64_t) m.u32("irodori-tts.sampler.default_steps"), 1, (double) INT_MAX},
    };
    const bool caption = m.boolean("irodori-tts.caption_condition");
    if (caption) {
        info.options.push_back({SPEECH_OPT_INSTRUCTIONS, false, true, std::string()});
    } else {
        info.lacks.push_back({SPEECH_OPT_INSTRUCTIONS, "",
                              m.path() + " has layout " + std::to_string(m.layout_version()) +
                                  ", which lacks the caption's encoder that instructions need; " + m.remedy()});
    }
    const bool rf = m.one_of("irodori-tts.flow", {"meanflow", "rf_velocity"}) == "rf_velocity";
    const std::string s = "irodori-tts.sampler.";
    if (rf) {
        const std::vector<std::string> modes(std::begin(irodori::kGuidanceModes), std::end(irodori::kGuidanceModes));
        info.options.insert(info.options.end(), {
            float32({SPEECH_OPT_CFG_SCALE_TEXT, false, true, (double) m.f32(s + "cfg_text"), 0, INFINITY}),
            float32({SPEECH_OPT_CFG_SCALE_SPEAKER, false, true, (double) m.f32(s + "cfg_speaker"), 0, INFINITY}),
            {SPEECH_OPT_CFG_GUIDANCE_MODE, false, true, modes[0], -INFINITY, INFINITY, false, modes},
            float32({SPEECH_OPT_CFG_MIN_T, false, true, (double) m.f32(s + "cfg_min_t"), 0, 1}),
            float32({SPEECH_OPT_CFG_MAX_T, false, true, (double) m.f32(s + "cfg_max_t"), 0, 1}),
            float32({SPEECH_OPT_TRUNCATION_FACTOR, false, true, std::nullopt, 0, INFINITY, true}),
            float32({SPEECH_OPT_RESCALE_K, false, true, std::nullopt, 0, INFINITY, true}),
            float32({SPEECH_OPT_RESCALE_SIGMA, false, true, std::nullopt, 0, INFINITY, true}),
            {SPEECH_OPT_SPEAKER_UNCOND_MODE, false, true, std::string(irodori::kSpeakerMasked), -INFINITY, INFINITY, false,
             {irodori::kSpeakerMasked, irodori::kSpeakerNoise}},
            float32({SPEECH_OPT_SWAY_COEFF, false, true, 0.0}),
        });
    }
    // A window as long as the longest speech reaches every frame of any speech.
    const int longest = (int) std::floor(m.f32("irodori-tts.length.max_seconds") * m.u32("speech.sample_rate") / (double) m.u32("irodori-tts.codec.hop_length"));
    info.options.insert(info.options.end(), {
        {SPEECH_OPT_KEEP_TAIL, false, true, false},
        {SPEECH_OPT_TAIL_WINDOW_SIZE, false, true, (int64_t) m.u32("irodori-tts.tail.window"), 1, (double) longest},
        float32({SPEECH_OPT_TAIL_STD_THRESHOLD, false, true, shortest(m.f32("irodori-tts.tail.std_threshold")), 0, INFINITY, true}),
        float32({SPEECH_OPT_TAIL_MEAN_THRESHOLD, false, true, shortest(m.f32("irodori-tts.tail.mean_threshold")), 0, INFINITY, true}),
    });
    if (rf) {
        const int64_t layers = m.u32("irodori-tts.dit.num_layers");
        info.options.insert(info.options.end(), {
            float32({SPEECH_OPT_SPEAKER_KV_SCALE, false, true, 1.0, 0, INFINITY, true}),
            float32({SPEECH_OPT_SPEAKER_KV_MIN_T, false, true, shortest(m.f32(s + "speaker_kv_min_t")), 0, 1}),
            {SPEECH_OPT_SPEAKER_KV_MAX_LAYERS, false, true, layers, 1, (double) layers},
        });
    }
    if (rf && caption) info.options.push_back(float32({SPEECH_OPT_CFG_SCALE_INSTRUCTIONS, false, true, (double) m.f32(s + "cfg_caption"), 0, INFINITY}));
    const auto counter = std::make_shared<Lazy<TokenCounter>>(file);
    info.count_tokens = [counter](const std::string & text) { return counter->get().count(text); };
    return info;
}

std::unique_ptr<Engine> load_irodori_tts(const std::string & path, ggml_backend_t backend, int) {
    return std::make_unique<IrodoriTtsEngine>(path, backend);
}

void make_irodori_tts_voice(const std::string & model_path, const VoiceRecipe & recipe, const std::string & voice_path, ggml_backend_t backend) {
    if (!recipe.embedding.empty()) {
        irodori::make_embedding_voice_file(model_path, recipe.embedding, (int) recipe.embedding_tokens, voice_path);
        return;
    }
    std::vector<std::string> references;
    for (const VoicePart & part : recipe.parts) {
        // Irodori-TTS hears a voice in its recordings alone, and has one speaker.
        if (part.voice_file || part.transcript || part.speaker) {
            throw Error(Fault::InvalidArgument,
                        std::string("Irodori-TTS makes a voice of WAVE files alone, without ") +
                            (part.voice_file ? "voice files" : part.transcript ? "transcripts" : "speakers") + "; give it the recordings",
                        "references");
        }
        references.push_back(part.path);
    }
    irodori::make_voice_file(model_path, references, {recipe.normalize, recipe.lufs}, voice_path, backend);
}
