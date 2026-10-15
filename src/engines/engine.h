#pragma once

#include <cmath>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "ggml-backend.h"
#include "model-file.h"
#include "speech.h"

/** A value of an option, the alternative of its speech_type: a string, an integer, a number or a boolean. */
using OptionValue = std::variant<std::string, int64_t, double, bool>;

/** The largest seed, 2^53 - 1: the integers a JSON reader in JavaScript holds exactly. */
constexpr int64_t kMaxSeed = (int64_t(1) << 53) - 1;

/**
 * One option a family takes, as its table declares it: whether a request must set it, whether its value steers the
 * model or is only checked, its default, the range of a number, its minimum excluded where `minimum_exclusive` says
 * so, and the values a string option takes. The values of voice and language are the model's voices and languages,
 * which the information holds, and a string option without choices, such as prompt, takes any text.
 */
struct OptionSpec {
    speech_option option;
    bool required = false;
    bool steers = true;
    std::optional<OptionValue> default_value = std::nullopt;
    double minimum = -INFINITY, maximum = INFINITY;
    bool minimum_exclusive = false;
    /** The values of a string option other than voice and language, compared with case. */
    std::vector<std::string> choices = {};
};

/**
 * A part of a voice: a reference recording in a WAVE file, with the words it speaks where they were given, or a voice
 * file of the family whose references it takes as they are; and the speaker it is given, where one was.
 */
struct VoicePart {
    bool voice_file = false;
    std::string path;
    std::optional<std::string> transcript;
    std::optional<int> speaker;
};

/**
 * What a voice file is made of: its parts, in the order given, and the loudness each recording is brought to before it
 * is encoded, the model's unless `lufs` gives another, or the recording's own without `normalize`; or, instead of
 * parts, a speaker-inversion embedding of `embedding_tokens` vectors, row by row.
 */
struct VoiceRecipe {
    std::vector<VoicePart> parts;
    bool normalize = true;
    std::optional<double> lufs;
    std::vector<float> embedding;
    size_t embedding_tokens = 0;
};

/** A voice as the model information shows it; what the model file does not say is empty. */
struct VoiceInfo {
    std::string name, language, gender, description;
};

/**
 * What a model file lacks that its family takes with files that have it: an option, or one voice, with the message a
 * request for it is refused with, which names what the file lacks and how to get a file that has it.
 */
struct Lack {
    speech_option option;
    /** The voice the file lacks, for SPEECH_OPT_VOICE; empty for an option the file does not take. */
    std::string voice;
    std::string message;
};

/**
 * What a family says of a model from its file's metadata, besides what every model file says: what the information
 * shows of it, the table of the options it takes in the order of the vocabulary, and how a synthesis counts a text's
 * tokens.
 */
struct FamilyInfo {
    bool incremental = false;
    std::vector<VoiceInfo> voices;
    /** The hash a voice file must carry, for a family that takes voice files. */
    std::string voice_codec;
    /** Whether add_voice() takes a recording in a WAVE file, which it encodes, besides a voice file. */
    bool voice_recordings = false;
    size_t max_text_tokens = 0;
    std::vector<OptionSpec> options;
    /** What the file lacks of what the family takes, which a request is refused with by its own message. */
    std::vector<Lack> lacks;
    /** The tokens of a text as a synthesis counts them against max_text_tokens; empty for a recognition model. */
    std::function<size_t(const std::string & text)> count_tokens;
};

/**
 * An object made from a model file's metadata when it is first asked for, from any thread: a tokenizer, which the
 * information needs only to count a text's tokens.
 */
template <typename T>
class Lazy {
public:
    explicit Lazy(std::shared_ptr<const ModelFile> file) : file_(std::move(file)) {}

    const T & get() const {
        std::call_once(once_, [&] { value_ = std::make_unique<T>(*file_); });
        return *value_;
    }

private:
    std::shared_ptr<const ModelFile> file_;
    mutable std::once_flag once_;
    mutable std::unique_ptr<T> value_;
};

/**
 * The options of a request that runs: each one it set, checked against the family's table as it was set, the table's
 * default of each one it did not set that has one, and the seed the C API drew where the request set none.
 */
class RequestValues {
public:
    /** Sets an option's value, `given` where the request set it rather than the table's default or a drawn seed. */
    void set(speech_option option, OptionValue value, bool given) {
        values_[option] = std::move(value);
        if (given) given_.insert(option);
    }
    bool has(speech_option option) const { return values_.count(option) > 0; }
    /** Whether the request set the option, for a rule that refuses a value the rest of the request leaves unused. */
    bool given(speech_option option) const { return given_.count(option) > 0; }
    const std::string & string(speech_option option) const { return std::get<std::string>(at(option)); }
    int64_t integer(speech_option option) const { return std::get<int64_t>(at(option)); }
    double number(speech_option option) const { return std::get<double>(at(option)); }
    bool boolean(speech_option option) const { return std::get<bool>(at(option)); }

private:
    const OptionValue & at(speech_option option) const {
        const auto it = values_.find(option);
        if (it == values_.end()) throw std::logic_error(std::string("a request has no value of ") + speech_option_name(option));
        return it->second;
    }

    std::map<speech_option, OptionValue> values_;
    std::set<speech_option> given_;
};

/** What a request that runs passes to its caller and asks of it. */
class Run {
public:
    virtual ~Run() = default;
    /** Passes audio on, at least one sample; false once the request is to stop. */
    virtual bool audio(const float * samples, size_t n) = 0;
    /** Says how far work that passes no audio has come, from 0 to 1; false once the request is to stop. */
    virtual bool progress(double done) = 0;
    /** Whether the request is to stop. */
    virtual bool stopped() = 0;
};

/** A token or a segment of a recognition: its start and end in seconds from the start of the audio, and its text. */
struct TimedText {
    double start, end;
    std::string text;
};

/**
 * What a recognition found: the text, its segments and tokens when the request set timestamps, why it ended: complete,
 * or at the model's limit of what it writes, with the text written up to it; and the languages the model wrote, as tags
 * of general.languages in the order of the audio, none for a family that writes none.
 */
struct Recognized {
    std::string text;
    std::vector<TimedText> segments, tokens;
    speech_stop stop = SPEECH_STOP_COMPLETE;
    std::vector<std::string> languages;
};

/**
 * A detection of mono audio at the model's rate that arrives a piece at a time, with a request's options: the regions
 * given so far, in seconds from the start and in order, each with an empty text, which never change once given, and,
 * once the audio has ended, are those detect() gives for the whole audio.
 */
class DetectionStream {
public:
    virtual ~DetectionStream() = default;

    /** Takes the next `n` samples. */
    virtual void push(const float * samples, size_t n) = 0;

    /** Ends the audio and gives the regions left. */
    virtual void end() = 0;

    virtual const std::vector<TimedText> & regions() const = 0;

    /** A region begun and not given yet: its start in seconds, and whether it is certain to be given. */
    struct Begun {
        double start;
        bool kept;
    };

    /**
     * The earliest region begun but not given yet: one that has ended whose end waits on what follows, or else the one
     * under way at the end of the audio heard, kept once it can no longer be dropped as too short; none when there is
     * neither.
     */
    virtual std::optional<Begun> open() const = 0;
};

/**
 * One loaded model of a family behind the C API, which has checked each value of a request against the family's table
 * and the request as a whole for what the table shows. A synthesis family overrides speak(), a recognition family
 * transcribe() and a detection family detect(); a family that takes voice files overrides add_voice().
 */
class Engine {
public:
    virtual ~Engine() = default;

    /**
     * Speaks `text`, passing its audio to `run`, after checking what the family's rules ask of the request as a whole.
     * Returns why the speech ended, unless `run` stopped it: complete, at max_seconds or at the model's limit.
     */
    virtual speech_stop speak(const std::string & text, const RequestValues & values, Run & run);

    /**
     * Recognizes mono samples at the model's sample rate, telling `run` how far it has come, after checking what the
     * family's rules ask of the request as a whole.
     */
    virtual Recognized transcribe(const std::vector<float> & samples, const RequestValues & values, Run & run);

    /**
     * The regions of mono samples at the model's sample rate where someone speaks, in seconds from the start and in
     * order, each with an empty text, telling `run` how far it has come, after checking what the family's rules ask of
     * the request as a whole.
     */
    virtual std::vector<TimedText> detect(const std::vector<float> & samples, const RequestValues & values, Run & run);

    /**
     * A detection of audio that arrives a piece at a time, with the request's options, after checking what the family's
     * rules ask of them. It computes on the engine, which runs one of its calls at a time, and lives no longer.
     */
    virtual std::unique_ptr<DetectionStream> start_detection(const RequestValues & values);

    /**
     * Adds a voice from a voice file or a WAVE file under a name the C API has checked is new, and returns what the
     * model information says of it, or "" for nothing.
     */
    virtual std::string add_voice(const std::string & name, const std::string & path);

    /** Runs a short request of the model's task, so that a GPU compiles its kernels before the first request. */
    virtual void warm_up() = 0;

    /** Whether some of the graphs of a model on a GPU compute on the CPU, with the threads the model was loaded with. */
    virtual bool computes_on_cpu() const { return false; }
};

FamilyInfo describe_qwen3_tts(const std::shared_ptr<const ModelFile> & file);
std::unique_ptr<Engine> load_qwen3_tts(const std::string & path, ggml_backend_t backend, int threads);

FamilyInfo describe_irodori_tts(const std::shared_ptr<const ModelFile> & file);
std::unique_ptr<Engine> load_irodori_tts(const std::string & path, ggml_backend_t backend, int threads);
void make_irodori_tts_voice(const std::string & model_path, const VoiceRecipe & recipe, const std::string & voice_path, ggml_backend_t backend);

FamilyInfo describe_drama3(const std::shared_ptr<const ModelFile> & file);
std::unique_ptr<Engine> load_drama3(const std::string & path, ggml_backend_t backend, int threads);
void make_drama3_voice(const std::string & model_path, const VoiceRecipe & recipe, const std::string & voice_path, ggml_backend_t backend);

FamilyInfo describe_fastconformer(const std::shared_ptr<const ModelFile> & file);
std::unique_ptr<Engine> load_fastconformer(const std::string & path, ggml_backend_t backend, int threads);

FamilyInfo describe_qwen3_asr(const std::shared_ptr<const ModelFile> & file);
std::unique_ptr<Engine> load_qwen3_asr(const std::string & path, ggml_backend_t backend, int threads);

FamilyInfo describe_silero_vad(const std::shared_ptr<const ModelFile> & file);
std::unique_ptr<Engine> load_silero_vad(const std::string & path, ggml_backend_t backend, int threads);
