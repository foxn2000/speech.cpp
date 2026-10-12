#pragma once

#include <cstdlib>

#include "ggml-cpu.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "decoder.h"
#include "detokenizer.h"
#include "encoder.h"
#include "frontend.h"
#include "model-file.h"
#include "times.h"
#include "transducer.h"

namespace fastconformer {

/** What recognize() found: the decoding's tokens and their text. */
struct Transcript {
    Decoding decoding;
    std::string text;
};

/** A FastConformer checkpoint with a TDT or an RNN-T decoder, from audio to text on one backend. */
class Recognizer {
public:
    Recognizer(const std::string & path, ggml_backend_t backend);
    ~Recognizer();
    Recognizer(const Recognizer &) = delete;
    Recognizer & operator=(const Recognizer &) = delete;

    /** The names of the model's decodings, the default first (fastconformer::decodings()). */
    const std::vector<std::string> & decodings() const { return decoding_names_; }

    /** The text of mono samples at sample_rate(), decoded with the decoding of decodings() named `name`. */
    Transcript recognize(const std::vector<float> & samples, const std::string & name);
    /** The same with the default decoding. */
    Transcript recognize(const std::vector<float> & samples) { return recognize(samples, decodings().front()); }

    /** The encoder's output for `features` ([frames, mels] row-major) projected for the joint, [T, hidden] row-major. */
    std::vector<float> encode(const std::vector<float> & features, int64_t frames);

    /**
     * The tokens of encode()'s output with the frames they were emitted on, by the decoding of decodings() named
     * `name`, telling `progress` how far it has come.
     */
    Decoding decoding(const std::vector<float> & projected, const std::string & name, const DecodingProgress & progress = {}) const {
        static ggml_backend_t cpu = std::getenv("SPEECH_FC_CPU_DECODE") ? ggml_backend_cpu_init() : nullptr;
        return decoder(name).decode(projected, cpu ? cpu : backend_, progress);
    }
    /** The same with the default decoding. */
    Decoding decoding(const std::vector<float> & projected) const { return decoding(projected, decodings().front()); }

    /** The token ids of encode()'s output, by the default decoding. */
    std::vector<int32_t> decode(const std::vector<float> & projected) const { return decoding(projected).ids; }

    /**
     * The time in seconds of encoder frame `frame` as process_timestamp_outputs() computes it: the frame times the
     * window stride times the subsampling factor, in double precision and in that order. The window stride is the
     * hop length over the sample rate; NeMo's 0.01 s is 160 samples at 16 kHz, and the division gives the double of
     * the literal.
     */
    double seconds(int64_t frame) const;

    /**
     * The segments of a decoding at the frames its tokens were emitted on, cut where the file's
     * fastconformer.segment.separators end a word and wherever its fastconformer.segment.breaks stand.
     */
    std::vector<Segment> segments(const Decoding & decoding) const;

    int sample_rate() const { return frontend_.sample_rate(); }
    const ModelFile & model() const { return model_; }
    const Frontend & frontend() const { return frontend_; }
    const Encoder & encoder() const { return encoder_; }
    const PredictionNetwork & prediction() const { return prediction_; }
    const Joint & joint() const { return joint_; }
    /** The decoding of decodings() named `name`. */
    const Decoder & decoder(const std::string & name) const;
    const Detokenizer & detokenizer() const { return detokenizer_; }

private:
    ggml_backend_t backend_;
    ModelFile model_;
    Frontend frontend_;
    Encoder encoder_;
    PredictionNetwork prediction_;
    Joint joint_;
    std::vector<std::string> decoding_names_;
    /** The decoders of decoding_names_, in their order. */
    std::vector<std::unique_ptr<Decoder>> decoders_;
    Detokenizer detokenizer_;
    std::vector<std::string> separators_, breaks_;
    ggml_gallocr_t allocr_;
};

}  // namespace fastconformer
