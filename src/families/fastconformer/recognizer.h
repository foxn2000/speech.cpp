#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "decoder.h"
#include "detokenizer.h"
#include "encoder.h"
#include "nemo-frontend.h"
#include "ggml-cpu.h"
#include "model-file.h"
#include "times.h"
#include "transducer.h"

namespace fastconformer {

/** What recognize() found: the decoding's tokens and their text. */
struct Transcript {
    Decoding decoding;
    std::string text;
};

/**
 * Where a decoding computes: on the CPU, from a copy of the tensors it reads there, or on the device the model is on.
 * Each step of a decoding is a small graph whose round trip to a GPU costs more than computing it on the CPU.
 */
enum class DecodeOn { Cpu, Device };

/**
 * A FastConformer checkpoint with a TDT or an RNN-T decoder, from audio to text on one backend, and, for a model on
 * another device than the CPU, the tensors its decodings read on the CPU too.
 */
class Recognizer {
public:
    /**
     * Loads the model at `path` onto `backend`. When `backend` is not the CPU, the first decoding on the CPU copies the
     * tensors of decoding_tensor() from it onto a CPU backend of its own that computes with `threads` threads.
     */
    Recognizer(const std::string & path, ggml_backend_t backend, int threads);
    ~Recognizer();
    Recognizer(const Recognizer &) = delete;
    Recognizer & operator=(const Recognizer &) = delete;

    /** The names of the model's decodings, the default first (fastconformer::decodings()). */
    const std::vector<std::string> & decodings() const { return decoding_names_; }

    /** The text of mono samples at sample_rate(), decoded with the decoding of decodings() named `name` where `on` says. */
    Transcript recognize(const std::vector<float> & samples, const std::string & name, DecodeOn on = DecodeOn::Cpu);
    /** The same with the default decoding. */
    Transcript recognize(const std::vector<float> & samples) { return recognize(samples, decodings().front()); }

    /** The encoder's output for `features` ([frames, mels] row-major) projected for the joint, [T, hidden] row-major. */
    std::vector<float> encode(const std::vector<float> & features, int64_t frames);

    /**
     * The tokens of encode()'s output with the frames they were emitted on, by the decoding of decodings() named
     * `name`, computed where `on` says, telling `progress` how far it has come.
     */
    Decoding decoding(const std::vector<float> & projected, const std::string & name, DecodeOn on = DecodeOn::Cpu,
                      const DecodingProgress & progress = {});
    /** The same with the default decoding. */
    Decoding decoding(const std::vector<float> & projected) { return decoding(projected, decodings().front()); }

    /** The token ids of encode()'s output, by the default decoding. */
    std::vector<int32_t> decode(const std::vector<float> & projected) { return decoding(projected).ids; }

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
    const NemoFrontend & frontend() const { return frontend_; }
    const Encoder & encoder() const { return encoder_; }
    /** The prediction network and the joint on the model's device. */
    const PredictionNetwork & prediction() const { return *device_.prediction; }
    const Joint & joint() const { return device_.joint; }
    /** The decoding of decodings() named `name` on the model's device. */
    const Decoder & decoder(const std::string & name) const { return device_.decoder(name); }
    const Detokenizer & detokenizer() const { return detokenizer_; }
    /** Whether the model is on the CPU, where DecodeOn::Device decodes too. */
    bool on_cpu() const { return on_cpu_; }

private:
    /** The prediction network, the joint and the decoders of decodings() over the tensors of one model file. */
    struct Transducer {
        Transducer(const ModelFile & m, const std::vector<std::string> & names);
        Transducer(const Transducer &) = delete;
        Transducer & operator=(const Transducer &) = delete;
        const Decoder & decoder(const std::string & name) const;

        std::unique_ptr<PredictionNetwork> prediction;
        Joint joint;
        const std::vector<std::string> & names;
        /** The decoders of `names`, in their order. */
        std::vector<std::unique_ptr<Decoder>> decoders;
    };

    /**
     * The CPU's backend and its copy of the tensors of decoding_tensor(), with the decoders over them. The backend
     * computes with threads of its own that wait between graphs, where ggml's CPU backend otherwise starts and joins
     * its threads for every graph: on an Apple M5 with reazonspeech-nemo-v2 in Q8_0 on Metal, recognitions of 6, 26 and
     * 28 s by beam search took 89, 357 and 410 ms with 4 such threads, 109, 420 and 422 ms with 4 threads started for
     * each graph, 116, 451 and 504 ms with 1 thread, and 119, 446 and 526 ms decoding on Metal (2026-10-13, medians).
     * They keep ggml's default of spinning a while before they sleep, which held up best while other processes kept the
     * cores busy: with 4 and 8 such processes on the M5's 10 cores, 8 recognitions took 1,867 and 2,604 ms with these
     * threads, 1,967 and 4,280 ms with threads that sleep at once, and 2,392 and 3,167 ms with threads started for each
     * graph (2026-10-13, sums of medians). A model on the CPU computes without such threads: there they gained 2 to
     * 3% on a quiet M5 but lost 62% with reazonspeech-nemo-v2 and 73% with parakeet-tdt_ctc-0.6b-ja, both in Q8_0 with
     * 4 threads, while 8 processes kept the cores busy (2026-10-13).
     */
    struct OnCpu {
        OnCpu(const ModelFile & device, int threads, const std::vector<std::string> & names);

        std::unique_ptr<ggml_threadpool, decltype(&ggml_threadpool_free)> pool;
        std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> backend;
        ModelFile model;
        Transducer transducer;
    };

    /** The copy of the decoding's tensors on the CPU, made when a decoding first computes there. */
    OnCpu & cpu();

    ggml_backend_t backend_;
    bool on_cpu_;
    int threads_;
    ModelFile model_;
    NemoFrontend frontend_;
    Encoder encoder_;
    std::vector<std::string> decoding_names_;
    Transducer device_;
    /**
     * None for a model on the CPU, whose decodings compute on its own tensors, and for a model on another device until
     * a decoding first computes on the CPU: the copy takes the host's memory, 35 MB for parakeet-tdt-0.6b-v3's F16
     * file, which a model whose requests decode on its device never reads.
     */
    std::unique_ptr<OnCpu> cpu_;
    Detokenizer detokenizer_;
    std::vector<std::string> separators_, breaks_;
    ggml_gallocr_t allocr_;
};

}  // namespace fastconformer
