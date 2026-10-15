#pragma once

#include <cstdint>
#include <functional>
#include <map>
#include <string>
#include <vector>

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "graph.h"
#include "model-file.h"

namespace drama3 {

/** What a codec computed on the way, by stage name, each in ggml order ([channels, time] with the channel fastest). */
using CodecStages = std::map<std::string, std::vector<float>>;

/**
 * The codec of Drama 3, Fish Audio S2-Pro's: a causal DAC at 44.1 kHz. Its encoder turns 512 samples into a vector of
 * 1,024, and its quantizer halves the rate twice, runs a transformer and draws a semantic code from 4,096 and 9
 * residual codes from 1,024 each for every 2,048 samples; its decoder turns the codes back into samples the other way.
 * Every convolution is causal, padded on the left alone, and every attention is causal within a window.
 */
class Codec {
public:
    /** The codec of the model file `m`, which outlives it. */
    Codec(const ModelFile & m, ggml_backend_t backend);
    ~Codec();
    Codec(const Codec &) = delete;
    Codec & operator=(const Codec &) = delete;

    int sample_rate() const { return sample_rate_; }
    /** The samples of a frame of codes. */
    int hop() const { return hop_; }
    /** The codes of a frame: the semantic code, then the residual ones. */
    int codebooks() const { return 1 + residual_codebooks_; }

    /** The frames encode() gives `samples` samples, the last of them padded to a whole frame. */
    int64_t frames(size_t samples) const { return (int64_t) ((samples + (size_t) hop_ - 1) / (size_t) hop_); }

    /**
     * The codes of `audio`, mono at sample_rate(), as [codebooks, frames] with the frame fastest, the audio padded with
     * silence on the right to whole frames as the official encoder pads it. `stages`, when given, receives the encoder
     * blocks' outputs (encoder_0 to encoder_6), the downsampling's (downsample) and the transformer's before the
     * quantizer (pre_module).
     */
    std::vector<int32_t> encode(const std::vector<float> & audio, CodecStages * stages = nullptr);

    /**
     * The codes of `latent`, [1024, frames] with the channel fastest as the transformer before the quantizer gives it
     * (encode()'s stage pre_module), as encode() gives them: the semantic code of each frame, then each residual code
     * of what the codes before it leave.
     */
    std::vector<int32_t> quantize(const std::vector<float> & latent, int64_t frames);

    /**
     * The audio of `codes`, [codebooks, frames] with the frame fastest as encode() gives them, frames * hop() samples.
     * A code past its codebook is taken as the codebook's last, as the official decoder clamps it. `stages`, when given,
     * receives the codebooks' vectors summed (quantized), the transformer's output after the quantizer (post_module),
     * the upsampling's (upsample) and the decoder blocks' outputs (decoder_0 to decoder_7).
     */
    std::vector<float> decode(const std::vector<int32_t> & codes, int64_t frames, CodecStages * stages = nullptr);

    /** Starts decoding an utterance's frames a few at a time with decode_more(). */
    void start_decoding();

    /**
     * The audio of the next `frames` frames of the utterance, `codes` as decode() takes them, after those decoded since
     * start_decoding(): the transformer after the quantizer carries each layer's inputs of the rows its attention reaches,
     * at their positions, and the convolutions start from its outputs of the kHeldFrames frames before, so that the
     * audio is that of decoding every frame at once but for the rounding of other shapes of products.
     */
    std::vector<float> decode_more(const std::vector<int32_t> & codes, int64_t frames);

    /**
     * The frames of the transformer's outputs that the convolutions after it decode the next frames from: they reach
     * fewer than 10 frames before a frame, and 16 leave a margin.
     */
    static constexpr int64_t kHeldFrames = 16;

private:
    /**
     * Builds the decoding of `frames` frames of `codes` from position_: the codebooks' vectors, the transformer, whose
     * rows before these `before` carries where given, and the convolutions, from the transformer's outputs `held` of the
     * frames before where given; `post` receives the transformer's outputs and `stage` each stage.
     */
    ggml_tensor * build_decoder(Graph & g, const std::vector<int32_t> & codes, int64_t frames, std::vector<ggml_tensor *> * before,
                                ggml_tensor * held, ggml_tensor ** post, const std::function<void(const std::string &, ggml_tensor *)> & stage);

    ggml_backend_t backend_;
    const ModelFile & m_;
    ggml_gallocr_t allocr_ = nullptr;
    int sample_rate_ = 0, hop_ = 0, latent_ = 0, residual_codebooks_ = 0, semantic_size_ = 0, codebook_size_ = 0;
    std::vector<int32_t> encoder_rates_, decoder_rates_, factors_;
    /** The frames decoded since start_decoding(), each transformer layer's inputs of the last rows, and the held outputs. */
    int64_t position_ = 0;
    std::vector<std::vector<float>> before_;
    std::vector<float> held_;
};

}  // namespace drama3
