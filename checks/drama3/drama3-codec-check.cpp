// Checks the codec of Drama 3 against the official implementation on a dump of reference/drama3/dump.py, each stage
// in the order data flows. Encoder: its blocks, the downsampling and the transformer before the quantizer on the dump's
// reference audio; the quantizer on the dump's own input to it, and the codes of the whole encoding. Decoder: the
// codebooks' vectors, the transformer after the quantizer, the upsampling and the decoder's blocks on the codes of the
// dump's frames, and the audio. Decoding a few frames at a time, as a synthesis passes its audio: the reference's codes,
// longer than the transformer's attention reaches through its layers, in windows of 4, 8, 16, 32, 64 and then 128
// frames, against decoding them at once.
//
// With the dump's reference recording, also its reading, mixed to mono and resampled to the codec's rate as a voice
// file reads it, against the official reading, and the codes of that reading against the official codes.
//
// usage: drama3-codec-check <model.gguf> <dump dir> [reference.wav] [gpu|cpu|device name]

#include <chrono>
#include <cstdio>
#include <string>
#include <vector>

#include "args.h"
#include "backend.h"
#include "compare.h"
#include "drama3/codec.h"
#include "drama3/layout.h"
#include "drama3/voice-file.h"
#include "ggml-cpu.h"
#include "npy.h"
#include "weight-types.h"

using namespace drama3;

namespace {

double seconds_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

/** A dump of [channels, time] in ggml's order: [time][channels], the channel fastest. */
std::vector<float> channel_first(const Npy & a) {
    const size_t rows = (size_t) a.shape[0], cols = (size_t) a.shape[1];
    std::vector<float> out(a.f32.size());
    for (size_t r = 0; r < rows; r++)
        for (size_t c = 0; c < cols; c++) out[c * rows + r] = a.f32[r * cols + c];
    return out;
}

/** Compares each stage with the dump of codec_<stage>, and fails a stage below `min_snr`. */
bool check_stages(const CodecStages & stages, const std::vector<std::string> & names, const std::string & dir, double min_snr) {
    bool ok = true;
    for (const std::string & name : names) {
        const Npy want = read_npy(dir + "/codec_" + name + ".npy");
        const std::vector<float> & got = stages.at(name);
        const Diff d = compare(got, channel_first(want));
        print_diff(name, d);
        ok = ok && got.size() == want.f32.size() && d.snr_db > min_snr;
    }
    return ok;
}

/** The share of each codebook's codes of `got` that are those of `want`, both [codebooks, frames] with the frame fastest. */
std::vector<double> agreement(const std::vector<int32_t> & got, const std::vector<int32_t> & want, int codebooks) {
    const size_t frames = want.size() / codebooks;
    std::vector<double> share(codebooks);
    for (int b = 0; b < codebooks; b++) {
        size_t same = 0;
        for (size_t f = 0; f < frames; f++) same += got[b * frames + f] == want[b * frames + f];
        share[b] = (double) same / (double) frames;
    }
    return share;
}

void print_agreement(const std::string & what, const std::vector<double> & share) {
    std::printf("%-34s", what.c_str());
    for (double s : share) std::printf(" %5.1f%%", 100 * s);
    std::printf("\n");
}

}  // namespace

int main(int argc, char ** argv) {
    const std::vector<std::string> args = utf8_args(argc, argv);
    if (args.size() < 3) {
        std::fprintf(stderr, "usage: %s <model.gguf> <dump dir> [reference.wav] [gpu|cpu|device name]\n", args[0].c_str());
        return 2;
    }
    const std::string dir = args[2];
    const auto is_wav = [](const std::string & a) { return a.size() > 4 && a.compare(a.size() - 4, 4, ".wav") == 0; };
    const std::string reference = args.size() > 3 && is_wav(args[3]) ? args[3] : "";
    const size_t device_arg = reference.empty() ? 3 : 4;
    if (!takes_weights(args[1], model_layout, {"F32"}, "its thresholds are those measured with them")) return 2;
    ggml_backend_t backend = init_backend(args.size() > device_arg ? args[device_arg] : "");
    std::printf("backend: %s\n", ggml_backend_name(backend));
    const ModelFile model(args[1], backend, model_layout, [](const std::string & name) { return name.rfind("codec.", 0) == 0; });
    Codec codec(model, backend);
    // Measured on an Apple M5 on 2026-10-15 for the ja-ref dump, the lowest SNR of the stages: on the CPU 96.9 dB for the
    // encoder and 119.1 dB for the decoder, and on Metal, whose matrix kernel rounds its inputs to half precision, 35.9
    // and 61.9 dB. On the CPU, RoPE tables of float32 rather than of the official bfloat16 give 69.6 dB for the encoder
    // and 82.0 dB for the decoder.
    const bool cpu = ggml_backend_is_cpu(backend);
    const double encoder_db = cpu ? 85 : 30, decoder_db = cpu ? 100 : 50;
    bool ok = true;

    const Npy audio = read_npy(dir + "/reference_audio.npy");
    const Npy reference_codes = read_npy(dir + "/reference_codes.npy");
    CodecStages stages;
    auto t0 = std::chrono::steady_clock::now();
    const std::vector<int32_t> codes = codec.encode(audio.f32, &stages);
    std::printf("encoding %.2f s of audio: %.3f s\n", (double) audio.f32.size() / codec.sample_rate(), seconds_since(t0));
    ok = check_stages(stages, {"encoder_0", "encoder_1", "encoder_2", "encoder_3", "encoder_4", "encoder_5", "encoder_6",
                               "downsample", "pre_module"}, dir, encoder_db) && ok;
    const Npy pre_module = read_npy(dir + "/codec_pre_module.npy");
    const std::vector<int32_t> quantized = codec.quantize(channel_first(pre_module), pre_module.shape[1]);
    const std::vector<double> own = agreement(quantized, reference_codes.i32, codec.codebooks());
    print_agreement("codes of the dump's pre_module", own);
    // The quantizer on its own input draws every code of the dump on the CPU and on Metal alike; a frame whose nearest
    // codes lie as close as rounding may draw another, and the rest of its residual codes then differ too. The codes of
    // the whole encoding carry the encoder's rounding, which on Metal changes 1 to 12 % of each codebook's codes.
    for (double share : own) ok = ok && share >= 0.99;
    print_agreement("codes of the encoding", agreement(codes, reference_codes.i32, codec.codebooks()));
    ok = ok && codes.size() == reference_codes.i32.size();

    const Npy frames = read_npy(dir + "/frames.npy");
    const int64_t n = frames.shape[0], columns = frames.shape[1];
    std::vector<int32_t> frame_codes((size_t) (n * codec.codebooks()));
    for (int64_t f = 0; f < n; f++)
        for (int b = 0; b < codec.codebooks(); b++) frame_codes[(size_t) (b * n + f)] = frames.i32[(size_t) (f * columns + 1 + b)];
    stages.clear();
    t0 = std::chrono::steady_clock::now();
    const std::vector<float> decoded = codec.decode(frame_codes, n, &stages);
    std::printf("decoding %lld frames: %.3f s\n", (long long) n, seconds_since(t0));
    ok = check_stages(stages, {"quantized", "post_module", "upsample", "decoder_0", "decoder_1", "decoder_2", "decoder_3",
                               "decoder_4", "decoder_5", "decoder_6", "decoder_7"}, dir, decoder_db) && ok;
    const Npy official = read_npy(dir + "/audio.npy");
    const Diff da = compare(decoded, official.f32);
    print_diff("audio", da);
    ok = ok && decoded.size() == official.f32.size() && da.snr_db > decoder_db;
    if (!reference.empty()) {
        const std::vector<float> read = read_recording(reference, codec.sample_rate());
        const Diff dr = compare(read, audio.f32);
        print_diff("reading the WAVE file", dr);
        const std::vector<double> codes_read = agreement(codec.encode(read), reference_codes.i32, codec.codebooks());
        print_agreement("codes of the reading", codes_read);
        // Measured on an Apple M5 on 2026-10-15: 129.2 dB on any device, since the reading runs on the host, and every
        // code of the reading the official one on the CPU, 88 to 99% of each codebook's on Metal. librosa's kaiser_best
        // filter in place of torchaudio's default gives 66.5 dB, and 73 to 98% of the codes on the CPU.
        ok = ok && read.size() == audio.f32.size() && dr.snr_db > 100;
    }

    const int64_t reference_frames = reference_codes.shape[1];
    const std::vector<float> whole = codec.decode(reference_codes.i32, reference_frames);
    std::vector<float> windowed;
    codec.start_decoding();
    for (int64_t first = 0, i = 0; first < reference_frames; i++) {
        const int64_t n = std::min<int64_t>(int64_t(4) << std::min<int64_t>(i, 5), reference_frames - first);
        std::vector<int32_t> part((size_t) (n * codec.codebooks()));
        for (int b = 0; b < codec.codebooks(); b++)
            for (int64_t f = 0; f < n; f++) part[(size_t) (b * n + f)] = reference_codes.i32[(size_t) (b * reference_frames + first + f)];
        const std::vector<float> audio = codec.decode_more(part, n);
        windowed.insert(windowed.end(), audio.begin(), audio.end());
        first += n;
    }
    const Diff dw = compare(windowed, whole);
    print_diff(std::to_string(reference_frames) + " frames in windows against at once", dw);
    // Measured on an Apple M5 on 2026-10-15: 128.9 dB on the CPU in F32 and 61.5 dB on Metal, whose matrix kernel
    // rounds its inputs to half precision. Windows decoded at once from the 136 frames before them, which one layer's
    // attention reaches but not the eight layers together, give 34.6 dB on the CPU.
    ok = ok && windowed.size() == whole.size() && dw.snr_db > (cpu ? 100 : 50);
    ggml_backend_free(backend);
    return ok ? 0 : 1;
}
