// Checks the resampler against torchaudio.functional.resample() in float64 with librosa's kaiser_best parameters, and
// with its default Hann filter where a dump's name ends in -hann, on the chirps and the noise of reference/resample/dump.py: each output has torchaudio's length and its samples, to what
// double precision explains. Each input given a piece at a time, in pieces of one sample, 10 ms and 100 ms and of random
// sizes from a fixed seed, must give the whole input's float samples bit for bit. Then checks that audio at the rate it
// is asked for passes through unchanged, whole and in pieces.
//
// usage: resample-check <reference/resample out dir>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "args.h"
#include "compare.h"
#include "npy.h"
#include "resample.h"

namespace {

/** The samples at `to` Hz of `input` given in pieces of `size` samples, or of random sizes for 0. */
std::vector<float> in_pieces(int from, int to, const std::vector<float> & input, size_t size, ResampleFilter filter = ResampleFilter::KaiserBest) {
    ResampleStream stream(from, to, filter);
    std::vector<float> out;
    std::mt19937 draw(20261008);
    for (size_t at = 0; at < input.size();) {
        const size_t n = std::min(size ? size : 1 + (size_t) (draw() % 9600), input.size() - at);
        stream.push(input.data() + at, n, out);
        at += n;
    }
    stream.end(out);
    return out;
}

}  // namespace

int main(int argc, char ** argv) {
    const std::vector<std::string> args = utf8_args(argc, argv);
    if (args.size() < 2) {
        std::fprintf(stderr, "usage: %s <reference/resample out dir>\n", args[0].c_str());
        return 2;
    }
    try {
        std::vector<std::filesystem::path> dumps;
        for (const auto & e : std::filesystem::directory_iterator(std::filesystem::u8path(args[1]))) {
            if (std::filesystem::is_regular_file(e.path() / "output.npy")) dumps.push_back(e.path());
        }
        std::sort(dumps.begin(), dumps.end());
        if (dumps.empty()) throw std::runtime_error("no dump of reference/resample/dump.py is in " + args[1]);
        bool ok = true;
        for (const auto & d : dumps) {
            const std::string name = d.filename().u8string();
            int from = 0, to = 0;
            if (std::sscanf(name.c_str(), "%d-%d-", &from, &to) != 2) throw std::runtime_error(name + " does not name two rates");
            const Npy input = read_npy((d / "input.npy").u8string());
            const Npy want = read_npy((d / "output.npy").u8string());
            const auto t0 = std::chrono::steady_clock::now();
            const ResampleFilter filter = name.size() > 5 && name.compare(name.size() - 5, 5, "-hann") == 0 ? ResampleFilter::Hann : ResampleFilter::KaiserBest;
            const std::vector<double> got = Resampler(from, to, filter).in_double(input.f32);
            const double took = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            const Diff diff = compare(got, want.f64);
            std::printf("%-22s %zu -> %zu samples (torchaudio %zu), %.3f s\n", name.c_str(), input.f32.size(), got.size(),
                        want.f64.size(), took);
            print_diff("  against torchaudio", diff);
            // Measured on an Apple M5 on 2026-10-06: 301 to 308 dB, largest difference 8e-16 to 1.7e-15, the rounding
            // of double precision summed in another order. The window's beta unrounded, where torchaudio rounds it to
            // float32, gives 154 dB, and taps rounded to float32 149 dB.
            ok = ok && got.size() == want.f64.size() && diff.snr_db > 250;

            const std::vector<float> whole = Resampler(from, to, filter)(input.f32);
            bool same = true;
            for (const size_t size : {(size_t) 1, (size_t) from / 100, (size_t) from / 10, (size_t) 0}) {
                const std::vector<float> streamed = in_pieces(from, to, input.f32, size, filter);
                same = same && streamed.size() == whole.size() && std::memcmp(streamed.data(), whole.data(), whole.size() * sizeof(float)) == 0;
            }
            std::printf("  %-32s %s\n", "in pieces", same ? "the whole's samples" : "DIFFER");
            ok = ok && same;
        }
        std::vector<float> same(48000);
        for (size_t i = 0; i < same.size(); i++) same[i] = (float) ((double) i / same.size() - 0.5);
        const bool unchanged = Resampler(48000, 48000)(same) == same && Resampler(16000, 16000).in_double(same) ==
                                                                            std::vector<double>(same.begin(), same.end()) &&
                               in_pieces(16000, 16000, same, 0) == same;
        std::printf("equal rates: the samples %s\n", unchanged ? "unchanged" : "changed");
        return ok && unchanged ? 0 : 1;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
