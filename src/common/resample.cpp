#include "resample.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <numeric>
#include <stdexcept>
#include <string>

namespace {

constexpr double kKaiserZeroCrossings = 64;
constexpr double kKaiserRolloff = 0.9475937167399596;
constexpr double kBeta = 14.769656459379492;
constexpr double kHannZeroCrossings = 6;
constexpr double kHannRolloff = 0.99;
constexpr double kPi = 3.141592653589793;

/**
 * The largest term of the ratio of two rates in lowest terms. The kernel holds `to` phases of about
 * 135 · max(1, from / to) taps, under 5 MB up to 4096, and takes 0.05 s to make at 4096:4095 on an Apple M5
 * (2026-10-06); both grow with the terms. Rates whose ratio needs larger terms (44101 Hz and 16000 Hz) are not the
 * rates of recorded audio.
 */
constexpr int kMaxRatioTerm = 4096;

/** The modified Bessel function of the first kind of order 0, by its power series, whose terms are all positive. */
double bessel_i0(double x) {
    const double q = x * x / 4;
    double sum = 1, term = 1;
    for (int k = 1; term > sum * 1e-17; k++) {
        term *= q / ((double) k * k);
        sum += term;
    }
    return sum;
}

}  // namespace

Resampler::Resampler(int from_rate, int to_rate, ResampleFilter filter) {
    const std::string rates = "from " + std::to_string(from_rate) + " Hz to " + std::to_string(to_rate) + " Hz";
    if (from_rate <= 0 || to_rate <= 0) {
        throw std::invalid_argument("cannot resample " + rates + ": a sample rate is a positive number of Hz; give the rate "
                                    "the audio was recorded at");
    }
    const int gcd = std::gcd(from_rate, to_rate);
    from_ = from_rate / gcd;
    to_ = to_rate / gcd;
    if (from_ > kMaxRatioTerm || to_ > kMaxRatioTerm) {
        throw std::invalid_argument("cannot resample " + rates + ": their ratio in lowest terms, " + std::to_string(from_) +
                                    ":" + std::to_string(to_) + ", has a term above " + std::to_string(kMaxRatioTerm) +
                                    "; convert the audio to a common rate such as 16000, 44100 or 48000 Hz first");
    }
    if (from_ == to_) return;
    // torchaudio's _get_sinc_resample_kernel() step by step, so that each tap is torchaudio's to the last bit or two.
    const bool hann = filter == ResampleFilter::Hann;
    const double crossings = hann ? kHannZeroCrossings : kKaiserZeroCrossings;
    const double base = std::min(from_, to_) * (hann ? kHannRolloff : kKaiserRolloff);
    width_ = (long long) std::ceil(crossings * from_ / base);
    const double scale = base / from_;
    // torchaudio makes beta a float32 tensor, so its window's beta is kBeta rounded to float32 and the window's peak
    // i0(beta), which it divides by, is computed in float32 (271903.5625 with torch 2.10.0, the double value
    // rounded). With exact values the output departs from torchaudio's by up to 1.1e-8, 154 dB SNR.
    const double beta = (float) kBeta;
    const double peak = (float) bessel_i0(beta);
    // torchaudio evaluates 2 · width + from taps per phase and clamps those past the last zero crossing to it, where
    // each is below 1e-21; only the taps before it are kept.
    const long long span = 2 * width_ + from_;
    std::vector<std::vector<double>> phases((size_t) to_);
    first_.assign((size_t) to_, 0);
    for (int j = 0; j < to_; j++) {
        for (long long k = 0; k < span; k++) {
            const double t = ((double) -j / to_ + (double) (k - width_) / from_) * base;
            if (std::fabs(t) >= crossings) continue;
            if (phases[j].empty()) first_[j] = k;
            const double r = t / crossings;
            const double half_cosine = std::cos(t * kPi / crossings / 2);
            const double window = hann ? half_cosine * half_cosine : bessel_i0(beta * std::sqrt(1 - r * r)) / peak;
            const double x = t * kPi;
            phases[j].push_back((x == 0 ? 1.0 : std::sin(x) / x) * (window * scale));
        }
        taps_ = std::max(taps_, (int) phases[j].size());
    }
    kernel_.assign((size_t) to_ * taps_, 0.0);
    for (int j = 0; j < to_; j++) std::copy(phases[j].begin(), phases[j].end(), kernel_.begin() + (ptrdiff_t) j * taps_);
}

size_t Resampler::length(long long n) const {
    if (from_ == to_) return (size_t) n;
    // torchaudio rounds n · to / from to float32 (torch.as_tensor of a Python float) before its ceiling, which once
    // the output reaches about 2^24 / from samples is now and then one below the exact ceiling: 180697 samples from
    // 44100 Hz become 65559 at 16000 Hz, not 65560. Its length is kept so that the output is torchaudio's sample for
    // sample. Its convolution makes (n / from + 1) · to samples, of which it keeps that many.
    return std::min((size_t) std::ceil((float) ((double) n * to_ / from_)), (size_t) (n / from_ + 1) * to_);
}

long long Resampler::needs(size_t m) const {
    return (long long) (m / (size_t) to_) * from_ + first_[m % (size_t) to_] - width_ + taps_;
}

long long Resampler::keeps(size_t m) const {
    return std::max(0LL, (long long) (m / (size_t) to_) * from_ - width_);
}

double Resampler::sample(size_t m, const float * x, long long first, long long n) const {
    const int phase = (int) (m % (size_t) to_);
    const long long start = (long long) (m / (size_t) to_) * from_ + first_[phase] - width_;
    const long long lo = std::max(0LL, -start), hi = std::min((long long) taps_, n - start);
    const double * h = kernel_.data() + (size_t) phase * taps_;
    const long long at = start - first;
    // Four sums, so that the additions do not wait for each other.
    double sum[4] = {0, 0, 0, 0};
    long long r = lo;
    for (; r + 4 <= hi; r += 4) {
        for (int s = 0; s < 4; s++) sum[s] += h[r + s] * x[at + r + s];
    }
    for (; r < hi; r++) sum[0] += h[r] * x[at + r];
    return (sum[0] + sum[1]) + (sum[2] + sum[3]);
}

template <typename T>
std::vector<T> Resampler::apply(const std::vector<float> & samples) const {
    if (from_ == to_) return std::vector<T>(samples.begin(), samples.end());
    const long long n = (long long) samples.size();
    std::vector<T> out(length(n));
    for (size_t m = 0; m < out.size(); m++) out[m] = (T) sample(m, samples.data(), 0, n);
    return out;
}

std::vector<float> Resampler::operator()(std::vector<float> samples) const {
    if (from_ == to_) return samples;
    return apply<float>(samples);
}

std::vector<double> Resampler::in_double(const std::vector<float> & samples) const {
    return apply<double>(samples);
}

ResampleStream::ResampleStream(int from_rate, int to_rate, ResampleFilter filter) : resampler_(from_rate, to_rate, filter) {}

void ResampleStream::push(const float * samples, size_t n, std::vector<float> & out) {
    if (resampler_.unchanged()) {
        out.insert(out.end(), samples, samples + n);
        return;
    }
    input_.insert(input_.end(), samples, samples + n);
    received_ += (long long) n;
    // An output sample is final once its taps lie within the input received, and the output of the input so far, which
    // never shrinks as the input grows, bounds the whole's.
    const size_t ready = resampler_.length(received_);
    for (; given_ < ready && resampler_.needs(given_) <= received_; given_++) {
        out.push_back((float) resampler_.sample(given_, input_.data(), first_, received_));
    }
    const long long keep = std::min(resampler_.keeps(given_), received_);
    if (keep > first_) {
        input_.erase(input_.begin(), input_.begin() + (ptrdiff_t) (keep - first_));
        first_ = keep;
    }
}

void ResampleStream::end(std::vector<float> & out) {
    if (resampler_.unchanged()) return;
    for (const size_t length = resampler_.length(received_); given_ < length; given_++) {
        out.push_back((float) resampler_.sample(given_, input_.data(), first_, received_));
    }
}
