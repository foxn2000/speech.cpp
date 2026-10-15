#pragma once

#include <cstddef>
#include <vector>

/** The filters of torchaudio.functional.resample() that the official implementations resample with. */
enum class ResampleFilter {
    /**
     * The parameters torchaudio's documentation gives for librosa's kaiser_best: 64 zero crossings on each side, cut
     * off at 0.9475937167399596 of the lower rate's Nyquist frequency, under a Kaiser window of beta 14.769656459379492,
     * which torchaudio rounds to float32 together with the window's peak i0(beta). Irodori-TTS's runtime takes it.
     */
    KaiserBest,
    /** torchaudio's defaults: 6 zero crossings, cut off at 0.99, under a Hann window. fish-speech takes it. */
    Hann,
};

/**
 * Changes the sample rate of mono audio as torchaudio.functional.resample() does with the parameters of a
 * ResampleFilter: a rational polyphase windowed sinc computed in double precision. The output has torchaudio's length, ceil(n · to / from) as torchaudio rounds it,
 * and the audio is padded with zeros at both ends as torchaudio pads it. At equal rates the samples pass through
 * unchanged.
 */
class Resampler {
public:
    /**
     * Prepares the filter from `from_rate` to `to_rate` Hz. Throws for a rate that is not positive and for two rates
     * whose ratio in lowest terms has a term above 4096.
     */
    Resampler(int from_rate, int to_rate, ResampleFilter filter = ResampleFilter::KaiserBest);

    /** The samples at the new rate, each rounded to float; at equal rates, `samples` themselves. */
    std::vector<float> operator()(std::vector<float> samples) const;

    /** The samples at the new rate in double precision, before the rounding to float. */
    std::vector<double> in_double(const std::vector<float> & samples) const;

    /** Whether the rates are equal, and the samples pass through unchanged. */
    bool unchanged() const { return from_ == to_; }

    /** The number of samples at the new rate of `n` samples, torchaudio's, which never falls as `n` grows. */
    size_t length(long long n) const;

    /** The input sample after the last one that output sample `m` weighs, whatever the input's length. */
    long long needs(size_t m) const;

    /** An input sample before every one that output sample `m`, or any after it, weighs. */
    long long keeps(size_t m) const;

    /**
     * Output sample `m` of `n` input samples, of which `x` holds those from `first` on, at least every one that the
     * sample weighs: the same sum, in the same order, whether `x` holds the whole input or a piece of it.
     */
    double sample(size_t m, const float * x, long long first, long long n) const;

private:
    template <typename T>
    std::vector<T> apply(const std::vector<float> & samples) const;

    /** The ratio in lowest terms: `from_` input samples become `to_` output samples. */
    int from_ = 1, to_ = 1;
    /** torchaudio's padding of the input on the left, in input samples. */
    long long width_ = 0;
    /** The taps of each of the `to_` phases: phase j starts at input offset first_[j] - width_. */
    int taps_ = 0;
    std::vector<long long> first_;
    /** [to_][taps_], zero past the end of a phase's taps. */
    std::vector<double> kernel_;
};

/**
 * Resampler's resampling of audio that arrives a piece at a time. An output sample is computed once every input sample it
 * weighs has arrived, and the last ones, which weigh the zeros after the audio, once the audio has ended, so that the
 * output is the whole audio's sample for sample. An output sample therefore follows the input by the half of the filter
 * after it, 64 zero crossings of the lower rate's cutoff, 64 / (0.9476 · the lower rate) seconds: 4.3 ms from 24, 44.1
 * or 48 kHz to 16 kHz and 8.6 ms from 8 kHz. The input it keeps is that half and the half before.
 */
class ResampleStream {
public:
    /** Throws as Resampler does. */
    ResampleStream(int from_rate, int to_rate, ResampleFilter filter = ResampleFilter::KaiserBest);

    /** Takes the next samples and appends to `out` the output samples they complete. */
    void push(const float * samples, size_t n, std::vector<float> & out);

    /** Ends the audio and appends the rest of the output, which then has the whole audio's length. */
    void end(std::vector<float> & out);

private:
    Resampler resampler_;
    /** The input from sample `first_` on. */
    std::vector<float> input_;
    long long first_ = 0, received_ = 0;
    /** The output samples given. */
    size_t given_ = 0;
};
