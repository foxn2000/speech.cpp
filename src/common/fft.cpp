#include "fft.h"

#include <cstdlib>
#include <stdexcept>
#include <string>

Fft::Fft(int size) : size_(size) {
    if (size < 1) throw std::logic_error("a Fourier transform of " + std::to_string(size) + " values");
    for (int n = size, p = 2; n > 1;) {
        if (n % p == 0) {
            factors_.push_back(p);
            n /= p;
        } else {
            p = p * p > n ? n : p + 1;
        }
    }
    twiddles_.reserve((size_t) size);
    for (int t = 0; t < size; t++) twiddles_.push_back(std::polar(1.0, -2 * 3.14159265358979323846 * t / size));
}

void Fft::transform(const std::complex<double> * in, std::complex<double> * out) const {
    if (factors_.empty()) {
        out[0] = in[0];
        return;
    }
    work(in, out, 1, 0);
}

void Fft::work(const std::complex<double> * in, std::complex<double> * out, size_t stride, size_t level) const {
    const size_t p = (size_t) factors_[level], n = (size_t) size_ / stride, m = n / p;
    // out[j m + k] becomes the transform of the j-th of the p interleaved subsequences at frequency k.
    for (size_t j = 0; j < p; j++) {
        if (m == 1) out[j] = in[j * stride];
        else work(in + j * stride, out + j * m, stride * p, level + 1);
    }
    // X[k + s m] = Σ_j exp(-2πi j (k + s m) / n) Y_j[k], where exp(-2πi e / n) is twiddle e × stride.
    static const bool plain = std::getenv("SPEECH_FFT_PLAIN") != nullptr;
    if (p == 2 && plain) {
        for (size_t k = 0; k < m; k++) {
            const double ar = out[k].real(), ai = out[k].imag(), br = out[m + k].real(), bi = out[m + k].imag();
            const std::complex<double> t0 = twiddles_[stride * k], t1 = twiddles_[stride * (k + m)];
            out[k] = {ar + (br * t0.real() - bi * t0.imag()), ai + (br * t0.imag() + bi * t0.real())};
            out[m + k] = {ar + (br * t1.real() - bi * t1.imag()), ai + (br * t1.imag() + bi * t1.real())};
        }
        return;
    }
    if (p == 2) {
        for (size_t k = 0; k < m; k++) {
            const auto a = out[k];
            const auto b = out[m + k];
            out[k] = a + b * twiddles_[stride * k];
            out[m + k] = a + b * twiddles_[stride * (k + m)];
        }
        return;
    }
    std::vector<std::complex<double>> y(p);
    for (size_t k = 0; k < m; k++) {
        for (size_t j = 0; j < p; j++) y[j] = out[j * m + k];
        for (size_t s = 0; s < p; s++) {
            std::complex<double> sum = y[0];
            for (size_t j = 1; j < p; j++) sum += y[j] * twiddles_[(stride * j * (k + s * m)) % (size_t) size_];
            out[k + s * m] = sum;
        }
    }
}
