#include "wav.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>

#include "error.h"

namespace {

constexpr uint16_t kFormatPcm = 1, kFormatFloat = 3, kFormatExtensible = 0xFFFE;

uint16_t u16(const uint8_t * p) { return (uint16_t) (p[0] | p[1] << 8); }
uint32_t u32(const uint8_t * p) { return (uint32_t) p[0] | (uint32_t) p[1] << 8 | (uint32_t) p[2] << 16 | (uint32_t) p[3] << 24; }

}  // namespace

std::vector<float> Wav::mono() const {
    if (channels == 1) return samples;
    std::vector<float> out(samples.size() / channels);
    for (size_t i = 0; i < out.size(); i++) {
        float sum = 0;
        for (int c = 0; c < channels; c++) sum += samples[i * channels + c];
        out[i] = sum / (float) channels;
    }
    return out;
}

Wav read_wav(const std::string & path) {
    std::ifstream f(std::filesystem::u8path(path), std::ios::binary);
    if (!f) throw Error(Fault::Io, "cannot open " + path + "; check the path and that the file can be read");
    return parse_wav(std::string((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>()), path);
}

bool starts_with_riff(const std::string & path) {
    std::ifstream f(std::filesystem::u8path(path), std::ios::binary);
    if (!f) throw Error(Fault::Io, "cannot open " + path + "; check the path and that the file can be read");
    char magic[4] = {};
    f.read(magic, 4);
    return std::memcmp(magic, "RIFF", 4) == 0;
}

Wav parse_wav(const std::string & file, const std::string & path) {
    const auto * bytes = reinterpret_cast<const uint8_t *>(file.data());
    const size_t size_of_file = file.size();
    if (size_of_file < 12 || std::memcmp(bytes, "RIFF", 4) != 0 || std::memcmp(bytes + 8, "WAVE", 4) != 0) {
        throw Error(Fault::File, path + " is not a WAVE file");
    }
    uint16_t format = 0, bits = 0;
    Wav wav;
    const uint8_t * data = nullptr;
    size_t data_size = 0;
    for (size_t at = 12; at + 8 <= size_of_file;) {
        const uint8_t * chunk = bytes + at;
        const size_t size = u32(chunk + 4);
        if (at + 8 + size > size_of_file) throw Error(Fault::File, path + " ends inside a chunk");
        if (std::memcmp(chunk, "fmt ", 4) == 0 && size >= 16) {
            format = u16(chunk + 8);
            wav.channels = u16(chunk + 10);
            wav.sample_rate = (int) u32(chunk + 12);
            bits = u16(chunk + 22);
            if (format == kFormatExtensible && size >= 40) format = u16(chunk + 32);
        } else if (std::memcmp(chunk, "data", 4) == 0) {
            data = chunk + 8;
            data_size = size;
        }
        at += 8 + size + (size & 1);
    }
    if (!data || wav.channels == 0) throw Error(Fault::File, path + " has no format or no data");
    const bool pcm = format == kFormatPcm && (bits == 16 || bits == 24 || bits == 32);
    const bool ieee = format == kFormatFloat && bits == 32;
    if (!pcm && !ieee) {
        throw Error(Fault::File, path + " is neither 16-, 24- or 32-bit PCM nor 32-bit float WAVE (format " +
                                 std::to_string(format) + ", " + std::to_string(bits) + " bits)");
    }
    const size_t width = bits / 8, n = data_size / width;
    wav.samples.resize(n);
    for (size_t i = 0; i < n; i++) {
        const uint8_t * p = data + i * width;
        if (ieee) {
            std::memcpy(&wav.samples[i], p, 4);
        } else if (bits == 16) {
            wav.samples[i] = (float) (int16_t) u16(p) / 32768.0f;
        } else if (bits == 24) {
            const int32_t v = (int32_t) ((uint32_t) p[0] << 8 | (uint32_t) p[1] << 16 | (uint32_t) p[2] << 24) >> 8;
            wav.samples[i] = (float) v / 8388608.0f;
        } else {
            wav.samples[i] = (float) ((double) (int32_t) u32(p) / 2147483648.0);
        }
    }
    wav.samples.resize(n / wav.channels * wav.channels);
    return wav;
}

void write_wav(const std::string & path, const std::vector<float> & pcm, int rate) {
    std::ofstream f(std::filesystem::u8path(path), std::ios::binary);
    if (!f) throw Error(Fault::Io, "cannot write " + path);
    const uint32_t data_size = (uint32_t) pcm.size() * 2;
    auto put32 = [&](uint32_t v) { f.write((const char *) &v, 4); };
    auto put16 = [&](uint16_t v) { f.write((const char *) &v, 2); };
    f.write("RIFF", 4);
    put32(36 + data_size);
    f.write("WAVEfmt ", 8);
    put32(16);
    put16(1);
    put16(1);
    put32(rate);
    put32(rate * 2);
    put16(2);
    put16(16);
    f.write("data", 4);
    put32(data_size);
    for (float s : pcm) {
        const int16_t v = (int16_t) std::lround(std::max(-1.0f, std::min(1.0f, s)) * 32767.0f);
        f.write((const char *) &v, 2);
    }
}
