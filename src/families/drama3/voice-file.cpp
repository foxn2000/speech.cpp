#include "voice-file.h"

#include <algorithm>
#include <memory>
#include <set>
#include <stdexcept>

#include "backend.h"
#include "codec.h"
#include "error.h"
#include "gguf.h"
#include "layout.h"
#include "prompt.h"
#include "resample.h"
#include "wav.h"

namespace drama3 {

namespace {

void require(bool condition, const ModelFile & m, const std::string & what) {
    if (!condition) throw Error(Fault::File, m.path() + ": " + what + "; " + m.remedy());
}

}  // namespace

std::vector<float> read_recording(const std::string & path, int rate) {
    const Wav wav = read_wav(path);
    const Resampler resample = [&] {
        try {
            return Resampler(wav.sample_rate, rate, ResampleFilter::Hann);
        } catch (const std::invalid_argument & e) {
            throw Error(Fault::InvalidArgument, path + ": " + e.what());
        }
    }();
    return resample(wav.mono());
}

namespace {

void write_voice_file(const std::string & path, const std::vector<VoiceReference> & references, const std::string & codec_sha256,
                      const std::string & device) {
    std::unique_ptr<gguf_context, decltype(&gguf_free)> g(gguf_init_empty(), gguf_free);
    const std::string p = std::string(kVoiceArchitecture) + ".";
    gguf_set_val_str(g.get(), "general.architecture", kVoiceArchitecture);
    gguf_set_val_u32(g.get(), "speech.layout", kVoiceLayout);
    gguf_set_val_str(g.get(), "speech.requires", kVoiceLayoutRequires);
    gguf_set_val_str(g.get(), (p + "codec_sha256").c_str(), codec_sha256.c_str());
    gguf_set_val_str(g.get(), (p + "device_kind").c_str(), device.c_str());
    std::vector<int32_t> speakers, frames, codes;
    std::vector<const char *> transcripts;
    for (const VoiceReference & r : references) {
        speakers.push_back(r.speaker);
        frames.push_back((int32_t) r.frames);
        transcripts.push_back(r.transcript.c_str());
        codes.insert(codes.end(), r.codes.begin(), r.codes.end());
    }
    gguf_set_arr_data(g.get(), (p + "speakers").c_str(), GGUF_TYPE_INT32, speakers.data(), speakers.size());
    gguf_set_arr_str(g.get(), (p + "transcripts").c_str(), transcripts.data(), transcripts.size());
    gguf_set_arr_data(g.get(), (p + "frames").c_str(), GGUF_TYPE_INT32, frames.data(), frames.size());
    gguf_set_arr_data(g.get(), (p + "codes").c_str(), GGUF_TYPE_INT32, codes.data(), codes.size());
    if (!gguf_write_to_file(g.get(), path.c_str(), false)) {
        throw Error(Fault::Io, "cannot write " + path + "; check that its folder exists and is writable", "voice_path");
    }
}

}  // namespace

Layout voice_layout(const ModelFile & model) {
    return {kVoiceArchitecture, kVoiceLayout, "make it again with speech voice from its WAVE files and their transcripts", [&model](const ModelFile & m) {
                const std::string p = std::string(kVoiceArchitecture) + ".";
                const std::string codec = m.str(p + "codec_sha256"), own = model.str("drama3.codec.sha256");
                require(codec.size() == 64, m, p + "codec_sha256 is not a SHA-256 in hexadecimal");
                if (codec != own) {
                    throw Error(Fault::InvalidArgument, m.path() + " was made with the codec of SHA-256 " + codec + ", and " + model.str("general.name") +
                                                            " has the codec " + own + "; make the voice again from its WAVE files with this model");
                }
                const std::vector<int32_t> speakers = m.i32_array(p + "speakers"), frames = m.i32_array(p + "frames");
                const std::vector<std::string> transcripts = m.str_array(p + "transcripts");
                require(!speakers.empty() && speakers.size() == frames.size() && speakers.size() == transcripts.size(), m,
                        "the voice file does not give each of its references a speaker, frames and a transcript");
                const int most = model.size("drama3.generation.speakers");
                const int64_t codebooks = 1 + model.count("drama3.codec.residual_codebooks");
                int64_t total = 0;
                for (size_t i = 0; i < speakers.size(); i++) {
                    require(speakers[i] >= 0 && speakers[i] < most && frames[i] >= 1 && !transcripts[i].empty(), m,
                            "the voice file gives a reference a speaker the model does not have, no frames or no transcript");
                    total += frames[i];
                }
                const std::vector<int32_t> codes = m.i32_array(p + "codes");
                require((int64_t) codes.size() == codebooks * total, m, "the voice file's codes are not its references' frames of codes");
                // The semantic codebook's codes, first in each reference, and the residual codebooks' after them.
                const int32_t semantic = model.size("drama3.codec.semantic_codebook_size"), residual = model.size("drama3.codec.codebook_size");
                size_t at = 0;
                for (int32_t f : frames) {
                    for (int64_t b = 0; b < codebooks; b++) {
                        for (int32_t i = 0; i < f; i++, at++) require(codes[at] >= 0 && codes[at] < (b == 0 ? semantic : residual), m, "a code of the voice file is outside its codebook");
                    }
                }
                m.one_of(p + "device_kind", {"cpu", "gpu", "igpu"});
                return std::vector<TensorSpec>{};
            }};
}

std::vector<VoiceReference> read_voice_file(const std::string & path, const ModelFile & model) {
    const ModelFile m(path, voice_layout(model));
    const std::string p = std::string(kVoiceArchitecture) + ".";
    const std::vector<int32_t> speakers = m.i32_array(p + "speakers"), frames = m.i32_array(p + "frames"), codes = m.i32_array(p + "codes");
    const std::vector<std::string> transcripts = m.str_array(p + "transcripts");
    const int64_t codebooks = 1 + model.count("drama3.codec.residual_codebooks");
    std::vector<VoiceReference> references;
    size_t at = 0;
    for (size_t i = 0; i < speakers.size(); i++) {
        const size_t n = (size_t) (codebooks * frames[i]);
        references.push_back({speakers[i], transcripts[i], std::vector<int32_t>(codes.begin() + (std::ptrdiff_t) at, codes.begin() + (std::ptrdiff_t) (at + n)),
                              frames[i]});
        at += n;
    }
    return references;
}

Voice voice_of(const std::vector<VoiceReference> & references) {
    Voice voice;
    for (const VoiceReference & r : references) voice.references.push_back({"<|speaker:" + std::to_string(r.speaker) + "|> " + r.transcript, r.codes, r.frames});
    return voice;
}

std::string describe_speakers(const std::vector<VoiceReference> & references) {
    std::set<int> speakers;
    for (const VoiceReference & r : references) speakers.insert(r.speaker);
    std::string out = speakers.size() == 1 ? "speaker " : "speakers ";
    size_t i = 0;
    for (int s : speakers) {
        out += (i == 0 ? "" : i + 1 == speakers.size() ? " and " : ", ") + std::to_string(s);
        i++;
    }
    return out;
}

void make_voice_file(const std::string & model_path, const std::vector<VoiceSource> & sources, const std::string & voice_path,
                     ggml_backend_t backend) {
    const auto model = naming("model_path", [&] {
        return std::make_unique<const ModelFile>(model_path, backend, model_layout, [](const std::string & name) { return name.rfind("codec.", 0) == 0; });
    });
    Codec codec(*model, backend);
    const Prompt prompt(*model);
    const int most = model->size("drama3.generation.speakers");
    std::vector<VoiceReference> references;
    // Each recording's audio by the index of its reference, encoded once the references are known to fit a prompt.
    std::vector<std::pair<size_t, std::vector<float>>> recordings;
    for (size_t i = 0; i < sources.size(); i++) {
        const VoiceSource & s = sources[i];
        if (s.voice_file) {
            std::vector<VoiceReference> added = naming("references", [&] { return read_voice_file(s.path, *model); });
            if (s.speaker) {
                std::set<int> speakers;
                for (const VoiceReference & r : added) speakers.insert(r.speaker);
                if (speakers.size() > 1) {
                    throw Error(Fault::InvalidArgument, s.path + " holds " + describe_speakers(added) +
                                                            ", and a speaker is given to a voice file of one speaker alone; leave it out to keep its speakers",
                                "speaker");
                }
                for (VoiceReference & r : added) r.speaker = *s.speaker;
            }
            references.insert(references.end(), added.begin(), added.end());
            continue;
        }
        if (!s.transcript) {
            throw Error(Fault::InvalidArgument, s.path + " has no transcript; give the words it speaks, which Drama 3 reads beside it", "transcript");
        }
        const std::string what = "the transcript of " + s.path;
        prompt.check(*s.transcript, "transcript", what);
        if (s.transcript->find("<|speaker") != std::string::npos) {
            throw Error(Fault::InvalidArgument, what + " holds a speaker tag; give the recording's speaker as its number instead", "transcript");
        }
        std::vector<float> audio = naming("references", [&] { return read_recording(s.path, codec.sample_rate()); });
        if (audio.empty()) throw Error(Fault::InvalidArgument, s.path + " holds no audio; give a recording of the speaker", "references");
        VoiceReference r;
        r.speaker = s.speaker.value_or(0);
        r.transcript = *s.transcript;
        r.frames = codec.frames(audio.size());
        r.codes.assign((size_t) (r.frames * codec.codebooks()), 0);
        recordings.push_back({references.size(), std::move(audio)});
        references.push_back(std::move(r));
    }
    for (const VoiceReference & r : references) {
        if (r.speaker >= most) {
            throw Error(Fault::OutOfRange, "the speaker " + std::to_string(r.speaker) + " is past the model's, 0 to " + std::to_string(most - 1), "speaker");
        }
    }
    // The references and the shortest text must fit a prompt, which counts the frames of a recording before its codes
    // stand in place of the zeros. It bounds what the codec is given: the graph of a recording of 1,000 s outgrows
    // ggml's context and aborts.
    try {
        prompt.rows(".", voice_of(references).references, {});
    } catch (const Error & e) {
        if (e.fault() != Fault::OutOfRange) throw;
        throw Error(Fault::OutOfRange, "the references leave a prompt no room for a text; give shorter references", "references");
    }
    for (auto & [at, audio] : recordings) references[at].codes = codec.encode(audio);
    write_voice_file(voice_path, references, model->str("drama3.codec.sha256"), device_kind(backend));
}

}  // namespace drama3
