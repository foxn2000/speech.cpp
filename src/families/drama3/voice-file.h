#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "ggml-backend.h"
#include "model-file.h"
#include "synthesizer.h"

namespace drama3 {

/** The general.architecture and the speech.layout of a voice file, which write_voice_file() writes. */
constexpr const char * kVoiceArchitecture = "drama3-voice";
constexpr uint32_t kVoiceLayout = 1;

/** The release whose reader first takes voice files of kVoiceLayout, which a voice file names as speech.requires. */
constexpr const char * kVoiceLayoutRequires = "0.8.8";

/** A reference of a voice: the speaker it belongs to, the words it speaks, and its codes, [codebooks, frames] with the frame fastest. */
struct VoiceReference {
    int speaker = 0;
    std::string transcript;
    std::vector<int32_t> codes;
    int64_t frames = 0;
};

/**
 * Layout 1 of a voice file for the model file `model`, which outlives it: each reference's speaker, transcript, frames
 * and codes, in the order the prompt lists them, the hash of the codec that encoded them and the kind of device that
 * did, all in its metadata, with no tensor. A voice file of a codec other than the model's is refused as the caller's
 * mistake.
 */
Layout voice_layout(const ModelFile & model);

/** What a voice is made of: a reference recording with its transcript, or a voice file, and the speaker it is given. */
struct VoiceSource {
    bool voice_file = false;
    std::string path;
    std::optional<std::string> transcript;
    std::optional<int> speaker;
};

/**
 * Makes the voice file at `voice_path` for the model file at `model_path`, reading only the codec from it and encoding
 * each recording on `backend`: a recording, mixed to mono and resampled to the codec's rate, with its transcript, as
 * speaker 0 unless it is given another; a voice file's references as they are, under the speaker given where the file
 * has one speaker. A recording without a transcript, a transcript with a speaker tag or a token only the prompt writes,
 * a speaker past the model's, and references that leave a prompt no room for a text are refused.
 */
void make_voice_file(const std::string & model_path, const std::vector<VoiceSource> & sources, const std::string & voice_path,
                     ggml_backend_t backend);

/** The recording of the WAVE file at `path`, mixed to mono and resampled to `rate` by torchaudio's default filter. */
std::vector<float> read_recording(const std::string & path, int rate);

/** The references of the voice file at `path`, read against the model `model`. */
std::vector<VoiceReference> read_voice_file(const std::string & path, const ModelFile & model);

/** The voice of `references` as a prompt lists it: each transcript after its speaker's tag. */
Voice voice_of(const std::vector<VoiceReference> & references);

/** "speakers 0 and 2", what the model information says of a voice of `references`. */
std::string describe_speakers(const std::vector<VoiceReference> & references);

}  // namespace drama3
