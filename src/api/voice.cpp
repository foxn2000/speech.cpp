#include <cmath>
#include <memory>
#include <string>

#include "api.h"
#include "backend.h"
#include "error.h"
#include "speech.h"

// The C API's making of voice files: of one reference, and of the voice parameters' references at their loudness or of
// an embedding.

struct speech_voice_params {
    VoiceRecipe recipe;
};

namespace {

/** Makes the voice file at `voice_path` of `recipe` for the model file at `model_path`, on the device of `params`. */
void make_voice(const char * model_path, const VoiceRecipe & recipe, const char * voice_path, const speech_load_params * params) {
    require(model_path, "model_path", "model_path");
    require(voice_path, "voice_path", "voice_path");
    const speech_load_params defaults;
    const speech_load_params & p = params ? *params : defaults;
    const Family & family = naming("model_path", [&]() -> const Family & { return family_of(model_path); });
    if (!family.make_voice) {
        throw ApiError(SPEECH_ERROR_UNSUPPORTED, std::string(model_path) + " is a model of " + family.layout.architecture +
                                                     ", which takes no voices made from recordings",
                       "model_path");
    }
    std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> backend(
        start_device(find_device(p.device), p.threads > 0 ? p.threads : default_threads()), ggml_backend_free);
    family.make_voice(model_path, recipe, voice_path, backend.get());
}

}  // namespace

extern "C" {

speech_status speech_voice_make(const char * model_path, const char * reference_path, const char * voice_path,
                                const speech_load_params * params) {
    return guarded([&] {
        require(reference_path, "reference_path", "reference_path");
        VoiceRecipe recipe;
        recipe.parts.push_back({false, reference_path});
        make_voice(model_path, recipe, voice_path, params);
        return SPEECH_OK;
    });
}

speech_status speech_voice_params_new(speech_voice_params ** params) {
    return guarded([&] {
        require(params, "params");
        *params = new speech_voice_params();
        return SPEECH_OK;
    });
}

void speech_voice_params_free(speech_voice_params * params) {
    delete params;
}

speech_status speech_voice_params_add_reference(speech_voice_params * params, const char * path) {
    return guarded([&] {
        require(params, "params");
        require(path, "path", "references");
        params->recipe.parts.push_back({false, path});
        return SPEECH_OK;
    });
}

speech_status speech_voice_params_set_transcript(speech_voice_params * params, const char * text) {
    return guarded([&] {
        require(params, "params");
        require(text, "text", "transcript");
        if (params->recipe.parts.empty() || params->recipe.parts.back().voice_file) {
            throw ApiError(SPEECH_ERROR_INVALID_ARGUMENT, "a transcript belongs to the reference added last, and none was; add the reference first",
                           "transcript");
        }
        params->recipe.parts.back().transcript = text;
        return SPEECH_OK;
    });
}

speech_status speech_voice_params_set_speaker(speech_voice_params * params, int speaker) {
    return guarded([&] {
        require(params, "params");
        if (params->recipe.parts.empty()) {
            throw ApiError(SPEECH_ERROR_INVALID_ARGUMENT, "a speaker is given to the reference or voice added last, and none was; add it first",
                           "speaker");
        }
        if (speaker < 0) throw ApiError(SPEECH_ERROR_OUT_OF_RANGE, "the speaker is " + std::to_string(speaker) + "; speakers are numbered from 0", "speaker");
        params->recipe.parts.back().speaker = speaker;
        return SPEECH_OK;
    });
}

speech_status speech_voice_params_add_voice(speech_voice_params * params, const char * path) {
    return guarded([&] {
        require(params, "params");
        require(path, "path", "references");
        params->recipe.parts.push_back({true, path});
        return SPEECH_OK;
    });
}

speech_status speech_voice_params_set_loudness(speech_voice_params * params, double lufs) {
    return guarded([&] {
        require(params, "params");
        if (!std::isfinite(lufs)) throw ApiError(SPEECH_ERROR_INVALID_ARGUMENT, "the loudness is not a finite number of LUFS", "loudness");
        params->recipe.normalize = true;
        params->recipe.lufs = lufs;
        return SPEECH_OK;
    });
}

speech_status speech_voice_params_keep_loudness(speech_voice_params * params) {
    return guarded([&] {
        require(params, "params");
        params->recipe.normalize = false;
        params->recipe.lufs.reset();
        return SPEECH_OK;
    });
}

speech_status speech_voice_params_set_embedding(speech_voice_params * params, const float * values, size_t tokens, size_t dim) {
    return guarded([&] {
        require(params, "params");
        require(values, "values", "embedding");
        if (tokens == 0 || dim == 0) throw ApiError(SPEECH_ERROR_INVALID_ARGUMENT, "the embedding has no tokens or no values in a token", "embedding");
        for (size_t i = 0; i < tokens * dim; i++) {
            if (!std::isfinite(values[i])) {
                throw ApiError(SPEECH_ERROR_INVALID_ARGUMENT, "value " + std::to_string(i) + " of the embedding is not a finite number", "embedding");
            }
        }
        params->recipe.embedding.assign(values, values + tokens * dim);
        params->recipe.embedding_tokens = tokens;
        return SPEECH_OK;
    });
}

speech_status speech_voice_make_from(const char * model_path, const speech_voice_params * voice, const char * voice_path,
                                     const speech_load_params * params) {
    return guarded([&] {
        require(voice, "voice");
        const VoiceRecipe & recipe = voice->recipe;
        if (!recipe.embedding.empty() && (!recipe.parts.empty() || !recipe.normalize || recipe.lufs)) {
            throw ApiError(SPEECH_ERROR_INVALID_ARGUMENT,
                           "a voice of an embedding has no references and no loudness; make it of the embedding alone or of references alone",
                           "embedding");
        }
        if (recipe.parts.empty() && recipe.embedding.empty()) {
            throw ApiError(SPEECH_ERROR_INVALID_ARGUMENT,
                           "the voice has no reference and no embedding; add one with speech_voice_params_add_reference() or "
                           "speech_voice_params_set_embedding()",
                           "references");
        }
        make_voice(model_path, voice->recipe, voice_path, params);
        return SPEECH_OK;
    });
}

}  // extern "C"
