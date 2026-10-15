#include <cmath>
#include <string>
#include <utility>
#include <variant>

#include "api.h"
#include "json.h"
#include "speech.h"

// The model information: what a model file says of its model, with the device, the threads and the added voices of a
// loaded model, through its accessors and as the one JSON object every program that shows it carries.

namespace {

const char * type_name(speech_type type) {
    switch (type) {
        case SPEECH_TYPE_STRING: return "string";
        case SPEECH_TYPE_INT: return "int";
        case SPEECH_TYPE_FLOAT: return "float";
        case SPEECH_TYPE_BOOL: return "bool";
    }
    return "";
}

std::string value_json(const OptionValue & v) {
    switch (v.index()) {
        case 0: return json_string(std::get<std::string>(v));
        case 1: return std::to_string(std::get<int64_t>(v));
        case 2: return json_number(std::get<double>(v));
        default: return std::get<bool>(v) ? "true" : "false";
    }
}

/** A bound of an integer option as an integer, of a number option as the shortest number that reads back as it. */
std::string bound_json(double v, speech_type type) {
    return type == SPEECH_TYPE_INT ? std::to_string((int64_t) v) : json_number(v);
}

template <typename Get>
std::string string_array(size_t n, Get get) {
    std::string out = "[";
    for (size_t i = 0; i < n; i++) out += (i ? "," : "") + json_string(get(i));
    return out + "]";
}

/**
 * The values a string option takes: the voices for voice, the languages for language and the choices the family
 * declares for another, or none for an option without them.
 */
const std::vector<std::string> * choices(const speech_model_info & info, speech_option option, std::vector<std::string> & voices) {
    if (option == SPEECH_OPT_LANGUAGE) return &info.file->languages;
    if (option == SPEECH_OPT_VOICE) {
        voices.clear();
        for (const VoiceInfo & v : info.voices) voices.push_back(v.name);
        return &voices;
    }
    const OptionSpec * s = info.file->spec(option);
    return s && !s->choices.empty() ? &s->choices : nullptr;
}

/** The members of the information in the order the JSON form gives them; a member that does not apply is left out. */
std::string info_json(const speech_model_info & info) {
    const FileInfo & f = *info.file;
    const FamilyInfo & d = f.described;
    const bool synthesis = f.family->task == SPEECH_TASK_SYNTHESIS;
    const ModelIdentity & id = f.identity;
    std::string out = "{\"name\":" + json_string(id.name) + ",\"organization\":" + json_string(id.organization) +
                      ",\"basename\":" + json_string(id.basename) + ",\"size_label\":" + json_string(id.size_label);
    if (id.finetune) out += ",\"finetune\":" + json_string(*id.finetune);
    if (id.version) out += ",\"version\":" + json_string(*id.version);
    out += ",\"license\":" + json_string(id.license) + ",\"source\":{\"repository\":" + json_string(id.repository) +
           ",\"revision\":" + json_string(id.revision) + "},\"weight_type\":" + json_string(id.weight_type) +
           ",\"architecture\":" + json_string(f.family->layout.architecture) + ",\"layout\":" + std::to_string(f.file->layout_version()) +
           ",\"task\":\"" + task_name(f.family->task) + "\",\"sample_rate\":" + std::to_string(f.sample_rate);
    if (synthesis) out += std::string(",\"incremental\":") + (d.incremental ? "true" : "false");
    out += ",\"languages\":" + string_array(f.languages.size(), [&](size_t i) { return f.languages[i]; });
    if (synthesis) {
        out += ",\"voices\":[";
        for (size_t i = 0; i < info.voices.size(); i++) {
            const VoiceInfo & v = info.voices[i];
            out += std::string(i ? "," : "") + "{\"name\":" + json_string(v.name) + ",\"language\":" + json_string(v.language) +
                   ",\"gender\":" + json_string(v.gender) + ",\"description\":" + json_string(v.description) + "}";
        }
        out += "]";
    }
    out += std::string(",\"voice_files\":") + (f.family->make_voice ? "true" : "false");
    if (f.family->make_voice) out += ",\"voice_codec\":" + json_string(d.voice_codec);
    out += std::string(",\"voice_recordings\":") + (d.voice_recordings ? "true" : "false");
    if (synthesis) out += ",\"max_text_tokens\":" + std::to_string(d.max_text_tokens);
    out += ",\"options\":[";
    std::vector<std::string> voices;
    for (size_t i = 0; i < d.options.size(); i++) {
        const OptionSpec & s = d.options[i];
        const speech_type type = speech_option_type(s.option);
        out += std::string(i ? "," : "") + "{\"name\":" + json_string(speech_option_name(s.option)) + ",\"type\":\"" + type_name(type) +
               "\",\"required\":" + (s.required ? "true" : "false") + ",\"steers\":" + (s.steers ? "true" : "false");
        if (s.default_value) out += ",\"default\":" + value_json(*s.default_value);
        if (std::isfinite(s.minimum)) out += std::string(s.minimum_exclusive ? ",\"exclusive_minimum\":" : ",\"minimum\":") + bound_json(s.minimum, type);
        if (std::isfinite(s.maximum)) out += ",\"maximum\":" + bound_json(s.maximum, type);
        if (const std::vector<std::string> * c = choices(info, s.option, voices)) {
            out += ",\"choices\":" + string_array(c->size(), [&](size_t k) { return (*c)[k]; });
        }
        out += "}";
    }
    out += "],\"file_bytes\":" + std::to_string(f.file_bytes) + ",\"weight_bytes\":" + std::to_string(f.weight_bytes);
    if (info.device) out += ",\"device\":" + json_string(*info.device) + ",\"threads\":" + std::to_string(info.threads);
    return out + "}";
}

/** The declaration of an option the model takes, of type `type`; another option is an error. */
const OptionSpec & taken(const speech_model_info * info, speech_option option, speech_type type) {
    require(info, "info");
    const OptionSpec * s = info->file->spec(option);
    const char * name = speech_option_name(option);
    if (!s) {
        throw ApiError(SPEECH_ERROR_UNSUPPORTED, info->file->identity.name + " does not take the option " + (name ? name : std::to_string(option)), name);
    }
    if (speech_option_type(option) != type) {
        throw ApiError(SPEECH_ERROR_INVALID_ARGUMENT, std::string("the option ") + name + " is of the type " + type_name(speech_option_type(option)) +
                                                          ", not " + type_name(type),
                       name);
    }
    return *s;
}

template <typename T>
speech_status default_of(const speech_model_info * info, speech_option option, speech_type type, T * value) {
    return guarded([&] {
        const OptionSpec & s = taken(info, option, type);
        require(value, "value");
        if (!s.default_value) {
            throw ApiError(SPEECH_ERROR_INVALID_ARGUMENT, std::string("the option ") + speech_option_name(option) + " has no default",
                           speech_option_name(option));
        }
        *value = std::get<T>(*s.default_value);
        return SPEECH_OK;
    });
}

const OptionSpec * spec_of(const speech_model_info * info, speech_option option) {
    return info ? info->file->spec(option) : nullptr;
}

}  // namespace

speech_model_info * make_info(std::shared_ptr<const FileInfo> file, std::vector<VoiceInfo> added, std::optional<std::string> device,
                              int threads) {
    auto info = std::make_unique<speech_model_info>();
    info->voices = file->described.voices;
    info->voices.insert(info->voices.end(), added.begin(), added.end());
    info->file = std::move(file);
    info->device = std::move(device);
    info->threads = threads;
    info->json = info_json(*info);
    return info.release();
}

extern "C" {

speech_status speech_model_info_open(const char * path, speech_model_info ** info) {
    if (info) *info = nullptr;
    return guarded([&] {
        require(path, "path");
        require(info, "info");
        *info = make_info(read_file_info(path), {}, std::nullopt, 0);
        return SPEECH_OK;
    });
}

void speech_model_info_free(speech_model_info * info) {
    delete info;
}

const char * speech_model_info_name(const speech_model_info * info) {
    return info ? info->file->identity.name.c_str() : nullptr;
}

const char * speech_model_info_organization(const speech_model_info * info) {
    return info ? info->file->identity.organization.c_str() : nullptr;
}

const char * speech_model_info_basename(const speech_model_info * info) {
    return info ? info->file->identity.basename.c_str() : nullptr;
}

const char * speech_model_info_size_label(const speech_model_info * info) {
    return info ? info->file->identity.size_label.c_str() : nullptr;
}

const char * speech_model_info_finetune(const speech_model_info * info) {
    return info && info->file->identity.finetune ? info->file->identity.finetune->c_str() : nullptr;
}

const char * speech_model_info_version(const speech_model_info * info) {
    return info && info->file->identity.version ? info->file->identity.version->c_str() : nullptr;
}

const char * speech_model_info_license(const speech_model_info * info) {
    return info ? info->file->identity.license.c_str() : nullptr;
}

speech_status speech_model_info_source(const speech_model_info * info, const char ** repository, const char ** revision) {
    return guarded([&] {
        require(info, "info");
        require(repository, "repository");
        require(revision, "revision");
        *repository = info->file->identity.repository.c_str();
        *revision = info->file->identity.revision.c_str();
        return SPEECH_OK;
    });
}

const char * speech_model_info_weight_type(const speech_model_info * info) {
    return info ? info->file->identity.weight_type.c_str() : nullptr;
}

const char * speech_model_info_architecture(const speech_model_info * info) {
    return info ? info->file->family->layout.architecture : nullptr;
}

uint32_t speech_model_info_layout(const speech_model_info * info) {
    return info ? info->file->file->layout_version() : 0;
}

speech_task speech_model_info_task(const speech_model_info * info) {
    return info ? info->file->family->task : SPEECH_TASK_SYNTHESIS;
}

int speech_model_info_sample_rate(const speech_model_info * info) {
    return info ? info->file->sample_rate : 0;
}

int speech_model_info_incremental(const speech_model_info * info) {
    return info && info->file->described.incremental ? 1 : 0;
}

size_t speech_model_info_language_count(const speech_model_info * info) {
    return info ? info->file->languages.size() : 0;
}

const char * speech_model_info_language(const speech_model_info * info, size_t index) {
    return info && index < info->file->languages.size() ? info->file->languages[index].c_str() : nullptr;
}

size_t speech_model_info_voice_count(const speech_model_info * info) {
    return info ? info->voices.size() : 0;
}

const char * speech_model_info_voice_name(const speech_model_info * info, size_t index) {
    return info && index < info->voices.size() ? info->voices[index].name.c_str() : nullptr;
}

const char * speech_model_info_voice_language(const speech_model_info * info, size_t index) {
    return info && index < info->voices.size() ? info->voices[index].language.c_str() : nullptr;
}

const char * speech_model_info_voice_gender(const speech_model_info * info, size_t index) {
    return info && index < info->voices.size() ? info->voices[index].gender.c_str() : nullptr;
}

const char * speech_model_info_voice_description(const speech_model_info * info, size_t index) {
    return info && index < info->voices.size() ? info->voices[index].description.c_str() : nullptr;
}

int speech_model_info_voice_files(const speech_model_info * info) {
    return info && info->file->family->make_voice ? 1 : 0;
}

const char * speech_model_info_voice_codec(const speech_model_info * info) {
    return info && info->file->family->make_voice ? info->file->described.voice_codec.c_str() : nullptr;
}

int speech_model_info_voice_recordings(const speech_model_info * info) {
    return info && info->file->described.voice_recordings ? 1 : 0;
}

size_t speech_model_info_max_text_tokens(const speech_model_info * info) {
    return info ? info->file->described.max_text_tokens : 0;
}

speech_status speech_model_info_text_tokens(const speech_model_info * info, const char * text, size_t * n_tokens) {
    return guarded([&] {
        require(info, "info");
        require(text, "text", "text");
        require(n_tokens, "n_tokens");
        if (!info->file->described.count_tokens) {
            throw ApiError(SPEECH_ERROR_UNSUPPORTED, info->file->identity.name + " is a model of speech " + task_name(info->file->family->task) +
                                                         " and has no text to count");
        }
        *n_tokens = info->file->described.count_tokens(text);
        return SPEECH_OK;
    });
}

size_t speech_model_info_option_count(const speech_model_info * info) {
    return info ? info->file->described.options.size() : 0;
}

speech_option speech_model_info_option(const speech_model_info * info, size_t index) {
    return info && index < info->file->described.options.size() ? info->file->described.options[index].option : (speech_option) 0;
}

int speech_model_info_takes(const speech_model_info * info, speech_option option) {
    return spec_of(info, option) ? 1 : 0;
}

int speech_model_info_option_required(const speech_model_info * info, speech_option option) {
    const OptionSpec * s = spec_of(info, option);
    return s && s->required ? 1 : 0;
}

int speech_model_info_option_steers(const speech_model_info * info, speech_option option) {
    const OptionSpec * s = spec_of(info, option);
    return s && s->steers ? 1 : 0;
}

int speech_model_info_option_has_default(const speech_model_info * info, speech_option option) {
    const OptionSpec * s = spec_of(info, option);
    return s && s->default_value ? 1 : 0;
}

speech_status speech_model_info_option_default_string(const speech_model_info * info, speech_option option, const char ** value) {
    return guarded([&] {
        const OptionSpec & s = taken(info, option, SPEECH_TYPE_STRING);
        require(value, "value");
        if (!s.default_value) {
            throw ApiError(SPEECH_ERROR_INVALID_ARGUMENT, std::string("the option ") + speech_option_name(option) + " has no default",
                           speech_option_name(option));
        }
        *value = std::get<std::string>(*s.default_value).c_str();
        return SPEECH_OK;
    });
}

speech_status speech_model_info_option_default_int(const speech_model_info * info, speech_option option, int64_t * value) {
    return default_of(info, option, SPEECH_TYPE_INT, value);
}

speech_status speech_model_info_option_default_float(const speech_model_info * info, speech_option option, double * value) {
    return default_of(info, option, SPEECH_TYPE_FLOAT, value);
}

speech_status speech_model_info_option_default_bool(const speech_model_info * info, speech_option option, int * value) {
    return guarded([&] {
        bool b = false;
        const speech_status status = default_of(info, option, SPEECH_TYPE_BOOL, &b);
        if (status != SPEECH_OK) return status;
        require(value, "value");
        *value = b ? 1 : 0;
        return SPEECH_OK;
    });
}

speech_status speech_model_info_option_range(const speech_model_info * info, speech_option option, double * minimum, double * maximum,
                                             int * minimum_exclusive) {
    return guarded([&] {
        require(info, "info");
        const speech_type type = speech_option_type(option);
        if (type != SPEECH_TYPE_INT && type != SPEECH_TYPE_FLOAT) {
            const char * name = speech_option_name(option);
            throw ApiError(SPEECH_ERROR_INVALID_ARGUMENT, std::string("the option ") + (name ? name : std::to_string(option)) + " is not a number and has no range",
                           name);
        }
        const OptionSpec & s = taken(info, option, type);
        require(minimum, "minimum");
        require(maximum, "maximum");
        require(minimum_exclusive, "minimum_exclusive");
        *minimum = s.minimum;
        *maximum = s.maximum;
        *minimum_exclusive = s.minimum_exclusive ? 1 : 0;
        return SPEECH_OK;
    });
}

size_t speech_model_info_option_choice_count(const speech_model_info * info, speech_option option) {
    if (!spec_of(info, option)) return 0;
    std::vector<std::string> voices;
    const std::vector<std::string> * c = choices(*info, option, voices);
    return c ? c->size() : 0;
}

const char * speech_model_info_option_choice(const speech_model_info * info, speech_option option, size_t index) {
    const OptionSpec * s = spec_of(info, option);
    if (!s) return nullptr;
    if (option == SPEECH_OPT_VOICE) return index < info->voices.size() ? info->voices[index].name.c_str() : nullptr;
    if (option == SPEECH_OPT_LANGUAGE) return index < info->file->languages.size() ? info->file->languages[index].c_str() : nullptr;
    return index < s->choices.size() ? s->choices[index].c_str() : nullptr;
}

uint64_t speech_model_info_file_bytes(const speech_model_info * info) {
    return info ? info->file->file_bytes : 0;
}

uint64_t speech_model_info_weight_bytes(const speech_model_info * info) {
    return info ? info->file->weight_bytes : 0;
}

const char * speech_model_info_device(const speech_model_info * info) {
    return info && info->device ? info->device->c_str() : nullptr;
}

int speech_model_info_threads(const speech_model_info * info) {
    return info ? info->threads : 0;
}

const char * speech_model_info_json(const speech_model_info * info) {
    return info ? info->json.c_str() : nullptr;
}

size_t speech_model_info_meta_count(const speech_model_info * info) {
    return info ? info->file->file->meta_count() : 0;
}

const char * speech_model_info_meta_key(const speech_model_info * info, size_t index) {
    return info ? quietly<const char *>(nullptr, [&] { return info->file->meta_key(index); }) : nullptr;
}

const char * speech_model_info_meta_value(const speech_model_info * info, size_t index) {
    return info ? quietly<const char *>(nullptr, [&] { return info->file->meta_value(index); }) : nullptr;
}

}  // extern "C"
