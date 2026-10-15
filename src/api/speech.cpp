#include "speech.h"

#include <cctype>
#include <cstring>
#include <functional>
#include <memory>
#include <mutex>
#include <new>
#include <stdexcept>
#include <string>
#include <utility>

#include "api.h"
#include "backend.h"
#include "drama3/layout.h"
#include "error.h"
#include "fastconformer/layout.h"
#include "irodori-tts/layout.h"
#include "log.h"
#include "quantize.h"
#include "qwen3-asr/layout.h"
#include "qwen3-tts/layout.h"
#include "silero-vad/layout.h"

// The C API's versions, statuses and errors, log, devices, load parameters, the table of families, the quantizing of
// model files, and the models with their voices. The model information is in info.cpp, the requests in request.cpp
// and the making of voice files in voice.cpp.

namespace {

/**
 * The families speech.cpp runs: the task of each, the layout its reader takes, which names its general.architecture,
 * its table of options with what else its information says, its engine, and how it makes a voice file where it takes
 * them. Adding a family is adding its line.
 */
const Family families[] = {
    {SPEECH_TASK_SYNTHESIS, qwen3_tts_layout, describe_qwen3_tts, load_qwen3_tts, nullptr},
    {SPEECH_TASK_SYNTHESIS, irodori::model_layout, describe_irodori_tts, load_irodori_tts, make_irodori_tts_voice},
    {SPEECH_TASK_SYNTHESIS, drama3::model_layout, describe_drama3, load_drama3, make_drama3_voice},
    {SPEECH_TASK_RECOGNITION, fastconformer::layout, describe_fastconformer, load_fastconformer, nullptr},
    {SPEECH_TASK_RECOGNITION, qwen3_asr::layout, describe_qwen3_asr, load_qwen3_asr, nullptr},
    {SPEECH_TASK_DETECTION, silero_vad::layout, describe_silero_vad, load_silero_vad, nullptr},
};

thread_local std::string last_error;
thread_local std::string last_input;
thread_local bool has_input = false;

speech_status status_of(Fault fault) {
    switch (fault) {
        case Fault::InvalidArgument: return SPEECH_ERROR_INVALID_ARGUMENT;
        case Fault::OutOfRange: return SPEECH_ERROR_OUT_OF_RANGE;
        case Fault::File: return SPEECH_ERROR_MODEL_FILE;
        case Fault::Device: return SPEECH_ERROR_DEVICE;
        case Fault::OutOfMemory: return SPEECH_ERROR_OUT_OF_MEMORY;
        case Fault::Io: return SPEECH_ERROR_IO;
    }
    return SPEECH_ERROR_INTERNAL;
}

struct OptionName {
    const char * name;
    speech_type type;
};

/** The vocabulary of options, by their value. */
const OptionName vocabulary[] = {
    {"voice", SPEECH_TYPE_STRING},  {"language", SPEECH_TYPE_STRING},      {"seed", SPEECH_TYPE_INT},
    {"speed", SPEECH_TYPE_FLOAT},   {"seconds", SPEECH_TYPE_FLOAT},        {"duration_scale", SPEECH_TYPE_FLOAT},
    {"steps", SPEECH_TYPE_INT},     {"max_seconds", SPEECH_TYPE_FLOAT},    {"timestamps", SPEECH_TYPE_BOOL},
    {"prompt", SPEECH_TYPE_STRING}, {"decoding", SPEECH_TYPE_STRING},
    {"do_sample", SPEECH_TYPE_BOOL},
    {"top_k", SPEECH_TYPE_INT},
    {"top_p", SPEECH_TYPE_FLOAT},
    {"temperature", SPEECH_TYPE_FLOAT},
    {"repetition_penalty", SPEECH_TYPE_FLOAT},
    {"code_predictor_do_sample", SPEECH_TYPE_BOOL},
    {"code_predictor_top_k", SPEECH_TYPE_INT},
    {"code_predictor_top_p", SPEECH_TYPE_FLOAT},
    {"code_predictor_temperature", SPEECH_TYPE_FLOAT},
    {"instructions", SPEECH_TYPE_STRING},
    {"cfg_scale_text", SPEECH_TYPE_FLOAT},
    {"cfg_scale_speaker", SPEECH_TYPE_FLOAT},
    {"cfg_guidance_mode", SPEECH_TYPE_STRING},
    {"cfg_min_t", SPEECH_TYPE_FLOAT},
    {"cfg_max_t", SPEECH_TYPE_FLOAT},
    {"truncation_factor", SPEECH_TYPE_FLOAT},
    {"rescale_k", SPEECH_TYPE_FLOAT},
    {"rescale_sigma", SPEECH_TYPE_FLOAT},
    {"speaker_uncond_mode", SPEECH_TYPE_STRING},
    {"sway_coeff", SPEECH_TYPE_FLOAT},
    {"keep_tail", SPEECH_TYPE_BOOL},
    {"tail_window_size", SPEECH_TYPE_INT},
    {"tail_std_threshold", SPEECH_TYPE_FLOAT},
    {"tail_mean_threshold", SPEECH_TYPE_FLOAT},
    {"speaker_kv_scale", SPEECH_TYPE_FLOAT},
    {"speaker_kv_min_t", SPEECH_TYPE_FLOAT},
    {"speaker_kv_max_layers", SPEECH_TYPE_INT},
    {"cfg_scale_instructions", SPEECH_TYPE_FLOAT},
    {"threshold", SPEECH_TYPE_FLOAT},
    {"min_speech_duration_ms", SPEECH_TYPE_INT},
    {"min_silence_duration_ms", SPEECH_TYPE_INT},
    {"speech_pad_ms", SPEECH_TYPE_INT},
    {"max_speech_duration_s", SPEECH_TYPE_FLOAT},
    {"decode_on_cpu", SPEECH_TYPE_BOOL},
    {"draft", SPEECH_TYPE_STRING},
};
constexpr size_t kOptions = sizeof vocabulary / sizeof vocabulary[0];

const char * const status_names[] = {"internal", "io", "out_of_memory", "device", "model_file", "out_of_range", "unsupported",
                                     "invalid_argument", "ok", "cancelled"};

const char * const stop_names[] = {"complete", "max_seconds", "model_limit", "cancelled"};

/** The weight type speech_quantize() writes that `name` names, compared without case. */
const WeightType & quantized_type(const std::string & name) {
    const auto lower = [](std::string s) {
        for (char & c : s) c = (char) std::tolower((unsigned char) c);
        return s;
    };
    std::string names;
    for (const WeightType & t : weight_types()) {
        if (t.type == GGML_TYPE_F32) continue;
        if (lower(name) == lower(tensor_type_text(t.type))) return t;
        names += (names.empty() ? "" : ", ") + tensor_type_text(t.type);
    }
    throw ApiError(SPEECH_ERROR_INVALID_ARGUMENT, "\"" + name + "\" is not a type speech quantize writes; give one of " + names, "type");
}

ggml_backend_dev_t device_at(size_t index) {
    const auto & list = devices();
    if (index >= list.size()) {
        throw ApiError(SPEECH_ERROR_INVALID_ARGUMENT,
                       "there is no device " + std::to_string(index) + "; speech_device_count() gives " + std::to_string(list.size()));
    }
    return list[index];
}

}  // namespace

speech_stop Engine::speak(const std::string &, const RequestValues &, Run &) {
    throw std::logic_error("the family's engine does not speak, though the table of families says it does");
}

Recognized Engine::transcribe(const std::vector<float> &, const RequestValues &, Run &) {
    throw std::logic_error("the family's engine does not recognize speech, though the table of families says it does");
}

std::vector<TimedText> Engine::detect(const std::vector<float> &, const RequestValues &, Run &) {
    throw std::logic_error("the family's engine does not detect speech, though the table of families says it does");
}

std::unique_ptr<DetectionStream> Engine::start_detection(const RequestValues &) {
    throw std::logic_error("the family's engine does not detect speech, though the table of families says it does");
}

std::string Engine::add_voice(const std::string &, const std::string &) {
    throw std::logic_error("the family's engine takes no voice files, though the table of families says it does");
}

speech_status record_failure() {
    speech_status status = SPEECH_ERROR_INTERNAL;
    std::string message, input;
    try {
        throw;
    } catch (const ApiError & e) {
        status = e.status();
        message = e.what();
        input = e.input();
    } catch (const Error & e) {
        status = status_of(e.fault());
        message = e.what();
        input = e.input();
    } catch (const std::bad_alloc &) {
        status = SPEECH_ERROR_OUT_OF_MEMORY;
        message = "the memory of the host ran out";
    } catch (const std::invalid_argument & e) {
        status = SPEECH_ERROR_INVALID_ARGUMENT;
        message = e.what();
    } catch (const std::exception & e) {
        message = e.what();
    } catch (...) {
        message = "an unknown C++ exception";
    }
    last_error = message;
    last_input = input;
    has_input = !input.empty();
    return status;
}

const Family & family_of(const std::string & path) {
    const std::string architecture = gguf_architecture(path);
    for (const Family & f : families) {
        if (architecture == f.layout.architecture) return f;
    }
    std::string known;
    for (const Family & f : families) known += (known.empty() ? "" : ", ") + std::string(f.layout.architecture);
    throw Error(Fault::File, path + " is a model of " + architecture + ", which speech.cpp does not run; it runs " + known);
}

const OptionSpec * FileInfo::spec(speech_option option) const {
    for (const OptionSpec & s : described.options) {
        if (s.option == option) return &s;
    }
    return nullptr;
}

const Lack * FileInfo::lack(speech_option option, const std::string & voice) const {
    for (const Lack & l : described.lacks) {
        if (l.option == option && l.voice == voice) return &l;
    }
    return nullptr;
}

const char * FileInfo::meta_key(size_t index) const {
    std::lock_guard<std::mutex> lock(meta_mutex_);
    if (meta_keys_.empty()) meta_keys_.resize(file->meta_count());
    if (index >= meta_keys_.size()) return nullptr;
    if (!meta_keys_[index]) meta_keys_[index] = std::make_unique<std::string>(file->meta_key(index));
    return meta_keys_[index]->c_str();
}

const char * FileInfo::meta_value(size_t index) const {
    std::lock_guard<std::mutex> lock(meta_mutex_);
    if (meta_values_.empty()) meta_values_.resize(file->meta_count());
    if (index >= meta_values_.size()) return nullptr;
    if (!meta_values_[index]) meta_values_[index] = std::make_unique<std::string>(file->meta_json(index));
    return meta_values_[index]->c_str();
}

std::shared_ptr<const FileInfo> read_file_info(const std::string & path) {
    const Family & family = family_of(path);
    auto info = std::make_shared<FileInfo>();
    info->family = &family;
    info->file = std::make_shared<const ModelFile>(path, family.layout);
    const ModelFile & m = *info->file;
    info->identity = read_identity(m);
    info->sample_rate = (int) m.u32("speech.sample_rate");
    if (task_has_languages(family.task)) info->languages = m.str_array("general.languages");
    info->described = family.describe(info->file);
    for (size_t i = 1; i < info->described.options.size(); i++) {
        if (info->described.options[i - 1].option >= info->described.options[i].option) {
            throw std::logic_error(std::string("the options of ") + family.layout.architecture + " are not in the order of the vocabulary");
        }
    }
    if (info->described.voice_codec.empty() != !family.make_voice || (info->described.voice_recordings && !family.make_voice)) {
        throw std::logic_error(std::string("the family ") + family.layout.architecture +
                               " names a voice codec or takes recordings without making voice files, or makes them without a codec");
    }
    info->file_bytes = m.file_bytes();
    info->weight_bytes = m.weight_bytes();
    return info;
}

bool speech_model::has_voice(const std::string & name) const {
    for (const VoiceInfo & v : file->described.voices) {
        if (v.name == name) return true;
    }
    std::lock_guard<std::mutex> lock(voices_mutex_);
    for (const VoiceInfo & v : added_) {
        if (v.name == name) return true;
    }
    return false;
}

extern "C" {

int speech_api_version_major(void) {
    return SPEECH_API_VERSION_MAJOR;
}

int speech_api_version_minor(void) {
    return SPEECH_API_VERSION_MINOR;
}

const char * speech_version(void) {
    return SPEECH_VERSION;
}

const char * speech_status_name(speech_status status) {
    const int at = (int) status - SPEECH_ERROR_INTERNAL;
    return at >= 0 && at < (int) (sizeof status_names / sizeof status_names[0]) ? status_names[at] : nullptr;
}

const char * speech_stop_name(speech_stop stop) {
    return (int) stop >= 0 && (int) stop < (int) (sizeof stop_names / sizeof stop_names[0]) ? stop_names[stop] : nullptr;
}

const char * speech_last_error(void) {
    return last_error.c_str();
}

const char * speech_last_error_option(void) {
    return has_input ? last_input.c_str() : nullptr;
}

void speech_log_set(speech_log_callback callback, void * user_data) {
    quietly(0, [&] {
        if (callback) {
            set_log_sink([callback, user_data](LogLevel level, const char * text) { callback((speech_log_level) level, text, user_data); });
        } else {
            set_log_sink({});
        }
        return 0;
    });
}

size_t speech_device_count(void) {
    return quietly<size_t>(0, [] { return devices().size(); });
}

const char * speech_device_name(size_t index) {
    return quietly<const char *>(nullptr, [&] { return index < devices().size() ? ggml_backend_dev_name(devices()[index]) : nullptr; });
}

const char * speech_device_description(size_t index) {
    return quietly<const char *>(nullptr, [&] { return index < devices().size() ? ggml_backend_dev_description(devices()[index]) : nullptr; });
}

speech_status speech_device_get_kind(size_t index, speech_device_kind * kind) {
    return guarded([&] {
        require(kind, "kind");
        switch (ggml_backend_dev_type(device_at(index))) {
            case GGML_BACKEND_DEVICE_TYPE_CPU: *kind = SPEECH_DEVICE_CPU; break;
            case GGML_BACKEND_DEVICE_TYPE_GPU: *kind = SPEECH_DEVICE_GPU; break;
            default: *kind = SPEECH_DEVICE_IGPU; break;
        }
        return SPEECH_OK;
    });
}

speech_status speech_device_memory(size_t index, uint64_t * total, uint64_t * free_bytes) {
    return guarded([&] {
        require(total, "total");
        require(free_bytes, "free_bytes");
        size_t free = 0, all = 0;
        ggml_backend_dev_memory(device_at(index), &free, &all);
        *total = all;
        *free_bytes = free;
        return SPEECH_OK;
    });
}

size_t speech_option_count(void) {
    return kOptions;
}

const char * speech_option_name(speech_option option) {
    return (size_t) option < kOptions ? vocabulary[option].name : nullptr;
}

speech_status speech_option_from_name(const char * name, speech_option * option) {
    return guarded([&] {
        require(name, "name");
        require(option, "option");
        std::string names;
        for (size_t i = 0; i < kOptions; i++) {
            if (std::strcmp(name, vocabulary[i].name) == 0) {
                *option = (speech_option) i;
                return SPEECH_OK;
            }
            names += (i ? ", " : "") + std::string(vocabulary[i].name);
        }
        throw ApiError(SPEECH_ERROR_INVALID_ARGUMENT, std::string("no option is named \"") + name + "\"; the options are " + names);
    });
}

speech_type speech_option_type(speech_option option) {
    return (size_t) option < kOptions ? vocabulary[option].type : (speech_type) -1;
}

speech_status speech_load_params_new(speech_load_params ** params) {
    return guarded([&] {
        require(params, "params");
        *params = new speech_load_params();
        return SPEECH_OK;
    });
}

void speech_load_params_free(speech_load_params * params) {
    delete params;
}

speech_status speech_load_params_set_device(speech_load_params * params, const char * device) {
    return guarded([&] {
        require(params, "params");
        require(device, "device", "device");
        if (!*device) {
            throw ApiError(SPEECH_ERROR_INVALID_ARGUMENT, "the device is empty; give \"auto\", \"gpu\", \"cpu\" or a device's name", "device");
        }
        params->device = device;
        return SPEECH_OK;
    });
}

speech_status speech_load_params_set_threads(speech_load_params * params, int threads) {
    return guarded([&] {
        require(params, "params");
        if (threads < 1) {
            throw ApiError(SPEECH_ERROR_INVALID_ARGUMENT, "the threads are " + std::to_string(threads) + "; give 1 or more", "threads");
        }
        params->threads = threads;
        return SPEECH_OK;
    });
}

speech_status speech_load_params_set_warmup(speech_load_params * params, int warmup) {
    return guarded([&] {
        require(params, "params");
        params->warmup = warmup != 0;
        return SPEECH_OK;
    });
}

speech_status speech_model_load(const char * path, const speech_load_params * params, speech_model ** model) {
    if (model) *model = nullptr;
    return guarded([&] {
        require(path, "path");
        require(model, "model");
        const speech_load_params defaults;
        const speech_load_params & p = params ? *params : defaults;
        auto m = std::make_unique<speech_model>();
        m->file = read_file_info(path);
        ggml_backend_dev_t device = find_device(p.device);
        const int threads = p.threads > 0 ? p.threads : default_threads();
        m->backend.reset(start_device(device, threads));
        m->engine = m->file->family->load(path, m->backend.get(), threads);
        m->device = ggml_backend_dev_name(device);
        m->threads = ggml_backend_dev_type(device) == GGML_BACKEND_DEVICE_TYPE_CPU || m->engine->computes_on_cpu() ? threads : 0;
        if (p.warmup) m->engine->warm_up();
        *model = m.release();
        return SPEECH_OK;
    });
}

void speech_model_free(speech_model * model) {
    delete model;
}

speech_status speech_model_get_info(const speech_model * model, speech_model_info ** info) {
    if (info) *info = nullptr;
    return guarded([&] {
        require(model, "model");
        require(info, "info");
        *info = make_info(model->file, model->added(), model->device, model->threads);
        return SPEECH_OK;
    });
}

speech_status speech_quantize(const char * model_path, const char * type, const char * out_path) {
    return guarded([&] {
        require(model_path, "model_path", "model_path");
        require(type, "type", "type");
        require(out_path, "out_path", "out_path");
        const WeightType & target = quantized_type(type);
        const Family & family = naming("model_path", [&]() -> const Family & { return family_of(model_path); });
        quantize_model_file(model_path, out_path, target, family.layout);
        return SPEECH_OK;
    });
}

speech_status speech_voice_add(speech_model * model, const char * name, const char * path) {
    return guarded([&] {
        require(model, "model");
        require(name, "name", "name");
        require(path, "path", "path");
        const FileInfo & file = *model->file;
        if (!file.family->make_voice) {
            throw ApiError(SPEECH_ERROR_UNSUPPORTED, file.identity.name + " takes no voices made from recordings; " +
                                                         (file.family->task == SPEECH_TASK_SYNTHESIS ? std::string("it speaks with its own voices")
                                                                                                     : "it is a model of speech " + std::string(task_name(file.family->task))));
        }
        if (!*name) throw ApiError(SPEECH_ERROR_INVALID_ARGUMENT, "the voice's name is empty; give the voice a name", "name");
        std::lock_guard<std::mutex> lock(model->busy);
        if (model->has_voice(name)) {
            throw ApiError(SPEECH_ERROR_INVALID_ARGUMENT, std::string(name) + " is already a voice of " + file.identity.name + "; give the voice another name",
                           "name");
        }
        if (file.lack(SPEECH_OPT_VOICE, name)) {
            throw ApiError(SPEECH_ERROR_INVALID_ARGUMENT, std::string(name) + " is the name of a voice that other files of " + file.family->layout.architecture +
                                                              " have; give the voice another name",
                           "name");
        }
        const std::string description = naming("path", [&] { return model->engine->add_voice(name, path); });
        model->add({name, "", "", description});
        return SPEECH_OK;
    });
}

}  // extern "C"
