#ifndef SPEECH_H
#define SPEECH_H

#include <stddef.h>
#include <stdint.h>

/*
 * The C API of speech.cpp on ggml: speech synthesis, speech recognition and the detection of speech with the models of
 * the families the library runs. A model does one task (speech_model_info_task()): a synthesis model turns the text of
 * a request into audio through speech_synthesize(), a recognition model turns the audio of a request into text through
 * speech_transcribe(), and a detection model finds where someone speaks in the audio of a request through
 * speech_detect().
 *
 * What a request may ask of a model is one vocabulary of options (speech_option), each with a fixed name that the
 * worker, the HTTP server and the command line use verbatim. Each family declares which options it takes, with their
 * type, default and range or choices, and the library checks every value a caller sets against that declaration as
 * it is set. The same declaration is readable before the model is loaded, through speech_model_info_open(), so a
 * caller learns what a model takes without trying. An option a model does not take is an error, except at the
 * option's neutral value, which every model accepts.
 *
 * Every string passed in or returned is UTF-8, paths included. A function that can fail returns a speech_status; on
 * failure speech_last_error() gives the message and speech_last_error_option() the input it concerns. Every object
 * the library returns is freed only through the function named for it, and every string it returns lives as long as
 * the object it was read from unless the declaration says otherwise. Nothing the caller passes in is kept after the
 * call returns.
 *
 * Threads: a model serves one request at a time, so requests that several threads start on the same model run one
 * after another. A request is used by one thread at a time, except for speech_request_cancel(), which any thread may
 * call while the request exists. Model information never changes once made and may be read from any thread.
 * Different models are independent of each other.
 *
 * The library checks what it is given before ggml sees it, so that no input a caller passes makes ggml abort the
 * process.
 */

#ifdef _WIN32
#    if defined(SPEECH_BUILD_SHARED)
#        define SPEECH_API __declspec(dllexport)
#    elif defined(SPEECH_SHARED)
#        define SPEECH_API __declspec(dllimport)
#    else
#        define SPEECH_API
#    endif
#else
#    define SPEECH_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

/**
 * The major version of this API. It rises when a declaration changes in a way an existing caller notices, and a
 * program built against one major version runs only against a library of the same major version. The shared
 * library's SOVERSION is this number.
 */
#define SPEECH_API_VERSION_MAJOR 3

/**
 * The minor version of this API. It rises when a function, an option or an enum value is added, and returns to 0
 * when the major version rises. A program built against minor version m runs against a library of the same major
 * version and a minor version of m or more. Version 3.0 is first released with speech.cpp 0.7.0, the option prompt
 * included, and a declaration added since says the version that added it.
 */
#define SPEECH_API_VERSION_MINOR 3

/** The SPEECH_API_VERSION_MAJOR the library was built with, for a caller that loads it at run time. */
SPEECH_API int speech_api_version_major(void);

/** The SPEECH_API_VERSION_MINOR the library was built with. */
SPEECH_API int speech_api_version_minor(void);

/**
 * The release of speech.cpp the library was built from, such as "0.7.0": MAJOR.MINOR.PATCH under Semantic
 * Versioning, the release's tag without its "v". The string lives as long as the process.
 */
SPEECH_API const char * speech_version(void);

/** What a call that can fail returns. A negative value is an error. */
typedef enum speech_status {
    /** The call did what it was asked. */
    SPEECH_OK = 0,
    /** A request was stopped by speech_request_cancel() or by one of its callbacks returning nonzero. */
    SPEECH_CANCELLED = 1,
    /** The caller's mistake: a NULL pointer, an empty text, a value of the wrong type, a call out of order. */
    SPEECH_ERROR_INVALID_ARGUMENT = -1,
    /** The model cannot do what was asked: an option it does not take, given a value other than the neutral one. */
    SPEECH_ERROR_UNSUPPORTED = -2,
    /**
     * The model takes the option, but not this value: outside its range, not one of its choices, or too long; or a
     * synthesis whose values, each within its range, are together more than the model computes, whose speech came out
     * not finite (speech_synthesize()), naming no option.
     */
    SPEECH_ERROR_OUT_OF_RANGE = -3,
    /**
     * A model or voice file that cannot be used: not GGUF, an architecture no family has, a layout this release does
     * not read, or a key or tensor that is missing, of the wrong type, or with a value the layout does not allow.
     */
    SPEECH_ERROR_MODEL_FILE = -4,
    /** A device that is not there, does not start, or fails while it computes. */
    SPEECH_ERROR_DEVICE = -5,
    /** The memory of the host or of the device ran out. */
    SPEECH_ERROR_OUT_OF_MEMORY = -6,
    /** A file that cannot be opened, read or written. */
    SPEECH_ERROR_IO = -7,
    /** A defect of the library. */
    SPEECH_ERROR_INTERNAL = -8
} speech_status;

/**
 * The name of a status in snake_case, as the worker, the HTTP server and the command line write it: "ok",
 * "cancelled", "invalid_argument", "unsupported", "out_of_range", "model_file", "device", "out_of_memory", "io" or
 * "internal". NULL for a value the library does not know. The string lives as long as the process.
 */
SPEECH_API const char * speech_status_name(speech_status status);

/**
 * The message of the last call on the calling thread that failed, naming what failed and what to do. It stays valid
 * until the next call into the library on the same thread.
 */
SPEECH_API const char * speech_last_error(void);

/**
 * The input the last failed call on the calling thread concerns: an option's name as speech_option_name() gives it,
 * "text" or "audio" for a request's input, "device", "threads" or "warmup" for a load parameter, "name" or "path"
 * for a voice being added, "model_path", "reference_path", "references", "loudness", "embedding" or "voice_path" for
 * a voice file being made ("references", "loudness" and "embedding" added in 3.1), or "model_path", "type" or "out_path"
 * for a model file being quantized (added in 3.1). NULL when the failure concerns no
 * single input. It stays valid until the next call into the library on the same thread.
 */
SPEECH_API const char * speech_last_error_option(void);

/** How much a log message matters. */
typedef enum speech_log_level {
    SPEECH_LOG_DEBUG = 0,
    SPEECH_LOG_INFO = 1,
    SPEECH_LOG_WARN = 2,
    SPEECH_LOG_ERROR = 3
} speech_log_level;

/**
 * Receives the library's log messages and ggml's. `text` is one line or a piece of one, as ggml writes it, and is
 * valid only during the call. It is called on whichever thread is inside the library, so it must be safe to call
 * from several threads at once.
 */
typedef void (*speech_log_callback)(speech_log_level level, const char * text, void * user_data);

/**
 * Sends every later log message to `callback` with `user_data`, or drops every message when `callback` is NULL. Until
 * it is called, warnings and errors go to stderr and the rest is dropped. It holds for the whole process.
 */
SPEECH_API void speech_log_set(speech_log_callback callback, void * user_data);

/*
 * Devices: the CPU and the GPUs that ggml can run a whole model on, in ggml's order. An accelerator that ggml runs
 * beside the CPU, such as BLAS, cannot run a model by itself and is not listed. The first call that touches a device
 * sets ggml up for the process. ggml's Metal backend sets the environment variable AGX_RELAX_CDM_CTXSTORE_TIMEOUT to 1
 * for the process while it lists its devices. Setting the environment is not safe while another thread reads it, so
 * that first call must not run while another thread of the process reads the environment.
 */

/** The kind of a device. */
typedef enum speech_device_kind {
    SPEECH_DEVICE_CPU = 0,
    /** A GPU with memory of its own. */
    SPEECH_DEVICE_GPU = 1,
    /** A GPU that shares the CPU's memory. */
    SPEECH_DEVICE_IGPU = 2
} speech_device_kind;

/** The number of devices the library can run a model on. */
SPEECH_API size_t speech_device_count(void);

/**
 * The name of the device at `index` as a load parameter gives it, such as "MTL0", "Vulkan1" or "CPU", or NULL when
 * `index` is not below speech_device_count(). The string lives as long as the process.
 */
SPEECH_API const char * speech_device_name(size_t index);

/** What the device at `index` is, such as "NVIDIA GeForce RTX 2080", or NULL past the last device. */
SPEECH_API const char * speech_device_description(size_t index);

/** The kind of the device at `index`. An index past the last device is an error. */
SPEECH_API speech_status speech_device_get_kind(size_t index, speech_device_kind * kind);

/** The memory of the device at `index` in bytes: its total, and what is free when it is called. */
SPEECH_API speech_status speech_device_memory(size_t index, uint64_t * total, uint64_t * free_bytes);

/*
 * Options: the vocabulary of what a request may ask, the same for every family. A value is never reused, and a later
 * minor version only appends. Each option has one type. Some have a neutral value that every model accepts, whether
 * or not it takes the option: speed 1, duration_scale 1, language "auto", timestamps false, prompt "" and
 * instructions "". The others have none, and a model that does not take one refuses any value of it.
 */
typedef enum speech_option {
    /**
     * "voice", a string: the name of one of the model's voices, compared with case. A synthesis model that takes it
     * requires it.
     */
    SPEECH_OPT_VOICE = 0,
    /**
     * "language", a string: "auto" (the neutral value), or a BCP 47 tag that names one of the model's languages, by
     * its code of two letters or, for a language that has none, of three, alone or with a region or script ("ja",
     * "ja-JP", "zh-Hant", "yue", "fil-PH"), compared without case. The model's declaration says whether the language
     * steers the model or is only checked against its languages (speech_model_info_option_steers()).
     */
    SPEECH_OPT_LANGUAGE = 1,
    /**
     * "seed", an integer from 0 to 2^53 - 1: the seed of the request's sampling, so that the same request with the
     * same seed on the same device gives the same audio. Without it the library draws one from the same range and
     * reports it in the result.
     */
    SPEECH_OPT_SEED = 2,
    /** "speed", a number: the speaking rate against the model's own, 1 being neutral. */
    SPEECH_OPT_SPEED = 3,
    /** "seconds", a number: the length of the speech in seconds, fixed before the speech is made. */
    SPEECH_OPT_SECONDS = 4,
    /** "duration_scale", a number: the factor of the length the model predicts, 1 being neutral. */
    SPEECH_OPT_DURATION_SCALE = 5,
    /** "steps", an integer: the steps of the model's sampler. */
    SPEECH_OPT_STEPS = 6,
    /**
     * "max_seconds", a number: the longest the speech may be. A synthesis that reaches it stops there and reports
     * SPEECH_STOP_MAX_SECONDS.
     */
    SPEECH_OPT_MAX_SECONDS = 7,
    /**
     * "timestamps", a boolean: whether a recognition's result carries its segments and its tokens with their times,
     * false being neutral.
     */
    SPEECH_OPT_TIMESTAMPS = 8,
    /**
     * "prompt", a string: what a recognition is told of the audio before it hears it, such as the names and terms it may
     * hold, which the model takes as context rather than as instructions; "" (the neutral value) tells it nothing. A
     * prompt too long for the model beside the audio is refused when the request runs, naming the option.
     */
    SPEECH_OPT_PROMPT = 9,
    /**
     * "decoding", a string: how a recognition chooses its tokens from the model's outputs, one of the names the model
     * takes (speech_model_info_option_choice()), compared with case: "beam", a beam search, or "greedy", the most
     * likely token at each step, which takes less time and can write another text. Taken by a model that runs more
     * than one; it has no neutral value. Added in 3.1.
     */
    SPEECH_OPT_DECODING = 10,
    /**
     * "do_sample", a boolean: whether a model that predicts tokens one by one draws each from its distribution (true)
     * or takes the most likely (false), as transformers' generate() names it. A model that has a second stack takes the
     * same of it under the stack's name (SPEECH_OPT_CODE_PREDICTOR_DO_SAMPLE). Setting top_k, top_p or temperature of a
     * stack that does not draw is refused when the request runs, since they would change nothing. Added in 3.1.
     */
    SPEECH_OPT_DO_SAMPLE = 11,
    /** "top_k", an integer: the most likely tokens a draw keeps, 0 keeping every token. Added in 3.1. */
    SPEECH_OPT_TOP_K = 12,
    /**
     * "top_p", a number: the smallest set of the most likely tokens whose probability reaches it, which a draw keeps.
     * Added in 3.1.
     */
    SPEECH_OPT_TOP_P = 13,
    /** "temperature", a number above 0: what the logits are divided by before a draw. Added in 3.1. */
    SPEECH_OPT_TEMPERATURE = 14,
    /**
     * "repetition_penalty", a number above 0: what the logit of a token that the speech has already taken is divided
     * by where it is positive, and multiplied by where it is negative, before a pick. Added in 3.1.
     */
    SPEECH_OPT_REPETITION_PENALTY = 15,
    /**
     * "code_predictor_do_sample", a boolean: do_sample of the code predictor, the second stack of a model that makes
     * each frame of its speech as several codes, which predicts the codes after the frame's first (Qwen3-TTS). Added
     * in 3.1.
     */
    SPEECH_OPT_CODE_PREDICTOR_DO_SAMPLE = 16,
    /** "code_predictor_top_k", an integer: top_k of the code predictor. Added in 3.1. */
    SPEECH_OPT_CODE_PREDICTOR_TOP_K = 17,
    /** "code_predictor_top_p", a number: top_p of the code predictor. Added in 3.1. */
    SPEECH_OPT_CODE_PREDICTOR_TOP_P = 18,
    /** "code_predictor_temperature", a number above 0: temperature of the code predictor. Added in 3.1. */
    SPEECH_OPT_CODE_PREDICTOR_TEMPERATURE = 19,
    /**
     * "instructions", a string: how to speak the text, described in words ("怒った口調で", "Speak slowly and softly."), as
     * OpenAI's speech API names it, for a model trained to follow such a description, which may describe the voice too
     * (Irodori-TTS's caption); "" (the neutral value) gives none. One too long for the model is refused when the request
     * runs, naming the option. Added in 3.1.
     */
    SPEECH_OPT_INSTRUCTIONS = 20,
    /**
     * "cfg_scale_text", a number of 0 or more: how strongly a guided sampler follows the text, against a branch of the
     * batch that runs without it; 0 runs no such branch. Added in 3.1.
     */
    SPEECH_OPT_CFG_SCALE_TEXT = 21,
    /**
     * "cfg_scale_speaker", a number of 0 or more: how strongly a guided sampler follows the voice, as cfg_scale_text.
     * Added in 3.1.
     */
    SPEECH_OPT_CFG_SCALE_SPEAKER = 22,
    /**
     * "cfg_guidance_mode", a string: how a guided sampler forms the branches it guides against. "independent", a
     * branch without each condition that guides; "joint", one branch without every condition, which takes the scales
     * that guide equal; "alternating", one branch, without each condition that guides in turn from step to step. Added
     * in 3.1.
     */
    SPEECH_OPT_CFG_GUIDANCE_MODE = 23,
    /**
     * "cfg_min_t", a number from 0 to 1, and "cfg_max_t": the guidance runs at the sampler's steps whose time, which
     * falls from 1 at the noise to 0 at the speech, lies from cfg_min_t to cfg_max_t. Added in 3.1.
     */
    SPEECH_OPT_CFG_MIN_T = 24,
    /** "cfg_max_t", a number from 0 to 1: the latest time of the guidance, as cfg_min_t says. Added in 3.1. */
    SPEECH_OPT_CFG_MAX_T = 25,
    /** "truncation_factor", a number above 0: the factor of the sampler's starting noise. Added in 3.1. */
    SPEECH_OPT_TRUNCATION_FACTOR = 26,
    /**
     * "rescale_k", a number above 0, and "rescale_sigma": the temporal score rescaling of each step's velocity (Xu et
     * al., 2025), which a request sets with both or neither. Added in 3.1.
     */
    SPEECH_OPT_RESCALE_K = 27,
    /** "rescale_sigma", a number above 0: the other setting of the rescaling, as rescale_k says. Added in 3.1. */
    SPEECH_OPT_RESCALE_SIGMA = 28,
    /**
     * "speaker_uncond_mode", a string: what the branch without the voice attends to in its place. "mask", nothing;
     * "noise", noise of the voice condition's spread. Added in 3.1.
     */
    SPEECH_OPT_SPEAKER_UNCOND_MODE = 29,
    /**
     * "sway_coeff", a number: the coefficient of Sway Sampling (F5-TTS), which moves the sampler's steps toward the
     * noise when below 0 and toward the speech when above; 0 spaces them evenly. A coefficient that leaves two steps
     * at the same time with the request's steps is refused when the request runs. Added in 3.1.
     */
    SPEECH_OPT_SWAY_COEFF = 30,
    /**
     * "keep_tail", a boolean: whether the speech keeps all of its length, rather than ending where the model's output
     * turns flat and quiet. Added in 3.1.
     */
    SPEECH_OPT_KEEP_TAIL = 31,
    /**
     * "tail_window_size", an integer: the frames over which the model's output must be flat for the speech to end.
     * Added in 3.1.
     */
    SPEECH_OPT_TAIL_WINDOW_SIZE = 32,
    /** "tail_std_threshold", a number: the standard deviation below which the window counts as flat. Added in 3.1. */
    SPEECH_OPT_TAIL_STD_THRESHOLD = 33,
    /**
     * "tail_mean_threshold", a number: how near to 0 the window's mean must lie for it to count as quiet. Added in
     * 3.1.
     */
    SPEECH_OPT_TAIL_MEAN_THRESHOLD = 34,
    /**
     * "speaker_kv_scale", a number above 0: the factor of the keys and values a guided sampler's model attends to of the
     * voice, which above 1 makes the speech follow the voice more closely, in the first speaker_kv_max_layers layers and
     * at every step that starts at a time of speaker_kv_min_t or more; 1 leaves them as they are. Added in 3.1.
     */
    SPEECH_OPT_SPEAKER_KV_SCALE = 35,
    /**
     * "speaker_kv_min_t", a number from 0 to 1: the time below which speaker_kv_scale stops, as speaker_kv_scale says;
     * one above the first step's time keeps it to the end. Added in 3.1.
     */
    SPEECH_OPT_SPEAKER_KV_MIN_T = 36,
    /**
     * "speaker_kv_max_layers", an integer of 1 or more: the layers, from the first, that speaker_kv_scale reaches.
     * Added in 3.1.
     */
    SPEECH_OPT_SPEAKER_KV_MAX_LAYERS = 37,
    /**
     * "cfg_scale_instructions", a number of 0 or more: how strongly a guided sampler follows the instructions, as
     * cfg_scale_text. Added in 3.1.
     */
    SPEECH_OPT_CFG_SCALE_INSTRUCTIONS = 38,
    /**
     * "threshold", a number from 0 to 1: the speech probability from which a chunk of a detection's audio counts as
     * speech, as Silero VAD's get_speech_timestamps() names it. A region ends once the probability has stayed below the
     * threshold less 0.15, or 0.01 at least, for min_silence_duration_ms. Added in 3.1.
     */
    SPEECH_OPT_THRESHOLD = 39,
    /**
     * "min_speech_duration_ms", an integer of 0 or more: the milliseconds a region of a detection must last, before its
     * padding, to be kept. Added in 3.1.
     */
    SPEECH_OPT_MIN_SPEECH_DURATION_MS = 40,
    /**
     * "min_silence_duration_ms", an integer of 0 or more: the milliseconds the speech probability must stay below the
     * threshold of a region's end before the region ends. Added in 3.1.
     */
    SPEECH_OPT_MIN_SILENCE_DURATION_MS = 41,
    /**
     * "speech_pad_ms", an integer of 0 or more: the milliseconds a detection adds before and after each region, where
     * two regions closer than twice it share the silence between them half and half. Added in 3.1.
     */
    SPEECH_OPT_SPEECH_PAD_MS = 42,
    /**
     * "max_speech_duration_s", a number above 0: the longest a region of a detection may be, in seconds, its padding
     * included. A region that would grow longer is cut at the longest silence within it that lasted longer than the
     * model's least (98 ms for Silero VAD), the next region beginning where that silence ended, or, without one, at the
     * chunk that reaches the limit. Without it a region has no limit. Added in 3.1.
     */
    SPEECH_OPT_MAX_SPEECH_DURATION_S = 43,
    /**
     * "decode_on_cpu", a boolean: whether a recognition with a model on a GPU decodes its tokens on the CPU, from a
     * copy of the weights its decoding reads that the model makes there when a request first decodes on the CPU and
     * then keeps, rather than on the GPU. Each step of such a decoding is a small computation, which the CPU computes
     * in less time than a round trip to a GPU takes where its weights are quantized, so the model's default is true
     * where they are quantized and false where they are F16 or F32 (speech_model_info_option_default_bool()). The two
     * compute in different arithmetic, which can choose another token where two score almost the same. A model on the
     * CPU decodes there either way. Added in 3.2.
     */
    SPEECH_OPT_DECODE_ON_CPU = 44,
    /**
     * "draft", text: what a recognition is likely to write, such as the text of an earlier recognition of the beginning
     * of the same audio, which live transcription gives each reading of an utterance. A model that takes it checks
     * several of the draft's tokens in one step of its decoding and keeps those it would have written itself, so that it
     * writes the text it writes without the draft, in less time where the draft is right and in about the same time
     * where it is wrong. A step that checks several tokens computes several rows at once, which rounds differently from
     * a step of one, so that where two tokens score almost the same the draft can choose the other. Added in 3.3.
     */
    SPEECH_OPT_DRAFT = 45
} speech_option;

/** The type of an option's values, which names the setter that takes them. */
typedef enum speech_type {
    /** UTF-8 text, set with speech_request_set_string(). */
    SPEECH_TYPE_STRING = 0,
    /** A whole number, set with speech_request_set_int(). */
    SPEECH_TYPE_INT = 1,
    /** A finite number, set with speech_request_set_float(). */
    SPEECH_TYPE_FLOAT = 2,
    /** True or false, set with speech_request_set_bool(). */
    SPEECH_TYPE_BOOL = 3
} speech_type;

/** The number of options the library knows; they are the values from 0 to one below it. */
SPEECH_API size_t speech_option_count(void);

/**
 * The fixed name of an option in snake_case, such as "duration_scale", which the command line writes in kebab-case
 * ("--duration-scale"); NULL for a value the library does not know. The string lives as long as the process.
 */
SPEECH_API const char * speech_option_name(speech_option option);

/** The option named `name` in snake_case. A name the library does not know is an error. */
SPEECH_API speech_status speech_option_from_name(const char * name, speech_option * option);

/** The type of an option's values. */
SPEECH_API speech_type speech_option_type(speech_option option);

/*
 * Load parameters: what speech_model_load() and speech_voice_make() take besides the path, the same for every family.
 * NULL in place of a parameters object means the defaults.
 */
typedef struct speech_load_params speech_load_params;

/** Parameters at their defaults: the device "auto", the library's number of threads and no warm-up. */
SPEECH_API speech_status speech_load_params_new(speech_load_params ** params);

/** Frees parameters. NULL is ignored. */
SPEECH_API void speech_load_params_free(speech_load_params * params);

/**
 * The device to run on: "auto" (the default) for the first GPU or integrated GPU the library lists, or the CPU when
 * there is none; "gpu" for the first GPU or integrated GPU; "cpu"; or a device's name as speech_device_name() gives
 * it, compared without case. A device asked for by "gpu" or by name that is not there or does not start is an error
 * when the model loads, and no other device takes its place. The model's information names the device it runs on.
 */
SPEECH_API speech_status speech_load_params_set_device(speech_load_params * params, const char * device);

/**
 * The number of threads the CPU computes with, 1 or more: a preference about how the work runs, which applies to
 * whatever the model computes on the CPU: all of its graphs for a model on the CPU, and for a model on a GPU those it
 * computes on the CPU, such as a decoding with decode_on_cpu, or none. Without it the library takes the machine's
 * performance cores, or its physical cores where the system does not tell them apart. The model's information reports
 * the number in effect (speech_model_info_threads()).
 */
SPEECH_API speech_status speech_load_params_set_threads(speech_load_params * params, int threads);

/**
 * Whether loading also runs a short request of the model's task (nonzero), so that a GPU's kernels are compiled
 * before the first request rather than during it. Off by default; a program that loads a model to answer requests as
 * they come, such as a server, turns it on.
 */
SPEECH_API speech_status speech_load_params_set_warmup(speech_load_params * params, int warmup);

/** What a model does. */
typedef enum speech_task {
    /** Text to audio, through speech_synthesize(). */
    SPEECH_TASK_SYNTHESIS = 0,
    /** Audio to text, through speech_transcribe(). */
    SPEECH_TASK_RECOGNITION = 1,
    /** Audio to the regions where someone speaks, through speech_detect(). Added in 3.1. */
    SPEECH_TASK_DETECTION = 2
} speech_task;

/*
 * Model information: what a model file says about its model, read from the metadata without the weights, or taken
 * from a loaded model together with what loading added (the device, the voices added since). Information never
 * changes once made, any thread may read it, and its strings live until speech_model_info_free(). An index past the
 * end gives NULL or 0.
 */
typedef struct speech_model_info speech_model_info;

/**
 * Reads the information of the model file at `path` without loading its weights or touching a device. The file is
 * checked as loading checks it, short of reading the weights: its architecture and layout, every key by its type and
 * value, and every tensor's name, shape and type.
 */
SPEECH_API speech_status speech_model_info_open(const char * path, speech_model_info ** info);

/** Frees information. NULL is ignored. */
SPEECH_API void speech_model_info_free(speech_model_info * info);

/*
 * The model's identity, from the GGUF specification's general keys of its file, named in parentheses. The name of a
 * converted file is made of them under GGUF's naming convention: <basename>-<size label>-<finetune>-<version>-<weight
 * type>.gguf, without the parts the model has none of.
 */

/**
 * The model's name (general.name), the name of the repository it was converted from, such as
 * "Qwen3-TTS-12Hz-0.6B-CustomVoice" or "parakeet-tdt-0.6b-v3".
 */
SPEECH_API const char * speech_model_info_name(const speech_model_info * info);

/**
 * The organization that publishes the model (general.organization), the namespace of the repository it was converted
 * from, such as "Qwen" or "nvidia".
 */
SPEECH_API const char * speech_model_info_organization(const speech_model_info * info);

/**
 * The model's line (general.basename), such as "Qwen3-TTS-12Hz" or "parakeet-tdt". Models of several lines can run in
 * one family (speech_model_info_architecture()).
 */
SPEECH_API const char * speech_model_info_basename(const speech_model_info * info);

/**
 * The model's size (general.size_label): the one its name gives, such as "0.6B", or, where its name gives none, the
 * parameters of the file's tensors counted and rounded, such as "619M".
 */
SPEECH_API const char * speech_model_info_size_label(const speech_model_info * info);

/**
 * What the model was trained toward beyond its line (general.finetune), such as "CustomVoice" or "ja", or NULL for a
 * model that has none.
 */
SPEECH_API const char * speech_model_info_finetune(const speech_model_info * info);

/** The model's version as its name gives it (general.version), such as "v3", or NULL for a model whose name has none. */
SPEECH_API const char * speech_model_info_version(const speech_model_info * info);

/** The license of the model's weights as an SPDX expression (general.license), such as "Apache-2.0". */
SPEECH_API const char * speech_model_info_license(const speech_model_info * info);

/**
 * The repository the model file was converted from, as a URL (general.source.repo_url), such as
 * "https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice", and the revision of it that was converted, a commit's
 * hash (general.source.url, which is "<repository>/tree/<revision>").
 */
SPEECH_API speech_status speech_model_info_source(const speech_model_info * info, const char ** repository,
                                                  const char ** revision);

/**
 * The type that holds most of the bytes of the model's weights (general.file_type), as ggml names it in capitals:
 * "F32", "F16" or "Q8_0", or "Q6_K", "Q5_K" or "Q4_K" (added in 3.1). The rest, such as the norms and the biases, is in
 * other types.
 */
SPEECH_API const char * speech_model_info_weight_type(const speech_model_info * info);

/**
 * The architecture of the model's family (general.architecture), such as "irodori-tts": the code that runs the file,
 * which models of several lines share, as "fastconformer" runs parakeet and ReazonSpeech.
 */
SPEECH_API const char * speech_model_info_architecture(const speech_model_info * info);

/** The version of the family's file layout that the file has. */
SPEECH_API uint32_t speech_model_info_layout(const speech_model_info * info);

/** Whether the model speaks, recognizes speech or detects it. */
SPEECH_API speech_task speech_model_info_task(const speech_model_info * info);

/**
 * The sample rate in Hz of the audio a synthesis model makes, or of the audio a recognition or detection model takes.
 * Audio given at another rate, to recognize, to detect speech in or as a voice's reference, is resampled to it.
 */
SPEECH_API int speech_model_info_sample_rate(const speech_model_info * info);

/**
 * Whether a synthesis model passes audio while it is still generating the rest (1), so that its first audio does not
 * wait on the length of the text, or generates a request's whole speech before it decodes the speech into audio (0).
 * 0 for a recognition or detection model.
 */
SPEECH_API int speech_model_info_incremental(const speech_model_info * info);

/**
 * The number of languages the model speaks or recognizes; 0 for a detection model, whose file names no language and
 * which takes the language "auto" alone.
 */
SPEECH_API size_t speech_model_info_language_count(const speech_model_info * info);

/**
 * The language at `index` as a BCP 47 tag, such as "ja" or "yue": the file's general.languages, the shortest ISO 639
 * code of each language, two letters where ISO 639-1 has one and three where it has none, which are BCP 47 tags.
 */
SPEECH_API const char * speech_model_info_language(const speech_model_info * info, size_t index);

/**
 * The number of voices: the model's own, then, for information taken from a loaded model, those added with
 * speech_voice_add() in the order they were added.
 */
SPEECH_API size_t speech_model_info_voice_count(const speech_model_info * info);

/** The name of the voice at `index`, which the voice option takes. */
SPEECH_API const char * speech_model_info_voice_name(const speech_model_info * info, size_t index);

/** The language the voice at `index` was recorded in as a BCP 47 tag, or "" when the model file does not say. */
SPEECH_API const char * speech_model_info_voice_language(const speech_model_info * info, size_t index);

/** "female" or "male" for the voice at `index`, or "" when the model file does not say. */
SPEECH_API const char * speech_model_info_voice_gender(const speech_model_info * info, size_t index);

/** The model's description of the voice at `index` in English, or "" when the model file has none. */
SPEECH_API const char * speech_model_info_voice_description(const speech_model_info * info, size_t index);

/**
 * Whether the model takes voices made from reference recordings (1), through speech_voice_add() and
 * speech_voice_make_from().
 */
SPEECH_API int speech_model_info_voice_files(const speech_model_info * info);

/**
 * The hash a voice file must carry to work with the model, the SHA-256 of its codec's official weights in lowercase
 * hexadecimal, or NULL for a model that takes no voice files.
 */
SPEECH_API const char * speech_model_info_voice_codec(const speech_model_info * info);

/**
 * Whether speech_voice_add() takes a reference recording in a WAVE file as well as a voice file (1). A model that needs
 * the words of each recording (Drama 3) takes voice files alone, made with speech_voice_make_from(), and is 0, as is a
 * model that takes no voice files. Added in 3.3.
 */
SPEECH_API int speech_model_info_voice_recordings(const speech_model_info * info);

/** The most tokens of text a synthesis request may have, in the model's own tokens; 0 for a recognition or detection model. */
SPEECH_API size_t speech_model_info_max_text_tokens(const speech_model_info * info);

/**
 * The number of the model's tokens that `text` takes, counted as a synthesis request counts it against
 * speech_model_info_max_text_tokens(), so that a caller can split a long text before it sends it. A recognition or
 * detection model is SPEECH_ERROR_UNSUPPORTED.
 */
SPEECH_API speech_status speech_model_info_text_tokens(const speech_model_info * info, const char * text,
                                                       size_t * n_tokens);

/** The number of options the model takes. */
SPEECH_API size_t speech_model_info_option_count(const speech_model_info * info);

/** The option at `index` of those the model takes, in the order of the vocabulary. */
SPEECH_API speech_option speech_model_info_option(const speech_model_info * info, size_t index);

/**
 * Whether the model takes `option` (1). An option it does not take is accepted only at the option's neutral value,
 * and refused at any value when the option has none.
 */
SPEECH_API int speech_model_info_takes(const speech_model_info * info, speech_option option);

/** Whether a request to the model must set `option` (1). */
SPEECH_API int speech_model_info_option_required(const speech_model_info * info, speech_option option);

/**
 * Whether the value of `option` steers what the model does (1), or is only checked against the option's choices and
 * then not used (0), as a language is by a model that has one language or finds the language itself.
 */
SPEECH_API int speech_model_info_option_steers(const speech_model_info * info, speech_option option);

/**
 * Whether `option` has a default that the model uses when a request does not set it (1). An option without one is
 * required (voice), drawn by the library when it is not set (seed), or without effect until it is set (seconds,
 * max_seconds).
 */
SPEECH_API int speech_model_info_option_has_default(const speech_model_info * info, speech_option option);

/**
 * The default of a string option. An option the model does not take, an option of another type or one without a
 * default is an error.
 */
SPEECH_API speech_status speech_model_info_option_default_string(const speech_model_info * info, speech_option option,
                                                                 const char ** value);

/** The default of an integer option, under the rules of speech_model_info_option_default_string(). */
SPEECH_API speech_status speech_model_info_option_default_int(const speech_model_info * info, speech_option option,
                                                              int64_t * value);

/** The default of a number option, under the rules of speech_model_info_option_default_string(). */
SPEECH_API speech_status speech_model_info_option_default_float(const speech_model_info * info, speech_option option,
                                                                double * value);

/** The default of a boolean option, under the rules of speech_model_info_option_default_string(). */
SPEECH_API speech_status speech_model_info_option_default_bool(const speech_model_info * info, speech_option option,
                                                               int * value);

/**
 * The range of an integer or number option: `*minimum` and `*maximum`, -INFINITY or INFINITY where the option has no
 * bound, and `*minimum_exclusive` nonzero when the minimum itself lies outside the range. A value within it can still
 * be refused when the request runs, for what it does together with another option or with the text, and the error
 * then names the option. An option the model does not take, or one of another type, is an error.
 */
SPEECH_API speech_status speech_model_info_option_range(const speech_model_info * info, speech_option option,
                                                        double * minimum, double * maximum, int * minimum_exclusive);

/**
 * The number of values a string option takes: the voices for "voice", the languages for "language", which also
 * takes "auto" and a region or script of each, the decodings for "decoding", the default among them, and the modes for
 * "cfg_guidance_mode" and "speaker_uncond_mode". 0 for "prompt" and "instructions", which take any text, and for an
 * option of another type.
 */
SPEECH_API size_t speech_model_info_option_choice_count(const speech_model_info * info, speech_option option);

/** The value at `index` of those a string option takes. */
SPEECH_API const char * speech_model_info_option_choice(const speech_model_info * info, speech_option option,
                                                        size_t index);

/** The size of the model file in bytes. */
SPEECH_API uint64_t speech_model_info_file_bytes(const speech_model_info * info);

/** The bytes the model's weights take on a device: the sum of its tensors as the file stores them. */
SPEECH_API uint64_t speech_model_info_weight_bytes(const speech_model_info * info);

/**
 * The name of the device the model runs on as speech_device_name() gives it, for information taken from a loaded
 * model, or NULL for information read from a file.
 */
SPEECH_API const char * speech_model_info_device(const speech_model_info * info);

/**
 * The number of threads the CPU computes a loaded model's graphs with: the number the load parameters set, or the
 * library's default, for a model on the CPU and for a model on a GPU that can compute some of its graphs on the CPU (a
 * model that takes decode_on_cpu), and 0 for a model on a GPU none of whose graphs runs on the CPU. 0 for information
 * read from a file.
 */
SPEECH_API int speech_model_info_threads(const speech_model_info * info);

/**
 * All of the information as one JSON object: the form the worker's ready message, the HTTP server's model object and
 * `speech info --json` carry, so that every program that shows it shows the same.
 */
SPEECH_API const char * speech_model_info_json(const speech_model_info * info);

/** The number of metadata entries in the model file, for a caller that needs a key the information does not name. */
SPEECH_API size_t speech_model_info_meta_count(const speech_model_info * info);

/** The key of the metadata entry at `index`, such as "general.license". */
SPEECH_API const char * speech_model_info_meta_key(const speech_model_info * info, size_t index);

/**
 * The value of the metadata entry at `index` as JSON text: a string, a number, true or false, or an array of them. A
 * long array, such as a tokenizer's vocabulary, is written whole.
 */
SPEECH_API const char * speech_model_info_meta_value(const speech_model_info * info, size_t index);

/**
 * Writes the model file at `model_path`, whose weights are F32, to `out_path` with its weights in `type`: "F16", "Q8_0",
 * "Q6_K", "Q5_K" or "Q4_K", compared without case, the names speech_model_info_weight_type() gives. Each tensor takes
 * the type the family's layout gives it in a file of that type: a matrix that the model only multiplies by or looks
 * rows up in takes `type`, Q8_0 instead of a K-quant (Q6_K, Q5_K, Q4_K) where its rows are not whole blocks of 256
 * values, and F16 instead of Q8_0 where they are not whole blocks of 32; a codec's or a convolution's weights that the
 * family keeps out of quantization take F16; and the rest stays F32. Every key and tensor of the file is kept, in its
 * order, but general.file_type, general.quantization_version, which a quantized file has, and speech.requires, which
 * names the first release that reads the file: 0.8.0 for Q6_K, Q5_K and Q4_K and for FastConformer's Q8_0, which
 * releases before it refuse. A file
 * whose weights are not F32 is SPEECH_ERROR_INVALID_ARGUMENT naming "model_path": a quantized file is not quantized
 * again, and a type made of F16 weights would hold other bytes than the same type made of the F32 weights. A weight that
 * is not finite, NaN or an infinity, is SPEECH_ERROR_MODEL_FILE naming "model_path". `out_path`, which may not be the
 * model file itself, is replaced, and removed again when the writing fails. It runs on the CPU, with the machine's
 * threads; a thread that the host cannot start is SPEECH_ERROR_OUT_OF_MEMORY. Added in 3.1.
 */
SPEECH_API speech_status speech_quantize(const char * model_path, const char * type, const char * out_path);

/** A loaded model on its device, with its voices. */
typedef struct speech_model speech_model;

/**
 * Loads the model file at `path`, which holds every part of the model, its codec included, on the device that the
 * parameters name, or with the defaults when `params` is NULL. The file's general.architecture chooses the family. On
 * success `*model` is a model that speech_model_free() frees; on failure it is NULL.
 */
SPEECH_API speech_status speech_model_load(const char * path, const speech_load_params * params,
                                           speech_model ** model);

/**
 * Frees a model and its device. Every request and detection made for it must be freed first. NULL is ignored. A
 * process frees every model it loaded before it returns from main() or calls exit(): on Metal, ggml aborts a process
 * that ends with a model still loaded, after its work is done, with SIGABRT.
 */
SPEECH_API void speech_model_free(speech_model * model);

/**
 * The information of a loaded model as it is at the call: the file's, with the device the model runs on and the
 * voices added so far. The caller frees it with speech_model_info_free(). Any thread may call it.
 */
SPEECH_API speech_status speech_model_get_info(const speech_model * model, speech_model_info ** info);

/*
 * Voices made from reference recordings, for a model whose information says it takes voice files. A voice file holds
 * the codec's latent of one or more reference recordings joined, each recording's length and rate, the loudness they
 * were brought to, the kind of device that encoded them and the hash of the codec they were encoded with, and it works
 * with every model whose codec has that hash; or a speaker-inversion embedding and the model it was made for. Every
 * voice file this release makes, of one reference at the model's loudness too, is of the voice file's layout 2, added
 * in 3.1, which releases before 0.8.0 refuse; a voice file of layout 1, which releases from 0.7.0 made, is read as
 * before.
 */

/**
 * Adds a voice named `name` to the model, from a voice file or from a reference recording in a WAVE file, which is
 * mixed to mono, resampled to the model's rate and encoded with the model's codec on the model's device. The name is
 * compared with case and must not name a voice the model has. A request that is running finishes first; any thread
 * may call it. A model that takes no voice files is SPEECH_ERROR_UNSUPPORTED, and a voice file made with another
 * codec is SPEECH_ERROR_INVALID_ARGUMENT. A model whose speech_model_info_voice_recordings() is 0 (Drama 3, which needs
 * a recording's words) takes a voice file alone, and a recording is SPEECH_ERROR_INVALID_ARGUMENT naming "path"; its
 * information describes the voice by its speakers.
 */
SPEECH_API speech_status speech_voice_add(speech_model * model, const char * name, const char * path);

/**
 * Makes a voice file at `voice_path` from the reference recording in the WAVE file at `reference_path`, for the model
 * file at `model_path`, reading only the codec's encoder from the model file and running it on the device that the
 * parameters name. The recording is mixed to mono, resampled to the model's rate, and may be at most as long as the
 * model's file allows. On the CPU the latent is the official encoder's to 99 dB SNR; a GPU computes it in less
 * precision, and the voice file records which kind of device made it.
 */
SPEECH_API speech_status speech_voice_make(const char * model_path, const char * reference_path,
                                           const char * voice_path, const speech_load_params * params);

/*
 * Voice parameters: what speech_voice_make_from() makes a voice file of: several reference recordings and the loudness
 * each is brought to, as Irodori-TTS's runtime's ref_wavs and ref_normalize_db, or a speaker-inversion embedding, as
 * its ref_embed; or, for Drama 3, recordings with their transcripts and voice files, each given to one of the speakers
 * a text names. Added in 3.1.
 */
typedef struct speech_voice_params speech_voice_params;

/** Parameters without a reference, at the model's loudness. Added in 3.1. */
SPEECH_API speech_status speech_voice_params_new(speech_voice_params ** params);

/** Frees parameters. NULL is ignored. Added in 3.1. */
SPEECH_API void speech_voice_params_free(speech_voice_params * params);

/**
 * Adds a reference recording in the WAVE file at `path`, which is encoded on its own and joined after the references
 * added before. The references together may be at most as long as the model's file allows; a failure names
 * "references". Added in 3.1.
 */
SPEECH_API speech_status speech_voice_params_add_reference(speech_voice_params * params, const char * path);

/**
 * Brings each reference to `lufs`, a finite loudness in LUFS, before it is encoded, instead of the model's
 * (irodori-tts.reference.lufs, -16 LUFS), which speech_voice_make() and speech_voice_add() take. Added in 3.1.
 */
SPEECH_API speech_status speech_voice_params_set_loudness(speech_voice_params * params, double lufs);

/**
 * Keeps each reference's loudness as recorded, scaling down one whose peak exceeds 1, instead of bringing it to a
 * loudness. Added in 3.1.
 */
SPEECH_API speech_status speech_voice_params_keep_loudness(speech_voice_params * params);

/**
 * Makes the voice of a speaker-inversion embedding instead of references, as the official runtime's ref_embed:
 * `tokens` vectors of `dim` finite values, row by row, which the model attends to in place of the speaker condition it
 * makes of a reference, and whose first vector the duration predictor takes as the speaker's. `dim` must be the model's
 * irodori-tts.speaker.dim. The values are copied. An embedding is learned against one model, which the voice file names
 * and no other model takes; it has no references and no loudness. Added in 3.1.
 */
SPEECH_API speech_status speech_voice_params_set_embedding(speech_voice_params * params, const float * values, size_t tokens,
                                                           size_t dim);

/**
 * Gives the reference recording added last its transcript, the words it speaks in UTF-8, which a model that writes a
 * reference's words into its prompt needs for every recording (Drama 3) and any other refuses. Added in 3.3.
 */
SPEECH_API speech_status speech_voice_params_set_transcript(speech_voice_params * params, const char * text);

/**
 * Gives what was added last, a reference recording or a voice file of one speaker, the speaker `speaker`, from 0, under
 * which a text's speaker tag finds it (Drama 3: `<|speaker:N|>`). A recording is speaker 0's and a voice file keeps its
 * speakers unless one is given; a model of one speaker refuses it. Added in 3.3.
 */
SPEECH_API speech_status speech_voice_params_set_speaker(speech_voice_params * params, int speaker);

/**
 * Adds the references of the voice file at `path`, made for a model of the same family and codec, as they are, without
 * encoding them again, so that a voice of several speakers is put together from voices made before (Drama 3); a model
 * that joins its references into one voice refuses it. Added in 3.3.
 */
SPEECH_API speech_status speech_voice_params_add_voice(speech_voice_params * params, const char * path);

/**
 * Makes a voice file at `voice_path` of `voice`, its references, at least one, or its embedding, for the model file at
 * `model_path`, as speech_voice_make() makes one of a single reference at the model's loudness, which this gives for one
 * reference and no loudness of its own. Added in 3.1.
 */
SPEECH_API speech_status speech_voice_make_from(const char * model_path, const speech_voice_params * voice, const char * voice_path,
                                                const speech_load_params * params);

/*
 * Requests. A request is made for one model and checks every value as it is set against what that model declares:
 * an option the model does not take, at a value other than the option's neutral one, is SPEECH_ERROR_UNSUPPORTED; a
 * value outside the option's range or choices is SPEECH_ERROR_OUT_OF_RANGE; a setter of another type than the
 * option's is SPEECH_ERROR_INVALID_ARGUMENT. A refused value leaves the request as it was, and setting an option
 * again replaces its value. An option the request does not set takes the model's default. What only the whole request
 * shows (a required option left out, two options that cannot go together, a text longer than the model takes) is
 * refused when the request runs, before any audio, result or progress, and the error names the option or input; such
 * a request is left as it was, to be fixed and run again. A request that has started its work runs once.
 */
typedef struct speech_request speech_request;

/** A request for `model`, with no input and no option set, freed with speech_request_free() before the model. */
SPEECH_API speech_status speech_request_new(speech_model * model, speech_request ** request);

/** Frees a request and its result. It must not be running. NULL is ignored. */
SPEECH_API void speech_request_free(speech_request * request);

/**
 * The text a synthesis request speaks, copied; it may not be empty. A recognition or detection model is
 * SPEECH_ERROR_UNSUPPORTED.
 */
SPEECH_API speech_status speech_request_set_text(speech_request * request, const char * text);

/**
 * The audio a recognition request recognizes, or a detection request finds speech in: `n_samples` mono samples,
 * nominally within [-1, 1], at `sample_rate` Hz, copied. Audio at a rate other than the model's is resampled to it when
 * the request runs. At least one sample is needed. A synthesis model is SPEECH_ERROR_UNSUPPORTED.
 */
SPEECH_API speech_status speech_request_set_audio(speech_request * request, const float * samples, size_t n_samples,
                                                  int sample_rate);

/** Sets a string option, copied. */
SPEECH_API speech_status speech_request_set_string(speech_request * request, speech_option option, const char * value);

/** Sets an integer option. */
SPEECH_API speech_status speech_request_set_int(speech_request * request, speech_option option, int64_t value);

/** Sets a number option. NaN and the infinities are refused. */
SPEECH_API speech_status speech_request_set_float(speech_request * request, speech_option option, double value);

/** Sets a boolean option, to true when `value` is nonzero. */
SPEECH_API speech_status speech_request_set_bool(speech_request * request, speech_option option, int value);

/**
 * Receives how far a request has come while it does work that passes no audio: `done` rises from 0 to 1 as the
 * family's steps finish (the sampler's steps of a speech made whole before its audio, the stages and the decoding of
 * a recognition, the blocks of chunks of a detection). The time between two calls is that of one step, which for the
 * encoder of a long recording can be seconds. Returning nonzero stops the request. It runs on the thread that runs the
 * request.
 */
typedef int (*speech_progress_callback)(double done, void * user_data);

/** The callback that receives the request's progress with `user_data`, or NULL for none, the default. */
SPEECH_API speech_status speech_request_set_progress(speech_request * request, speech_progress_callback on_progress,
                                                     void * user_data);

/**
 * Stops the request at the next point where the work under way can stop: before the next piece of audio, the next
 * step or the next stage. The request then returns SPEECH_CANCELLED. A request cancelled before it runs returns
 * SPEECH_CANCELLED without doing any work, and one that has returned is not affected. Any thread may call it while
 * the request exists, and it affects this request alone.
 */
SPEECH_API void speech_request_cancel(speech_request * request);

/**
 * Receives the audio of a synthesis as it is made: `n_samples` mono samples, at least one, at the model's sample
 * rate, nominally within [-1, 1] and not clamped, each a finite number, valid only during the call. Returning nonzero
 * stops the request. It runs on the thread that called speech_synthesize() and must not call into the library with the
 * same model.
 */
typedef int (*speech_audio_callback)(const float * samples, size_t n_samples, void * user_data);

/**
 * Speaks a request, passing its audio to `on_audio` with `user_data`. It returns SPEECH_OK once the speech has ended
 * or has been stopped by a limit, with the reason in the result; SPEECH_CANCELLED once it was cancelled; or an error,
 * after which no more audio comes. The request is checked as a whole before any audio or progress: a model that
 * predicts the speech's length, as Irodori-TTS does, refuses a length it cannot make once it has predicted it. Speech
 * that comes out not finite, as from a guidance scale that a float32 holds but whose products it does not, is never
 * passed on: the request fails with SPEECH_ERROR_OUT_OF_RANGE naming no option, since each value was within its range
 * and none is known to be at fault, and the caller makes a new request with smaller values. A recognition or detection
 * model is SPEECH_ERROR_UNSUPPORTED.
 */
SPEECH_API speech_status speech_synthesize(speech_request * request, speech_audio_callback on_audio, void * user_data);

/**
 * Recognizes the speech in a request's audio. It returns SPEECH_OK once the result holds the text, SPEECH_CANCELLED
 * once it was cancelled, or an error. The whole audio is recognized at once, or a model that takes a limited length
 * at once cuts it into parts it recognizes one after another and joins their texts, as Qwen3-ASR does past 1200 s;
 * how its time and memory grow with its length depends on the model. A synthesis or detection model is
 * SPEECH_ERROR_UNSUPPORTED.
 */
SPEECH_API speech_status speech_transcribe(speech_request * request);

/**
 * Finds the regions of a request's audio where someone speaks, which the result gives as its segments: each with its
 * start and end in seconds from the start of the audio and an empty text, in order and apart. It returns SPEECH_OK once
 * the result holds them, none where no one speaks, SPEECH_CANCELLED once it was cancelled, or an error. The model takes
 * the audio in chunks, each after the one before, so that its time grows with the audio's length; the request's options
 * (threshold, min_speech_duration_ms, min_silence_duration_ms, speech_pad_ms, max_speech_duration_s) only turn the
 * chunks' speech probabilities into regions. A synthesis or recognition model is SPEECH_ERROR_UNSUPPORTED. Added in
 * 3.1.
 */
SPEECH_API speech_status speech_detect(speech_request * request);

/*
 * Detections of audio that arrives a piece at a time, as from a microphone, for a detection model: the regions where
 * someone speaks, each given as soon as the audio heard makes it certain. Once the audio has ended, the regions are those
 * speech_detect() gives for all of it, sample for sample, however it was cut into pieces. A detection is used by one
 * thread at a time. Added in 3.1.
 */
typedef struct speech_detection speech_detection;

/**
 * Starts a detection of audio at `sample_rate` Hz, which speech_detection_push() gives, with the options of `request`,
 * checked as speech_detect() checks them and copied; the request is left as it was. A request that has audio is
 * SPEECH_ERROR_INVALID_ARGUMENT naming "audio", as its audio would go unused, and so is a rate that
 * speech_request_set_audio() refuses. A synthesis or recognition model is SPEECH_ERROR_UNSUPPORTED. On success
 * `*detection` is a detection that speech_detection_free() frees before the model. Added in 3.1.
 */
SPEECH_API speech_status speech_detection_start(const speech_request * request, int sample_rate, speech_detection ** detection);

/** Frees a detection. NULL is ignored. Added in 3.1. */
SPEECH_API void speech_detection_free(speech_detection * detection);

/**
 * Gives the detection the next `n_samples` mono samples, nominally within [-1, 1], at its rate, copied; none at all is
 * allowed, and `samples` may then be NULL. The samples are resampled to the model's rate, which holds back the last 4.3 ms
 * of audio from 24, 44.1 or 48 kHz (8.6 ms from 8 kHz) until the audio after them arrives; the model computes each of
 * its chunks (32 ms for Silero VAD) once its samples have arrived, and the regions that the audio heard makes certain
 * are given (speech_detection_region_count()). It computes on the model's device while the model does nothing else, so
 * that the pushes of detections on one model and the model's requests wait for each other. A detection that has ended,
 * or whose push or end failed, takes no more audio: SPEECH_ERROR_INVALID_ARGUMENT. Added in 3.1.
 */
SPEECH_API speech_status speech_detection_push(speech_detection * detection, const float * samples, size_t n_samples);

/**
 * Ends the audio: the samples held back are resampled, the model computes its last chunk, padded with zeros, and every
 * region left is given, after which the regions are those speech_detect() gives for all of the audio pushed. A detection
 * that has ended, or whose push failed, is SPEECH_ERROR_INVALID_ARGUMENT. Added in 3.1.
 */
SPEECH_API speech_status speech_detection_end(speech_detection * detection);

/**
 * The number of regions given so far. A region is given once the audio heard makes its padded start and end certain.
 * With min_silence_duration_ms at least twice speech_pad_ms, as with the defaults, that is the chunk where the silence
 * after the region has lasted min_silence_duration_ms and ends it. With less, the region's end waits on whether speech
 * resumes within twice speech_pad_ms of it, for at most 2 · speech_pad_ms + min_speech_duration_ms +
 * min_silence_duration_ms and two chunks after the region. A region that max_speech_duration_s cuts is given at the cut
 * at the earliest, and may wait as long after it. Added in 3.1.
 */
SPEECH_API size_t speech_detection_region_count(const speech_detection * detection);

/**
 * The region at `index` in the order given: its start and end in seconds from the start of the audio pushed, as
 * speech_detect() times a region. A region once given never changes. Added in 3.1.
 */
SPEECH_API speech_status speech_detection_region(const speech_detection * detection, size_t index, double * start, double * end);

/**
 * Whether a region has begun that is not given yet (1), with the start it will have in `*start` and in `*kept` whether it
 * is certain to be given, each unless NULL: a region that has ended and whose end waits on what follows, as with
 * min_silence_duration_ms below twice speech_pad_ms or after a cut at max_speech_duration_s, which is kept, or else the
 * region under way, where someone speaks at the end of the audio heard. A region under way is kept once no end it can
 * still take leaves it as short as min_speech_duration_ms: its speech has lasted longer than that, to the audio heard or,
 * where its probability has fallen below the threshold less 0.15 (0.01 at least) and not risen above the threshold
 * since, to where it fell. Until then it is dropped, and no region is given from its start, if its speech ends that
 * soon. 0 when neither, and after speech_detection_end(). Added in 3.1.
 */
SPEECH_API int speech_detection_speaking(const speech_detection * detection, double * start, int * kept);

/** Why a request ended. */
typedef enum speech_stop {
    /** It did all it was asked: the speech came to its end, or the audio was recognized or searched whole. */
    SPEECH_STOP_COMPLETE = 0,
    /** The speech reached the request's max_seconds and was stopped there. */
    SPEECH_STOP_MAX_SECONDS = 1,
    /**
     * The request reached the most that the model makes, which its file gives, and was stopped there: the longest speech
     * of a synthesis, or the most tokens a recognition writes, whose text then holds what was written up to it.
     */
    SPEECH_STOP_MODEL_LIMIT = 2,
    /** speech_request_cancel() or a callback stopped it. */
    SPEECH_STOP_CANCELLED = 3
} speech_stop;

/**
 * The name of a stop reason in snake_case: "complete", "max_seconds", "model_limit" or "cancelled"; NULL for a value
 * the library does not know. The string lives as long as the process.
 */
SPEECH_API const char * speech_stop_name(speech_stop stop);

/** What a request did, owned by the request. */
typedef struct speech_result speech_result;

/**
 * The result of a request that returned SPEECH_OK or SPEECH_CANCELLED, or NULL before the request has run and after
 * it failed. It lives until speech_request_free().
 */
SPEECH_API const speech_result * speech_request_result(const speech_request * request);

/** Why the request ended. */
SPEECH_API speech_stop speech_result_stop(const speech_result * result);

/** The seed a synthesis used, the request's or the one the library drew; -1 for a recognition and a detection. */
SPEECH_API int64_t speech_result_seed(const speech_result * result);

/**
 * The number of samples a synthesis passed to its callback, whose length in seconds is this over the sample rate; 0 for
 * a recognition and a detection.
 */
SPEECH_API uint64_t speech_result_samples(const speech_result * result);

/** The text a recognition found, "" when it heard none or was cancelled; NULL for a synthesis and a detection. */
SPEECH_API const char * speech_result_text(const speech_result * result);

/**
 * The number of segments of a recognition that set timestamps: runs of tokens that end where the model's file says a
 * sentence ends (a separator at the end of a word, as NeMo cuts, or a break after any token, for a language written
 * without spaces), or with the last token. 0 without timestamps and for a synthesis. For a detection, the regions where
 * someone speaks, 0 where no one does and for a cancelled request.
 */
SPEECH_API size_t speech_result_segment_count(const speech_result * result);

/**
 * The segment at `index`: the start of its first token and the end of its last in seconds from the start of the
 * audio, and its text. The texts of the segments, joined in order, are the result's text. A region of a detection has
 * its start and end and the text "".
 */
SPEECH_API speech_status speech_result_segment(const speech_result * result, size_t index, double * start, double * end,
                                               const char ** text);

/** The number of tokens of a recognition that set timestamps; 0 without timestamps and for a synthesis. */
SPEECH_API size_t speech_result_token_count(const speech_result * result);

/**
 * The token at `index`: its start in seconds, the encoder frame the decoder emitted it on; its end, the frame that
 * its predicted duration reaches for a model that predicts one, or the next frame; and its text as it stands in the
 * result's text. A punctuation mark of a model that predicts durations starts and ends where the token before it
 * ends, as NeMo times it. The texts of the tokens, joined in order, are the result's text.
 */
SPEECH_API speech_status speech_result_token(const speech_result * result, size_t index, double * start, double * end,
                                             const char ** text);

/**
 * The number of languages a recognition heard: those the model wrote before its text, or the language a request forced
 * on a model that the language steers, which the model's prompt writes for it. 0 for a synthesis, a cancelled request,
 * a model that writes no language (FastConformer, whose parakeet model of many languages tells them apart without
 * writing any), audio in which the model heard no speech (Qwen3-ASR's language None, or nothing written after a forced
 * language), and a name the model wrote that is none of its languages, which a warning in the log reports. A
 * recognition of audio that the model takes in parts (Qwen3-ASR past 1200 s) has the languages of its parts in the order
 * of the audio, one for each run of parts of the same language, as qwen-asr's merge_languages() merges them; one of a
 * single part has one at most. Added in 3.1.
 */
SPEECH_API size_t speech_result_language_count(const speech_result * result);

/**
 * The language at `index` as a BCP 47 tag, one of the model's languages as speech_model_info_language() gives them
 * ("ja", "yue"), or NULL past the last. Added in 3.1.
 */
SPEECH_API const char * speech_result_language(const speech_result * result, size_t index);

#ifdef __cplusplus
}
#endif

#endif
