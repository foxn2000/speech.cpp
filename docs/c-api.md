# C API

This page describes the C API, `include/speech.h`, through which every program reaches the models. The header
documents each function; this page shows the shape of a program and the rules that hold across functions.

## A program

A program that speaks one sentence:

```c
#include <stdio.h>
#include "speech.h"

static int on_audio(const float * samples, size_t n, void * user_data) {
    /* n mono float samples at the model's sample rate, at least one. Returning nonzero stops the request. */
    return 0;
}

static int check(speech_status status) {
    if (status < 0) {
        const char * option = speech_last_error_option();
        fprintf(stderr, "%s (%s): %s\n", speech_status_name(status), option ? option : "-", speech_last_error());
    }
    return status >= 0;
}

int main(void) {
    speech_model * model = NULL;
    speech_request * request = NULL;
    if (check(speech_model_load("Qwen3-TTS-12Hz-0.6B-CustomVoice-Q8_0.gguf", NULL, &model)) &&
        check(speech_request_new(model, &request)) &&
        check(speech_request_set_text(request, "明日の東京は晴れです。")) &&
        check(speech_request_set_string(request, SPEECH_OPT_VOICE, "ono_anna")) &&
        check(speech_request_set_int(request, SPEECH_OPT_SEED, 42)) &&
        check(speech_synthesize(request, on_audio, NULL))) {
        const speech_result * result = speech_request_result(request);
        printf("%s, seed %lld, %llu samples\n", speech_stop_name(speech_result_stop(result)),
               (long long) speech_result_seed(result), (unsigned long long) speech_result_samples(result));
    }
    speech_request_free(request);
    speech_model_free(model);
    return 0;
}
```

A recognition of mono samples at any rate, with the times of its segments:

```c
if (check(speech_model_load("parakeet-tdt_ctc-0.6B-ja-F16.gguf", NULL, &model)) &&
    check(speech_request_new(model, &request)) &&
    check(speech_request_set_audio(request, samples, n_samples, 48000)) &&
    check(speech_request_set_bool(request, SPEECH_OPT_TIMESTAMPS, 1)) &&
    check(speech_transcribe(request))) {
    const speech_result * result = speech_request_result(request);
    for (size_t i = 0; i < speech_result_segment_count(result); i++) {
        double start, end;
        const char * text;
        speech_result_segment(result, i, &start, &end, &text);
        printf("%.2f %.2f %s\n", start, end, text);
    }
}
```

The regions of a recording where someone speaks, each a segment with an empty text, of 10 s at most:

```c
if (check(speech_model_load("silero-vad-309K-v6.2-F32.gguf", NULL, &model)) &&
    check(speech_request_new(model, &request)) &&
    check(speech_request_set_audio(request, samples, n_samples, 48000)) &&
    check(speech_request_set_float(request, SPEECH_OPT_MAX_SPEECH_DURATION_S, 10)) &&
    check(speech_detect(request))) {
    const speech_result * result = speech_request_result(request);
    for (size_t i = 0; i < speech_result_segment_count(result); i++) {
        double start, end;
        speech_result_segment(result, i, &start, &end, NULL);
        printf("%.3f %.3f\n", start, end);
    }
}
```

The same regions while the audio arrives, as from a microphone, in pieces of any size at any rate:

```c
speech_detection * detection = NULL;
size_t given = 0;
if (check(speech_request_new(model, &request)) &&
    check(speech_request_set_int(request, SPEECH_OPT_MIN_SILENCE_DURATION_MS, 500)) &&
    check(speech_detection_start(request, 48000, &detection))) {
    while (next_piece(&samples, &n_samples)) {  /* the caller's audio, at 48 kHz */
        check(speech_detection_push(detection, samples, n_samples));
        double start, end;
        int kept = 0;
        if (speech_detection_speaking(detection, &start, &kept) && kept) { /* someone speaks, from start */ }
        for (; given < speech_detection_region_count(detection); given++) {
            speech_detection_region(detection, given, &start, &end);
            printf("%.3f %.3f\n", start, end);  /* an utterance has ended */
        }
    }
    check(speech_detection_end(detection));  /* and gives the regions left */
}
speech_detection_free(detection);
```

The library loads files and never reaches the network: a program passes a model file's path. `speech pull NAME` prints
the path of a model of the catalog.

Link `libspeech` from a release archive ([install.md](install.md#release-archives)). On Windows, define `SPEECH_SHARED`
and link `speech.lib`. Inside this CMake project, link the target `speech` (shared) or `speech-static`.

## Functions

- **Loading.** `speech_model_load()` loads a model from its one GGUF file, which holds its codec. A file made for a
  later release is refused with a message that names the release that reads it. Its parameters, from
  `speech_load_params_new()` or NULL for the defaults, are the same for every family:
  - the device: `auto` (the default: the first GPU, or the CPU when there is none), `gpu`, `cpu`, or a device's name,
    compared without case. A device asked for by `gpu` or by name that is not there or does not start is an error, and
    no other device takes its place.
  - the CPU's threads, which apply to whatever runs on the CPU; by default the machine's performance cores, or its
    physical cores where the system does not tell them apart.
  - a warm-up that runs a short request while the model loads, so that a GPU compiles its kernels before the first
    request. It is off by default; `speech worker` and `speech serve` turn it on.
- **Model information.** `speech_model_info_open()` reads what a model file says of its model from its metadata, without
  its weights or a device: its identity, architecture and layout, task and sample rate, languages, voices, whether it
  takes voice files, the codec they must carry and whether a recording alone makes one, the longest text, the options it takes, its sizes, and every metadata
  entry. `speech_model_get_info()` gives the same for a loaded model, with the device, the threads in effect and the
  voices added. `speech_model_info_text_tokens()` counts a text's tokens as a synthesis counts them against the longest
  text, so that a caller can split a long text first. `speech_model_info_json()` writes the whole as one JSON object
  (below).
- **Quantizing.** `speech_quantize()` writes a model file of F32 weights in F16, Q8_0, Q6_K, Q5_K or Q4_K, on the CPU,
  as `speech quantize` does ([cli.md](cli.md#speech-quantize)). Silero VAD's file is F32 alone, and is refused.
- **Voices.** `speech_voice_add()` adds a voice to a loaded model from a voice file, or from a reference WAVE file at any
  rate where `speech_model_info_voice_recordings()` says so. `speech_voice_make()` writes a voice file from a reference,
  reading only the codec from the model file. `speech_voice_make_from()` writes one from the parameters
  `speech_voice_params_new()` makes: several references (`speech_voice_params_add_reference()`), another loudness
  (`speech_voice_params_set_loudness()`) or the recordings' own (`speech_voice_params_keep_loudness()`), or a
  speaker-inversion embedding instead (`speech_voice_params_set_embedding()`); for Drama 3, each reference's words
  (`speech_voice_params_set_transcript()`), voice files made before (`speech_voice_params_add_voice()`) and the speaker
  each reference or voice file is given (`speech_voice_params_set_speaker()`). A model that takes no voice files answers
  all three with `SPEECH_ERROR_UNSUPPORTED` ([models/irodori-tts.md](models/irodori-tts.md#voices),
  [models/drama3.md](models/drama3.md#voices)).
- **Requests.** `speech_request_new()` makes a request for one model. It takes a text (`speech_request_set_text()`) or
  audio (`speech_request_set_audio()`), and options through the setter of each option's type. `speech_synthesize()`
  passes the audio to its callback as it is made, `speech_transcribe()` recognizes the whole audio at once, and
  `speech_detect()` finds where someone speaks in it. A model does one task, and the calls of the others are
  `unsupported`.
  `speech_request_set_progress()` gives a callback that hears how far a request has come while it passes no audio.
  `speech_request_cancel()` stops a request, from any thread, at the next audio, step or stage; a request cancelled
  before it runs returns at once. Either returns `SPEECH_CANCELLED`.
- **Detections of audio that arrives.** `speech_detection_start()` starts a detection with a request's options and the
  rate of the audio to come; `speech_detection_push()` gives it samples, `speech_detection_end()` ends the audio, and
  `speech_detection_region_count()`, `speech_detection_region()` and `speech_detection_speaking()` read what it has found
  so far ([below](#detections)).
- **Results.** `speech_request_result()` gives what a request that returned `SPEECH_OK` or `SPEECH_CANCELLED` did: why it
  stopped (`complete`, `max_seconds`, `model_limit` or `cancelled`), the seed of a synthesis, the samples it passed, and
  the regions of a detection as its segments, each with an empty text and no result text, the text of a recognition with
  its segments and tokens when the request set `timestamps`, and the languages it heard
  (`speech_result_language_count()`, `speech_result_language()`), none for a model that names none, such as
  FastConformer, and none for a cancelled request. The seed is the request's, or one the library drew
  from 0 to 2^53 - 1, with which the same request repeats its audio on the same device.
- **Devices.** `speech_device_count()`, `speech_device_name()`, `speech_device_description()`,
  `speech_device_get_kind()` and `speech_device_memory()` list the CPU and the GPUs a model can run on.
- **Logs.** `speech_log_set()` sends the library's messages and ggml's to a callback. Until it is called, warnings and
  errors go to stderr and the rest is dropped.

## Requests

- Each value is checked against what the model declares as it is set, and the request as a whole when it runs, before
  any work. A refused value or request is left as it was, to be fixed and run again. A request that has started its work
  runs once.
- Audio may be at any common rate. The library resamples a recording to recognize or to detect speech in, and a reference
  recording of a voice, to the model's rate (`speech_model_info_sample_rate()`); the audio it makes is at the model's rate.
- Text is normalized as each model's official code normalizes it, so Japanese copied from a macOS file name gets the
  same tokens as typed text.
- A text longer than `speech_model_info_max_text_tokens()` is `out_of_range`, option `text`.
- Audio shorter than two of the model's mel frames (20 ms for FastConformer) is `out_of_range`, option `audio`. Qwen3-ASR
  pads audio under 0.5 s and takes any.
- A recognition stops at `model_limit` when it reaches the most tokens the model writes (4096 for Qwen3-ASR), with the
  text written up to there.
- Audio that comes out not finite, as from extreme option values, is never passed on: the request ends with
  `out_of_range` naming no option.
- Nothing is cut short or moved to another device behind the caller's back.

## Detections

A detection finds the regions of audio that arrives a piece at a time, with a detection model such as
[Silero VAD](models/silero-vad.md).

- **The regions are `speech_detect()`'s.** Once the audio has ended, they are the regions `speech_detect()` gives for all
  of it, sample for sample, however it was cut into pieces.
- **A region is given once the audio heard makes it certain**, and never changes. With `min_silence_duration_ms` at
  least twice `speech_pad_ms`, as with the defaults, that is when `min_silence_duration_ms` of silence has ended it.
  With less, its end waits on whether speech resumes within twice `speech_pad_ms`, for at most
  2 × `speech_pad_ms` + `min_speech_duration_ms` + `min_silence_duration_ms` and two of the model's chunks (64 ms for
  Silero VAD). A region that `max_speech_duration_s` cuts at an earlier silence is given at the cut at the earliest.
- **`speech_detection_speaking()`** says whether a region has begun that is not given yet, the start it will have, and
  whether it is kept, certain to be given: a region that has ended and whose end waits, as with
  `min_silence_duration_ms` below twice `speech_pad_ms` or after a cut at `max_speech_duration_s`, which is kept, or else
  the region under way, where someone speaks at the end of the audio heard. Speech that ends no longer than
  `min_speech_duration_ms` makes no region, so the region under way is kept once its speech has lasted longer than that
  without falling silent: with the defaults, about 250 ms after it began, where a region is given only after the
  silence that ends it.
- **Any rate.** The audio is resampled to the model's rate a piece at a time, which holds back the last 4.3 ms of the
  audio pushed from 24, 44.1 or 48 kHz (8.6 ms from 8 kHz) until more arrives. The model computes a chunk (32 ms for
  Silero VAD) once its samples have arrived.
- **Refusals.** A request that has audio, and a rate `speech_request_set_audio()` refuses, are `invalid_argument` with
  the option `audio`; a push after the end, or after a push that failed, is `invalid_argument`. A synthesis or
  recognition model is `unsupported`.
- **Threads.** A detection is used by one thread at a time. A push computes while the model does nothing else, so the
  pushes of several detections on one model and its requests wait for each other.

## Errors

A function that can fail returns a `speech_status`, negative for an error, whose category says what kind of failure it
is. `speech_status_name()` gives its name, `speech_last_error()` the message, and `speech_last_error_option()` the input
it concerns (an option's name, `text`, `audio`, `device`, `threads`, `name`, `path`, or a parameter's name for
`speech_voice_make()` or `speech_quantize()`), on the same thread. No C++ exception crosses the API.

| Status | Name | Meaning |
|---|---|---|
| `SPEECH_OK` | `ok` | the call did what it was asked |
| `SPEECH_CANCELLED` | `cancelled` | `speech_request_cancel()` or a callback stopped the request |
| `SPEECH_ERROR_INVALID_ARGUMENT` | `invalid_argument` | the caller's mistake: a NULL pointer, an empty text, a value of the wrong type, a required option left out, a request run again after its work |
| `SPEECH_ERROR_UNSUPPORTED` | `unsupported` | the model cannot do it: an option it does not take, at a value other than the neutral one, or the other task |
| `SPEECH_ERROR_OUT_OF_RANGE` | `out_of_range` | a value the model does not take: outside its range or choices, or a text too long |
| `SPEECH_ERROR_MODEL_FILE` | `model_file` | a model or voice file that cannot be used |
| `SPEECH_ERROR_DEVICE` | `device` | a device that is not there, does not start, or fails while it computes |
| `SPEECH_ERROR_OUT_OF_MEMORY` | `out_of_memory` | the host's or the device's memory ran out |
| `SPEECH_ERROR_IO` | `io` | a file that cannot be opened, read or written |
| `SPEECH_ERROR_INTERNAL` | `internal` | a defect of the library |

## Ownership, threads and versions

- **Ownership.** Every object the library returns is freed only through the function named for it. Every string it
  returns lives as long as the object it was read from: an information's until `speech_model_info_free()`, a result's
  until `speech_request_free()`, the version, a device's and a name's (`speech_option_name()`, `speech_status_name()`,
  `speech_stop_name()`) as long as the process, and the error message until the next call on the thread. Nothing the
  caller passes is kept after the call.
- **Freeing before the process ends.** Free every request and detection, then every model, before the program returns
  from `main()` or calls `exit()`. On Metal, a process that ends with a model still loaded aborts as it ends (exit
  status 134, `GGML_ASSERT([rsets->data count] == 0)`), after its work is done. A binding frees its models when the
  program shuts down, such as from Python's `atexit`, not only when they are collected.
- **Threads.** A model serves one request at a time; requests that several threads run on the same model wait for each
  other and for the pushes of its detections. A request is used by one thread at a time, but `speech_request_cancel()`
  may be called from any thread; a detection is used by one thread at a time. Information never changes once made and
  may be read from any thread. Separate models are independent.
- **Versions.** `speech_version()` gives the release the library was built from (`"0.8.7"`). `SPEECH_API_VERSION_MAJOR`
  and `SPEECH_API_VERSION_MINOR`, and `speech_api_version_major()` and `speech_api_version_minor()` for a caller that
  loads the library at run time, give the API's version, 3.3. The major rises when a declaration changes in a way an
  existing caller notices, and the minor when a function, an option or an enum value is added. A program built against
  M.m runs against a library of the same major version and a minor version of m or more. The shared library's SOVERSION
  is the major version (`libspeech.3.dylib`, `libspeech.so.3`).

## Options

What a request may ask of a model is one vocabulary of options (`speech_option`), each with a name in snake_case
(`speech_option_name()`, `speech_option_from_name()`) and one type. Every entry point uses the same name: a member of the
worker's messages and the server's speech request, and a flag in kebab-case on the command line (`--duration-scale`).
Each model takes some of them, with ranges and defaults from its file, and `speech info MODEL` lists them.

| Option | Type | Neutral | Taken by |
|---|---|---|---|
| `voice` | string | none | [Qwen3-TTS](models/qwen3-tts.md#options), [Irodori-TTS](models/irodori-tts.md#options), [Drama 3](models/drama3.md#options); required |
| `language` | string | `auto` | every model but Silero VAD, which takes `auto` alone |
| `seed` | int | none | Qwen3-TTS, Irodori-TTS, Drama 3 |
| `speed`, `seconds`, `duration_scale`, `steps` | float, float, float, int | 1, none, 1, none | Irodori-TTS |
| `max_seconds` | float | none | Qwen3-TTS, Drama 3 (each chunk of a long text) |
| `timestamps` | bool | false | [FastConformer](models/fastconformer.md#options) |
| `prompt` | string | `""` | [Qwen3-ASR](models/qwen3-asr.md#options) |
| `draft` | string | `""` | [Qwen3-ASR](models/qwen3-asr.md#options) |
| `decoding` | string | none | reazonspeech-nemo-v2 |
| `decode_on_cpu` | bool | none | FastConformer |
| `do_sample`, `top_k`, `top_p`, `temperature`, `repetition_penalty` | bool, int, float, float, float | none | Qwen3-TTS; `top_k`, `top_p` and `temperature` also Drama 3 |
| `code_predictor_do_sample`, `code_predictor_top_k`, `code_predictor_top_p`, `code_predictor_temperature` | bool, int, float, float | none | Qwen3-TTS |
| `instructions` | string | `""` | Qwen3-TTS 1.7B, Irodori-TTS |
| `cfg_scale_text`, `cfg_scale_speaker`, `cfg_guidance_mode`, `cfg_min_t`, `cfg_max_t`, `truncation_factor`, `rescale_k`, `rescale_sigma`, `speaker_uncond_mode`, `sway_coeff`, `speaker_kv_scale`, `speaker_kv_min_t`, `speaker_kv_max_layers`, `cfg_scale_instructions` | float, but `cfg_guidance_mode` and `speaker_uncond_mode` string and `speaker_kv_max_layers` int | none | Irodori-TTS v4.1-Small (RF) |
| `keep_tail`, `tail_window_size`, `tail_std_threshold`, `tail_mean_threshold` | bool, int, float, float | none | Irodori-TTS |
| `threshold`, `min_speech_duration_ms`, `min_silence_duration_ms`, `speech_pad_ms`, `max_speech_duration_s` | float, int, int, int, float | none | [Silero VAD](models/silero-vad.md#options) |

- A value at an option's neutral value is accepted by every model. Any other value of an option a model does not take is
  `unsupported`. An option whose neutral value is "none" has none.
- A string option's value is one of its choices. `voice`, `decoding`, `cfg_guidance_mode` and `speaker_uncond_mode` are
  compared with case; `language` also takes `auto` and a region or script of a choice, compared without case.
- A number outside the range is `out_of_range`.
- An option "steers" when it changes what the model does. A language that is only "checked" is compared with the model's
  languages and then not used.
- A required option left out is `invalid_argument` when the request runs.

## Model information as JSON

`speech_model_info_json()` writes one object, its members in this order. A member that does not apply is left out rather
than null. `speech info --json` prints it, the worker's `ready` and `info` carry it as `model`, and the server's model
object as `speech`, so every program that shows a model shows the same. This is `Qwen3-TTS-12Hz-0.6B-CustomVoice-Q8_0.gguf`
loaded on Metal, with two of its nine voices and four of its options shown:

```json
{
  "name": "Qwen3-TTS-12Hz-0.6B-CustomVoice",
  "organization": "Qwen",
  "basename": "Qwen3-TTS-12Hz",
  "size_label": "0.6B",
  "finetune": "CustomVoice",
  "license": "Apache-2.0",
  "source": {"repository": "https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice", "revision": "85e237c12c027371202489a0ec509ded67b5e4b5"},
  "weight_type": "Q8_0",
  "architecture": "qwen3-tts",
  "layout": 1,
  "task": "synthesis",
  "sample_rate": 24000,
  "incremental": true,
  "languages": ["de", "en", "es", "fr", "it", "ja", "ko", "pt", "ru", "zh"],
  "voices": [
    {"name": "aiden", "language": "en", "gender": "male", "description": "Sunny American male voice."},
    {"name": "ono_anna", "language": "ja", "gender": "female", "description": "Playful Japanese female voice."}
  ],
  "voice_files": false,
  "voice_recordings": false,
  "max_text_tokens": 24565,
  "options": [
    {"name": "voice", "type": "string", "required": true, "steers": true, "choices": ["aiden", "dylan", "eric", "ono_anna", "ryan", "serena", "sohee", "uncle_fu", "vivian"]},
    {"name": "seed", "type": "int", "required": false, "steers": true, "minimum": 0, "maximum": 9007199254740991},
    {"name": "max_seconds", "type": "float", "required": false, "steers": true, "exclusive_minimum": 0, "maximum": 655.36},
    {"name": "temperature", "type": "float", "required": false, "steers": true, "default": 0.9, "exclusive_minimum": 0}
  ],
  "file_bytes": 1213534464,
  "weight_bytes": 1208175044,
  "device": "MTL0",
  "threads": 0
}
```

- `organization` to `weight_type` are the model's identity, from the general keys of its file.
  `finetune` and `version` are left out for a model whose name has none. `source` is the repository the file was
  converted from and the revision. `weight_type` is the type that holds most of the weights: `F32`, `F16`, `Q8_0`, `Q6_K`, `Q5_K` or `Q4_K`.
- `architecture` is the family: `qwen3-tts`, `irodori-tts`, `drama3`, `fastconformer`, `qwen3-asr` or `silero-vad`. `layout` is
  the version of the family's file layout. `task` is `synthesis`, `recognition` or `detection`; a detection model's
  `languages` is empty.
- `incremental` is true for a model that passes audio while it still makes the rest (Qwen3-TTS, Drama 3), so that its first
  audio does not wait on the length of the text, and false for one that makes a request's whole speech before it decodes
  it (Irodori-TTS).
- Each voice's `language`, `gender` and `description` are `""` where the file does not say. A voice added from a voice
  file or a recording says none, but for Drama 3, whose description names the voice's speakers (`speakers 0 and 1`). A
  recognition or detection model has no `voices`, `incremental` or `max_text_tokens`.
- `voice_files` is true for a model that takes voices made from recordings; `voice_codec` then follows it, the hash a
  voice file must carry. `voice_recordings` is true for a model that adds a voice from a WAVE file as well (Irodori-TTS),
  and false for one that needs a voice file (Drama 3, whose voice files hold each recording's words).
- An option has `type` (`string`, `int`, `float`, `bool`), `required`, `steers`, and where they apply `default`,
  `minimum` or `exclusive_minimum`, `maximum`, and `choices`.
- `device` and `threads` are there for a loaded model alone. `threads` is the number of CPU threads in effect for a model
  on the CPU or for a model on a GPU that can compute part of a request on the CPU (FastConformer's decoding with
  `decode_on_cpu`), and 0 for a model on a GPU that computes nothing there. Setting threads is never an error, since
  with the device `auto` a caller cannot know where the model will run.
