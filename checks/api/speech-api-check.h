#pragma once

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#include "speech.h"

/** Opens a file whose path is UTF-8, as fopen() does with `mode`; on Windows fopen() reads the ANSI code page. */
FILE * open_utf8(const char * path, const char * mode);

/** Prints a failure with the last error's status, option and message, and returns 1. */
int fail(const char * what);

/**
 * Whether `got` is `want` and the last error names `option` (NULL for none); prints the outcome, the status's name and
 * the message, and returns 1 when it is as expected.
 */
int expect(speech_status got, speech_status want, const char * option, const char * what);

double now_seconds(void);
void sleep_seconds(double seconds);

/** A thread running `body(arg)`, started and joined in the platform's way. */
typedef struct {
    void * handle;
} Thread;
void thread_start(Thread * thread, void (*body)(void * arg), void * arg);
void thread_join(Thread * thread);

/** A mutex and a condition variable in the platform's way. */
typedef struct {
    void * lock;
    void * changed;
} Monitor;
void monitor_init(Monitor * m);
void monitor_free(Monitor * m);
void monitor_lock(Monitor * m);
void monitor_unlock(Monitor * m);
void monitor_wait(Monitor * m);
void monitor_signal(Monitor * m);

/**
 * Checks the versions, the names of the statuses, the stop reasons and the options, and the devices. Returns nonzero
 * on a failure.
 */
int check_library(void);

/** Checks the load parameters and the loads of `model_path` that are refused. Returns nonzero on a failure. */
int check_load_refusals(const char * model_path);

/**
 * Checks that speech_quantize() refuses what it cannot write, with the category and the input at fault, and that a file
 * whose weights are not F32 is refused before anything is written. Returns nonzero on a failure.
 */
int check_quantize_refusals(const char * model_path);

/**
 * Checks the information of the model file at `path` read without loading against the information of `model`, which
 * was loaded from it and has no voice added: the same JSON object but for the device and the threads, which only the
 * loaded model's has. Checks as well that the identity's accessors give the file's general keys. Returns nonzero on a
 * failure.
 */
int check_info_matches(const char * path, const speech_model * model);

/**
 * Checks that a request for `model` refuses each value of every option that the model does not take, or takes but not
 * at that value, with the category and the option's name, and accepts each option's neutral value. Returns nonzero on
 * a failure.
 */
int check_option_refusals(speech_model * model);

/**
 * The progress a request reports: how many reports, the first and the last, and whether one fell below the one before.
 * It stops the request once `stop_at` is reached when that is above 0.
 */
typedef struct {
    int n, fell;
    double first, last;
    double stop_at;
} Progress;
int record_progress(double done, void * user_data);

/** Whether the recorded progress rose from 0 or more to `last` without falling; prints a failure when not. */
int progress_rises(const Progress * p, double last, const char * what);

/** A whole file, NUL-terminated, or NULL when it cannot be read; `size` is its length. The caller frees it. */
char * read_whole(const char * path, size_t * size);

/**
 * The samples of a dump's audio.npy, a one-dimensional little-endian float32 array in NumPy's format 1.0, or NULL when
 * the file is not one. The caller frees them.
 */
float * read_audio(const char * dump, size_t * n);

/** The checks of a recognition model: `argv` is <model.gguf> <dump folder>... [--device NAME]. Returns the exit status. */

/**
 * The checks of a detection model: `argv` is <model.gguf> <dump folder of reference/silero-vad/dump.py>... [--device
 * NAME]. Returns the exit status.
 */
int check_detection(int argc, char ** argv);

/**
 * The checks of detections of audio given a piece at a time, on a loaded detection model, with the audio of `n_dumps`
 * dumps at the model's rate, named by their folders. Returns nonzero on a failure.
 */
int check_detection_stream(speech_model * model, float * const * audio, const size_t * lengths, const char * const * names, int n_dumps);
int check_recognition(int argc, char ** argv);

/** The sentence a synthesis model speaks in most checks. */
#define SENTENCE "明日の東京は晴れで、最高気温は二十四度の予報です。"

/** The audio of one request, grown as it arrives. */
typedef struct {
    float * samples;
    size_t n, capacity;
} Audio;

/** An audio callback that appends to the Audio at `user_data`. */
int collect(const float * samples, size_t n, void * user_data);

/** A request of `text` in `voice` (NULL for none) with `seed` (negative for none); a failure ends the process. */
speech_request * new_request(speech_model * model, const char * text, const char * voice, int64_t seed);

/**
 * Runs `r`, appending its audio to `audio`, and returns its status; the request is freed. A result that counts other
 * samples than this request passed to the callback is a failure, returned as SPEECH_ERROR_INTERNAL, which no check
 * expects, rather than an exit: main then frees the model, where an exit with the model loaded makes ggml's Metal
 * backend abort on the residency sets it still holds.
 */
speech_status speak(speech_request * r, Audio * audio, speech_stop * stop, int64_t * seed);

/**
 * Runs `request`, which it frees, while another thread cancels it once it has reported progress or passed audio;
 * whether it stopped as cancelled with at most one call of a callback after the cancel.
 */
int cancelled_from_another_thread(speech_request * request, const char * what);

/**
 * Irodori-TTS's rules of a request, its options of the official runtime's request, the voice none and instructions,
 * spoken in `voice`. Returns nonzero on a failure.
 */
int check_irodori(speech_model * model, const speech_model_info * info, const char * voice);

/**
 * Irodori-TTS's voice files of several references, of another loudness and of an embedding, made for the model file
 * at `model_path` beside `made`, which speech_voice_make() made of `reference`: one reference at the model's loudness
 * gives its bytes, the others other files, and what cannot be made is refused naming its input. The embedding's is
 * "<made>.embedding.gguf". Returns nonzero on a failure.
 */
int check_voice_params(const char * model_path, const char * reference, const char * made, const speech_load_params * params);

/**
 * Drama 3's voice files: `made` of `reference` and its `transcript`, "<made>.cast.gguf" put together of it as speakers 0
 * and 1, and the voices that cannot be made refused naming their input. Returns nonzero on a failure.
 */
int check_drama3_voices(const char * model_path, const char * reference, const char * transcript, const char * made, const speech_load_params * params);
