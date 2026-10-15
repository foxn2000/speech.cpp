/*
 * Checks that the C API alone, through the shared libspeech, does what a program needs. It is written in C so that
 * speech.h is checked to be plain C.
 *
 * In every mode: the versions, the names of the statuses, the stop reasons and the options, the devices, the load
 * parameters and loads it refuses, and the quantizing it refuses. With a synthesis model: optionally makes an Irodori-TTS voice file, and voice files
 * of several references, of another loudness and of an embedding, loads the model without a warm-up, compares the
 * information read without loading with the loaded model's, adds the voices, refuses each option's values with the
 * category and the option's name and accepts the neutral ones, speaks one sentence into a WAVE file, repeats a drawn
 * seed's audio, gives the same audio with every option set at its default as with none, checks the family's rules of
 * the whole request (Qwen3-TTS's max_seconds, longest text, sampling and instructions, Drama 3's max_seconds, longest
 * text and temperature, Irodori-TTS's lengths, steps, progress, cut at the tail, guidance, voice none and instructions),
 * cancels a request from another thread and one before it runs, and runs requests from two threads at once.
 * Irodori-TTS's own checks are in speech-api-irodori.c.
 *
 * With a recognition model (transcribe), speech-api-recognition.c, which takes F32 or F16 weights only, and with a
 * detection model (detect), speech-api-detection.c and speech-api-detection-stream.c.
 *
 * With --make-drama3-voice, Drama 3's voice files of a recording and its transcript, of them put together as two
 * speakers, and the ones refused (speech-api-drama3.c), then refuses the recording added alone and speaks both voices.
 *
 * usage: speech-api-check <model.gguf> <out.wav> [--device NAME] [--voice NAME=FILE]...
 *                         [--make-voice <reference.wav> <voice.gguf>]
 *                         [--make-drama3-voice <reference.wav> <transcript> <voice.gguf>]
 *        speech-api-check transcribe <model.gguf> <dump folder>... [--device NAME]
 *        speech-api-check detect <model.gguf> <dump folder>... [--device NAME]
 */

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "speech-api-check.h"
#include "speech.h"

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <shellapi.h>
#endif

#define MAX_VOICES 16

#ifdef _WIN32
/** argv as UTF-8; the C runtime gives it in the ANSI code page, which cannot hold Japanese on most systems. */
static char ** utf8_argv(int * argc) {
    wchar_t ** wide = CommandLineToArgvW(GetCommandLineW(), argc);
    char ** argv = (char **) calloc((size_t) *argc + 1, sizeof(char *));
    for (int i = 0; i < *argc; i++) {
        const int bytes = WideCharToMultiByte(CP_UTF8, 0, wide[i], -1, NULL, 0, NULL, NULL);
        argv[i] = (char *) malloc((size_t) bytes);
        WideCharToMultiByte(CP_UTF8, 0, wide[i], -1, argv[i], bytes, NULL, NULL);
    }
    LocalFree(wide);
    return argv;
}
#endif

static void put_u32(FILE * f, unsigned v) {
    const unsigned char b[4] = {(unsigned char) v, (unsigned char) (v >> 8), (unsigned char) (v >> 16), (unsigned char) (v >> 24)};
    fwrite(b, 1, 4, f);
}

static void put_u16(FILE * f, unsigned v) {
    const unsigned char b[2] = {(unsigned char) v, (unsigned char) (v >> 8)};
    fwrite(b, 1, 2, f);
}

/** Writes 16-bit mono PCM; returns 0 when the file cannot be written. */
static int write_wav(const char * path, const Audio * a, int rate) {
    FILE * f = open_utf8(path, "wb");
    if (!f) return 0;
    const unsigned bytes = (unsigned) (a->n * 2);
    fwrite("RIFF", 1, 4, f);
    put_u32(f, 36 + bytes);
    fwrite("WAVEfmt ", 1, 8, f);
    put_u32(f, 16);
    put_u16(f, 1);
    put_u16(f, 1);
    put_u32(f, (unsigned) rate);
    put_u32(f, (unsigned) rate * 2);
    put_u16(f, 2);
    put_u16(f, 16);
    fwrite("data", 1, 4, f);
    put_u32(f, bytes);
    for (size_t i = 0; i < a->n; i++) {
        float s = a->samples[i];
        s = s < -1.0f ? -1.0f : s > 1.0f ? 1.0f : s;
        const int v = (int) (s * 32767.0f + (s < 0 ? -0.5f : 0.5f));
        put_u16(f, (unsigned) (v & 0xFFFF));
    }
    return fclose(f) == 0;
}

static int log_counts[4];

/** Counts the library's and ggml's messages by level and prints the warnings and errors. */
static void on_log(speech_log_level level, const char * text, void * user_data) {
    (void) user_data;
    if ((int) level >= 0 && (int) level < 4) log_counts[level]++;
    if (level >= SPEECH_LOG_WARN) fputs(text, stderr);
}

/** Adds the voices of the command line after the model's own, and checks the voices a model refuses. */
static int check_voices(speech_model * model, const char * model_path, char ** names, char ** paths, size_t n_voices, const char * recording) {
    speech_model_info * info = NULL;
    if (speech_model_get_info(model, &info) != SPEECH_OK) return fail("speech_model_get_info");
    const size_t own = speech_model_info_voice_count(info);
    const int voice_files = speech_model_info_voice_files(info), recordings = speech_model_info_voice_recordings(info);
    speech_model_info_free(info);
    for (size_t i = 0; i < n_voices; i++) {
        const double start = now_seconds();
        if (speech_voice_add(model, names[i], paths[i]) != SPEECH_OK) return fail("speech_voice_add");
        printf("added the voice %s from %s in %.3f s\n", names[i], paths[i], now_seconds() - start);
    }
    int ok = 1;
    if (!voice_files) {
        ok &= expect(speech_voice_add(model, "added", "voice.gguf"), SPEECH_ERROR_UNSUPPORTED, NULL, "a voice added to a model with its own");
        ok &= expect(speech_voice_make(model_path, "reference.wav", "voice.gguf", NULL), SPEECH_ERROR_UNSUPPORTED, "model_path",
                     "a voice file made for a model with its own voices");
        return ok ? 0 : 1;
    }
    ok &= expect(speech_voice_add(model, names[0], paths[0]), SPEECH_ERROR_INVALID_ARGUMENT, "name", "a voice under a name the model has");
    ok &= expect(speech_voice_add(model, "", paths[0]), SPEECH_ERROR_INVALID_ARGUMENT, "name", "a voice without a name");
    ok &= expect(speech_voice_add(model, "missing", "no-such-folder/voice.gguf"), SPEECH_ERROR_IO, "path", "a voice file that is not there");
    // The voice none speaks without a reference in a file that holds the null speaker, and its name stays its own in
    // one that does not.
    ok &= expect(speech_voice_add(model, "none", paths[0]), SPEECH_ERROR_INVALID_ARGUMENT, "name", "a voice added under the name none");
    if (recording && !recordings) {
        ok &= expect(speech_voice_add(model, "recording", recording), SPEECH_ERROR_INVALID_ARGUMENT, "path",
                     "a recording added to a model that takes voice files alone");
    }
    if (speech_model_get_info(model, &info) != SPEECH_OK) return fail("speech_model_get_info");
    ok &= speech_model_info_voice_count(info) == own + n_voices &&
          !strcmp(speech_model_info_voice_name(info, own + n_voices - 1), names[n_voices - 1]) &&
          !strcmp(speech_model_info_voice_language(info, own), "");
    if (!ok) fprintf(stderr, "FAIL: the information does not list the voices added\n");
    speech_model_info_free(info);
    return ok ? 0 : 1;
}

/**
 * A model that decides the length of its speech as it goes, Qwen3-TTS or Drama 3: max_seconds stops the speech and a
 * refused value leaves it, and a text past the longest is refused and leaves its request to run again with a shorter
 * one.
 */
static int check_length_limits(speech_model * model, const speech_model_info * info, const char * voice) {
    const int rate = speech_model_info_sample_rate(info);
    speech_request * r = new_request(model, "これは最大の長さで止める、少し長めの文です。止まったところで終わります。", voice, 3);
    int ok = expect(speech_request_set_float(r, SPEECH_OPT_MAX_SECONDS, 0.5), SPEECH_OK, NULL, "max_seconds 0.5");
    ok &= expect(speech_request_set_float(r, SPEECH_OPT_MAX_SECONDS, 1e9), SPEECH_ERROR_OUT_OF_RANGE, "max_seconds", "max_seconds 1e9");
    Audio audio = {NULL, 0, 0};
    speech_stop stop;
    if (speak(r, &audio, &stop, NULL) != SPEECH_OK) return fail("a request with max_seconds");
    printf("max_seconds 0.5 gave %.3f s of audio, stopped at %s\n", (double) audio.n / rate, speech_stop_name(stop));
    ok &= stop == SPEECH_STOP_MAX_SECONDS && audio.n > 0 && audio.n <= (size_t) (0.5 * rate);
    if (!ok) fprintf(stderr, "FAIL: max_seconds does not stop the speech within it, or a refused value replaced it\n");
    free(audio.samples);

    size_t unit = 0, total = 0;
    if (speech_model_info_text_tokens(info, SENTENCE, &unit) != SPEECH_OK) return fail("speech_model_info_text_tokens");
    const size_t copies = speech_model_info_max_text_tokens(info) / unit + 2, length = strlen(SENTENCE);
    char * text = (char *) malloc(copies * length + 1);
    for (size_t i = 0; i < copies; i++) memcpy(text + i * length, SENTENCE, length);
    text[copies * length] = '\0';
    if (speech_model_info_text_tokens(info, text, &total) != SPEECH_OK) return fail("speech_model_info_text_tokens");
    printf("a text of %zu copies of the sentence is %zu tokens, the model takes %zu\n", copies, total, speech_model_info_max_text_tokens(info));
    Audio none = {NULL, 0, 0};
    r = new_request(model, text, voice, 4);
    ok &= total > speech_model_info_max_text_tokens(info) &&
          expect(speech_synthesize(r, collect, &none), SPEECH_ERROR_OUT_OF_RANGE, "text", "a text past the longest") && none.n == 0 &&
          speech_request_set_text(r, "短い文です。") == SPEECH_OK &&
          expect(speak(r, &none, NULL, NULL), SPEECH_OK, NULL, "the same request run again with a shorter text") && none.n > 0;
    if (!ok) fprintf(stderr, "FAIL: a text past the longest is not refused, or its request cannot run again with a shorter one\n");
    free(text);
    free(none.samples);
    return ok ? 0 : 1;
}

/**
 * Qwen3-TTS's sampling: with neither stack drawing, the seed changes nothing; a stack's top_k, top_p or temperature
 * set while it does not draw is refused and leaves the request to run once fixed; and a temperature that the
 * sampler's float cannot hold is refused before any work.
 */
static int check_qwen3_tts_sampling(speech_model * model, const char * voice) {
    Audio a = {NULL, 0, 0}, b = {NULL, 0, 0};
    speech_request * r = new_request(model, "同じ音です。", voice, 1);
    speech_request_set_float(r, SPEECH_OPT_MAX_SECONDS, 2);
    int ok = expect(speech_request_set_bool(r, SPEECH_OPT_DO_SAMPLE, 0), SPEECH_OK, NULL, "do_sample false") &&
             expect(speech_request_set_bool(r, SPEECH_OPT_CODE_PREDICTOR_DO_SAMPLE, 0), SPEECH_OK, NULL, "code_predictor_do_sample false") &&
             expect(speak(r, &a, NULL, NULL), SPEECH_OK, NULL, "a request that draws nothing, seed 1");
    r = new_request(model, "同じ音です。", voice, 2);
    speech_request_set_float(r, SPEECH_OPT_MAX_SECONDS, 2);
    speech_request_set_bool(r, SPEECH_OPT_DO_SAMPLE, 0);
    speech_request_set_bool(r, SPEECH_OPT_CODE_PREDICTOR_DO_SAMPLE, 0);
    ok &= expect(speak(r, &b, NULL, NULL), SPEECH_OK, NULL, "a request that draws nothing, seed 2");
    if (ok && (a.n == 0 || a.n != b.n || memcmp(a.samples, b.samples, a.n * sizeof(float)) != 0)) {
        fprintf(stderr, "FAIL: a request that draws nothing gives other audio for another seed\n");
        ok = 0;
    }
    if (ok) printf("a request that draws nothing gives the same %zu samples for the seeds 1 and 2\n", a.n);

    const speech_option unused[] = {SPEECH_OPT_TOP_K, SPEECH_OPT_TOP_P, SPEECH_OPT_TEMPERATURE, SPEECH_OPT_CODE_PREDICTOR_TOP_K,
                                    SPEECH_OPT_CODE_PREDICTOR_TOP_P, SPEECH_OPT_CODE_PREDICTOR_TEMPERATURE};
    for (size_t i = 0; i < sizeof unused / sizeof unused[0]; i++) {
        const speech_option o = unused[i];
        const int talker = o == SPEECH_OPT_TOP_K || o == SPEECH_OPT_TOP_P || o == SPEECH_OPT_TEMPERATURE;
        const speech_option do_sample = talker ? SPEECH_OPT_DO_SAMPLE : SPEECH_OPT_CODE_PREDICTOR_DO_SAMPLE;
        char what[128];
        snprintf(what, sizeof what, "%s with %s false", speech_option_name(o), speech_option_name(do_sample));
        r = new_request(model, "はい。", voice, 3);
        speech_request_set_bool(r, do_sample, 0);
        ok &= expect(speech_option_type(o) == SPEECH_TYPE_INT ? speech_request_set_int(r, o, 5) : speech_request_set_float(r, o, 0.5), SPEECH_OK,
                     NULL, speech_option_name(o));
        b.n = 0;
        ok &= expect(speech_synthesize(r, collect, &b), SPEECH_ERROR_INVALID_ARGUMENT, speech_option_name(o), what) && b.n == 0 &&
              speech_request_set_bool(r, do_sample, 1) == SPEECH_OK &&
              expect(speak(r, &b, NULL, NULL), SPEECH_OK, NULL, "the same request run again drawing") && b.n > 0;
    }
    r = new_request(model, "はい。", voice, 3);
    ok &= expect(speech_request_set_float(r, SPEECH_OPT_TEMPERATURE, 1e-50), SPEECH_OK, NULL, "temperature 1e-50") &&
          expect(speak(r, &b, NULL, NULL), SPEECH_ERROR_OUT_OF_RANGE, "temperature", "a temperature that a float rounds to 0");
    if (!ok) fprintf(stderr, "FAIL: Qwen3-TTS's sampling options are not followed\n");
    free(a.samples);
    free(b.samples);
    return ok ? 0 : 1;
}

/**
 * Drama 3's temperature, which every draw of a frame divides by: one that the sampler's float rounds to 0 is refused
 * before any work, naming the option, and leaves the request to run once fixed.
 */
static int check_drama3_sampling(speech_model * model, const char * voice) {
    Audio a = {NULL, 0, 0};
    speech_request * r = new_request(model, "はい。", voice, 3);
    int ok = expect(speech_request_set_float(r, SPEECH_OPT_TEMPERATURE, 1e-50), SPEECH_OK, NULL, "temperature 1e-50") &&
             expect(speech_synthesize(r, collect, &a), SPEECH_ERROR_OUT_OF_RANGE, "temperature", "a temperature that a float rounds to 0") &&
             a.n == 0 && speech_request_set_float(r, SPEECH_OPT_TEMPERATURE, 1) == SPEECH_OK &&
             expect(speak(r, &a, NULL, NULL), SPEECH_OK, NULL, "the same request run again with temperature 1") && a.n > 0;
    if (!ok) fprintf(stderr, "FAIL: Drama 3's temperature is not followed\n");
    free(a.samples);
    return ok ? 0 : 1;
}

/**
 * Drama 3's voices made by check_drama3_voices(): the information names their speakers, and a scene of the two speakers
 * of the cast speaks. Nothing to check where no voice was made.
 */
static int check_drama3_cast(speech_model * model, const speech_model_info * info, int made) {
    if (!made) return 0;
    const size_t n = speech_model_info_voice_count(info);
    int ok = n >= 2 && !strcmp(speech_model_info_voice_name(info, n - 2), "made") && !strcmp(speech_model_info_voice_name(info, n - 1), "cast");
    if (ok) {
        const char * one = speech_model_info_voice_description(info, n - 2), * two = speech_model_info_voice_description(info, n - 1);
        ok = strchr(one, '0') && !strchr(one, '1') && strchr(two, '0') && strchr(two, '1');
    }
    if (!ok) fprintf(stderr, "FAIL: the information does not name the speakers of Drama 3's voices\n");
    Audio a = {NULL, 0, 0};
    speech_request * r = new_request(model, "<|speaker:0|> こんにちは。<|speaker:1|> こんばんは。", "cast", 5);
    ok &= expect(speak(r, &a, NULL, NULL), SPEECH_OK, NULL, "a scene of the cast's two speakers") && a.n > 0;
    if (ok) printf("the cast's voices are described by their speakers, and a scene of both spoke %zu samples\n", a.n);
    free(a.samples);
    return ok ? 0 : 1;
}

/**
 * Qwen3-TTS's instructions, for a model that takes them: one changes the audio of the same seed, and one whose tokens
 * leave the text no room is refused naming the option and leaves the request to run once it is cleared.
 * check_option_refusals() covers a model that takes none.
 */
static int check_qwen3_tts_instructions(speech_model * model, const speech_model_info * info, const char * voice) {
    if (!speech_model_info_takes(info, SPEECH_OPT_INSTRUCTIONS)) return 0;
    Audio plain = {NULL, 0, 0}, told = {NULL, 0, 0};
    speech_request * r = new_request(model, "同じ文です。", voice, 5);
    speech_request_set_float(r, SPEECH_OPT_MAX_SECONDS, 2);
    int ok = expect(speak(r, &plain, NULL, NULL), SPEECH_OK, NULL, "a request without an instruction");
    r = new_request(model, "同じ文です。", voice, 5);
    speech_request_set_float(r, SPEECH_OPT_MAX_SECONDS, 2);
    ok &= expect(speech_request_set_string(r, SPEECH_OPT_INSTRUCTIONS, "怒った口調で話してください。"), SPEECH_OK, NULL, "an instruction") &&
          expect(speak(r, &told, NULL, NULL), SPEECH_OK, NULL, "the same request with an instruction");
    if (ok && plain.n == told.n && memcmp(plain.samples, told.samples, plain.n * sizeof(float)) == 0) {
        fprintf(stderr, "FAIL: an instruction leaves the audio of the same seed as it was\n");
        ok = 0;
    }
    if (ok) printf("an instruction gives %zu samples where the same seed gave %zu without it\n", told.n, plain.n);

    size_t unit = 0;
    if (speech_model_info_text_tokens(info, SENTENCE, &unit) != SPEECH_OK) return fail("speech_model_info_text_tokens");
    const size_t copies = speech_model_info_max_text_tokens(info) / unit + 1, length = strlen(SENTENCE);
    char * instruction = (char *) malloc(copies * length + 1);
    for (size_t i = 0; i < copies; i++) memcpy(instruction + i * length, SENTENCE, length);
    instruction[copies * length] = '\0';
    Audio none = {NULL, 0, 0};
    r = new_request(model, "はい。", voice, 6);
    ok &= expect(speech_request_set_string(r, SPEECH_OPT_INSTRUCTIONS, instruction), SPEECH_OK, NULL, "an instruction of any length") &&
          expect(speech_synthesize(r, collect, &none), SPEECH_ERROR_OUT_OF_RANGE, "instructions", "an instruction that leaves the text no room") &&
          none.n == 0 && speech_request_set_string(r, SPEECH_OPT_INSTRUCTIONS, "") == SPEECH_OK &&
          expect(speak(r, &none, NULL, NULL), SPEECH_OK, NULL, "the same request run again without the instruction") && none.n > 0;
    if (!ok) fprintf(stderr, "FAIL: Qwen3-TTS's instructions are not followed or not limited\n");
    free(instruction);
    free(plain.samples);
    free(told.samples);
    free(none.samples);
    return ok ? 0 : 1;
}

/**
 * A request that sets every option the model declares with a default, at that default, gives the audio of one that
 * sets none, for the same seed: the defaults the information shows are the ones a request runs with.
 */
static int check_defaults(speech_model * model, const speech_model_info * info, const char * voice) {
    Audio none = {NULL, 0, 0}, all = {NULL, 0, 0};
    if (speak(new_request(model, "三つ目です。", voice, 21), &none, NULL, NULL) != SPEECH_OK) return fail("a request that sets no option");
    speech_request * r = new_request(model, "三つ目です。", voice, 21);
    for (size_t i = 0; i < speech_model_info_option_count(info); i++) {
        const speech_option o = speech_model_info_option(info, i);
        if (!speech_model_info_option_has_default(info, o)) continue;
        const char * text = NULL;
        int64_t integer = 0;
        double number = 0;
        int boolean = 0;
        speech_status s = SPEECH_OK;
        switch (speech_option_type(o)) {
            case SPEECH_TYPE_STRING:
                s = speech_model_info_option_default_string(info, o, &text);
                if (s == SPEECH_OK) s = speech_request_set_string(r, o, text);
                break;
            case SPEECH_TYPE_INT:
                s = speech_model_info_option_default_int(info, o, &integer);
                if (s == SPEECH_OK) s = speech_request_set_int(r, o, integer);
                break;
            case SPEECH_TYPE_FLOAT:
                s = speech_model_info_option_default_float(info, o, &number);
                if (s == SPEECH_OK) s = speech_request_set_float(r, o, number);
                break;
            case SPEECH_TYPE_BOOL:
                s = speech_model_info_option_default_bool(info, o, &boolean);
                if (s == SPEECH_OK) s = speech_request_set_bool(r, o, boolean);
                break;
        }
        if (s != SPEECH_OK) return fail("an option set at its default");
    }
    if (speak(r, &all, NULL, NULL) != SPEECH_OK) return fail("a request that sets every option at its default");
    const int same = none.n > 0 && none.n == all.n && memcmp(none.samples, all.samples, none.n * sizeof(float)) == 0;
    if (same) printf("a request that sets every option at its default gives the same %zu samples as one that sets none\n", none.n);
    else fprintf(stderr, "FAIL: a request that sets every option at its default gives other audio than one that sets none\n");
    free(none.samples);
    free(all.samples);
    return same ? 0 : 1;
}

/** A request run on a thread of its own, for two at once. */
typedef struct {
    speech_request * request;
    Audio audio;
    speech_status status;
} Concurrent;

static void run_concurrent(void * arg) {
    Concurrent * c = (Concurrent *) arg;
    c->status = speech_synthesize(c->request, collect, &c->audio);
}

/** Cancels a request from another thread and one before it runs, and runs two at once. */
static int check_threads(speech_model * model, const char * voice) {
    int ok = cancelled_from_another_thread(new_request(model, "これは途中で止める長めの文です。止まったら、残りの音声は届きません。", voice, 8),
                                           "a request cancelled from another thread");

    speech_request * early = new_request(model, SENTENCE, voice, 9);
    speech_request_cancel(early);
    Audio none = {NULL, 0, 0};
    speech_stop stop;
    ok &= expect(speak(early, &none, &stop, NULL), SPEECH_CANCELLED, NULL, "a request cancelled before it runs") && none.n == 0 &&
          stop == SPEECH_STOP_CANCELLED;

    Concurrent two[2];
    Thread threads[2];
    for (int i = 0; i < 2; i++) {
        memset(&two[i], 0, sizeof two[i]);
        two[i].request = new_request(model, i ? "二つ目です。" : "一つ目です。", voice, 10 + i);
        thread_start(&threads[i], run_concurrent, &two[i]);
    }
    for (int i = 0; i < 2; i++) {
        thread_join(&threads[i]);
        ok &= expect(two[i].status, SPEECH_OK, NULL, "one of two requests run at once") && two[i].audio.n > 0;
        speech_request_free(two[i].request);
        free(two[i].audio.samples);
    }
    if (ok) printf("two requests from two threads ran one after another\n");
    else fprintf(stderr, "FAIL: a request is not cancelled alone, or two requests at once do not both run\n");
    return ok ? 0 : 1;
}

/** The checks of a loaded synthesis model, which main frees whatever they find. */
static int check_model(speech_model * model, const char * model_path, const char * out_wav, char ** names, char ** paths, size_t n_voices,
                       int made, const char * recording) {
    if (check_info_matches(model_path, model) != 0) return 1;
    speech_model_info * info = NULL;
    if (speech_model_get_info(model, &info) != SPEECH_OK) return fail("speech_model_get_info");
    const int irodori = !strcmp(speech_model_info_architecture(info), "irodori-tts");
    const int drama3 = !strcmp(speech_model_info_architecture(info), "drama3");
    if (speech_model_info_task(info) != SPEECH_TASK_SYNTHESIS || speech_model_info_max_text_tokens(info) == 0 ||
        (irodori && (!speech_model_info_voice_codec(info) || n_voices == 0)) || (!irodori && speech_model_info_voice_count(info) == 0) ||
        speech_model_info_voice_recordings(info) != irodori) {
        fprintf(stderr, "FAIL: a synthesis model is not described as speech.h declares, or Irodori-TTS was given no voice\n");
        return 1;
    }
    if (check_voices(model, model_path, names, paths, n_voices, recording) != 0 || check_option_refusals(model) != 0) return 1;
    speech_model_info_free(info);
    if (speech_model_get_info(model, &info) != SPEECH_OK) return fail("speech_model_get_info");
    const int rate = speech_model_info_sample_rate(info);
    // Irodori-TTS speaks in the first voice added, since its own voice none has no reference.
    const char * voice = made ? "made" : irodori ? names[0] : speech_model_info_voice_name(info, 0);

    size_t tokens = 0;
    if (speech_model_info_text_tokens(info, SENTENCE, &tokens) != SPEECH_OK) return fail("speech_model_info_text_tokens");
    speech_request * r = new_request(model, SENTENCE, voice, 7);
    if (speech_request_set_string(r, SPEECH_OPT_LANGUAGE, "ja") != SPEECH_OK) return fail("the language ja");
    Audio audio = {NULL, 0, 0};
    speech_stop stop;
    int64_t seed = 0;
    const double start = now_seconds();
    if (speak(r, &audio, &stop, &seed) != SPEECH_OK) return fail("speech_synthesize");
    printf("spoke %zu tokens as %.2f s of audio in %.2f s in the voice %s, stopped at %s\n", tokens, (double) audio.n / rate,
           now_seconds() - start, voice, speech_stop_name(stop));
    if (audio.n == 0 || stop != SPEECH_STOP_COMPLETE || seed != 7 || !write_wav(out_wav, &audio, rate)) {
        fprintf(stderr, "FAIL: the sentence gave no audio, did not complete, did not report its seed, or %s cannot be written\n", out_wav);
        return 1;
    }

    Audio drawn = {NULL, 0, 0}, again = {NULL, 0, 0};
    if (speak(new_request(model, "二つ目の文です。", voice, -1), &drawn, NULL, &seed) != SPEECH_OK) return fail("a request without a seed");
    if (speak(new_request(model, "二つ目の文です。", voice, seed), &again, NULL, NULL) != SPEECH_OK) return fail("a request with the drawn seed");
    if (seed < 0 || seed > 9007199254740991LL || drawn.n != again.n || memcmp(drawn.samples, again.samples, drawn.n * sizeof(float)) != 0) {
        fprintf(stderr, "FAIL: the seed %lld that the library drew does not give the same audio again\n", (long long) seed);
        return 1;
    }
    printf("the drawn seed %lld gives the same %zu samples again\n", (long long) seed, again.n);
    if (made && irodori) {
        Audio embedded = {NULL, 0, 0};
        if (speak(new_request(model, SENTENCE, "embedded", 7), &embedded, NULL, NULL) != SPEECH_OK || embedded.n == 0) {
            return fail("a request in the voice of an embedding");
        }
        printf("the voice of an embedding spoke %.2f s\n", (double) embedded.n / rate);
        free(embedded.samples);
    }

    r = new_request(model, "どの値も中立です。", voice, 12);
    int ok = expect(speech_request_set_float(r, SPEECH_OPT_SPEED, 1), SPEECH_OK, NULL, "speed 1") &&
             expect(speech_request_set_float(r, SPEECH_OPT_DURATION_SCALE, 1), SPEECH_OK, NULL, "duration_scale 1") &&
             expect(speech_request_set_string(r, SPEECH_OPT_LANGUAGE, "auto"), SPEECH_OK, NULL, "language auto") &&
             expect(speech_request_set_bool(r, SPEECH_OPT_TIMESTAMPS, 0), SPEECH_OK, NULL, "timestamps false");
    audio.n = 0;
    ok &= expect(speak(r, &audio, NULL, NULL), SPEECH_OK, NULL, "a request of neutral values");
    r = new_request(model, NULL, voice, 13);
    ok &= expect(speech_request_set_audio(r, audio.samples, audio.n, rate), SPEECH_ERROR_UNSUPPORTED, "audio", "audio for a synthesis");
    ok &= expect(speech_transcribe(r), SPEECH_ERROR_UNSUPPORTED, NULL, "speech_transcribe() of a synthesis model");
    ok &= expect(speech_detect(r), SPEECH_ERROR_UNSUPPORTED, NULL, "speech_detect() of a synthesis model");
    speech_detection * detection = NULL;
    ok &= expect(speech_detection_start(r, rate, &detection), SPEECH_ERROR_UNSUPPORTED, NULL, "a detection of a synthesis model") &&
          detection == NULL;
    ok &= expect(speak(r, &audio, NULL, NULL), SPEECH_ERROR_INVALID_ARGUMENT, "text", "a request without a text");
    // A request refused before its work is fixed and run again; one that has done its work runs once.
    r = new_request(model, "一回だけです。", NULL, 14);
    audio.n = 0;
    ok &= expect(speech_synthesize(r, collect, &audio), SPEECH_ERROR_INVALID_ARGUMENT, "voice", "a request without a voice") &&
          speech_request_set_string(r, SPEECH_OPT_VOICE, voice) == SPEECH_OK &&
          expect(speech_synthesize(r, collect, &audio), SPEECH_OK, NULL, "the same request run again with a voice") && audio.n > 0 &&
          expect(speech_synthesize(r, collect, &audio), SPEECH_ERROR_INVALID_ARGUMENT, NULL, "the same request run again after its work");
    speech_request_free(r);
    if (!ok) return 1;
    free(audio.samples);
    free(drawn.samples);
    free(again.samples);

    if (check_defaults(model, info, voice) != 0) return 1;
    if (irodori ? check_irodori(model, info, voice) != 0
        : drama3 ? check_length_limits(model, info, voice) != 0 || check_drama3_sampling(model, voice) != 0 || check_drama3_cast(model, info, made)
                 : check_length_limits(model, info, voice) != 0 || check_qwen3_tts_sampling(model, voice) != 0 ||
                       check_qwen3_tts_instructions(model, info, voice) != 0) {
        return 1;
    }
    if (check_threads(model, voice) != 0) return 1;
    speech_model_info_free(info);
    return 0;
}

int main(int argc, char ** argv) {
#ifdef _WIN32
    argv = utf8_argv(&argc);
#endif
    const int recognition = argc > 1 && !strcmp(argv[1], "transcribe"), detection = argc > 1 && !strcmp(argv[1], "detect");
    if (argc < 3) {
        fprintf(stderr,
                "usage: %s <model.gguf> <out.wav> [--device NAME] [--voice NAME=FILE]... "
                "[--make-voice <reference.wav> <voice.gguf>] [--make-drama3-voice <reference.wav> <transcript> <voice.gguf>]\n"
                "       %s transcribe <model.gguf> <dump folder>... [--device NAME]\n"
                "       %s detect <model.gguf> <dump folder>... [--device NAME]\n",
                argv[0], argv[0], argv[0]);
        return 2;
    }
    speech_log_set(on_log, NULL);
    const char * checked = argv[recognition || detection ? 2 : 1];
    if (check_library() != 0 || check_load_refusals(checked) != 0 || check_quantize_refusals(checked) != 0) return 1;
    if (recognition) return check_recognition(argc - 2, argv + 2);
    if (detection) return check_detection(argc - 2, argv + 2);
    const char * model_path = argv[1];

    speech_load_params * params = NULL;
    if (speech_load_params_new(&params) != SPEECH_OK) return fail("speech_load_params_new");
    char * names[MAX_VOICES], * paths[MAX_VOICES];
    size_t n_voices = 0;
    const char * reference = NULL, * made = NULL, * transcript = NULL;
    for (int i = 3; i < argc; i++) {
        if (!strcmp(argv[i], "--device") && i + 1 < argc) {
            if (speech_load_params_set_device(params, argv[++i]) != SPEECH_OK) return fail("speech_load_params_set_device");
        } else if (!strcmp(argv[i], "--voice") && i + 1 < argc && n_voices < MAX_VOICES - 1) {
            char * eq = strchr(argv[++i], '=');
            if (!eq) {
                fprintf(stderr, "--voice takes NAME=FILE\n");
                return 2;
            }
            *eq = '\0';
            names[n_voices] = argv[i];
            paths[n_voices++] = eq + 1;
        } else if (!strcmp(argv[i], "--make-voice") && i + 2 < argc) {
            reference = argv[++i];
            made = argv[++i];
        } else if (!strcmp(argv[i], "--make-drama3-voice") && i + 3 < argc) {
            reference = argv[++i];
            transcript = argv[++i];
            made = argv[++i];
        } else {
            fprintf(stderr, "unknown or incomplete option %s\n", argv[i]);
            return 2;
        }
    }
    if (made && transcript) {
        if (check_drama3_voices(model_path, reference, transcript, made, params) != 0) return 1;
        static char cast[4096];
        snprintf(cast, sizeof cast, "%s.cast.gguf", made);
        names[n_voices] = (char *) "made";
        paths[n_voices++] = (char *) made;
        names[n_voices] = (char *) "cast";
        paths[n_voices++] = cast;
    } else if (made) {
        const double start = now_seconds();
        if (speech_voice_make(model_path, reference, made, params) != SPEECH_OK) return fail("speech_voice_make");
        printf("made the voice file %s in %.3f s\n", made, now_seconds() - start);
        if (check_voice_params(model_path, reference, made, params) != 0) return 1;
        names[n_voices] = (char *) "made";
        paths[n_voices++] = (char *) made;
        static char embedded[4096];
        snprintf(embedded, sizeof embedded, "%s.embedding.gguf", made);
        names[n_voices] = (char *) "embedded";
        paths[n_voices++] = embedded;
    }

    speech_model * model = NULL;
    double start = now_seconds();
    if (speech_model_load(model_path, params, &model) != SPEECH_OK) return fail("speech_model_load");
    speech_load_params_free(params);
    printf("loaded without a warm-up in %.2f s\n", now_seconds() - start);
    const int status = check_model(model, model_path, argv[2], names, paths, n_voices, made != NULL, reference);
    speech_model_free(model);
    if (status == 0) {
        printf("log messages: %d debug, %d info, %d warnings, %d errors\n", log_counts[0], log_counts[1], log_counts[2], log_counts[3]);
        printf("ok\n");
    }
    return status;
}
