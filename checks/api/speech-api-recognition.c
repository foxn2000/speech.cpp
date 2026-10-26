/*
 * The recognition half of speech-api-check: the C API with a recognition model, through the shared libspeech. Loads the
 * model without a warm-up, compares the information read without loading with the loaded model's, refuses each option's
 * values with the category and the option's name, recognizes the audio of each dump of reference/fastconformer/dump.py
 * or reference/qwen3-asr/dump.py and compares the text with the dump's text byte for byte, the stop with the dump's, at
 * the model's limit where qwen-asr's decoding of a part reached max_new_tokens, and the languages with the one qwen-asr
 * parsed, or those it merged from the parts of an input it split, by their tags, or with none for a dump of
 * FastConformer, which writes no language, gives each its tokens and segments with their times where the model takes
 * timestamps, recognizes the audio of each dump with the dump's text as its draft where the model takes one and
 * compares the text, recognizes each other request of a Qwen3-ASR dump of an input qwen-asr did not split, with its
 * language forced and with its prompt, recognizes the audio with each decoding other than the model's default and
 * compares the text and the languages with those in the dump's folder of the decoding's name, which
 * reference/fastconformer/dump.py --greedy writes for an RNN-T's greedy decoding, recognizes the audio with each
 * decoding and decode_on_cpu the other way than the model's default where it takes it and compares the text with the
 * dump's of the decoding, recognizes audio at three times the model's rate, refuses what a recognition cannot take and
 * runs a request it refused before its work again, reports and stops on progress, and cancels a request from another
 * thread while it runs, its result then without text or language.
 */

#include <stdlib.h>
#include <string.h>

#include "speech-api-check.h"
#include "speech.h"

#define MAX_DUMPS 16

/**
 * What a recognition asks besides its audio: timestamps, and a language, a prompt and a decoding where they are not
 * NULL, decode_on_cpu where it is 0 or 1 rather than -1, and a draft where it is not NULL.
 */
typedef struct {
    int timestamps;
    const char * language;
    const char * prompt;
    const char * decoding;
    int decode_on_cpu;
    const char * draft;
} Asked;

/**
 * The languages of a result as tags joined with commas, "" for none, which the caller frees, or NULL when the index
 * past the last gives a language.
 */
static char * languages_of(const speech_result * result) {
    const size_t count = speech_result_language_count(result);
    size_t length = 1;
    for (size_t i = 0; i < count; i++) length += strlen(speech_result_language(result, i)) + 1;
    char * out = (char *) calloc(length, 1);
    for (size_t i = 0; i < count; i++) {
        if (i) strcat(out, ",");
        strcat(out, speech_result_language(result, i));
    }
    if (speech_result_language(result, count) != NULL) {
        free(out);
        return NULL;
    }
    return out;
}

/**
 * Recognizes `n` samples at `rate` as `asked` asks, returning the status, the result's text in `text` and its languages
 * as languages_of() gives them in `languages` when that is not NULL (which the caller frees), and its stop reason in
 * `stop` when that is not NULL, checking the tokens and segments against the text when they are asked for.
 */
static speech_status recognize(speech_model * model, const float * samples, size_t n, int rate, Asked asked, Progress * progress,
                               char ** text, char ** languages, speech_stop * stop) {
    speech_request * r = NULL;
    *text = NULL;
    if (languages) *languages = NULL;
    if (speech_request_new(model, &r) != SPEECH_OK || speech_request_set_audio(r, samples, n, rate) != SPEECH_OK ||
        speech_request_set_bool(r, SPEECH_OPT_TIMESTAMPS, asked.timestamps) != SPEECH_OK ||
        (asked.language && speech_request_set_string(r, SPEECH_OPT_LANGUAGE, asked.language) != SPEECH_OK) ||
        (asked.prompt && speech_request_set_string(r, SPEECH_OPT_PROMPT, asked.prompt) != SPEECH_OK) ||
        (asked.decoding && speech_request_set_string(r, SPEECH_OPT_DECODING, asked.decoding) != SPEECH_OK) ||
        (asked.decode_on_cpu >= 0 && speech_request_set_bool(r, SPEECH_OPT_DECODE_ON_CPU, asked.decode_on_cpu) != SPEECH_OK) ||
        (asked.draft && speech_request_set_string(r, SPEECH_OPT_DRAFT, asked.draft) != SPEECH_OK) ||
        (progress && speech_request_set_progress(r, record_progress, progress) != SPEECH_OK)) {
        fail("a request's audio or option");
        speech_request_free(r);
        return SPEECH_ERROR_INTERNAL;
    }
    const speech_status status = speech_transcribe(r);
    const speech_result * result = speech_request_result(r);
    const int timestamps = asked.timestamps;
    if (result) {
        const char * t = speech_result_text(result);
        *text = (char *) malloc(strlen(t) + 1);
        strcpy(*text, t);
        if (stop) *stop = speech_result_stop(result);
        if (languages && !(*languages = languages_of(result))) {
            fprintf(stderr, "FAIL: the result gives a language past its last\n");
            speech_request_free(r);
            return SPEECH_ERROR_INTERNAL;
        }
    }
    if (status == SPEECH_OK && timestamps) {
        const size_t length = strlen(*text);
        char * joined_tokens = (char *) calloc(length + 1, 1), * joined_segments = (char *) calloc(length + 1, 1);
        const double seconds = (double) n / rate;
        double last = 0;
        int ok = 1;
        for (size_t i = 0; i < speech_result_token_count(result); i++) {
            double start = 0, end = 0;
            const char * piece = NULL;
            ok = ok && speech_result_token(result, i, &start, &end, &piece) == SPEECH_OK && start >= last - 1e-9 && end >= start &&
                 end <= seconds + 0.5 && strlen(joined_tokens) + strlen(piece) <= length;
            if (!ok) break;
            strcat(joined_tokens, piece);
            last = start;
        }
        printf("  %zu tokens, %zu segments:\n", speech_result_token_count(result), speech_result_segment_count(result));
        for (size_t i = 0; ok && i < speech_result_segment_count(result); i++) {
            double start = 0, end = 0;
            const char * piece = NULL;
            ok = speech_result_segment(result, i, &start, &end, &piece) == SPEECH_OK && end >= start && strlen(joined_segments) + strlen(piece) <= length;
            if (ok) {
                strcat(joined_segments, piece);
                printf("  %8.2f %8.2f %s\n", start, end, piece);
            }
        }
        ok = ok && !strcmp(joined_tokens, *text) && !strcmp(joined_segments, *text) && (length == 0 || speech_result_segment_count(result) > 0) &&
             speech_result_segment(result, speech_result_segment_count(result), NULL, NULL, NULL) == SPEECH_ERROR_INVALID_ARGUMENT;
        if (!ok) fprintf(stderr, "FAIL: the tokens and segments do not make the text, or their times do not rise within the audio\n");
        free(joined_tokens);
        free(joined_segments);
        if (!ok) {
            speech_request_free(r);
            return SPEECH_ERROR_INTERNAL;
        }
    }
    speech_request_free(r);
    return status;
}

/** Whether a file can be opened for reading. */
static int exists(const char * path) {
    FILE * f = open_utf8(path, "rb");
    if (f) fclose(f);
    return f != NULL;
}

/**
 * The folder of a dump that holds the text of a request without options: the dump's own for
 * reference/fastconformer/dump.py, and its auto/ for reference/qwen3-asr/dump.py, which keeps each request in a folder
 * of its own.
 */
static void request_folder(const char * dump, const char * request, char * out, size_t size) {
    snprintf(out, size, "%s/%s/meta.json", dump, request);
    if (exists(out)) snprintf(out, size, "%s/%s", dump, request);
    else snprintf(out, size, "%s", dump);
}

/**
 * The string member `key` of the JSON object in `json`, which the caller frees, or NULL when it is absent, not a
 * string, or holds an escape json.dump(ensure_ascii=False) does not write: it escapes '"', '\\' and the control
 * characters alone.
 */
static char * json_string_member(const char * json, const char * key) {
    char pattern[128];
    snprintf(pattern, sizeof pattern, "\"%s\": ", key);
    const char * at = strstr(json, pattern);
    if (!at || at[strlen(pattern)] != '"') return NULL;
    at += strlen(pattern) + 1;
    char * out = (char *) malloc(strlen(at) + 1);
    size_t n = 0;
    for (; *at && *at != '"'; at++) {
        if (*at != '\\') {
            out[n++] = *at;
            continue;
        }
        at++;
        switch (*at) {
            case 'b': out[n++] = '\b'; break;
            case 'f': out[n++] = '\f'; break;
            case 'n': out[n++] = '\n'; break;
            case 'r': out[n++] = '\r'; break;
            case 't': out[n++] = '\t'; break;
            case 'u': {
                char hex[5] = {0};
                for (int k = 0; k < 4 && at[k + 1]; k++) hex[k] = at[k + 1];
                const unsigned long cp = strtoul(hex, NULL, 16);
                if (cp >= 0x20) {
                    free(out);
                    return NULL;
                }
                out[n++] = (char) cp;
                at += 4;
                break;
            }
            case '"':
            case '\\':
            case '/': out[n++] = *at; break;
            default: free(out); return NULL;
        }
    }
    out[n] = '\0';
    return out;
}

/** The index of `item` in a metadata value that is a JSON array of strings without escapes, or -1. */
static int index_in_array(const char * array, const char * item) {
    char quoted[256];
    snprintf(quoted, sizeof quoted, "\"%s\"", item);
    const char * at = strstr(array, quoted);
    if (!at) return -1;
    int index = 0;
    for (const char * c = array; c < at; c++) index += *c == ',';
    return index;
}

/** The value of the metadata entry `key` as JSON text, or NULL. */
static const char * meta_of(const speech_model_info * info, const char * key) {
    for (size_t i = 0; i < speech_model_info_meta_count(info); i++) {
        if (!strcmp(speech_model_info_meta_key(info, i), key)) return speech_model_info_meta_value(info, i);
    }
    return NULL;
}

/** The string member `key` of the meta.json in `folder`, which the caller frees, or NULL as json_string_member() gives it. */
static char * meta_member(const char * folder, const char * key) {
    char path[4096 + 16];
    size_t size = 0;
    snprintf(path, sizeof path, "%s/meta.json", folder);
    char * meta = read_whole(path, &size);
    char * out = meta ? json_string_member(meta, key) : NULL;
    free(meta);
    return out;
}

/**
 * Whether `dump` is one of reference/qwen3-asr/dump.py of an input that qwen-asr split, which holds its split.npy and
 * the request without options alone: under auto/, the texts of the parts joined and their languages merged as
 * transcribe() merges them, its meta.json's language, and each part's request under auto/part<k>/.
 */
static int split_dump(const char * dump) {
    char path[4096 + 16];
    snprintf(path, sizeof path, "%s/split.npy", dump);
    return exists(path);
}

/**
 * The languages qwen-asr gave the request in `folder`, the names joined with commas under `key` in its meta.json, as
 * the tags of the model's languages that qwen3-asr.language_names names so, joined with commas: "" for none and for a
 * dump of reference/fastconformer/dump.py, which has none; the caller frees it. NULL when a name is none of the model's.
 */
static char * dump_languages(const speech_model_info * info, const char * folder, const char * key) {
    char * names = meta_member(folder, key);
    const char * known = meta_of(info, "qwen3-asr.language_names");
    char * out = (char *) calloc(names ? 4 * strlen(names) + 1 : 1, 1);
    for (char * name = names && names[0] ? names : NULL; name && out;) {
        char * comma = strchr(name, ',');
        if (comma) *comma = '\0';
        const int at = known ? index_in_array(known, name) : -1;
        if (at < 0) {
            free(out);
            out = NULL;
        } else {
            if (out[0]) strcat(out, ",");
            strcat(out, speech_model_info_language(info, (size_t) at));
        }
        name = comma ? comma + 1 : NULL;
    }
    free(names);
    return out;
}

/** Whether qwen-asr's decoding of the request in `folder` stopped at max_new_tokens, as its meta.json's stop says. */
static int stopped_at_limit(const char * folder) {
    char * stop = meta_member(folder, "stop");
    const int limited = stop && !strcmp(stop, "max_new_tokens");
    free(stop);
    return limited;
}

/**
 * The stop of the dump's request in `folder`: the model's limit where qwen-asr's decoding stopped at max_new_tokens, or,
 * for the request of a split dump, where any part's did, and complete otherwise and for a dump of
 * reference/fastconformer/dump.py, which has no stop.
 */
static speech_stop dump_stop(const char * folder, int split) {
    int limited = !split && stopped_at_limit(folder);
    for (int k = 0; split; k++) {
        char part[4096 + 16], meta[4096 + 32];
        snprintf(part, sizeof part, "%s/part%d", folder, k);
        snprintf(meta, sizeof meta, "%s/meta.json", part);
        if (!exists(meta)) break;
        limited = limited || stopped_at_limit(part);
    }
    return limited ? SPEECH_STOP_MODEL_LIMIT : SPEECH_STOP_COMPLETE;
}

static int ignore_audio(const float * samples, size_t n, void * user_data) {
    (void) samples;
    (void) n;
    (void) user_data;
    return 0;
}

/** A thread that cancels a request a fixed time after it starts, and notes when it did. */
typedef struct {
    speech_request * request;
    double delay, cancelled_at;
} Canceller;

static void cancel_later(void * arg) {
    Canceller * c = (Canceller *) arg;
    const double start = now_seconds();
    sleep_seconds(c->delay);
    speech_request_cancel(c->request);
    c->cancelled_at = now_seconds() - start;
}

/** What a recognition model refuses: options it does not take, the wrong input, and voices. */
static int check_refusals(speech_model * model, const char * model_path, const float * samples, size_t n, int rate) {
    speech_request * r = NULL;
    if (speech_request_new(model, &r) != SPEECH_OK) return fail("speech_request_new");
    int ok = expect(speech_request_set_audio(r, samples, n, 0), SPEECH_ERROR_INVALID_ARGUMENT, "audio", "audio without a sample rate");
    ok &= expect(speech_request_set_audio(r, samples, n, 44101), SPEECH_ERROR_INVALID_ARGUMENT, "audio", "audio at 44101 Hz");
    ok &= expect(speech_request_set_audio(r, samples, 0, rate), SPEECH_ERROR_INVALID_ARGUMENT, "audio", "no audio");
    ok &= expect(speech_request_set_audio(r, NULL, n, rate), SPEECH_ERROR_INVALID_ARGUMENT, "audio", "NULL samples");
    ok &= expect(speech_request_set_text(r, "明日の東京は晴れです。"), SPEECH_ERROR_UNSUPPORTED, "text", "a text for a recognition");
    ok &= expect(speech_synthesize(r, NULL, NULL), SPEECH_ERROR_INVALID_ARGUMENT, NULL, "speech_synthesize() without a callback");
    ok &= expect(speech_synthesize(r, ignore_audio, NULL), SPEECH_ERROR_UNSUPPORTED, NULL, "speech_synthesize() of a recognition model");
    ok &= expect(speech_transcribe(r), SPEECH_ERROR_INVALID_ARGUMENT, "audio", "a recognition without audio");
    ok &= expect(speech_detect(r), SPEECH_ERROR_UNSUPPORTED, NULL, "speech_detect() of a recognition model");
    speech_detection * detection = NULL;
    ok &= expect(speech_detection_start(r, rate, &detection), SPEECH_ERROR_UNSUPPORTED, NULL, "a detection of a recognition model") &&
          detection == NULL;
    speech_request_free(r);
    speech_model_info * info = NULL;
    if (speech_model_get_info(model, &info) != SPEECH_OK) return fail("speech_model_get_info");
    const int takes_prompt = speech_model_info_takes(info, SPEECH_OPT_PROMPT), takes_timestamps = speech_model_info_takes(info, SPEECH_OPT_TIMESTAMPS);
    // A request refused before its work is fixed and run again: one with a prompt whose tokens, beside the audio's and
    // the most the model writes, are more than the decoder's positions, or, for a model that takes no prompt, one of
    // less audio than the model recognizes.
    char * long_prompt = NULL;
    if (takes_prompt) {
        const size_t words = 70000;
        long_prompt = (char *) malloc(words * 5 + 1);
        for (size_t i = 0; i < words; i++) memcpy(long_prompt + 5 * i, " word", 5);
        long_prompt[words * 5] = '\0';
    }
    if (speech_request_new(model, &r) != SPEECH_OK || speech_request_set_audio(r, samples, takes_prompt ? n : 100, rate) != SPEECH_OK ||
        (takes_prompt && speech_request_set_string(r, SPEECH_OPT_PROMPT, long_prompt) != SPEECH_OK)) {
        return fail("a request");
    }
    ok &= (takes_prompt ? expect(speech_transcribe(r), SPEECH_ERROR_OUT_OF_RANGE, "prompt", "a prompt of 70000 words")
                        : expect(speech_transcribe(r), SPEECH_ERROR_OUT_OF_RANGE, "audio", "100 samples of audio")) &&
          (takes_prompt ? speech_request_set_string(r, SPEECH_OPT_PROMPT, "") : speech_request_set_audio(r, samples, n, rate)) == SPEECH_OK &&
          expect(speech_transcribe(r), SPEECH_OK, NULL, takes_prompt ? "the same request run again without its prompt" : "the same request run again with its whole audio") &&
          speech_result_text(speech_request_result(r))[0] != '\0' &&
          expect(speech_transcribe(r), SPEECH_ERROR_INVALID_ARGUMENT, NULL, "the same request run again after its work");
    speech_request_free(r);
    free(long_prompt);
    // A refused value leaves the one set before it.
    const speech_option kept = takes_timestamps ? SPEECH_OPT_TIMESTAMPS : SPEECH_OPT_PROMPT;
    if (speech_request_new(model, &r) != SPEECH_OK || speech_request_set_audio(r, samples, n, rate) != SPEECH_OK ||
        (takes_timestamps ? speech_request_set_bool(r, kept, 1) : speech_request_set_string(r, kept, "x")) != SPEECH_OK) {
        return fail("a request");
    }
    ok &= expect(speech_request_set_float(r, kept, 0), SPEECH_ERROR_INVALID_ARGUMENT, speech_option_name(kept), "a value of another type");
    ok &= expect(speech_transcribe(r), SPEECH_OK, NULL, "a recognition after a refused value") &&
          (!takes_timestamps || speech_result_segment_count(speech_request_result(r)) > 0);
    speech_request_free(r);
    size_t tokens = 0;
    ok &= expect(speech_model_info_text_tokens(info, "明日", &tokens), SPEECH_ERROR_UNSUPPORTED, NULL, "the tokens of a text for a recognition");
    speech_model_info_free(info);
    ok &= expect(speech_voice_add(model, "voice", "voice.gguf"), SPEECH_ERROR_UNSUPPORTED, NULL, "a voice for a recognition");
    ok &= expect(speech_voice_make(model_path, "reference.wav", "voice.gguf", NULL), SPEECH_ERROR_UNSUPPORTED, "model_path",
                 "a voice file for a recognition model");
    return ok ? 0 : 1;
}

/** The checks of a loaded recognition model, which check_recognition() frees whatever they find. */
static int check_loaded(speech_model * model, const char * model_path, const char ** dumps, int n_dumps) {
    if (check_info_matches(model_path, model) != 0 || check_option_refusals(model) != 0) return 1;
    speech_model_info * info = NULL;
    if (speech_model_get_info(model, &info) != SPEECH_OK) return fail("speech_model_get_info");
    // The texts are compared with the dumps byte for byte, which quantized weights do not reproduce: Qwen3-ASR 0.6B in
    // Q8_0 writes 3 of its 40 dumped requests otherwise, each at a step where the reference's two likeliest tokens lie
    // within the quantization's error (2026-10-06).
    const char * file_type = meta_of(info, "general.file_type");
    if (!file_type || (strcmp(file_type, "0") != 0 && strcmp(file_type, "1") != 0)) {
        fprintf(stderr, "FAIL: %s does not hold F32 or F16 weights, whose texts alone can equal the dumps' byte for byte\n", model_path);
        return 1;
    }
    const int rate = speech_model_info_sample_rate(info);
    const char * use = meta_of(info, "speech.language_use");
    if (speech_model_info_task(info) != SPEECH_TASK_RECOGNITION || speech_model_info_incremental(info) || speech_model_info_voice_count(info) ||
        speech_model_info_voice_files(info) || speech_model_info_max_text_tokens(info) || rate <= 0 || speech_model_info_language_count(info) == 0 ||
        !use || speech_model_info_option_steers(info, SPEECH_OPT_LANGUAGE) != !strcmp(use, "\"steers\"")) {
        fprintf(stderr, "FAIL: the recognition model is not described as speech.h declares\n");
        return 1;
    }
    const int timestamps = speech_model_info_takes(info, SPEECH_OPT_TIMESTAMPS);
    const Asked plain = {timestamps, NULL, NULL, NULL, -1};
    const char * default_decoding = NULL;
    // The model's default of decode_on_cpu, or -1 for a model that does not take it.
    int decode_on_cpu = -1;
    if (speech_model_info_takes(info, SPEECH_OPT_DECODE_ON_CPU) &&
        speech_model_info_option_default_bool(info, SPEECH_OPT_DECODE_ON_CPU, &decode_on_cpu) != SPEECH_OK) {
        return fail("the default of decode_on_cpu");
    }
    if (speech_model_info_takes(info, SPEECH_OPT_DECODING) &&
        speech_model_info_option_default_string(info, SPEECH_OPT_DECODING, &default_decoding) != SPEECH_OK) {
        return fail("the default of decoding");
    }

    float * audio[MAX_DUMPS];
    size_t lengths[MAX_DUMPS];
    char * texts[MAX_DUMPS];
    int shortest = -1, longest = 0, ok = 1;
    for (int d = 0; d < n_dumps; d++) {
        size_t n = 0, text_size = 0;
        char folder[4096], path[4096 + 16];
        request_folder(dumps[d], "auto", folder, sizeof folder);
        snprintf(path, sizeof path, "%s/text.txt", folder);
        audio[d] = read_audio(dumps[d], &n);
        lengths[d] = n;
        texts[d] = read_whole(path, &text_size);
        if (!audio[d] || !texts[d]) {
            fprintf(stderr, "FAIL: %s has no audio.npy of float32 samples or no text.txt\n", dumps[d]);
            return 1;
        }
        const int split = split_dump(dumps[d]);
        char * text = NULL, * languages = NULL, * want_languages = dump_languages(info, folder, split ? "language" : "parsed_language");
        const speech_stop want_stop = dump_stop(folder, split);
        Progress progress;
        memset(&progress, 0, sizeof progress);
        const double start = now_seconds();
        speech_stop stop = SPEECH_STOP_CANCELLED;
        if (!want_languages) {
            fprintf(stderr, "FAIL: %s: qwen-asr parsed a language the model does not name\n", folder);
            return 1;
        }
        if (recognize(model, audio[d], n, rate, plain, &progress, &text, &languages, &stop) != SPEECH_OK) return fail("speech_transcribe");
        const int equal = !strcmp(text, texts[d]) && stop == want_stop, same_languages = !strcmp(languages, want_languages);
        printf("%s: %.2f s of audio in %.3f s, text %s, stop %s, languages %s (%s)\n", dumps[d], (double) n / rate, now_seconds() - start,
               equal ? "equal to the dump's" : "DIFFERS", speech_stop_name(stop), same_languages ? "the dump's" : "DIFFER",
               languages[0] ? languages : "none");
        if (!equal) printf("  got  %s (%s)\n  want %s (%s)\n", text, speech_stop_name(stop), texts[d], speech_stop_name(want_stop));
        if (!same_languages) printf("  got the languages \"%s\" where qwen-asr parsed \"%s\"\n", languages, want_languages);
        ok = ok && equal && same_languages && progress_rises(&progress, 1, "the recognition");
        free(text);
        free(languages);
        free(want_languages);
        // The same audio with the dump's text as its draft, whose tokens the model checks against those it writes.
        if (speech_model_info_takes(info, SPEECH_OPT_DRAFT)) {
            Asked drafted = plain;
            drafted.draft = texts[d];
            if (recognize(model, audio[d], n, rate, drafted, NULL, &text, NULL, NULL) != SPEECH_OK) return fail("speech_transcribe with a draft");
            const int same = !strcmp(text, texts[d]);
            printf("  with its text as the draft: text %s\n", same ? "equal to the dump's" : "DIFFERS");
            if (!same) printf("    got  %s\n    want %s\n", text, texts[d]);
            ok = ok && same;
            free(text);
        }
        // The other requests of a dump of reference/qwen3-asr/dump.py of an input it did not split: its language forced,
        // its prompt, and both.
        static const char * const others[] = {"forced", "auto-prompt", "forced-prompt"};
        for (size_t k = 0; strcmp(folder, dumps[d]) != 0 && !split && k < sizeof others / sizeof others[0]; k++) {
            char request[4096], file[4096 + 16];
            request_folder(dumps[d], others[k], request, sizeof request);
            snprintf(file, sizeof file, "%s/meta.json", request);
            char * meta = read_whole(file, &text_size);
            snprintf(file, sizeof file, "%s/text.txt", request);
            char * want = read_whole(file, &text_size);
            char * name = meta ? json_string_member(meta, "language") : NULL, * prompt = meta ? json_string_member(meta, "prompt") : NULL;
            const char * names = meta_of(info, "qwen3-asr.language_names");
            const int at = name && names ? index_in_array(names, name) : -1;
            Asked asked = {timestamps, at >= 0 ? speech_model_info_language(info, (size_t) at) : NULL, prompt && prompt[0] ? prompt : NULL,
                           NULL, -1};
            char * languages = NULL, * want_languages = dump_languages(info, request, "parsed_language");
            if (!want || !prompt || (name && at < 0) || !want_languages ||
                recognize(model, audio[d], n, rate, asked, NULL, &text, &languages, NULL) != SPEECH_OK) {
                fprintf(stderr, "FAIL: %s: the request cannot be read or recognized\n", request);
                return 1;
            }
            const int same = !strcmp(text, want), same_languages = !strcmp(languages, want_languages);
            printf("  %s (language %s, prompt \"%s\"): text %s, languages %s (%s)\n", others[k], asked.language ? asked.language : "auto", prompt,
                   same ? "equal to the dump's" : "DIFFERS", same_languages ? "the dump's" : "DIFFER", languages[0] ? languages : "none");
            if (!same) printf("    got  %s\n    want %s\n", text, want);
            if (!same_languages) printf("    got the languages \"%s\" where qwen-asr parsed \"%s\"\n", languages, want_languages);
            ok = ok && same && same_languages;
            free(text);
            free(languages);
            free(want_languages);
            free(meta);
            free(want);
            free(name);
            free(prompt);
        }
        // The decodings other than the default.
        for (size_t k = 0; k < speech_model_info_option_choice_count(info, SPEECH_OPT_DECODING); k++) {
            const char * decoding = speech_model_info_option_choice(info, SPEECH_OPT_DECODING, k);
            if (!strcmp(decoding, default_decoding)) continue;
            char request[4096], file[4096 + 16];
            snprintf(request, sizeof request, "%s/%s", dumps[d], decoding);
            snprintf(file, sizeof file, "%s/text.txt", request);
            char * want = read_whole(file, &text_size), * languages = NULL, * want_languages = dump_languages(info, request, "parsed_language");
            const Asked asked = {timestamps, NULL, NULL, decoding, -1};
            if (!want || !want_languages || recognize(model, audio[d], n, rate, asked, NULL, &text, &languages, NULL) != SPEECH_OK) {
                fprintf(stderr, "FAIL: %s: no text.txt, or the audio cannot be recognized with the decoding %s\n", dumps[d], decoding);
                return 1;
            }
            const int same = !strcmp(text, want), same_languages = !strcmp(languages, want_languages);
            printf("  decoding %s: text %s, languages %s (%s)\n", decoding, same ? "equal to the dump's" : "DIFFERS",
                   same_languages ? "the dump's" : "DIFFER", languages[0] ? languages : "none");
            if (!same) printf("    got  %s\n    want %s\n", text, want);
            ok = ok && same && same_languages;
            free(text);
            free(want);
            free(languages);
            free(want_languages);
        }
        // Each decoding where the model does not decode by default, on the CPU or on the model's device: the same text.
        const size_t decodings = speech_model_info_option_choice_count(info, SPEECH_OPT_DECODING);
        for (size_t k = 0; decode_on_cpu >= 0 && k < (decodings ? decodings : 1); k++) {
            const char * decoding = decodings ? speech_model_info_option_choice(info, SPEECH_OPT_DECODING, k) : NULL;
            const int other = decoding && strcmp(decoding, default_decoding) != 0;
            char file[4096 + 16];
            snprintf(file, sizeof file, "%s/%s/text.txt", dumps[d], other ? decoding : ".");
            char * want = read_whole(file, &text_size);
            const Asked asked = {timestamps, NULL, NULL, decoding, !decode_on_cpu};
            if (!want || recognize(model, audio[d], n, rate, asked, NULL, &text, NULL, NULL) != SPEECH_OK) {
                fprintf(stderr, "FAIL: %s: no %s, or the audio cannot be recognized with decode_on_cpu %s\n", dumps[d], file,
                        decode_on_cpu ? "false" : "true");
                return 1;
            }
            const int same = !strcmp(text, want);
            printf("  decoding %s with decode_on_cpu %s: text %s\n", decoding ? decoding : "(the model's)", decode_on_cpu ? "false" : "true",
                   same ? "equal to the dump's" : "DIFFERS");
            if (!same) printf("    got  %s\n    want %s\n", text, want);
            ok = ok && same;
            free(text);
            free(want);
        }
        if (texts[d][0] && (shortest < 0 || n < lengths[shortest])) shortest = d;
        if (n > lengths[longest]) longest = d;
    }
    speech_model_info_free(info);
    if (!ok || shortest < 0) {
        fprintf(stderr, "FAIL: a text differs from the dump's, the progress does not rise, or no dump has a text\n");
        return 1;
    }

    // Each sample three times is audio at three times the rate, which the library resamples to the model's.
    const size_t n = lengths[shortest];
    float * tripled = (float *) malloc(3 * n * sizeof(float));
    for (size_t i = 0; i < 3 * n; i++) tripled[i] = audio[shortest][i / 3];
    char * text = NULL;
    if (recognize(model, tripled, 3 * n, 3 * rate, (Asked){0, NULL, NULL, NULL, -1}, NULL, &text, NULL, NULL) != SPEECH_OK || !text[0]) {
        return fail("audio at three times the rate");
    }
    printf("the same audio at %d Hz: %s\n", 3 * rate, text);
    free(text);
    free(tripled);
    if (check_refusals(model, model_path, audio[shortest], n, rate) != 0) return 1;

    Progress stopping;
    memset(&stopping, 0, sizeof stopping);
    stopping.stop_at = 0.5;
    char * languages = NULL;
    if (!expect(recognize(model, audio[shortest], n, rate, plain, &stopping, &text, &languages, NULL), SPEECH_CANCELLED, NULL,
                "a progress callback that stops at 0.5") ||
        strcmp(text, "") != 0 || strcmp(languages, "") != 0) {
        fprintf(stderr, "FAIL: a stopped recognition has a text or a language\n");
        return 1;
    }
    free(text);
    free(languages);

    // The cancel comes 20 ms into the longest audio, within the encoder; the text comes within milliseconds of the
    // encoder's end, or of the decoding step under way, so a request still running well after the cancel must stop.
    speech_request * r = NULL;
    if (speech_request_new(model, &r) != SPEECH_OK || speech_request_set_audio(r, audio[longest], lengths[longest], rate) != SPEECH_OK) {
        return fail("a request");
    }
    Canceller c = {r, 0.02, 0};
    Thread thread;
    thread_start(&thread, cancel_later, &c);
    const double start = now_seconds();
    const speech_status status = speech_transcribe(r);
    const double took = now_seconds() - start;
    thread_join(&thread);
    const speech_result * result = speech_request_result(r);
    if (status == SPEECH_CANCELLED && result && !strcmp(speech_result_text(result), "") && speech_result_language_count(result) == 0 &&
        speech_result_stop(result) == SPEECH_STOP_CANCELLED) {
        printf("cancelled from another thread after %.3f s: stopped at %.3f s without text\n", c.cancelled_at, took);
    } else if (status == SPEECH_OK && took < c.cancelled_at + 0.01) {
        printf("the request finished in %.3f s, before the cancel at %.3f s took effect; cancellation was not exercised\n", took, c.cancelled_at);
    } else {
        fprintf(stderr, "FAIL: a cancel at %.3f s did not stop a request that ran %.3f s (status %d)\n", c.cancelled_at, took, (int) status);
        return 1;
    }
    speech_request_free(r);

    if (recognize(model, audio[shortest], n, rate, (Asked){0, NULL, NULL, NULL, -1}, NULL, &text, NULL, NULL) != SPEECH_OK ||
        strcmp(text, texts[shortest]) != 0) {
        fprintf(stderr, "FAIL: the request after a cancelled one does not give its text\n");
        return 1;
    }
    printf("the request after the cancelled one gives its text\n");
    free(text);
    for (int d = 0; d < n_dumps; d++) {
        free(audio[d]);
        free(texts[d]);
    }
    return 0;
}

int check_recognition(int argc, char ** argv) {
    const char * model_path = argv[0];
    speech_load_params * params = NULL;
    if (speech_load_params_new(&params) != SPEECH_OK) return fail("speech_load_params_new");
    const char * dumps[MAX_DUMPS];
    int n_dumps = 0;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--device") && i + 1 < argc) {
            if (speech_load_params_set_device(params, argv[++i]) != SPEECH_OK) return fail("speech_load_params_set_device");
        } else if (argv[i][0] != '-' && n_dumps < MAX_DUMPS) {
            dumps[n_dumps++] = argv[i];
        } else {
            fprintf(stderr, "unknown or incomplete option %s\n", argv[i]);
            return 2;
        }
    }
    if (n_dumps == 0) {
        fprintf(stderr, "give at least one dump folder of reference/fastconformer/dump.py or reference/qwen3-asr/dump.py\n");
        return 2;
    }

    speech_model * model = NULL;
    const double load_start = now_seconds();
    if (speech_model_load(model_path, params, &model) != SPEECH_OK) return fail("speech_model_load");
    speech_load_params_free(params);
    printf("loaded without a warm-up in %.2f s\n", now_seconds() - load_start);
    const int status = check_loaded(model, model_path, dumps, n_dumps);
    speech_model_free(model);
    if (status == 0) printf("ok\n");
    return status;
}
