/*
 * Drama 3's voice files through the C API: one made of a recording and its transcript, one put together of it as two
 * speakers without encoding it again, and what cannot be made refused naming its input, before any is written.
 */

#include <stdio.h>
#include <string.h>

#include "speech-api-check.h"

/** Makes a voice file at `path` of `v`, expecting `status` and the error's `option`. */
static int make(const char * model_path, speech_voice_params * v, const char * path, const speech_load_params * params, speech_status status,
                const char * option, const char * what) {
    return expect(speech_voice_make_from(model_path, v, path, params), status, option, what);
}

int check_drama3_voices(const char * model_path, const char * reference, const char * transcript, const char * made, const speech_load_params * params) {
    char cast[4096], refused[4096];
    snprintf(cast, sizeof cast, "%s.cast.gguf", made);
    snprintf(refused, sizeof refused, "%s.refused.gguf", made);
    int ok = 1;
    speech_voice_params * v = NULL;
    if (speech_voice_params_new(&v) != SPEECH_OK) return fail("speech_voice_params_new");
    ok &= expect(speech_voice_params_set_transcript(v, transcript), SPEECH_ERROR_INVALID_ARGUMENT, "transcript", "a transcript before any reference");
    ok &= expect(speech_voice_params_set_speaker(v, 1), SPEECH_ERROR_INVALID_ARGUMENT, "speaker", "a speaker before any reference");
    ok &= expect(speech_voice_params_add_reference(v, reference), SPEECH_OK, NULL, "a reference");
    ok &= make(model_path, v, refused, params, SPEECH_ERROR_INVALID_ARGUMENT, "transcript", "a reference without its transcript");
    ok &= expect(speech_voice_params_set_speaker(v, -1), SPEECH_ERROR_OUT_OF_RANGE, "speaker", "speaker -1");
    ok &= expect(speech_voice_params_set_transcript(v, "<|speaker:1|> 話者のタグのある書き起こし"), SPEECH_OK, NULL, "a tagged transcript") &&
          make(model_path, v, refused, params, SPEECH_ERROR_INVALID_ARGUMENT, "transcript", "a transcript with a speaker tag");
    ok &= expect(speech_voice_params_set_transcript(v, transcript), SPEECH_OK, NULL, "the transcript") &&
          expect(speech_voice_params_set_speaker(v, 16), SPEECH_OK, NULL, "speaker 16") &&
          make(model_path, v, refused, params, SPEECH_ERROR_OUT_OF_RANGE, "speaker", "speaker 16, past the model's 16 speakers");
    ok &= expect(speech_voice_params_set_speaker(v, 0), SPEECH_OK, NULL, "speaker 0") &&
          expect(speech_voice_params_set_loudness(v, -23), SPEECH_OK, NULL, "a loudness") &&
          make(model_path, v, refused, params, SPEECH_ERROR_INVALID_ARGUMENT, "loudness", "a loudness, which Drama 3 does not take");
    speech_voice_params_free(v);
    ok &= expect(speech_voice_make(model_path, reference, refused, params), SPEECH_ERROR_INVALID_ARGUMENT, "transcript",
                 "speech_voice_make(), which gives no transcript");
    FILE * f = fopen(refused, "rb");
    if (f) {
        fclose(f);
        fprintf(stderr, "FAIL: a refused voice file was written\n");
        ok = 0;
    }

    if (speech_voice_params_new(&v) != SPEECH_OK) return fail("speech_voice_params_new");
    const double start = now_seconds();
    ok &= expect(speech_voice_params_add_reference(v, reference), SPEECH_OK, NULL, "a reference") &&
          expect(speech_voice_params_set_transcript(v, transcript), SPEECH_OK, NULL, "its transcript") &&
          make(model_path, v, made, params, SPEECH_OK, NULL, "a voice of a recording and its transcript");
    speech_voice_params_free(v);
    if (ok) printf("made the voice file %s of a recording and its transcript in %.3f s\n", made, now_seconds() - start);

    if (speech_voice_params_new(&v) != SPEECH_OK) return fail("speech_voice_params_new");
    const double copied = now_seconds();
    ok &= expect(speech_voice_params_add_voice(v, made), SPEECH_OK, NULL, "a voice file") &&
          expect(speech_voice_params_set_transcript(v, transcript), SPEECH_ERROR_INVALID_ARGUMENT, "transcript", "a transcript of a voice file") &&
          expect(speech_voice_params_add_voice(v, made), SPEECH_OK, NULL, "the voice file again") &&
          expect(speech_voice_params_set_speaker(v, 1), SPEECH_OK, NULL, "speaker 1") &&
          make(model_path, v, cast, params, SPEECH_OK, NULL, "a voice of speakers 0 and 1 put together of voice files");
    speech_voice_params_free(v);
    if (ok) printf("put %s together of the voice file as speakers 0 and 1 in %.3f s\n", cast, now_seconds() - copied);

    if (speech_voice_params_new(&v) != SPEECH_OK) return fail("speech_voice_params_new");
    ok &= expect(speech_voice_params_add_voice(v, cast), SPEECH_OK, NULL, "the voice of two speakers") &&
          expect(speech_voice_params_set_speaker(v, 2), SPEECH_OK, NULL, "speaker 2") &&
          make(model_path, v, refused, params, SPEECH_ERROR_INVALID_ARGUMENT, "speaker", "a speaker given to a voice file of two speakers");
    speech_voice_params_free(v);
    if (!ok) fprintf(stderr, "FAIL: Drama 3's voice files are not made as speech.h declares\n");
    return ok ? 0 : 1;
}
