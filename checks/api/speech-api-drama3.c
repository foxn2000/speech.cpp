/*
 * Drama 3's voice files through the C API: one made of a recording and its transcript, one put together of it as two
 * speakers without encoding it again, and what cannot be made refused naming its input, before any is written: among
 * them recordings of no audio, of samples that are not finite and of 1,000 s, whose graph would outgrow ggml's context
 * were it encoded before the prompt's room is checked.
 */

#include <math.h>
#include <stdio.h>
#include <string.h>

#include "speech-api-check.h"

/** Makes a voice file at `path` of `v`, expecting `status` and the error's `option`. */
static int make(const char * model_path, speech_voice_params * v, const char * path, const speech_load_params * params, speech_status status,
                const char * option, const char * what) {
    return expect(speech_voice_make_from(model_path, v, path, params), status, option, what);
}

/** Writes `n` copies of `sample`, `bits` wide, as a mono WAVE file at 44.1 kHz of `format` (1 PCM, 3 float). */
static int write_wave(const char * path, unsigned format, unsigned bits, size_t n, const void * sample) {
    FILE * f = open_utf8(path, "wb");
    if (!f) return 0;
    const unsigned width = bits / 8, rate = 44100, data = (unsigned) (n * width);
    const unsigned fields[][2] = {{36 + data, 4}, {16, 4}, {format, 2}, {1, 2}, {rate, 4}, {rate * width, 4}, {width, 2}, {bits, 2}};
    unsigned char header[44], *at = header;
    memcpy(at, "RIFF", 4);
    at += 4;
    for (size_t i = 0; i < sizeof fields / sizeof fields[0]; i++) {
        for (unsigned b = 0; b < fields[i][1]; b++) *at++ = (unsigned char) (fields[i][0] >> (8 * b));
        if (i == 0) {
            memcpy(at, "WAVEfmt ", 8);
            at += 8;
        }
    }
    memcpy(at, "data", 4);
    for (unsigned b = 0; b < 4; b++) header[40 + b] = (unsigned char) (data >> (8 * b));
    fwrite(header, 1, sizeof header, f);
    unsigned char block[4096 * 4];
    const size_t per_block = sizeof block / width;
    for (size_t i = 0; i < per_block; i++) memcpy(block + i * width, sample, width);
    for (size_t done = 0; done < n; done += per_block) fwrite(block, width, n - done < per_block ? n - done : per_block, f);
    return fclose(f) == 0;
}

/** Expects a voice of one recording, written as write_wave() writes it, with `transcript`, to be refused with `status` and `option`. */
static int refuse_recording(const char * model_path, const char * wave, unsigned format, unsigned bits, size_t n, const void * sample,
                            const char * transcript, const char * refused, const speech_load_params * params, speech_status status,
                            const char * option, const char * what) {
    if (!write_wave(wave, format, bits, n, sample)) return fail("writing a WAVE file");
    speech_voice_params * v = NULL;
    if (speech_voice_params_new(&v) != SPEECH_OK) return fail("speech_voice_params_new");
    const int ok = expect(speech_voice_params_add_reference(v, wave), SPEECH_OK, NULL, "a reference") &&
                   expect(speech_voice_params_set_transcript(v, transcript), SPEECH_OK, NULL, "its transcript") &&
                   make(model_path, v, refused, params, status, option, what);
    speech_voice_params_free(v);
    remove(wave);
    return ok;
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
    ok &= expect(speech_voice_params_set_transcript(v, "\xff\xfe"), SPEECH_OK, NULL, "a transcript that is not UTF-8") &&
          make(model_path, v, refused, params, SPEECH_ERROR_INVALID_ARGUMENT, "transcript", "a transcript that is not UTF-8");
    ok &= expect(speech_voice_params_set_transcript(v, transcript), SPEECH_OK, NULL, "the transcript again") &&
          expect(speech_voice_params_set_speaker(v, 0), SPEECH_OK, NULL, "speaker 0") &&
          expect(speech_voice_params_set_loudness(v, -23), SPEECH_OK, NULL, "a loudness") &&
          make(model_path, v, refused, params, SPEECH_ERROR_INVALID_ARGUMENT, "loudness", "a loudness, which Drama 3 does not take");
    speech_voice_params_free(v);
    ok &= expect(speech_voice_make(model_path, reference, refused, params), SPEECH_ERROR_INVALID_ARGUMENT, "transcript",
                 "speech_voice_make(), which gives no transcript");
    char wave[4096];
    snprintf(wave, sizeof wave, "%s.recording.wav", made);
    const short silence = 0;
    const float not_finite = NAN;
    const double refusing = now_seconds();
    ok &= refuse_recording(model_path, wave, 1, 16, (size_t) 1000 * 44100, &silence, transcript, refused, params, SPEECH_ERROR_OUT_OF_RANGE,
                           "references", "a recording of 1,000 s, past what a prompt holds");
    if (ok) printf("refused the recording of 1,000 s in %.3f s, before encoding it\n", now_seconds() - refusing);
    ok &= refuse_recording(model_path, wave, 3, 32, 2048, &not_finite, transcript, refused, params, SPEECH_ERROR_MODEL_FILE, "references",
                           "a recording of float samples that are not finite");
    ok &= refuse_recording(model_path, wave, 1, 16, 0, &silence, transcript, refused, params, SPEECH_ERROR_INVALID_ARGUMENT, "references",
                           "a recording without audio");
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
