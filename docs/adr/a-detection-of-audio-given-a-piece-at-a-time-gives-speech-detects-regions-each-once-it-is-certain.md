# A detection of audio given a piece at a time gives speech_detect()'s regions, each once it is certain

## Context

speech.cpp 0.8.0 transcribes speech as it is said: from a microphone with `speech asr --live`, from programs through the
worker, and through `speech serve` with OpenAI's Realtime transcription over a WebSocket. Each needs to know, while the
audio still arrives, where an utterance starts and where it has ended, to cut it and give it to a recognizer.
`speech_detect()` takes a request's whole audio
([the record of detection](detecting-speech-is-a-task-of-its-own-run-by-speech-detect-with-silero-vads-options-and-f32-files.md)).

Silero VAD's rule, `get_speech_timestamps()`, walks the chunks' probabilities once, left to right, with a small state
(whether a region is under way, its start, where a silence began, the silences a limit may cut at). Only its padding
looks ahead: a region's padded end is half the silence to the next region when that starts within twice
`speech_pad_ms`, and `speech_pad_ms` otherwise. The package's `VADIterator`, which streams, has a rule of its own: it
has no `min_speech_duration_ms` and no `max_speech_duration_s`, pads every region by `speech_pad_ms` whatever follows,
and ends a region below the threshold less 0.15 without the floor of 0.01, so its regions are not
`get_speech_timestamps()`'s.

The audio comes at the caller's rate: OpenAI's Realtime API sends 24 kHz PCM and a browser records at 48 kHz, where the
model takes 16 kHz. Silero VAD's graphs multiplied the frames of all their chunks in one product, and Metal's kernel for
more than eight columns rounds its inputs to half precision, so on Metal a chunk's probability moved by up to 2.2e-3
with the number of chunks its graph held (Apple M5, 2026-10-08).

## Decision

- **A detection is an object of its own**, `speech_detection`, started from a request's options and the rate of the
  audio to come, given samples in any amounts, and ended. It gives the regions found so far, which never change once
  given, and whether a region has begun that is not given yet, with the start it will have and whether it is kept.
- **Its regions are `speech_detect()`'s.** Once the audio has ended they are the regions `speech_detect()` gives for
  all of it, sample for sample, at any rate and however it was cut into pieces. Each stage gives the same numbers in
  pieces as whole: the resampler computes an output sample through the same sum once its input has arrived; a chunk is
  computed once its samples have arrived, with the same context and cell state, and every product of the network
  multiplies one column, which each backend computes alike in a graph of any size; and the rule walks the chunks as the
  official does.
- **A region is given once it is certain**: when no region still to come can start within twice `speech_pad_ms` of its
  end, or the region that did is certain to be kept, longer than `min_speech_duration_ms`. With `min_silence_duration_ms`
  at least twice `speech_pad_ms`, as with the defaults, that is the chunk where silence ends the region. Otherwise it is
  at most 2 · `speech_pad_ms` + `min_speech_duration_ms` + `min_silence_duration_ms` and two chunks after the region's
  end, and a region that `max_speech_duration_s` cuts at an earlier silence is given at the cut at the earliest.
- **What has begun and is not given yet is said oldest first**: a region whose end waits on what follows before the
  region under way, so that a caller that announces a start and an end for each region announces them in order.
- **The region begun is said to be kept once it can no longer be dropped**, as shorter than `min_speech_duration_ms`:
  a region whose end waits is kept, and the region under way once no end it can still take, where its probability fell
  below the threshold less 0.15 without rising above the threshold since, or else the next chunk or the end of the audio
  heard, leaves it that short. With the defaults that is 250 ms after its speech began, where its region is given only
  after the silence that ends it, so that a caller can say someone speaks long before it can give the utterance, and
  never for a noise it then drops. `speech_detection_speaking()` gives it beside the start, in one call that reads one
  region: whether a region has begun, its start, and whether it is kept.
- **The library resamples the pieces** with the filter of the whole audio
  ([the record of resampling](the-library-resamples-input-audio-with-torchaudios-kaiser-best-filter-and-drama3-references-as-fish-speech-does.md)), which holds
  back the half of the filter after the last sample until more arrives: 4.2 to 4.3 ms from 24, 32, 44.1 or 48 kHz to
  16 kHz, 8.6 ms from 8 kHz.
- **A push holds the model as a request does**, so that several detections and the model's requests share one model.

The alternatives were turned down:

- A request that takes its audio in pieces (an append to `speech_request`) and runs once at the end. A request runs once
  and holds one result; a detection gives regions while it runs, keeps the state of its resampler, its chunks and its
  rule from piece to piece, and outlives any one call, and every call of a request would have to ask whether it streams.
- `VADIterator`'s rule. Its regions are neither `speech_detect()`'s nor the official `get_speech_timestamps()`'s.
- Giving a region as soon as silence ends it, padded as if no region followed. Where speech resumes within twice
  `speech_pad_ms`, its end would differ from `speech_detect()`'s.
- Audio at the model's rate alone. Every Realtime client and browser would resample with a filter of its own, and the
  regions would not be those `speech_detect()` gives for the audio the caller has.
- A shorter filter for pieces, with less delay. The regions would not be `speech_detect()`'s, for 4.3 ms, an eighth of
  a chunk.
- Products of many columns, with the regions allowed to differ near a threshold. A region would move by a chunk or
  more wherever a probability lies within Metal's error of a threshold, and the contract would hold on the CPU alone.
- The region under way said without whether it is kept, which a caller would guess from the time it has lasted. Its
  probability may fall below the lower threshold and stay between the two thresholds for any time, keeping the region
  open without lengthening its speech, and it is dropped once a chunk falls below again; a guess from elapsed time
  announces it, and waiting out `min_speech_duration_ms` and `min_silence_duration_ms` instead cost 0.6 s.
- Whether it is kept as a function of its own beside `speech_detection_speaking()`, or as a third return value. A
  second call would have to name the same region again, and a value of 2 reads less plainly than an output named for
  what it says.

## Consequences

The assembly that cuts utterances for `speech asr -`, `speech asr --live` and `server_vad` of `speech serve` uses the
detection. On Metal, `speech_detect()` computes in single precision too: its probabilities lie within 3.7e-6 of the
official's, where they lay within 6.2e-3, and a minute takes 0.057 s, where it took 0.045 s. A small piece's time is
mostly the device's to start a graph: a minute in pieces of 20 ms takes 0.2 s on the CPU and 0.7 s on Metal, so a live
caller loads the model on the CPU.
