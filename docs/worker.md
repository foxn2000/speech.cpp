# Worker protocol

This page describes `speech worker`, a process that another program starts to speak texts or recognize speech, as
[ASIST](https://github.com/nyosegawa/asist) does, and the protocol it speaks.

```
speech worker MODEL [--add-voice NAME=FILE]... [--device NAME] [--threads N] [--no-warmup] [request options]
```

```sh
speech worker qwen3-tts-0.6b
speech worker irodori-tts-mf --add-voice me=me.voice.gguf --add-voice calm=calm-reference.wav
speech worker irodori-tts-mf --steps 8 --add-voice me=me.voice.gguf
speech worker reazonspeech-v2
```

A request option given as a flag ([cli.md](cli.md#speech-serve-and-speech-worker)) is the default of every request that
does not set it, in place of the model's own; a request's own member wins. `ready` and `info` list these defaults.

The worker serves one model. It speaks [JSON Lines](https://jsonlines.org): one JSON object per line on stdin and on
stdout, in UTF-8.

- stdout carries the protocol and nothing else; every log goes to stderr. A caller treats a line on stdout that is not a
  JSON object as a defect of the worker and fails, rather than skipping it.
- Every line in either direction is one JSON object with a string member `type`. A member set to `null` counts as left
  out. A member that the message's type does not have is refused, a name that is not an option included.
- Option members take the option's type ([c-api.md](c-api.md#options)): a JSON string; an integer without a fraction or
  an exponent; any JSON number; `true` or `false`.
- Requests run one at a time, in the order they become complete. `info` and `count_tokens` only read the model's
  information and are answered as they arrive, also while a request runs.

## Start

```
out {"type":"ready","protocol":3,"version":"0.8.7","defaults":{"steps":8},"model":{...model information...}}
out {"type":"fatal","error":{"code":"model_file","option":null,"message":"..."}}
```

`ready` comes once the model is loaded, warmed up unless `--no-warmup` so that a GPU has compiled its kernels before the
first request, and has the voices of `--add-voice`. `version` is the release, `defaults` the request options given as
flags with their values (`{}` without any), and `model` is the model information as JSON
([c-api.md](c-api.md#model-information-as-json)), with the device, the threads and the voices added. The model
information keeps the model's own defaults.

`fatal` comes instead when the worker cannot start. The worker then exits with 1, or with 2 for a command line it cannot
run. A detection model (Silero VAD) is `unsupported`: the worker serves synthesis and recognition models. A request
option flag the model does not take is `unsupported`, and a value out of its range `out_of_range`, each naming the
option.

`protocol` rises when a caller must change to keep working: a message or member removed, renamed or given another
meaning. A member or message added, an option added to the vocabulary, or a member added to the model information does
not raise it.

## Requests

| `type` | Members | Task | Answer |
|---|---|---|---|
| `synthesize` | `id`, `text`, option members | synthesis | `progress` and `chunk`s, then one terminal message |
| `chunk` | `id`, `seq`, `pcm` | recognition | none of its own; it adds audio to request `id` |
| `transcribe` | `id`, `sample_rate`, option members | recognition | `progress`, then one terminal message |
| `add_voice` | `id`, `name`, `path` | synthesis models that take voice files | one terminal message |
| `info` | `id` | any | one terminal message, at once |
| `count_tokens` | `id`, `text` | synthesis | one terminal message, at once |
| `cancel` | `id` | any | none of its own |

- `id` is a non-empty string, unique among the requests that have not had their terminal message.
- A recognition request is its `chunk` lines, `seq` 0, 1, 2 … in order, then its `transcribe` line, which makes it
  complete. The chunks of several requests may interleave. `sample_rate` is the rate of the chunks' audio, any rate; the
  library resamples it.
- `pcm`, in both directions, is base64 of 16-bit little-endian mono samples. Audio from the worker is at the model's
  `sample_rate`, each sample `round(clamp(x, −1, 1) × 32767)`. Audio to the worker is read as `x / 32768`.
- `path` of `add_voice` is a voice file, or a WAVE file for a model whose `voice_recordings` is true (Irodori-TTS);
  Drama 3 takes a voice file alone ([models.md](models.md#voices)).
- `info` gives the voices added so far: one sent before an `add_voice` has had its `end` may not list that voice yet.
- `count_tokens` counts a text's tokens as a synthesis counts them against `max_text_tokens`, so that a caller can split a
  long text before it sends it, also while a synthesis runs.
- `cancel` of a request that is collecting chunks or waiting ends it with `cancelled` at once. A running request stops at
  the next point where its work can stop and ends with `cancelled`; `add_voice` cannot be stopped once it runs. A cancel
  of an id that no request in flight has is ignored.
- Once a recognition request has had its `error` or `cancelled` while it collected chunks, its later `chunk` lines are
  dropped up to and including its `transcribe` line, which frees its id, or until a chunk 0 under its id starts a new
  request. A caller that cancels need not send the `transcribe`.

When stdin closes, the worker answers the requests it has, answers each request still collecting chunks with an `error`,
since no `transcribe` can follow, and exits with 0.

## Answers

```
out {"type":"chunk","id":"a","seq":0,"pcm":"..."}
out {"type":"progress","id":"r","done":0.42}
out {"type":"end","id":"a","seed":1234,"samples":96000,"stop":"complete"}
out {"type":"end","id":"r","text":"...","stop":"complete","segments":[{"start":0.0,"end":2.48,"text":"..."}],"tokens":[{"start":0.0,"end":0.16,"text":"..."}]}
out {"type":"end","id":"q","text":"...","stop":"complete","languages":["ja"]}
out {"type":"end","id":"v"}
out {"type":"end","id":"i","defaults":{"steps":8},"model":{...model information...}}
out {"type":"end","id":"c","tokens":14}
out {"type":"error","id":"a","error":{"code":"out_of_range","option":"speed","message":"..."}}
out {"type":"cancelled","id":"a"}
```

- Every request gets exactly one terminal message, `end`, `error` or `cancelled`, and nothing for its id after it.
- `end` of a synthesis has `seed` (the request's, or the one the library drew), `samples` (the number sent in its chunks)
  and `stop` (`complete`, `max_seconds` or `model_limit`).
- `end` of a recognition has `text` and `stop` (`complete`, or `model_limit` when it reached the most tokens the model
  writes, with the text written up to there). It has `languages`, the tags of the languages the model heard in the
  order of the audio, where there are any (Qwen3-ASR), and `segments` and `tokens` when the request set `timestamps`.
- `end` of `info` has `model`, `end` of `count_tokens` has `tokens`, and `end` of `add_voice` has nothing more.
- `error` has `code` (an error category, [c-api.md](c-api.md#errors)), `option` (the input it concerns: an option's
  name, `text`, `audio`, `id`, `type`, `seq`, `pcm`, `name`, `path`, `sample_rate` or another member's name, or null)
  and `message`.
- `progress` comes for a running request that is passing no chunks, when the library reports progress and at least 1 s
  has passed since the request's last message. `done` is a fraction from 0 to 1. A caller that takes a silent worker for
  a hung one allows for the longest single step: the encoder of a long recording runs as one (3.7 s for 311 s of audio
  with reazonspeech-nemo-v2 in F16 on an Apple M5).
- A line that names no request it could belong to (not a JSON object, no string `id`, a `cancel` with a member it does
  not have, or a line that starts a request under an `id` already in flight) gets an `error` without `id`.
- A request of the other task is `unsupported` with the option `type`: a `synthesize` to a recognition model, and a
  `chunk` or `transcribe` to a synthesis model, whose later lines are dropped. A member the type does not have is
  `invalid_argument` with that member as `option`. Option values are checked by the library's setters when the request
  runs, so their errors are the C API's.

## What each family does

- **Qwen3-TTS** passes audio as it makes it, from the first 0.08 s, at 24 kHz. A cancel takes effect at the next chunk.
- **Irodori-TTS** makes a request's speech at once and passes it as the codec decodes it, at 48 kHz, so a request should
  be one sentence: the first audio waits for the whole text. A text it predicts to last over 30 s is cut after its
  sentences, or after its commas, or at its spaces, and spoken in pieces as `speech tts` speaks it, a sentence's end
  holding 0.9 s of silence; a text with no place to cut is refused, naming it. A cancel takes effect at the next sampler
  step or piece of audio.
- **FastConformer** recognizes a request's audio at once, and loses whole sentences of a stretch that holds several, so
  a request should be one utterance. A cancel takes effect before or after the encoder, or during the decoding.
- **Qwen3-ASR** recognizes up to 1200 s at once, and longer audio in parts. A cancel takes effect within a fraction of a
  second.

## A session

Irodori-TTS speaking a sentence, and a recognizer taking a request in two chunks:

```
in  {"type":"synthesize","id":"1","text":"明日の東京は晴れです。","voice":"me","seed":42}
out {"type":"chunk","id":"1","seq":0,"pcm":"..."}
out {"type":"chunk","id":"1","seq":1,"pcm":"..."}
out {"type":"end","id":"1","seed":42,"samples":134400,"stop":"complete"}

in  {"type":"chunk","id":"r","seq":0,"pcm":"..."}
in  {"type":"chunk","id":"r","seq":1,"pcm":"..."}
in  {"type":"transcribe","id":"r","sample_rate":16000}
out {"type":"end","id":"r","text":"群島や湖では必ずしもヨットは必要ありません。","stop":"complete"}
```

`checks/smoke/worker_client.py` is a client in Python that checks every line and one terminal message per request.
