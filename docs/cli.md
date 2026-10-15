# Command line

This page lists the subcommands of `speech` and their options. `speech <subcommand> --help` prints the same in short.

| Subcommand | What it does |
|---|---|
| [`speech tts`](#speech-tts) | speaks text into a WAVE file or to stdout |
| [`speech asr`](#speech-asr) | writes the text of WAVE files, or of the microphone as someone speaks |
| [`speech vad`](#speech-vad) | writes where someone speaks in WAVE files |
| [`speech voice`](#speech-voice) | makes a voice file for Irodori-TTS or Drama 3 from reference recordings |
| [`speech info`](#speech-info) | prints what a model file says of its model, without loading it |
| [`speech devices`](#speech-devices) | lists the devices a model can run on |
| [`speech models`](#speech-models) | lists the models a name fetches, what is fetched, and which model to start with |
| [`speech pull`](#speech-pull) | fetches models ahead of their use |
| [`speech rm`](#speech-rm) | removes fetched models, or the old files no model names |
| [`speech quantize`](#speech-quantize) | writes a model file of F32 weights in another weight type |
| [`speech update`](#speech-update) | installs the latest release, or says whether one is out |
| [`speech serve`](server.md) | serves a model of each task over HTTP with OpenAI's audio API, and a page to try models on |
| [`speech worker`](worker.md) | serves a model over JSON Lines on stdin and stdout, for programs such as ASIST |

## Examples

```sh
# One sentence to a file, with a Qwen3-TTS speaker
speech tts qwen3-tts-0.6b --voice ono_anna --seed 42 -o out.wav "明日の東京は晴れです。"

# A voice file from a reference recording, then a text file, one sentence per line, into one WAVE file in that voice
speech voice irodori-tts-mf me.wav me.voice.gguf
speech tts irodori-tts-mf --add-voice me=me.voice.gguf --voice me -o story.wav < story.txt

# Straight into a player, which starts as the first audio arrives
echo "こんにちは。" | speech tts irodori-tts-mf --add-voice me=me.voice.gguf --voice me -o - | ffplay -nodisp -autoexit -

# The text of recordings, and their segments with their times as JSON Lines
speech asr reazonspeech-v2 meeting.wav
speech asr parakeet-tdt_ctc-0.6b-ja --timestamps --format json one.wav two.wav > texts.jsonl

# The text of a recording in Japanese, told the names it holds
speech asr qwen3-asr-1.7b --language ja --prompt "Claude Code、渋谷" meeting.wav

# The text of a long recording, recognized region by region where someone speaks
speech asr reazonspeech-v2 --vad silero-vad meeting.wav

# The microphone, each utterance's text as it is said, until Ctrl-C
speech asr reazonspeech-v2 --vad silero-vad --live

# Where someone speaks in a recording, in regions of 10 s at most, and each region as a WAVE file of its own
speech vad silero-vad --max-speech-duration-s 10 meeting.wav
speech vad silero-vad --split regions/ meeting.wav       # regions/meeting-01.wav, regions/meeting-02.wav, ...

# What a model file holds, without loading it
speech info irodori-tts-mf
```

## Rules for every subcommand

- **MODEL** is a model file, whose path ends in `.gguf`, or `NAME[:TYPE]` of a model in `speech models`, which is
  fetched the first time it is used ([models.md](models.md)).
- **`--device NAME`** is `auto` (the default: the first GPU, or the CPU on a machine without one), `gpu`, `cpu`, or a
  name that `speech devices` lists. **`--threads N`** is the CPU's threads; by default the machine's performance cores.
- **Request options.** `speech tts`, `speech asr` and `speech vad` take every option of the C API's vocabulary as a flag
  of its name in kebab-case: `duration_scale` is `--duration-scale` ([c-api.md](c-api.md#options)). The model refuses
  an option it does not take, as the C API does: Qwen3-TTS answers `--speed 1.5` with `speech: unsupported (speed): ...`.
  `speech info MODEL` lists the options a model takes, with their defaults and ranges. `speech serve` and
  `speech worker` take the same flags as the defaults of their requests.
- A value follows its flag or an `=`: `--seed 7` or `--seed=7`. A number is read whole: `--steps 4x` is a usage error.
- A boolean flag alone means true, and takes `true` or `false` only after an `=`: `--timestamps`, `--do-sample=false`.
- An argument that begins with `-`, such as a text, follows `--`.
- stdout carries the output alone, and every log goes to stderr.
- On Windows the command line is read as UTF-8, and stdin and stdout are binary.
- `speech --version` prints the release and the C API's version: `speech.cpp 0.8.7, C API 3.3`.
- On a terminal, a subcommand says on stderr when a later release is out, once it ends (`speech serve` as it starts):
  `speech: speech.cpp 0.8.8 is out (this is 0.8.7); run speech update to install it`. It asks github.com at most once a
  day, and only when stderr is a terminal, so a script or a program that starts `speech` never makes the request; the
  worker never asks. `SPEECH_NO_UPDATE_CHECK=1` turns it off.

| Exit | Meaning |
|---|---|
| 0 | done |
| 1 | a failure, printed on stderr as `speech: <code> (<option>): <message>`. The code is the library's error category ([c-api.md](c-api.md#errors)) and the option is the input at fault; the parentheses are left out when there is none |
| 2 | a command line that cannot run, printed with a pointer to `speech <subcommand> --help` |
| 3 | a request stopped at the most the model makes (`model_limit`): for `speech tts` the longest speech, and the WAVE file is complete; for `speech asr` the most tokens of a text, and every file's text is written |

## speech tts

```
speech tts MODEL -o FILE|- [options] [TEXT]
  --add-voice NAME=FILE    add a voice from a voice file or a WAVE file before speaking; repeatable
  --device NAME --threads N
  -v                       also report the model, and each request's seed and stop reason
  --voice NAME --language TAG --seed N ... every request option
```

- Speaks TEXT, or without it each non-empty line of stdin as a text of its own, into one 16-bit mono WAVE at the model's
  rate. `-o -` writes it to stdout.
- Irodori-TTS, which speaks at most 30 s at a time, speaks a text of several sentences one sentence at a time, with a
  pause of 0.9 s between them; a sentence too long for it is cut at its commas, or at its spaces. A run of text with
  neither that is too long fails. Qwen3-TTS speaks a text whole, up to 24565 tokens.
- `--voice` is required. Qwen3-TTS has named speakers; Irodori-TTS and Drama 3 take voices added with `--add-voice`
  ([models.md](models.md#voices)).
- `--seed` applies to every text and every sentence. Without it, each text draws its own seed, which `-v` reports; the
  same seed gives the same audio on the same device.
- A request that stops at `--max-seconds`, or at the longest speech the model makes (655 s for Qwen3-TTS), is reported on
  stderr whatever `-v` says.
- The WAVE is written as the audio is made, so a player that reads stdout starts on the first audio. A run that fails
  removes the WAVE file it was writing.
- Text on stdin is UTF-8.
- stderr reports the load time and, for each text, its seconds of audio, how many requests it took when more than one,
  its time to first audio, the time to its end, and its RTFx, the seconds of audio over the seconds they took.

## speech asr

```
speech asr MODEL [options] AUDIO.wav...|-|--live
  --format text|json       text (the default) or one JSON object per file, or per event, and line
  --vad MODEL              transcribe by the regions where this detection model finds speech
  --live                   transcribe the default microphone until Ctrl-C, with --vad
  --rate HZ                the rate of the PCM that - reads from stdin, 16000 unless given
  --device NAME --threads N
  -v                       also report the release, the sample rate and the model's languages, and each utterance
  --language TAG --timestamps --prompt TEXT --decoding beam|greedy ... every request option
```

- Recognizes each WAVE file in the order given: 16-, 24- or 32-bit PCM or 32-bit float at any rate, with its channels
  averaged. The library resamples it to the model's rate. Convert other formats first: `ffmpeg -i in.mp3 out.wav`.
- `text` writes one line per file. With `--timestamps` it writes one line per segment instead,
  `FILE<TAB>START<TAB>END<TAB>TEXT`, with the times in seconds and three decimals.
- `json` writes `{"file":…,"text":…,"stop":…}` for each file, with `"languages"` where the model names the languages it
  heard (Qwen3-ASR), and `"segments"` and `"tokens"` with `--timestamps`, in the form of the worker's `end`
  ([worker.md](worker.md#answers)).
- stderr reports the load time and, for each file, its seconds of audio, the time to its text and the RTFx. It also
  reports a recognition that stopped at the most tokens the model writes (4096 for Qwen3-ASR); the command then exits
  with 3 once every file's text is written.
- A failure names the file. The lines of the files before it are already on stdout.

### Transcribing by regions

`--vad MODEL` names a detection model ([models/silero-vad.md](models/silero-vad.md)), which runs on the CPU with one
thread whatever `--device` and `--threads` say. It finds where someone speaks in each file, each region is recognized
alone, and the texts are joined in order: with a space between two regions, unless either side is Japanese or Chinese
text or already has one. Segments and tokens have the times of the whole file, and
`languages` lists the languages heard in order. A file in which no one speaks gives an empty text.

A recognizer given a long stretch with several sentences can drop whole sentences, and can write words for audio in
which no one speaks; by regions it hears one or a few sentences at a time.

The detection takes the flags of `speech vad` with these defaults, which are OpenAI's for its `server_vad` and a longest
region:

| Flag | Default | Meaning |
|---|---|---|
| `--threshold X` | 0.5 | the speech probability from which audio counts as speech |
| `--speech-pad-ms N` | 300 | the audio kept before and after each region |
| `--min-silence-duration-ms N` | 500 | the silence that ends a region |
| `--max-speech-duration-s S` | 10 | the longest region; a longer one is cut at its longest silence |
| `--min-speech-duration-ms N` | 250 | the shortest region kept |

`speech vad --split` with the same flags writes the regions that `speech asr --vad` recognizes.

### Transcribing as someone speaks

With `--vad`, `--live` transcribes the default microphone until Ctrl-C, and `-` reads 16-bit little-endian mono PCM from
stdin at `--rate`, 8000 Hz or more, until it ends, for programs that have audio as it is recorded. Each region is written as it ends, so
the same audio gives the same regions and texts as a file.

```sh
speech asr reazonspeech-v2 --vad silero-vad --live
speech asr qwen3-asr-0.6b --vad silero-vad --live --format json > events.jsonl
```

- `text` writes each utterance's text as a line. On a terminal, the line shows the text as it is said, the beginning that
  two readings in a row agree on, and the final text replaces it once the utterance ends. With Qwen3-ASR, each reading
  and the final text are given the reading before as their `draft`.
- `json` writes OpenAI's Realtime server events, one JSON object per line, as `/v1/realtime` sends them
  ([server.md](server.md#realtime-transcription)): `input_audio_buffer.speech_started` with `audio_start_ms` and the
  utterance's `item_id`, `input_audio_buffer.speech_stopped` with `audio_end_ms`, `input_audio_buffer.committed`,
  `conversation.item.input_audio_transcription.delta`, text that adds to the end of what came before, and
  `conversation.item.input_audio_transcription.completed`, whose `transcript` is the whole text and replaces the deltas.
- On an Apple M5, speech started comes 0.26 s after the speech begins, the first text 0.5 to 1.5 s after it, and the
  final text 0.7 to 1.3 s after the speech ends.
- stderr reports the microphone and its rate, and at the end the utterances and the time the detection took; `-v` also
  reports each utterance as it ends. A recognition that fails ends the run with exit 1, after the `failed` event in JSON.
- `--timestamps` is refused: the times of an utterance are in its events.
- Utterances wait for their recognition while the recognizer is behind, up to 13.1 million samples (13.6 min at 16 kHz).
  Past that, `-` reads stdin no faster than the recognition goes, and `--live` ends with `out_of_memory`. Options that
  keep that much audio before an utterance is committed, such as a `--speech-pad-ms` of minutes, end either with
  `out_of_memory`.

On macOS, a program reaches the microphone through the app it runs in, such as Terminal or iTerm2. The first time,
macOS asks whether that app may use the microphone; afterwards it is in System Settings, Privacy & Security,
Microphone. Without it the program hears only silence, and `speech` says so on stderr after 3 s.

## speech vad

```
speech vad MODEL [options] AUDIO.wav...
  --format text|json       text (the default) or one JSON object per file and line
  --split DIR              also write each region as a WAVE file in DIR
  --device NAME --threads N
  -v                       also report the release and the sample rate
  --threshold X --min-speech-duration-ms N --min-silence-duration-ms N --speech-pad-ms N
  --max-speech-duration-s S ... every request option
```

- Finds where someone speaks in each WAVE file in the order given, with a detection model
  ([models/silero-vad.md](models/silero-vad.md)). It reads the files as `speech asr` does.
- `text` writes one line per region, `FILE<TAB>START<TAB>END`, with the times in seconds and three decimals. A file
  without speech writes no line.
- `json` writes `{"file":…,"regions":[{"start":…,"end":…}]}` for each file.
- `--split DIR` also writes each region as a mono 16-bit WAVE file at the file's rate in DIR, which it makes if it is not
  there: `meeting-1.wav`, `meeting-2.wav` and so on, numbered with as many digits as the last number so that they sort
  in order. Files of one name in different folders are refused, since their regions would take the same names.
- stderr reports the load time and, for each file, its seconds of audio, the time to its regions, the RTFx and the
  number of regions.

## speech voice

```
speech voice MODEL ((REFERENCE.wav | VOICE.gguf)... | EMBEDDING.speaker.safetensors) OUT.gguf [options]
  --lufs LUFS              Irodori-TTS: bring each recording to this loudness instead of the model's (-16)
  --keep-loudness          Irodori-TTS: keep each recording's loudness, scaling down a peak above 1
  --transcript TEXT        Drama 3: the words of a recording, one for each recording in their order
  --speaker N              Drama 3: the speaker of a recording or voice file, one for each in their order
  --device NAME            cpu (the default here), auto, gpu or a name speech devices lists
  --threads N -v
```

- Makes a voice file for a model that takes voice files (Irodori-TTS, Drama 3) from one or more recordings at any rate.
  Each is encoded on its own and the results are joined in order. Only the codec is read from MODEL.
- Drama 3 needs the words each recording speaks, one `--transcript` per recording. A recording is speaker 0 unless
  `--speaker` gives another, and a Drama 3 voice file given as an input keeps its speakers unless `--speaker` gives one.
  So a voice for a scene is put together from voices made before, without encoding them again:

  ```sh
  speech voice Drama-35B-A3B-v3-Q4_K.gguf woman.wav --transcript "今日はいい天気ですね。" woman.voice.gguf
  speech voice Drama-35B-A3B-v3-Q4_K.gguf man.wav --transcript "明日は雨の予報です。" man.voice.gguf
  speech voice Drama-35B-A3B-v3-Q4_K.gguf woman.voice.gguf man.voice.gguf --speaker 0 --speaker 1 cast.voice.gguf
  speech tts Drama-35B-A3B-v3-Q4_K.gguf --add-voice cast=cast.voice.gguf --voice cast -o scene.wav \
    "<|speaker:0|> 晴れたね。<|speaker:1|> 明日は雨だって。"
  ```

- A `.speaker.safetensors` file, a speaker-inversion embedding as Irodori-TTS's runtime saves one, makes a voice for
  MODEL alone. It is the whole voice: a second embedding or a recording beside it is refused.
- The device is `cpu` by default, which encodes a voice closest to the official encoder.
- [models/irodori-tts.md](models/irodori-tts.md#voices) and [models/drama3.md](models/drama3.md#voices) say what a
  voice file holds.

## speech info

```
speech info MODEL [--json] [--meta]
```

- Prints the model's information without loading it: its name, organization, model line, size label, finetune, version,
  license, source and weight type; its architecture and layout; its task and rate; its languages and voices; each option
  with its type, default, range or choices and whether it steers; the longest text; and the sizes.
- `--json` prints the model information as JSON instead ([c-api.md](c-api.md#model-information-as-json)), the object the
  worker's `ready` and the server's `/v1/models` carry.
- `--meta` adds every metadata entry of the GGUF file: as `key = value` in text, with an array of more than eight items
  shortened and its length given, and whole as a `meta` object in JSON.

## speech devices

```
speech devices [--json]
```

Lists the CPU and the GPUs, with their kind (`cpu`, `gpu` or `igpu`), description and memory. `--json` prints one
object, for a program that chooses a device before it starts a worker:

```json
{"devices":[{"name":"Vulkan0","description":"NVIDIA GeForce RTX 2080","kind":"gpu","memory_total":8589934592,"memory_free":7516192768}]}
```

## speech models

```
speech models [--json]
```

Lists the catalog: each file of each model, by the name that fetches it, the name alone for the type the name means and
`NAME:TYPE` for the others, with its type, size, whether it is fetched (`yes`, `no`, or how much of a stopped fetch is
there), its task and languages; the model to start with for each language; and the old files in the model folder.

```
NAME                           TYPE  SIZE     FETCHED  TASK         LANGUAGES
qwen3-tts-0.6b                 q8_0  1.21 GB  no       synthesis    de en es fr it ja ko pt ru zh
qwen3-tts-0.6b:q6_k            q6_k  0.99 GB  no       synthesis    de en es fr it ja ko pt ru zh
qwen3-tts-0.6b:q5_k            q5_k  0.87 GB  no       synthesis    de en es fr it ja ko pt ru zh
qwen3-tts-0.6b:q4_k            q4_k  0.76 GB  no       synthesis    de en es fr it ja ko pt ru zh
...
reazonspeech-v2                q8_0  0.66 GB  no       recognition  ja
reazonspeech-v2:f16            f16   1.24 GB  no       recognition  ja
...

The model to start with for each language:
  synthesis    qwen3-tts-0.6b                 de en es fr it ja ko pt ru zh
  recognition  qwen3-asr-0.6b                 ar cs da de el en es fa fi fil fr hi hu id it ko mk ms nl pl pt ro ru sv th tr vi yue zh
               reazonspeech-v2                ja
               parakeet-tdt-0.6b-v3           bg et hr lt lv mt sk sl uk

A subcommand given a NAME[:TYPE] fetches its file the first time; speech pull fetches it ahead, and speech rm removes it.

Old files, which no model of this release names; speech rm --old removes them:
  1.89 GB  sakasegawa--Irodori-TTS-v4.1-Small-MF-GGUF/99e5d77f6d92a70d4b9bff52829ac118fa1bf7d3/Irodori-TTS-848M-MF-v4.1-F16.gguf
```

`--json` prints one object for programs: `version`, `directory`, `models` and `old`. Each model has `name`,
`repository`, `revision`, `task`, `languages`, `voice_files`, `start` (the languages it is the model to start with for),
`type` (the type its name alone means) and `files`. Each file has `type`, `file`, `size`, `sha256`, `url`, `path` in the
model folder, `fetched`, and `partial`, the bytes of a stopped fetch. Each old file has `path` and `size`.

## speech pull

```
speech pull NAME[:TYPE]...
```

Fetches each model's file unless it is in the model folder, and prints its path on stdout, so that
`model=$(speech pull qwen3-asr-0.6b)` gives a script the path. On stderr it says what it fetches, from where, how far it
is and how fast, and that it checked the SHA-256. On a terminal the progress is one line, rewritten as it moves.

```
fetching qwen3-asr-0.6b, Qwen3-ASR-0.6B-Q8_0.gguf (0.84 GB), from https://huggingface.co/sakasegawa/Qwen3-ASR-0.6B-GGUF into ...
resuming at 7 MB of 842 MB
  87 MB of 842 MB, 10%, 5.2 MB/s
  ...
fetched qwen3-asr-0.6b in 134 s and checked its SHA-256
```

## speech rm

```
speech rm NAME[:TYPE]... | speech rm --old
```

Removes each model's file, with what was fetched of it and the folders that this leaves empty. `--old` removes every
file that no model of this release names. `speech rm` waits for a process that fetches the same file. A model that is not
in the folder is refused with exit 1, before anything is removed.

## speech quantize

```
speech quantize MODEL OUT --type f16|q8_0|q6_k|q5_k|q4_k
```

- Writes MODEL, a model file of F32 weights, in another weight type, on the CPU. Some tensors stay in a wider type, as
  each family needs. A file in another type is refused, and so is Silero VAD's, which is F32 alone.
- OUT is the file to write, or a folder, in which the file takes its usual name, such as `Qwen3-ASR-0.6B-Q4_K.gguf`.
- The F32 file comes from the official checkpoint through the model's converter, in a clone of this repository with
  [uv](https://docs.astral.sh/uv/):

  ```sh
  cd reference/qwen3-asr && uv run python convert.py Qwen3-ASR-0.6B ../../models   # models/Qwen3-ASR-0.6B-F32.gguf
  speech quantize ../../models/Qwen3-ASR-0.6B-F32.gguf ../../models --type q4_k
  ```

  The other converters are `reference/qwen3-tts/convert.py 0.6b|1.7b`, `reference/irodori-tts/convert.py mf|rf` and
  `reference/fastconformer/convert.py <model name>`.

`q6_k`, `q5_k` and `q4_k` make smaller files that compute less exactly. In the stage checks against the official
implementation:

| Family | Q6_K to Q4_K |
|---|---|
| Qwen3-TTS | in Q4_K, the 0.6B model spoke 2 of 20 sentences as other words, where F32 did so with 1 |
| Irodori-TTS | the speech still says the text, with lengths that part from the official's |
| FastConformer | reazonspeech-nemo-v2 writes 1.5% to 3.5% of its characters otherwise, against 0.9% in Q8_0 |
| Qwen3-ASR | the 0.6B model keeps the official text on 34 (Q6_K) to 21 (Q4_K) of 40 requests, the 1.7B on 36 to 28 |

## speech update

```
speech update [--check]
```

Asks github.com for the latest release. When it is later than this one, and this `speech` is the one the installer put
in place, `speech update` runs the installer with `--no-modify-path` (`-NoModifyPath` on Windows), which downloads the
release, checks it against the release's `SHA256SUMS` and replaces this version ([install.md](install.md#update)), and
exits with the installer's status. `--check` only says whether a later release is out. Messages go to stderr.

```
$ speech update
speech.cpp 0.8.8 is out (this is 0.8.7); running the installer
Downloading https://github.com/nyosegawa/speech.cpp/releases/download/v0.8.8/speech-0.8.8-macos-arm64-metal.zip
...
```

A `speech` built from source or copied elsewhere is left as it is, and `speech update` exits with 1, saying where it is
and how to update it: run the installer, or pull and build again. Without the network it exits with 1 and an `io`
failure.

## speech serve and speech worker

```
speech serve [MODEL [MODEL [MODEL]]] [--open] [--host 127.0.0.1] [--port 8080] [--cors-origin ORIGIN|*]...
             [--add-voice NAME=FILE]... [--device NAME] [--threads N] [--no-warmup] [request options]
speech worker MODEL [--add-voice NAME=FILE]... [--device NAME] [--threads N] [--no-warmup] [request options]
```

[server.md](server.md) and [worker.md](worker.md) describe them. `speech serve` takes a detection model beside a
synthesis and a recognition model, for transcriptions by regions; the worker takes none.

A request option given as a flag is the default of every request that does not set it, in place of the model's own:
`speech serve irodori-tts-mf --steps 8` samples in 8 steps unless a request sets `steps`. A request's own value wins,
and an option no model given takes, or a value out of its range, stops the program before it serves.

`speech serve` gives a flag to every model given that takes the option. After `tts.`, `asr.` or `vad.` it gives it to
the synthesis, recognition or detection model alone, and wins over the same flag without one:
`speech serve qwen3-tts-0.6b qwen3-asr-0.6b --language ja --asr.language auto`. The worker's `ready` and the server's
`/v1/models` list the defaults given.
