# Models

This page lists the models speech.cpp runs, how to name them, where their files are kept, and their voices and
languages.

## The catalog

Each release carries a catalog of models. A command takes a model's name and fetches its file from Hugging Face the
first time.

| Name | Task | Languages | File the name fetches | Family |
|---|---|---|---|---|
| [`qwen3-tts-0.6b`](https://huggingface.co/sakasegawa/Qwen3-TTS-12Hz-0.6B-CustomVoice-GGUF) | synthesis | de en es fr it ja ko pt ru zh | `Qwen3-TTS-12Hz-0.6B-CustomVoice-Q8_0.gguf`, 1.21 GB | [Qwen3-TTS](models/qwen3-tts.md) |
| [`qwen3-tts-1.7b`](https://huggingface.co/sakasegawa/Qwen3-TTS-12Hz-1.7B-CustomVoice-GGUF) | synthesis | the same | `Qwen3-TTS-12Hz-1.7B-CustomVoice-Q8_0.gguf`, 2.29 GB | [Qwen3-TTS](models/qwen3-tts.md) |
| [`irodori-tts-mf`](https://huggingface.co/sakasegawa/Irodori-TTS-v4.1-Small-MF-GGUF) | synthesis | ja | `Irodori-TTS-866M-MF-v4.1-F16.gguf`, 1.92 GB | [Irodori-TTS](models/irodori-tts.md) |
| [`irodori-tts`](https://huggingface.co/sakasegawa/Irodori-TTS-v4.1-Small-GGUF) | synthesis | ja | `Irodori-TTS-859M-v4.1-F16.gguf`, 1.91 GB | [Irodori-TTS](models/irodori-tts.md) |
| [`irodori-tts-large`](https://huggingface.co/sakasegawa/Irodori-TTS-v4-Large-GGUF) | synthesis | ja | `Irodori-TTS-3.4B-v4-Q8_0.gguf`, 3.88 GB | [Irodori-TTS](models/irodori-tts.md) |
| [`qwen3-asr-0.6b`](https://huggingface.co/sakasegawa/Qwen3-ASR-0.6B-GGUF) | recognition | 30 languages | `Qwen3-ASR-0.6B-Q8_0.gguf`, 0.84 GB | [Qwen3-ASR](models/qwen3-asr.md) |
| [`qwen3-asr-1.7b`](https://huggingface.co/sakasegawa/Qwen3-ASR-1.7B-GGUF) | recognition | 30 languages | `Qwen3-ASR-1.7B-Q8_0.gguf`, 2.18 GB | [Qwen3-ASR](models/qwen3-asr.md) |
| [`reazonspeech-v2`](https://huggingface.co/sakasegawa/reazonspeech-nemo-v2-GGUF) | recognition | ja | `reazonspeech-nemo-619M-v2-Q8_0.gguf`, 0.66 GB | [FastConformer](models/fastconformer.md) |
| [`parakeet-tdt_ctc-0.6b-ja`](https://huggingface.co/sakasegawa/parakeet-tdt_ctc-0.6b-ja-GGUF) | recognition | ja | `parakeet-tdt_ctc-0.6B-ja-Q8_0.gguf`, 0.66 GB | [FastConformer](models/fastconformer.md) |
| [`parakeet-tdt-0.6b-v3`](https://huggingface.co/sakasegawa/parakeet-tdt-0.6b-v3-GGUF) | recognition | 25 European languages | `parakeet-tdt-0.6B-v3-F16.gguf`, 1.26 GB | [FastConformer](models/fastconformer.md) |
| [`silero-vad`](https://huggingface.co/sakasegawa/silero-vad-GGUF) | detection | any | `silero-vad-309K-v6.2-F32.gguf`, 1.2 MB | [Silero VAD](models/silero-vad.md) |

`speech models` prints the same, with what is fetched. Each family's page links the upstream models.

### Which model to start with

No model is chosen for you, since the right one depends on the language. A command that names no model stops and says
which one to start with:

| Task | Model | Languages |
|---|---|---|
| synthesis | `qwen3-tts-0.6b` | de en es fr it ja ko pt ru zh, with its own named speakers, and as it makes the audio |
| recognition | `reazonspeech-v2` | ja: it writes Japanese with punctuation and takes recordings of many minutes whole |
| recognition | `qwen3-asr-0.6b` | the 29 other languages of Qwen3-ASR |
| recognition | `parakeet-tdt-0.6b-v3` | bg et hr lt lv mt sk sl uk, the European languages Qwen3-ASR does not recognize |

A detection model takes no language, and `speech vad` without a model lists every detection model.

## Names

Every subcommand that takes a model (`tts`, `asr`, `vad`, `voice`, `info`, `serve`, `worker`) takes a model file's path,
which ends in `.gguf`, or `NAME[:TYPE]`:

- NAME is a name of the catalog, or the Hugging Face repository of its converted files.
- TYPE is the file's weight type, `f16`, `q8_0`, `q6_k`, `q5_k` or `q4_k`, where the repository holds it. Without it,
  the name means the file in the table above.

```sh
speech asr qwen3-asr-0.6b meeting.wav                       # Qwen3-ASR-0.6B-Q8_0.gguf
speech asr sakasegawa/Qwen3-ASR-0.6B-GGUF meeting.wav       # the same file, named by its repository
speech asr qwen3-asr-0.6b:q8_0 meeting.wav                  # the same file, its type given
speech asr qwen3-asr-0.6b:q4_k meeting.wav                  # Qwen3-ASR-0.6B-Q4_K.gguf, 0.51 GB
speech serve reazonspeech-v2                                # fetched before the server starts
```

Every model but Silero VAD also comes in Q6_K, Q5_K and Q4_K, which take less memory and are less accurate. Each
family's page gives the size, the memory and the accuracy of every type, and `speech models` lists the files.

A name or a type that the catalog does not hold is refused, and the message lists what it holds. A release fetches the
files it was checked with, so a file updated on Hugging Face reaches you with the next release.

## Fetching

- A fetch runs the system's curl, so the system's proxies, `https_proxy` among them, and certificates apply.
- It checks the file's size and SHA-256 against the catalog before the file is used.
- A fetch that stops keeps what it has as `<file>.part`, and the same command resumes it.
- Two processes that need the same file take turns: the second waits for the first's fetch, then uses the file.

## Where models are kept

| System | Model folder |
|---|---|
| macOS | `~/Library/Caches/speech.cpp/models` |
| Linux | `$XDG_CACHE_HOME/speech.cpp/models`, or `~/.cache/speech.cpp/models` |
| Windows | `%LOCALAPPDATA%\speech.cpp\models` |

`SPEECH_MODEL_DIR`, an absolute path, replaces the folder. A file goes to `<owner>--<name>/<sha256>/<file>` in it, by
its content, so a release whose catalog pins the same file at a later commit uses it as it is.
Nothing is removed automatically: `speech models` lists the files that no model of this release names as old, and
`speech rm --old` removes them.

## Voices

A synthesis request needs a `voice`.

- **Qwen3-TTS** speaks with the model's nine speakers: `aiden`, `dylan`, `eric`, `ono_anna`, `ryan`, `serena`, `sohee`,
  `uncle_fu` and `vivian`. `speech info qwen3-tts-0.6b` lists each with its language, gender and description. `dylan`
  (Beijing) and `eric` (Sichuan) speak a Chinese dialect when the language is `zh` or left to the model.
- **Irodori-TTS** speaks in the voice of a reference recording that you add, or with the voice `none`, with which the
  model chooses a voice itself ([models/irodori-tts.md](models/irodori-tts.md#voices)). A voice is added from a WAVE
  file, which is encoded each time it is added, or from a voice file that `speech voice` made once.
- **Drama 3** speaks with the voice `none`, with which the model chooses a voice for each speaker the text names, or
  in the voices of a voice file that `speech voice` made of recordings and their transcripts, each given to a speaker
  ([models/drama3.md](models/drama3.md#voices)). It takes a voice file alone, not a WAVE file.

| Entry point | How to add a voice |
|---|---|
| `speech tts` | `--add-voice NAME=FILE`, repeatable |
| `speech worker` | `--add-voice NAME=FILE`, or the request `add_voice` |
| `speech serve` | `--add-voice NAME=FILE`, or a recording on the page (Irodori-TTS) |
| C API | `speech_voice_add()` |

A voice's name is case-sensitive, and may be neither a voice of the model nor `none`.

## Languages

- A model's languages are BCP 47 tags: two letters, or three for a language without a two-letter code (`yue`, `fil`).
- A request's `language` is one of them, a region or script of one (`ja-JP`, `zh-Hant`), or `auto`, the default, which
  leaves the choice to the model. Any other language is refused.
- Qwen3-TTS and Qwen3-ASR are steered by the language. Irodori-TTS, Drama 3 and the FastConformer models only check it
  against their languages and do not use it. Silero VAD has no languages and takes `auto` alone.
- Qwen3-ASR's results say which language it heard.

## Files on Hugging Face

Each converted repository holds the GGUF file, the codec inside it, and beside it the same name with `.json` added: the
output of `speech info --json` for the file ([c-api.md](c-api.md#model-information-as-json)). A program can show a
model's identity, voices, languages, options and sizes, and choose a file, without downloading the model. The files run
in speech.cpp alone.
