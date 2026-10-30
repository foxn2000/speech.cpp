# FastConformer

This page describes speech.cpp's port of NVIDIA NeMo's [FastConformer](https://arxiv.org/abs/2305.05084) recognizers:
the parakeet models, ReazonSpeech and Haas-ja-v1.0, which write the text of a recording.

The catalog names fetch these files:

| Name | Upstream | Languages | File |
|---|---|---|---|
| `reazonspeech-v2` | [reazon-research/reazonspeech-nemo-v2](https://huggingface.co/reazon-research/reazonspeech-nemo-v2) | ja, with punctuation | `reazonspeech-nemo-619M-v2-Q8_0.gguf`, 0.66 GB |
| `parakeet-tdt_ctc-0.6b-ja` | [nvidia/parakeet-tdt_ctc-0.6b-ja](https://huggingface.co/nvidia/parakeet-tdt_ctc-0.6b-ja) | ja | `parakeet-tdt_ctc-0.6B-ja-Q8_0.gguf`, 0.66 GB |
| `parakeet-tdt-0.6b-v3` | [nvidia/parakeet-tdt-0.6b-v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) | 25 European languages, with punctuation and capitals | `parakeet-tdt-0.6B-v3-F16.gguf`, 1.26 GB |

parakeet-tdt-0.6b-v3's languages are those of its model card: `bg` `cs` `da` `de` `el` `en` `es` `et` `fi` `fr` `hr`
`hu` `it` `lt` `lv` `mt` `nl` `pl` `pt` `ro` `ru` `sk` `sl` `sv` `uk`.

```sh
speech asr parakeet-tdt_ctc-0.6b-ja utterance.wav                  # the text on stdout
speech asr parakeet-tdt-0.6b-v3 --timestamps utterance.wav         # with the times of its segments
speech asr reazonspeech-v2 --vad silero-vad meeting.wav            # a recording of minutes, by its regions of speech
speech asr reazonspeech-v2 --decoding greedy meeting.wav           # faster, with greedy decoding
speech worker parakeet-tdt_ctc-0.6b-ja                             # a recognition worker
speech serve parakeet-tdt-0.6b-v3                                  # POST /v1/audio/transcriptions
```

## Haas-ja-v1.0

[third-intelligence/Haas-ja-v1.0](https://huggingface.co/third-intelligence/Haas-ja-v1.0) recognizes Japanese.
Its converted GGUF files are not published on Hugging Face and it has no catalog name. Use a local converted file:

```sh
speech asr models/Haas-0.6B-ja-v1.0-F32.gguf utterance.wav
speech asr models/Haas-0.6B-ja-v1.0-F32.gguf --timestamps utterance.wav
speech asr models/Haas-0.6B-ja-v1.0-F32.gguf --vad silero-vad meeting.wav
```

A request uses greedy TDT decoding and recognizes its audio whole; `--vad` recognizes each region of speech
separately. Haas attends over the whole request, so its encoder's memory grows with the square of the input length.
Custom vocabulary boosting and splitting at quiet points are not supported.

The F32 conversion can be written as F16, Q8_0, Q6_K, Q5_K or Q4_K with
[speech quantize](../cli.md#speech-quantize).

## What it does

- Each model writes the text NeMo's `transcribe()` writes by default: ReazonSpeech with a beam search, the parakeet models
  and Haas with greedy decoding.
- A request is recognized whole, however long, but a stretch that holds several sentences loses whole sentences, as in
  NeMo itself: on minutes of Common Voice ja clips joined with pauses, reazonspeech-v2 dropped 522 of 600 and
  parakeet-ja 266. Recognize a long recording by the regions where someone speaks, with `--vad silero-vad` or
  `chunking_strategy` (the [record](../adr/transcription-by-regions-recognizes-each-region-where-someone-speaks-alone-in-the-tools-and-joins-the-texts.md)
  has the measurement). ReazonSpeech's time and memory grow with the length; the parakeet models and Haas attend over the whole
  request, so theirs grow with its square.
- parakeet-v3 finds the language of the audio itself. None of the models takes a language: a request's `language` is only
  checked against the model's languages.
- With `timestamps`, the result also has the tokens and segments with their times in seconds. A segment ends at a full
  stop, a question mark or an exclamation mark.
- On a GPU, the encoder runs on the GPU, and the decoding runs on the CPU where the file's decoding weights are
  quantized (Q8_0, Q6_K, Q5_K and Q4_K files) and on the GPU where they are F16 or F32: each step of the decoding is a
  small computation, which the CPU finishes before a round trip to the GPU would with quantized weights but not with F16
  ones. `decode_on_cpu` chooses either way per request. When a request first decodes on the CPU, the model copies the
  weights the decoding reads into the CPU's memory and keeps them, 12 MB for the Q8_0 files of reazonspeech-v2 and
  parakeet-ja and 35 MB for the F16 file of parakeet-v3, and the CPU computes with `--threads`. The CPU's arithmetic
  differs from the GPU's, so a long recording can come out a word different where two words score almost the same.

Not supported: the CTC head of parakeet-tdt_ctc-0.6b-ja, which NeMo does not decode with by default.

## Options

| Option | Default | Notes |
|---|---|---|
| `language` | `auto` | one of the model's languages; only checked |
| `timestamps` | false | add the times of the tokens and segments |
| `decoding` | `beam` | reazonspeech-v2 only: `beam` or `greedy`. Greedy is faster and can write another text, often with fewer commas and full stops |
| `decode_on_cpu` | true for Q8_0 and K-quant files, false for F16 and F32 | for a model on a GPU, decode on the CPU (true) or on the GPU (false). A model on the CPU decodes there either way |

## Accuracy

Every stage is checked against NeMo's tensors on utterances of FLEURS' test split, on an Apple M5:

| Model | Inputs | Text from the audio, CPU F32, CPU F16 and Metal |
|---|---|---|
| parakeet-tdt_ctc-0.6b-ja | 3 ja_jp utterances, 6 to 26 s | NeMo's text on all three |
| parakeet-tdt-0.6b-v3 | 12 utterances of en_us, de_de, fr_fr and es_419, 6 to 23 s | NeMo's text on all twelve |
| reazonspeech-nemo-v2 | 8 ja_jp utterances and 2 joined inputs of 65 s and 311 s | NeMo's text on all ten, with beam search and with greedy decoding |

The times of tokens and segments are NeMo's exactly with F32 weights, and with F16 weights decoding on Metal. Decoding on
the CPU with F16 weights, a token can fall one frame (80 ms) from NeMo's where two frames score almost the same: one
token in 2 of reazonspeech-nemo-v2's 20 decodings.

On the 4,483 utterances of Common Voice 8.0's Japanese test set, trimmed to the voice, the CER in Q8_0 is 7.93% for
parakeet-tdt_ctc-0.6b-ja, 12.04% for reazonspeech-nemo-v2 with beam search and 12.47% with greedy decoding (3.01%,
7.12% and 7.66% accepting other spellings of the same words), on the M5 and within 0.01 points on the RTX 2080.
parakeet-tdt-0.6b-v3's WER in F16 on 300 English utterances of FLEURS is 8.76% on the M5 and 8.78% on the RTX 2080.

## Weight types

Each catalog model comes in five types; `NAME:TYPE` fetches one (`speech asr parakeet-tdt-0.6b-v3:q8_0`). speech-bench, two
rounds, on an Apple M5 with Metal on 2026-10-16 and an RTX 2080 with Vulkan on 2026-10-14. The error is the CER on the first 500
utterances of Common Voice 8.0's Japanese test set, on which the CER is higher than on the whole set, and after the
slash the CER accepting other spellings of the same words; for parakeet-tdt-0.6b-v3 it is the WER on 300 English
utterances of FLEURS. The error is the M5's, and the RTX 2080's is within 0.05 points of it. The peak memory is the
process's on the M5, whose GPU shares it, and the VRAM the video memory in use on the RTX 2080.

| Model | Type | File | Error | Peak memory, M5 | VRAM, RTX 2080 | RTFx, M5 / RTX 2080 |
|---|---|---|---|---|---|---|
| parakeet-tdt_ctc-0.6b-ja | F16 | 1.24 GB | 9.09% / 2.37% | 1.35 GB | 1.32 GB | 105 / 69 |
| parakeet-tdt_ctc-0.6b-ja | Q8_0 | 0.66 GB | 9.14% / 2.35% | 0.78 GB | 0.74 GB | 131 / 76 |
| parakeet-tdt_ctc-0.6b-ja | Q6_K | 0.51 GB | 9.08% / 2.48% | 0.63 GB | 0.60 GB | 125 / 80 |
| parakeet-tdt_ctc-0.6b-ja | Q5_K | 0.43 GB | 8.79% / 2.35% | 0.55 GB | 0.52 GB | 120 / 72 |
| parakeet-tdt_ctc-0.6b-ja | Q4_K | 0.36 GB | 9.26% / 2.48% | 0.47 GB | 0.44 GB | 132 / 77 |
| reazonspeech-nemo-v2, beam search | F16 | 1.24 GB | 14.12% / 7.14% | 1.41 GB | 1.34 GB | 49 / 28 |
| reazonspeech-nemo-v2, beam search | Q8_0 | 0.66 GB | 14.17% / 7.15% | 0.84 GB | 0.75 GB | 75 / 35 |
| reazonspeech-nemo-v2, beam search | Q6_K | 0.51 GB | 13.91% / 6.98% | 0.69 GB | 0.61 GB | 74 / 34 |
| reazonspeech-nemo-v2, beam search | Q5_K | 0.43 GB | 13.96% / 7.06% | 0.61 GB | 0.53 GB | 70 / 34 |
| reazonspeech-nemo-v2, beam search | Q4_K | 0.36 GB | 14.20% / 7.10% | 0.53 GB | 0.45 GB | 75 / 34 |
| parakeet-tdt-0.6b-v3 | F16 | 1.26 GB | 8.76% | 1.40 GB | 1.45 GB | 132 / 85 |
| parakeet-tdt-0.6b-v3 | Q8_0 | 0.67 GB | 9.04% | 0.82 GB | 0.86 GB | 165 / 101 |
| parakeet-tdt-0.6b-v3 | Q6_K | 0.52 GB | 8.87% | 0.67 GB | 0.72 GB | 159 / 103 |
| parakeet-tdt-0.6b-v3 | Q5_K | 0.44 GB | 9.15% | 0.58 GB | 0.63 GB | 153 / 102 |
| parakeet-tdt-0.6b-v3 | Q4_K | 0.36 GB | 9.51% | 0.52 GB | 0.56 GB | 166 / 102 |

In the Q6_K, Q5_K and Q4_K files the prediction network and the joint stay Q8_0.

## Speed

speech-bench, on the M5 on 2026-10-16 and the RTX 2080 on 2026-10-14, one request at a time, with the files the
names fetch: Q8_0 for parakeet-tdt_ctc-0.6b-ja and reazonspeech-nemo-v2 on the 4,483 utterances of Common Voice 8.0's
Japanese test set (4,479 kept by VAD), one round, and F16 for parakeet-tdt-0.6b-v3 on 300 English utterances of FLEURS (299 kept), the medians of three rounds. The RTFx
is the seconds of audio over the seconds they took, so larger is faster.

| Model | Utterances | p50 latency, M5 / RTX 2080 | p90 latency, M5 / RTX 2080 | RTFx, M5 / RTX 2080 |
|---|---|---|---|---|
| parakeet-tdt_ctc-0.6b-ja | Common Voice ja | 0.027 s / 0.050 s | 0.036 s / 0.076 s | 137 / 76 |
| reazonspeech-nemo-v2, beam search | Common Voice ja | 0.046 s / 0.092 s | 0.076 s / 0.179 s | 76 / 36 |
| reazonspeech-nemo-v2, greedy | Common Voice ja | 0.033 s / 0.043 s | 0.049 s / 0.083 s | 108 / 73 |
| parakeet-tdt-0.6b-v3 | FLEURS en | 0.058 s / 0.095 s | 0.091 s / 0.125 s | 130 / 86 |

A long recording is recognized whole. ReazonSpeech, parakeet-tdt_ctc-0.6b-ja and Haas use global attention, whose memory
grows with the square of the input length. Use `speech asr --vad silero-vad` to transcribe long recordings by region.
