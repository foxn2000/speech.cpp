# Drama 3

This page describes speech.cpp's port of Fish Audio's [Drama 3](https://huggingface.co/fishaudio/drama-3), a speech
model of 35B parameters, 3B of them active, that speaks a text as one speaker or as a scene of several, with cues such as
laughter written in the text.

Built with Fish Audio. The weights are under the
[Fish Audio Community License 1.0](https://huggingface.co/fishaudio/drama-3/blob/main/LICENSE): commercial use is allowed
up to a yearly revenue of 500,000 US dollars, and a speech or voice-cloning service for others needs a separate agreement
with Fish Audio.

```sh
speech tts Drama-35B-A3B-v3-Q4_K.gguf --voice none -o out.wav "こんにちは、今日はいい天気ですね。"
speech tts Drama-35B-A3B-v3-Q4_K.gguf --voice none -o out.wav "<|speaker:0|> ねえ、聞いた？ <|speaker:1|> [laughs] うん、本当に驚いたよね。"
speech worker Drama-35B-A3B-v3-Q4_K.gguf     # behind the worker protocol
```

## Files

[`reference/drama3/convert.py`](../../reference/drama3/convert.py) converts the checkpoint into one F32 file of 143 GB,
and `speech quantize` makes the other types from it (`--type q4_k` writes 20.8 GB in three and a half minutes on an Apple
M5). A model needs its whole file in memory: speaking a short text with the Q4_K file took 21.1 GB at its peak on the M5.

## What it does

- **Speakers** are written in the text as `<|speaker:0|>` to `<|speaker:15|>`; text before the first tag is speaker 0's.
  The voice `none` gives every speaker a voice of the model's choosing, and a voice file gives its speakers the voices
  of its recordings ([Voices](#voices)).
- **Cues** in brackets, such as `[laughs]` or `[sighs]`, and the words of the text steer how it is spoken.
- **The language** is read from the text. `language` is checked against the model's languages and changes nothing.
- **A long text** is spoken in chunks of about 300 bytes of UTF-8 with at most three speakers, cut at sentence ends,
  then at commas, then at spaces, never inside a tag or a cue. Each chunk is prompted with the first and the latest chunk
  before it, so the voices and the way of speaking carry from one to the next.
- **Audio** is passed as it is made: the first piece after 4 frames, 0.19 s of speech, then pieces of 8, 16, 32, 64 and
  128 frames. The same request with the same seed gives the same samples.

## Voices

A voice file holds recordings of one or more speakers with the words each speaks, made by `speech voice`, and is added
with `--add-voice` as any voice is. Each recording needs its transcript and is speaker 0 unless `--speaker` gives
another; a voice file given to `speech voice` adds its recordings as they are, so a voice for a scene is put together
from voices made before:

```sh
speech voice Drama-35B-A3B-v3-Q4_K.gguf woman.wav --transcript "今日はいい天気ですね。" woman.voice.gguf
speech voice Drama-35B-A3B-v3-Q4_K.gguf man.wav --transcript "明日は雨の予報です。" man.voice.gguf
speech voice Drama-35B-A3B-v3-Q4_K.gguf woman.voice.gguf man.voice.gguf --speaker 0 --speaker 1 cast.voice.gguf
speech tts Drama-35B-A3B-v3-Q4_K.gguf --add-voice cast=cast.voice.gguf --voice cast -o scene.wav \
  "<|speaker:0|> 晴れたね。<|speaker:1|> 明日は雨だって。"
```

- The text names the speakers the voice file holds; a speaker it does not hold gets a voice of the model's choosing.
  The model information describes an added voice by its speakers (`speakers 0 and 1`).
- A recording is read at any rate, mixed to mono and resampled to 44.1 kHz. On the CPU, `speech voice`'s default, the
  codes of a 12.8 s recording were all the official encoder's; on Metal, 88 to 99% of each codebook's were.
- A voice file holds the codec's codes, not the audio, and works with every file of Drama 3 whose codec has the hash it
  carries, whatever its weight type.
- Every chunk of a text is prompted with all of the voice's recordings, so they lengthen the prompt and the wait for the
  first audio: with two recordings of 28 s in all, the first audio came after 1.6 s on the Apple M5 with the Q4_K file.
  A voice whose recordings leave a chunk no room for a text is refused.
- A WAVE file given to `--add-voice` is refused, since it lacks its words; make a voice file of it first. The page of
  `speech serve` makes no voices of Drama 3 for the same reason.

## Options

| Option | Default | Notes |
|---|---|---|
| `voice` | required | `none`, or a voice added from a voice file |
| `language` | `auto` | one of the model's languages; checked, not used |
| `seed` | drawn | 0 to 2^53 - 1 |
| `max_seconds` | 47.6 | the longest each chunk may speak, at most 95.1; a chunk that reaches it stops there, and the result says so |
| `top_k` | 30 | the most likely tokens a draw keeps, 1 to 4097 |
| `top_p` | 0.9 | the share of probability a draw keeps; the token that passes it is dropped |
| `temperature` | 1 | what the logits are divided by |

The defaults are those of the model card, and they apply to every draw of a frame. A text may have up to 8,192 tokens,
and a chunk's prompt up to 8,192 rows. A text is refused when it is empty, holds `<|im_start|>`, `<|im_end|>`,
`<|voice|>`, `<|text|>`, `<|interleave|>` or `<|semantic:`, which only the prompt writes, or names a speaker other than 0
to 15. `instructions`, `speed`, `seconds` and `repetition_penalty` are refused.

## Accuracy

Every stage is checked against the official implementation's tensors, on the CPU with the F32 file:

| Check | Result |
|---|---|
| Codec encoder and decoder | 96.9 dB SNR or more at every stage; the official codes of a 12.8 s reference |
| Text model, 40 layers from each one's own input | 120.2 dB or more at every stage; the official head's logits at 132.9 dB |
| Audio decoder | 107.7 dB or more in every layer and head |
| Sampler | the official probabilities within 3.2e-6 |
| Tokenizer, prompt, chunks | the official tokens, prompts and chunks of every case |

## Speed

`speech tts` on an Apple M5 with Metal and the Q4_K file, the voice `none`, on 2026-10-15: five Japanese and English
texts of 6.0 to 27.4 s of speech, each from a fresh process.

| Time to first audio | RTFx |
|---|---|
| 0.31 to 0.42 s | 1.1 to 1.2 |
