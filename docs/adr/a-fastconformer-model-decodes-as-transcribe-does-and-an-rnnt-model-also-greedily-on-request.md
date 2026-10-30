# A FastConformer model decodes as transcribe() does, and an RNN-T model also greedily on request

## Context

NeMo's `transcribe()` decodes each FastConformer model with the decoding its checkpoint configures by default, and the
model cards' results are that decoding's. For the parakeet models it is greedy TDT (`greedy_batch`, durations 0 to 4, at
most 10 tokens on one frame). parakeet-tdt_ctc-0.6b-ja is a hybrid with a CTC head beside its TDT decoder, and the two
do not always agree: on one of the three FLEURS utterances the dumps cover, CTC writes ブラックスタフ where TDT writes
ブラックスタッフ. parakeet-tdt-0.6b-v3 has no CTC head. On Metal, TDT took 0.28 s for a 25.5 s utterance in `speech asr` where
CTC took 0.25 to 0.27 s.

For reazonspeech-nemo-v2 it is NeMo's alignment-length synchronous beam search with a beam of 4. Each step of the search
is one graph on the device, and the steps number about the frames and the labels together: 4,864 graphs for the 311 s
input of the checks, where the encoder takes one, and 3.3 s of the 4.9 s that input took on an Apple M5 with F16
weights on Metal (2026-10-06). A caller that waits for the text once a person stops speaking waits for those steps. The
checkpoint also configures greedy decoding (`decoding.greedy.max_symbols` 10), which NeMo runs by label looping with the
strategy `greedy_batch` or frame by frame with `greedy`; the two emit the same tokens on the same frames on all ten
inputs of the checks. Its text differs from the beam search's on five of the ten: the greedy texts lack some of the
commas and full stops, one has サウスカロライナ where the beam search writes サウスカロロナイナ, and the 311 s input has a few
other words.

On the 4,483 utterances of Common Voice 8.0's Japanese test set, through speech-bench with each utterance's audio
trimmed by VAD with a 0.2 s margin, reazonspeech-nemo-v2 in F16 on an Apple M5 (Metal) gave on 2026-10-08:

| Decoding | CER | CER with accepted spellings | Empty results | Median time per utterance | p90 |
|---|---|---|---|---|---|
| Beam search | 12.03% | 7.13% | 0 | 0.110 s | 0.167 s |
| Greedy | 12.50% | 7.69% | 4 | 0.077 s | 0.104 s |

## Decision

- **A model decodes by default with the decoding `transcribe()` runs for it**, ported from NeMo and checked step by step
  against it: greedy TDT for the parakeet models and Haas-ja-v1.0, ported from NeMo's label-looping decoder, and the beam search for
  reazonspeech-nemo-v2. A hybrid's CTC head is not converted, and there is no option to choose it.
- **An RNN-T model also decodes greedily when a request asks**, through the option `decoding`, a string whose choices
  the model declares: reazonspeech-nemo-v2 takes `beam`, the default, and `greedy`. The choices name the decodings, not
  NeMo's strategies, two of which, `greedy_batch` and `greedy`, are one decoding. Only a model with a choice declares
  the option; the parakeet models and Haas do not take it, as a model that cannot change its speed does not take `speed`. The
  option has no neutral value, since each model's default is its own.
- **Greedy decoding takes one graph per token.** The prediction network's output changes only when a token is emitted,
  so one graph computes the prediction for the last token and the joint at a run of frames from the current one, and
  the host takes the first frame whose label is a token, the first of equal values as torch's argmax takes it. A run of
  8 frames follows each token, and a run twice as long a run of blanks, up to 64 frames. Measured on an Apple M5 on
  2026-10-07 while other work loaded the machine: the 311 s input took 1,008 graphs for its 931 tokens against the beam
  search's 4,864, about 1.2 s against 3.1 s with F16 weights on Metal, where NeMo evaluates the joint 4,822 times; the
  25.5 s FLEURS utterance 107 graphs against 638, about 0.09 s against 0.55 s.
- **The limit of greedy decoding is in the file.** The most tokens it emits on one frame, the checkpoint's 10, is
  `fastconformer.decoder.rnnt.max_symbols` of fastconformer's layout 2. The reader brings a file of layout 1 up to
  layout 2, giving an RNN-T file the limit of 10: layout 1's converter pinned one RNN-T checkpoint,
  reazonspeech-nemo-v2, and the upgrade refuses an RNN-T file of layout 1 under another `general.name`.

The alternatives were turned down:

- CTC as an option beside TDT. Two decoders double the checks, the file and the surface a caller has to understand, for
  a head NeMo itself does not use by default and that the other FastConformer models lack.
- CTC as the default for its speed. TDT costs a few hundredths of a second more, and its text is the one NeMo and the
  model card give.
- The beam search alone. Greedy decoding takes a fifth of the graphs and 70% of the median time per utterance, which a
  caller that waits for the text can prefer to the commas.
- Greedy decoding as the default. `transcribe()` runs the beam search, whose text has the commas and full stops and the
  lower error rate, and a caller that compares speech.cpp with NeMo or the model card compares it with the beam search.
- Greedy decoding frame by frame, one graph per evaluation of the joint, as `GreedyRNNTInfer` runs it: 4,822 graphs for
  the 311 s input, as many as the beam search's steps.
- The argmax on the device with `ggml_argmax`. ggml's CPU backend keeps the last of equal values where torch keeps the
  first, so the blank, the last output, would win a tie that NeMo gives to a token.
- A beam size as the option, greedy being a beam of 1. NeMo's beam search with a beam of 1 runs a greedy search of its
  own, not the decoding of the strategy `greedy_batch`, and beams of 2 and 3 have no dump to check them against.
- A boolean `greedy`. It leaves no room for another decoding, and its false would misdescribe the parakeet models.
- The parakeet models or Haas declaring `decoding` with `greedy` its one choice. It steers nothing, and no client sends a
  decoding as a hint, as OpenAI's clients send a language.
- The limit of 10 in the C++, NeMo's default as well as the checkpoint's. It is the checkpoint's setting, which another
  RNN-T checkpoint can set otherwise.
- The new key optional, read as 10 where a file lacks it. A file whose converter forgot the key and a file of layout 1
  would look the same.

## Consequences

A request that leaves `decoding` unset gives the text NeMo's `transcribe()` returns. A caller finds the option in a
model's information and chooses per request through every entry point: the worker's `transcribe`, the server's form
field `decoding` and `speech asr --decoding`. `reference/fastconformer/dump.py --greedy` dumps NeMo's
greedy decoding beside the beam search's, and the transducer, times and C API checks compare `greedy` with it.
