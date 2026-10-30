# Haas uses KDA prediction in the FastConformer family, checked against FLA's CPU recurrence

## Context

Haas-ja-v1.0 shares NeMo's FastConformer encoder and greedy TDT decoder with the parakeet models. Its prediction
network replaces the LSTM with Kimi Delta Attention (KDA), and its SentencePiece vocabulary includes byte fallback.
flash-linear-attention (FLA) 0.5.2's KimiDeltaAttention needs Triton and CUDA, which the macOS reference environment
cannot run. FLA also supplies a pure PyTorch `naive_recurrent_kda` reference.

## Decision

Haas belongs to the `fastconformer` family. Layout 3 selects `lstm` or `kda` with
`fastconformer.decoder.prediction` and records `fastconformer.tokenizer.byte_fallback` and
`fastconformer.tokenizer.remove_extra_whitespaces`. The latter preserves the tokenizer's rule for repeated leading
space markers separately from whether it strips a dummy prefix at all. KDA's checkpoint parameters
stay in the file, including its decay parameters in their checkpoint form. Layouts 1 and 2 read as LSTM models
without byte fallback; their owners need no replacement files. The upgrade takes the whitespace flag from the
pinned tokenizers of the three model names those layouts support and refuses another name.

The port computes the KDA recurrence with `ggml_gated_delta_net`. Its reference keeps Haas's official prediction
network around a CPU float32 substitute for KimiDeltaAttention, installed through `haas_asr._fla.load_kda_class`.
The substitute adapts FLA's projections, short convolutions, normalization, gates and cache contract, and loads
FLA 0.5.2's own naive recurrence rather than rewriting it. The reference keeps the family's NeMo 3.0.0 pin despite
haas_asr's older NeMo bound.

On three FLEURS Japanese dumps, Haas F32 prediction outputs agree with that reference at 128.6–132.8 dB SNR on
the CPU and Metal; token ids and text agree exactly on both. This checks the port against FLA's naive recurrence,
not against execution of its CUDA kernels.

The alternatives were turned down:

- A separate Haas family. It duplicates the encoder, joint and TDT decoding for a change confined to the prediction
  network and tokenizer.
- A CUDA-only reference. It cannot produce the stage dumps on the macOS machines that develop and check the port.
- An independently rewritten reference recurrence. A mistake could be shared with the C++ port, whereas FLA's own
  reference supplies the arithmetic being ported.
- A separate recurrence kernel in speech.cpp. ggml already provides the gated delta operation for KDA, so another
  implementation creates a second backend responsibility for the same operation.

## Consequences

Haas uses the family's existing entry points and request options. An older reader refuses its layout 3 file;
existing model files remain readable through the layout upgrade. The CPU reference does not establish parity with
FLA's CUDA kernels or with haas_asr's default bfloat16 CUDA execution.
