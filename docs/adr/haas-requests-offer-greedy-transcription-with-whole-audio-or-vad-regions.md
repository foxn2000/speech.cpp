# Haas requests offer greedy transcription with whole audio or VAD regions

## Context

The official haas_asr package wraps NeMo with plain transcription, quiet-point splitting for long audio, and custom
vocabulary boosting. Its `Vocabulary` contains written forms, readings and a boosting strength. Boosting compiles
token sequences into NeMo's `GPUBoostingTreeModel`, decodes with and without the bonus using one encoder pass,
guards changes by complete listed words and a volume check, and formats matched spellings in their written form.
These operations change the transcript; they are not a text prompt or a replacement of words alone.

## Decision

| Official input or operation | speech.cpp request |
|---|---|
| `transcribe(model, audios)` without a vocabulary | One audio per request, with the family's greedy TDT decoding. Multiple audios are separate requests. |
| `batch_size` | Not offered. One model serves one request at a time; batching is the Python wrapper's scheduling choice. |
| `transcribe_longform` | No implicit splitting. Recognize the audio whole, as the family's other full-context models do, or use the tools' transcription by regions with `--vad` or the server's `chunking_strategy`. |
| `chunk_seconds`, `search_seconds`, `window_seconds` | Not offered. They configure the wrapper's volume-based cuts, not the recognition model. |
| `separator` | Not offered as a recognition option. The tools join VAD regions by their existing language-aware rule. |
| `vocabulary`, its words, readings and `alpha`, and `guard` | Not offered. NeMo's boosting tree, two decodes, guards and spelling formatter need a separate checked port; the existing TDT decoder has none of them. |
| `load_model`'s `device` and `dtype` | The caller selects a speech.cpp device and GGUF weight type. There is no bfloat16 CUDA request mode. |

The ordinary recognition options, including timestamps and the decoding device, remain those of the family.
Haas does not declare a `decoding` choice because greedy TDT is its only decoding.

The alternatives were turned down:

- Reproducing `transcribe_longform` inside a recognition request. Its default 30 s chunks move cuts to low-volume
  windows and change the context at each cut. The library's whole-audio request and the tools' explicit VAD regions
  already give callers those two responsibilities without silently substituting a third policy.
- Exposing `Vocabulary` through `instructions` or text replacement. Neither computes token bonuses, their withdrawal
  for incomplete phrases, or the guarded comparison with an unbiased decode, so neither preserves its meaning.
- Offering a vocabulary option before porting boosting. It would accept a request that the decoder cannot honor.
- Matching haas_asr's default bfloat16 CUDA cast through a request option. speech.cpp's precision comes from its GGUF
  weight types and backend arithmetic; a dtype string would promise an execution mode those backends do not provide.

## Consequences

Comparisons with the official implementation use plain greedy transcription without vocabulary or longform
splitting, and state the weight type and device. A caller that needs the official phrase boosting uses haas_asr;
transcription by VAD regions can differ from both whole-audio recognition and the wrapper's quiet-point chunks.
