# Speech recognition runs NeMo models through one FastConformer port

## Context

NVIDIA NeMo's parakeet-tdt-0.6b-v3 (25 European languages) and parakeet-tdt_ctc-0.6b-ja (Japanese), and
reazon-research's reazonspeech-nemo-v2 and third-intelligence's Haas-ja-v1.0 (both Japanese) are one architecture,
FastConformer: a log-mel frontend, a subsampling
by 8 with depthwise convolutions, and conformer layers with relative positional attention. They differ in their decoders
and a few options: TDT for parakeet-v3, TDT and a CTC head for parakeet-ja, RNN-T and local attention for ReazonSpeech,
and the number of mel bins. Haas uses TDT with a KDA prediction network instead of an LSTM and a tokenizer with byte
fallback.

## Decision

speech.cpp ports FastConformer once, as the family `src/families/fastconformer/`, on ggml, and runs these NeMo
models through it, each stage checked against tensors dumped from NeMo itself (`reference/fastconformer/`). Their
decoders run on the same encoder ([the decoding
record](a-fastconformer-model-decodes-as-transcribe-does-and-an-rnnt-model-also-greedily-on-request.md)).

The alternatives were turned down:

- Running parakeet through CrispASR or audio.cpp. Each is a runtime of its own with its own copy of ggml, so `speech`
  and `libspeech` would carry two ggml builds and two ways of choosing a device, and their ports are not checked stage
  by stage against NeMo as this repository's are. audio.cpp v0.8.2's Irodori-TTS adds a distorted copy of the voice on
  Metal, a defect such checks catch.
- Running the models as ONNX through sherpa-onnx. It brings ONNX Runtime as a second dependency beside ggml, runs
  exported weights rather than NeMo's, and reaches the GPU through ONNX Runtime's providers rather than the Metal and
  Vulkan builds every release already has.

## Consequences

Recognition with these models costs one family, whose encoder they share; the family's layout grows with each
decoder it runs. A model outside FastConformer and Qwen3-ASR needs a decision of its own.
