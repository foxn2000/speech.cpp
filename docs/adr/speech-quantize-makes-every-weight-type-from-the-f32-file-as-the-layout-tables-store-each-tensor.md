# speech quantize makes every weight type from the F32 file, as the layout tables store each tensor

## Context

Each family's `layout.cpp` lists the tensors of its file and the types its reader takes each in. gguf-py 0.19 quantizes
Q8_0, Q4_0, Q5_0 and a few others, and none of the K-quants (Q6_K, Q5_K, Q4_K), which ggml quantizes
(`ggml_quantize_chunk()`). Quantizing in a converter takes the Python environment and the checkpoint. ASIST and others
pin the released F16 and Q8_0 files by SHA-256.

A K-quant's block holds 256 values of a row. The rows of some matrices are not whole blocks of it: Qwen3-ASR 0.6B's
encoder and projector, 896 wide; FastConformer's prediction network and joint, 640; Irodori-TTS's AdaLN, of rank 192,
the DiT's feed-forward output, 3680, the speaker encoder's input, 128, and its feed-forward output, 1996, which is not
whole blocks of Q8_0's 32 either. Some weights reach operations that ggml runs from F16 or F32 alone: Qwen3-ASR's
convolution kernels through `ggml_im2col()`, and FastConformer's through `ggml_conv_2d_direct()`,
`ggml_conv_2d_dw_direct()` and `ggml_ssm_conv()`. The matrices and embeddings of every family are read by
`ggml_mul_mat()` and `ggml_get_rows()` alone, which ggml v0.27.0 runs from every K-quant on the CPU, Metal and Vulkan
(each backend's `supports_op`).

On the nine released models, ggml's Q8_0 and F16 are gguf-py's on every tensor, and `speech quantize` of each F32 file
writes the released F16 and Q8_0 files byte for byte (`checks/compare/quantize_compare.py`, 2026-10-08).

## Decision

- **The library quantizes.** `speech_quantize()` and `speech quantize IN OUT --type` write a model file of F32 weights in
  F16, Q8_0, Q6_K, Q5_K or Q4_K with ggml's own quantizers, on the CPU. They take F32 weights alone: a quantized file is
  not quantized again, and F16 weights are refused too, since a type made of them would hold other bytes than the same
  type made of the F32 weights, under the same name.
- **The layout tables decide each tensor's type.** A `TensorSpec` carries a `Storage`: for a file of each weight type,
  the types the tensor takes in order, of which the file holds it in the first whose blocks its rows are whole blocks
  of. The reader takes a tensor in any of its storage's types, so that the choice can change without refusing the files
  made before; `speech quantize` writes the one the order gives. Three kinds of storage serve the families:
  `quantized_storage()` for a matrix or embedding that only `ggml_mul_mat()` and `ggml_get_rows()` read, `half_storage()`
  for Qwen3-TTS's codec and Qwen3-ASR's convolution kernels, F16 in every file but an F32 one, as the released files
  hold them, and `float32_storage()` for the rest, every tensor of Silero VAD among them. Nothing else chooses a type,
  and a type that would leave most of a file's bytes in another type, as every type does with Silero VAD's, is refused
  ([the record of detection](detecting-speech-is-a-task-of-its-own-run-by-speech-detect-with-silero-vads-options-and-f32-files.md)).
- **A tensor a type cannot hold takes the next wider type the storage lists.** In a K-quant file a matrix whose rows are
  not whole blocks of 256 values takes Q8_0, and F16 where they are not whole blocks of 32; the same F16 stands for Q8_0
  in a Q8_0 file. A weight whose operation reads F16 or F32 alone has no quantized type in its storage at all.
- **The converters write F32 alone.** `speech quantize` writes the released F16 byte for byte, and a converter that
  wrote another type would be a second place deciding the types.
- **A file names the release that reads it.** `speech.requires` is the latest of the input's own, the first release
  that reads the weight type's `general.file_type`, which every family reads alike, and the first release whose reader
  of the family takes each tensor in its type, which each storage gives per type from the family's readers: the reader
  of 0.7 took FastConformer's matrices in F16 and F32 alone, where the other families' took Q8_0 as well. A Q6_K, Q5_K
  or Q4_K file and a FastConformer file in Q8_0 name 0.8.0, and the other F16 and Q8_0 files 0.7.0, so that they are
  the released ones. A reader that does not know a file's `general.file_type`, or finds a tensor in a type its layout
  does not list, names the release `speech.requires` gives when it comes after its own, as a newer layout is refused.
  `general.file_type` is 18 (MOSTLY_Q6_K), 16 (MOSTLY_Q5_K_S) or 14 (MOSTLY_Q4_K_S), the one of gguf-py's two values for
  Q5_K and Q4_K that names no mix of wider types, and the file is named `-Q6_K`, `-Q5_K` or `-Q4_K`.
- **The metadata is copied from the file's own bytes.** gguf's writer writes a tensor's axes up to its last above 1,
  where gguf-py writes every axis of the numpy array: through it a kernel of width 1, `[768, 768, 1]` in a released
  file, would become `[768, 768]`, and the file would differ from the released one.

The alternatives were turned down:

- Quantizing in the converters, with gguf-py. It does not quantize the K-quants, and the converters and the tables would
  both decide the types.
- Q5_0, Q5_1 or IQ4_NL for a matrix a K-quant cannot hold, as llama.cpp substitutes them. Each is a type more for every
  backend and every check, where Q8_0 already runs everywhere and is checked.
- One first release per weight type for every family. A FastConformer file in Q8_0 would name 0.7.0, which refuses it.
- A reader that takes a tensor only in the type the rule gives for the file's weight type. The rule could then never
  change without refusing the files made under the one before.
- A `speech.layout` raised for the K-quant files, so that older readers would name the release. A layout says which
  keys and tensors a file holds, which a Q4_K file holds as an F16 one does.
- Writing the metadata through gguf's writer, which would change the released files' bytes as above.

## Consequences

Converting a model is two steps, and the F32 file, 8.1 GB for the 1.7B models, is written first. FastConformer's files
can be Q8_0 and K-quants. Qwen3-ASR 0.6B's encoder, 896 wide, is Q8_0 in its K-quant files, and FastConformer's
prediction network and joint, 640 wide, too.
