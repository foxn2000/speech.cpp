# The Qwen3 decoder attends with flash attention where a GPU computes it

## Context

The Qwen3 decoder of `src/common/` runs Qwen3-TTS's talker and code predictor and Qwen3-ASR's decoder. It can attend
through two matrix products and a softmax, reading a cache whose values are kept transposed, a row per channel, so that
a step reads the cache once and copies none of it; or through ggml_flash_attn_ext, which llama.cpp runs by default on
Metal and Vulkan, which reads the values a row per position and holds no scores, and which sums the values in half
precision on the CPU. With the two products, the 0.6B Qwen3-ASR model decoded 123 to 132 steps a second on an Apple M5
with Q8_0 on Metal, where llama.cpp b11246 decodes 158 to 165. Measured on that M5 on 2026-10-07, at the lengths of
speech recognition:

| | Products | Flash attention |
|---|---|---|
| Attention of 28 layers of 16 query and 8 key/value heads of 128, 400 positions, Metal | 0.85 ms | 1.17 ms, 0.62 ms read as 416 positions with a mask |
| The same, 2,000 positions, Metal | 2.62 ms | 2.20 ms with the mask |
| The same, 8,000 positions, Metal | 9.59 ms | 8.10 ms with the mask |
| The same, 400 positions, CPU | 1.71 ms | 1.33 ms with the mask |
| The same, 8,000 positions, CPU | 22.1 ms | 51.2 ms with the mask |
| A step of the 0.6B Qwen3-ASR decoder at 350 to 550 positions, Metal, the two interleaved in one process | 7.61 ms | 7.25 ms, read in blocks of 32 with the mask |
| The talker's prompt of 3,665 rows in blocks of 512, 0.6B, F32, Metal | 1.86 s | 1.57 s |

Metal's flash attention reads keys 32 positions at a time in its kernel for a few rows and 64 in the one for many, and
first copies a last block that the positions do not fill into a buffer of its own, which is what makes it slower than
the products at 400 positions.

## Decision

The decoder chooses its attention once, when the model loads, from what its device computes, and logs the choice. On a
GPU whose backend computes ggml_flash_attn_ext for the stack's heads and cache type (`ggml_backend_supports_op`), it
attends with flash attention: it reads a whole number of blocks of 64 positions, with a mask that hides the positions
past each row, and its cache keeps the values a row per position, the positions not yet written set to zero on the
device, since the mask hides a position whose key or value was never written but cannot hide a NaN there. On the CPU,
and on a GPU whose backend does not compute flash attention, it attends with the two products over transposed values.
The choice is no fallback after a failure: it is made from the device's capability before anything runs, and the
decoder checks take an argument that forces either attention, so that both are checked on a GPU.

The alternatives were turned down:

- Flash attention on every device. On the CPU it takes twice as long as the products at 8,000 positions, which a long
  recognition or a long speech reaches, and it sums the values in half precision.
- The products on every device. They are slower than flash attention at every length on Metal once the positions fill
  whole blocks, and they hold the scores of a whole block of rows.
- Refusing a GPU that cannot compute flash attention. A GPU that runs Qwen3-TTS and Qwen3-ASR with the products would
  stop running them.

## Consequences

The cache's layout depends on the attention: `read_cache()` gives keys and values a row per position either way. On
Metal the decoder's error is smaller with flash attention: the prompt's logits of qwen3-asr-decoder-check from the
dump's input lie 51.2 to 68.5 dB from transformers' with the 0.6B model in F32 (43.3 to 70.3 dB with the products) and
50.1 to 73.2 dB with the 1.7B (46.6 to 67.1), every text from the audio and every teacher-forced argmax are the same
with either attention in every type. Where Metal runs flash attention without the tensor API, a prefill in blocks
gives exactly what one graph gives. On the M5, M6, A19 and A20, ggml runs it through the tensor API where a graph's
rows fill enough of the GPU and through its simdgroup kernel where they do not, so that the 1,064 rows of
qwen3-decoder-check's prefill give the same caches and logits in blocks of 512 as in one graph up to 1.9e-3 and 1.3e-3
in relative error on an Apple M5, as the two products' rounding gives (2026-10-17). Forced on Metal, the two products
still pass every decoder check with the same thresholds. Vulkan runs the same flash attention where its
device has the subgroup operations ggml's needs; its speed there has not been measured against the products.
