# A FastConformer model on a GPU decodes on the CPU where its decoding weights are quantized, and a request can choose either device

## Context

A FastConformer recognition runs the encoder as one graph and then decodes: each step of greedy TDT, greedy RNN-T or
the beam search is one small graph of the prediction network and the joint. With the parakeet and ReazonSpeech LSTM
networks it contains four products of 2,560 by 640 with a vector, one of 640 by 640 and the joint's output layer, of
up to 8,198 by 640, with a column or four. Haas's KDA prediction network uses the same device-selection rule. The beam search
computes one graph per step of its search, 638 for 25.5 s of audio; greedy decoding about one per token. On a GPU each
graph is a round trip of about 0.3 ms on Metal and 0.6 to 0.7 ms on Vulkan.

On the CPU a step's time follows the bytes of weights it reads. With ggml's CPU backend alone and a pool of threads,
the median microseconds of one step at 1, 2, 4 and 8 threads, measured on 2026-10-13:

| Step | Weights | Apple M5 (1, 2, 4 threads) | Intel i9-9900K, WSL (1, 2, 4, 8 threads) |
|---|---|---|---|
| parakeet-tdt-0.6b-v3, greedy (8,198 outputs) | F32 | 931, 714, 595 | 2,488, 1,616, 1,347, 1,332 |
| | F16 | 410, 261, 209 | 1,428, 791, 562, 530 |
| | Q8_0 | 251, 149, 116 | 768, 421, 228, 182 |
| reazonspeech-nemo-v2, beam of 4 (3,001 outputs) | F32 | 805, 596, 471 | 1,908, 1,203, 960, 913 |
| | F16 | 400, 224, 133 | 1,244, 700, 381, 303 |
| | Q8_0 | 300, 175, 92 | 755, 396, 266, 207 |
| reazonspeech-nemo-v2, greedy | F16 | 245, 147, 97 | 931, 505, 296, 230 |
| | Q8_0 | 180, 106, 57 | 482, 409, 147, 93 |
| parakeet-tdt_ctc-0.6b-ja (3,078 outputs) | F16 | 292, 158, 98 | 932, 552, 308, 214 |
| | Q8_0 | 185, 128, 64 | 516, 257, 138, 109 |

The weights the decoding reads, the prediction network and the joint but for its projection of the encoder, are 2 to 3%
of a file: 11.5 MB of reazonspeech-nemo-v2's Q8_0 file, 11.6 MB of parakeet-tdt_ctc-0.6b-ja's Q8_0 file and 35.0 MB of
parakeet-tdt-0.6b-v3's F16 file, the files the names fetch. In the Q6_K, Q5_K and Q4_K files they are Q8_0.

ggml's CPU backend, built without OpenMP as every release is, starts and joins its threads for every graph unless it is
given a pool of its own.

## Decision

- **A FastConformer model loaded on a GPU copies the tensors its decodings read into the CPU's memory when a request
  first decodes on the CPU**, once, in the file's type, from its tensors on the GPU, and keeps them beside those. A
  warm-up decodes on the CPU where the file's default does, so that a server's first request finds the copy made. A
  model on the CPU keeps no second copy.
- **The request option `decode_on_cpu` chooses where the decoding computes, and its default follows the file's
  decoding weights**: true where every matrix the decoding multiplies by is quantized (Q8_0 or a K-quant), false where
  they are F16 or F32. The model's information gives the default per file, and a request can ask for either. Every
  FastConformer model declares the option, so that a request does not depend on the device the model was loaded on; on a
  model on the CPU both values decode there.
- **The CPU decodes with the load's threads, in a pool that waits between graphs.** The model's information reports the
  threads of a FastConformer model on a GPU, which can compute part of a request on the CPU.

The default follows the weights' type because the CPU computes a quantized step in less time than a round trip to the
GPU on both machines measured, and an F16 step on the i9-9900K in about as long as Vulkan's round trip or longer. The
whole recognition, on an Apple M5 with Metal on 2026-10-13, the median of 3 runs in each of 3 rounds that alternate the
builds, summed over 4 utterances of 6 to 28 s:

| Model, file, decoding | Decoding on Metal | On the CPU | On the CPU from an F32 copy |
|---|---|---|---|
| reazonspeech-nemo-v2, Q8_0, beam search | 1,262 ms | 1,001 ms | 1,641 ms |
| reazonspeech-nemo-v2, F16, beam search | 1,409 ms | 1,312 ms | 1,839 ms |
| reazonspeech-nemo-v2, Q8_0, greedy | 714 ms | 707 ms | 855 ms |
| parakeet-tdt_ctc-0.6b-ja, Q8_0 | 635 ms | 585 ms | 678 ms |
| parakeet-tdt-0.6b-v3, F16 | 583 ms | 555 ms | 678 ms |

On Windows with an RTX 2080 on Vulkan and an i9-9900K, the total of 100 utterances in 3 alternating rounds against
0.8.4, which decodes on the GPU, with the decoding on the CPU: reazonspeech-nemo-v2 in Q8_0 13.93 s against 11.56 s,
parakeet-tdt_ctc-0.6b-ja in Q8_0 4.65 s against 3.97 s, and parakeet-tdt-0.6b-v3 in F16 8.78 s against 10.38 s.

The alternatives were turned down:

- Decoding on the CPU whatever the weights. parakeet-tdt-0.6b-v3 in F16 took 18% longer on the i9-9900K than decoding
  on the RTX 2080, where the M5 gained 5%.
- A copy converted to F32, whose arithmetic would be nearer the GPU's. It gives the text of decoding on Metal on every input measured,
  with no token a frame from NeMo's, but each step reads twice the bytes of F16 and four times those of Q8_0: it was 7
  to 30% slower than decoding on Metal for every model on the M5, and holds 43 to 70 MB.
- A copy of F16 weights converted to Q8_0, which the CPU computes fastest. It changes the precision of the weights the
  file holds, which a user chose with the file.
- The copy loaded with the model. parakeet-tdt-0.6b-v3's F16 file, whose requests decode on the GPU by default, then
  holds 35 MB that no request reads unless it asks; making the copy when a request first asks took 1 to 8 ms on the
  M5 for each of the files the names fetch (2026-10-13).
- The copy read from the file when a request first asks. The file can have been replaced or removed since the model
  loaded, and the request would fail or decode with other weights than the GPU's.
- A second copy of the whole model on the CPU. It doubles the memory of the weights, 0.66 GB more for a Q8_0 file and
  1.26 GB for an F16 one, where the decoding reads 2 to 3% of them.
- The CPU reading the GPU's own buffers. It works only where the GPU's memory is the host's, as Metal's is on Apple
  silicon; a discrete GPU on Vulkan has no buffer the CPU can read in place.
- Decoding on the GPU alone. With quantized weights every step pays a round trip that costs more than the step's
  arithmetic.
- ggml's threads started for each graph. On the M5, reazonspeech-nemo-v2's beam search on 3 of the utterances above
  took 950 ms with 4 threads started for each graph and 856 ms with a pool of 4, where Metal took 1,091 ms.

## Consequences

The decoding on the CPU computes in the CPU's arithmetic, which differs from the GPU's: the CPU multiplies Q8_0 weights
by inputs it rounds to 8 bits and, on ARM, sums products of F16 weights in half precision. Where two choices score
almost the same, the text or a token's frame can differ from the GPU's. On the checks' dumps the texts are NeMo's either
way; with F16 weights decoding on the CPU, one token in 2 of reazonspeech-nemo-v2's 20 decodings falls one frame from
NeMo's, and with the Q8_0 file the joined 311 s input comes out one word different, once nearer NeMo's text and once
further, with each decoding. The beam search on that input then runs to its last step, 7,782 where Metal stops at
4,889.

For the parakeet and ReazonSpeech files measured here, the CPU copy adds 12 to 35 MB of host memory once a request
decodes there. A recognition that decodes on the CPU occupies its threads while it decodes. A caller that wants the CPU free or the
GPU's arithmetic sets `decode_on_cpu` to false; one whose CPU decodes F16 weights faster than its GPU's round trip, as
the M5 does, can set it to true.
