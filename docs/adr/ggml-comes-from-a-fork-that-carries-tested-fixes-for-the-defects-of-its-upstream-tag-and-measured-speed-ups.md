# ggml comes from a fork that carries tested fixes for the defects of its upstream tag and measured speed-ups

## Context

speech.cpp builds ggml from a git submodule. ggml-org/ggml v0.27.0 (2026-10-15) gives speech.cpp's models wrong
results on two supported platforms, in five places, each shown by test-backend-ops cases that fail on the tag on an
Apple M5 or an RTX 2080 (2026-10-17):

- **Metal's tensor API.** On the M5, M6, A19 and A20, Metal runs matrix products through Metal 4's tensor API by
  default, on the matrix units those chips have. ggml's `kernel_mul_mm` for it reads the activations in float and asks
  for relaxed precision, under which it computes less precisely than
  the simdgroup kernel that the earlier chips run: Irodori-TTS v4.1-Small's synthesis lay 2.3 to 48.3 dB from the
  official implementation's against 8.7 to 62.5 dB, reazonspeech-nemo-v2's beam search of a 311 s input changed its
  text, and on speech.cpp's products its result lies 2e-7 to 3e-7 in NMSE from the exact product of the operands
  rounded to half, where the simdgroup kernel's lies 1e-14 to 2e-12 (2026-10-15). Without relaxed precision the units
  multiply its float activations in float, and it is slower than the simdgroup kernel, by speech-bench on the M5
  (2026-10-12): median first audio 0.351 s against 0.251 s for Irodori-TTS v4.1-Small-MF, 1.762 s against 1.239 s for
  v4.1-Small at 16 steps and 0.072 s against 0.039 s for Qwen3-TTS 0.6B, and median recognition 0.150 s against
  0.123 s for Qwen3-ASR 0.6B. Given both operands in half precision, as the simdgroup kernel stages them, the units
  multiply them without relaxed precision as that kernel does, half by half summed in float, faster than it and at
  least as fast as ggml's relaxed kernel. On an Apple M5 with Q8_0 files (2026-10-15), against the simdgroup kernel:
  Irodori-TTS v4.1-Small speaks 11.2 s at RTFx 3.67 against 2.37 (ggml's relaxed kernel 3.61) and v4.1-Small-MF at
  9.76 against 7.72; Qwen3-ASR 0.6B prefills in 18 ms against 41 and encodes in 20 ms against 32;
  parakeet-tdt_ctc-0.6b-ja encodes 25.5 s in 95 ms against 174; Qwen3-TTS 1.7B gives its first audio in 39 ms
  against 56. ggml stores each tile of the kernel clipped by the tile's own extents at a 64-bit offset (llama/28748),
  and the products of 64 and 192 columns, which a store through a slice of the whole output corrupted, pass on v0.27.0.
- **Vulkan's q8_1 activations.** Vulkan quantizes a matrix product's activations to q8_1 for its integer dot products,
  and q8_1 carries the sum of each block of 32 activations in half precision. The products of Q4_0, Q4_1, Q5_0, Q5_1,
  Q4_K, Q5_K, Q2_0 and IQ1_S read that sum, and Qwen3-TTS's activations sum past 65504 in a block, so its Q4_K and Q5_K
  files gave no finite logit on an RTX 2080, where the CPU and Metal spoke. On v0.27.0 on that RTX 2080, products of
  such activations give NaN, inf or a relative error of 0.17 to 64. Where no q8_1 kernel takes a product, v0.27.0
  converts the activations of a quantized weight to half precision on a device with cooperative matrices, the RTX 2080
  (NV_coopmat2) among them, whose range activations whose block sums pass 65504 approach.
- **Vulkan's ROPE reading its positions in a view.** Vulkan binds a buffer at its offset rounded down to the device's
  `minStorageBufferOffsetAlignment`, and a shader adds the rest back. ROPE's shaders do so for their input and output
  but not for the positions or the frequency factors, so a ROPE whose positions are a view that does not start on that
  alignment rotates by another row's positions. Qwen3-TTS's code predictor, which reads each step's positions as a view
  of one input when it runs a frame in one graph, gave logits 1.37 in relative error from the reference on an RTX 2080
  (16 bytes of alignment), against 0.034 a step at a time, and its CER rose from 15.9% to 20.0% (2026-10-13).
- **argmax's equal values.** torch.argmax takes the first of equal values, and so do the greedy decoders that
  speech.cpp checks against it. ggml's argmax takes the last on the CPU and on Metal, and on Vulkan whichever its
  threads' layout gives, so that a decoder that picks its tokens on the device and one that picks them from the logits
  on the host could write different tokens at a tie.
- **Metal's 4-bit scales.** Metal dequantizes Q4_0, Q4_1 and Q4_K dividing the half-precision scale by 16 in half
  precision. For a model's weights, a few hundredths at most, a Q4_K scale is about 1e-4 and its sixteenth falls among
  half precision's subnormal numbers, so every Metal kernel reads the weights of the upper nibbles about 1% off: rows of
  Q4_K weights drawn from [-0.04, 0.04] come back 2.0e-5 in NMSE from the CPU's, and from [-0.004, 0.004] 6.8e-4, on an
  Apple M5 with v0.27.0 (2026-10-17).

ggml reads no setting that avoids the first two through its API, only environment variables: `GGML_METAL_TENSOR_DISABLE`,
which it reads only while it lists its devices, so that a library sets it in its host's process, unsafely while another
thread reads the environment; and `GGML_VK_DISABLE_MMVQ`, which it reads when it first creates each Vulkan device, and
which also moves Q8_0's batched products off their kernels. v0.27.0 reads them as before.

ggml's kernels also leave some of speech.cpp's graphs slower than they need be. On Metal on an Apple M5 (2026-10-13),
Irodori-TTS v4.1-Small-MF spends 80 to 85% of its synthesis in its codec, and 23 to 26% of the codec's time in Snake,
which it writes as five operations (a product, a sine, a square, a product and a sum), each a pass over the
activations. ggml v0.27.0 runs those five as one kernel on Metal and Vulkan only when the channels run along `ne[1]`, while the codecs of Irodori-TTS and Qwen3-TTS keep their activations with the
channels along `ne[0]`, the layout in which their convolutions run as matrix products and their strided convolutions
as reshapes. And ggml's argmax reads a row with one threadgroup of at most 256 threads shared among the rows on Metal,
and with one subgroup, 32 threads on NVIDIA, on Vulkan: an argmax over Qwen3's 151,936 tokens took 59 µs on an Apple M5
and 190 µs on an RTX 2080 (2026-10-15), longer than the head's product that made the logits on the RTX 2080, so that
picking Qwen3-ASR's tokens on the device was slower than reading the logits back.

## Decision

- **The submodule is the branch `speech.cpp` of github.com/nyosegawa/ggml**, pinned by commit: an upstream release tag,
  then one commit of `test-backend-ops` cases that show the defects, then one commit for each fix, then one commit for
  each speed-up, and nothing else.
- **Metal runs matrix products on the matrix units of the chips that have them**, the M5, M6, A19 and A20, through the
  tensor API as ggml does, but with a `kernel_mul_mm` of the fork's: it stages both operands in threadgroup memory in
  half precision, as the simdgroup kernel does, multiplies them without relaxed precision, and stores each tile
  clipped by its own extents at a 64-bit offset, as ggml's kernel does. Its products are the
  simdgroup kernel's up to the order of the float sums: on speech.cpp's shapes in F32, F16, Q8_0, Q4_K, Q5_K and Q6_K,
  the RMS of their difference is at most 1.6e-6 of the output's and its largest value 1.6e-5 of it, and the NMSE from
  the exact product is the simdgroup kernel's to four digits, or lower (2026-10-15). ggml's own tensor
  `kernel_mul_mm` is not in the fork. The chips before the M5 keep the simdgroup kernel, as in ggml:
  `GGML_METAL_TENSOR_ENABLE` turns the tensor API on there and `GGML_METAL_TENSOR_DISABLE` off on any chip.
- **Vulkan keeps the types whose q8_1 dot products read the block sums off q8_1 activations**, through the
  dequantizing kernels on activations in F32, never converted to half precision, which then accumulate in F32 as the
  q8_1 kernels do, whatever precision the product asks for: in half precision, which they take by default, a product
  past 65504 is clipped. Q8_0, Q6_K and the other types keep the q8_1 path.
- **Vulkan's ROPE adds the offsets of its positions and frequency factors in a view**, as it adds those of its input
  and output.
- **Metal and Vulkan fuse Snake with the channels along `ne[0]`** as they fuse it with the channels along `ne[1]`.
- **Metal divides the 4-bit scales by 16 in float**, as it already did for Q5_K.
- **Every backend's argmax takes the first of equal values**, and **Metal and Vulkan read a row of argmax with as many
  threads as a threadgroup takes**, up to 512 on Vulkan.
- **A speed-up is a commit of its own, with `test-backend-ops` cases that show it correct in the same commit, and its
  effect measured on Metal and on the Windows machine's Vulkan GPU, with the numbers and their conditions (the model,
  the type, the device and the date) in its commit message.** A speed-up whose effect on one of speech.cpp's models
  is not measured does not go in.
- **Each commit comes with what shows it, and goes when its upstream has the same:**

  | Commit | Shown by | Dropped when |
  |---|---|---|
  | Vulkan's q8_1 block sums | `test-backend-ops -o MUL_MAT -p large_b`: Q4_0, Q5_0, Q4_K and Q5_K products of activations from 2048 to 4096, and of activations from 512 to 1024 with weights from 0.5 to 1, on Vulkan | the tag passes those cases |
  | Vulkan's ROPE offsets | `test-backend-ops -o ROPE` and `-o ROPE_BACK`: the cases with `v=4`, whose positions and frequency factors are views one element into a longer tensor, on Vulkan | the tag passes those cases |
  | Snake fused with the channels along `ne[0]` (speed-up) | `test-backend-ops -o SNAKE_FUSE`: the cases with `channels_first=1`, on Metal and Vulkan | the tag fuses Snake in that layout |
  | argmax takes the first of equal values | `test-backend-ops -o ARGMAX`: the cases with `ties=1`, whose rows repeat their largest value in every thread, on Metal and Vulkan | the tag's argmax takes the first of equal values on every backend |
  | argmax reads a row with a threadgroup's threads (speed-up) | `test-backend-ops -o ARGMAX`: rows of 151,936 values, one to eight of them, on Metal and Vulkan | the tag reads a row of argmax with more than one subgroup on Vulkan and more than 256 threads among all rows on Metal |
  | Metal's matrix units without relaxed precision (fix and speed-up) | `test-backend-ops -o MUL_MAT` on an M5, M6, A19 or A20: the F16 products of 64 and 192 columns, and the products of F32, F16, BF16, Q8_0, Q4_K, Q5_K and Q6_K weights whose outputs end inside a tile, whose K does not fill a step, with broadcast batches, permuted or strided activations and activations in half precision; and the stage checks of speech.cpp on Metal | the tag's tensor-API `kernel_mul_mm` passes those cases, gives the simdgroup kernel's products up to the order of the float sums, and is no slower |
  | Metal's 4-bit scales in float | `test-backend-ops -o GET_ROWS`: the Q4_0, Q4_1 and Q4_K cases with `range` 0.04 and 0.004, on Metal | the tag passes those cases |

  Between tags, a fix found later is a commit of its own on top, with its cases. Moving ggml to a newer tag starts the
  branch again from that tag: the commit of cases first, on which they run on the
  devices above, then each fix the table still calls for, then each speed-up the tag still lacks.

The alternatives were turned down:

- The environment variables above, set by the library. They change the environment of a host's process, and Vulkan's
  would also move Q8_0 off its kernels.
- Keeping the tensor API off on every chip, so that the M5 runs the simdgroup kernel as the earlier chips do. It is as
  precise, and leaves the matrix units idle, at the speeds above.
- ggml's tensor `kernel_mul_mm` as ggml runs it, under relaxed precision. It computes less precisely, as above.
- ggml's tensor `kernel_mul_mm` without relaxed precision. The units multiply its float activations in float, slower
  than the simdgroup kernel, as above.
- Converting the activations to half precision once per product, into memory beside the output, for the units to read
  from device memory. As precise, and up to 1.9 times as fast on products far larger than speech.cpp's, but slower
  than the simdgroup kernel on the codecs' largest products, with K of 96 (0.6 to 0.8 of its speed, where staging both
  operands gives 1.1 to 1.4 times it, on an M5 on 2026-10-15), and it takes memory beside every product.
- Avoiding the shapes and the activations in speech.cpp, such as changing the codec's windows or keeping Qwen3-TTS
  from K-quants on Vulkan. Every graph can meet the shapes, and every model with large activations the sums.
- Waiting for ggml-org/ggml to fix them. The models give wrong results on supported platforms until then.
- A fork that carries fixes alone, with speed-ups only in speech.cpp's graphs. ggml's custom operations run on the CPU
  alone, and laying out the codecs' activations with the channels along `ne[1]` for upstream's Snake kernel would put a
  transposition of the activations before and after each Snake, or change every convolution of both codecs.
- Waiting for ggml-org/ggml to add a speed-up. The models run more slowly on supported platforms until then.

## Consequences

Metal computes on the M5 as the simdgroup kernel computes but for the order of the float sums, which moves the stage
checks' errors as any change of that order does: a build of the simdgroup kernel that only sums the odd steps of K
apart moves them as far, in both directions. The library does not touch the environment. A ggml update runs the cases
on the M5 and on the Windows machine's Vulkan GPU, from the Windows build of CI with checks, before the submodule
moves. A speed-up changes the results of the graphs it covers on Metal
and Vulkan by rounding alone, and the CPU's not at all.
