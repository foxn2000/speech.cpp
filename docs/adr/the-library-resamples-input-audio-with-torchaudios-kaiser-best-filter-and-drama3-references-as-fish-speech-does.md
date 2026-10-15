# The library resamples input audio with torchaudio's kaiser_best filter, and Drama 3's references as fish-speech does

## Context

Audio comes in at whatever rate it was recorded: OpenAI's clients send files of 44.1 or 48 kHz, a microphone gives its
own rate, and a reference recording for Irodori-TTS or Drama 3 may be at any rate. The recognizers take 16 kHz,
Irodori-TTS's codec 48 kHz and Drama 3's 44.1 kHz. The official implementations resample, each its own way: Irodori-TTS's
runtime and fish-speech, which runs Drama 3, with `torchaudio.functional.resample()` at its defaults, NeMo's
`transcribe()` with librosa's soxr. Measured with torchaudio 2.10.0 from 48 to 16 kHz on 2026-10-06, torchaudio's
defaults lose 2.4 dB at 0.9 of the lower Nyquist frequency and fold a tone at 1.1 of it back into the band at −14 dB.

## Decision

- **The library resamples what it is given**, a recognition's or a detection's audio and a voice's references, to the
  model's rate, and never what it makes; audio given a piece at a time gets the samples of the whole
  ([the record of detections](a-detection-of-audio-given-a-piece-at-a-time-gives-speech-detects-regions-each-once-it-is-certain.md)).
  Audio at the model's rate passes unchanged, so the checks against the dumps are unaffected.
- **The method is torchaudio's rational polyphase windowed sinc** with the parameters torchaudio documents as librosa's
  `kaiser_best` (a Kaiser window of beta 14.769656459379492, 64 zero crossings, a cutoff at 0.9475937167399596 of the
  lower Nyquist frequency), in double precision except where torchaudio rounds to float32: beta, the window's peak
  i0(beta) and the output length. Measured as above, it passes the band to 0.9 of the lower Nyquist frequency within
  0.022 dB and keeps everything from 1.05 of it at −146 dB or below, and anyone with torchaudio can reproduce its
  output, which `resample-check` compares with.
- **Drama 3's references are resampled with torchaudio's defaults** (a Hann window, 6 zero crossings, a cutoff at 0.99
  of the lower Nyquist frequency), as fish-speech resamples them. A Drama 3 voice holds the codec's codes, which a small
  difference in the samples changes one by one: on a reference of 12.8 s at 48 kHz on 2026-10-15, the defaults read it
  as fish-speech does at 129.2 dB SNR and give every code of the official encoder on the CPU, where kaiser_best reads
  it at 66.5 dB and gives 73 to 98% of each codebook's codes. A reference goes up to 44.1 kHz or down from a rate near
  it, where what the defaults fold back lies above 21.8 kHz, outside hearing.

The alternatives were turned down:

- Refusing audio at any rate but the model's. Every caller and every OpenAI client would resample with a tool of its
  own.
- torchaudio's defaults for all audio, which would give Irodori-TTS's references the official runtime's latent exactly,
  at the aliasing measured above, which recognition would hear as well. Irodori-TTS's references keep kaiser_best: its
  voice is a continuous latent, which another filter moves slightly, while a code either is the official's or is not.
- kaiser_best for Drama 3's references too. A quarter of some codebook's codes would differ from the official encoder's,
  and the codec's check could not compare the reading of a recording with fish-speech's.
- A port of soxr, which NeMo's files go through: more code to port and check for a quality the Kaiser filter reaches.

## Consequences

Audio at another rate gives a text or a latent slightly different from the official implementation's, whose resampler is
not this one; a Drama 3 voice made on the CPU holds the official codes. Two rates whose ratio in lowest terms has a term
above 4096 are refused, which no two common rates of recorded audio have.
