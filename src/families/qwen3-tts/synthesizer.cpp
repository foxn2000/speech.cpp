#include "synthesizer.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <stdexcept>

#include "error.h"
#include "layout.h"

int chunk_frames(int sent) {
    return sent < 2 ? 1 : sent == 2 ? 2 : 4;
}

int text_token_limit(const ModelFile & m) {
    return (int) m.u32("qwen3-tts.talker.max_position_embeddings") - (int) m.u32("qwen3-tts.generation.max_frames") - kPromptRows;
}

Synthesizer::Synthesizer(const std::string & path, ggml_backend_t backend)
    : model_(path, backend, qwen3_tts_layout),
      talker_(model_, backend),
      codec_(model_, backend),
      tokenizer_(model_, kTextTokenizer),
      ids_(model_),
      generation_(model_) {}

namespace {

/** Adds the seconds from its construction to its destruction to `total`, when `total` is set. */
struct Timer {
    double * total;
    std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
    ~Timer() {
        if (total) *total += std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    }
};

}  // namespace

SynthesisOutcome Synthesizer::synthesize(const SynthesisRequest & r, const AudioSink & sink, SynthesisStats * stats) {
    SynthesisStats local;
    SynthesisStats & st = stats ? *stats : local;
    Prompt prompt;
    {
        Timer t{&st.prompt};
        std::vector<int32_t> text_ids = {ids_.im_start, ids_.assistant, ids_.newline};
        const std::vector<int32_t> body = naming("text", [&] { return tokenizer_.encode(r.text); });
        if (body.empty()) throw Error(Fault::InvalidArgument, "the text is empty; give a text to speak", "text");
        if ((int) body.size() > max_text_tokens()) {
            throw Error(Fault::OutOfRange, "the text is " + std::to_string(body.size()) + " tokens long and Qwen3-TTS takes at most " +
                                               std::to_string(max_text_tokens()) + "; split it into shorter texts",
                        "text");
        }
        text_ids.insert(text_ids.end(), body.begin(), body.end());
        text_ids.insert(text_ids.end(), {ids_.im_end, ids_.newline, ids_.im_start, ids_.assistant, ids_.newline});
        std::vector<int32_t> instruction;
        if (!r.instructions.empty()) {
            instruction = naming("instructions", [&] { return instruction_ids(tokenizer_, ids_, r.instructions); });
            // The instruction's rows take positions of the talker that the text could otherwise have.
            if (body.size() + instruction.size() > (size_t) max_text_tokens()) {
                throw Error(Fault::OutOfRange, "the instruction is " + std::to_string(instruction.size()) + " tokens long and the text " +
                                                   std::to_string(body.size()) + ", and Qwen3-TTS takes at most " +
                                                   std::to_string(max_text_tokens()) + " of both together; give a shorter instruction or text",
                            "instructions");
            }
        }
        prompt = build_prompt(talker_, ids_, text_ids, r.speaker, r.language, instruction);
    }

    const int n_groups = talker_.num_code_groups();
    const int vocab = talker_.vocab();
    const std::vector<bool> none;
    std::mt19937_64 rng(r.seed);
    const int max_frames = std::min(r.max_frames, generation_.max_frames);
    const SamplingParams talker_sampling = r.talker.value_or(generation_.talker);
    SamplingParams cp_sampling = r.code_predictor.value_or(generation_.code_predictor);
    static const bool cp_greedy = std::getenv("SPEECH_CP_GREEDY") != nullptr, cp_chain = std::getenv("SPEECH_CP_CHAIN") != nullptr;
    if (cp_greedy) cp_sampling.greedy = true;
    // The code predictor's generate() starts each frame from embeddings alone, so the codes its repetition penalty sees
    // are the frame's that it has made so far.
    std::vector<int32_t> cp_history;
    const auto sample_code = [&](const std::vector<float> & logits) {
        try {
            return sample(logits, cp_sampling, cp_history, none, rng);
        } catch (const Error & e) {
            // The sampler names the talker's options; the code predictor's carry its name before them.
            throw Error(e.fault(), std::string("in the code predictor, ") + e.what(), "code_predictor_" + e.input());
        }
    };

    {
        Timer t{&st.talker};
        // Every frame takes one position of the talker's cache after the prompt's.
        talker_.prefill(prompt.embeds, prompt.n, (int64_t) prompt.n + max_frames);
    }
    codec_.reset();
    std::vector<int32_t> history, frame(n_groups), pending;
    std::vector<float> audio;
    int frames = 0, pending_frames = 0, chunks = 0;
    bool stopped = false, ended = false;

    auto flush = [&]() {
        if (pending_frames == 0) return;
        audio.clear();
        {
            Timer t{&st.codec};
            codec_.decode(pending.data(), pending_frames, audio);
        }
        pending.clear();
        pending_frames = 0;
        chunks++;
        if (!sink(audio.data(), audio.size())) stopped = true;
    };

    while (frames < max_frames && !stopped) {
        frame[0] = sample(talker_.logits(), talker_sampling, history, generation_.banned(vocab, ids_.codec_eos, frames), rng);
        if (frame[0] == ids_.codec_eos) {
            ended = true;
            break;
        }
        history.push_back(frame[0]);
        {
            Timer t{&st.code_predictor};
            cp_history.clear();
            if (cp_chain) {
                const std::vector<int32_t> codes = talker_.cp_greedy_frame(frame[0]);
                std::copy(codes.begin(), codes.end(), frame.begin() + 1);
            } else {
            frame[1] = sample_code(talker_.cp_begin(frame[0]));
            for (int g = 1; g < n_groups - 1; g++) {
                cp_history.push_back(frame[g]);
                frame[g + 1] = sample_code(talker_.cp_next(g, frame[g]));
            }
            }
        }
        pending.insert(pending.end(), frame.begin(), frame.end());
        pending_frames++;
        frames++;
        if (pending_frames == chunk_frames(chunks)) flush();
        if (stopped) break;
        Timer t{&st.talker};
        talker_.step(frame.data(), prompt.frame_extra);
    }
    if (!stopped) flush();
    if (std::getenv("SPEECH_TTS_STATS")) {
        std::fprintf(stderr, "qwen3-tts stats: frames %d prompt %.4f talker %.4f code_predictor %.4f codec %.4f\n", frames,
                     st.prompt, st.talker, st.code_predictor, st.codec);
    }
    return {frames, ended};
}
