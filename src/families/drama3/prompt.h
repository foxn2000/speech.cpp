#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "added-vocabulary.h"
#include "model-file.h"
#include "qwen2-tokenizer.h"

namespace drama3 {

/** A reference of a voice, or a chunk spoken before: its text and its codes, [codebooks, frames] with the frame fastest. */
struct Utterance {
    std::string text;
    std::vector<int32_t> codes;
    int64_t frames = 0;
};

/**
 * Drama 3's prompt as fish-speech's build_drama3_prompt() writes it, from the pieces of the file's template: the system
 * turn with its instruction and the references, then each chunk spoken before as a user turn and an assistant voice
 * turn with its codes, then the user turn of the text and the assistant voice turn that the frames follow. Each piece
 * of text is tokenized on its own, as the official builder tokenizes each: tokenizing them joined changes ids at the
 * seams. A row of text is its token with codes of 0; a row of audio is the first code's semantic id and the codes.
 */
class Prompt {
public:
    explicit Prompt(const ModelFile & m);

    /**
     * The rows of the prompt, 1 + codebooks ids each, row-major. A text that is empty, holds a token only the builder
     * writes or a speaker tag other than <|speaker:0|> to <|speaker:15|> throws an Error naming `text`, a reference's or
     * a turn's naming "voice"; codes outside their codebook throw likewise, and a prompt of more rows than the model
     * takes throws an Error of its range naming "text".
     */
    std::vector<int32_t> rows(const std::string & text, const std::vector<Utterance> & references, const std::vector<Utterance> & history) const;

    /**
     * Refuses `text`, which is `what` ("the text") of the caller's `input`, where it is empty, holds a token only the
     * prompt writes or a speaker tag other than <|speaker:0|> to the model's last, with an Error naming `input`.
     */
    void check(const std::string & text, const std::string & input, const std::string & what) const;

    /** The ids of `text`, its added tokens found first, as the official builder's tokenizer gives them. */
    std::vector<int32_t> tokenize(const std::string & text) const;

    int width() const { return 1 + codebooks_; }

private:
    void append_text(std::vector<int32_t> & rows, const std::string & text) const;
    void append_codes(std::vector<int32_t> & rows, const Utterance & u) const;

    Qwen2Tokenizer bpe_;
    AddedVocabulary added_;
    std::string system_, instruction_, reference_instruction_, reference_open_, reference_text_, reference_speech_, end_, user_, assistant_;
    std::vector<std::string> reserved_;
    int codebooks_ = 0, codes_ = 0, speakers_ = 0;
    int32_t semantic_begin_ = 0;
    int64_t max_rows_ = 0;
};

/**
 * Whether `text` holds anything but Unicode white space (the White_Space property, which leaves out U+001C to U+001F
 * where Python's str.strip() takes them); text that is not UTF-8 is refused naming `input`.
 */
bool has_content(const std::string & text, const std::string & input);

/** `text` with <|speaker:0|> and a space before it, unless a speaker tag leads it after white space. */
std::string with_speaker_tag(const std::string & text);

}  // namespace drama3
