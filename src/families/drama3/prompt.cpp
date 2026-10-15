#include "prompt.h"

#include <algorithm>
#include <optional>

#include "error.h"
#include "utf8.h"

namespace drama3 {

namespace {

/** The White_Space property of Unicode, which fish-speech strips texts of. */
bool is_white_space(uint32_t c) {
    return (c >= 0x9 && c <= 0xD) || c == 0x20 || c == 0x85 || c == 0xA0 || c == 0x1680 || (c >= 0x2000 && c <= 0x200A) || c == 0x2028 ||
           c == 0x2029 || c == 0x202F || c == 0x205F || c == 0x3000;
}

std::vector<uint32_t> code_points(const std::string & text, const std::string & input) {
    std::optional<std::vector<uint32_t>> cps = decode_utf8(text);
    if (!cps) throw Error(Fault::InvalidArgument, "the " + input + " is not valid UTF-8; give UTF-8", input);
    return std::move(*cps);
}

/** The bytes of the white space that leads `text`. */
size_t leading_space(const std::string & text) {
    const std::vector<uint32_t> cps = code_points(text, "text");
    size_t i = 0;
    while (i < cps.size() && is_white_space(cps[i])) i++;
    return encode_utf8(std::vector<uint32_t>(cps.begin(), cps.begin() + (std::ptrdiff_t) i)).size();
}

/** The length of the speaker tag <|speaker:N|> at `at`, N one or more ASCII digits, or 0 where none is. */
size_t speaker_tag_at(const std::string & text, size_t at, std::string * digits = nullptr) {
    static const std::string open = "<|speaker:";
    if (text.compare(at, open.size(), open) != 0) return 0;
    size_t i = at + open.size();
    while (i < text.size() && text[i] >= '0' && text[i] <= '9') i++;
    if (i == at + open.size() || text.compare(i, 2, "|>") != 0) return 0;
    if (digits) *digits = text.substr(at + open.size(), i - at - open.size());
    return i + 2 - at;
}

}  // namespace

bool has_content(const std::string & text) {
    const std::vector<uint32_t> cps = code_points(text, "text");
    return std::any_of(cps.begin(), cps.end(), [](uint32_t c) { return !is_white_space(c); });
}

std::string with_speaker_tag(const std::string & text) {
    return speaker_tag_at(text, leading_space(text)) ? text : "<|speaker:0|> " + text;
}

Prompt::Prompt(const ModelFile & m) : bpe_(m, "drama3.tokenizer", [&] {
                                          Qwen2Pattern pattern = Qwen2Pattern::Qwen2;
                                          qwen2_pattern(m.str("drama3.tokenizer.pre_tokenizer"), &pattern);
                                          return pattern;
                                      }()),
                                      added_(m, "drama3.tokenizer") {
    const std::string p = "drama3.prompt.";
    system_ = m.str(p + "system");
    instruction_ = m.str(p + "instruction");
    reference_instruction_ = m.str(p + "reference_instruction");
    reference_open_ = m.str(p + "reference_open");
    reference_text_ = m.str(p + "reference_text");
    reference_speech_ = m.str(p + "reference_speech");
    end_ = m.str(p + "end");
    user_ = m.str(p + "user");
    assistant_ = m.str(p + "assistant");
    reserved_ = m.str_array(p + "reserved");
    codebooks_ = m.count("drama3.audio_decoder.num_codebooks");
    codes_ = m.size("drama3.audio_decoder.vocab_size");
    speakers_ = m.size("drama3.generation.speakers");
    semantic_begin_ = (int32_t) m.u32("drama3.semantic_start_token_id");
    max_rows_ = m.size("drama3.generation.max_prompt_rows");
}

std::vector<int32_t> Prompt::tokenize(const std::string & text) const {
    return added_.encode(text, bpe_);
}

void Prompt::append_text(std::vector<int32_t> & rows, const std::string & text) const {
    for (int32_t id : tokenize(text)) {
        rows.push_back(id);
        rows.insert(rows.end(), (size_t) codebooks_, 0);
    }
}

void Prompt::append_codes(std::vector<int32_t> & rows, const Utterance & u) const {
    for (int64_t f = 0; f < u.frames; f++) {
        rows.push_back(u.codes[(size_t) f] + semantic_begin_);
        for (int b = 0; b < codebooks_; b++) rows.push_back(u.codes[(size_t) (b * u.frames + f)]);
    }
}

void Prompt::check(const std::string & t, const std::string & input, const std::string & what) const {
    if (!has_content(t)) throw Error(Fault::InvalidArgument, what + " is empty; give it words to speak", input);
    for (const std::string & token : reserved_) {
        if (t.find(token) != std::string::npos) {
            throw Error(Fault::InvalidArgument, what + " holds " + token + ", which only the prompt writes; take it out", input);
        }
    }
    for (size_t at = t.find("<|speaker"); at != std::string::npos; at = t.find("<|speaker", at + 1)) {
        std::string digits;
        const bool canonical = speaker_tag_at(t, at, &digits) && digits.size() <= 2 && (digits.size() == 1 || digits[0] != '0') &&
                               std::stoi(digits) < speakers_;
        if (!canonical) {
            throw Error(Fault::InvalidArgument,
                        what + " holds a speaker tag other than <|speaker:0|> to <|speaker:" + std::to_string(speakers_ - 1) +
                            "|>; write each speaker as one of them",
                        input);
        }
    }
}

std::vector<int32_t> Prompt::rows(const std::string & text, const std::vector<Utterance> & references, const std::vector<Utterance> & history) const {
    const auto check_codes = [&](const Utterance & u, const std::string & what) {
        const bool fits = u.frames >= 1 && (int64_t) u.codes.size() == u.frames * codebooks_ &&
                          std::all_of(u.codes.begin(), u.codes.end(), [&](int32_t c) { return c >= 0 && c < codes_; });
        if (!fits) throw Error(Fault::InvalidArgument, what + " holds codes that are not frames of " + std::to_string(codebooks_) + " codes of the codec", "voice");
    };
    check(text, "text", "the text");
    for (size_t i = 0; i < references.size(); i++) {
        check(references[i].text, "voice", "the transcript of reference " + std::to_string(i + 1));
        check_codes(references[i], "reference " + std::to_string(i + 1));
    }
    for (size_t i = 0; i < history.size(); i++) {
        check(history[i].text, "text", "chunk " + std::to_string(i + 1) + " spoken before");
        check_codes(history[i], "chunk " + std::to_string(i + 1) + " spoken before");
    }

    std::vector<int32_t> rows;
    append_text(rows, system_);
    append_text(rows, references.empty() ? instruction_ : reference_instruction_);
    for (size_t i = 0; i < references.size(); i++) {
        // The speaker tag follows "Text: " directly, and the white space that led the transcript moves behind it.
        const std::string tagged = with_speaker_tag(references[i].text);
        const size_t lead = leading_space(tagged);
        const size_t tag = speaker_tag_at(tagged, lead);
        const std::string transcript = tagged.substr(lead, tag) + tagged.substr(0, lead) + tagged.substr(lead + tag);
        append_text(rows, reference_open_ + std::to_string(i + 1) + reference_text_ + transcript + reference_speech_);
        append_codes(rows, references[i]);
    }
    append_text(rows, end_);
    for (const Utterance & turn : history) {
        append_text(rows, user_);
        append_text(rows, with_speaker_tag(turn.text));
        append_text(rows, end_);
        append_text(rows, assistant_);
        append_codes(rows, turn);
        append_text(rows, end_);
    }
    append_text(rows, user_);
    append_text(rows, with_speaker_tag(text));
    append_text(rows, end_);
    append_text(rows, assistant_);
    const int64_t n = (int64_t) rows.size() / width();
    if (n > max_rows_) {
        throw Error(Fault::OutOfRange,
                    "the prompt takes " + std::to_string(n) + " rows, more than the model's " + std::to_string(max_rows_) +
                        "; give a shorter text or shorter references",
                    "text");
    }
    return rows;
}

}  // namespace drama3
