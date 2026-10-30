#include "detokenizer.h"

#include <algorithm>
#include <stdexcept>

#include "error.h"
#include "utf8.h"

namespace fastconformer {

namespace {

const std::string kSpaceSymbol = "\xe2\x96\x81";

std::string replace_all(std::string s, const std::string & from, const std::string & to) {
    for (size_t at = s.find(from); at != std::string::npos; at = s.find(from, at + to.size())) s.replace(at, from.size(), to);
    return s;
}

/** Recognizes Python's Unicode \s, which NeMo strips before punctuation and BYTE pieces can supply. */
bool is_whitespace(uint32_t cp) {
    return (cp >= 0x09 && cp <= 0x0d) || (cp >= 0x1c && cp <= 0x20) || cp == 0x85 || cp == 0xa0 || cp == 0x1680 ||
           (cp >= 0x2000 && cp <= 0x200a) || cp == 0x2028 || cp == 0x2029 || cp == 0x202f || cp == 0x205f || cp == 0x3000;
}

size_t whitespace_bytes(const std::string & s, size_t at) {
    const size_t n = utf8_scalar_length(s, at);
    return n && is_whitespace(decode_utf8(s.substr(at, n))->front()) ? n : 0;
}

/** Decodes a SentencePiece BYTE piece, or returns -1 for an ordinary piece. */
int byte_piece(const std::string & piece) {
    if (piece.size() != 6 || piece.compare(0, 3, "<0x") != 0 || piece[5] != '>') return -1;
    const std::string hex = "0123456789ABCDEF";
    const size_t hi = hex.find(piece[3]), lo = hex.find(piece[4]);
    return hi == std::string::npos || lo == std::string::npos ? -1 : (int) (hi * 16 + lo);
}

}  // namespace

Detokenizer::Detokenizer(const ModelFile & m)
    : pieces_(m.str_array("fastconformer.tokenizer.tokens")),
      punctuation_(m.str_array("fastconformer.tokenizer.punctuation")),
      unknown_id_((int) m.u32("fastconformer.tokenizer.unknown_id")),
      unknown_surface_(m.str("fastconformer.tokenizer.unknown_surface")),
      strip_leading_space_(m.boolean("fastconformer.tokenizer.strip_leading_space")),
      remove_extra_whitespaces_(m.boolean("fastconformer.tokenizer.remove_extra_whitespaces")),
      byte_fallback_(m.boolean("fastconformer.tokenizer.byte_fallback")) {
    bytes_.assign(pieces_.size(), -1);
    if (byte_fallback_) {
        std::vector<int> counts(256, 0);
        for (size_t i = 0; i < pieces_.size(); i++) {
            bytes_[i] = (int) i == unknown_id_ ? -1 : byte_piece(pieces_[i]);
            if (bytes_[i] >= 0) counts[(size_t) bytes_[i]]++;
        }
        if (!std::all_of(counts.begin(), counts.end(), [](int count) { return count == 1; })) {
            throw Error(Fault::File, m.path() + ": byte fallback requires each of the 256 BYTE pieces exactly once; " + m.remedy());
        }
    }
}

std::string Detokenizer::piece_text(int32_t id, bool strip_space) const {
    if (id < 0 || (size_t) id >= pieces_.size()) throw std::runtime_error("token id " + std::to_string(id) + " is not in the vocabulary");
    if (id == unknown_id_) return unknown_surface_;
    if (bytes_[(size_t) id] >= 0) return replace_invalid_utf8(std::string(1, (char) bytes_[(size_t) id]));
    std::string piece = pieces_[(size_t) id];
    if (strip_leading_space_ && strip_space && piece.compare(0, kSpaceSymbol.size(), kSpaceSymbol) == 0) piece.erase(0, kSpaceSymbol.size());
    return replace_all(piece, kSpaceSymbol, " ");
}

std::vector<std::string> Detokenizer::token_texts(const std::vector<int32_t> & ids) const {
    std::vector<std::string> pieces;
    std::string text;
    pieces.resize(ids.size());
    for (size_t i = 0; i < ids.size();) {
        const int32_t id = ids[i];
        if (id < 0 || (size_t) id >= pieces_.size()) throw std::runtime_error("token id " + std::to_string(id) + " is not in the vocabulary");
        if (bytes_[(size_t) id] < 0) {
            pieces[i] = piece_text(id, remove_extra_whitespaces_ ? text.empty() : i == 0);
            text += pieces[i++];
            continue;
        }
        const size_t first = i;
        std::string run;
        while (i < ids.size() && ids[i] >= 0 && (size_t) ids[i] < bytes_.size() &&
               bytes_[(size_t) ids[i]] >= 0) {
            run += (char) bytes_[(size_t) ids[i++]];
        }
        for (size_t at = 0; at < run.size();) {
            const size_t n = utf8_scalar_length(run, at);
            // SentencePiece puts a scalar on its final byte token and replaces each invalid byte separately.
            const std::string scalar = n ? run.substr(at, n) : "\xef\xbf\xbd";
            pieces[first + at + (n ? n : 1) - 1] = scalar;
            text += scalar;
            at += n ? n : 1;
        }
    }
    std::vector<std::string> out(pieces.size());
    size_t i = 0;
    for (size_t k = 0; k < pieces.size(); k++) {
        for (size_t at = 0; at < pieces[k].size();) {
            const size_t space = byte_fallback_ ? whitespace_bytes(text, i) : text[i] == ' ' ? 1 : 0;
            bool before_mark = false;
            if (space) {
                for (const std::string & p : punctuation_) before_mark = before_mark || text.compare(i + space, p.size(), p) == 0;
            }
            const size_t n = space ? space : 1;
            if (!before_mark) out[k].append(pieces[k], at, n);
            i += n;
            at += n;
        }
    }
    return out;
}

std::string Detokenizer::text(const std::vector<int32_t> & ids) const {
    std::string text;
    for (const std::string & t : token_texts(ids)) text += t;
    return text;
}

bool Detokenizer::punctuation(int32_t id) const {
    const std::string alone = piece_text(id, true);
    return std::find(punctuation_.begin(), punctuation_.end(), alone) != punctuation_.end();
}

std::vector<bool> Detokenizer::word_starts(const std::vector<int32_t> & ids) const {
    std::vector<bool> starts;
    for (int32_t id : ids) {
        const std::string alone = piece_text(id, true);
        starts.push_back(bytes_[(size_t) id] < 0 && alone != pieces_[(size_t) id] && !punctuation(id));
    }
    return starts;
}

}  // namespace fastconformer
