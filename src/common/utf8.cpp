#include "utf8.h"

namespace {

/**
 * The length of the UTF-8 sequence at `s[i]`: a whole character's when `whole` comes back true, and otherwise that of
 * the longest start of one, at least one byte, the maximal subpart that becomes one U+FFFD.
 */
size_t sequence(const std::string & s, size_t i, bool & whole) {
    const unsigned char c = (unsigned char) s[i];
    whole = false;
    if (c < 0x80) {
        whole = true;
        return 1;
    }
    size_t len = 0;
    unsigned char lo = 0x80, hi = 0xBF;
    if (c >= 0xC2 && c <= 0xDF) {
        len = 2;
    } else if (c >= 0xE0 && c <= 0xEF) {
        len = 3;
        if (c == 0xE0) lo = 0xA0;
        if (c == 0xED) hi = 0x9F;
    } else if (c >= 0xF0 && c <= 0xF4) {
        len = 4;
        if (c == 0xF0) lo = 0x90;
        if (c == 0xF4) hi = 0x8F;
    } else {
        return 1;
    }
    for (size_t k = 1; k < len; k++) {
        if (i + k >= s.size()) return k;
        const unsigned char cc = (unsigned char) s[i + k];
        if (cc < (k == 1 ? lo : 0x80) || cc > (k == 1 ? hi : 0xBF)) return k;
    }
    whole = true;
    return len;
}

}  // namespace

size_t utf8_scalar_length(const std::string & s, size_t i) {
    if (i >= s.size()) return 0;
    bool whole;
    const size_t len = sequence(s, i, whole);
    return whole ? len : 0;
}

std::optional<std::vector<uint32_t>> decode_utf8(const std::string & s) {
    std::vector<uint32_t> out;
    for (size_t i = 0; i < s.size();) {
        bool whole;
        const size_t len = sequence(s, i, whole);
        if (!whole) return std::nullopt;
        const unsigned char c = (unsigned char) s[i];
        uint32_t cp = len == 1 ? c : len == 2 ? (c & 0x1F) : len == 3 ? (c & 0x0F) : (c & 0x07);
        for (size_t k = 1; k < len; k++) cp = (cp << 6) | ((unsigned char) s[i + k] & 0x3F);
        out.push_back(cp);
        i += len;
    }
    return out;
}

std::string encode_utf8(const std::vector<uint32_t> & code_points) {
    std::string s;
    for (uint32_t cp : code_points) {
        if (cp < 0x80) {
            s += (char) cp;
        } else if (cp < 0x800) {
            s += (char) (0xC0 | (cp >> 6));
            s += (char) (0x80 | (cp & 0x3F));
        } else if (cp < 0x10000) {
            s += (char) (0xE0 | (cp >> 12));
            s += (char) (0x80 | ((cp >> 6) & 0x3F));
            s += (char) (0x80 | (cp & 0x3F));
        } else {
            s += (char) (0xF0 | (cp >> 18));
            s += (char) (0x80 | ((cp >> 12) & 0x3F));
            s += (char) (0x80 | ((cp >> 6) & 0x3F));
            s += (char) (0x80 | (cp & 0x3F));
        }
    }
    return s;
}

std::string replace_invalid_utf8(const std::string & s) {
    std::string out;
    for (size_t i = 0; i < s.size();) {
        bool whole;
        const size_t len = sequence(s, i, whole);
        if (whole) out.append(s, i, len);
        else out += "\xEF\xBF\xBD";
        i += len;
    }
    return out;
}
