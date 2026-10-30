#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

/** The length of the well-formed UTF-8 scalar at `s[i]`, or zero when there is none. */
size_t utf8_scalar_length(const std::string & s, size_t i);

/**
 * The code points of `s`, or nothing when `s` is not well-formed UTF-8 by the Unicode Standard's table 3-7, which
 * refuses overlong forms, surrogates and code points past U+10FFFF as Python's and Rust's decoders do.
 */
std::optional<std::vector<uint32_t>> decode_utf8(const std::string & s);

/** The UTF-8 of `code_points`, each of which is a Unicode scalar value. */
std::string encode_utf8(const std::vector<uint32_t> & code_points);

/**
 * `s` with every ill-formed sequence replaced by U+FFFD, one for each maximal subpart, as Rust's
 * String::from_utf8_lossy() and Python's errors="replace" replace them.
 */
std::string replace_invalid_utf8(const std::string & s);
