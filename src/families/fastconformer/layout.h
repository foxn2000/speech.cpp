#pragma once

#include "model-file.h"

namespace fastconformer {

/**
 * Layout 3 holds the FastConformer frontend, encoder, LSTM or KDA prediction network, joint, decoding settings and
 * SentencePiece vocabulary. Layouts 1 and 2 read with an LSTM and without byte fallback; layout 1's RNN-T also
 * receives reazonspeech-nemo-v2's greedy limit of 10 symbols per frame.
 */
extern const Layout layout;

}  // namespace fastconformer
