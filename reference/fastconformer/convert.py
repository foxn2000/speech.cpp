"""Converts a pinned NeMo FastConformer checkpoint with an RNN-T or TDT decoder to the one GGUF file the C++ port reads.

usage: uv run python convert.py <model> <out dir>

Writes a file named under GGUF's naming convention, including Haas-0.6B-ja-v1.0-F32.gguf, in layout 3: the frontend's
window and mel filterbank, the subsampling, the conformer layers, the LSTM or Kimi Delta Attention (KDA) prediction
network and the joint, with the SentencePiece pieces and byte-fallback setting the token ids use, the settings of the
decoding transcribe() runs (greedy TDT's durations and limit, or the beam and length of RNN-T's alignment-length
synchronous beam search) and, for RNN-T, the limit of the greedy decoding NeMo also runs on it, every other constant the
C++ reads and the model's identity in the GGUF specification's general keys. A hybrid checkpoint's CTC head is left out,
since NeMo decodes with its transducer.

The tokenizer's normalizer_spec.remove_extra_whitespaces is stored separately from strip_leading_space, which is
add_dummy_prefix or remove_extra_whitespaces. When prefix stripping is enabled, remove_extra_whitespaces controls
whether a leading space marker is stripped while the text is empty or only from the first piece, including when
byte fallback supplies the first pieces.

Tensor shapes follow ggml, whose ne[0] is the last numpy axis: a Linear weight [out, in] is stored as is
(ne = [in, out]). Every tensor is written in float32; `speech quantize` makes the other weight types of the file, each
tensor in the type src/families/fastconformer/layout.cpp gives it. The batch norm of each convolution
module is folded into its depthwise convolution, which it follows in evaluation, and each LSTM layer's two biases are
summed. A checkpoint whose conformer layers have no biases (ConformerEncoder's use_bias false) is written without them,
and fastconformer.encoder.use_bias says so.
"""

import argparse
import os
import re

import numpy as np
from gguf import GGUFValueType, GGUFWriter, LlamaFileType, naming_convention, size_label
from nemo.collections.asr.models import EncDecHybridRNNTCTCBPEModel, EncDecRNNTBPEModel
from nemo.collections.asr.parts.preprocessing import features
from nemo.collections.asr.parts.submodules import multi_head_attention
from sentencepiece import sentencepiece_model_pb2

from pins import MODELS, restore

ARCH = "fastconformer"
# Each layout this converter has written, with the first release of speech.cpp whose reader takes it; it writes the
# last.
RELEASES = {1: "0.7.0", 2: "0.8.0", 3: "0.9.3"}
LAYOUT = max(RELEASES)
# The ISO 639 two-letter codes of the languages each checkpoint transcribes, from its model card, which requests give as
# BCP 47 tags. None takes a language: parakeet-tdt-0.6b-v3 finds the language of the audio itself.
LANGUAGES = {
    "Haas-ja-v1.0": ["ja"],
    "parakeet-tdt_ctc-0.6b-ja": ["ja"],
    "reazonspeech-nemo-v2": ["ja"],
    "parakeet-tdt-0.6b-v3": ["en", "es", "fr", "de", "bg", "hr", "cs", "da", "nl", "et", "fi", "el", "hu", "it", "lv",
                             "lt", "mt", "pl", "pt", "ro", "sk", "sl", "sv", "ru", "uk"],
}
# The SPDX identifier of each checkpoint's license, from its model card.
LICENSES = {
    "Haas-ja-v1.0": "Apache-2.0",
    "parakeet-tdt_ctc-0.6b-ja": "CC-BY-4.0",
    "parakeet-tdt-0.6b-v3": "CC-BY-4.0",
    "reazonspeech-nemo-v2": "Apache-2.0",
}
# The parts of each model's name under GGUF's naming convention (ggml's docs/gguf.md): its line, its size where the name
# gives one, what it was trained toward (ja, the language of parakeet-tdt_ctc-0.6b-ja) and its version. Where the name
# gives no size, the size label is counted from the parameters of the file's tensors.
NAMES = {
    "Haas-ja-v1.0": {"basename": "Haas", "size_label": "0.6B", "finetune": "ja", "version": "v1.0"},
    "parakeet-tdt_ctc-0.6b-ja": {"basename": "parakeet-tdt_ctc", "size_label": "0.6B", "finetune": "ja", "version": None},
    "parakeet-tdt-0.6b-v3": {"basename": "parakeet-tdt", "size_label": "0.6B", "finetune": None, "version": "v3"},
    "reazonspeech-nemo-v2": {"basename": "reazonspeech-nemo", "size_label": None, "finetune": None, "version": "v2"},
}
# speech.cpp's addition to NeMo's segments: the marks that end a segment wherever they stand, for a model whose
# languages are written without spaces. NeMo ends a segment at one of its separators only where a word ends, which
# text without spaces between its words never reaches.
UNSPACED_LANGUAGES = {"ja", "zh"}
UNSPACED_BREAKS = ["。", "？", "！", "?", "!"]


class Writer(GGUFWriter):
    """A GGUFWriter that writes an empty array of a given element type, which GGUFWriter refuses."""

    def _pack_val(self, val, vtype, add_vtype, sub_type=None):
        if vtype == GGUFValueType.ARRAY and sub_type is not None and len(val) == 0:
            return (self._pack("I", vtype) if add_vtype else b"") + self._pack("I", sub_type) + self._pack("Q", 0)
        return super()._pack_val(val, vtype, add_vtype, sub_type)


parser = argparse.ArgumentParser()
parser.add_argument("model", choices=sorted(MODELS))
parser.add_argument("out_dir")
args = parser.parse_args()
os.makedirs(args.out_dir, exist_ok=True)

pin = MODELS[args.model]
model = restore(pin)
cfg = model.cfg
enc = cfg.encoder
featurizer = model.preprocessor.featurizer
encoder = model.encoder
assert enc.subsampling == "dw_striding" and not enc.get("causal_downsampling", False)
assert enc.conv_norm_type == "batch_norm" and encoder.conv_context_size == [(enc.conv_kernel_size - 1) // 2] * 2
# Attention over the whole utterance, or Longformer's local attention with as many frames on either side and one or
# more global tokens from the first frame on, whose queries, keys and values are the layer's own.
attention = encoder.self_attention_model
assert encoder.att_context_style == "regular" and len(encoder.att_context_size_all) == 1
context = list(encoder.att_context_size)
if attention == "rel_pos":
    assert context == [-1, -1]
else:
    assert attention == "rel_pos_local_attn" and context[0] == context[1] > 0, (attention, context)
    assert encoder.global_tokens == 1 or (encoder.global_tokens > 1 and encoder.global_tokens_spacing == 1)
    assert not encoder.global_attn_separate
assert featurizer.exact_pad is False and featurizer.frame_splicing == 1 and featurizer.mag_power == 2.0
assert featurizer.normalize == "per_feature" and featurizer.log and featurizer.log_zero_guard_type == "add"
assert featurizer.pad_value == 0
# The decoding transcribe() runs by default, which the C++ ports for each kind of transducer: TDT's greedy label
# looping (greedy_batch, GreedyBatchedTDTInfer), or RNN-T's alignment-length synchronous beam search (alsd,
# BeamRNNTInfer). Both feed the blank first as the prediction network's padding (its embedding zero, so the blank is
# the start of the sequence) and use a ReLU joint. The joint's log-softmax, which NeMo applies on the CPU alone, is
# left out; the beam search takes its own log-softmax over the tokens and the blank, which the C++ computes.
# A model with a prompt (EncDecRNNTBPEModelWithPrompt and its hybrid) is told its language through an input the C++
# does not have; a hybrid's CTC head is left aside while cur_decoder is "rnnt".
dec, joint, decoding = model.decoder, model.joint, model.decoding
assert type(model) in (EncDecRNNTBPEModel, EncDecHybridRNNTCTCBPEModel)
assert type(model) is EncDecRNNTBPEModel or model.cur_decoder == "rnnt"
decoder = {"GreedyBatchedTDTInfer": "tdt", "BeamRNNTInfer": "rnnt"}[type(decoding.decoding).__name__]
if decoder == "tdt":
    assert decoding.cfg.strategy == "greedy_batch" and not decoding.cfg.get("big_blank_durations")
    assert type(decoding.decoding.decoding_computer).__name__ == "GreedyBatchedTDTLabelLoopingComputer"
    assert not decoding.decoding.decoding_computer.has_fusion_models()
    durations = [int(d) for d in decoding.cfg.durations]
else:
    beam = decoding.decoding
    # A beam of 1 runs greedy_search() instead. The blank is the last class (index_incr 0), and a float target
    # length is a multiple of the frames.
    assert decoding.cfg.strategy == "alsd" and beam.search_algorithm == beam.align_length_sync_decoding
    assert beam.beam_size > 1 and beam.return_best_hypothesis and beam.softmax_temperature == 1.0
    assert beam.language_model is None and beam.ngram_lm is None and not beam.hat_subtract_ilm
    assert beam.blank == beam.vocab_size and isinstance(beam.alsd_max_target_length, float)
    # The greedy decoding a caller may choose instead, GreedyBatchedRNNTInfer or GreedyRNNTInfer, takes the most tokens
    # on one frame as RNNTDecoding reads it from the configuration's greedy settings; without a limit it would emit
    # tokens on one frame for as long as the joint gives them.
    greedy_max_symbols = decoding.cfg.greedy.get("max_symbols", None) or decoding.cfg.greedy.get("max_symbols_per_step", None)
    assert isinstance(greedy_max_symbols, int) and greedy_max_symbols > 0, greedy_max_symbols
    durations = []
assert dec.blank_as_pad and dec.blank_idx == decoding.blank_id and not dec.random_state_sampling
prednet = dec.prediction.dec_rnn
if type(prednet).__name__ == "KDAPredictionNetwork":
    prediction = "kda"
    assert cfg.decoder.prednet.kda_final_norm and prednet.final_norm is not None
    assert cfg.decoder.get("normalization_mode") is None
    assert len(prednet.layers) == len(prednet.norms) == dec.pred_rnn_layers
    for layer, norm in zip(prednet.layers, prednet.norms):
        assert layer.num_v_heads == layer.num_heads == cfg.decoder.prednet.kda_num_heads
        assert layer.head_v_dim == layer.head_k_dim == cfg.decoder.prednet.kda_head_dim
        assert layer.expand_v == cfg.decoder.prednet.kda_expand_v == 1.0
        assert layer.conv_size == cfg.decoder.prednet.kda_conv_size
        assert layer.use_short_conv and not layer.conv_bias
        assert not layer.allow_neg_eigval and not layer.safe_gate and layer.lower_bound is None
        assert layer.o_norm.activation == "sigmoid"
        assert norm.eps == prednet.final_norm.eps and layer.o_norm.eps == prednet.layers[0].o_norm.eps
        assert all(conv.bias is None for conv in (layer.q_conv1d, layer.k_conv1d, layer.v_conv1d))
        assert layer.key_dim == layer.value_dim == layer.num_heads * layer.head_k_dim
        assert layer.f_proj[0].out_features == layer.g_proj[0].out_features == layer.head_v_dim
else:
    prediction = "lstm"
    # LSTMDropout is the plain LSTM rnn() makes without a normalization; its dropout is off in evaluation.
    assert type(prednet).__name__ == "LSTMDropout" and prednet.lstm.proj_size == 0
    assert prednet.lstm.bias and not prednet.lstm.bidirectional
assert not dec.is_adapter_available() and not joint.is_adapter_available()
assert float(dec.prediction.embed.weight[dec.blank_idx].abs().max()) == 0
assert joint.activation == "relu" and joint.temperature == 1.0
assert joint.num_extra_outputs == len(durations) == len(decoding.durations or [])
assert joint.joint_net[-1].out_features == decoding.blank_id + 1 + len(durations)

w = Writer(None, ARCH)


def add_array(key, values, element):
    """An array with its element type given, not inferred from its first value."""
    w.add_key_value(key, list(values), GGUFValueType.ARRAY, sub_type=element)


w.add_uint32("speech.layout", LAYOUT)
w.add_string("speech.requires", RELEASES[LAYOUT])
w.add_string("speech.task", "recognition")
w.add_uint32("speech.sample_rate", int(featurizer.sample_rate))
w.add_string("speech.language_use", "checked")

# The frontend: FilterbankFeatures' parameters, and the normalization guard, CONSTANT in features.py.
w.add_uint32("fastconformer.frontend.n_fft", int(featurizer.n_fft))
w.add_uint32("fastconformer.frontend.hop_length", int(featurizer.hop_length))
w.add_uint32("fastconformer.frontend.n_mels", int(featurizer.nfilt))
w.add_float32("fastconformer.frontend.preemphasis", float(featurizer.preemph))
w.add_float32("fastconformer.frontend.log_guard", float(featurizer.log_zero_guard_value))
w.add_float32("fastconformer.frontend.std_guard", float(features.CONSTANT))

# The encoder. pos_base is INF_VAL of multi_head_attention.py, the base of RelPositionalEncoding's
# wavelengths; ff_factor is ConformerLayer's fc_factor, the weight of each half-step feed-forward.
w.add_uint32("fastconformer.encoder.d_model", int(enc.d_model))
w.add_uint32("fastconformer.encoder.num_layers", int(enc.n_layers))
w.add_uint32("fastconformer.encoder.num_heads", int(enc.n_heads))
w.add_uint32("fastconformer.encoder.conv_kernel", int(enc.conv_kernel_size))
w.add_uint32("fastconformer.encoder.subsampling_factor", int(enc.subsampling_factor))
w.add_float32("fastconformer.encoder.norm_eps", float(model.encoder.layers[0].norm_out.eps))
w.add_float32("fastconformer.encoder.pos_base", float(multi_head_attention.INF_VAL))
w.add_float32("fastconformer.encoder.xscale", float(model.encoder.xscale or 1.0))
w.add_float32("fastconformer.encoder.ff_factor", float(model.encoder.layers[0].fc_factor))
# Whether the linear layers of the feed-forward modules and the attention and the pointwise convolutions have biases;
# the depthwise convolution has one either way once the batch norm is folded into it.
use_bias = bool(model.encoder.layers[0].feed_forward1.use_bias)
assert all(bool(m.use_bias) == use_bias for layer in model.encoder.layers
           for m in (layer.feed_forward1, layer.feed_forward2, layer.self_attn, layer.conv))
w.add_bool("fastconformer.encoder.use_bias", use_bias)
w.add_string("fastconformer.encoder.attention", attention)
if attention == "rel_pos_local_attn":
    w.add_uint32("fastconformer.encoder.attention_context", int(context[0]))
    w.add_uint32("fastconformer.encoder.global_tokens", int(encoder.global_tokens))

# The joint's classes are the tokenizer's ids, a blank after them, then one per duration.
tokenizer = model.tokenizer
assert not tokenizer.legacy
proto = sentencepiece_model_pb2.ModelProto()
proto.ParseFromString(tokenizer.tokenizer.serialized_model_proto())
# SentencePiece's byte fallback represents each byte exactly once, as <0x00> through <0xFF>.
byte_fallback = bool(proto.trainer_spec.byte_fallback)
allowed = (proto.SentencePiece.NORMAL, proto.SentencePiece.USER_DEFINED, proto.SentencePiece.UNKNOWN)
byte_pieces = []
for piece in proto.pieces:
    is_byte = piece.type == proto.SentencePiece.BYTE
    assert bool(re.fullmatch(r"<0x[0-9A-Fa-f]{2}>", piece.piece)) == is_byte
    if is_byte:
        assert byte_fallback
        byte_pieces.append(piece.piece)
    else:
        assert piece.type in allowed
assert sorted(byte_pieces) == ([f"<0x{i:02X}>" for i in range(256)] if byte_fallback else [])
assert not proto.trainer_spec.treat_whitespace_as_suffix
blank = int(decoding.blank_id)
assert blank == len(proto.pieces) == tokenizer.vocab_size
w.add_string("fastconformer.decoder.kind", decoder)
w.add_string("fastconformer.decoder.prediction", prediction)
if prediction == "kda":
    from kda import QK_NORM_EPS

    # Haas's prednet configuration supplies the head count; its block and final LayerNorm share epsilon.
    w.add_uint32("fastconformer.decoder.kda.num_heads", int(cfg.decoder.prednet.kda_num_heads))
    w.add_float32("fastconformer.decoder.kda.norm_eps", float(prednet.final_norm.eps))
    # FLA's output norm and q/k normalization use separate epsilons.
    w.add_float32("fastconformer.decoder.kda.gate_norm_eps", float(prednet.layers[0].o_norm.eps))
    w.add_float32("fastconformer.decoder.kda.qk_norm_eps", QK_NORM_EPS)
w.add_uint32("fastconformer.decoder.blank_id", blank)
w.add_uint32("fastconformer.decoder.prediction_layers", int(dec.pred_rnn_layers))
if decoder == "tdt":
    add_array("fastconformer.decoder.tdt.durations", durations, GGUFValueType.INT32)
    # The most tokens emitted on one frame before the decoding moves to the next.
    w.add_uint32("fastconformer.decoder.tdt.max_symbols", int(decoding.decoding.max_symbols))
else:
    w.add_uint32("fastconformer.decoder.rnnt.beam_size", int(beam.beam_size))
    # Whether the best of the finished hypotheses is the one with the highest score per label, the blank it starts
    # with counted.
    w.add_bool("fastconformer.decoder.rnnt.score_norm", bool(beam.score_norm))
    # The most labels a hypothesis takes, as a multiple of the encoder's frames.
    w.add_float32("fastconformer.decoder.rnnt.max_target_ratio", float(beam.alsd_max_target_length))
    # The most tokens greedy decoding emits on one frame before it moves to the next.
    w.add_uint32("fastconformer.decoder.rnnt.max_symbols", greedy_max_symbols)
# The marks that end a segment where a word ends: the checkpoint's decoding segment_seperators, or NeMo's default
# (".", "?", "!") when it sets none, as the decoding object takes them.
add_array("fastconformer.segment.separators", list(decoding.segment_seperators), GGUFValueType.STRING)
unspaced = all(language in UNSPACED_LANGUAGES for language in LANGUAGES[args.model])
add_array("fastconformer.segment.breaks", UNSPACED_BREAKS if unspaced else [], GGUFValueType.STRING)
w.add_bool("fastconformer.tokenizer.byte_fallback", byte_fallback)
add_array("fastconformer.tokenizer.tokens", [p.piece for p in proto.pieces], GGUFValueType.STRING)
w.add_uint32("fastconformer.tokenizer.unknown_id", int(tokenizer.tokenizer.unk_id()))
w.add_string("fastconformer.tokenizer.unknown_surface", proto.trainer_spec.unk_surface)
# SentencePiece strips a dummy leading "▁" when either normalizer option is set.
w.add_bool("fastconformer.tokenizer.strip_leading_space",
           bool(proto.normalizer_spec.add_dummy_prefix or proto.normalizer_spec.remove_extra_whitespaces))
w.add_bool("fastconformer.tokenizer.remove_extra_whitespaces", bool(proto.normalizer_spec.remove_extra_whitespaces))
# Normal pieces hold no whitespace, and the unknown surface holds only spaces; BYTE pieces can supply Unicode whitespace.
assert not any(c.isspace() for p in proto.pieces for c in p.piece)
assert all(c == " " or not c.isspace() for c in proto.trainer_spec.unk_surface)
add_array("fastconformer.tokenizer.punctuation", sorted(decoding.supported_punctuation or []), GGUFValueType.STRING)
# The decoding strips no language tags from the text (strip_lang_tags, for models that write them), which the C++
# does not do.
assert not decoding.strip_lang_tags

sd = {k: v.detach().float().numpy() for k, v in model.state_dict().items()}


def add(name, data):
    w.add_tensor(name, np.ascontiguousarray(data, dtype=np.float32))


add("frontend.window", sd["preprocessor.featurizer.window"])
add("frontend.filterbank", sd["preprocessor.featurizer.fb"][0])

# The subsampling's nn.Sequential: Conv2d, ReLU, then (depthwise Conv2d, pointwise Conv2d, ReLU) per halving.
s = "encoder.pre_encode."
add("sub.conv.0.weight", sd[s + "conv.0.weight"])
add("sub.conv.0.bias", sd[s + "conv.0.bias"])
for i in range(1, int(np.log2(enc.subsampling_factor))):
    dw, pw = 3 * i - 1, 3 * i
    add(f"sub.conv.{i}.dw.weight", sd[s + f"conv.{dw}.weight"])
    add(f"sub.conv.{i}.dw.bias", sd[s + f"conv.{dw}.bias"])
    add(f"sub.conv.{i}.pw.weight", sd[s + f"conv.{pw}.weight"][:, :, 0, 0])
    add(f"sub.conv.{i}.pw.bias", sd[s + f"conv.{pw}.bias"])
add("sub.out.weight", sd[s + "out.weight"])
add("sub.out.bias", sd[s + "out.bias"])

d = int(enc.d_model)
for l in range(int(enc.n_layers)):
    a, o = f"encoder.layers.{l}.", f"blk.{l}."

    def linear(src, dst, bias=use_bias):
        add(dst + ".weight", sd[a + src + ".weight"])
        if bias:
            add(dst + ".bias", sd[a + src + ".bias"])

    def norm(src, dst):
        add(dst + ".weight", sd[a + src + ".weight"])
        add(dst + ".bias", sd[a + src + ".bias"])

    for ff in ("1", "2"):
        norm(f"norm_feed_forward{ff}", o + f"ff{ff}_norm")
        linear(f"feed_forward{ff}.linear1", o + f"ff{ff}_up")
        linear(f"feed_forward{ff}.linear2", o + f"ff{ff}_down")
    norm("norm_self_att", o + "attn_norm")
    for x in ("q", "k", "v", "out"):
        linear(f"self_attn.linear_{x}", o + f"attn_{x}")
    linear("self_attn.linear_pos", o + "attn_pos", bias=False)
    add(o + "attn_pos_bias_u", sd[a + "self_attn.pos_bias_u"])
    add(o + "attn_pos_bias_v", sd[a + "self_attn.pos_bias_v"])

    norm("norm_conv", o + "conv_norm")
    # GLU keeps the first half of the pointwise convolution's channels and gates them with the second.
    pw1 = sd[a + "conv.pointwise_conv1.weight"][:, :, 0]
    add(o + "conv_pw1_a.weight", pw1[:d])
    add(o + "conv_pw1_gate.weight", pw1[d:])
    if use_bias:
        add(o + "conv_pw1_a.bias", sd[a + "conv.pointwise_conv1.bias"][:d])
        add(o + "conv_pw1_gate.bias", sd[a + "conv.pointwise_conv1.bias"][d:])
    bn = a + "conv.batch_norm."
    scale = sd[bn + "weight"] / np.sqrt(sd[bn + "running_var"] + model.encoder.layers[l].conv.batch_norm.eps)
    dw_bias = sd[a + "conv.depthwise_conv.bias"] if use_bias else 0
    add(o + "conv_dw.weight", sd[a + "conv.depthwise_conv.weight"][:, 0, :] * scale[:, None])
    add(o + "conv_dw.bias", (dw_bias - sd[bn + "running_mean"]) * scale + sd[bn + "bias"])
    add(o + "conv_pw2.weight", sd[a + "conv.pointwise_conv2.weight"][:, :, 0])
    if use_bias:
        add(o + "conv_pw2.bias", sd[a + "conv.pointwise_conv2.bias"])
    norm("norm_out", o + "out_norm")

add("pred.embed.weight", sd["decoder.prediction.embed.weight"])
if prediction == "lstm":
    # PyTorch stacks the gates as input, forget, cell and output; its two biases are summed.
    for l in range(int(dec.pred_rnn_layers)):
        r = "decoder.prediction.dec_rnn.lstm."
        add(f"pred.lstm.{l}.ih.weight", sd[r + f"weight_ih_l{l}"])
        add(f"pred.lstm.{l}.hh.weight", sd[r + f"weight_hh_l{l}"])
        add(f"pred.lstm.{l}.bias", sd[r + f"bias_ih_l{l}"] + sd[r + f"bias_hh_l{l}"])
else:
    # Haas prednet.py keeps the blocks, their pre-norms and final norm under dec_rnn.
    r = "decoder.prediction.dec_rnn."
    for l in range(int(dec.pred_rnn_layers)):
        src, dst = r + f"layers.{l}.", f"pred.kda.{l}."
        for suffix in ("weight", "bias"):
            add(dst + "norm." + suffix, sd[r + f"norms.{l}." + suffix])
        for name, source in (("q", "q_proj"), ("k", "k_proj"), ("v", "v_proj"),
                             ("f_down", "f_proj.0"), ("f_up", "f_proj.1"), ("beta", "b_proj"),
                             ("g_down", "g_proj.0"), ("g_up", "g_proj.1"),
                             ("out_norm", "o_norm"), ("out", "o_proj")):
            add(dst + name + ".weight", sd[src + source + ".weight"])
        add(dst + "g_up.bias", sd[src + "g_proj.1.bias"])
        for name in ("q", "k", "v"):
            add(dst + name + "_conv.weight", sd[src + name + "_conv1d.weight"][:, 0, :])
        add(dst + "a_log", sd[src + "A_log"])
        add(dst + "dt_bias", sd[src + "dt_bias"])
    for suffix in ("weight", "bias"):
        add("pred.norm." + suffix, sd[r + "final_norm." + suffix])
for x in ("enc", "pred"):
    add(f"joint.{x}.weight", sd[f"joint.{x}.weight"])
    add(f"joint.{x}.bias", sd[f"joint.{x}.bias"])
add("joint.out.weight", sd["joint.joint_net.2.weight"])
add("joint.out.bias", sd["joint.joint_net.2.bias"])

# The model's identity and languages in the GGUF specification's general keys, which name the file. They follow the
# tensors, whose parameters give a size label the name does not, and then go first after general.architecture, since
# GGUFWriter writes the keys in the order they were added.
parts = NAMES[args.model]
label = parts["size_label"] or size_label(*w.get_total_parameter_count())
repository = f"https://huggingface.co/{pin['repository']}"
w.add_name(pin["repository"].split("/")[1])
w.add_organization(pin["repository"].split("/")[0])
w.add_basename(parts["basename"])
w.add_size_label(label)
if parts["finetune"]:
    w.add_finetune(parts["finetune"])
if parts["version"]:
    w.add_version(parts["version"])
w.add_license(LICENSES[args.model])
w.add_source_url(f"{repository}/tree/{pin['revision']}")
w.add_source_repo_url(repository)
w.add_file_type(LlamaFileType.ALL_F32)
w.add_languages(sorted(LANGUAGES[args.model]))
w.kv_data[0] = dict(sorted(w.kv_data[0].items(), key=lambda item: not item[0].startswith("general.")))

path = os.path.join(args.out_dir, naming_convention(None, parts["basename"], parts["finetune"], parts["version"], label, "f32") + ".gguf")
w.write_header_to_file(path)
w.write_kv_data_to_file()
w.write_tensors_to_file()
w.close()
print("wrote", path)
