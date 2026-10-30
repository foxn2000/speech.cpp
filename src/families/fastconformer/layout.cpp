#include "layout.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include "error.h"
namespace fastconformer {

namespace {

/**
 * How the layout stores a tensor in a file of each weight type: a matrix of a linear layer or the
 * prediction network's embedding, which ggml_mul_mat() and ggml_get_rows() alone read, in the file's type, which 0.7's
 * reader took in F16 and F32 alone, so that a file of it in Q8_0 is read from 0.8.0; and the convolution kernels, which
 * ggml's direct and depthwise convolutions and ggml_ssm_conv() read in F32 alone, the norms, the biases, the frontend and
 * the rest in F32.
 */
const Storage kMatrix = quantized_storage({{GGML_TYPE_F32, kFirstLayoutRelease},
                                           {GGML_TYPE_F16, kFirstLayoutRelease},
                                           {GGML_TYPE_Q8_0, "0.8.0"},
                                           {GGML_TYPE_Q6_K, "0.8.0"},
                                           {GGML_TYPE_Q5_K, "0.8.0"},
                                           {GGML_TYPE_Q4_K, "0.8.0"}});
const Storage kF32 = float32_storage();

/** The width of the subsampling's convolutions, NeMo's default, which the encoder pads by 1 on either side. */
constexpr int64_t kSubsamplingWidth = 3;

void require(bool condition, const ModelFile & m, const std::string & what) {
    if (!condition) throw Error(Fault::File, m.path() + ": " + what + "; " + m.remedy());
}

/** Reads every key of the layout, checks the ones that must agree, and names the tensors they call for. */
std::vector<TensorSpec> tensors(const ModelFile & m) {
    check_model_keys(m, "recognition", "checked");
    const std::string p = "fastconformer.";
    const int n_fft = m.size(p + "frontend.n_fft"), mels = m.size(p + "frontend.n_mels");
    m.size(p + "frontend.hop_length");
    for (const char * key : {"preemphasis", "log_guard", "std_guard"}) m.f32(p + "frontend." + key);
    // The frontend's real FFT packs the frame's values in pairs.
    require(n_fft % 2 == 0, m, "fastconformer.frontend.n_fft is odd");
    const int d = m.size(p + "encoder.d_model"), heads = m.size(p + "encoder.num_heads"), kernel = m.size(p + "encoder.conv_kernel");
    // The relative positions are encoded as a sine and a cosine for each pair of channels.
    require(d % 2 == 0 && d % heads == 0, m, "fastconformer.encoder.d_model is odd or not a multiple of fastconformer.encoder.num_heads");
    require(kernel % 2 == 1, m, "fastconformer.encoder.conv_kernel is not odd");
    for (const char * key : {"norm_eps", "pos_base", "xscale", "ff_factor"}) m.f32(p + "encoder." + key);
    if (m.one_of(p + "encoder.attention", {"rel_pos", "rel_pos_local_attn"}) == "rel_pos_local_attn") {
        m.size(p + "encoder.attention_context");
        m.size(p + "encoder.global_tokens");
    }
    const uint32_t factor = m.u32(p + "encoder.subsampling_factor");
    int halvings = 0;
    for (uint32_t f = factor; f > 1; f /= 2) halvings++;
    require(factor >= 2 && (1u << halvings) == factor, m, "fastconformer.encoder.subsampling_factor is not a power of two");
    const bool use_bias = m.boolean(p + "encoder.use_bias");
    const int layers = m.count(p + "encoder.num_layers");

    const bool tdt = m.one_of(p + "decoder.kind", {"tdt", "rnnt"}) == "tdt";
    const uint32_t blank = m.u32(p + "decoder.blank_id");
    int64_t outputs = (int64_t) blank + 1;
    if (tdt) {
        const std::vector<int32_t> durations = m.i32_array(p + "decoder.tdt.durations");
        require(!durations.empty() && std::all_of(durations.begin(), durations.end(), [](int32_t duration) { return duration >= 0; }), m,
                "fastconformer.decoder.tdt.durations is empty or holds a negative duration");
        outputs += (int64_t) durations.size();
        m.size(p + "decoder.tdt.max_symbols");
    } else {
        // BeamRNNTInfer runs greedy_search() instead with a beam of 1.
        require(m.size(p + "decoder.rnnt.beam_size") >= 2, m, "fastconformer.decoder.rnnt.beam_size is less than 2");
        m.boolean(p + "decoder.rnnt.score_norm");
        const float ratio = m.f32(p + "decoder.rnnt.max_target_ratio");
        require(ratio >= 0 && std::isfinite(ratio), m, "fastconformer.decoder.rnnt.max_target_ratio is negative or not finite");
        m.size(p + "decoder.rnnt.max_symbols");
    }
    const int prediction_layers = m.count(p + "decoder.prediction_layers");
    const std::string prediction = m.one_of(p + "decoder.prediction", {"lstm", "kda"});
    m.str_array(p + "segment.separators");
    m.str_array(p + "segment.breaks");
    require(m.str_array(p + "tokenizer.tokens").size() == blank, m, "the blank does not follow the last of fastconformer.tokenizer.tokens");
    require(m.u32(p + "tokenizer.unknown_id") < blank, m, "fastconformer.tokenizer.unknown_id is not a piece");
    m.str(p + "tokenizer.unknown_surface");
    m.boolean(p + "tokenizer.strip_leading_space");
    m.boolean(p + "tokenizer.remove_extra_whitespaces");
    m.boolean(p + "tokenizer.byte_fallback");
    m.str_array(p + "tokenizer.punctuation");

    // The window's length and the widths of the subsampling's channels, the feed-forward layers, the prediction network
    // and the joint have no key: each is the width of one tensor, which the others are checked against.
    const int64_t window = m.width("frontend.window", 0), channels = m.width("sub.conv.0.weight", 3);
    require(window <= n_fft, m, "frontend.window is longer than fastconformer.frontend.n_fft");
    const int64_t ffn = m.width("blk.0.ff1_up.weight", 1), predicted = m.width("pred.embed.weight", 0), joint = m.width("joint.enc.weight", 1);
    // Each stride-2 convolution of the subsampling halves the mel axis, rounding up.
    int64_t subsampled_mels = mels;
    for (int i = 0; i < halvings; i++) subsampled_mels = (subsampled_mels - 1) / 2 + 1;

    const int64_t k = kSubsamplingWidth;
    std::vector<TensorSpec> t = {{"frontend.window", {window}, kF32},
                                 {"frontend.filterbank", {n_fft / 2 + 1, mels}, kF32},
                                 {"sub.conv.0.weight", {k, k, 1, channels}, kF32},
                                 {"sub.conv.0.bias", {channels}, kF32},
                                 {"sub.out.weight", {channels * subsampled_mels, d}, kMatrix},
                                 {"sub.out.bias", {d}, kF32},
                                 {"pred.embed.weight", {predicted, (int64_t) blank + 1}, kMatrix},
                                 {"joint.enc.weight", {d, joint}, kMatrix},
                                 {"joint.enc.bias", {joint}, kF32},
                                 {"joint.pred.weight", {predicted, joint}, kMatrix},
                                 {"joint.pred.bias", {joint}, kF32},
                                 {"joint.out.weight", {joint, outputs}, kMatrix},
                                 {"joint.out.bias", {outputs}, kF32}};
    for (int i = 1; i < halvings; i++) {
        add_block(t, "sub.conv." + std::to_string(i) + ".",
                  {{"dw.weight", {k, k, 1, channels}, kF32},
                   {"dw.bias", {channels}, kF32},
                   {"pw.weight", {channels, channels}, kMatrix},
                   {"pw.bias", {channels}, kF32}});
    }
    add_numbered(t, "blk.", layers,
                 {{"ff1_norm.weight", {d}, kF32},          {"ff1_norm.bias", {d}, kF32},
                  {"ff2_norm.weight", {d}, kF32},          {"ff2_norm.bias", {d}, kF32},
                  {"attn_norm.weight", {d}, kF32},         {"attn_norm.bias", {d}, kF32},
                  {"conv_norm.weight", {d}, kF32},         {"conv_norm.bias", {d}, kF32},
                  {"out_norm.weight", {d}, kF32},          {"out_norm.bias", {d}, kF32},
                  {"attn_pos.weight", {d, d}, kMatrix},    {"attn_pos_bias_u", {d / heads, heads}, kF32},
                  {"attn_pos_bias_v", {d / heads, heads}, kF32}, {"conv_dw.weight", {kernel, d}, kF32},
                  {"conv_dw.bias", {d}, kF32},             {"ff1_up.weight", {d, ffn}, kMatrix},
                  {"ff1_down.weight", {ffn, d}, kMatrix},  {"ff2_up.weight", {d, ffn}, kMatrix},
                  {"ff2_down.weight", {ffn, d}, kMatrix},  {"attn_q.weight", {d, d}, kMatrix},
                  {"attn_k.weight", {d, d}, kMatrix},      {"attn_v.weight", {d, d}, kMatrix},
                  {"attn_out.weight", {d, d}, kMatrix},    {"conv_pw1_a.weight", {d, d}, kMatrix},
                  {"conv_pw1_gate.weight", {d, d}, kMatrix}, {"conv_pw2.weight", {d, d}, kMatrix}});
    if (use_bias) {
        add_numbered(t, "blk.", layers,
                     {{"ff1_up.bias", {ffn}, kF32},    {"ff1_down.bias", {d}, kF32},   {"ff2_up.bias", {ffn}, kF32},
                      {"ff2_down.bias", {d}, kF32},    {"attn_q.bias", {d}, kF32},     {"attn_k.bias", {d}, kF32},
                      {"attn_v.bias", {d}, kF32},      {"attn_out.bias", {d}, kF32},   {"conv_pw1_a.bias", {d}, kF32},
                      {"conv_pw1_gate.bias", {d}, kF32}, {"conv_pw2.bias", {d}, kF32}});
    }
    if (prediction == "lstm") {
        // An LSTM layer's four gates, input, forget, cell and output, are stacked in its matrices and its summed biases.
        add_numbered(t, "pred.lstm.", prediction_layers,
                     {{"ih.weight", {predicted, 4 * predicted}, kMatrix}, {"hh.weight", {predicted, 4 * predicted}, kMatrix},
                      {"bias", {4 * predicted}, kF32}});
    } else {
        const int h = m.size(p + "decoder.kda.num_heads");
        for (const char * key : {"norm_eps", "gate_norm_eps", "qk_norm_eps"}) {
            const float eps = m.f32(p + "decoder.kda." + key);
            require(std::isfinite(eps) && eps > 0, m, p + "decoder.kda." + key + " is not finite and positive");
        }
        const int64_t kh = m.width("pred.kda.0.q.weight", 1), w = m.width("pred.kda.0.q_conv.weight", 0);
        const int64_t r = m.width("pred.kda.0.f_down.weight", 1), hd = kh / h;
        require(kh % h == 0 && (hd == 32 || hd == 64 || hd == 128), m,
                "pred.kda.0.q.weight does not give a KDA head width of 32, 64 or 128");
        require(w >= 2, m, "pred.kda.0.q_conv.weight has fewer than two convolution inputs");
        add_numbered(t, "pred.kda.", prediction_layers,
                     {{"norm.weight", {predicted}, kF32}, {"norm.bias", {predicted}, kF32},
                      {"q.weight", {predicted, kh}, kMatrix}, {"k.weight", {predicted, kh}, kMatrix},
                      {"v.weight", {predicted, kh}, kMatrix},
                      {"q_conv.weight", {w, kh}, kF32}, {"k_conv.weight", {w, kh}, kF32}, {"v_conv.weight", {w, kh}, kF32},
                      {"f_down.weight", {predicted, r}, kMatrix}, {"f_up.weight", {r, kh}, kMatrix},
                      {"a_log", {h}, kF32}, {"dt_bias", {kh}, kF32}, {"beta.weight", {predicted, h}, kMatrix},
                      {"g_down.weight", {predicted, r}, kMatrix}, {"g_up.weight", {r, kh}, kMatrix}, {"g_up.bias", {kh}, kF32},
                      {"out_norm.weight", {hd}, kF32}, {"out.weight", {kh, predicted}, kMatrix}});
        add_block(t, "pred.norm.", {{"weight", {predicted}, kF32}, {"bias", {predicted}, kF32}});
    }
    return t;
}

/** Brings layouts 1 and 2 up with their LSTM prediction and tokenizer without byte fallback. */
void upgrade(ModelFile & m) {
    const std::string name = m.str("general.name");
    // The pinned tokenizers' normalizer_spec sets remove_extra_whitespaces for parakeet-ja and ReazonSpeech only.
    const struct { const char * name; bool remove_extra_whitespaces; } legacy[] = {
        {"parakeet-tdt_ctc-0.6b-ja", true}, {"reazonspeech-nemo-v2", true}, {"parakeet-tdt-0.6b-v3", false}};
    const bool * remove_extra = nullptr;
    std::string names;
    for (const auto & model : legacy) {
        if (!names.empty()) names += ", ";
        names += model.name;
        if (name == model.name) remove_extra = &model.remove_extra_whitespaces;
    }
    require(remove_extra != nullptr, m, "it is a model of layout " + std::to_string(m.layout_version()) + " other than " + names);
    m.upgrade_bool("fastconformer.tokenizer.remove_extra_whitespaces", *remove_extra);
    if (m.layout_version() == 1 && m.one_of("fastconformer.decoder.kind", {"tdt", "rnnt"}) == "rnnt") {
        // Layout 1's converter takes one RNN-T checkpoint, reazonspeech-nemo-v2, whose decoding.greedy.max_symbols is 10.
        require(name == "reazonspeech-nemo-v2", m, "it is an RNN-T model of layout 1 other than reazonspeech-nemo-v2");
        m.upgrade_u32("fastconformer.decoder.rnnt.max_symbols", 10);
    }
    m.upgrade_str("fastconformer.decoder.prediction", "lstm");
    m.upgrade_bool("fastconformer.tokenizer.byte_fallback", false);
}

}  // namespace

const Layout layout = {"fastconformer", 3,
                       "convert it again with reference/fastconformer/convert.py, or download it again from its Hugging Face "
                       "repository",
                       tensors, upgrade};

}  // namespace fastconformer
