// Checks FastConformer's transducer, RNN-T or TDT, against NeMo on each dump of reference/fastconformer/dump.py, in
// the order data flows: the prediction network on the dump's labels, the joint on the dump's encoder frames and
// prediction outputs, each decoding of the model and the detokenization from the dump's encoder output, then the
// whole path from the dump's audio to the text, which it also times with the graphs each decoding computes. On a GPU
// each decoding runs twice, on the CPU's copy of its tensors and on the GPU. The
// dumps are those of the model, in <reference out dir>/<its general.name>/: the model's default decoding is checked
// against the dump's own tokens and text, and each other one against those in the dump's folder of its name, which
// dump.py --greedy writes for an RNN-T's greedy decoding.
//
// usage: fastconformer-transducer-check <model.gguf> <reference out dir> [gpu|cpu|device name]

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <stdexcept>

#include "args.h"
#include "backend.h"
#include "compare.h"
#include "reference-dumps.h"
#include "fastconformer/layout.h"
#include "fastconformer/recognizer.h"
#include "npy.h"
#include "weight-types.h"

using namespace fastconformer;

namespace {

using Clock = std::chrono::steady_clock;

double ms(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

/** The log-softmax of each row of `logits` ([rows, classes] row-major), in double precision. */
std::vector<float> log_softmax(const std::vector<float> & logits, int classes) {
    std::vector<float> out(logits.size());
    for (size_t r = 0; r < logits.size() / (size_t) classes; r++) {
        const float * row = &logits[r * classes];
        double top = row[0];
        for (int c = 1; c < classes; c++) top = std::max(top, (double) row[c]);
        double sum = 0;
        for (int c = 0; c < classes; c++) sum += std::exp((double) row[c] - top);
        const double log_sum = top + std::log(sum);
        for (int c = 0; c < classes; c++) out[r * classes + c] = (float) ((double) row[c] - log_sum);
    }
    return out;
}

/** The share of rows of two [rows, stride] arrays whose largest element among columns [from, to) is the same. */
double argmax_agreement(const std::vector<float> & a, const std::vector<float> & b, int stride, int from, int to) {
    const size_t rows = a.size() / (size_t) stride;
    size_t same = 0;
    for (size_t r = 0; r < rows; r++) {
        int ia = from, ib = from;
        for (int c = from + 1; c < to; c++) {
            if (a[r * stride + c] > a[r * stride + ia]) ia = c;
            if (b[r * stride + c] > b[r * stride + ib]) ib = c;
        }
        same += ia == ib;
    }
    return rows == 0 ? 0 : (double) same / (double) rows;
}

std::string ids_text(const std::vector<int32_t> & ids) {
    std::string s;
    for (int32_t id : ids) s += " " + std::to_string(id);
    return s;
}

}  // namespace

int main(int argc, char ** argv) {
    const std::vector<std::string> args = utf8_args(argc, argv);
    if (args.size() < 3) {
        std::fprintf(stderr, "usage: %s <model.gguf> <reference out dir> [gpu|cpu|device name]\n", args[0].c_str());
        return 2;
    }
    try {
        if (!takes_weights(args[1], layout, {"F32", "F16"},
                           "its bounds are those of float32 and half precision, and it wants NeMo's tokens")) {
            return 2;
        }
        ggml_backend_t backend = init_backend(args.size() > 3 ? args[3] : "");
        std::printf("backend: %s\n", ggml_backend_name(backend));
        bool ok = true;
        {
            Recognizer recognizer(args[1], backend, default_threads());
            const PredictionNetwork & prediction = recognizer.prediction();
            const Joint & joint = recognizer.joint();
            const std::vector<std::string> & decodings = recognizer.decodings();
            const int hidden = prediction.hidden(), outputs = joint.outputs(), blank = recognizer.decoder(decodings.front()).blank();
            const bool tdt = recognizer.model().str("fastconformer.decoder.kind") == "tdt";
            std::vector<DecodeOn> places{DecodeOn::Cpu};
            if (!recognizer.on_cpu()) places.push_back(DecodeOn::Device);
            auto place_name = [](DecodeOn on) { return on == DecodeOn::Cpu ? "cpu" : "device"; };
            ggml_gallocr_t allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
            for (const auto & d : reference_dumps(args[2], recognizer.model())) {
                const Npy encoded = read_npy((d / "encoded.npy").u8string());
                const Npy pred_labels = read_npy((d / "pred_labels.npy").u8string());
                const Npy pred_output = read_npy((d / "pred_output.npy").u8string());
                const Npy joint_frames = read_npy((d / "joint_frames.npy").u8string());
                const Npy joint_steps = read_npy((d / "joint_steps.npy").u8string());
                const Npy joint_output = read_npy((d / "joint_output.npy").u8string());
                const Npy audio = read_npy((d / "audio.npy").u8string());
                // NeMo's tokens and text of each decoding of the model, in its order.
                std::vector<std::vector<int32_t>> want_ids;
                std::vector<std::string> want_text;
                for (size_t k = 0; k < decodings.size(); k++) {
                    const std::filesystem::path from = k == 0 ? d : d / std::filesystem::u8path(decodings[k]);
                    want_ids.push_back(read_npy((from / "ids.npy").u8string()).i32);
                    want_text.push_back(dump_text(from / "text.txt"));
                }
                std::printf("%s (%.2f s, %zu prediction steps, %zu joint evaluations)\n", d.filename().u8string().c_str(),
                            (double) audio.f32.size() / recognizer.sample_rate(), pred_labels.i32.size(), joint_frames.i32.size());

                // The prediction network on the dump's labels, each step from the state of the step it continues: the
                // one before it in greedy TDT decoding, the step of its hypothesis' labels in RNN-T's beam search.
                std::vector<int32_t> parents((size_t) pred_labels.i32.size());
                for (size_t n = 0; n < parents.size(); n++) parents[n] = (int32_t) n - 1;
                if (!tdt) parents = read_npy((d / "pred_parents.npy").u8string()).i32;
                if (parents.size() != pred_labels.i32.size()) throw std::runtime_error("pred_parents.npy and pred_labels.npy differ in length");
                std::vector<float> outputs_got;
                std::vector<PredictionState> states;
                for (size_t n = 0; n < pred_labels.i32.size(); n++) {
                    if (parents[n] >= (int32_t) n) throw std::runtime_error("a step of pred_parents.npy continues one after it");
                    Graph g(512);
                    const PredictionNetwork::Step step =
                        prediction.build(g, pred_labels.i32[n], parents[n] < 0 ? prediction.initial_state() : states[(size_t) parents[n]]);
                    g.output(step.output);
                    for (auto * buffer : step.state) g.output(buffer);
                    g.compute(backend, allocr);
                    const std::vector<float> out = Graph::read(step.output);
                    outputs_got.insert(outputs_got.end(), out.begin(), out.end());
                    states.push_back(PredictionNetwork::read_state(step));
                }
                const Diff dp = compare(outputs_got, pred_output.f32);
                print_diff("  prediction network", dp);

                // The joint on the dump's encoder frames and prediction outputs, each projected here.
                Graph g;
                ggml_tensor * f = joint.project_encoder(g.ctx(), g.input(encoded.f32, encoded.shape[1], encoded.shape[0]));
                ggml_tensor * p = joint.project_prediction(g.ctx(), g.input(pred_output.f32, hidden, pred_output.shape[0]));
                const int64_t k = (int64_t) joint_frames.i32.size();
                f = ggml_get_rows(g.ctx(), f, g.input(joint_frames.i32, k));
                p = ggml_get_rows(g.ctx(), p, g.input(joint_steps.i32, k));
                ggml_tensor * logits = joint.build(g.ctx(), f, p);
                g.output(logits);
                g.compute(backend, allocr);
                const std::vector<float> got = Graph::read(logits);
                const Diff dj = compare(log_softmax(got, outputs), joint_output.f32);
                print_diff("  joint log-probabilities", dj);
                const double labels_agree = argmax_agreement(got, joint_output.f32, outputs, 0, blank + 1);
                std::printf("  argmax agreement: labels %.4f", labels_agree);
                if (outputs > blank + 1) std::printf(", durations %.4f", argmax_agreement(got, joint_output.f32, outputs, blank + 1, outputs));
                std::printf("\n");

                // Measured on an Apple M5 on 2026-10-06 with parakeet-tdt_ctc-0.6b-ja and parakeet-tdt-0.6b-v3: the
                // prediction network 130 to 133 dB on the CPU with F32 weights, 56 to 60 dB with F16, 130 to 134 dB on
                // Metal with F32 and 66 to 70 dB with F16; the joint 140 to 142 dB on the CPU with F32, 76 to 77 dB with
                // F16 and 85 to 89 dB on Metal with either. reazonspeech-nemo-v2's the same day: the prediction network
                // 125 to 128, 61 to 64, 125 to 128 and 69 to 73 dB, the joint 124 to 126, 71 to 72 and 83 to 88 dB. A
                // wrong gate or a wrong output falls far below.
                ok = ok && dp.snr_db > 40 && dj.snr_db > 40;

                // Each decoding and its text from the dump's encoder output.
                Graph e;
                ggml_tensor * projected = joint.project_encoder(e.ctx(), e.input(encoded.f32, encoded.shape[1], encoded.shape[0]));
                e.output(projected);
                e.compute(backend, allocr);
                const std::vector<float> dump_projected = Graph::read(projected);
                for (size_t k = 0; k < decodings.size(); k++) {
                    for (const DecodeOn on : places) {
                        const Decoding decoding = recognizer.decoding(dump_projected, decodings[k], on);
                        const std::string text = recognizer.detokenizer().text(decoding.ids);
                        std::printf("  %s decoding on the %s: ids %s, text %s, %zu graphs\n", decodings[k].c_str(), place_name(on),
                                    decoding.ids == want_ids[k] ? "equal" : "DIFFER", text == want_text[k] ? "equal" : "DIFFERS", decoding.graphs);
                        if (decoding.ids != want_ids[k]) {
                            std::printf("    got%s\n    want%s\n", ids_text(decoding.ids).c_str(), ids_text(want_ids[k]).c_str());
                        }
                        if (text != want_text[k]) std::printf("    got  %s\n    want %s\n", text.c_str(), want_text[k].c_str());
                        ok = ok && decoding.ids == want_ids[k] && text == want_text[k];
                    }
                }

                // The whole path with each decoding, timed after a first run that builds Metal's pipelines.
                for (const std::string & name : decodings) recognizer.recognize(audio.f32, name);
                for (const std::string & name : decodings) recognizer.decoding(dump_projected, name, DecodeOn::Device);
                const auto t0 = Clock::now();
                const std::vector<float> features = recognizer.frontend().features(audio.f32);
                const auto t1 = Clock::now();
                const std::vector<float> path_projected = recognizer.encode(features, recognizer.frontend().frames(audio.f32.size()));
                const auto t2 = Clock::now();
                std::printf("  audio to text: frontend %.1f ms, encoder %.1f ms\n", ms(t0, t1), ms(t1, t2));
                for (size_t k = 0; k < decodings.size(); k++) {
                    for (const DecodeOn on : places) {
                        const auto t3 = Clock::now();
                        const Decoding decoding = recognizer.decoding(path_projected, decodings[k], on);
                        const auto t4 = Clock::now();
                        const std::string text = recognizer.detokenizer().text(decoding.ids);
                        std::printf("    %s decoding on the %s: ids %s, text %s; %.1f ms, %zu graphs\n", decodings[k].c_str(), place_name(on),
                                    decoding.ids == want_ids[k] ? "equal" : "DIFFER", text == want_text[k] ? "equal" : "DIFFERS", ms(t3, t4),
                                    decoding.graphs);
                        if (text != want_text[k]) std::printf("      got  %s\n      want %s\n", text.c_str(), want_text[k].c_str());
                        ok = ok && text == want_text[k];
                    }
                }
            }
            ggml_gallocr_free(allocr);
        }
        ggml_backend_free(backend);
        return ok ? 0 : 1;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
