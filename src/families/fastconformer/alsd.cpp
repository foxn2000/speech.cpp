#include "alsd.h"

#include <algorithm>
#include <cmath>
#include <map>
#include <memory>
#include <stdexcept>

#include "error.h"

namespace fastconformer {

namespace {

/** The prediction network's output for a hypothesis' labels and the state it leaves. */
struct Prediction {
    std::vector<float> output;
    PredictionState state;
};

struct Hypothesis {
    /** The labels, the blank first. */
    std::vector<int32_t> labels;
    /** The frame each label after the blank was added on. */
    std::vector<int64_t> frames;
    double score;
    /** The prediction of the labels before the last, whose state the last is fed with; none for the starting blank. */
    std::shared_ptr<const Prediction> from;
};

/** numpy.logaddexp() as it computes it in double precision. */
double logaddexp(double x, double y) {
    if (x == y) return x + std::log(2.0);
    const double d = x - y;
    return d > 0 ? x + std::log1p(std::exp(-d)) : y + std::log1p(std::exp(d));
}

}  // namespace

AlsdDecoder::AlsdDecoder(const ModelFile & m, const PredictionNetwork & prediction, const Joint & joint)
    : prediction_(prediction),
      joint_(joint),
      blank_((int) m.u32("fastconformer.decoder.blank_id")),
      beam_((int) m.u32("fastconformer.decoder.rnnt.beam_size")),
      score_norm_(m.boolean("fastconformer.decoder.rnnt.score_norm")),
      max_target_ratio_(m.f32("fastconformer.decoder.rnnt.max_target_ratio")) {}

Decoding AlsdDecoder::decode(const std::vector<float> & projected, ggml_backend_t backend, const DecodingProgress & progress) const {
    const int hidden = joint_.hidden(), outputs = joint_.outputs(), predicted = prediction_.hidden();
    const int64_t frames = (int64_t) (projected.size() / (size_t) hidden);
    const int beam = std::min(beam_, blank_);
    // int(alsd_max_target_len * T) for a float target length.
    const int64_t max_labels = (int64_t) ((double) max_target_ratio_ * (double) frames);
    std::unique_ptr<ggml_gallocr, decltype(&ggml_gallocr_free)> allocr(ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend)),
                                                                       &ggml_gallocr_free);
    if (!allocr) throw Error(Fault::OutOfMemory, "cannot create a graph allocator");

    // NeMo caches the prediction of each sequence of labels it has computed. Only the beam's sequences can be asked
    // for again in the next step, so the cache keeps those; one asked for after it was dropped is computed again from
    // the same state, which gives the same output.
    std::map<std::vector<int32_t>, std::shared_ptr<const Prediction>> cache;
    using Hypotheses = std::vector<std::shared_ptr<Hypothesis>>;
    Hypotheses beam_hyps{std::make_shared<Hypothesis>(Hypothesis{{blank_}, {}, 0.0, nullptr})};
    // NeMo keeps every finished hypothesis and returns the first of them in a stable sort by this key, so the search
    // keeps only the best so far, which a later one replaces with a larger key alone. A finished hypothesis' score is
    // final once its step has merged the beam: it is never active again, and later steps merge only their own. Keeping
    // all of them raised the peak footprint of reazonspeech-nemo-v2's 311 s input in Q8_0, whose search then runs to its
    // last step, from 1.84 to 2.11 GB on the CPU of an Apple M5 and from 1.01 to 1.26 GB on Metal decoding on the CPU
    // (2026-10-13).
    auto key = [&](const Hypothesis & h) { return score_norm_ ? h.score / (double) h.labels.size() : h.score; };
    std::shared_ptr<Hypothesis> finished;
    std::vector<double> logp((size_t) outputs);
    std::vector<int> tokens((size_t) blank_);
    // The frame the search has passed for every hypothesis, which only rises.
    int64_t passed = 0;
    size_t graphs = 0;
    for (int64_t i = 0; i < frames + max_labels; i++) {
        Hypotheses active;
        std::vector<int64_t> at;
        for (const auto & h : beam_hyps) {
            const int64_t t = i - (int64_t) (h->labels.size() - 1);
            if (t > frames - 1) continue;
            active.push_back(h);
            at.push_back(t);
        }
        if (active.empty()) break;
        passed = std::max(passed, *std::min_element(at.begin(), at.end()));
        if (progress && !progress((double) passed / (double) frames)) break;
        const size_t n = active.size();

        // The predictions the cache lacks and the joint of every active hypothesis at its frame, in one graph.
        Graph g(2048);
        std::vector<std::shared_ptr<const Prediction>> predictions(n);
        std::vector<std::pair<size_t, PredictionNetwork::Step>> computed;
        ggml_tensor * columns = nullptr;
        std::vector<float> f;
        for (size_t j = 0; j < n; j++) {
            const auto it = cache.find(active[j]->labels);
            ggml_tensor * column;
            if (it != cache.end()) {
                predictions[j] = it->second;
                column = g.input(it->second->output, predicted);
            } else {
                const Hypothesis & h = *active[j];
                const PredictionNetwork::Step step = prediction_.build(g, h.labels.back(), h.from ? h.from->state : prediction_.initial_state());
                g.output(step.output);
                for (auto * buffer : step.state) g.output(buffer);
                computed.emplace_back(j, step);
                column = step.output;
            }
            columns = columns ? ggml_concat(g.ctx(), columns, column, 1) : column;
            f.insert(f.end(), projected.begin() + at[j] * hidden, projected.begin() + (at[j] + 1) * hidden);
        }
        ggml_tensor * logits = joint_.build(g.ctx(), g.input(f, hidden, (int64_t) n), joint_.project_prediction(g.ctx(), columns));
        g.output(logits);
        g.compute(backend, allocr.get());
        graphs++;
        for (const auto & c : computed) {
            predictions[c.first] = std::make_shared<const Prediction>(Prediction{Graph::read(c.second.output), PredictionNetwork::read_state(c.second)});
            cache[active[c.first]->labels] = predictions[c.first];
        }
        const std::vector<float> out = Graph::read(logits);

        Hypotheses expanded, finishing;
        for (size_t j = 0; j < n; j++) {
            const Hypothesis & h = *active[j];
            const float * row = &out[j * (size_t) outputs];
            double top = row[0], sum = 0;
            for (int c = 1; c < outputs; c++) top = std::max(top, (double) row[c]);
            for (int c = 0; c < outputs; c++) sum += std::exp((double) row[c] - top);
            for (int c = 0; c < outputs; c++) logp[(size_t) c] = (double) row[c] - top - std::log(sum);

            auto blank = std::make_shared<Hypothesis>(Hypothesis{h.labels, h.frames, h.score + logp[(size_t) blank_], h.from});
            expanded.push_back(blank);
            if (at[j] == frames - 1) finishing.push_back(blank);
            // topk() over the tokens: the largest first, and the lower id first between equal ones.
            for (int c = 0; c < blank_; c++) tokens[(size_t) c] = c;
            std::partial_sort(tokens.begin(), tokens.begin() + beam, tokens.end(),
                              [&](int a, int b) { return logp[(size_t) a] > logp[(size_t) b] || (logp[(size_t) a] == logp[(size_t) b] && a < b); });
            for (int k = 0; k < beam; k++) {
                const int token = tokens[(size_t) k];
                std::vector<int32_t> labels = h.labels;
                labels.push_back(token);
                std::vector<int64_t> emitted = h.frames;
                emitted.push_back(at[j]);
                expanded.push_back(
                    std::make_shared<Hypothesis>(Hypothesis{std::move(labels), std::move(emitted), h.score + logp[(size_t) token], predictions[j]}));
            }
        }

        std::stable_sort(expanded.begin(), expanded.end(), [](const auto & a, const auto & b) { return a->score > b->score; });
        if (expanded.size() > (size_t) beam) expanded.resize((size_t) beam);
        beam_hyps.clear();
        for (const auto & h : expanded) {
            const auto same = std::find_if(beam_hyps.begin(), beam_hyps.end(), [&](const auto & k) { return k->labels == h->labels; });
            if (same != beam_hyps.end()) {
                (*same)->score = logaddexp((*same)->score, h->score);
            } else {
                beam_hyps.push_back(h);
            }
        }
        std::map<std::vector<int32_t>, std::shared_ptr<const Prediction>> kept;
        for (const auto & h : beam_hyps) {
            const auto it = cache.find(h->labels);
            if (it != cache.end()) kept.insert(*it);
        }
        cache.swap(kept);
        for (const auto & h : finishing) {
            if (!finished || key(*h) > key(*finished)) finished = h;
        }
    }

    const Hypothesis & best = finished ? *finished : *beam_hyps.front();
    return Decoding{std::vector<int32_t>(best.labels.begin() + 1, best.labels.end()), best.frames, {}, graphs};
}

}  // namespace fastconformer
