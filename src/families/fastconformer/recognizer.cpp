#include "recognizer.h"

#include <cstdlib>
#include <stdexcept>

#include "error.h"

#include "layout.h"

namespace fastconformer {

namespace {

std::vector<std::unique_ptr<Decoder>> make_decoders(const ModelFile & m, const PredictionNetwork & prediction, const Joint & joint,
                                                    const std::vector<std::string> & names) {
    std::vector<std::unique_ptr<Decoder>> decoders;
    for (const std::string & name : names) decoders.push_back(make_decoder(m, prediction, joint, name));
    return decoders;
}

}  // namespace

Recognizer::Recognizer(const std::string & path, ggml_backend_t backend)
    : backend_(backend),
      model_(path, backend, layout),
      frontend_(model_),
      encoder_(model_, backend),
      prediction_(model_),
      joint_(model_),
      decoding_names_(fastconformer::decodings(model_)),
      decoders_(make_decoders(model_, prediction_, joint_, decoding_names_)),
      detokenizer_(model_),
      separators_(model_.str_array("fastconformer.segment.separators")),
      breaks_(model_.str_array("fastconformer.segment.breaks")),
      allocr_(ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend))) {
    if (!allocr_) throw Error(Fault::OutOfMemory, "cannot create a graph allocator");
    if (std::getenv("SPEECH_FC_CPU_COPY")) cpu_ = std::make_unique<CpuDecoding>(path, decoding_names_);
}

Recognizer::CpuDecoding::CpuDecoding(const std::string & path, const std::vector<std::string> & names)
    : backend(ggml_backend_cpu_init()), model(path, backend, layout), prediction(model), joint(model),
      decoders(make_decoders(model, prediction, joint, names)) {}

Recognizer::~Recognizer() {
    ggml_gallocr_free(allocr_);
}

std::vector<float> Recognizer::encode(const std::vector<float> & features, int64_t frames) {
    Graph g;
    ggml_tensor * out = joint_.project_encoder(g.ctx(), encoder_.build(g, features, frames));
    g.output(out);
    g.compute(backend_, allocr_);
    return Graph::read(out);
}

Transcript Recognizer::recognize(const std::vector<float> & samples, const std::string & name) {
    const std::vector<float> features = frontend_.features(samples);
    Transcript t;
    t.decoding = decoding(encode(features, frontend_.frames(samples.size())), name);
    t.text = detokenizer_.text(t.decoding.ids);
    return t;
}

const Decoder & Recognizer::decoder(const std::string & name) const {
    for (size_t i = 0; i < decoding_names_.size(); i++) {
        if (decoding_names_[i] == name) return *decoders_[i];
    }
    throw std::logic_error(model_.path() + " has no decoding named " + name);
}

std::vector<Segment> Recognizer::segments(const Decoding & decoding) const {
    return fastconformer::segments(detokenizer_.token_texts(decoding.ids), detokenizer_.word_starts(decoding.ids),
                                   token_spans(decoding, detokenizer_), separators_, breaks_);
}

double Recognizer::seconds(int64_t frame) const {
    const double window_stride = (double) frontend_.hop_length() / (double) frontend_.sample_rate();
    return (double) frame * window_stride * (double) encoder_.subsampling_factor();
}

}  // namespace fastconformer
