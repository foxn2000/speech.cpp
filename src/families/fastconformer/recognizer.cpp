#include "recognizer.h"

#include <stdexcept>

#include "backend.h"
#include "error.h"
#include "layout.h"

namespace fastconformer {

Recognizer::Transducer::Transducer(const ModelFile & m, const std::vector<std::string> & names) : prediction(make_prediction(m)), joint(m), names(names) {
    for (const std::string & name : names) decoders.push_back(make_decoder(m, *prediction, joint, name));
}

const Decoder & Recognizer::Transducer::decoder(const std::string & name) const {
    for (size_t i = 0; i < names.size(); i++) {
        if (names[i] == name) return *decoders[i];
    }
    throw std::logic_error("a FastConformer model has no decoding named " + name);
}

Recognizer::OnCpu::OnCpu(const ModelFile & device, int threads, const std::vector<std::string> & names)
    : pool(nullptr, ggml_threadpool_free),
      backend(start_device(find_device("cpu"), threads), ggml_backend_free),
      model(device, backend.get(), decoding_tensor),
      transducer(model, names) {
    ggml_threadpool_params params = ggml_threadpool_params_default(threads);
    pool.reset(ggml_threadpool_new(&params));
    if (!pool) throw Error(Fault::OutOfMemory, "cannot start the CPU's threads for the decoding");
    ggml_backend_cpu_set_threadpool(backend.get(), pool.get());
}

Recognizer::Recognizer(const std::string & path, ggml_backend_t backend, int threads)
    : backend_(backend),
      on_cpu_(ggml_backend_dev_type(ggml_backend_get_device(backend)) == GGML_BACKEND_DEVICE_TYPE_CPU),
      threads_(threads),
      model_(path, backend, layout),
      frontend_(model_, "fastconformer.", true),
      encoder_(model_, backend),
      decoding_names_(fastconformer::decodings(model_)),
      device_(model_, decoding_names_),
      detokenizer_(model_),
      separators_(model_.str_array("fastconformer.segment.separators")),
      breaks_(model_.str_array("fastconformer.segment.breaks")),
      allocr_(ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend))) {
    if (!allocr_) throw Error(Fault::OutOfMemory, "cannot create a graph allocator");
}

Recognizer::~Recognizer() {
    ggml_gallocr_free(allocr_);
}

std::vector<float> Recognizer::encode(const std::vector<float> & features, int64_t frames) {
    Graph g;
    ggml_tensor * out = device_.joint.project_encoder(g.ctx(), encoder_.build(g, features, frames));
    g.output(out);
    g.compute(backend_, allocr_);
    return Graph::read(out);
}

Transcript Recognizer::recognize(const std::vector<float> & samples, const std::string & name, DecodeOn on) {
    const std::vector<float> features = frontend_.features(samples);
    Transcript t;
    t.decoding = decoding(encode(features, frontend_.frames(samples.size())), name, on);
    t.text = detokenizer_.text(t.decoding.ids);
    return t;
}

Recognizer::OnCpu & Recognizer::cpu() {
    if (!cpu_) cpu_ = std::make_unique<OnCpu>(model_, threads_, decoding_names_);
    return *cpu_;
}

Decoding Recognizer::decoding(const std::vector<float> & projected, const std::string & name, DecodeOn on,
                              const DecodingProgress & progress) {
    if (on == DecodeOn::Device || on_cpu_) return device_.decoder(name).decode(projected, backend_, progress);
    OnCpu & copy = cpu();
    return copy.transducer.decoder(name).decode(projected, copy.backend.get(), progress);
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
