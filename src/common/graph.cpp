#include "graph.h"

#include <cstring>
#include <stdexcept>
#include <string>
#include <system_error>

#include "error.h"

void compute_graph(ggml_backend_t backend, ggml_cgraph * gf) {
    const std::string device = ggml_backend_name(backend);
    ggml_status status = GGML_STATUS_FAILED;
    try {
        status = ggml_backend_graph_compute(backend, gf);
    } catch (const std::system_error & e) {
        // ggml's Vulkan backend allocates buffers of its own while it computes and throws vulkan-hpp's exceptions out of
        // ggml_backend_graph_compute() when one does not fit: vk::OutOfDeviceMemoryError and vk::OutOfHostMemoryError,
        // whose codes are VK_ERROR_OUT_OF_DEVICE_MEMORY (-2) and VK_ERROR_OUT_OF_HOST_MEMORY (-1) in the category
        // "vk::Result". Caught as any exception, they read as a defect of the library.
        const bool vulkan = std::string(e.code().category().name()) == "vk::Result";
        if (vulkan && (e.code().value() == -1 || e.code().value() == -2)) {
            throw Error(Fault::OutOfMemory, "the memory of " + device + " ran out while it computed a graph: " + e.what());
        }
        throw Error(Fault::Device, "the device " + device + " failed to compute a graph: " + e.what());
    }
    if (status == GGML_STATUS_ALLOC_FAILED) throw Error(Fault::OutOfMemory, "the memory of " + device + " ran out while it computed a graph");
    if (status != GGML_STATUS_SUCCESS) throw Error(Fault::Device, "the device " + device + " failed to compute a graph");
}

Graph::Graph(int max_nodes) {
    ggml_init_params params = {ggml_tensor_overhead() * max_nodes + ggml_graph_overhead_custom(max_nodes, false),
                               nullptr, true};
    ctx_ = ggml_init(params);
    if (!ctx_) throw Error(Fault::OutOfMemory, "cannot create a ggml context");
    gf_ = ggml_new_graph_custom(ctx_, max_nodes, false);
}

Graph::~Graph() {
    if (ctx_) ggml_free(ctx_);
}

ggml_tensor * Graph::input(const std::vector<float> & data, int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3) {
    ggml_tensor * t = ggml_new_tensor_4d(ctx_, GGML_TYPE_F32, ne0, ne1, ne2, ne3);
    if ((int64_t) data.size() != ggml_nelements(t)) throw std::runtime_error("an input's data does not fit its shape");
    ggml_set_input(t);
    Upload u{t, std::vector<uint8_t>(ggml_nbytes(t))};
    std::memcpy(u.bytes.data(), data.data(), u.bytes.size());
    uploads_.push_back(std::move(u));
    return t;
}

ggml_tensor * Graph::input(const std::vector<int32_t> & data, int64_t ne0) {
    ggml_tensor * t = ggml_new_tensor_1d(ctx_, GGML_TYPE_I32, ne0);
    if ((int64_t) data.size() != ne0) throw std::runtime_error("an input's data does not fit its shape");
    ggml_set_input(t);
    Upload u{t, std::vector<uint8_t>(ggml_nbytes(t))};
    std::memcpy(u.bytes.data(), data.data(), u.bytes.size());
    uploads_.push_back(std::move(u));
    return t;
}

ggml_tensor * Graph::half_input(const std::vector<float> & data, int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3) {
    ggml_tensor * t = ggml_new_tensor_4d(ctx_, GGML_TYPE_F16, ne0, ne1, ne2, ne3);
    ggml_set_input(t);
    uploads_.push_back({t, std::vector<uint8_t>(ggml_nbytes(t))});
    set(t, data);
    return t;
}

ggml_tensor * Graph::zeros(int64_t ne0, int64_t ne1) {
    return input(std::vector<float>((size_t) (ne0 * ne1), 0.0f), ne0, ne1);
}

void Graph::output(ggml_tensor * t) {
    ggml_set_output(t);
    ggml_build_forward_expand(gf_, t);
}

void Graph::copy(ggml_tensor * src, ggml_tensor * dst) {
    ggml_build_forward_expand(gf_, ggml_cpy(ctx_, src, dst));
}

void Graph::expand(ggml_tensor * t) {
    ggml_build_forward_expand(gf_, t);
}

void Graph::compute(ggml_backend_t backend, ggml_gallocr_t allocr) {
    if (!ggml_gallocr_alloc_graph(allocr, gf_)) throw Error(Fault::OutOfMemory, "cannot allocate the memory of a graph on the device");
    compute_again(backend);
}

void Graph::compute_scheduled(ggml_backend_sched_t sched) {
    ggml_backend_sched_reset(sched);
    if (!ggml_backend_sched_alloc_graph(sched, gf_)) throw Error(Fault::OutOfMemory, "cannot allocate a scheduled graph");
    for (const Upload & u : uploads_) ggml_backend_tensor_set(u.tensor, u.bytes.data(), 0, u.bytes.size());
    if (ggml_backend_sched_graph_compute(sched, gf_) != GGML_STATUS_SUCCESS) throw Error(Fault::Device, "a scheduled graph failed");
}

Graph::Upload & Graph::upload_of(const ggml_tensor * input) {
    for (Upload & u : uploads_) {
        if (u.tensor == input) return u;
    }
    throw std::logic_error("a tensor that is no input of a graph was given data");
}

void Graph::set(ggml_tensor * input, const std::vector<float> & data) {
    Upload & u = upload_of(input);
    if ((int64_t) data.size() != ggml_nelements(input)) throw std::runtime_error("an input's data does not fit its shape");
    if (input->type == GGML_TYPE_F16) ggml_fp32_to_fp16_row(data.data(), reinterpret_cast<ggml_fp16_t *>(u.bytes.data()), ggml_nelements(input));
    else if (input->type == GGML_TYPE_F32) std::memcpy(u.bytes.data(), data.data(), u.bytes.size());
    else throw std::logic_error("float data was given to an input of another type");
}

void Graph::set(ggml_tensor * input, const std::vector<int32_t> & data) {
    Upload & u = upload_of(input);
    if ((int64_t) data.size() != ggml_nelements(input) || input->type != GGML_TYPE_I32) throw std::runtime_error("an input's data does not fit its shape");
    std::memcpy(u.bytes.data(), data.data(), u.bytes.size());
}

void Graph::compute_again(ggml_backend_t backend) {
    for (const Upload & u : uploads_) ggml_backend_tensor_set(u.tensor, u.bytes.data(), 0, u.bytes.size());
    compute_graph(backend, gf_);
}

std::vector<float> Graph::read(const ggml_tensor * t) {
    std::vector<float> out;
    read(t, out);
    return out;
}

void Graph::read(const ggml_tensor * t, std::vector<float> & out) {
    if (t->type != GGML_TYPE_F32) throw std::runtime_error("a result is not float32");
    out.resize((size_t) ggml_nelements(t));
    ggml_backend_tensor_get(t, out.data(), 0, ggml_nbytes(t));
}

