#pragma once

#include <cstdint>
#include <vector>

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"

/**
 * A matrix product that asks for float32 accumulation. Vulkan otherwise accumulates in half precision on
 * GPUs that have it; the CPU and Metal ignore the request (Metal rounds the inputs of its matrix kernel to
 * half precision either way).
 */
inline ggml_tensor * mul_mat(ggml_context * ctx, ggml_tensor * a, ggml_tensor * b) {
    ggml_tensor * t = ggml_mul_mat(ctx, a, b);
    ggml_prec_set_acc(t, GGML_PREC_F32);
    return t;
}

/**
 * Computes `gf`, whose tensors have their memory, on `backend`. A backend that runs out of memory while it computes
 * throws an Error of Fault::OutOfMemory, and one that fails otherwise an Error of Fault::Device.
 */
void compute_graph(ggml_backend_t backend, ggml_cgraph * gf);

/**
 * One computation: a ggml context and graph whose inputs carry their data from the moment they are made,
 * so that the code that builds a stage also says what goes into it. The data is uploaded when the graph
 * is computed, after its tensors have been allocated.
 */
class Graph {
public:
    explicit Graph(int max_nodes = 32768);
    ~Graph();
    Graph(const Graph &) = delete;
    Graph & operator=(const Graph &) = delete;

    ggml_context * ctx() const { return ctx_; }

    ggml_tensor * input(const std::vector<float> & data, int64_t ne0, int64_t ne1 = 1, int64_t ne2 = 1, int64_t ne3 = 1);
    ggml_tensor * input(const std::vector<int32_t> & data, int64_t ne0);
    /** A half-precision input of `data`, [ne0, ne1]: what ggml_flash_attn_ext takes as its mask. */
    ggml_tensor * half_input(const std::vector<float> & data, int64_t ne0, int64_t ne1);
    /** A float32 input of zeros, for padding with a concatenation where a backend pads only on the right. */
    ggml_tensor * zeros(int64_t ne0, int64_t ne1);

    /** Marks `t` as a result to read back after compute(). */
    void output(ggml_tensor * t);

    /**
     * Writes `src` into `dst`, a tensor that outlives the graph, at this point of the graph: a node added afterwards
     * that reads `dst` through a view of it sees what was written.
     */
    void copy(ggml_tensor * src, ggml_tensor * dst);

    /**
     * Puts `t`, and what it needs that the graph does not hold yet, into the graph now, rather than where the first node
     * that reads it is added: the graph runs its nodes in the order they were put in.
     */
    void expand(ggml_tensor * t);

    /** Allocates the graph with `allocr`, uploads the inputs and computes it on `backend`. */
    void compute(ggml_backend_t backend, ggml_gallocr_t allocr);

    /** Gives `input`, an input of this graph, the data `data` from the next compute on, converted to its type. */
    void set(ggml_tensor * input, const std::vector<float> & data);
    void set(ggml_tensor * input, const std::vector<int32_t> & data);

    /**
     * Uploads the inputs and computes the graph on `backend` again where compute() allocated it, which holds while
     * no other graph is allocated with the same allocator.
     */
    void compute_again(ggml_backend_t backend);

    /** The graph's nodes, which outputs and copies have put in, for asking a backend whether it computes each. */
    ggml_cgraph * cgraph() const { return gf_; }
    /** Uploads the inputs' data. */
    void upload();

    /** A result, as float32 in ggml order (ne0 fastest). */
    static std::vector<float> read(const ggml_tensor * t);
    /** A result into `out`, whose memory it reuses. */
    static void read(const ggml_tensor * t, std::vector<float> & out);

private:
    struct Upload {
        ggml_tensor * tensor;
        std::vector<uint8_t> bytes;
    };

    /** The upload of `input`; a tensor that is no input of this graph throws. */
    Upload & upload_of(const ggml_tensor * input);

    ggml_context * ctx_ = nullptr;
    ggml_cgraph * gf_ = nullptr;
    std::vector<Upload> uploads_;
};

