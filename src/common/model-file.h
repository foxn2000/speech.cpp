#pragma once

#include <cstdint>
#include <cstdio>
#include <functional>
#include <initializer_list>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "ggml-backend.h"
#include "ggml.h"
#include "gguf.h"
#include "storage.h"

class ModelFile;

/**
 * What a reader takes of a GGUF file: the general.architecture it reads, the speech.layout it knows, what to tell
 * the owner of a file it cannot read, the tensors a file calls for, which `tensors` names with the shape and the
 * types of each after it has read and checked the file's keys, and how a file of an earlier layout is brought up to
 * this one.
 */
struct Layout {
    const char * architecture;
    uint32_t version;
    /** What to do with a file this reader refuses, such as "convert it again with reference/x/convert.py". */
    const char * remedy;
    std::function<std::vector<TensorSpec>(const ModelFile & file)> tensors;
    /**
     * Brings the metadata of a file of an earlier layout, whose version ModelFile::layout_version() gives, up to this
     * layout before anything else reads it, setting what the earlier layout lacks with ModelFile's upgrade setters
     * (upgrade_u32() and the others). Empty for a family whose layout has had one version, whose reader refuses every
     * other.
     */
    std::function<void(ModelFile & file)> upgrade = {};
};

/** The bytes of one value of a GGUF type that is neither a string nor an array. */
size_t gguf_scalar_size(gguf_type type);

/** Moves `f` to `offset` from its start, which fseek() cannot do past 2 GiB on Windows, whose long is 32 bits. */
bool seek_file(FILE * f, uint64_t offset);

/**
 * The general.architecture of a GGUF file, read without loading its tensors. A file without speech.layout, written
 * for a release before layouts began, throws.
 */
std::string gguf_architecture(const std::string & path);

/**
 * A GGUF file of a known layout whose metadata stays readable and whose tensors, all of them or those a reader asks
 * for, live in one backend buffer. A key is read by its exact type: one that is missing or of another type throws
 * with a message naming it, where ggml's own getters would abort the process. Every failure throws an Error: Io for
 * a file that cannot be opened or read, File for one whose content this release cannot use.
 */
class ModelFile {
public:
    /**
     * Opens the file, checks its architecture and layout against `layout` and its tensors against the ones its keys
     * call for, a tensor missing, one not called for, and one of another shape or type included, and only then loads
     * the tensors onto `backend`: every tensor, or those whose names `keep` accepts.
     */
    ModelFile(const std::string & path, ggml_backend_t backend, const Layout & layout,
              const std::function<bool(const std::string & name)> & keep = {});
    /** Opens and checks the file as above and loads no tensor, for a reader of its metadata alone. */
    ModelFile(const std::string & path, const Layout & layout);
    /**
     * A copy onto `backend` of the loaded tensors of `source` whose names `keep` accepts, read from the backend that
     * holds them and not from the file, which may have changed or gone since `source` read it. It shares the metadata of
     * `source`.
     */
    ModelFile(const ModelFile & source, ggml_backend_t backend, const std::function<bool(const std::string & name)> & keep);
    ModelFile(const ModelFile &) = delete;
    ModelFile & operator=(const ModelFile &) = delete;

    /** A loaded tensor; one the file does not hold or that was not loaded throws. */
    ggml_tensor * tensor(const std::string & name) const;

    /** Experiment: whether the tensor `name` was loaded. */
    bool loaded(const std::string & name) const { return by_name_.count(name) > 0; }

    /** Whether the file has `key`, for a key that a file has only where its model has a value for it. */
    bool has(const std::string & key) const;
    uint32_t u32(const std::string & key) const;
    /**
     * A u32 key that sizes the model, such as a width or a number of heads: 0 throws, and so does a value past the
     * range of int, in which the families hold sizes.
     */
    int size(const std::string & key) const;
    /**
     * A u32 key that numbers layers or blocks of tensors: 0 throws, and so does a number past the tensors the file
     * holds, since each layer holds one at least.
     */
    int count(const std::string & key) const;
    /**
     * Axis `axis` of the tensor `name` as the file stores it, for a layout to take a width that no key gives from one
     * tensor and check the others against it. A tensor the file does not hold throws, and so does a width of 0 or one
     * past the range of int.
     */
    int64_t width(const std::string & name, int axis) const;
    /** The type of the tensor `name` as the file stores it, whether or not it is loaded; one the file does not hold throws. */
    ggml_type stored_type(const std::string & name) const;
    /** The number of tensors the file holds. */
    int64_t tensor_count() const;
    float f32(const std::string & key) const;
    double f64(const std::string & key) const;
    bool boolean(const std::string & key) const;
    std::string str(const std::string & key) const;
    /** A string that names a kind; a value other than `values` throws. */
    std::string one_of(const std::string & key, std::initializer_list<const char *> values) const;
    std::vector<int32_t> i32_array(const std::string & key) const;
    std::vector<float> f32_array(const std::string & key) const;
    std::vector<double> f64_array(const std::string & key) const;
    std::vector<std::string> str_array(const std::string & key) const;

    /**
     * Sets a key of the metadata, of the type each setter names, as it is read, not in the file, for a layout's upgrade
     * of a file of an earlier layout; the model information then lists it among the file's entries.
     */
    void upgrade_u32(const std::string & key, uint32_t value);
    void upgrade_f32(const std::string & key, float value);
    void upgrade_bool(const std::string & key, bool value);
    void upgrade_str(const std::string & key, const std::string & value);
    void upgrade_i32_array(const std::string & key, const std::vector<int32_t> & values);
    void upgrade_f32_array(const std::string & key, const std::vector<float> & values);

    const std::string & path() const { return path_; }
    /** The layout's remedy, for a message about the file that a family's reader throws. */
    const std::string & remedy() const { return remedy_; }
    /**
     * What to tell the owner of a file that holds what this release does not know, such as a weight type: the release
     * that the file's speech.requires names, where it comes after this one, or else the layout's remedy.
     */
    std::string remedy_for_unknown() const;
    /** The file's own speech.layout, which an upgrade does not change. */
    uint32_t layout_version() const { return version_; }

    /** The size of the file in bytes. */
    uint64_t file_bytes() const;
    /** The bytes of every tensor the file holds, as it stores them. */
    uint64_t weight_bytes() const;
    /** The bytes of the tensors the file holds in each type, as it stores them. */
    std::map<ggml_type, uint64_t> type_bytes() const;
    /** The number of metadata entries, the key of the one at `index`, and its value as JSON text. */
    size_t meta_count() const;
    std::string meta_key(size_t index) const;
    std::string meta_json(size_t index) const;

private:
    /** Opens and checks the file, which both constructors do. */
    void open(const Layout & layout);
    /** The key's id, once it is known to have `type` and, for an array, `element`. */
    int64_t key_id(const std::string & key, gguf_type type, gguf_type element = GGUF_TYPE_COUNT) const;
    /** "layout 1 of qwen3-tts", or the architecture alone while the layout is not yet read. */
    std::string layout_name() const;
    void check_tensors(const Layout & layout) const;
    /** Allocates the tensors whose names `keep` accepts, or every tensor, on `backend`, without their values. */
    void allocate(ggml_backend_t backend, const std::function<bool(const std::string & name)> & keep);
    /** Reads the values of the allocated tensors from the file. */
    void load();

    std::string path_, architecture_, remedy_;
    uint32_t version_ = 0;
    std::shared_ptr<gguf_context> gguf_;
    /** The shapes and types of every tensor of the file, which ggml reads with its metadata. */
    std::shared_ptr<ggml_context> ctx_;
    /** The copies of the tensors a reader keeps, when it keeps only some. */
    std::unique_ptr<ggml_context, decltype(&ggml_free)> kept_{nullptr, ggml_free};
    /** The context of the loaded tensors: ctx_, kept_, or none for a reader of the metadata alone. */
    ggml_context * loaded_ = nullptr;
    /**
     * The loaded tensors by name. ggml_get_tensor() compares the name with every tensor of the context before it, and
     * a decoder looks up its layers' tensors for every graph it builds: on an Apple M5, building a step of the 0.6B
     * Qwen3-ASR decoder, 308 lookups, took 0.36 ms with it and takes 0.12 ms here, of a step of 7.5 ms (2026-10-07).
     */
    std::unordered_map<std::string, ggml_tensor *> by_name_;
    std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)> buffer_{nullptr, ggml_backend_buffer_free};
};

/**
 * What a model file says of the model it holds in the GGUF specification's general keys (ggml's docs/gguf.md): its
 * names, its license, where it was converted from, and the type of its weights.
 */
struct ModelIdentity {
    /** general.name, the name of the upstream repository. */
    std::string name;
    /** general.organization, and general.basename and general.size_label, which the file's name begins with. */
    std::string organization, basename, size_label;
    /** general.finetune and general.version, which a model whose name has none does not have. */
    std::optional<std::string> finetune, version;
    /** general.license, an SPDX expression. */
    std::string license;
    /** general.source.repo_url, and the revision that general.source.url, `<repository>/tree/<revision>`, names. */
    std::string repository, revision;
    /** The type general.file_type names, which holds most of the tensors' bytes: "F32", "F16", "Q8_0", "Q6_K", "Q5_K" or "Q4_K". */
    std::string weight_type;
};

/**
 * Reads and checks the general keys of a model file's identity: every one but general.finetune and general.version
 * is required, and none is an empty string. general.file_type must name one of weight_types() and the type that holds
 * most of the tensors' bytes, and a file with a quantized tensor must have general.quantization_version.
 */
ModelIdentity read_identity(const ModelFile & file);


/**
 * Reads and checks the keys every model file has: its identity (read_identity()), speech.sample_rate and speech.task,
 * which must be the family's `task`; and for a task that takes languages, synthesis and recognition,
 * general.languages, ISO 639 codes of two or three lowercase letters sorted without repeats, and speech.language_use,
 * which must be the family's `language_use`. A model of detection takes no language, and its file has neither key,
 * which its family gives as a `language_use` of nullptr.
 */
void check_model_keys(const ModelFile & file, const char * task, const char * language_use);

/** Appends a tensor of each of `parts`, named `prefix` and the part's name, to `tensors`: the tensors of one block. */
void add_block(std::vector<TensorSpec> & tensors, const std::string & prefix, std::initializer_list<TensorSpec> parts);

/**
 * Appends, for each of 0 to `count` - 1, a tensor of each of `parts`, named `prefix`, the number and the part's name
 * after a dot, or `prefix` and the number alone for a part of an empty name, to `tensors`: the tensors of numbered
 * blocks of the same shapes.
 */
void add_numbered(std::vector<TensorSpec> & tensors, const std::string & prefix, int count, std::initializer_list<TensorSpec> parts);
