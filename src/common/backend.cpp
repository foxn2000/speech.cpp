#include "backend.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <mutex>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>

#include "error.h"
#include "ggml-cpu.h"
#include "log.h"

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#elif defined(__APPLE__)
#include <sys/sysctl.h>
#else
#include <fstream>
#include <sstream>
#endif

namespace {

std::string lower(std::string s) {
    for (char & c : s) c = (char) std::tolower((unsigned char) c);
    return s;
}

bool is_gpu(ggml_backend_dev_t dev) {
    const enum ggml_backend_dev_type type = ggml_backend_dev_type(dev);
    return type == GGML_BACKEND_DEVICE_TYPE_GPU || type == GGML_BACKEND_DEVICE_TYPE_IGPU;
}

std::string device_names() {
    std::string out;
    for (ggml_backend_dev_t dev : devices()) out += (out.empty() ? "" : ", ") + std::string(ggml_backend_dev_name(dev));
    return out;
}

#if !defined(_WIN32) && !defined(__APPLE__)
/** The numbers of a Linux CPU list such as "0-3,8-11". */
std::vector<int> cpu_list(const std::string & text) {
    std::vector<int> out;
    std::stringstream ranges(text);
    std::string range;
    while (std::getline(ranges, range, ',')) {
        const size_t dash = range.find('-');
        const int first = std::atoi(range.c_str()), last = dash == std::string::npos ? first : std::atoi(range.c_str() + dash + 1);
        for (int c = first; c <= last; c++) out.push_back(c);
    }
    return out;
}

std::optional<std::string> read_line(const std::string & path) {
    std::ifstream f(path);
    std::string line;
    if (!f || !std::getline(f, line)) return std::nullopt;
    return line;
}
#endif

}  // namespace

void start_ggml() {
    route_ggml_log();
    static std::once_flag once;
    std::call_once(once, [] { ggml_backend_dev_count(); });
}

const std::vector<ggml_backend_dev_t> & devices() {
    start_ggml();
    static const std::vector<ggml_backend_dev_t> list = [] {
        std::vector<ggml_backend_dev_t> out;
        for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
            ggml_backend_dev_t dev = ggml_backend_dev_get(i);
            // An accelerator such as BLAS runs beside the CPU and cannot run a graph alone.
            if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU || is_gpu(dev)) out.push_back(dev);
        }
        return out;
    }();
    return list;
}

ggml_backend_dev_t find_device(const std::string & name) {
    const std::string wanted = lower(name);
    ggml_backend_dev_t cpu = nullptr, gpu = nullptr;
    for (ggml_backend_dev_t dev : devices()) {
        if (!cpu && ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU) cpu = dev;
        if (!gpu && is_gpu(dev)) gpu = dev;
    }
    if (wanted == "auto") {
        if (gpu) return gpu;
        if (cpu) return cpu;
        throw Error(Fault::Device, "ggml lists no device to run on", "device");
    }
    if (wanted == "gpu") {
        if (gpu) return gpu;
        throw Error(Fault::Device, "no GPU was found; the devices are " + device_names() + ". Give \"cpu\" or \"auto\" to run on the CPU",
                    "device");
    }
    if (wanted == "cpu" && cpu) return cpu;
    for (ggml_backend_dev_t dev : devices()) {
        if (lower(ggml_backend_dev_name(dev)) == wanted) return dev;
    }
    throw Error(Fault::Device, "no device is named \"" + name + "\"; the devices are " + device_names() + ", and \"auto\", \"gpu\" and \"cpu\" choose one",
                "device");
}

ggml_backend_t start_device(ggml_backend_dev_t device, int threads) {
    ggml_backend_t backend = ggml_backend_dev_init(device, nullptr);
    if (!backend) throw Error(Fault::Device, std::string("cannot start the device ") + ggml_backend_dev_name(device), "device");
    if (ggml_backend_is_cpu(backend)) ggml_backend_cpu_set_n_threads(backend, threads);
    return backend;
}

int default_threads() {
    static const int threads = [] {
#ifdef __APPLE__
        for (const char * key : {"hw.perflevel0.physicalcpu", "hw.physicalcpu"}) {
            int n = 0;
            size_t size = sizeof n;
            if (sysctlbyname(key, &n, &size, nullptr, 0) == 0 && n > 0) return n;
        }
#elif defined(_WIN32)
        DWORD length = 0;
        GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &length);
        std::vector<char> buffer(length);
        auto * first = reinterpret_cast<SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX *>(buffer.data());
        if (length > 0 && GetLogicalProcessorInformationEx(RelationProcessorCore, first, &length)) {
            // A core of a higher efficiency class is a faster one; on a CPU of one kind every core has class 0.
            int fastest = -1, cores = 0;
            for (DWORD at = 0; at < length;) {
                const auto * info = reinterpret_cast<const SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX *>(buffer.data() + at);
                const int efficiency = info->Processor.EfficiencyClass;
                if (efficiency > fastest) {
                    fastest = efficiency;
                    cores = 0;
                }
                if (efficiency == fastest) cores++;
                at += info->Size;
            }
            if (cores > 0) return cores;
        }
#else
        // An Intel CPU with performance and efficiency cores lists its performance cores' logical processors here.
        const std::optional<std::string> listed = read_line("/sys/devices/cpu_core/cpus");
        const std::optional<std::string> online = listed ? listed : read_line("/sys/devices/system/cpu/online");
        std::set<std::pair<std::string, std::string>> cores;
        if (online) {
            for (int c : cpu_list(*online)) {
                const std::string topology = "/sys/devices/system/cpu/cpu" + std::to_string(c) + "/topology/";
                const auto package = read_line(topology + "physical_package_id"), core = read_line(topology + "core_id");
                if (package && core) cores.insert({*package, *core});
            }
        }
        if (!cores.empty()) return (int) cores.size();
#endif
        return (int) std::max(1u, std::thread::hardware_concurrency());
    }();
    return threads;
}

ggml_backend_t init_backend(const std::string & name) {
    return start_device(find_device(name.empty() ? "auto" : name), default_threads());
}

std::string device_kind(ggml_backend_t backend) {
    switch (ggml_backend_dev_type(ggml_backend_get_device(backend))) {
        case GGML_BACKEND_DEVICE_TYPE_CPU: return "cpu";
        case GGML_BACKEND_DEVICE_TYPE_GPU: return "gpu";
        case GGML_BACKEND_DEVICE_TYPE_IGPU: return "igpu";
        default: throw std::logic_error(std::string("the device ") + ggml_backend_name(backend) + " is of a kind a voice file cannot name");
    }
}
