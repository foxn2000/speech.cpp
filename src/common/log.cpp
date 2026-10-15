#include "log.h"

#include <cstdio>
#include <memory>
#include <cstdlib>
#include <mutex>

#include "ggml.h"

namespace {

std::mutex sink_mutex;
/** The sink, or none for the default; a message holds its own copy while it is written. */
std::shared_ptr<const LogSink> sink;

void to_stderr(LogLevel level, const char * text) {
    if (level == LogLevel::Warn || level == LogLevel::Error || std::getenv("SPEECH_LOG_ALL")) std::fputs(text, stderr);
}

/** The level of ggml's last message on this thread, which a continuation (GGML_LOG_LEVEL_CONT) keeps. */
thread_local LogLevel continued = LogLevel::Info;

void from_ggml(enum ggml_log_level level, const char * text, void *) {
    switch (level) {
        case GGML_LOG_LEVEL_DEBUG: continued = LogLevel::Debug; break;
        case GGML_LOG_LEVEL_WARN: continued = LogLevel::Warn; break;
        case GGML_LOG_LEVEL_ERROR: continued = LogLevel::Error; break;
        case GGML_LOG_LEVEL_CONT: break;
        default: continued = LogLevel::Info; break;
    }
    std::shared_ptr<const LogSink> s;
    {
        std::lock_guard<std::mutex> lock(sink_mutex);
        s = sink;
    }
    if (!s) to_stderr(continued, text);
    else if (*s) (*s)(continued, text);
}

}  // namespace

void set_log_sink(LogSink s) {
    route_ggml_log();
    auto next = std::make_shared<const LogSink>(std::move(s));
    std::lock_guard<std::mutex> lock(sink_mutex);
    sink = std::move(next);
}

void log_message(LogLevel level, const std::string & text) {
    std::shared_ptr<const LogSink> s;
    {
        std::lock_guard<std::mutex> lock(sink_mutex);
        s = sink;
    }
    const std::string line = text + "\n";
    if (!s) to_stderr(level, line.c_str());
    else if (*s) (*s)(level, line.c_str());
}

void route_ggml_log() {
    static std::once_flag once;
    std::call_once(once, [] { ggml_log_set(from_ggml, nullptr); });
}
