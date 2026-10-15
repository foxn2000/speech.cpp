#pragma once

#include <string>
#include <vector>

#include "ggml-backend.h"

/**
 * Sets ggml up for this process once, its messages going through log.h. Every function here calls it before it touches
 * a device.
 */
void start_ggml();

/** The devices a model can run on: ggml's CPU and GPUs in ggml's order, without the accelerators it runs beside the CPU. */
const std::vector<ggml_backend_dev_t> & devices();

/**
 * The device that `name` asks for: "auto" for the first GPU or integrated GPU, or the CPU when there is none; "gpu"
 * for the first GPU or integrated GPU; "cpu"; or a device's name, compared without case. A device that is not there
 * throws an Error of the device.
 */
ggml_backend_dev_t find_device(const std::string & name);

/**
 * Starts the backend of `device`, whose CPU computes with `threads` threads when it is the CPU. A device that does not
 * start throws an Error of the device.
 */
ggml_backend_t start_device(ggml_backend_dev_t device, int threads);

/**
 * The CPU threads the library computes with when the caller does not choose: the machine's performance cores, or its
 * physical cores where the system does not tell them apart, or its logical processors where it tells neither.
 */
int default_threads();

/**
 * The backend of the device `name` asks for as find_device() reads it, "auto" when it is empty, with the default
 * threads: what the checks run on.
 */
ggml_backend_t init_backend(const std::string & name);

/** The kind of device `backend` runs on, as a voice file names it: "cpu", "gpu" or "igpu". */
std::string device_kind(ggml_backend_t backend);
