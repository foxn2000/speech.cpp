"""Runs the stage checks of the given families on the given devices and prints one line per check and a summary.

It finds the check executables in a build folder (<build>/ or, as CI's checks artifact and Visual Studio lay them out,
<build>/Release/), the model files under the given paths, which default to the repository's models/ folder and the
folder `speech pull` fetches into, and the dumps under <reference>/<family>/out/. A model file is known by its GGUF
keys: its architecture names its family, and of the files with the same general.name and weight type the one of the
newest speech.layout runs. Each family's checks run in the order data flows, for each model and each device; a check
that takes no device runs once per model, on the first device's turn, and a check that forces the attention runs on a
GPU once with each attention. A check whose dumps or inputs are missing is skipped and says what it looked for.

A check that exits with 2 refused its arguments, as a check refuses weights it cannot judge; that is reported apart and
does not fail the run. Any other exit but 0 fails it. Each check's whole output goes to a log of its own in --logs.

families: qwen3-tts, irodori-tts, drama3, qwen3-asr, fastconformer, silero-vad, common (the checks that take no model), all

usage: python3 checks/run.py <family>... [--build DIR] [--devices cpu,gpu] [--types F32] [--models PATH...]
                             [--reference DIR] [--references DIR] [--decoder-text FILE] [--only CHECK...]
                             [--logs DIR] [--gpu-lock FILE] [--list]
"""

import argparse
import json
import os
import re
import struct
import subprocess
import sys
import tempfile
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FAMILIES = ["qwen3-tts", "irodori-tts", "drama3", "qwen3-asr", "fastconformer", "silero-vad", "common"]
# general.file_type's values of the weight types speech.cpp writes, as weight_types() of src/common/storage.cpp lists
# them; a script cannot read that table.
FILE_TYPES = {0: "F32", 1: "F16", 7: "Q8_0", 18: "Q6_K", 16: "Q5_K", 14: "Q4_K"}
REFUSED = 2

SCALARS = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}


def gguf_keys(path):
    """The key-value pairs of a GGUF file's header, without the contents of its arrays, or None for another file."""

    def read(f, fmt):
        return struct.unpack(fmt, f.read(struct.calcsize(fmt)))[0]

    def string(f):
        return f.read(read(f, "<Q")).decode("utf-8", "replace")

    def value(f, kind):
        if kind == 8:
            return string(f)
        if kind == 9:
            item, count = read(f, "<I"), read(f, "<Q")
            if item == 8:
                for _ in range(count):
                    f.seek(read(f, "<Q"), 1)
            elif item in SCALARS:
                f.seek(count * struct.calcsize(SCALARS[item]), 1)
            else:
                for _ in range(count):
                    value(f, item)
            return None
        return read(f, SCALARS[kind])

    with open(path, "rb") as f:
        if f.read(4) != b"GGUF":
            return None
        read(f, "<I")
        read(f, "<Q")
        keys = {}
        for _ in range(read(f, "<Q")):
            key = string(f)
            keys[key] = value(f, read(f, "<I"))
        return keys


class Model:
    def __init__(self, path, keys):
        self.path = path
        self.family = keys["general.architecture"]
        self.name = keys["general.name"]
        self.type = FILE_TYPES.get(keys.get("general.file_type"), str(keys.get("general.file_type")))
        self.layout = keys["speech.layout"]
        self.repository = keys.get("general.source.repo_url", "")
        # general.source.url names the revision, as https://huggingface.co/<repository>/tree/<revision>.
        self.revision = keys.get("general.source.url", "").rstrip("/").rsplit("/", 1)[-1]
        self.captions = bool(keys.get("irodori-tts.caption_condition"))

    def label(self):
        return f"{self.name} {self.type}"


def default_model_paths():
    paths = [os.path.join(REPO, "models")]
    chosen = os.environ.get("SPEECH_MODEL_DIR")
    if chosen:
        paths.append(chosen)
    elif sys.platform == "win32":
        paths.append(os.path.join(os.environ.get("LOCALAPPDATA", ""), "speech.cpp", "models"))
    elif sys.platform == "darwin":
        paths.append(os.path.expanduser("~/Library/Caches/speech.cpp/models"))
    else:
        xdg = os.environ.get("XDG_CACHE_HOME", "")
        paths.append(os.path.join(xdg if os.path.isabs(xdg) else os.path.expanduser("~/.cache"), "speech.cpp", "models"))
    return paths


def find_models(paths, families, types):
    """The model files of the families and types under the paths, the newest layout of each name and type, in order."""
    files = []
    for p in paths:
        if os.path.isfile(p):
            files.append(p)
        for root, _, names in os.walk(p, followlinks=True):
            files += [os.path.join(root, n) for n in sorted(names) if n.endswith(".gguf")]
    chosen = {}
    for f in files:
        try:
            keys = gguf_keys(f)
        except (OSError, struct.error, UnicodeDecodeError):
            keys = None
        # Files of no layout, such as voice files and the converters' files before layouts, are not models.
        if not keys or "speech.layout" not in keys or keys.get("general.architecture") not in families:
            continue
        m = Model(f, keys)
        if m.type not in types:
            continue
        key = (m.family, m.name, m.type)
        if key not in chosen or m.layout > chosen[key].layout:
            chosen[key] = m
    return sorted(chosen.values(), key=lambda m: (families.index(m.family), m.name, types.index(m.type)))


class Run:
    """One run of a check: its executable, its arguments, the device it runs on or None, and what it ran on."""

    def __init__(self, check, args, device=None, what="", skip=None):
        self.check, self.args, self.device, self.what, self.skip = check, args, device, what, skip


def runs_on(device, check, model, args, what=""):
    return Run(check, [model.path] + args + [device], device, f"{model.label()} {device}{what}")


def attentions(device, check, model, args, what=""):
    """A check that forces the attention runs as the device chooses on the CPU, and with each attention on a GPU."""
    if device == "cpu":
        return [runs_on(device, check, model, args, what)]
    return [Run(check, [model.path] + args + [device, a], device, f"{model.label()} {device} {a}{what}") for a in ("flash", "products")]


def missing(check, what, path):
    return Run(check, [], None, what, f"no {path}")


def reading(paths, runs):
    """The runs of a check, or, where a file or folder it reads besides the model's dumps is missing, a skip of each that
    names it."""
    absent = [p for p in paths if not os.path.exists(p)]
    return [Run(r.check, [], None, r.what, f"no {absent[0]}") for r in runs] if absent else runs


def named_dumps(check, model, out):
    """The skip of a check whose model has no dumps under <out>/<general.name>/, or None."""
    d = os.path.join(out, model.name)
    return None if os.path.isdir(d) else missing(check, model.label(), d)


def qwen3_tts(model, devices, ctx):
    out = os.path.join(ctx.reference, "qwen3-tts", "out")
    dumps = []
    if os.path.isdir(out):
        for d in sorted(os.listdir(out)):
            meta = os.path.join(out, d, "meta.json")
            if os.path.isfile(meta):
                with open(meta, encoding="utf-8") as f:
                    dumps.append((os.path.join(out, d), json.load(f)))
    runs = []
    cases, decode = os.path.join(out, "tokenizer-cases.tsv"), os.path.join(out, "decode-cases.tsv")
    if os.path.isfile(cases) and os.path.isfile(decode):
        runs.append(Run("tokenizer-check", [model.path, cases, decode], None, model.label()))
    else:
        runs.append(missing("tokenizer-check", model.label(), cases))
    text = decoder_text(dumps, ctx)
    sampling = os.path.join(out, "sampling")
    for i, device in enumerate(devices):
        if text:
            runs += attentions(device, "qwen3-decoder-check", model, [text])
        else:
            runs.append(missing("qwen3-decoder-check", model.label(), "text: no dump's meta.json under " + out))
        # dump.py records the base name of the checkpoint's folder: the repository's name, or the revision of a Hugging
        # Face snapshot.
        checkpoint = {model.repository.rstrip("/").rsplit("/", 1)[-1], model.revision}
        own = [d for d, meta in dumps if meta.get("model_dir") in checkpoint]
        runs += [runs_on(device, "qwen3-tts-talker-check", model, [d], " " + os.path.basename(d)) for d in own]
        if not own:
            runs.append(missing("qwen3-tts-talker-check", model.label(), f"dump of {model.name} under {out}"))
        if os.path.isdir(sampling):
            if i == 0:
                runs.append(Run("qwen3-tts-sampler-check", [model.path, sampling], None, model.label()))
            runs.append(runs_on(device, "qwen3-tts-graph-sampler-check", model, [sampling]))
        elif i == 0:
            runs.append(missing("qwen3-tts-sampler-check", model.label(), sampling))
        # Every Qwen3-TTS file holds the same codec, so each dump's codes check it.
        runs += [runs_on(device, "qwen3-tts-codec-check", model, [d], " " + os.path.basename(d)) for d, _ in dumps]
    return runs


def decoder_text(dumps, ctx):
    """A text for qwen3-decoder-check, which repeats it until its rows fill the blocks it needs: --decoder-text, or the
    Qwen3-TTS dumps' texts, written to the logs folder."""
    if ctx.decoder_text:
        return ctx.decoder_text
    texts = [meta["text"] for _, meta in dumps if meta.get("text")]
    if not texts:
        return None
    path = os.path.join(ctx.logs, "qwen3-decoder-text.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write("".join(texts))
    return path


def irodori_tts(model, devices, ctx):
    out = os.path.join(ctx.reference, "irodori-tts", "out")
    if not os.path.isdir(out):
        return [missing("irodori-tts", model.label(), out)]
    runs = [Run("irodori-window-check", [], None, "")] if model is ctx.first.get("irodori-tts") else []
    codec = []
    for d in sorted(os.listdir(out)):
        meta = os.path.join(out, d, "meta.json")
        if not os.path.isfile(meta):
            continue
        with open(meta, encoding="utf-8") as f:
            m = json.load(f)
        if "https://huggingface.co/" + m["model"]["repository"] != model.repository or m.get("embedding") or not m.get("reference"):
            continue
        refs = m["reference"] if isinstance(m["reference"], list) else [m["reference"]]
        # Dumps of the same references at the same loudness check the encoder alike; the first of them checks it.
        if all((refs, m.get("ref_normalize_db", "model")) != (r, n) for _, r, n in codec):
            codec.append((os.path.join(out, d), refs, m.get("ref_normalize_db", "model")))
    def cases_of(file):
        """The cases text_cases.py or caption_cases.py made with the model, in the folder whose meta.json names it."""
        for d in sorted(os.listdir(out)):
            meta = os.path.join(out, d, "meta.json")
            if os.path.isfile(os.path.join(out, d, file)) and os.path.isfile(meta):
                with open(meta, encoding="utf-8") as f:
                    if "https://huggingface.co/" + json.load(f)["model"]["repository"] == model.repository:
                        return os.path.join(out, d, file)
        return os.path.join(out, f"<{model.name}'s folder>", file)

    for i, device in enumerate(devices):
        cases = [cases_of("text-cases.tsv")]
        if model.captions:
            cases.append(cases_of("caption-cases.tsv"))
        runs += reading(cases, [runs_on(device, "irodori-text-check", model, [out])])
        runs.append(runs_on(device, "irodori-condition-check", model, [out]))
        runs += attentions(device, "irodori-dit-check", model, [out])
        if not ctx.references:
            if i == 0:
                runs.append(missing("irodori-codec-check", model.label(), "--references, the folder of the dumps' reference WAVE files"))
        else:
            for d, refs, _ in codec:
                wavs = [os.path.join(ctx.references, r) for r in refs]
                absent = [w for w in wavs if not os.path.isfile(w)]
                if absent:
                    runs.append(missing("irodori-codec-check", f"{model.label()} {device} {os.path.basename(d)}", absent[0]))
                else:
                    runs.append(Run("irodori-codec-check", [model.path, d] + wavs + [device], device,
                                    f"{model.label()} {device} {os.path.basename(d)}"))
        runs += attentions(device, "irodori-synthesis-check", model, [out])
    return runs


def drama3(model, devices, ctx):
    out = os.path.join(ctx.reference, "drama3", "out")
    dumps = []
    if os.path.isdir(out):
        for d in sorted(os.listdir(out)):
            meta = os.path.join(out, d, "meta.json")
            if os.path.isfile(meta):
                with open(meta, encoding="utf-8") as f:
                    m = json.load(f)
                if "https://huggingface.co/" + m["model"]["repository"] == model.repository and m.get("reference"):
                    dumps.append(os.path.join(out, d))
    sampling, prompts = os.path.join(out, "sampling"), os.path.join(out, "prompt-cases.jsonl")
    runs = [Run("drama3-prompt-check", [model.path, prompts], None, model.label()) if os.path.isfile(prompts)
            else missing("drama3-prompt-check", model.label(), prompts),
            Run("drama3-sampler-check", [model.path, sampling], None, model.label()) if os.path.isdir(sampling)
            else missing("drama3-sampler-check", model.label(), sampling)]
    if not dumps:
        return runs + [missing("drama3-codec-check", model.label(), f"dump of {model.name} with a reference under {out}")]
    for device in devices:
        for d in dumps:
            # The codec check reads the dump's reference recording too, where --references holds it.
            with open(os.path.join(d, "meta.json"), encoding="utf-8") as f:
                wav = os.path.join(ctx.references, json.load(f)["reference"]["file"]) if ctx.references else None
            runs.append(runs_on(device, "drama3-codec-check", model, [d] + ([wav] if wav and os.path.isfile(wav) else []), " " + os.path.basename(d)))
        runs += [runs_on(device, f"drama3-{stage}-check", model, [d], " " + os.path.basename(d)) for stage in ("text-model", "audio-decoder") for d in dumps]
    return runs


def qwen3_asr(model, devices, ctx):
    out = os.path.join(ctx.reference, "qwen3-asr", "out")
    absent = named_dumps("qwen3-asr", model, out)
    if absent:
        return [absent]
    runs = [Run("qwen3-asr-frontend-check", [model.path, out], None, model.label())]
    cases, decode = os.path.join(out, "tokenizer-cases.tsv"), os.path.join(out, "decode-cases.tsv")
    if os.path.isfile(cases) and os.path.isfile(decode):
        runs.append(Run("tokenizer-check", [model.path, cases, decode], None, model.label()))
    else:
        runs.append(missing("tokenizer-check", model.label(), cases))
    for device in devices:
        runs.append(runs_on(device, "qwen3-asr-encoder-check", model, [out]))
        runs += reading([os.path.join(out, "parse-cases.jsonl")], attentions(device, "qwen3-asr-decoder-check", model, [out]))
        runs += reading([os.path.join(out, "split")], [runs_on(device, "qwen3-asr-split-check", model, [out])])
    return runs


def fastconformer(model, devices, ctx):
    out = os.path.join(ctx.reference, "fastconformer", "out")
    absent = named_dumps("fastconformer", model, out)
    if absent:
        return [absent]
    runs = [Run("fastconformer-frontend-check", [model.path, out], None, model.label())]
    for device in devices:
        runs += [runs_on(device, f"fastconformer-{stage}-check", model, [out]) for stage in ("encoder", "transducer", "times")]
    return runs


def silero_vad(model, devices, ctx):
    out = os.path.join(ctx.reference, "silero-vad", "out")
    absent = named_dumps("silero-vad", model, out)
    return [absent] if absent else [runs_on(device, "silero-vad-check", model, [out]) for device in devices]


def common(ctx):
    runs = [Run("fft-check", [], None, "")]
    resample = os.path.join(ctx.reference, "resample", "out")
    runs.append(Run("resample-check", [resample], None, "") if os.path.isdir(resample) else missing("resample-check", "", resample))
    cases = os.path.join(ctx.reference, "unicode", "out", "normalization-cases.tsv")
    runs.append(Run("unicode-check", [cases], None, "") if os.path.isfile(cases) else missing("unicode-check", "", cases))
    runs.append(Run("quantize-check", [os.path.join(ctx.logs, "quantize-check")], None, ""))
    return runs


PLANS = {"qwen3-tts": qwen3_tts, "irodori-tts": irodori_tts, "drama3": drama3, "qwen3-asr": qwen3_asr, "fastconformer": fastconformer, "silero-vad": silero_vad}


def executable(build, check):
    suffix = ".exe" if sys.platform == "win32" else ""
    for folder in (build, os.path.join(build, "Release")):
        path = os.path.join(folder, check + suffix)
        if os.path.isfile(path):
            return path
    return None


class GpuLock:
    """An exclusive lock on a file, or nothing without one, held while a check runs on a device other than the CPU, so
    that several runners or other programs that take the same lock (lockf -k on macOS) share one GPU a check at a
    time."""

    def __init__(self, path):
        self.path = path
        self.f = None

    def __enter__(self):
        if self.path:
            import fcntl

            self.f = open(self.path, "a")
            fcntl.flock(self.f, fcntl.LOCK_EX)

    def __exit__(self, *_):
        if self.f:
            self.f.close()


SNR = re.compile(r"SNR\s+(-?[0-9.]+|-?inf)\s*dB")


def key_figure(text, code):
    """The line the summary shows of a check's output: where it failed, the first line that says FAIL or error; then
    the line of the least SNR it printed, of the comparisons it printed, where it printed one; and otherwise its last
    line."""
    lines = [" ".join(line.split()) for line in text.splitlines() if line.strip()]
    if code != 0:
        for line in lines:
            if "FAIL" in line or line.startswith("error:"):
                return line
    snrs = [(float(m.group(1)), line) for line in lines for m in SNR.finditer(line)]
    if snrs:
        return f"{min(snrs)[1]} (least of {len(snrs)})"
    return lines[-1] if lines else ""


def main():
    p = argparse.ArgumentParser(description="Runs the stage checks of the given families.")
    p.add_argument("families", nargs="+", choices=FAMILIES + ["all"])
    p.add_argument("--build", default=os.path.join(REPO, "build"), help="the folder of the check executables")
    p.add_argument("--devices", default="cpu", help="comma-separated: cpu, gpu or a device's name")
    p.add_argument("--types", default="F32", help="comma-separated weight types of the files to check, or all")
    p.add_argument("--models", nargs="+", help="model files or folders to look in (default: models/ and the model cache)")
    p.add_argument("--reference", default=os.path.join(REPO, "reference"), help="the folder of <family>/out/")
    p.add_argument("--references", help="the folder of the reference WAVE files Irodori-TTS's and Drama 3's dumps name")
    p.add_argument("--decoder-text", help="the text qwen3-decoder-check runs (default: the Qwen3-TTS dumps' texts)")
    p.add_argument("--only", nargs="+", help="run only these checks, by executable name")
    p.add_argument("--logs", help="the folder for each check's output (default: a new temporary folder)")
    p.add_argument("--gpu-lock", help="a file to lock while a check runs on a GPU")
    p.add_argument("--list", action="store_true", help="print each run's command instead of running it")
    ctx = p.parse_args()
    # A check prints model output in other scripts, which a console or a file in Windows' code page cannot hold.
    sys.stdout.reconfigure(errors="replace")

    families = FAMILIES if "all" in ctx.families else [f for f in FAMILIES if f in ctx.families]
    devices = [d.strip() for d in ctx.devices.split(",") if d.strip()]
    types = list(FILE_TYPES.values()) if ctx.types == "all" else [t.strip().upper() for t in ctx.types.split(",")]
    unknown = [t for t in types if t not in FILE_TYPES.values()]
    if unknown:
        p.error(f"--types names {', '.join(unknown)}; the types are {', '.join(FILE_TYPES.values())} and all")
    if ctx.gpu_lock and sys.platform == "win32":
        p.error("--gpu-lock takes a lock with flock(), which Windows does not have")
    if ctx.logs:
        os.makedirs(ctx.logs, exist_ok=True)
    else:
        ctx.logs = tempfile.mkdtemp(prefix="speech-checks-")

    model_paths = ctx.models or default_model_paths()
    models = find_models(model_paths, families, types)
    ctx.first = {}
    for m in models:
        ctx.first.setdefault(m.family, m)
    runs = []
    for family in families:
        if family == "common":
            runs += common(ctx)
            continue
        found = [m for m in models if m.family == family]
        if not found:
            print(f"error: no {family} file of {', '.join(types)} under {', '.join(model_paths)}; give one with --models", file=sys.stderr)
            return 2
        for m in found:
            runs += PLANS[family](m, devices, ctx)
    if ctx.only:
        runs = [r for r in runs if r.check in ctx.only]

    if ctx.list:
        for r in runs:
            print(f"skip {r.check} {r.what}: {r.skip}" if r.skip else " ".join([r.check] + r.args))
        return 0

    counts = {"pass": 0, "FAIL": 0, "refused": 0, "skip": 0}
    failed_logs = []
    for i, r in enumerate(runs):
        label = f"{r.check:32} {r.what}"
        if r.skip:
            counts["skip"] += 1
            print(f"skip              {label}: {r.skip}", flush=True)
            continue
        exe = executable(ctx.build, r.check)
        if not exe:
            counts["FAIL"] += 1
            print(f"FAIL              {label}: no {r.check} in {ctx.build}; build the checks", flush=True)
            continue
        log = os.path.join(ctx.logs, f"{i:03d}-{r.check}.log")
        locked = ctx.gpu_lock and r.device and r.device != "cpu"
        with open(log, "w", encoding="utf-8") as f, GpuLock(ctx.gpu_lock if locked else None):
            f.write(" ".join([exe] + r.args) + "\n\n")
            f.flush()
            start = time.monotonic()
            code = subprocess.run([exe] + r.args, stdout=f, stderr=subprocess.STDOUT).returncode
            seconds = time.monotonic() - start
        with open(log, encoding="utf-8", errors="replace") as f:
            figure = key_figure(f.read().split("\n\n", 1)[-1], code)
        for a in r.args:
            figure = figure.replace(a, os.path.basename(a))
        status = "pass" if code == 0 else "refused" if code == REFUSED else "FAIL"
        counts[status] += 1
        if status == "FAIL":
            failed_logs.append(log)
            if code != 1:
                figure = f"exit {code}: {figure}"
        print(f"{status:7} {seconds:7.1f}s  {label}: {figure[:160]}", flush=True)

    print(f"\n{counts['pass']} passed, {counts['FAIL']} failed, {counts['refused']} refused, {counts['skip']} skipped; logs in {ctx.logs}")
    for log in failed_logs:
        print(f"  {log}")
    return 1 if counts["FAIL"] else 0


if __name__ == "__main__":
    sys.exit(main())
