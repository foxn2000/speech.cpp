"""A client of the worker protocol for the smoke scripts. It starts `speech worker`, checks that ready names the protocol
it speaks, and on every line it reads it checks that the line is one JSON object with a string "type", that a chunk,
progress or terminal message belongs to a request in flight, and that a request gets exactly one terminal message (end, error or cancelled) and
nothing for its id after it. It also compares the model information of ready with `speech info --json`, reads the
requests a recognition model's reference dumped, with the texts they must give, and holds the paragraph and the character error rate with which the
smoke scripts check a long text spoken a sentence at a time.
"""

import array
import ast
import json
import os
import struct
import subprocess
import tempfile
import time
import unicodedata

TERMINAL = {"end", "error", "cancelled"}
# The protocol this client speaks; a worker that says another in ready is one its callers must change for.
PROTOCOL = 3


class Worker:
    def __init__(self, speech, model, options, stderr=subprocess.DEVNULL):
        self.speech, self.model, self.options = speech, model, options
        self.proc = subprocess.Popen([speech, "worker", model, *options], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=stderr)
        self.in_flight = set()
        self.backlog = []
        self.started = time.perf_counter()
        self.ready = self._read_line()
        if self.ready["type"] != "ready":
            raise SystemExit(f"the worker did not start: {self.ready}")
        if self.ready["protocol"] != PROTOCOL:
            raise SystemExit(f"the worker speaks protocol {self.ready['protocol']!r}, and this client protocol {PROTOCOL}")

    def _read_line(self):
        line = self.proc.stdout.readline()
        if not line:
            raise SystemExit("the worker exited")
        text = line.decode("utf-8").rstrip("\n")
        if text.endswith("\r"):
            raise SystemExit("a line ends with \\r")
        try:
            message = json.loads(text)
        except json.JSONDecodeError:
            message = None
        if not isinstance(message, dict) or not isinstance(message.get("type"), str):
            raise SystemExit(f"a line on stdout is not a JSON object with a type: {text[:200]!r}")
        message["_at"] = time.perf_counter()
        return message

    def read(self):
        """The next message, checked against the requests in flight."""
        m = self._read_line()
        kind, id = m["type"], m.get("id")
        if kind in TERMINAL and id is None:
            if kind != "error":
                raise SystemExit(f"a terminal message without an id: {m}")
        elif kind in TERMINAL or kind in ("chunk", "progress"):
            if id not in self.in_flight:
                raise SystemExit(f"{kind} for {id!r}, which has no request in flight: {short(m)}")
            if kind in TERMINAL:
                self.in_flight.remove(id)
        return m

    def send_line(self, line):
        self.proc.stdin.write((line + "\n").encode("utf-8"))
        self.proc.stdin.flush()

    def send(self, message):
        """Sends a line that starts no request: a chunk after the first, or a cancel."""
        self.send_line(json.dumps(message, ensure_ascii=False))

    def expect(self, id):
        """Notes that a line sent under `id` starts a request, which must get one terminal message."""
        if id in self.in_flight:
            raise SystemExit(f"the smoke started a second request under {id!r}")
        self.in_flight.add(id)

    def request(self, message):
        self.expect(message["id"])
        self.send(message)

    def next_for(self, id):
        """The next message for `id`, keeping the others for later calls."""
        for i, m in enumerate(self.backlog):
            if m.get("id") == id:
                return self.backlog.pop(i)
        while True:
            m = self.read()
            if m.get("id") == id:
                return m
            self.backlog.append(m)

    def until(self, id):
        """The messages for `id` up to and including its terminal message."""
        out = []
        while True:
            m = self.next_for(id)
            out.append(m)
            if m["type"] in TERMINAL:
                return out

    def terminal(self, id, kind=None, code=None, option=None):
        """The terminal message of `id`, checked to be of `kind`, and for an error of `code` and `option`."""
        m = self.until(id)[-1]
        check_message(m, kind, code, option)
        return m

    def error_without_id(self, code="invalid_argument", option=None):
        for i, m in enumerate(self.backlog):
            if m["type"] == "error" and "id" not in m:
                self.backlog.pop(i)
                break
        else:
            while True:
                m = self.read()
                if m["type"] == "error" and "id" not in m:
                    break
                self.backlog.append(m)
        check_message(m, "error", code, option)
        return m

    def close(self):
        """Closes stdin, reads to the end, and checks that every request was answered and the worker exited with 0."""
        self.proc.stdin.close()
        rest = []
        while self.in_flight or self.backlog:
            if self.backlog:
                rest.append(self.backlog.pop(0))
            else:
                rest.append(self.read())
        tail = self.proc.stdout.read()
        if tail:
            raise SystemExit(f"the worker wrote after its last answer: {tail[:200]!r}")
        code = self.proc.wait(timeout=60)
        if code != 0:
            raise SystemExit(f"the worker exited with {code}")
        return rest

    def check_model_information(self, added_voices=()):
        """Checks that ready's model is `speech info --json` of the model with the device, the threads and the voices added."""
        check_model_information(self.speech, self.model, self.ready["model"], added_voices)


def check_message(m, kind=None, code=None, option=None):
    if kind and m["type"] != kind:
        raise SystemExit(f"expected {kind}, got {short(m)}")
    if code or option:
        error = m.get("error") or {}
        if code and error.get("code") != code:
            raise SystemExit(f"expected the code {code}, got {short(m)}")
        if option is not None and error.get("option") != option:
            raise SystemExit(f"expected the option {option}, got {short(m)}")
        if not isinstance(error.get("message"), str) or not error["message"]:
            raise SystemExit(f"an error without a message: {m}")


def check_model_information(speech, model, loaded, added_voices=()):
    """Checks the information of a loaded model against `speech info --json` of its file and the voices added to it."""
    r = subprocess.run([speech, "info", model, "--json"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    want = json.loads(r.stdout)
    # What the library says of an added voice is read from its file, which the model file's information cannot give.
    said = {v["name"]: v["description"] for v in loaded.get("voices", []) if isinstance(v.get("description"), str)}
    for name in added_voices:
        want["voices"].append({"name": name, "language": "", "gender": "", "description": said.get(name, "")})
        for option in want["options"]:
            if option["name"] == "voice":
                option["choices"].append(name)
    got = dict(loaded)
    if not isinstance(got.pop("device", None), str) or not isinstance(got.pop("threads", None), int):
        raise SystemExit(f"the loaded model's information lacks the device or the threads: {loaded}")
    if json.dumps(got) != json.dumps(want):
        raise SystemExit(f"the loaded model's information differs from speech info --json:\n  {json.dumps(got)[:600]}\n  {json.dumps(want)[:600]}")


def read_npy(path):
    """The float32 samples of a dump's .npy file."""
    with open(path, "rb") as f:
        data = f.read()
    header_length = struct.unpack("<H", data[8:10])[0]
    header = ast.literal_eval(data[10:10 + header_length].decode())
    assert header["descr"] == "<f4" and len(header["shape"]) == 1, header
    values = array.array("f")
    values.frombytes(data[10 + header_length:])
    return values


def pcm16(samples):
    """Float samples as 16-bit little-endian PCM, scaled as the worker scales them back (by 32768)."""
    return array.array("h", [max(-32768, min(32767, round(x * 32768))) for x in samples]).tobytes()


# general.file_type of the weight types whose texts are the reference's: F32 and F16.
REFERENCE_TYPES = (0, 1)


def dump_requests(speech, model, dump, options=()):
    """The requests of a dump of a recognition model's reference, each as (name, members, text, languages): for one of
    reference/fastconformer/dump.py, the request without options, the dump's text and no language, since FastConformer
    writes none, then, for a model that takes the option decoding (read with speech info --json), a request with each
    decoding other than the default and the text in the dump's folder of its name; for one of
    reference/qwen3-asr/dump.py, which keeps each request in a folder of its own, auto, forced, auto-prompt and
    forced-prompt, the forced language given by the tag of general.languages whose name qwen3-asr.language_names gives
    (read with speech info --json --meta), the prompt by its text, and the language qwen-asr parsed by its tag, or
    none.

    The text and languages are the reference's for weights in F32 or F16. A quantized file's rounding may change a word
    of a long recording, which points to no defect of the entry point a smoke script drives, so for one the text and
    languages are what `speech asr` of the same build gives for the dump's audio as 16-bit samples, with the request's
    members and `options`, the flags that started the worker or the server."""
    def text(folder):
        with open(os.path.join(folder, "text.txt"), encoding="utf-8") as f:
            return f.read()

    if not os.path.isfile(os.path.join(dump, "auto", "meta.json")):
        r = subprocess.run([speech, "info", model, "--json"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        decoding = [o for o in json.loads(r.stdout)["options"] if o["name"] == "decoding"]
        others = [c for o in decoding for c in o["choices"] if c != o["default"]]
        requests = [("auto", {}, text(dump), [])] + [(c, {"decoding": c}, text(os.path.join(dump, c)), []) for c in others]
        return library_texts(speech, model, dump, options, requests)
    r = subprocess.run([speech, "info", model, "--json", "--meta"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    meta = json.loads(r.stdout)["meta"]
    tags = dict(zip(meta["qwen3-asr.language_names"], meta["general.languages"]))
    requests = []
    for name in ("auto", "forced", "auto-prompt", "forced-prompt"):
        folder = os.path.join(dump, name)
        with open(os.path.join(folder, "meta.json"), encoding="utf-8") as f:
            asked = json.load(f)
        members = {}
        if asked["language"]:
            members["language"] = tags[asked["language"]]
        if asked["prompt"]:
            members["prompt"] = asked["prompt"]
        requests.append((name, members, text(folder), [tags[asked["parsed_language"]]] if asked["parsed_language"] else []))
    return library_texts(speech, model, dump, options, requests)


def recognition_defaults(options, takes):
    """The request options that the flags `options`, which started a worker or a server, set for its recognition model,
    by name, with the device and the threads: a flag of a request option the model takes (`takes`, its options from
    speech info --json, by name), alone or after the prefix asr., and --device and --threads. The flags of the process
    alone (--no-warmup, --port, --add-voice and the like) and those for the other tasks (tts., vad.) set nothing here.
    A flag of a bool option stands alone or with =false; any other takes the next word or the value after =."""
    defaults, at = {}, 0
    while at < len(options):
        flag = options[at]
        at += 1
        if not flag.startswith("--"):
            continue
        name, given, value = flag[2:].partition("=")
        name = name.replace("-", "_")
        task, dot, bare = name.partition(".")
        if dot:
            name = bare if task == "asr" else None
        option = takes.get(name)
        takes_value = (option is None or option["type"] != "bool") and not given
        if takes_value and at < len(options) and not options[at].startswith("--"):
            value = options[at]
            at += 1
        if option is not None and option["type"] == "bool":
            defaults[name] = value if given else "true"
        elif option is not None or name in ("device", "threads"):
            defaults[name] = value
    return defaults


def library_texts(speech, model, dump, options, requests):
    """`requests` as they are for weights in F32 or F16, and otherwise with the text and languages of `speech asr`, each
    with the defaults the flags `options` set, overridden by the request's own members as the worker and the server
    override them."""
    r = subprocess.run([speech, "info", model, "--json", "--meta"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    info = json.loads(r.stdout)
    if info["meta"]["general.file_type"] in REFERENCE_TYPES:
        return requests
    takes = {o["name"]: o for o in info["options"]}
    defaults = recognition_defaults(options, takes)
    rate = info["sample_rate"]
    pcm = pcm16(read_npy(os.path.join(dump, "audio.npy")))
    with tempfile.TemporaryDirectory() as temp:
        path = os.path.join(temp, "dump.wav")
        with open(path, "wb") as f:
            f.write(b"RIFF" + struct.pack("<I", 36 + len(pcm)) + b"WAVEfmt " + struct.pack("<IHHIIHH", 16, 1, 1, rate, rate * 2, 2, 16)
                    + b"data" + struct.pack("<I", len(pcm)) + pcm)
        found = []
        for name, members, _, _ in requests:
            flags = []
            for k, v in (defaults | members).items():
                flag = "--" + k.replace("_", "-")
                flags += [f"{flag}={str(v).lower()}"] if k in takes and takes[k]["type"] == "bool" else [flag, str(v)]
            r = subprocess.run([speech, "asr", model, path, "--format", "json", *flags], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if r.returncode != 0:
                raise SystemExit(f"speech asr exited with {r.returncode} on {dump}: {r.stderr.decode(errors='replace')[-800:]}")
            end = json.loads(r.stdout)
            found.append((name, members, end["text"], end.get("languages", [])))
    return found


# The blocks of the scripts written without spaces between words: Han, Hiragana and Katakana with their radicals,
# symbols, punctuation and full-width forms.
UNSPACED = ((0x2E80, 0x2FDF), (0x3000, 0x30FF), (0x31F0, 0x33FF), (0x3400, 0x4DBF), (0x4E00, 0x9FFF), (0xF900, 0xFAFF),
            (0xFF00, 0xFFEF), (0x1AFF0, 0x1B16F), (0x20000, 0x323AF))


def joined_transcript(parts, timestamps):
    """What transcription by regions answers for regions recognized alone, each given as (its offset in seconds, the
    worker's end of its samples): the texts joined with a space unless either side is written without spaces or already
    has one, the stop model_limit where any part stopped there, the languages in order with a run of one counted once,
    and with `timestamps` the segments and tokens moved by their part's offset, the space beginning the first of each."""
    def unspaced(c):
        return any(a <= ord(c) <= b for a, b in UNSPACED)

    whole = {"text": "", "stop": "complete", "languages": [], "segments": [], "tokens": []}
    for offset, end in parts:
        if end["stop"] == "model_limit":
            whole["stop"] = "model_limit"
        for language in end.get("languages", []):
            if not whole["languages"] or whole["languages"][-1] != language:
                whole["languages"].append(language)
        if not end["text"]:
            continue
        a, b = whole["text"][-1:], end["text"][0]
        space = " " if a and not (a.isspace() or b.isspace() or unspaced(a) or unspaced(b)) else ""
        whole["text"] += space + end["text"]
        if timestamps:
            for kind in ("segments", "tokens"):
                whole[kind] += [{"start": t["start"] + offset, "end": t["end"] + offset, "text": (space if i == 0 else "") + t["text"]}
                                for i, t in enumerate(end[kind])]
    if not whole["languages"]:
        del whole["languages"]
    if not timestamps:
        del whole["segments"], whole["tokens"]
    return whole


# Thirty short Japanese sentences: 276 tokens of Irodori-TTS, past the 256 it takes in a request, which it speaks in 137
# to 154 s, more than four times the 30 s of a request.
PARAGRAPH = [
    "今日は朝から雨が降っていました。", "駅までの道は水たまりだらけで、靴がすっかり濡れてしまいました。",
    "電車はいつもより混んでいて、窓の外はずっと灰色でした。", "会社に着くと、同僚が温かいお茶を入れてくれました。",
    "午後には雨が上がり、雲の間から青い空が見えてきました。", "帰り道、公園の木々が夕日に照らされて輝いていました。",
    "小さな子どもたちが水たまりで楽しそうに遊んでいました。", "家に帰ってから、久しぶりにゆっくりと本を読みました。",
    "夕方になると、風が少し冷たくなってきました。", "近くの店でパンと牛乳を買いました。",
    "友達から久しぶりに電話がかかってきました。", "来週の日曜日に一緒に山へ行く約束をしました。",
    "天気が良ければ、頂上から海が見えるそうです。", "お弁当を作って持っていくことにしました。",
    "窓の外では、虫の声が静かに聞こえていました。", "夜は早めに寝て、明日に備えることにしました。",
    "朝起きると、空はすっかり晴れていました。", "台所からは味噌汁のいい香りがしていました。",
    "妹はまだ眠そうな顔で新聞を読んでいました。", "庭の花に水をやってから、駅へ向かいました。",
    "途中の橋の上で、川の流れをしばらく眺めました。", "魚が跳ねて、小さな波が広がっていきました。",
    "会社の近くに新しい喫茶店ができていました。", "今度の休みに、家族を連れて行ってみようと思います。",
    "昼休みには、同僚と一緒に近くの公園を歩きました。", "池のそばのベンチで、鳥が羽を休めていました。",
    "午後の会議は思ったより早く終わりました。", "帰りの電車では、窓の外の景色をぼんやりと眺めていました。",
    "家の前まで来ると、隣の犬がしっぽを振って迎えてくれました。", "明日もきっと、いい一日になると思います。",
]
# The most character error rate the paragraph's speech may have as the recognizer hears it. Qwen3-ASR 0.6B heard 13
# syntheses of it by both Irodori-TTS v4.1 models, in the voice none and in a voice of a reference, at 0.47% to 1.25%
# (Apple M5, Metal, 2026-10-08), and its shortest sentence is 2.3% of it, so a sentence lost at a join fails.
MOST_CER = 0.02


def speaks_by_sentence(info):
    """Whether a synthesis model speaks a text a sentence at a time, as speech tts and speech serve speak it: one whose
    request speaks less than 60 s, by the upper bound of its option seconds or max_seconds."""
    longest = [o["maximum"] for o in info["options"] if o["name"] in ("seconds", "max_seconds")]
    return bool(longest) and min(longest) < 60


def cer(reference, hypothesis):
    """The character error rate of `hypothesis` against `reference`, both in NFKC without punctuation, symbols or spaces."""
    def plain(text):
        return [c for c in unicodedata.normalize("NFKC", text) if unicodedata.category(c)[0] not in "PSZ" and not c.isspace()]

    r, h = plain(reference), plain(hypothesis)
    previous = list(range(len(h) + 1))
    for i, a in enumerate(r, 1):
        current = [i] + [0] * len(h)
        for j, b in enumerate(h, 1):
            current[j] = min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (a != b))
        previous = current
    return previous[-1] / len(r)


def heard(speech, recognizer, wav_path, load_options):
    """The text `speech asr` writes of a WAVE file with the recognition model `recognizer`."""
    r = subprocess.run([speech, "asr", recognizer, *load_options, wav_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if r.returncode != 0:
        raise SystemExit(f"speech asr {recognizer} exited with {r.returncode}: {r.stderr.decode()[-600:]}")
    return r.stdout.decode().strip()


def short(m):
    m = {k: v for k, v in m.items() if k != "_at"}
    if "pcm" in m:
        m["pcm"] = f"<{len(m['pcm'])} base64 bytes>"
    text = json.dumps(m, ensure_ascii=False)
    return text if len(text) < 400 else text[:400] + "..."
