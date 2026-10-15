"""Drives `speech worker` with a synthesis model through its protocol the way a caller does, with every line checked by
worker_client.py: one JSON object per line, and one terminal message per request and nothing after it.

It checks ready (the release, and the model information of `speech info --json` with the device, the
threads and the voices of --add-voice), info and count_tokens, also answered while a synthesis runs; that a seed repeats the audio and a drawn seed is
reported and repeats it too; cancels while a request waits, while it runs, of an unknown id, and after an end, whose id
a later request then reuses; a second request under an id in flight; each error with its code and option (members the
message does not have, values of the wrong type, out of range or not taken, a missing text, lines that are not JSON
objects or have no id or type, the other task's messages); chunks to a synthesis model; add_voice, and a recording
refused by a model that takes voice files alone; for
Irodori-TTS a fixed length, the progress of a long sampler, a paragraph past what one request speaks spoken in one
synthesize, whose reported seed repeats it, and a run with no place to cut refused, naming it, for Drama 3 max_seconds,
the sampling options' ranges and a paragraph of several of its chunks spoken in one synthesize, and for Qwen3-TTS a
text past its tokens refused, max_seconds and the sampling options, and an instruction where the model takes one and its
refusal where it does not; and that a request option given to the worker as a flag is the default of a request that
does not set it, while one the model does not take or out of its range ends the worker with fatal. Writes the first
answer to a WAV.

usage: python3 checks/smoke/worker_smoke.py <out.wav> <speech> <model.gguf> [worker options...]
       A model that takes voice files needs --add-voice NAME=FILE, whose FILE add_voice adds again under another name.
"""

import base64
import json
import os
import subprocess
import sys
import tempfile
import time
import wave

from worker_client import PARAGRAPH, Worker, short, speaks_by_sentence

out_wav, speech, model, *options = sys.argv[1:]
added = [o.split("=", 1) for i, o in enumerate(options) if i > 0 and options[i - 1] == "--add-voice"]
w = Worker(speech, model, options)
ready = w.ready
info = ready["model"]
assert isinstance(ready["version"], str) and info["task"] == "synthesis", short(ready)
w.check_model_information([name for name, _ in added])
rate = info["sample_rate"]
# An added voice where there is one: the voice an Irodori-TTS file has of its own, none, speaks without a reference.
voice = added[0][0] if added else info["voices"][0]["name"]
own = [v["name"] for v in info["voices"]][: len(info["voices"]) - len(added)]
irodori = info["architecture"] == "irodori-tts"
drama3 = info["architecture"] == "drama3"
print(f"ready in {time.perf_counter() - w.started:.2f} s: {info['name']} on {info['device']}, protocol {ready['protocol']}, speech.cpp {ready['version']}, "
      f"{len(info['voices'])} voices, model information equal to speech info --json")

TEXT = "明日の東京は晴れで、最高気温は二十四度の予報です。"
LONG = "これは途中で止める長めの文です。止まったら終わりの知らせの代わりに取り消しの知らせが来ます。"


def speak(id, text=TEXT, **members):
    """The audio and the end of a synthesize."""
    w.request({"type": "synthesize", "id": id, "text": text, "voice": voice, **members})
    pcm = bytearray()
    for m in w.until(id):
        if m["type"] == "chunk":
            pcm += base64.b64decode(m["pcm"])
    if m["type"] != "end":
        raise SystemExit(f"{id}: {short(m)}")
    assert m["samples"] * 2 == len(pcm) and m["stop"] in ("complete", "max_seconds", "model_limit"), short(m)
    return bytes(pcm), m


def expect_error(message, code, option):
    w.request(message)
    m = w.terminal(message["id"], "error", code, option)
    print(f"{message['id']}: {code} ({option}) as expected: {m['error']['message']}")


w.request({"type": "info", "id": "info"})
assert w.terminal("info", "end")["model"] == info, "info's model differs from ready's"
w.request({"type": "count_tokens", "id": "count", "text": TEXT})
count = w.terminal("count", "end")
assert isinstance(count["tokens"], int) and 0 < count["tokens"] <= info["max_text_tokens"], short(count)
print(f"info: the model of ready; count_tokens: {count['tokens']} tokens")

t0 = time.perf_counter()
pcm_a, end_a = speak("a", seed=11)
assert end_a["seed"] == 11 and end_a["stop"] == "complete", short(end_a)
pcm_a2, _ = speak("a2", seed=11)
assert pcm_a == pcm_a2, "the same seed gave other audio"
print(f"a: {len(pcm_a) // 2 / rate:.2f} s of audio in {time.perf_counter() - t0:.2f} s, seed 11; the same seed repeats it byte for byte")
pcm_d, end_d = speak("d")
assert 0 <= end_d["seed"] < 2 ** 53, short(end_d)
assert speak("d2", seed=end_d["seed"])[0] == pcm_d, "the reported seed did not repeat the audio"
print(f"d: drew the seed {end_d['seed']}, which repeats the audio")

# A cancel while the request runs ends it with cancelled; one while it waits, at once, with no chunk.
w.request({"type": "synthesize", "id": "b", "text": LONG, "voice": voice})
while (m := w.next_for("b"))["type"] == "progress":
    pass
assert m["type"] == "chunk", short(m)
w.send({"type": "cancel", "id": "b"})
w.request({"type": "synthesize", "id": "c", "text": "三つ目です。", "voice": voice})
w.request({"type": "synthesize", "id": "c2", "text": "四つ目です。", "voice": voice})
w.send({"type": "cancel", "id": "c2"})
b = w.until("b")[-1]["type"]
c2 = w.until("c2")
assert [m["type"] for m in c2] == ["cancelled"], [short(m) for m in c2]
w.terminal("c", "end")
assert b == "cancelled", b
print("b: cancelled while it ran; c2: cancelled while it waited, with no chunk; c answered")

# A cancel of an id no request has, and one after its request ended, change nothing later: the id is reused.
w.send({"type": "cancel", "id": "nothing"})
w.send({"type": "cancel", "id": "a"})
assert speak("a", seed=11)[0] == pcm_a, "the reused id gave other audio"
print("a cancel of an unknown id and one after the end changed nothing; the id a was answered again")

# A request under an id in flight is refused without an id, and the request in flight is answered.
w.request({"type": "synthesize", "id": "x", "text": LONG, "voice": voice})
w.send({"type": "synthesize", "id": "x", "text": "二つ目。", "voice": voice})
m = w.error_without_id("invalid_argument", "id")
w.terminal("x", "end")
print(f"a second request under x: an error without an id ({m['error']['message'][:60]}...); x answered")

# count_tokens and info read the model's information alone, so they are answered while a synthesis runs.
w.request({"type": "synthesize", "id": "busy", "text": LONG, "voice": voice})
while (m := w.next_for("busy"))["type"] == "progress":
    pass
assert m["type"] == "chunk", short(m)
w.request({"type": "count_tokens", "id": "busy-count", "text": TEXT})
w.request({"type": "info", "id": "busy-info"})
ended = []
while "busy" not in ended:
    m = w.read()
    if m["type"] in ("end", "error", "cancelled"):
        assert m["type"] == "end", short(m)
        ended.append(m["id"])
        if m["id"] == "busy-count":
            assert m["tokens"] == count["tokens"], short(m)
        if m["id"] == "busy-info":
            assert m["model"] == info, short(m)
assert ended == ["busy-count", "busy-info", "busy"], ended
print("count_tokens and info sent while a synthesis ran: answered before it ended")

expect_error({"type": "synthesize", "id": "e1", "text": "あ。", "voice": voice, "bogus": 1}, "invalid_argument", "bogus")
expect_error({"type": "synthesize", "id": "e2", "text": "あ。", "voice": voice, "speed": "fast"}, "invalid_argument", "speed")
expect_error({"type": "synthesize", "id": "e3", "text": "あ。", "voice": voice, "seed": 1.5}, "invalid_argument", "seed")
expect_error({"type": "synthesize", "id": "e4", "text": "あ。", "voice": "no-such-voice"}, "out_of_range", "voice")
expect_error({"type": "synthesize", "id": "e5", "voice": voice}, "invalid_argument", "text")
expect_error({"type": "synthesize", "id": "e6", "text": "", "voice": voice}, "invalid_argument", "text")
expect_error({"type": "synthesize", "id": "e7", "text": "あ。"}, "invalid_argument", "voice")
expect_error({"type": "synthesize", "id": "e8", "text": "あ。", "voice": voice, "timestamps": True}, "unsupported", "timestamps")
if speaks_by_sentence(info):
    # A text the library refuses as too long is spoken in pieces, cut after its sentences, under the seed the first
    # piece drew; a run with no place to cut is refused, naming it.
    t0 = time.perf_counter()
    pcm_p, end_p = speak("paragraph", text="".join(PARAGRAPH))
    assert end_p["stop"] == "complete", short(end_p)
    assert speak("paragraph2", text="".join(PARAGRAPH), seed=end_p["seed"])[0] == pcm_p, "the reported seed did not repeat the paragraph"
    print(f"paragraph: {len(PARAGRAPH)} sentences past one request, {len(pcm_p) // 2 / rate:.2f} s of audio in "
          f"{time.perf_counter() - t0:.2f} s for two runs, the reported seed repeating it byte for byte")
    run_text = "あいうえおかきくけこ" * 80
    w.request({"type": "synthesize", "id": "e9", "text": run_text, "voice": voice})
    message = w.terminal("e9", "error", "out_of_range", "text")["error"]["message"]
    assert message.startswith('"' + run_text[:20]), message
    print(f"e9: a run of {len(run_text)} characters with no place to cut, refused: {message[:100]}")
else:
    too_long, tokens = TEXT, count["tokens"]
    while tokens <= info["max_text_tokens"]:
        too_long += too_long
        w.request({"type": "count_tokens", "id": "count-long", "text": too_long})
        tokens = w.terminal("count-long", "end")["tokens"]
    expect_error({"type": "synthesize", "id": "e9", "text": too_long, "voice": voice}, "out_of_range", "text")
expect_error({"type": "pause", "id": "e10"}, "invalid_argument", "type")
expect_error({"type": "add_voice", "id": "e11", "name": "x"}, "invalid_argument", "path")
# A member set to null counts as left out, and an option at its neutral value is taken by every model.
speak("n", text="あ。", language=None, speed=1, duration_scale=1, timestamps=False, instructions="")
print("n: null, a speed and a scale of 1, timestamps false and empty instructions were taken")
if irodori:
    expect_error({"type": "synthesize", "id": "e12", "text": "あ。", "voice": voice, "speed": 5}, "out_of_range", "speed")
    expect_error({"type": "synthesize", "id": "e13", "text": "あ。", "voice": voice, "max_seconds": 1}, "unsupported", "max_seconds")
    expect_error({"type": "synthesize", "id": "e14", "text": "あ。", "voice": voice, "seconds": 2, "duration_scale": 1.5},
                 "invalid_argument", "seconds")
    expect_error({"type": "synthesize", "id": "e15", "text": "あ。", "voice": voice, "temperature": 0.5}, "unsupported", "temperature")
    pcm, end = speak("s", text="三つ目です。", seconds=1)
    assert end["samples"] <= rate, short(end)
    print(f"s: a length of 1 s gave {end['samples'] / rate:.3f} s")
    # A sampler of many steps passes no audio for seconds, and the worker reports its progress at most once a second.
    w.request({"type": "synthesize", "id": "p", "text": "あ。", "voice": voice, "steps": 160})
    messages = w.until("p")
    progress = [m for m in messages if m["type"] == "progress"]
    assert messages[-1]["type"] == "end" and progress, [short(m) for m in messages if m["type"] != "chunk"]
    done = [m["done"] for m in progress]
    gaps = [b["_at"] - a["_at"] for a, b in zip(progress, progress[1:])]
    assert done == sorted(done) and 0 <= done[0] and done[-1] <= 1 and all(g > 0.9 for g in gaps), (done, gaps)
    print(f"p: 160 steps, {len(progress)} progress messages ({', '.join(f'{d:.2f}' for d in done)}) at least a second apart")
    # The official runtime's settings that the file declares, each taken once and refused where it has no effect.
    takes = {o["name"] for o in info["options"]}
    speak("k", text="はい。", keep_tail=True, seconds=2)
    expect_error({"type": "synthesize", "id": "e15", "text": "あ。", "voice": voice, "keep_tail": True, "tail_window_size": 5},
                 "invalid_argument", "tail_window_size")
    if "cfg_scale_text" in takes:
        speak("g", text="はい。", steps=4, cfg_guidance_mode="joint", cfg_scale_text=4, cfg_scale_speaker=4, speaker_kv_scale=1.5)
        expect_error({"type": "synthesize", "id": "e16", "text": "あ。", "voice": voice, "cfg_guidance_mode": "joint", "cfg_scale_text": 3,
                      "cfg_scale_speaker": 5}, "invalid_argument", "cfg_guidance_mode")
        # The largest scale the range takes, whose guidance overflows the float32, gives speech that is not finite, refused
        # naming no option.
        largest = next(o["maximum"] for o in info["options"] if o["name"] == "cfg_scale_text")
        expect_error({"type": "synthesize", "id": "e19", "text": "あ。", "voice": voice, "steps": 4, "cfg_scale_text": largest}, "out_of_range",
                     None)
    if "none" in [v["name"] for v in info["voices"]]:
        w.request({"type": "synthesize", "id": "none", "text": "はい。", "voice": "none"})
        w.terminal("none", "end")
    if "instructions" in takes:
        expect_error({"type": "synthesize", "id": "e17", "text": "あ。", "voice": voice, "instructions": "声、" * 600}, "out_of_range",
                     "instructions")
    print("k, g and none: the runtime's settings the file declares were taken, and settings without effect and a caption past the "
          "file's longest refused")
elif drama3:
    expect_error({"type": "synthesize", "id": "e12", "text": "あ。", "voice": voice, "speed": 1.5}, "unsupported", "speed")
    expect_error({"type": "synthesize", "id": "e13", "text": "あ。", "voice": voice, "seconds": 2}, "unsupported", "seconds")
    expect_error({"type": "synthesize", "id": "e14", "text": "あ。", "voice": voice, "steps": 4}, "unsupported", "steps")
    pcm, end = speak("s", text=LONG, max_seconds=0.5)
    assert end["stop"] == "max_seconds" and end["samples"] <= 0.5 * rate, short(end)
    print(f"s: max_seconds 0.5 stopped it at {end['samples'] / rate:.3f} s ({end['stop']})")
    expect_error({"type": "synthesize", "id": "e15", "text": "あ。", "voice": voice, "temperature": 1e-50}, "out_of_range", "temperature")
    expect_error({"type": "synthesize", "id": "e16", "text": "あ。", "voice": voice, "top_k": 0}, "out_of_range", "top_k")
    expect_error({"type": "synthesize", "id": "e17", "text": "あ。", "voice": voice, "top_p": 1.5}, "out_of_range", "top_p")
    expect_error({"type": "synthesize", "id": "e19", "text": "あ。", "voice": voice, "do_sample": False}, "unsupported", "do_sample")
    # A paragraph of more bytes than a chunk takes is spoken in one synthesize, chunk after chunk.
    paragraph = "".join(PARAGRAPH[:6])
    pcm, end = speak("pa", text=paragraph, seed=5)
    assert end["stop"] == "complete" and end["samples"] > 12 * rate, short(end)
    print(f"pa: a paragraph of {len(paragraph.encode())} bytes gave {end['samples'] / rate:.1f} s in one synthesize ({end['stop']})")
else:
    expect_error({"type": "synthesize", "id": "e12", "text": "あ。", "voice": voice, "speed": 1.5}, "unsupported", "speed")
    expect_error({"type": "synthesize", "id": "e13", "text": "あ。", "voice": voice, "seconds": 2}, "unsupported", "seconds")
    expect_error({"type": "synthesize", "id": "e14", "text": "あ。", "voice": voice, "steps": 4}, "unsupported", "steps")
    pcm, end = speak("s", text=LONG, max_seconds=0.5)
    assert end["stop"] == "max_seconds" and end["samples"] <= 0.5 * rate, short(end)
    print(f"s: max_seconds 0.5 stopped it at {end['samples'] / rate:.3f} s ({end['stop']})")
    # Neither stack draws, so the seed changes nothing; a temperature of a stack that does not draw is refused.
    greedy = dict(text="あ。", do_sample=False, code_predictor_do_sample=False, max_seconds=2)
    assert speak("g1", seed=1, **greedy)[0] == speak("g2", seed=2, **greedy)[0], "a request that draws nothing changed with the seed"
    print("g1, g2: do_sample and code_predictor_do_sample false gave the same audio for the seeds 1 and 2")
    expect_error({"type": "synthesize", "id": "e15", "text": "あ。", "voice": voice, "do_sample": False, "temperature": 0.5},
                 "invalid_argument", "temperature")
    expect_error({"type": "synthesize", "id": "e16", "text": "あ。", "voice": voice, "code_predictor_top_k": -1}, "out_of_range",
                 "code_predictor_top_k")
    expect_error({"type": "synthesize", "id": "e17", "text": "あ。", "voice": voice, "do_sample": "no"}, "invalid_argument", "do_sample")
# A model that takes an instruction speaks the same seed otherwise with one, and one that takes none refuses it.
if any(o["name"] == "instructions" for o in info["options"]):
    told = dict(text="あ。", seed=3)
    assert speak("i1", **told)[0] != speak("i2", instructions="怒った口調で話してください。", **told)[0], "an instruction changed nothing"
    print("i1, i2: an instruction gave other audio for the same seed")
else:
    expect_error({"type": "synthesize", "id": "e18", "text": "あ。", "voice": voice, "instructions": "怒った口調で"}, "unsupported",
                 "instructions")

# The other task's messages: a chunk opens a recognition request answered unsupported, whose later lines are dropped.
w.request({"type": "chunk", "id": "r", "seq": 0, "pcm": ""})
w.terminal("r", "error", "unsupported", "type")
w.send({"type": "chunk", "id": "r", "seq": 1, "pcm": ""})
w.send({"type": "transcribe", "id": "r", "sample_rate": 16000})
w.request({"type": "transcribe", "id": "r3", "sample_rate": 16000})
w.terminal("r3", "error", "unsupported", "type")
print("chunk and transcribe: unsupported once, the request's later lines dropped")

# Lines that name no request are answered without an id.
for line in ["this is not JSON", '["id", "h"]', '{"type": "synthesize", "text": "あ。"}', '{"type": "synthesize", "id": 5}',
             '{"type": "info", "id": ""}', '{"type": "cancel", "id": "a", "now": true}']:
    w.send_line(line)
    m = w.error_without_id()
    print(f"{line[:44]!r}: an error without an id: {m['error']['message'][:90]}")

if info["voice_files"]:
    name, path = added[0]
    w.request({"type": "add_voice", "id": "v", "name": "added", "path": path})
    w.terminal("v", "end")
    w.request({"type": "info", "id": "v-info"})
    assert [v["name"] for v in w.terminal("v-info", "end")["model"]["voices"]] == own + [n for n, _ in added] + ["added"]
    voice = "added"
    speak("v-speak", text="あ。")
    expect_error({"type": "add_voice", "id": "v2", "name": "added", "path": path}, "invalid_argument", "name")
    expect_error({"type": "add_voice", "id": "v3", "name": "other", "path": "/no/such/voice.gguf"}, "io", "path")
    if not info["voice_recordings"]:
        with tempfile.TemporaryDirectory() as folder:
            recording = os.path.join(folder, "recording.wav")
            with wave.open(recording, "wb") as f:
                f.setnchannels(1)
                f.setsampwidth(2)
                f.setframerate(rate)
                f.writeframes(bytes(2 * rate))
            expect_error({"type": "add_voice", "id": "v4", "name": "other", "path": recording}, "invalid_argument", "path")
    print("add_voice: the voice was added, info lists it, and a synthesis speaks with it")
else:
    expect_error({"type": "add_voice", "id": "v", "name": "x", "path": "x.wav"}, "unsupported", None)

# Requests queued when stdin closes are answered before the worker exits with 0.
w.request({"type": "synthesize", "id": "last", "text": "おしまい。", "voice": voice})
rest = w.close()
assert rest[-1]["type"] == "end" and rest[-1]["id"] == "last", short(rest[-1])

# A request option given to the worker as a flag is the default of every request that does not set it, and one the model
# does not take, or a value out of its range, ends the worker with fatal before ready.
assert ready["defaults"] == {}, ready["defaults"]
w = Worker(speech, model, [*options, "--seed", "11"])
assert w.ready["model"] == info and w.ready["defaults"] == {"seed": 11}, "ready's defaults or model information"
w.request({"type": "info", "id": "defaults-info"})
assert w.terminal("defaults-info", "end")["defaults"] == {"seed": 11}, "info's defaults"
voice = added[0][0] if added else info["voices"][0]["name"]
pcm_default, end_default = speak("default")
assert end_default["seed"] == 11 and pcm_default == pcm_a, "a request without a seed did not speak with the worker's --seed 11"
assert speak("own", seed=5)[1]["seed"] == 5, "a request's own seed did not win over the worker's --seed"
w.close()
print("--seed 11: in ready's and info's defaults, a request without a seed spoke seed 11's audio, and one with seed 5 kept it; "
      "ready's model information unchanged")
for flags, code, option in [(["--timestamps"], "unsupported", "timestamps"), (["--seed", "-1"], "out_of_range", "seed")]:
    run = subprocess.run([speech, "worker", model, *options, *flags], stdin=subprocess.DEVNULL, capture_output=True, text=True)
    first = json.loads(run.stdout.splitlines()[0])
    assert run.returncode != 0 and first["type"] == "fatal", (run.returncode, run.stdout[:300])
    assert (first["error"]["code"], first["error"]["option"]) == (code, option), short(first)
    print(f"{' '.join(flags)}: fatal {code} ({option}) before ready: {first['error']['message'][:90]}")

with wave.open(out_wav, "wb") as f:
    f.setnchannels(1)
    f.setsampwidth(2)
    f.setframerate(rate)
    f.writeframes(pcm_a)
print("ok: every request had one terminal message; the worker answered the last after stdin closed and exited with 0")
