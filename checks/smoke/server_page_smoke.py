"""Checks `speech serve` with a model of each task and its page: the page's files, its endpoints under /speech/, the
protections around them, the OpenAI endpoints with two models held, and the replacement of a model by the page.

The model files are local copies of the catalog's files, found by their size and SHA-256 and linked into a model
folder of the script's own (SPEECH_MODEL_DIR), so that loading one by its name fetches nothing. It checks:

- a server given no model and no --open, and --open on another address than a loopback one: exit 2;
- the page: GET / with its Content-Security-Policy, every file it and its modules name, a file it does not have, and a
  Host that is not 127.0.0.1, localhost or [::1] with the server's port, refused;
- the page's endpoints, GET /speech/models, POST /speech/load and POST /speech/voices, from a foreign Origin or for a
  foreign Host (403); the catalog as `speech models --json` gives it and the models held, from the server's own origin
  as well as from none, with no token;
- every endpoint refusing a foreign Origin, the OpenAI ones included, preflights too, while a request without one (curl,
  scripts) and one from --cors-origin's origin pass;
- the OpenAI endpoints with the two models: /v1/models lists both, speech and a transcription of that speech, and a
  request naming the model of the other task refused;
- a load of a name outside the catalog (404), of a model file's path (400) and of a body it does not take (400);
- switching: the page loads the third model in place of its task's, a second load of that task meanwhile is refused
  (409), the endpoint of that task serves the new model while the other task's stays, and the first model comes back
  by its catalog name;
- a fetch the page stops: the load of a model not in the folder, closed at its first progress, leaves a part and the
  model held as it was (a few megabytes come from Hugging Face; the part is removed after);
- a load whose page goes away while it waits for another process's lock on the file, which that process then puts in
  place, leaves the model held as it was;
- a voice added to a model that takes voice files, or refused by one that does not;
- with a recognition model as the third, two Realtime sessions at /v1/realtime through the page's replacement of the
  recognition model: a commit while the new model loads fails with model_loading, a session that names no model goes on
  with the new one, and one that named the old model, in session.update or in the address's query, fails with
  model_not_found;
- with a detection model, the place of detection, empty until the page loads that model into it by its catalog name,
  a transcription with chunking_strategy refused before and answered after, /v1/models listing it third, and its
  replacement while a Realtime session with server_vad holds it, which the load does not wait for and after which the
  session goes on with the new model;
- a server on 0.0.0.0, which has no page and answers its endpoints with how to reach them (on Windows its firewall may
  ask about it).

usage: python3 checks/smoke/server_page_smoke.py <speech> <work dir> <synthesis.gguf with voices of its own> <recognition.gguf> <third model.gguf>
                                                 [detection.gguf]
"""

import array
import base64
import hashlib
import http.client
import json
import os
import re
import shutil
import struct
import subprocess
import sys
import threading
import time
import uuid

from server_client import Server, expect_error

if len(sys.argv) not in (6, 7):
    raise SystemExit(__doc__.strip().splitlines()[-1])
speech, work, *files = sys.argv[1:]
folder = os.path.join(os.path.abspath(work), "page-smoke-models")
shutil.rmtree(folder, ignore_errors=True)
os.makedirs(folder)
env = {**os.environ, "SPEECH_MODEL_DIR": folder}
ALLOWED = "http://localhost:5173"
FOREIGN = "http://evil.example"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(16 << 20):
            h.update(chunk)
    return h.hexdigest()


catalog = json.loads(subprocess.run([speech, "models", "--json"], env=env, capture_output=True, check=True).stdout)
given = []
for path in files:
    size, digest = os.path.getsize(path), sha256(path)
    found = [(m, f) for m in catalog["models"] for f in m["files"] if f["size"] == size and f["sha256"] == digest]
    if not found:
        raise SystemExit(f"{path} is no file of the catalog")
    m, f = found[0]
    os.makedirs(os.path.dirname(f["path"]), exist_ok=True)
    os.symlink(os.path.abspath(path), f["path"])
    given.append({"path": path, "name": m["name"] if f["type"] == m["type"] else f"{m['name']}:{f['type']}", "task": m["task"]})
synthesis, recognition, third = given[:3]
detection = given[3] if len(given) > 3 else None
assert synthesis["task"] == "synthesis" and recognition["task"] == "recognition", "give a synthesis model, then a recognition model"
assert not detection or detection["task"] == "detection", "the fourth model is a detection model"
print(f"linked {', '.join(g['name'] for g in given)} into {folder}")

r = subprocess.run([speech, "serve", "--port", "0"], env=env, capture_output=True)
starting = {m["name"] for m in catalog["models"] if m["start"]}
assert r.returncode == 2 and all(n in r.stderr.decode() for n in starting), r.stderr
r = subprocess.run([speech, "serve", "--open", "--host", "0.0.0.0", "--port", "0"], env=env, capture_output=True)
assert r.returncode == 2, r.stderr
print("no model and no --open, and --open on 0.0.0.0: exit 2")

server = Server(speech, [synthesis["path"], recognition["path"], "--cors-origin", ALLOWED], env)
call, post_json, port = server.call, server.post_json, server.port
own = f"http://127.0.0.1:{port}"
assert server.page == f"{own}/", server.page

status, headers, page = call("GET", "/")
assert status == 200 and headers["content-type"].startswith("text/html"), (status, headers)
assert "frame-ancestors 'none'" in headers["content-security-policy"] and headers["referrer-policy"] == "no-referrer", headers
named, seen = set(re.findall(r'(?:href|src)="(/page/[a-z0-9-]+\.(?:js|css))"', page.decode())), set()
while named - seen:
    path = sorted(named - seen)[0]
    seen.add(path)
    status, headers, body = call("GET", path)
    assert status == 200 and headers["content-type"].startswith("text/css" if path.endswith(".css") else "text/javascript"), (path, status)
    named |= {"/page/" + n for n in re.findall(r"""(?:from |import\(|addModule\()'(?:\./|/page/)([a-z0-9-]+\.js)'""", body.decode())}
expect_error(call("GET", "/page/no-such-file.js"), 404, None, None, "a file the page does not have")
for host in ("evil.example", f"evil.example:{port}", f"127.0.0.1:{port + 1}"):
    expect_error(call("GET", "/", headers={"Host": host}), 403, "host_not_allowed", None, f"the page for the Host {host}")
print(f"the page and the {len(seen)} files it and its modules name")

for method, path in (("GET", "/speech/models"), ("POST", "/speech/load"), ("POST", "/speech/voices")):
    expect_error(call(method, path, headers={"Origin": FOREIGN}), 403, "origin_not_allowed", None, f"{method} {path} from {FOREIGN}")
    expect_error(call(method, path, headers={"Host": f"evil.example:{port}"}), 403, "host_not_allowed", None,
                 f"{method} {path} for a foreign Host")
for origin in (None, own, f"http://localhost:{port}"):
    status, _, body = call("GET", "/speech/models", headers={"Origin": origin} if origin else {})
    assert status == 200, (origin, body)
state = json.loads(body)
assert state["catalog"]["models"] == json.loads(subprocess.run([speech, "models", "--json"], env=env, capture_output=True).stdout)["models"]
for g in (synthesis, recognition):
    held = state[g["task"]]
    assert held["replacing"] is None and held["held"]["name"] is None and held["held"]["path"] == g["path"], held
assert state["detection"] == {"held": None, "replacing": None}, state["detection"]
print("the Host and the Origin of /speech/: foreign ones refused; the catalog and the two models given, with no token")

expect_error(call("GET", "/health", headers={"Origin": FOREIGN}), 403, "origin_not_allowed", None, "GET /health from a foreign origin")
expect_error(call("OPTIONS", "/v1/audio/speech", headers={"Origin": FOREIGN, "Access-Control-Request-Method": "POST"}), 403,
             "origin_not_allowed", None, "a preflight from a foreign origin")
status, _, body = call("GET", "/v1/models")
listed = json.loads(body)["data"]
assert status == 200 and [m["speech"]["task"] for m in listed] == ["synthesis", "recognition"], body
info = {m["speech"]["task"]: m["speech"] for m in listed}


def speak(text="明日の東京は晴れです。", headers=None):
    voices = info["synthesis"]["voices"]
    member = {"input": text, "seed": 5} | ({"voice": voices[0]["name"]} if voices else {})
    return post_json("/v1/audio/speech", member, headers)


def transcribe(wav, headers=None, chunking=False):
    boundary = uuid.uuid4().hex
    body = f"--{boundary}\r\nContent-Disposition: form-data; name=\"chunking_strategy\"\r\n\r\nauto\r\n".encode() if chunking else b""
    body += (f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"x.wav\"\r\nContent-Type: audio/wav\r\n\r\n").encode()
    body += wav + f"\r\n--{boundary}--\r\n".encode()
    return call("POST", "/v1/audio/transcriptions", body, {"Content-Type": f"multipart/form-data; boundary={boundary}", **(headers or {})})


def check_tasks():
    """Speech from the synthesis model, its text from the recognition model, and requests from origins."""
    if not info["synthesis"]["voices"]:
        return None
    status, headers, wav = speak()
    assert status == 200 and headers["content-type"] == "audio/wav", wav[:200]
    status, _, body = transcribe(wav)
    assert status == 200 and json.loads(body)["text"], body
    expect_error(speak(headers={"Origin": FOREIGN}), 403, "origin_not_allowed", None, "speech from a foreign origin")
    status, headers, _ = speak(headers={"Origin": ALLOWED})
    assert status == 200 and headers["access-control-allow-origin"] == ALLOWED, headers
    return wav, json.loads(body)["text"]


def check_voice(wav):
    """A voice made from a recording, added to a synthesis model that takes recordings and refused by another."""
    boundary = uuid.uuid4().hex
    form = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"name\"\r\n\r\nsmoke\r\n--{boundary}\r\nContent-Disposition: form-data; "
            f"name=\"file\"; filename=\"x.wav\"\r\n\r\n").encode() + wav + f"\r\n--{boundary}--\r\n".encode()
    got = call("POST", "/speech/voices", form, {"Content-Type": f"multipart/form-data; boundary={boundary}"})
    if info["synthesis"]["voice_recordings"]:
        assert got[0] == 200 and "smoke" in [v["name"] for v in json.loads(got[2])["held"]["model"]["voices"]], got[2][:300]
        info["synthesis"] = json.loads(got[2])["held"]["model"]
        print(f"a voice made from a recording added to {info['synthesis']['name']}")
    elif info["synthesis"]["voice_files"]:
        expect_error(got, 400, "invalid_value", "path", f"a recording for {info['synthesis']['name']}, which takes voice files alone")
    else:
        expect_error(got, 400, "unsupported_parameter", None, f"a voice for {info['synthesis']['name']}, which takes no voice files")


wav, text = check_tasks()
check_voice(wav)
expect_error(post_json("/v1/audio/speech", {"model": info["recognition"]["name"], "input": "あ。"}), 404, "model_not_found", "model",
             "speech naming the recognition model")
print(f"two models: /v1/models lists both, speech of {len(wav) - 44} bytes and its text {text!r}, from no origin and --cors-origin's")

expect_error(post_json("/speech/load", {"model": "no-such-model"}), 404, "model_not_found", "model", "a load outside the catalog")
expect_error(post_json("/speech/load", {"model": f"{recognition['name'].split(':')[0]}:q9_9"}), 404, "model_not_found", "model",
             "a load of a type the catalog does not hold")
expect_error(post_json("/speech/load", {"model": "/tmp/x.gguf"}), 400, "invalid_value", "model", "a load of a model file")
expect_error(call("POST", "/speech/load", b"{not json"), 400, "invalid_value", "model", "a load that is not JSON")
expect_error(post_json("/speech/load", {"model": third["name"], "path": "/tmp"}), 400, "invalid_value", "model", "a load with another member")


def load(name, task):
    r = call("POST", "/speech/load", json.dumps({"model": name}).encode(), {"Content-Type": "application/json"}, stream=True)
    assert r.status == 200 and r.getheader("Content-Type") == "text/event-stream", r.status
    expect_error(post_json("/speech/load", {"model": name}), 409, "model_loading", "model", f"a second load of {task} meanwhile")
    events = [json.loads(block[6:]) for block in r.read().decode().split("\n\n") if block]
    assert [e["type"] for e in events][-2:] == ["load", "loaded"], events
    held = events[-1]["held"]
    assert events[-1]["task"] == task and held["name"] == name and held["model"]["task"] == task, events[-1]
    state = json.loads(call("GET", "/speech/models")[2])
    assert state[task]["held"]["name"] == name and state[task]["replacing"] is None, state[task]
    return held


task, other = third["task"], "recognition" if third["task"] == "synthesis" else "synthesis"
before = json.loads(call("GET", "/speech/models")[2])[other]["held"]
held = load(third["name"], task)
info[task] = held["model"]
if task == "synthesis":
    check_voice(wav)
state = json.loads(call("GET", "/speech/models")[2])
assert state[other]["held"] == before, "the model of the other task changed"
status, _, body = call("GET", "/v1/models")
assert {m["id"] for m in json.loads(body)["data"]} == {info["synthesis"]["name"], info["recognition"]["name"]}, body
check_tasks()
back = {"synthesis": synthesis, "recognition": recognition}[task]
info[task] = load(back["name"], task)["model"]
check_tasks()
print(f"switching: {third['name']} in place of the {task} model, a second load meanwhile refused, both tasks served; "
      f"{back['name']} back by its name")

if task == "recognition":
    # Realtime sessions through the page's replacement of the recognition model: a commit while the new model loads fails
    # with model_loading, a session that names no model goes on with the new one, and one that named the old fails.
    def pcm24(wave):
        channels, at = struct.unpack("<HI", wave[22:28])
        values = array.array("h", wave[44:])
        n = len(values) * 24000 // at
        return array.array("h", [values[min(i * at // 24000, len(values) - 1)] for i in range(n)]).tobytes()

    def commit(ws, pcm):
        for start in range(0, len(pcm), 9600):
            ws.send({"type": "input_audio_buffer.append", "audio": base64.b64encode(pcm[start:start + 9600]).decode()})
        ws.send({"type": "input_audio_buffer.commit"})
        assert ws.event()["type"] == "input_audio_buffer.committed"
        while True:
            e = ws.event()
            if e["type"].endswith((".completed", ".failed")):
                return e

    audio = pcm24(wav)
    named, free = server.websocket("/v1/realtime"), server.websocket("/v1/realtime")
    queried = server.websocket(f"/v1/realtime?model={info['recognition']['name']}")
    assert named.event()["type"] == free.event()["type"] == queried.event()["type"] == "session.created"
    named.send({"type": "session.update", "session": {"type": "transcription", "audio": {"input": {"transcription": {"model": info["recognition"]["name"]}}}}})
    assert named.event()["type"] == "session.updated"
    # A long transcription keeps the old model busy, so that the new one is loaded only once it ends and the commit below
    # falls while the new model loads, however fast that load is.
    channels, at = struct.unpack("<HI", wav[22:28])
    data = wav[44:] * max(1, 300 * at * 2 // len(wav[44:]))
    long = wav[:4] + struct.pack("<I", 36 + len(data)) + wav[8:40] + struct.pack("<I", len(data)) + data
    busy = threading.Thread(target=lambda: transcribe(long))
    busy.start()
    time.sleep(0.5)
    r = call("POST", "/speech/load", json.dumps({"model": third["name"]}).encode(), {"Content-Type": "application/json"}, stream=True)
    buffer = b""
    while b'"type":"load"' not in buffer:
        chunk = r.read1(4096)
        assert chunk, buffer[-300:]
        buffer += chunk
    time.sleep(0.2)
    failed = commit(free, audio)
    assert failed["type"].endswith(".failed") and failed["error"]["code"] == "model_loading", failed
    assert json.loads((buffer + r.read()).decode().strip().split("\n\n")[-1][6:])["type"] == "loaded"
    busy.join()
    completed = commit(free, audio)
    assert completed["type"].endswith(".completed") and completed["transcript"], completed
    for how, session in (("in session.update", named), ("in the address", queried)):
        failed = commit(session, audio)
        assert failed["type"].endswith(".failed") and failed["error"]["code"] == "model_not_found", (how, failed)
        session.close()
    free.close()
    info["recognition"] = load(recognition["name"], "recognition")["model"]
    print(f"realtime through a replacement: a commit while {third['name']} loads fails with model_loading, a session without a model "
          f"goes on with it ({completed['transcript'][:20]!r}), and one that named {recognition['name']}, in session.update or in "
          f"the address, fails with model_not_found")

if detection:
    expect_error(transcribe(wav, chunking=True), 400, "unsupported_parameter", "chunking_strategy", "chunking_strategy without a detection model")
    held = load(detection["name"], "detection")
    status, _, body = call("GET", "/v1/models")
    assert [m["speech"]["task"] for m in json.loads(body)["data"]] == ["synthesis", "recognition", "detection"], body
    status, _, body = transcribe(wav, chunking=True)
    assert status == 200 and json.loads(body)["text"], body
    print(f"detection: {held['name']} loaded by the page into its empty place, /v1/models listing it third, and chunking_strategy "
          f"refused before and answered after: {json.loads(body)['text']!r}")
    # A Realtime session with server_vad keeps the detection model only until the page replaces it, so that the load does
    # not wait for the session, and goes on with the new one.
    vad_session = server.websocket("/v1/realtime")
    assert vad_session.event()["type"] == "session.created"
    vad_session.send({"type": "session.update", "session": {"type": "transcription", "audio": {"input": {"turn_detection": {"type": "server_vad"}}}}})
    assert vad_session.event()["type"] == "session.updated"
    started = time.perf_counter()
    load(detection["name"], "detection")
    took = time.perf_counter() - started
    assert took < 10, f"the page's load of the detection model waited {took:.1f} s for a Realtime session"
    at = struct.unpack("<I", wav[24:28])[0]
    values = array.array("h", wav[44:])
    spoken = array.array("h", [values[min(i * at // 24000, len(values) - 1)] for i in range(len(values) * 24000 // at)]).tobytes() + b"\0" * 72000
    for start in range(0, len(spoken), 960):
        vad_session.send({"type": "input_audio_buffer.append", "audio": base64.b64encode(spoken[start:start + 960]).decode()})
    e = vad_session.event()
    while not e["type"].endswith((".completed", ".failed")):
        e = vad_session.event()
    assert e["type"].endswith(".completed") and e["transcript"], e
    vad_session.close()
    print(f"detection replaced by the page in {took:.2f} s while a server_vad session held it, which went on with the new one: "
          f"{e['transcript']!r}")

absent = next(m for m in catalog["models"] if all(m["name"] != g["name"].split(":")[0] for g in given))
held_before = json.loads(call("GET", "/speech/models")[2])[absent["task"]]["held"]
connection = http.client.HTTPConnection("127.0.0.1", port, timeout=600)
connection.request("POST", "/speech/load", json.dumps({"model": absent["name"]}).encode(), {"Content-Type": "application/json"})
r = connection.getresponse()
assert r.status == 200, r.status
buffer = b""
while not re.search(rb'"type":"fetch","done":[1-9]', buffer):
    chunk = r.read1(4096)
    assert chunk, f"the load of {absent['name']} ended before its fetch: {buffer[-300:]!r}"
    buffer += chunk
connection.close()
for _ in range(100):
    state = json.loads(call("GET", "/speech/models")[2])
    if state[absent["task"]]["replacing"] is None:
        break
    time.sleep(0.1)
entry = next(m for m in state["catalog"]["models"] if m["name"] == absent["name"])
part = next(f for f in entry["files"] if f["type"] == entry["type"])
assert state[absent["task"]]["replacing"] is None and not part["fetched"] and part["partial"] > 0, (state[absent["task"]], part)
assert state[absent["task"]]["held"] == held_before, "a stopped fetch changed the model held"
subprocess.run([speech, "rm", absent["name"]], env=env, capture_output=True, check=True)
print(f"a fetch the page stopped: {part['partial']} bytes of {absent['name']} kept as a part, the {absent['task']} model as it was")

# Another process holds the third model's lock with its file out of place, as one that fetches it does; the page's load
# waits on the lock and goes away; the other process puts the file in place and lets the lock go.
third_file = next(f for m in catalog["models"] for f in m["files"] if (m["name"] if f["type"] == m["type"] else f"{m['name']}:{f['type']}") == third["name"])
moved = third_file["path"] + ".away"
os.rename(third_file["path"], moved)
holder = open(third_file["path"] + ".lock", "a+b")
if os.name == "nt":
    import msvcrt
    msvcrt.locking(holder.fileno(), msvcrt.LK_NBLCK, 1)
else:
    import fcntl
    fcntl.flock(holder.fileno(), fcntl.LOCK_EX)
held_before = json.loads(call("GET", "/speech/models")[2])[third["task"]]["held"]
connection = http.client.HTTPConnection("127.0.0.1", port, timeout=600)
connection.request("POST", "/speech/load", json.dumps({"model": third["name"]}).encode(), {"Content-Type": "application/json"})
r = connection.getresponse()
buffer = b""
while b'"type":"note"' not in buffer:
    chunk = r.read1(4096)
    assert chunk, f"the load of {third['name']} ended before it waited for the lock: {buffer[-300:]!r}"
    buffer += chunk
connection.close()
time.sleep(0.5)
os.rename(moved, third_file["path"])
if os.name == "nt":
    holder.seek(0)
    msvcrt.locking(holder.fileno(), msvcrt.LK_UNLCK, 1)
else:
    fcntl.flock(holder.fileno(), fcntl.LOCK_UN)
holder.close()
for _ in range(300):
    state = json.loads(call("GET", "/speech/models")[2])
    if state[third["task"]]["replacing"] is None:
        break
    time.sleep(0.1)
assert state[third["task"]]["replacing"] is None, "the load that lost its page did not end"
assert state[third["task"]]["held"] == held_before, f"a load whose page went away replaced the model by {state[third['task']]['held']['name']}"
print(f"a load of {third['name']} whose page went away while it waited for the lock: the {third['task']} model as it was")

out = server.stop()
assert out == b"", out[:200]

elsewhere = Server(speech, [synthesis["path"]], env, host="0.0.0.0")
assert elsewhere.page is None, elsewhere.lines
status, headers, body = elsewhere.call("GET", "/")
assert status == 404 and headers["content-type"].startswith("text/html") and b"127.0.0.1" in body, body
expect_error(elsewhere.call("GET", "/speech/models", headers={"Authorization": "Bearer " + "0" * 32}), 404, "page_off", None,
             "/speech/models on 0.0.0.0")
assert elsewhere.call("GET", "/v1/models")[0] == 200
elsewhere.stop()
print("on 0.0.0.0: no page address printed, the page saying how to reach it, and its endpoints off")
shutil.rmtree(folder)
print("ok")
