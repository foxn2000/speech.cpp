"""Runs a pinned NeMo FastConformer on the CPU in float32 and saves the tensors the C++ port is checked against.

usage: uv run python dump.py <model> <out dir> <audio.wav>...
       uv run python dump.py --times <model> <out dir>
       uv run python dump.py --greedy <model> <out dir>

Each WAVE file, mono at the model's sample rate, goes to <out dir>/<model>/<file name without .wav>/. The
stages are those of the official forward pass and of the default decoding of its transducer (greedy TDT, or beam
search over RNN-T), in evaluation (no dither), recorded with hooks:

  audio          the samples as read, [N]
  features       the normalized log-mel features of the valid frames, [frames, mels]
  subsampled     the subsampling's output, before the encoder scales it by sqrt(d_model), [T, d_model]
  layers         the output of each conformer layer, [layers, T, d_model]
  encoded        the encoder's output, [T, d_model]
  pred_labels    the label of each step of the prediction network in decoding order, the blank first, [n]
  pred_parents   the step whose state each step continues, -1 for the zero state at the start; in greedy decoding
                 each step continues the one before it, in beam search the step of its hypothesis' labels, [n]
  pred_output    the prediction network's output for each, [n, hidden]
  joint_frames   the encoder frame of each evaluation of the joint, [k]
  joint_steps    the index in pred_labels of the prediction network's output each evaluation used, [k]
  joint_output   the joint's output for each, a log-softmax over the tokens, the blank and a TDT joint's durations
                 together as the joint returns it on the CPU, [k, vocabulary + 1 + durations]
  ids            the token ids of the decoding, [m]

text.txt holds the text in UTF-8 without a newline, and meta.json the same text and the lengths. The text of the
stages is asserted to equal what the official transcribe() returns with the model's default decoding.

Then each file goes through transcribe(timestamps=True), whose ids and text are asserted to be the stages', and the
times it gives are saved beside them, the offsets in encoder frames and the times in seconds as NeMo computes them:

  token_timestep   the time index the decoding recorded with each token, [m]: the frame it was emitted on in greedy
                   TDT decoding, the step of the search (the frame plus the tokens before it) in the beam search
  token_duration   the duration greedy TDT decoding predicted with each token, in frames, [m]; TDT only
  token_offsets    the start and end offset of each token in frames, as compute_rnnt_timestamps() gives them, [m, 2]
  token_seconds    the same in seconds, float64, [m, 2]
  segment_offsets  the start and end offset of each segment in frames, [s, 2]
  segment_seconds  the same in seconds, float64, [s, 2]

segments.txt holds the text of each segment on a line of its own, and times.json all of it with the words, each
token's text and the settings the times depend on: the separators, whether the checkpoint sets them, the punctuation
marks, the subsampling factor and the window stride. With --times, the times alone are added to every dump of the
model under <out dir>/<model>/, from the audio.npy each holds.

With --greedy, for a model whose default decoding is RNN-T's beam search, every dump of the model under
<out dir>/<model>/ gains a folder greedy/ with NeMo's greedy decoding of the dump's encoder output, as
change_decoding_strategy() sets it up for the strategy greedy_batch with the rest of the checkpoint's decoding
settings, its greedy max_symbols among them: GreedyBatchedRNNTInfer's label looping. It holds ids.npy and text.txt,
meta.json with the text, the most tokens on one frame and the evaluations of the joint and steps of the prediction
network the decoding took, and the times of transcribe(timestamps=True) with the same decoding, as above, whose
token_timestep is the frame each token was emitted on. The ids and their frames are asserted to be those of the
strategy greedy as well (GreedyRNNTInfer, which goes frame by frame), and the text that of transcribe() of the dump's
audio. The folder holds no features.npy, so that the checks do not take it for a dump of the default decoding.
"""

import argparse
import copy
import json
import os
import tempfile

import numpy as np
import soundfile
import torch
from omegaconf import open_dict

from pins import MODELS, NEMO, restore

parser = argparse.ArgumentParser()
parser.add_argument("--times", action="store_true", help="add the times to the model's existing dumps")
parser.add_argument("--greedy", action="store_true", help="add NeMo's greedy decoding to the model's existing dumps")
parser.add_argument("model", choices=sorted(MODELS))
parser.add_argument("out_dir")
parser.add_argument("audio", nargs="*")
args = parser.parse_args()
if args.times and args.greedy:
    parser.error("give --times or --greedy, not both")
if (args.times or args.greedy) == bool(args.audio):
    parser.error("give WAVE files, or --times or --greedy without them")

pin = MODELS[args.model]
model = restore(pin)
sample_rate = int(model.cfg.preprocessor.sample_rate)
# The checkpoint's own decoding settings, which change_decoding_strategy() replaces with the ones it is given.
checkpoint_decoding = copy.deepcopy(model.cfg.decoding)
if args.greedy and type(model.decoding.decoding).__name__ != "BeamRNNTInfer":
    parser.error(f"{args.model} decodes greedily by default; its dumps hold that decoding already")
# A hybrid checkpoint has a CTC head beside its transducer; transcribe() decodes with the transducer while cur_decoder
# is "rnnt".
assert getattr(model, "cur_decoder", "rnnt") == "rnnt"
# transcribe(timestamps=True) merges every default of the decoding's configuration into the checkpoint's, so what the
# checkpoint itself sets is read before.
set_by_checkpoint = {key: key in model.cfg.decoding for key in ("segment_seperators", "word_seperator", "segment_gap_threshold")}

# The decoding calls the prediction network on the label each hypothesis ends with, from the state its labels before
# left, projects the outputs and the encoder's frames for the joint, and evaluates the joint on pairs of them: one at a
# time in greedy decoding, a batch of the beam's hypotheses at a time in beam search. These wrappers record every row.
# Rows are told apart by their bytes, which equal tensors share and which the decoding copies without change.
calls = {}
predict, project_encoder, project_prednet, joint_after_projection = (
    model.decoder.predict, model.joint.project_encoder, model.joint.project_prednet, model.joint.joint_after_projection)


def row_key(x):
    return x.detach().contiguous().numpy().tobytes()


def state_key(state, row):
    """Packs one batch row's state, whether LSTM or Haas's recurrent and convolution components."""
    return b"".join(row_key(component[:, row]) for component in state)


def recording_predict(y, state, *a, **k):
    assert y.shape[1] == 1
    # Haas's short convolutions can update views of the incoming state in place, so identify parents before predict().
    parents = [calls["states"][state_key(state, r)]
               if state is not None and any(bool(s[:, r].any()) for s in state) else -1
               for r in range(y.shape[0])]
    out = predict(y, state, *a, **k)
    for r, parent in enumerate(parents):
        calls["states"].setdefault(state_key(out[1], r), len(calls["pred"]))
        calls["pred"].append((int(y[r, 0]), parent, out[0][r, 0].clone()))
        calls["outputs"].setdefault(row_key(out[0][r, 0]), len(calls["pred"]) - 1)
    return out


def recording_project_encoder(f):
    out = project_encoder(f)
    for x, p in zip(f.reshape(-1, f.shape[-1]), out.reshape(-1, out.shape[-1])):
        calls["projected_frames"][row_key(p)] = calls["frames"][row_key(x)]
    return out


def recording_project_prednet(g):
    out = project_prednet(g)
    for x, p in zip(g.reshape(-1, g.shape[-1]), out.reshape(-1, out.shape[-1])):
        calls["projected_steps"][row_key(p)] = calls["outputs"][row_key(x)]
    return out


def recording_joint(f, g):
    out = joint_after_projection(f, g)
    f, g, o = f.reshape(-1, f.shape[-1]), g.reshape(-1, g.shape[-1]), out.reshape(-1, out.shape[-1])
    assert f.shape[0] == g.shape[0] == o.shape[0]
    for r in range(o.shape[0]):
        calls["joint"].append((calls["projected_frames"][row_key(f[r])], calls["projected_steps"][row_key(g[r])], o[r].clone()))
    return out


def record_decoding(enabled):
    """Enable stage recording, or restore the original methods for an independent transcribe() call."""
    methods = ((recording_predict, recording_project_encoder, recording_project_prednet, recording_joint) if enabled else
               (predict, project_encoder, project_prednet, joint_after_projection))
    model.decoder.predict, model.joint.project_encoder, model.joint.project_prednet, model.joint.joint_after_projection = methods


record_decoding(True)


def dump_stages(path):
    """Saves the stages of the WAVE file at `path` and returns the folder they went to."""
    samples, rate = soundfile.read(path, dtype="float32")
    assert rate == sample_rate and samples.ndim == 1, f"{path} is not mono at {sample_rate} Hz"
    name = os.path.splitext(os.path.basename(path))[0]
    out = os.path.join(args.out_dir, args.model, name)
    os.makedirs(out, exist_ok=True)
    saved = {"audio": samples}
    # transcribe() ends with the encoder's unfreeze(), which puts it in training: dropout would run, and every
    # batch norm would update its statistics, changing the model for all that follows.
    model.eval()

    signal = torch.from_numpy(samples)[None]
    features, frames = model.preprocessor(input_signal=signal, length=torch.tensor([signal.shape[1]]))
    frames = int(frames[0])
    saved["features"] = features[0, :, :frames].T

    hooked = {"layers": []}
    hooks = [model.encoder.pre_encode.register_forward_hook(lambda m, a, o: hooked.update(subsampled=o[0]))]
    hooks += [layer.register_forward_hook(lambda m, a, o: hooked["layers"].append(o)) for layer in model.encoder.layers]
    encoded, length = model.encoder(audio_signal=features, length=torch.tensor([frames]))
    for h in hooks:
        h.remove()
    t = int(length[0])
    saved["subsampled"] = hooked["subsampled"][0, :t]
    saved["layers"] = torch.stack([layer[0, :t] for layer in hooked["layers"]])
    saved["encoded"] = encoded[0, :, :t].T
    assert torch.equal(saved["layers"][-1], saved["encoded"])

    # The decoding reads only valid frames; the encoder can return additional padding frames.
    encoded = encoded[:, :, :t]
    frames_of = {}
    for i in range(t):
        frames_of.setdefault(row_key(encoded[0, :, i]), i)
    calls.update(pred=[], states={}, outputs={}, frames=frames_of, projected_frames={}, projected_steps={}, joint=[])
    hypothesis = model.decoding.rnnt_decoder_predictions_tensor(encoder_output=encoded, encoded_lengths=length, return_hypotheses=True)[0]
    saved["pred_labels"] = np.array([label for label, _, _ in calls["pred"]], dtype=np.int32)
    saved["pred_parents"] = np.array([parent for _, parent, _ in calls["pred"]], dtype=np.int32)
    saved["pred_output"] = torch.stack([g for _, _, g in calls["pred"]])
    saved["joint_frames"] = np.array([frame for frame, _, _ in calls["joint"]], dtype=np.int32)
    saved["joint_steps"] = np.array([step for _, step, _ in calls["joint"]], dtype=np.int32)
    saved["joint_output"] = torch.stack([o for _, _, o in calls["joint"]])
    ids = [int(i) for i in hypothesis.y_sequence.tolist() if i != model.decoding.blank_id]
    saved["ids"] = np.array(ids, dtype=np.int32)
    text = hypothesis.text
    assert model.decoding.decode_tokens_to_str_with_strip_punctuation(ids) == text
    record_decoding(False)
    try:
        official = model.transcribe([path], batch_size=1, verbose=False)[0].text
    finally:
        record_decoding(True)
    assert text == official, f"the stages give {text!r}, transcribe() gives {official!r}"

    for key, value in saved.items():
        array = value.detach().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)
        np.save(os.path.join(out, f"{key}.npy"), np.ascontiguousarray(array).astype(np.int32 if array.dtype.kind in "iu" else np.float32))
    meta = {
        "nemo": NEMO, "model": pin, "audio": os.path.basename(path), "samples": int(samples.shape[0]),
        "seconds": samples.shape[0] / sample_rate, "frames": frames, "encoded_frames": t,
        "prediction_steps": len(saved["pred_labels"]), "joint_evaluations": len(saved["joint_frames"]), "text": text,
    }
    with open(os.path.join(out, "text.txt"), "w", encoding="utf-8", newline="") as f:
        f.write(text)
    json.dump(meta, open(os.path.join(out, "meta.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps({k: v for k, v in meta.items() if k not in ("nemo", "model")}, ensure_ascii=False))
    return out


def dump_times(path, out):
    """Saves the times transcribe(timestamps=True) gives for the WAVE file at `path` into the dump folder `out`."""
    hypothesis = model.transcribe([path], batch_size=1, timestamps=True, verbose=False)[0]
    blank = model.decoding.blank_id
    ids = [int(i) for i in hypothesis.y_sequence.tolist()]
    assert blank not in ids
    assert ids == np.load(os.path.join(out, "ids.npy")).tolist(), f"transcribe(timestamps=True) gives other ids for {path}"
    with open(os.path.join(out, "text.txt"), encoding="utf-8", newline="") as f:
        assert hypothesis.text == f.read(), f"transcribe(timestamps=True) gives another text for {path}"
    stamps = hypothesis.timestamp
    timestep = [int(t) for t in stamps["timestep"]]
    tdt = model.decoding._is_tdt
    duration = [int(d) for d in hypothesis.token_duration] if tdt else None
    chars, words, segments = stamps["char"], stamps["word"], stamps["segment"]
    assert len(timestep) == len(chars) == len(ids) and (duration is None or len(duration) == len(ids))

    def pairs(entries, *keys):
        return [[entry[k] for k in keys] for entry in entries]

    arrays = {
        "token_timestep": np.array(timestep, dtype=np.int32),
        "token_offsets": np.array(pairs(chars, "start_offset", "end_offset"), dtype=np.int32).reshape(-1, 2),
        "token_seconds": np.array(pairs(chars, "start", "end"), dtype=np.float64).reshape(-1, 2),
        "segment_offsets": np.array(pairs(segments, "start_offset", "end_offset"), dtype=np.int32).reshape(-1, 2),
        "segment_seconds": np.array(pairs(segments, "start", "end"), dtype=np.float64).reshape(-1, 2),
    }
    if tdt:
        arrays["token_duration"] = np.array(duration, dtype=np.int32)
    for key, array in arrays.items():
        np.save(os.path.join(out, f"{key}.npy"), np.ascontiguousarray(array))
    assert all("\n" not in s["segment"] for s in segments)
    with open(os.path.join(out, "segments.txt"), "w", encoding="utf-8", newline="") as f:
        f.write("".join(s["segment"] + "\n" for s in segments))
    times = {
        "nemo": NEMO, "model": pin,
        "segment_seperators": list(model.decoding.segment_seperators), "word_seperator": model.decoding.word_seperator,
        "segment_gap_threshold": model.decoding.segment_gap_threshold, "set_by_checkpoint": set_by_checkpoint,
        "supported_punctuation": sorted(model.decoding.supported_punctuation),
        "tokenizer_type": model.decoding.tokenizer_type,
        "subsampling_factor": int(model.encoder.subsampling_factor),
        "window_stride": float(model.cfg.preprocessor.window_stride),
        "timestep": timestep, "token_duration": duration,
        "pieces": model.decoding.decode_ids_to_tokens(ids),
        "char": [dict(c, char=list(c["char"])) for c in chars], "word": words, "segment": segments,
    }
    with open(os.path.join(out, "times.json"), "w", encoding="utf-8") as f:
        json.dump(times, f, ensure_ascii=False, indent=1)
    print(json.dumps({"dump": out, "tokens": len(ids), "segments": len(segments)}, ensure_ascii=False))


def wave_file(out, tmp):
    """The dump's samples as a WAVE file in `tmp` for transcribe(), which reads files; written as 32-bit float, they
    read back unchanged."""
    path = os.path.join(tmp, os.path.basename(out) + ".wav")
    soundfile.write(path, np.load(os.path.join(out, "audio.npy")), sample_rate, subtype="FLOAT")
    return path


def use_decoding(strategy):
    """Switches the model to the decoding `strategy`, with the rest of the checkpoint's decoding settings."""
    cfg = copy.deepcopy(checkpoint_decoding)
    with open_dict(cfg):
        cfg.strategy = strategy
    model.change_decoding_strategy(cfg, verbose=False)


def decode(out):
    """The hypothesis of the model's decoding of the dump's encoder output, and the evaluations of the joint and the
    steps of the prediction network it took."""
    model.eval()
    encoded = torch.from_numpy(np.load(os.path.join(out, "encoded.npy"))).T[None].contiguous()
    counts = {"joint": 0, "prediction": 0}

    def counting_predict(y, *a, **k):
        counts["prediction"] += 1 if y is None else y.shape[0]
        return predict(y, *a, **k)

    def counting_joint(f, g):
        out = joint_after_projection(f, g)
        counts["joint"] += out.numel() // out.shape[-1]
        return out

    model.decoder.predict, model.joint.joint_after_projection = counting_predict, counting_joint
    try:
        hypothesis = model.decoding.rnnt_decoder_predictions_tensor(
            encoder_output=encoded, encoded_lengths=torch.tensor([encoded.shape[2]]), return_hypotheses=True)[0]
    finally:
        model.decoder.predict, model.joint.joint_after_projection = predict, joint_after_projection
    return hypothesis, counts


def ids_and_frames(hypothesis):
    """The token ids of a greedy hypothesis and the frame each was emitted on."""
    return [int(i) for i in torch.as_tensor(hypothesis.y_sequence).tolist()], [int(t) for t in torch.as_tensor(hypothesis.timestamp).tolist()]


def dump_greedy(dumps):
    """Saves NeMo's greedy decoding of each dump's encoder output, and its times, into the dump's folder greedy/."""
    use_decoding("greedy")
    assert type(model.decoding.decoding).__name__ == "GreedyRNNTInfer"
    frame_by_frame = {out: ids_and_frames(decode(out)[0]) for out in dumps}
    use_decoding("greedy_batch")
    greedy = model.decoding.decoding
    assert type(greedy).__name__ == "GreedyBatchedRNNTInfer" and greedy.loop_labels
    assert type(greedy.decoding_computer).__name__ == "GreedyBatchedRNNTLabelLoopingComputer"
    assert not greedy.decoding_computer.has_fusion_models()
    # NeMo takes greedy.max_symbols, or greedy.max_symbols_per_step where the configuration has no max_symbols.
    assert greedy.max_symbols == (checkpoint_decoding.greedy.get("max_symbols") or checkpoint_decoding.greedy.get("max_symbols_per_step"))
    outs = []
    for out in dumps:
        hypothesis, counts = decode(out)
        ids, frames = ids_and_frames(hypothesis)
        assert (ids, frames) == frame_by_frame[out], f"greedy and greedy_batch decode {out} differently"
        text = hypothesis.text
        assert model.decoding.decode_tokens_to_str_with_strip_punctuation(ids) == text
        with tempfile.TemporaryDirectory() as tmp:
            official = model.transcribe([wave_file(out, tmp)], batch_size=1, verbose=False)[0].text
        assert text == official, f"the greedy decoding of the encoder output gives {text!r}, transcribe() gives {official!r}"
        greedy_out = os.path.join(out, "greedy")
        os.makedirs(greedy_out, exist_ok=True)
        np.save(os.path.join(greedy_out, "ids.npy"), np.array(ids, dtype=np.int32))
        with open(os.path.join(greedy_out, "text.txt"), "w", encoding="utf-8", newline="") as f:
            f.write(text)
        meta = {
            "nemo": NEMO, "model": pin, "strategy": "greedy_batch", "max_symbols": int(greedy.max_symbols),
            "encoded_frames": int(np.load(os.path.join(out, "encoded.npy")).shape[0]), "tokens": len(ids),
            "prediction_steps": counts["prediction"], "joint_evaluations": counts["joint"], "text": text,
        }
        json.dump(meta, open(os.path.join(greedy_out, "meta.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print(json.dumps({"dump": greedy_out, **{k: v for k, v in meta.items() if k not in ("nemo", "model")}}, ensure_ascii=False))
        outs.append(greedy_out)
    for out, greedy_out in zip(dumps, outs):
        with tempfile.TemporaryDirectory() as tmp:
            dump_times(wave_file(out, tmp), greedy_out)


# The times come from transcribe() as it runs without the recording wrappers, and after the stages of every file, since
# transcribe(timestamps=True) changes the model's decoding for good.
outs = [] if args.times or args.greedy else [dump_stages(path) for path in args.audio]
record_decoding(False)
if args.times or args.greedy:
    root = os.path.join(args.out_dir, args.model)
    dumps = sorted(os.path.join(root, d) for d in os.listdir(root) if os.path.isfile(os.path.join(root, d, "audio.npy")))
    assert dumps, f"no dump of this script is under {root}"
    if args.greedy:
        dump_greedy(dumps)
    else:
        for out in dumps:
            with tempfile.TemporaryDirectory() as tmp:
                dump_times(wave_file(out, tmp), out)
else:
    for path, out in zip(args.audio, outs):
        dump_times(path, out)
