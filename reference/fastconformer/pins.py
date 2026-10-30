"""The pinned checkpoints every script in this folder runs, and the version of NeMo pyproject.toml pins."""

import hashlib
import importlib.metadata
import os

import torch
from huggingface_hub import hf_hub_download
from nemo.collections.asr.models import ASRModel

NEMO = importlib.metadata.version("nemo-toolkit")
MODELS = {
    "Haas-ja-v1.0": {
        "repository": "third-intelligence/Haas-ja-v1.0",
        "revision": "8a36fd5b1a69c75da25e016c5b81597c14a9a810",
        "file": "Haas-ja-v1.0.nemo",
        "size": 1232384000,
        "sha256": "c6ed38698300a371a86eadab4b56d5458f47623f7fc4a5f1e938b5adbe8458aa",
    },
    "parakeet-tdt_ctc-0.6b-ja": {
        "repository": "nvidia/parakeet-tdt_ctc-0.6b-ja",
        "revision": "44edb27eea9317daf89333e75eb830db4b1cc298",
        "file": "parakeet-tdt_ctc-0.6b-ja.nemo",
        "size": 2490951680,
        "sha256": "a2ed2aab9c82cce4e9b699dca669184d7525c7b5e1eb8c3f7b789421bc3637e8",
    },
    "parakeet-tdt-0.6b-v3": {
        "repository": "nvidia/parakeet-tdt-0.6b-v3",
        "revision": "541d1f99c6b0c3cd0b11a95167540bb8edefd82b",
        "file": "parakeet-tdt-0.6b-v3.nemo",
        "size": 2509332480,
        "sha256": "3cbdc85877e668ca7b82d0d56770eb1fac76691f55d6b97545e8d61ca588d10d",
    },
    "reazonspeech-nemo-v2": {
        "repository": "reazon-research/reazonspeech-nemo-v2",
        "revision": "33693408be76b7cba9fd4a7546a0a8772430211b",
        "file": "reazonspeech-nemo-v2.nemo",
        "size": 2477946880,
        "sha256": "d196d43ad03466ca88beeda4bf5fafb07bab7202d4b663b8e4f12cb0a4381fae",
    },
}


def checkpoint(pin):
    """The local path of a pinned .nemo checkpoint, downloaded on first use and checked against its pin."""
    path = hf_hub_download(pin["repository"], pin["file"], revision=pin["revision"])
    size = os.path.getsize(path)
    if size != pin["size"]:
        raise SystemExit(f"{path} has {size} bytes, not the pinned {pin['size']}")
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            digest.update(block)
    if digest.hexdigest() != pin["sha256"]:
        raise SystemExit(f"{path} has the SHA-256 {digest.hexdigest()}, not the pinned {pin['sha256']}")
    return path


def restore(pin):
    """The official model of a pinned checkpoint, on the CPU in float32 and in evaluation."""
    torch.set_grad_enabled(False)
    if pin["repository"] == "third-intelligence/Haas-ja-v1.0":
        import haas_asr._fla
        from kda import KimiDeltaAttention

        # prednet.py imports this loader; replace it before NeMo imports the checkpoint's decoder target.
        haas_asr._fla.check_fla_version()
        haas_asr._fla.load_kda_class = lambda: KimiDeltaAttention
    model = ASRModel.restore_from(checkpoint(pin), map_location="cpu")
    return model.float().eval()
