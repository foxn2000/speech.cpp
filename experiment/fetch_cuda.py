"""Fetches the parts of the CUDA toolkit a build of ggml needs from NVIDIA's redistributable archives, checks each
against the SHA-256 the release's manifest gives, and unpacks them into one folder.

usage: python fetch_cuda.py VERSION PLATFORM OUT [COMPONENT ...]
       python fetch_cuda.py 13.4.2 windows-x86_64 cuda cccl cuda_crt cuda_cudart cuda_nvcc libnvvm libcublas
"""
import hashlib
import json
import os
import shutil
import sys
import tarfile
import urllib.request
import zipfile

BASE = "https://developer.download.nvidia.com/compute/cuda/redist/"


def fetch(url, path):
    with urllib.request.urlopen(url) as r, open(path, "wb") as f:
        shutil.copyfileobj(r, f, 1 << 20)


def main():
    version, platform, out, components = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:]
    os.makedirs(out, exist_ok=True)
    manifest = json.load(urllib.request.urlopen(f"{BASE}redistrib_{version}.json"))
    for name in components:
        entry = manifest[name][platform]
        archive = os.path.join(out, os.path.basename(entry["relative_path"]))
        print(f"{name} {manifest[name]['version']}: {int(entry['size']) / 1e6:.0f} MB", flush=True)
        fetch(BASE + entry["relative_path"], archive)
        h = hashlib.sha256()
        with open(archive, "rb") as f:
            for block in iter(lambda: f.read(1 << 20), b""):
                h.update(block)
        if h.hexdigest() != entry["sha256"]:
            sys.exit(f"{name}: SHA-256 {h.hexdigest()} is not the manifest's {entry['sha256']}")
        # Each archive holds one top folder, <name>-<platform>-<version>-archive; its contents go into OUT.
        if archive.endswith(".zip"):
            with zipfile.ZipFile(archive) as z:
                for m in z.infolist():
                    rel = m.filename.split("/", 1)[1] if "/" in m.filename else ""
                    if not rel or m.is_dir():
                        continue
                    dest = os.path.join(out, rel)
                    os.makedirs(os.path.dirname(dest), exist_ok=True)
                    with z.open(m) as src, open(dest, "wb") as dst:
                        shutil.copyfileobj(src, dst, 1 << 20)
        else:
            with tarfile.open(archive) as t:
                for m in t.getmembers():
                    parts = m.name.split("/", 1)
                    if len(parts) < 2 or not parts[1]:
                        continue
                    m.name = parts[1]
                    if m.islnk():
                        m.linkname = m.linkname.split("/", 1)[1]
                    t.extract(m, out)
        os.remove(archive)
    print("unpacked into", out)


if __name__ == "__main__":
    main()
