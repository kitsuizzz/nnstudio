#!/usr/bin/env python3
"""HuggingFace tokenizers for nnstudio's text boxes — stdlib only until one is
actually loaded.

A tokenizer / de-tokenizer box may carry a huggingface repo (`Qwen/Qwen3-8B`,
`gpt2`, or the full URL). The repo's `tokenizer.json` is downloaded once into a
per-user cache and loaded with the `tokenizers` package — `pip install
tokenizers`, only needed for this; a character vocabulary needs nothing.

    NNSTUDIO_TOKENIZER_CACHE   where the files are cached
                               (~/.cache/nnstudio/tokenizers by default)
    HF_TOKEN / HUGGING_FACE_HUB_TOKEN   sent when the repo is gated
"""
import json
import os
import re
import urllib.error
import urllib.request

TIMEOUT = 60
_SKIP = {"tree", "blob", "resolve", "raw", "refs"}


def cache_dir():
    return os.environ.get("NNSTUDIO_TOKENIZER_CACHE") or os.path.join(
        os.path.expanduser("~"), ".cache", "nnstudio", "tokenizers")


def hf_repo_id(text):
    """'Qwen/Qwen3-8B', 'gpt2', 'https://huggingface.co/Qwen/Qwen3-8B/tree/main'…"""
    s = str(text or "").strip().rstrip("/")
    m = re.search(r"huggingface\.co/(.+)$", s)
    if m:
        keep = []
        for part in [p for p in m.group(1).split("/") if p]:
            if part in _SKIP:
                break
            keep.append(part)
            if len(keep) == 2:
                break
        s = "/".join(keep)
    if s.endswith(".git"):
        s = s[:-4]
    if re.match(r"^[A-Za-z0-9_.-]+(/[A-Za-z0-9_.-]+)?$", s):
        return s
    raise ValueError("not a huggingface repo: %r" % (text,))


def cache_path(repo):
    rid = hf_repo_id(repo)
    return os.path.join(cache_dir(), rid.replace("/", "__"), "tokenizer.json")


def tokenizer_path(repo, download=True):
    """The repo's tokenizer.json on disk, downloading it once if needed."""
    path = cache_path(repo)
    if os.path.isfile(path) and os.path.getsize(path) > 0:
        return path
    if not download:
        raise FileNotFoundError("no cached tokenizer.json for %s" % repo)
    rid = hf_repo_id(repo)
    url = "https://huggingface.co/%s/resolve/main/tokenizer.json" % rid
    req = urllib.request.Request(url, headers={"User-Agent": "nnstudio"})
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        req.add_header("Authorization", "Bearer " + token)
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
            data = r.read()
    except urllib.error.HTTPError as e:
        if e.code in (401, 403):
            raise RuntimeError("%s is gated — set HF_TOKEN to a token that can read it" % rid)
        raise RuntimeError("huggingface said %d for %s (no tokenizer.json there?)"
                           % (e.code, rid))
    except OSError as e:
        raise RuntimeError("cannot reach huggingface for %s: %s" % (rid, e))
    if not data:
        raise RuntimeError("huggingface returned an empty tokenizer.json for %s" % rid)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".part"
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, path)
    return path


def tokenizer_meta(repo):
    """What the app shows when a repo is pasted: model kind, real token count."""
    path = tokenizer_path(repo)
    with open(path, encoding="utf-8") as f:
        doc = json.load(f)
    model = doc.get("model") or {}
    ids = [int(v) for v in (model.get("vocab") or {}).values()]
    ids += [int(t.get("id") or 0) for t in doc.get("added_tokens") or []]
    kind = str(model.get("type") or "")
    if not kind:                             # older exports leave the type implied
        if "merges" in model:
            kind = "BPE"
        elif isinstance(model.get("vocab"), list):
            kind = "Unigram"
        elif isinstance(model.get("vocab"), dict):
            kind = "WordLevel"
        else:
            kind = "?"
    return {"repo": hf_repo_id(repo), "path": path, "model": kind,
            "tokens": (max(ids) + 1 if ids else 0),
            "bytes": os.path.getsize(path)}
