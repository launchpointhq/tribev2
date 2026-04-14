"""RunPod Serverless entry point for TRIBE v2 brain-activity timelines."""
from __future__ import annotations

import base64
import io
import os
import tempfile
import traceback

import numpy as np
import requests
import runpod

from tribev2 import TribeModel

from timeline import build_timeline

CACHE = os.environ.get("TRIBE_CACHE", "/runpod-volume/tribev2-cache")
os.makedirs(CACHE, exist_ok=True)

print("Loading TRIBE v2 from HuggingFace…", flush=True)
MODEL = TribeModel.from_pretrained("facebook/tribev2", cache_folder=CACHE)
TR = float(MODEL.data.TR)
print(f"Ready. TR={TR}s", flush=True)

DEFAULT_SUFFIX = {"video": ".mp4", "audio": ".wav", "text": ".txt"}
VALID_SUFFIX = {
    "video": {".mp4", ".avi", ".mkv", ".mov", ".webm"},
    "audio": {".wav", ".mp3", ".flac", ".ogg"},
    "text": {".txt"},
}


def _pick_suffix(spec: dict, kind: str) -> str:
    if "suffix" in spec:
        s = spec["suffix"].lower()
        if not s.startswith("."):
            s = "." + s
        return s
    if "url" in spec:
        from urllib.parse import urlparse
        path = urlparse(spec["url"]).path.lower()
        for ext in VALID_SUFFIX[kind]:
            if path.endswith(ext):
                return ext
    return DEFAULT_SUFFIX[kind]


UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/122.0 Safari/537.36"
)


def _download(url: str, dest: str) -> None:
    with requests.get(url, stream=True, allow_redirects=True,
                      headers={"User-Agent": UA}, timeout=120) as r:
        if r.status_code >= 400:
            raise ValueError(
                f"Could not fetch {url}: HTTP {r.status_code}. "
                "The URL must be publicly downloadable (no login, no preview "
                "page). For Google Drive, Dropbox, etc., use a direct "
                "download link, a signed S3 URL, or send the file as base64."
            )
        with open(dest, "wb") as f:
            for chunk in r.iter_content(1024 * 256):
                if chunk:
                    f.write(chunk)


def _materialize(spec: dict, kind: str) -> str:
    if "path" in spec:
        return spec["path"]
    fd, path = tempfile.mkstemp(suffix=_pick_suffix(spec, kind))
    os.close(fd)
    if "url" in spec:
        _download(spec["url"], path)
    elif "b64" in spec:
        with open(path, "wb") as f:
            f.write(base64.b64decode(spec["b64"]))
    else:
        raise ValueError(f"{kind} needs one of: url | b64 | path")
    return path


def handler(job):
    try:
        inp = job["input"]
        opts = inp.get("options", {})

        kwargs = {}
        if "video" in inp:
            kwargs["video_path"] = _materialize(inp["video"], "video")
        if "audio" in inp:
            kwargs["audio_path"] = _materialize(inp["audio"], "audio")
        if "text" in inp:
            kwargs["text_path"] = _materialize(inp["text"], "text")
        if not kwargs:
            return {"error": "provide one of: video, audio, text"}

        events = MODEL.get_events_dataframe(**kwargs)
        preds, segments = MODEL.predict(events, verbose=False)

        out = {
            "timeline": build_timeline(
                preds,
                segments,
                tr=TR,
                top_k=opts.get("top_k", 8),
                include_full_regions=opts.get("include_full_regions", True),
                z_score=opts.get("z_score", True),
            )
        }

        if opts.get("include_raw_vertices"):
            buf = io.BytesIO()
            np.savez_compressed(buf, preds=preds.astype(np.float32))
            out["raw_vertices_npz_b64"] = base64.b64encode(buf.getvalue()).decode()
            out["raw_vertices_shape"] = list(preds.shape)

        return out
    except Exception as e:
        return {"error": str(e), "traceback": traceback.format_exc()}


runpod.serverless.start({"handler": handler})
