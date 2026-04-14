"""Turn (n_segments, n_vertices) fsaverage5 predictions into a compact ROI timeline JSON."""
from __future__ import annotations

from functools import lru_cache

import numpy as np

FSA5_PER_HEMI = 10242


@lru_cache(maxsize=1)
def _destrieux_fsa5():
    """Load Destrieux atlas and downsample to fsaverage5.

    fsaverage is a hierarchical mesh: the first N vertices of fsaverage
    correspond 1:1 to fsaverageN, so slicing the first 10242 labels per
    hemisphere yields the fsaverage5 parcellation.
    """
    from nilearn import datasets

    atlas = datasets.fetch_atlas_surf_destrieux()
    names = [n.decode() if isinstance(n, bytes) else n for n in atlas["labels"]]
    lh = np.asarray(atlas["map_left"])[:FSA5_PER_HEMI]
    rh = np.asarray(atlas["map_right"])[:FSA5_PER_HEMI]
    return names, lh, rh


def _region_means(activations: np.ndarray, label_map: np.ndarray,
                  names: list[str], hemi: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for lbl_id in np.unique(label_map):
        name = names[lbl_id]
        if name in ("Unknown", "Medial_wall"):
            continue
        mask = label_map == lbl_id
        if not mask.any():
            continue
        out[f"{hemi}.{name}"] = float(activations[mask].mean())
    return out


def build_timeline(
    preds: np.ndarray,
    segments: list,
    tr: float,
    top_k: int = 8,
    include_full_regions: bool = True,
    z_score: bool = True,
) -> dict:
    """Build a timeline JSON from per-TR fsaverage5 predictions.

    Parameters
    ----------
    preds
        Array of shape (n_segments, 20484).
    segments
        Segment objects aligned with ``preds``; ``offset`` (seconds) is read when present.
    tr
        Seconds per timestep.
    top_k
        Number of strongest regions to surface per frame.
    z_score
        Z-score each region across time so activations are comparable between regions.
    """
    names, lh_labels, rh_labels = _destrieux_fsa5()
    n_seg, n_vert = preds.shape
    if n_vert != 2 * FSA5_PER_HEMI:
        raise ValueError(f"expected 20484 fsaverage5 vertices, got {n_vert}")

    frames: list[dict[str, float]] = []
    for i in range(n_seg):
        lh = preds[i, :FSA5_PER_HEMI]
        rh = preds[i, FSA5_PER_HEMI:]
        frames.append({
            **_region_means(lh, lh_labels, names, "lh"),
            **_region_means(rh, rh_labels, names, "rh"),
        })

    region_names = sorted(frames[0].keys())
    mat = np.array([[f[r] for r in region_names] for f in frames])

    if z_score and n_seg > 1:
        mu = mat.mean(0, keepdims=True)
        sd = mat.std(0, keepdims=True) + 1e-8
        mat = (mat - mu) / sd

    timeline = []
    for i, seg in enumerate(segments):
        row = mat[i]
        order = np.argsort(-row)
        top = [
            {
                "region": region_names[j],
                "hemi": region_names[j][:2],
                "z": round(float(row[j]), 3),
            }
            for j in order[:top_k]
        ]
        entry = {
            "t": round(float(getattr(seg, "offset", i * tr)), 3),
            "tr_index": i,
            "top_regions": top,
        }
        if include_full_regions:
            entry["regions"] = {
                region_names[j]: round(float(row[j]), 3)
                for j in range(len(region_names))
            }
        timeline.append(entry)

    return {
        "meta": {
            "mesh": "fsaverage5",
            "n_vertices": int(n_vert),
            "n_regions": len(region_names),
            "atlas": "destrieux_2009",
            "tr_seconds": float(tr),
            "duration_seconds": round(n_seg * float(tr), 3),
            "activation_unit": "z-score across timeline" if z_score else "raw model output",
        },
        "regions": region_names,
        "timeline": timeline,
    }
