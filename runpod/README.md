# TRIBE v2 on RunPod Serverless

Wraps `facebook/tribev2` in a queue-based RunPod Serverless worker that
returns a per-TR brain-activity timeline (Destrieux ROI z-scores) aligned
to video / audio / text stimuli.

## Build & push

From the repo root:

```bash
docker build --platform linux/amd64 -f runpod/Dockerfile \
  -t <your-dockerhub-user>/tribev2-runpod:latest .

docker push <your-dockerhub-user>/tribev2-runpod:latest
```

## Deploy

1. Serverless → **New Endpoint** → **Import from Docker Registry** →
   `docker.io/<your-dockerhub-user>/tribev2-runpod:latest`
2. Endpoint Type: **Queue**
3. GPU: **24 GB+** (A10 / L4 / A100). The multimodal stack does not fit in 16 GB.
4. Attach a **Network Volume** at `/runpod-volume` so HF weights and the
   nilearn atlas cache survive cold starts.
5. Max Execution Time: 5–10 min (video + audio + transcription chains).

## Request shape

```json
{
  "input": {
    "video": { "url": "https://example.com/clip.mp4" },
    "options": {
      "top_k": 10,
      "include_full_regions": false,
      "z_score": true,
      "include_raw_vertices": false
    }
  }
}
```

Exactly one of `video`, `audio`, `text` must be provided. Each accepts
`{ "url": ... }`, `{ "b64": ... }`, or `{ "path": ... }`.

## Response shape

```json
{
  "timeline": {
    "meta": {
      "mesh": "fsaverage5",
      "n_vertices": 20484,
      "n_regions": 146,
      "atlas": "destrieux_2009",
      "tr_seconds": 1.49,
      "duration_seconds": 14.9,
      "activation_unit": "z-score across timeline"
    },
    "regions": ["lh.G_Ins_lg_and_S_cent_ins", "..."],
    "timeline": [
      {
        "t": 0.0,
        "tr_index": 0,
        "top_regions": [
          { "region": "lh.G_temporal_middle", "hemi": "lh", "z": 2.41 }
        ],
        "regions": { "lh.G_temporal_middle": 2.41 }
      }
    ]
  }
}
```

With `include_full_regions=false` and `include_raw_vertices=false`, a
2-minute clip returns well under the 20 MB `/runsync` cap.
