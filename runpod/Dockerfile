FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/runpod-volume/hf \
    TRANSFORMERS_CACHE=/runpod-volume/hf \
    NILEARN_DATA=/runpod-volume/nilearn

RUN apt-get update && apt-get install -y \
      python3.11 python3.11-venv python3-pip \
      ffmpeg git build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.11 /usr/bin/python \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3

WORKDIR /app

COPY pyproject.toml README.md ./
COPY tribev2 ./tribev2

RUN pip install --upgrade pip \
 && pip install -e . \
 && pip install runpod nilearn scipy

RUN python -m spacy download en_core_web_sm || true

# Prime the Destrieux atlas cache so the first request doesn't fetch it.
RUN python -c "from nilearn import datasets; datasets.fetch_atlas_surf_destrieux()"

COPY runpod/timeline.py runpod/handler.py /app/

CMD ["python", "-u", "handler.py"]
