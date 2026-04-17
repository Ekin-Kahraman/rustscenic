# Pinned reference environment for rustscenic audit.
# This image is ground truth. Every PR runs against it; we do not upgrade.
#
# Known quirks (baked into our choices here):
#   - arboreto 0.1.6 sdist is broken on PyPI (build_meta misconfigured) — force --only-binary
#   - arboreto's Dask cluster path crashes on every modern dask; run_reference.py uses client=None
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl git build-essential libhdf5-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    numpy==1.26.4 \
    pandas==2.1.4 \
    scipy==1.13.1 \
    dask==2024.1.1 \
    "distributed==2024.1.1" \
    lightgbm==4.6.0 \
    scanpy==1.11.5 \
    anndata==0.10.9 \
    pyscenic==0.12.1 \
    pyarrow==15.0.0 \
    && pip install --no-cache-dir --only-binary=arboreto arboreto==0.1.6

COPY datasets.sh /
COPY run_reference.py /
RUN chmod +x /datasets.sh && /datasets.sh

ENTRYPOINT ["python", "/run_reference.py"]
