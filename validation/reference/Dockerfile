# Pinned reference environment for rustscenic audit.
# This image is ground truth. Every PR runs against it; we do not upgrade.
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
    arboreto==0.1.6 \
    pyscenic==0.12.1 \
    pyarrow==15.0.0

COPY datasets.sh /
COPY run_reference.py /
RUN chmod +x /datasets.sh && /datasets.sh

ENTRYPOINT ["python", "/run_reference.py"]
