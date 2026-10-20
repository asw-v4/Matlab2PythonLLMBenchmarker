# The sandbox model-written code runs in. Built once; no network is needed (or allowed) when it runs.
#   scripts/build_sandbox.ps1
FROM python:3.12-slim

# Pinned to the versions in uv.lock, so the code runs against the same numpy as the rest of the benchmark.
RUN pip install --no-cache-dir numpy==2.2.6 scipy==1.18.1

# One BLAS thread, so sums are added in the same order every run.
ENV OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0

WORKDIR /work
