# use this file with:
# docker buildx build --platform linux/amd64 --tag warbler .
# docker run --rm -ti -v $(pwd):/mnt warbler python xxx.py
#
# to troubleshoot:
# docker run --rm -ti -v $(pwd):/mnt --entrypoint /bin/bash warbler

FROM python:3.12.11-bullseye

ENV PATH=/opt/venv/bin:$PATH

WORKDIR /src
COPY requirements-docker.txt pyproject.toml ./
COPY fms_ehrs/ fms_ehrs/

RUN python -m venv /opt/venv \
 && pip install --no-cache-dir torch --extra-index-url https://download.pytorch.org/whl/cu128 \
 && pip install --no-build-isolation -r requirements-docker.txt \
 && pip install --no-deps .
