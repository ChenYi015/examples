ARG PYTORCH_IMAGE=pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

FROM ${PYTORCH_IMAGE}

COPY requirements.txt .

RUN set -eux && \
    pip install -r requirements.txt && \
    rm requirements.txt
