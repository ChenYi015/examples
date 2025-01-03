ARG TENSORFLOW_IMAGE=tensorflow/tensorflow:2.18.0-gpu

FROM ${TENSORFLOW_IMAGE}

COPY requirements.txt .

RUN set -eux && \
    pip install -r requirements.txt && \
    rm requirements.txt
