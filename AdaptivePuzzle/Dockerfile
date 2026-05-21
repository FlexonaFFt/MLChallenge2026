FROM python:3.11-slim

RUN pip install --no-cache-dir numpy tqdm \
    && pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch

COPY train.py solve.py gym.py /opt/participant/
COPY core/   /opt/participant/core/
COPY models/ /opt/participant/models/
COPY utils/  /opt/participant/utils/

WORKDIR /workspace
ENV PYTHONPATH=/workspace:/opt/participant
