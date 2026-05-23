FROM python:3.11-slim

RUN pip install --no-cache-dir numpy tqdm \
    && pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch

COPY train.py solve.py common.py model.py search.py /opt/participant/

WORKDIR /workspace
ENV PYTHONPATH=/workspace:/opt/participant
