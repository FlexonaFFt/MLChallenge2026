# Можно заменить на любой образ; этот — из внутреннего реестра Яндекса,
# собирается быстрее потому, что pytorch уже в слое.
FROM cr.yandex/crp2q2b12lka2f8enigt/pytorch/pytorch:2.8.0-cuda12.6-cudnn9-runtime

RUN pip3 install --no-cache-dir \
    "vllm==0.22.0"

WORKDIR /workspace
COPY . .

ENTRYPOINT ["python3", "solution.py"]
