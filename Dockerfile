FROM nvidia/cuda:12.6.3-cudnn9-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/hf_cache

RUN apt-get update && apt-get install -y \
    python3 python3-pip python3-dev \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3 /usr/bin/python

WORKDIR /app

COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

COPY download_model.py .
RUN python3 download_model.py

COPY . .

EXPOSE 9000
CMD ["bash", "start.sh"]
