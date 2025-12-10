FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    vim \
    wget \
    locales \
    && localedef -f UTF-8 -i ja_JP ja_JP.UTF-8 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

ENV LANG=ja_JP.UTF-8
ENV LC_ALL=ja_JP.UTF-8
ENV TZ=Asia/Tokyo

WORKDIR /workspaces/dl_scratch

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .
