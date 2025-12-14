# FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime
FROM pytorch/pytorch:2.9.1-cuda13.0-cudnn9-runtime

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

# uvから一部を持ってくる的なやつ。
COPY --from=ghcr.io/astral-sh/uv:0.9.17 /uv /uvx /bin/

# seedがあるとpipが入るよ。
RUN uv venv --seed -p 3.13

COPY requirements.txt .

# RUN uv pip install --no-cache-dir -r requirements.txt
# image内にキャッシュを保存する。肥大化したらdocker system prune
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install -r requirements.txt

COPY . .
