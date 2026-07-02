# PyTorch公式のGPU対応イメージを使用
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel

# タイムゾーンの選択をスキップするための設定を追加
ENV DEBIAN_FRONTEND=noninteractive

# システム依存関係のインストール
RUN apt-get update && apt-get install -y \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 作業ディレクトリの設定
WORKDIR /workspace

# リポジトリをコピー（ローカルでgit clone済みの場合）
COPY . /workspace

# Pythonライブラリのインストール
# plyfile: data_utils/RicePaddyDataLoader.py が .ply ファイル読み込みに使用
RUN pip install --no-cache-dir tqdm matplotlib scikit-learn plyfile
