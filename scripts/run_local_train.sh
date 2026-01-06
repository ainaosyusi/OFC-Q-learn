#!/bin/bash

# ====================================================
# OFC-Q-learn Local Training Launcher
# ====================================================

# リポジトリのルートディレクトリへ移動（スクリプトのある場所から一つ上へ）
SCRIPT_DIR=$(cd $(dirname $0); pwd)
REPO_DIR=$(dirname $SCRIPT_DIR)
cd $REPO_DIR

LOG_DIR="logs"
CKPT_DIR="ckpt"

# パラメータ（ローカルでのテスト用に少し少なめに設定、必要に応じて変更して使ってください）
EPISODES=10000
N_PLAYERS=2
HERO_IDX=0
SEED=11
MODEL_NAME="ofc_qnet_local.pt"
LOG_NAME="train_local.log"

# logs, ckpt 作成
mkdir -p $LOG_DIR
mkdir -p $CKPT_DIR

echo "Starting Local Training..."
echo "Episodes: $EPISODES"
echo "Model: $CKPT_DIR/$MODEL_NAME"
echo "Logging to: $LOG_DIR/$LOG_NAME (and stdout)"

# 学習コマンド実行
# 2>&1 | tee ... で標準出力・エラー出力を両方ファイルと画面に出す
python dqn_ofc_multi.py \
    --train \
    --episodes $EPISODES \
    --n_players $N_PLAYERS \
    --hero_idx $HERO_IDX \
    --seed $SEED \
    --model ./$CKPT_DIR/$MODEL_NAME \
    2>&1 | tee $LOG_DIR/$LOG_NAME

echo "Test Training Finished."
