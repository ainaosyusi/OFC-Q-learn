#!/bin/bash

# ====================================================
# OFC-Q-learn GCP Training Launcher
# ====================================================

SESSION_NAME="ofc-train"
PYTHON_ENV="ofc_env"  # 仮想環境名（必要に応じて変更）
REPO_DIR="$HOME/OFC-Q-learn/OFC-Q-learn"  # リポジトリパス
LOG_DIR="logs"
CKPT_DIR="ckpt"

# パラメータ
EPISODES=200000
N_PLAYERS=2
HERO_IDX=0
SEED=11
MODEL_NAME="ofc_qnet_mc_v1.pt"
LOG_NAME="train_mc_v1.log"

# ディレクトリへ移動
cd $REPO_DIR || { echo "Repository not found: $REPO_DIR"; exit 1; }

# logs, ckpt 作成
mkdir -p $LOG_DIR
mkdir -p $CKPT_DIR

# tmux セッション確認
tmux has-session -t $SESSION_NAME 2>/dev/null

if [ $? != 0 ]; then
  echo "Starting new tmux session: $SESSION_NAME"
  tmux new-session -d -s $SESSION_NAME
  
  # 仮想環境有効化
  if [ -d "$HOME/$PYTHON_ENV" ]; then
      tmux send-keys -t $SESSION_NAME "source ~/$PYTHON_ENV/bin/activate" C-m
  fi
  
  # 依存ライブラリ確認 (tqdm)
  tmux send-keys -t $SESSION_NAME "pip install tqdm" C-m

  # 学習コマンド実行
  CMD="python dqn_ofc_multi.py --train --episodes $EPISODES --n_players $N_PLAYERS --hero_idx $HERO_IDX --seed $SEED --model ./$CKPT_DIR/$MODEL_NAME > $LOG_DIR/$LOG_NAME 2>&1"
  
  echo "Command: $CMD"
  tmux send-keys -t $SESSION_NAME "$CMD" C-m
  
  echo "Training started in background."
else
  echo "Session $SESSION_NAME already exists. Attaching..."
fi

echo "===================================================="
echo "To monitor logs: tail -f $REPO_DIR/$LOG_DIR/$LOG_NAME"
echo "To attach tmux:  tmux attach -t $SESSION_NAME"
echo "===================================================="

# ログを少し表示して終了
sleep 2
if [ -f "$LOG_DIR/$LOG_NAME" ]; then
    tail -n 10 "$LOG_DIR/$LOG_NAME"
else
    # まだログがないかもしれないのでセッションの状態を表示
    tmux list-sessions
fi
