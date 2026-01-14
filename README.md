# OFC-Q-learn

Deep Q-Learning (DQN) を用いたオープンフェイス・チャイニーズポーカー (OFC) AI プロジェクト。

本プロジェクトは、強化学習を用いて OFC パイナップル（JOPTルール相当）の高性能な AI プレイヤーを構築することを目的としています。2〜3人対戦、不完全情報、および構造的な役の制約（Top < Mid < Bot）といった複雑な課題に取り組んでいます。

## 🚀 プロジェクト目標: ゼロからプロレベルへ
- **戦略的意思決定**: 相手の公開カードを観測し、最適なカード配置を行います。
- **制約の学習**: OFC の基本ルールを遵守し、ファウルを回避するように学習します。
- **スケーラブルな学習**: ローカル環境およびクラウド (GCP) での学習をサポートしています。

詳細な開発計画については、[ROADMAP.md](./ROADMAP.md) を参照してください。

## 🛠 特徴
- **環境**: 2〜3人プレイに対応したカスタム Gymnasium ライクな環境 (`multi_ofc_env.py`)。
- **アルゴリズム**:
    - **DQN**: 離散的なアクション空間におけるカード配置のための Deep Q-Network。
    - **Behavior Cloning (BC)**: 人間のデモデータからの事前学習。
- **プラットフォーム**:
    - ローカル学習スクリプト。
    - 長時間の実験のための GCP (Compute Engine) 連携スクリプト。

## 📦 インストール方法

```bash
# リポジトリのクローン
git clone https://github.com/naoai/OFC-Q-learn.git
cd OFC-Q-learn

# 仮想環境の作成
python -m venv venv
source venv/bin/activate

# 依存ライブラリのインストール
pip install torch numpy tqdm
```

## 🏃 使い方

### 学習 (マルチプレイヤー)
```bash
# 2人プレイでの学習 (ローカル)
./scripts/run_local_train.sh

# パラメータを指定して学習
python dqn_ofc_multi.py --train --episodes 1000 --n_players 3 --hero_idx 0 --seed 11
```

### 評価
```bash
# 学習済みモデルを使用した評価
python dqn_ofc_multi.py --eval 100 --n_players 2 --hero_idx 0 --model ./ckpt/ofc_qnet_v1.pt
```

## 📂 ファイル構成
- `multi_ofc_env.py`: ゲームロジックと報酬システム。
- `dqn_ofc_multi.py`: マルチプレイヤー環境向けの DQN 実装。
- `train_bc.py`: 事前学習用 Behavior Cloning。
- `scripts/`: 学習自動化用シェルスクリプト。
- `ckpt/`: モデルチェックポイント保存先 (Git 除外)。
- `logs/`: 学習ログ保存先 (Git 除外)。

## 📄 ライセンス
本プロジェクトは研究および教育目的で公開されています。