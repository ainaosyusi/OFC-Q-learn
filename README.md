FC-Q-learn % mkdir -p ckpt
python dqn_ofc.py --episodes 1000 --seed 11

でとりあえず動かして
# 例: パスを書き換えて新規で学習
python dqn_ofc.py --episodes 5000 --seed 11
# （dqn_ofc.py 内の save_path を "./ckpt/ofc_qnet_v2.pt" に変えておくと分かりやすい）

python dqn_ofc.py --test


学習経過
11/17　まだプレイが幼稚　ペアペアハイでギリ耐えできるレベル

multi
3. 使い方のイメージ
ローカル or GCP で、multi_ofc_env.py と dqn_ofc_multi.py をプロジェクト直下に置いて：
# 2人プレイで学習（hero=0）
python dqn_ofc_multi.py --train --episodes 1000 --n_players 2 --hero_idx 0 --seed 11

# 学習済みモデルで1回だけ試す
python dqn_ofc_multi.py --test --n_players 2 --hero_idx 0 --seed 11
3人プレイなら：
python dqn_ofc_multi.py --train --episodes 1000 --n_players 3 --hero_idx 0


python dqn_ofc_multi.py --eval 1000 --n_players 2 --hero_idx 0 --seed 11

# 実験A：2人戦、ベースライン
python dqn_ofc_multi.py --train --episodes 50000 --n_players 2 --hero_idx 0 --seed 11

# 実験B：3人戦
python dqn_ofc_multi.py --train --episodes 50000 --n_players 3 --hero_idx 0 --seed 11
