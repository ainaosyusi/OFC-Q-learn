FC-Q-learn % mkdir -p ckpt
python dqn_ofc.py --episodes 1000 --seed 11

でとりあえず動かして
# 例: パスを書き換えて新規で学習
python dqn_ofc.py --episodes 5000 --seed 11
# （dqn_ofc.py 内の save_path を "./ckpt/ofc_qnet_v2.pt" に変えておくと分かりやすい）

python dqn_ofc.py --test


学習経過
11/17　まだプレイが幼稚　ペアペアハイでギリ耐えできるレベル