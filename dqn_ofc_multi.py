# dqn_ofc_multi.py
import math
import random
from collections import deque, namedtuple
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from multi_ofc_env import (
    OFCMultiEnv, MultiOFCState, PlayerBoard,
    RANKS, SUITS, eval_3, eval_5, catname_3, catname_5
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =====================
# 状態・行動エンコード
# =====================
N_CARDS = 52
CARD2IDX = {r + s: i for i, (r, s) in enumerate((r, s) for r in RANKS for s in SUITS)}

# ロケーションカテゴリ:
# 0: unknown
# hero: 1..5 (top, mid, bot, dead, incoming)
# opp1: 6..10
# opp2: 11..15
LOC_DIM = 16

def encode_state(s: MultiOFCState) -> np.ndarray:
    """
    52枚 × 16ロケーション + turn one-hot(5) + hero-pos one-hot(3)
    """
    loc = np.zeros((N_CARDS, LOC_DIM), dtype=np.float32)
    loc[:, 0] = 1.0  # unknown に初期化

    hero_idx = s.hero_idx
    n_players = len(s.players)

    # hero のカテゴリベース
    HERO_BASE = 1
    OPP1_BASE = 6
    OPP2_BASE = 11

    def place_cards(cards, base, offset):
        for c in cards:
            ci = CARD2IDX[c]
            loc[ci, :] = 0.0
            loc[ci, base + offset] = 1.0

    # hero
    hero = s.players[hero_idx]
    place_cards(hero.top.cards, HERO_BASE, 0)
    place_cards(hero.mid.cards, HERO_BASE, 1)
    place_cards(hero.bot.cards, HERO_BASE, 2)
    place_cards(hero.dead,      HERO_BASE, 3)
    place_cards(hero.incoming,  HERO_BASE, 4)

    # opponent mapping
    opp_slots = []
    for idx in range(n_players):
        if idx == hero_idx:
            continue
        opp_slots.append(idx)
    # 最大2人分だけ扱う
    while len(opp_slots) < 2:
        opp_slots.append(None)

    # opp1
    if opp_slots[0] is not None:
        p = s.players[opp_slots[0]]
        place_cards(p.top.cards, OPP1_BASE, 0)
        place_cards(p.mid.cards, OPP1_BASE, 1)
        place_cards(p.bot.cards, OPP1_BASE, 2)
        place_cards(p.dead,      OPP1_BASE, 3)
        place_cards(p.incoming,  OPP1_BASE, 4)

    # opp2
    if opp_slots[1] is not None:
        p = s.players[opp_slots[1]]
        place_cards(p.top.cards, OPP2_BASE, 0)
        place_cards(p.mid.cards, OPP2_BASE, 1)
        place_cards(p.bot.cards, OPP2_BASE, 2)
        place_cards(p.dead,      OPP2_BASE, 3)
        place_cards(p.incoming,  OPP2_BASE, 4)

    loc_flat = loc.reshape(-1)  # 52 * 16 = 832

    # turn one-hot (0..4)
    turn_oh = np.zeros(5, dtype=np.float32)
    t = max(0, min(4, s.turn))
    turn_oh[t] = 1.0

    # hero position one-hot (最大3人想定)
    pos_oh = np.zeros(3, dtype=np.float32)
    pos_oh[min(s.hero_idx, 2)] = 1.0

    return np.concatenate([loc_flat, turn_oh, pos_oh], axis=0)

# 行動エンコード: hero のボードだけを見れば良い
def encode_action(s: MultiOFCState, action) -> np.ndarray:
    """
    初手5枚:
        - hero.incoming は 5枚
        - action は ['T','M','B',...] 長さ5
        → 1枚ごとに (カードone-hot 52 + 置き先one-hot3) = 55 を連結 → 275
    pineapple:
        - hero.incoming は 3枚
        - action は (discard_idx, p1, p2)
        → discard idx one-hot3 + 2枚分の(52+3)=55×2 = 113
        → 275まで0埋め
    """
    hero = s.players[s.hero_idx]
    if s.turn == 0:
        inc = hero.incoming
        assert len(inc) == 5
        feat = []
        for c, dst in zip(inc, action):
            card_oh = np.zeros(N_CARDS, dtype=np.float32)
            card_oh[CARD2IDX[c]] = 1.0
            dst_oh = np.zeros(3, dtype=np.float32)
            dst_oh["TMB".index(dst)] = 1.0
            feat.append(np.concatenate([card_oh, dst_oh], axis=0))  # 55
        return np.concatenate(feat, axis=0)  # 275
    else:
        inc = hero.incoming
        assert len(inc) == 3
        d, p1, p2 = action
        d_oh = np.zeros(3, dtype=np.float32)
        d_oh[d] = 1.0
        keep = [inc[i] for i in range(3) if i != d]

        def one(c, dst):
            card_oh = np.zeros(N_CARDS, dtype=np.float32)
            card_oh[CARD2IDX[c]] = 1.0
            dst_oh = np.zeros(3, dtype=np.float32)
            dst_oh["TMB".index(dst)] = 1.0
            return np.concatenate([card_oh, dst_oh], axis=0)

        a1 = one(keep[0], p1)
        a2 = one(keep[1], p2)
        vec = np.concatenate([d_oh, a1, a2], axis=0)  # 113
        pad = np.zeros(275 - 113, dtype=np.float32)
        return np.concatenate([vec, pad], axis=0)

STATE_DIM = 832 + 5 + 3  # 840
ACT_DIM   = 275
SA_DIM    = STATE_DIM + ACT_DIM

# =====================
# Qネットワーク Q(s,a)
# =====================
class QNet(nn.Module):
    def __init__(self, in_dim=SA_DIM, hidden=(512, 512, 256)):
        super().__init__()
        layers = []
        last = in_dim
        for h in hidden:
            layers += [nn.Linear(last, h), nn.ReLU()]
            last = h
        layers += [nn.Linear(last, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, sa):
        return self.net(sa).squeeze(-1)  # (B,)

# =====================
# リプレイバッファ
# =====================
Transition = namedtuple("Transition", ("s_vec", "a_vec", "r", "s2_vec", "done"))

class Replay:
    def __init__(self, cap=200_000):
        self.buf = deque(maxlen=cap)
    def __len__(self):
        return len(self.buf)
    def push(self, *args):
        self.buf.append(Transition(*args))
    def sample(self, n):
        idx = np.random.choice(len(self.buf), size=n, replace=False)
        return [self.buf[i] for i in idx]

# =====================
# ε-greedy 行動選択
# =====================
def select_action(env: OFCMultiEnv, qnet: QNet, s: MultiOFCState, epsilon: float):
    s_vec = encode_state(s)
    acts = env.legal_actions()
    if not acts:
        return None, s_vec, None

    # ε-greedy
    if random.random() < epsilon:
        a = random.choice(acts)
        a_vec = encode_action(s, a)
        return a, s_vec, a_vec

    # greedy
    with torch.no_grad():
        qs = []
        for a in acts:
            a_vec = encode_action(s, a)
            sa_np = np.concatenate([s_vec.astype(np.float32), a_vec.astype(np.float32)], axis=0)
            sa = torch.tensor(sa_np, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            q = qnet(sa).item()
            qs.append(q)
        best_idx = int(np.argmax(qs))
        a = acts[best_idx]
        a_vec = encode_action(s, a)
        return a, s_vec, a_vec

# =====================
# エピソード実行
# =====================
def play_episode(env: OFCMultiEnv, qnet: QNet, epsilon: float, gamma=0.99):
    s = env.reset()
    done = False
    total_r = 0.0
    traj = []

    while not done:
        a, s_vec, a_vec = select_action(env, qnet, s, epsilon)
        if a is None:
            # 行動無しで終局してしまう rare case
            break
        s2, r, done, _ = env.step(a)
        s2_vec = encode_state(s2)
        total_r += r
        traj.append((s_vec, a_vec, r, s2_vec, float(done)))
        s = s2

    # 将来割引は学習側で扱うので、ここでは単純にステップ毎報酬として記録
    return total_r, traj

# =====================
# 学習ループ
# =====================
def train(
    num_episodes=20000,
    n_players=2,
    hero_idx=0,
    start_learning=1024,
    batch_size=256,
    gamma=0.995,
    lr=2e-4,
    target_sync=2000,
    epsilon_start=0.7,
    epsilon_end=0.05,
    epsilon_decay=30000,
    seed=11,
    save_path="./ckpt/ofc_qnet_multi.pt",
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    env = OFCMultiEnv(n_players=n_players, hero_idx=hero_idx, seed=seed)
    qnet = QNet().to(DEVICE)
    tgt  = QNet().to(DEVICE)
    tgt.load_state_dict(qnet.state_dict())

    opt = optim.Adam(qnet.parameters(), lr=lr)
    buf = Replay(cap=300_000)

    step = 0
    log_every = 200

    for ep in range(1, num_episodes + 1):
        epsilon = epsilon_end + (epsilon_start - epsilon_end) * math.exp(-(step)/epsilon_decay)
        ret, traj = play_episode(env, qnet, epsilon, gamma=gamma)

        # バッファに追加
        for (s_vec, a_vec, r, s2_vec, done) in traj:
            buf.push(s_vec, a_vec, r, s2_vec, done)

        # 学習ステップ
        if len(buf) >= start_learning:
            batch = buf.sample(batch_size)
            s_batch  = torch.tensor(np.stack([b.s_vec  for b in batch]), dtype=torch.float32, device=DEVICE)
            a_batch  = torch.tensor(np.stack([b.a_vec  for b in batch]), dtype=torch.float32, device=DEVICE)
            r_batch  = torch.tensor(np.array([b.r      for b in batch], dtype=np.float32), device=DEVICE)
            s2_batch = torch.tensor(np.stack([b.s2_vec for b in batch]), dtype=torch.float32, device=DEVICE)
            d_batch  = torch.tensor(np.array([b.done   for b in batch], dtype=np.float32), device=DEVICE)

            sa = torch.cat([s_batch, a_batch], dim=1)
            q  = qnet(sa)  # (B,)

            with torch.no_grad():
                # 次状態での最大 Q を target ネットで計算
                q_next = []
                for i in range(batch_size):
                    s2_vec_np = s2_batch[i].cpu().numpy()
                    # s2 の「元状態オブジェクト」は持っていないが、行動のmax計算は難しいので
                    # ここでは「次状態のQを 0」として近似 (terminal のみ / 低頻度報酬ならそこそこ動く)
                    # → 本格的には rehydrate_state + legal_actions で DoubleDQN する
                    q_next.append(0.0)
                q_next = torch.tensor(q_next, dtype=torch.float32, device=DEVICE)

                y = r_batch + (1.0 - d_batch) * gamma * q_next

            loss = nn.SmoothL1Loss()(q, y)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(qnet.parameters(), 1.0)
            opt.step()

            if step % target_sync == 0:
                tgt.load_state_dict(qnet.state_dict())

        step += 1

        if ep % log_every == 0:
            print(f"[EP {ep:5d}] return={ret:+.3f} buf={len(buf)} eps={epsilon:.3f}")

    torch.save(qnet.state_dict(), save_path)
    print("saved:", save_path)

# =====================
# 評価/テスト
# =====================
def evaluate(model_path="./ckpt/ofc_qnet_multi.pt",
             n_players=2, hero_idx=0, seed=None, episodes=1000):
    env = OFCMultiEnv(n_players=n_players, hero_idx=hero_idx, seed=seed)
    model = QNet().to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    foul_cnt = 0
    ret_sum = 0.0
    legal_ret_sum = 0.0
    legal_cnt = 0

    for ep in range(episodes):
        s = env.reset()
        done = False
        total_r = 0.0
        while not done:
            s_vec = encode_state(s)
            acts = env.legal_actions()
            if not acts:
                break
            qs = []
            for a in acts:
                a_vec = encode_action(s, a)
                sa_np = np.concatenate([s_vec.astype(np.float32),
                                        a_vec.astype(np.float32)], axis=0)
                sa = torch.tensor(sa_np, dtype=torch.float32, device=DEVICE).unsqueeze(0)
                with torch.no_grad():
                    qv = model(sa).item()
                qs.append(qv)
            best_idx = int(np.argmax(qs))
            a = acts[best_idx]
            s, r, done, _ = env.step(a)
            total_r += r

        hero = s.players[s.hero_idx]
        ret_sum += total_r
        if hero.foul:
            foul_cnt += 1
        else:
            legal_cnt += 1
            legal_ret_sum += total_r

    print(f"[EVAL] episodes = {episodes}")
    print(f"[EVAL] foul rate         = {foul_cnt/episodes:.3f}")
    print(f"[EVAL] avg return (all)  = {ret_sum/episodes:.3f}")
    if legal_cnt > 0:
        print(f"[EVAL] avg return (legal only) = {legal_ret_sum/legal_cnt:.3f}")
    else:
        print("[EVAL] no legal hands 😇")

def run_test(model_path="./ckpt/ofc_qnet_multi.pt", n_players=2, hero_idx=0, seed=None):
    env = OFCMultiEnv(n_players=n_players, hero_idx=hero_idx, seed=seed)
    model = QNet().to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    print("=== 試行開始（multi） ===")
    s = env.reset()
    done = False
    total_r = 0.0

    while not done:
        s_vec = encode_state(s)
        acts = env.legal_actions()
        if not acts:
            break
        qs = []
        for a in acts:
            a_vec = encode_action(s, a)
            sa_np = np.concatenate([s_vec.astype(np.float32), a_vec.astype(np.float32)], axis=0)
            sa = torch.tensor(sa_np, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            with torch.no_grad():
                qv = model(sa).item()
            qs.append(qv)
        best_idx = int(np.argmax(qs))
        a = acts[best_idx]
        s, r, done, _ = env.step(a)
        total_r += r

    hero = s.players[s.hero_idx]
    print("報酬:", total_r, "ファウル:", hero.foul)
    print("Top役:", catname_3(*eval_3(hero.top.cards)))
    print("Mid役:", catname_5(*eval_5(hero.mid.cards)))
    print("Bot役:", catname_5(*eval_5(hero.bot.cards)))
    print("Top:", hero.top.cards)
    print("Mid:", hero.mid.cards)
    print("Bot:", hero.bot.cards)

# =====================
# CLI
# =====================
if __name__ == "__main__":
    import argparse, os
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="学習モードで実行")
    parser.add_argument("--test", action="store_true", help="テスト（1エピソード）を実行")
    parser.add_argument("--eval", type=int, default=0, help="評価を episodes 回実行")
    parser.add_argument("--episodes", type=int, default=20000)
    parser.add_argument("--n_players", type=int, default=2, help="プレイヤー人数(2〜3)")
    parser.add_argument("--hero_idx", type=int, default=0, help="学習対象プレイヤーのindex")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--model", type=str, default="./ckpt/ofc_qnet_multi.pt")
    args = parser.parse_args()

    os.makedirs("./ckpt", exist_ok=True)

    if args.eval > 0:
        evaluate(
            model_path=args.model,
            n_players=args.n_players,
            hero_idx=args.hero_idx,
            seed=args.seed,
            episodes=args.eval,
        )
    elif args.test:
        run_test(
            model_path=args.model,
            n_players=args.n_players,
            hero_idx=args.hero_idx,
            seed=args.seed,
        )
    else:
        train(
            num_episodes=args.episodes,
            n_players=args.n_players,
            hero_idx=args.hero_idx,
            seed=args.seed,
            save_path=args.model,
        )
