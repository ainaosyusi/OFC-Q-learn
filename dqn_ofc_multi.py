from __future__ import annotations

import argparse
import math
import os
import random
from collections import deque, namedtuple
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from multi_ofc_env import OFCMultiEnv, MultiOFCState, PlayerBoard, RANKS, SUITS

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =====================
# カード/行動 エンコード
# =====================

# Joker を 1種類 "X" として扱う（env 側が "X" を使う前提）
JOKER = "X"

RANKS52 = RANKS
SUITS52 = SUITS
DECK52: List[str] = [r + s for r in RANKS52 for s in SUITS52]

CARD_LIST: List[str] = DECK52 + [JOKER]
N_CARDS = len(CARD_LIST)  # 53

CARD2IDX: Dict[str, int] = {c: i for i, c in enumerate(CARD_LIST)}

def card_to_idx(c: str) -> int:
    # 万一未知が来たら落とす（黙って無視すると学習が壊れる）
    if c not in CARD2IDX:
        raise KeyError(f"Unknown card: {c}")
    return CARD2IDX[c]

# (card_idx, row) の全組み合わせ（初手は最大5枚、それ以降は3枚なので idx=0..4 を確保）
ACTIONS: List[Tuple[int, str]] = []
for ci in range(5):
    for row in ["T", "M", "B"]:
        ACTIONS.append((ci, row))

# 既存モデルとの互換性のため出力次元は 259 のまま確保し、実際に使うのは先頭 15 個のみ
N_REAL_ACTIONS = len(ACTIONS)  # 15
N_ACTIONS = 259


# =====================
# 状態エンコード
# =====================

def encode_state(s: MultiOFCState) -> np.ndarray:
    """
    env の MultiOFCState 前提:
      - s.turn: int  (0..4)
      - s.hand: List[str]  (turn0=5枚, それ以降=3枚)
      - s.hero: PlayerBoard
      - s.opps: List[PlayerBoard]
    """
    players: List[PlayerBoard] = [s.hero] + list(s.opps)
    n_players = len(players)

    # 1プレイヤあたり固定 13スロット (Top3 + Mid5 + Bot5)
    SLOTS_PER_PLAYER = 13
    MAX_CUR = 5  # current_cards は最大 5 を確保

    # 追加のスカラー特徴
    EXTRA = 2  # (turn/4, is_turn0)

    dim = n_players * SLOTS_PER_PLAYER * N_CARDS + MAX_CUR * N_CARDS + EXTRA
    vec = np.zeros(dim, dtype=np.float32)

    def put(base: int, card: str):
        vec[base + card_to_idx(card)] = 1.0

    # players 部分（hero を先頭、その後に opps を列挙）
    for pi, pb in enumerate(players):
        # スロット順: top0..2, mid0..4, bot0..4
        slot_cards: List[str] = []
        slot_cards += list(pb.top)[:3]
        slot_cards += list(pb.mid)[:5]
        slot_cards += list(pb.bot)[:5]

        # 足りない分は空スロット扱い（0のまま）
        for si, card in enumerate(slot_cards):
            base = (pi * SLOTS_PER_PLAYER + si) * N_CARDS
            put(base, card)

    # current_cards 部分（最大5枠）
    offset_cur = n_players * SLOTS_PER_PLAYER * N_CARDS
    for i, card in enumerate(s.hand[:MAX_CUR]):
        base = offset_cur + i * N_CARDS
        put(base, card)

    # extra
    offset_extra = offset_cur + MAX_CUR * N_CARDS
    vec[offset_extra + 0] = float(s.turn) / 4.0
    vec[offset_extra + 1] = 1.0 if s.turn == 0 else 0.0

    return vec


def legal_action_indices(env: OFCMultiEnv) -> List[int]:
    """env.available_actions() の戻り (idx,row) を固定 index へ変換"""
    acts = env.available_actions()
    mp = {a: i for i, a in enumerate(ACTIONS)}
    out: List[int] = []
    for a in acts:
        if a in mp:
            out.append(mp[a])
    return out


def decode_action(env: OFCMultiEnv, aidx: int):
    """固定index -> env に渡す action object"""
    if aidx < N_REAL_ACTIONS:
        return ACTIONS[aidx]
    # ありえない場合は最初の合法行動を返す（マスクされている前提）
    legal = env.available_actions()
    return legal[0] if legal else (0, "T")


# =====================
# DQN
# =====================

Transition = namedtuple("Transition", ("s", "a", "r", "ns", "done"))

class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buf = deque(maxlen=capacity)

    def push(self, *args):
        self.buf.append(Transition(*args))

    def sample(self, batch_size: int):
        idx = np.random.choice(len(self.buf), batch_size, replace=False)
        return [self.buf[i] for i in idx]

    def __len__(self):
        return len(self.buf)

class QNet(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, out_dim),
        )

    def forward(self, x):
        return self.net(x)

@torch.no_grad()
def select_action(qnet: QNet, svec: np.ndarray, legal_idxs: List[int], eps: float) -> int:
    if random.random() < eps:
        return random.choice(legal_idxs)

    x = torch.tensor(svec, dtype=torch.float32, device=DEVICE).unsqueeze(0)
    q = qnet(x).squeeze(0).detach().cpu().numpy()

    # illegal を -inf に
    mask = np.full_like(q, -1e9, dtype=np.float32)
    mask[legal_idxs] = q[legal_idxs]
    return int(mask.argmax())

def train(
    episodes: int,
    n_players: int,
    hero_idx: int,
    seed: int,
    n_jokers: int,
    model_path: str,
    log_interval: int = 200,
    save_interval: int = 5000,
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    env = OFCMultiEnv(n_players=n_players, hero_idx=hero_idx, seed=seed, n_jokers=n_jokers)

    # state dim を一度作って決める
    s0 = env.reset()
    in_dim = encode_state(s0).shape[0]

    qnet = QNet(in_dim, N_ACTIONS).to(DEVICE)
    tgt = QNet(in_dim, N_ACTIONS).to(DEVICE)
    tgt.load_state_dict(qnet.state_dict())

    opt = optim.Adam(qnet.parameters(), lr=1e-4)
    buf = ReplayBuffer(capacity=200_000)

    gamma = 0.99
    batch_size = 256
    warmup = 2000
    target_sync = 2000

    eps_start = 1.0
    eps_end = 0.05
    eps_decay = 200_000  # episodes で線形減衰

    def eps_by_ep(ep: int) -> float:
        if ep >= eps_decay:
            return eps_end
        return eps_start + (eps_end - eps_start) * (ep / eps_decay)

    step_count = 0

    for ep in range(1, episodes + 1):
        s = env.reset()
        done = False
        ep_ret = 0.0
        hero_foul = False

        while not done:
            svec = encode_state(s)
            legal_idxs = legal_action_indices(env)

            eps = eps_by_ep(ep)
            aidx = select_action(qnet, svec, legal_idxs, eps)
            action = decode_action(env, aidx)

            ns, r, done, info = env.step(action)
            ep_ret += r
            hero_foul = bool(info.get("hero_foul", False))

            buf.push(svec, aidx, r, encode_state(ns), done)
            s = ns

            # learn
            if len(buf) >= max(warmup, batch_size):
                batch = buf.sample(batch_size)
                bs = torch.tensor(np.stack([t.s for t in batch]), dtype=torch.float32, device=DEVICE)
                ba = torch.tensor([t.a for t in batch], dtype=torch.int64, device=DEVICE).unsqueeze(1)
                br = torch.tensor([t.r for t in batch], dtype=torch.float32, device=DEVICE).unsqueeze(1)
                bns = torch.tensor(np.stack([t.ns for t in batch]), dtype=torch.float32, device=DEVICE)
                bd = torch.tensor([t.done for t in batch], dtype=torch.float32, device=DEVICE).unsqueeze(1)

                q = qnet(bs).gather(1, ba)

                with torch.no_grad():
                    nq = tgt(bns).max(dim=1, keepdim=True)[0]
                    y = br + (1.0 - bd) * gamma * nq

                loss = nn.functional.smooth_l1_loss(q, y)

                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(qnet.parameters(), 5.0)
                opt.step()

                step_count += 1
                if step_count % target_sync == 0:
                    tgt.load_state_dict(qnet.state_dict())

        if ep % log_interval == 0:
            print(f"[TRAIN] ep={ep} eps={eps_by_ep(ep):.3f} return={ep_ret:.3f} foul={hero_foul} buf={len(buf)}")

        if ep % save_interval == 0:
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            torch.save(qnet.state_dict(), model_path)
            print(f"[SAVE] {model_path}")

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(qnet.state_dict(), model_path)
    print(f"[DONE] saved: {model_path}")

@torch.no_grad()
def evaluate(
    model_path: str,
    n_players: int,
    hero_idx: int,
    seed: int,
    n_jokers: int,
    episodes: int,
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    env = OFCMultiEnv(n_players=n_players, hero_idx=hero_idx, seed=seed, n_jokers=n_jokers)

    s0 = env.reset()
    in_dim = encode_state(s0).shape[0]

    qnet = QNet(in_dim, N_ACTIONS).to(DEVICE)
    qnet.load_state_dict(torch.load(model_path, map_location=DEVICE))
    qnet.eval()

    total = 0.0
    foul_cnt = 0

    for _ in range(episodes):
        s = env.reset()
        done = False
        ep_ret = 0.0
        hero_foul = False

        while not done:
            svec = encode_state(s)
            legal_idxs = legal_action_indices(env)
            aidx = select_action(qnet, svec, legal_idxs, eps=0.0)
            action = decode_action(env, aidx)
            s, r, done, info = env.step(action)
            ep_ret += r
            hero_foul = bool(info.get("hero_foul", False))

        total += ep_ret
        foul_cnt += 1 if hero_foul else 0

    print(f"[EVAL] episodes={episodes} foul_rate={foul_cnt/episodes:.3f} avg_return={total/episodes:.3f}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--episodes", type=int, default=2000)
    ap.add_argument("--eval", type=int, default=0, help="number of eval episodes")
    ap.add_argument("--n_players", type=int, default=3)
    ap.add_argument("--hero_idx", type=int, default=0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--n_jokers", type=int, default=0)
    ap.add_argument("--model", type=str, default="./ckpt/model.pt")
    args = ap.parse_args()

    if args.train:
        train(
            episodes=args.episodes,
            n_players=args.n_players,
            hero_idx=args.hero_idx,
            seed=args.seed,
            n_jokers=args.n_jokers,
            model_path=args.model,
        )

    if args.eval and args.eval > 0:
        evaluate(
            model_path=args.model,
            n_players=args.n_players,
            hero_idx=args.hero_idx,
            seed=args.seed,
            n_jokers=args.n_jokers,
            episodes=args.eval,
        )

if __name__ == "__main__":
    main()
