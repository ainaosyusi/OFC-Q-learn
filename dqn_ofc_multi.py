from __future__ import annotations

import argparse
import copy
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
# カード/行動 エンコード（Q(s,a) 用）
# =====================

N_CARDS = 52  # 既存 ckpt は 52枚前提
CARD2IDX: Dict[str, int] = {r + s: i for i, (r, s) in enumerate((r, s) for r in RANKS for s in SUITS)}

# ロケーションカテゴリ: 0=unknown, 1..5 hero(top/mid/bot/unused/incoming), 6..10 opp1, 11..15 opp2
LOC_DIM = 16


def encode_state(s: MultiOFCState) -> np.ndarray:
    """52枚×16ロケーション + turn one-hot(5) + hero-pos one-hot(3) + pad(21) = 861"""

    loc = np.zeros((N_CARDS, LOC_DIM), dtype=np.float32)
    loc[:, 0] = 1.0

    hero = s.hero
    opps = list(s.opps)

    HERO_BASE = 1
    OPP1_BASE = 6
    OPP2_BASE = 11

    def place_cards(cards: List[str], base: int, offset: int):
        for c in cards:
            if c not in CARD2IDX:
                continue
            ci = CARD2IDX[c]
            loc[ci, :] = 0.0
            loc[ci, base + offset] = 1.0

    # hero board + current hand を incoming として配置
    place_cards(hero.top, HERO_BASE, 0)
    place_cards(hero.mid, HERO_BASE, 1)
    place_cards(hero.bot, HERO_BASE, 2)
    place_cards([], HERO_BASE, 3)  # dead 行は未使用
    place_cards(s.hand, HERO_BASE, 4)

    # opp1
    if len(opps) > 0:
        p = opps[0]
        place_cards(p.top, OPP1_BASE, 0)
        place_cards(p.mid, OPP1_BASE, 1)
        place_cards(p.bot, OPP1_BASE, 2)
    # opp2
    if len(opps) > 1:
        p = opps[1]
        place_cards(p.top, OPP2_BASE, 0)
        place_cards(p.mid, OPP2_BASE, 1)
        place_cards(p.bot, OPP2_BASE, 2)

    loc_flat = loc.reshape(-1)  # 832

    turn_oh = np.zeros(5, dtype=np.float32)
    turn_oh[max(0, min(4, s.turn))] = 1.0

    pos_oh = np.zeros(3, dtype=np.float32)
    pos_oh[0] = 1.0  # hero は常に idx=0 扱い

    pad = np.zeros(21, dtype=np.float32)  # ckpt 入力長に合わせる

    return np.concatenate([loc_flat, turn_oh, pos_oh, pad], axis=0)


def encode_action(s: MultiOFCState, action: Tuple[int, str]) -> np.ndarray:
    """(card_idx, row) を固定長 275 に埋め込む"""
    vec = np.zeros(275, dtype=np.float32)
    idx, row = action
    if 0 <= idx < len(s.hand):
        c = s.hand[idx]
        if c in CARD2IDX:
            vec[CARD2IDX[c]] = 1.0
    row_oh = ["T", "M", "B"]
    if row in row_oh:
        vec[N_CARDS + row_oh.index(row)] = 1.0
    return vec


def legal_actions_from_state(s: MultiOFCState) -> List[Tuple[int, str]]:
    acts: List[Tuple[int, str]] = []
    hero = s.hero
    for i, _ in enumerate(s.hand):
        for row in ["T", "M", "B"]:
            if hero.can_place(row):
                acts.append((i, row))
    return acts


# =====================
# DQN
# =====================

Transition = namedtuple("Transition", ("s_vec", "a_vec", "r", "ns_state", "done"))

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
    def __init__(self, in_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)

@torch.no_grad()
def select_action(env: OFCMultiEnv, qnet: QNet, s: MultiOFCState, eps: float):
    s_vec = encode_state(s)
    legal = legal_actions_from_state(s)
    if not legal:
        return None, s_vec, None

    if random.random() < eps:
        a = random.choice(legal)
        return a, s_vec, encode_action(s, a)

    best_q = None
    best_a = None
    for a in legal:
        a_vec = encode_action(s, a)
        sa = torch.tensor(np.concatenate([s_vec, a_vec], axis=0), dtype=torch.float32, device=DEVICE).unsqueeze(0)
        q = qnet(sa).item()
        if (best_q is None) or (q > best_q):
            best_q = q
            best_a = a
    return best_a, s_vec, encode_action(s, best_a)

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

    # state/action dim を一度作って決める
    s0 = env.reset()
    state_dim = encode_state(s0).shape[0]
    action_dim = encode_action(s0, (0, "T")).shape[0]
    sa_dim = state_dim + action_dim

    qnet = QNet(sa_dim).to(DEVICE)
    tgt = QNet(sa_dim).to(DEVICE)
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
            eps = eps_by_ep(ep)
            action, s_vec, a_vec = select_action(env, qnet, s, eps)
            if action is None:
                break

            ns, r, done, info = env.step(action)
            ep_ret += r
            hero_foul = bool(info.get("hero_foul", False))

            buf.push(s_vec, a_vec, r, copy.deepcopy(ns), done)
            s = ns

            # learn
            if len(buf) >= max(warmup, batch_size):
                batch = buf.sample(batch_size)

                bs = torch.tensor(
                    np.stack([np.concatenate([t.s_vec, t.a_vec], axis=0) for t in batch]),
                    dtype=torch.float32,
                    device=DEVICE,
                )
                br = torch.tensor([t.r for t in batch], dtype=torch.float32, device=DEVICE)
                bd = torch.tensor([t.done for t in batch], dtype=torch.float32, device=DEVICE)

                q = qnet(bs)

                with torch.no_grad():
                    targets = []
                    for t, done_flag in zip(batch, bd):
                        if done_flag:
                            targets.append(t.r)
                            continue
                        ns_vec = encode_state(t.ns_state)
                        legal = legal_actions_from_state(t.ns_state)
                        if not legal:
                            targets.append(t.r)
                            continue
                        qs = []
                        for a in legal:
                            a_vec = encode_action(t.ns_state, a)
                            sa = torch.tensor(
                                np.concatenate([ns_vec, a_vec], axis=0),
                                dtype=torch.float32,
                                device=DEVICE,
                            ).unsqueeze(0)
                            qs.append(tgt(sa).item())
                        targets.append(t.r + gamma * max(qs))
                    y = torch.tensor(targets, dtype=torch.float32, device=DEVICE)

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
    state_dim = encode_state(s0).shape[0]
    action_dim = encode_action(s0, (0, "T")).shape[0]
    sa_dim = state_dim + action_dim

    qnet = QNet(sa_dim).to(DEVICE)
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
            action, _, _ = select_action(env, qnet, s, eps=0.0)
            if action is None:
                break
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
