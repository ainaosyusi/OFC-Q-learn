# dqn_ofc.py (patched)
import os, math, random
from collections import deque, namedtuple, Counter
from typing import List, Tuple, Any, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from ofc_env import (
    OFCEnv,
    OFCState,
    Line,
    RANKS,
    SUITS,
    RANK2IDX,
    eval_3,
    eval_5,
    catname_3,
    catname_5,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------
# 1) 状態と行動のエンコード
# ---------------------------

CARD2IDX = {r+s: i for i, (r, s) in enumerate((r, s) for r in RANKS for s in SUITS)}  # 52枚: 0..51

# 位置カテゴリ: 0=未使用(山/デック), 1=Top, 2=Mid, 3=Bot, 4=Dead, 5=Incoming
LOC_DIM = 6
N_CARDS = 52

def encode_state(s: OFCState) -> np.ndarray:
    """
    盤面を固定長ベクトルへ
      - 52枚 × 6ロケーションのワンホット（実際は各カードちょうど1カテゴリに所属）
      - turn (0..4) を one-hot(5)
      - 各ラインの空き (top, mid, bot) を正規化
    出力次元: 312 + 5 + 3 = 320
    """
    loc = np.zeros((N_CARDS, LOC_DIM), dtype=np.float32)  # (52, 6)
    loc[:, 0] = 1.0  # 未使用に初期化

    def place(cards: List[str], cat: int):
        for c in cards:
            idx = CARD2IDX[c]
            loc[idx, :] = 0.0
            loc[idx, cat] = 1.0

    place(s.top.cards, 1)
    place(s.mid.cards, 2)
    place(s.bot.cards, 3)
    place(s.dead, 4)
    place(s.incoming, 5)

    loc_flat = loc.reshape(-1)  # 52*6 = 312
    turn_oh = np.zeros(5, dtype=np.float32)
    t = max(0, min(4, s.turn))
    turn_oh[t] = 1.0

    spaces = np.array([
        (3 - len(s.top.cards))/3.0,
        (5 - len(s.mid.cards))/5.0,
        (5 - len(s.bot.cards))/5.0
    ], dtype=np.float32)

    return np.concatenate([loc_flat, turn_oh, spaces], axis=0)  # 320

def encode_action(s: OFCState, a: Union[List[str], Tuple[int,str,str]]) -> np.ndarray:
    """
    行動 a を固定長へ
    - turn==0: 5枚の (カードID one-hot 52 + 置き先 one-hot 3) を連結
    - turn>=1: (discard idx one-hot 3) + 2枚分の (カードID52 + 置き先3) を連結
    固定長: 初手=275, 以降=113 → パディングして 275 に統一
    """
    if s.turn == 0:
        inc = s.incoming  # 長さ5
        feat = []
        for c, dst in zip(inc, a):  # a は ['T','M','B',...]
            card_oh = np.zeros(N_CARDS, dtype=np.float32); card_oh[CARD2IDX[c]] = 1.0
            dst_oh  = np.zeros(3, dtype=np.float32);       dst_oh["TMB".index(dst)] = 1.0
            feat.append(np.concatenate([card_oh, dst_oh], axis=0))  # 55
        act_vec = np.concatenate(feat, axis=0)  # 275
        return act_vec

    else:
        inc = s.incoming  # 長さ3
        discard_idx, p1, p2 = a[0], a[1], a[2]
        d_oh = np.zeros(3, dtype=np.float32); d_oh[discard_idx] = 1.0
        keep = [inc[i] for i in range(3) if i != discard_idx]

        def one(c, dst):
            card_oh = np.zeros(N_CARDS, dtype=np.float32); card_oh[CARD2IDX[c]] = 1.0
            dst_oh  = np.zeros(3, dtype=np.float32);       dst_oh["TMB".index(dst)] = 1.0
            return np.concatenate([card_oh, dst_oh], axis=0)  # 55

        a1 = one(keep[0], p1); a2 = one(keep[1], p2)
        vec = np.concatenate([d_oh, a1, a2], axis=0)  # 113
        pad = np.zeros(275 - 113, dtype=np.float32)
        return np.concatenate([vec, pad], axis=0)

STATE_DIM = 320
ACT_DIM   = 275
SA_DIM    = STATE_DIM + ACT_DIM

# ---------------------------
# 2) Qネットワーク（Q(s,a))
# ---------------------------
class QNet(nn.Module):
    def __init__(self, in_dim=SA_DIM, hidden=(512, 512, 256)):
        super().__init__()
        layers = []
        last = in_dim
        for h in hidden:
            layers += [nn.Linear(last, h), nn.ReLU()]
            last = h
        layers += [nn.Linear(last, 1)]  # Q(s,a): スカラー
        self.net = nn.Sequential(*layers)

    def forward(self, sa):
        return self.net(sa).squeeze(-1)  # (B,)

# ---------------------------
# 3) リプレイバッファ（ベクトルを格納：ミューテーション安全）
# ---------------------------
Transition = namedtuple('Transition', ('s_vec', 'a_vec', 'r', 's2_vec', 'done'))

class Replay:
    def __init__(self, cap=200_000):
        self.buf = deque(maxlen=cap)
    def push(self, *args):
        self.buf.append(Transition(*args))
    def sample(self, n):
        idx = np.random.choice(len(self.buf), size=n, replace=False)
        return [self.buf[i] for i in idx]
    def __len__(self):
        return len(self.buf)

# ---------------------------
# Early-foul Monte Carlo mask
# ---------------------------
from copy import deepcopy
def _known_cards_from_state(s: OFCState):
    return set(s.top.cards + s.mid.cards + s.bot.cards + s.dead + s.incoming)

def _random_completion_and_foul(s: OFCState, samples_deck, rng):
    # 軽量コピー
    st = OFCState()
    st.top.cards = list(s.top.cards)
    st.mid.cards = list(s.mid.cards)
    st.bot.cards = list(s.bot.cards)
    st.dead      = list(s.dead)
    st.incoming  = list(s.incoming)
    st.turn      = s.turn
    st.foul      = False

    deck = list(samples_deck)  # 残り札プール

    def rnd_place(card):
        choices = []
        if len(st.top.cards) < 3: choices.append("T")
        if len(st.mid.cards) < 5: choices.append("M")
        if len(st.bot.cards) < 5: choices.append("B")
        if not choices:
            return
        dst = rng.choice(choices)
        if dst == "T": st.top.cards.append(card)
        elif dst == "M": st.mid.cards.append(card)
        else: st.bot.cards.append(card)

    if len(st.incoming) == 5 and st.turn == 0:
        for c in st.incoming:
            rnd_place(c)
        st.incoming = []
        st.turn = 1

    while True:
        if len(st.top.cards)==3 and len(st.mid.cards)==5 and len(st.bot.cards)==5:
            break
        if not deck: break
        draw = []
        for _ in range(3):
            if deck:
                draw.append(deck.pop())
        if not draw: break
        discard = rng.randrange(len(draw))
        keep = [draw[i] for i in range(len(draw)) if i != discard]
        for c in keep:
            rnd_place(c)

    # foul 判定（厳密 Top<Mid と Mid<Bot）
    from ofc_env import eval_3, eval_5, FIVE_ORDER, THREE_ORDER
    if len(st.top.cards)==3 and len(st.mid.cards)==5 and len(st.bot.cards)==5:
        wt_top, tb_top = eval_3(st.top.cards)
        wt_mid, tb_mid = eval_5(st.mid.cards)
        wt_bot, tb_bot = eval_5(st.bot.cards)

        # Mid<Bot
        def lt_5(a,b):
            wa,ta = a; wb,tb = b
            if wa!=wb: return wa<wb
            return ta<tb
        if not lt_5((wt_mid,tb_mid),(wt_bot,tb_bot)):
            return True

        # Top<Mid 厳密判定
        if wt_top == THREE_ORDER["HIGH"]:
            return not (wt_mid >= FIVE_ORDER["PAIR"])
        if wt_top == THREE_ORDER["PAIR"]:
            if wt_mid < FIVE_ORDER["PAIR"]:
                return True
            if wt_mid >= FIVE_ORDER["TWO_PAIR"]:
                return False
            return not (tb_mid[0] > tb_top[0])
        if wt_top == THREE_ORDER["SET"]:
            if wt_mid < FIVE_ORDER["TRIPS"]:
                return True
            if wt_mid == FIVE_ORDER["TRIPS"]:
                return not (tb_mid[0] > tb_top[0])
            return False
    return False

def filter_by_foul_montecarlo(env: OFCEnv, acts, samples=12, threshold=0.7, seed=777):
    rng = random.Random(seed)
    s0 = env.state
    assert s0 is not None
    from ofc_env import DECK52
    seen = _known_cards_from_state(s0)
    pool = [c for c in DECK52 if c not in seen]

    kept = []
    for a in acts:
        st = OFCState()
        st.top.cards = list(s0.top.cards)
        st.mid.cards = list(s0.mid.cards)
        st.bot.cards = list(s0.bot.cards)
        st.dead      = list(s0.dead)
        st.incoming  = list(s0.incoming)
        st.turn      = s0.turn
        st.foul      = False

        if st.turn == 0:
            for card, dst in zip(st.incoming, a):
                if dst == "T": st.top.cards.append(card)
                elif dst == "M": st.mid.cards.append(card)
                else: st.bot.cards.append(card)
            st.incoming = []
            st.turn = 1
        else:
            d, p1, p2 = a[0], a[1], a[2]
            discard_card = st.incoming[d]
            st.dead.append(discard_card)
            keep = [st.incoming[i] for i in range(3) if i != d]
            if p1 == "T": st.top.cards.append(keep[0])
            elif p1 == "M": st.mid.cards.append(keep[0])
            else: st.bot.cards.append(keep[0])
            if p2 == "T": st.top.cards.append(keep[1])
            elif p2 == "M": st.mid.cards.append(keep[1])
            else: st.bot.cards.append(keep[1])
            st.incoming = []

        if len(st.top.cards)>3 or len(st.mid.cards)>5 or len(st.bot.cards)>5:
            continue

        fouls = 0
        for _ in range(samples):
            fouls += int(_random_completion_and_foul(st, pool, rng))
        rate = fouls / max(1, samples)
        if rate < threshold:
            kept.append(a)

    return kept if kept else acts


# ---------------------------
# Heuristic action prior
# ---------------------------
def _preview_state_after_action(state: OFCState, action):
    """軽量コピー後に action を適用し、盤面のみ更新した状態を返す"""

    st = OFCState()
    st.top.cards = list(state.top.cards)
    st.mid.cards = list(state.mid.cards)
    st.bot.cards = list(state.bot.cards)
    st.dead = list(state.dead)
    st.incoming = list(state.incoming)
    st.turn = state.turn
    st.foul = state.foul

    if state.turn == 0:
        for card, dst in zip(state.incoming, action):
            if dst == "T":
                st.top.cards.append(card)
            elif dst == "M":
                st.mid.cards.append(card)
            else:
                st.bot.cards.append(card)
        st.incoming = []
        st.turn = 1
    else:
        d, p1, p2 = action
        cards = list(state.incoming)
        if not cards:
            return st
        discard_card = cards[d]
        st.dead.append(discard_card)
        keep = [cards[i] for i in range(len(cards)) if i != d]
        for card, dst in zip(keep, (p1, p2)):
            if dst == "T":
                st.top.cards.append(card)
            elif dst == "M":
                st.mid.cards.append(card)
            else:
                st.bot.cards.append(card)
        st.incoming = []
        st.turn = min(4, state.turn + 1)

    return st


def _line_summary(cards, cap, is_top=False):
    ranks = [RANK2IDX[c[0]] for c in cards]
    suits = Counter([c[1] for c in cards])
    cnt = Counter(ranks)
    pair_ranks = sorted([r for r, c in cnt.items() if c >= 2], reverse=True)
    trips_ranks = sorted([r for r, c in cnt.items() if c >= 3], reverse=True)
    fill = len(cards) / cap
    strength = sum(ranks)
    if trips_ranks:
        strength += 30 + trips_ranks[0] * 1.5
    elif pair_ranks:
        strength += 12 + pair_ranks[0]
    flush_potential = max(suits.values()) if suits else 0
    high = max(ranks) if ranks else -1
    two_pair = len(pair_ranks) >= 2 if not is_top else False
    return {
        "fill": fill,
        "pair": pair_ranks[0] if pair_ranks else None,
        "trips": trips_ranks[0] if trips_ranks else None,
        "two_pair": two_pair,
        "strength": strength,
        "flush": flush_potential,
        "high": high,
        "len": len(cards),
        "space": cap - len(cards),
    }


def _heuristic_board_score(st: OFCState) -> float:
    top = _line_summary(st.top.cards, 3, is_top=True)
    mid = _line_summary(st.mid.cards, 5)
    bot = _line_summary(st.bot.cards, 5)

    score = 0.0

    # 基本：ボトム重視だがミドル/トップも埋める
    score += 2.0 * bot["fill"] + 1.4 * mid["fill"] + 0.8 * top["fill"]

    # ボトムに偏り過ぎると罰
    score -= 1.3 * max(0.0, bot["fill"] - mid["fill"] - 0.25)
    score -= 0.8 * max(0.0, mid["fill"] - top["fill"] - 0.35)

    # ミドルは常にボトムより弱く
    if bot["fill"] >= 0.4 and mid["fill"] >= 0.4 and mid["strength"] >= bot["strength"]:
        score -= 1.5

    # ミドルに弱ペアを置いたら罰
    if mid["pair"] is not None and mid["pair"] < RANK2IDX["Q"] and mid["fill"] < 1.0:
        score -= 1.1

    # ボトムでセット / 2ペア確定なら加点
    if bot["trips"] is not None:
        score += 0.9 + 0.05 * bot["trips"]
    elif bot["two_pair"]:
        score += 0.6
    elif bot["pair"] is not None:
        score += 0.35 + 0.03 * bot["pair"]

    # Topの扱い：QQ+で大きく加点、ただしミドル/ボトムが未整備なら抑制
    if top["pair"] is not None:
        if top["pair"] >= RANK2IDX["Q"]:
            if mid["fill"] >= 0.5 and bot["fill"] >= 0.5:
                score += 1.2 + 0.08 * top["pair"]
            else:
                score += 0.3  # 構えるだけ
        elif top["fill"] < 1.0 and (mid["fill"] < 0.4 or bot["fill"] < 0.4):
            score -= 0.8

    # Topがハイカード中心なら微加点
    if top["high"] >= RANK2IDX.get("K", 11):
        score += 0.2
    if top["high"] >= RANK2IDX.get("A", 12) and top["len"] >= 2:
        score += 0.2

    # Flush ドロー維持
    if mid["flush"] >= 3 and mid["space"] >= 2:
        score += 0.25
    if bot["flush"] >= 3 and bot["space"] >= 2:
        score += 0.3

    # Top だけが進みすぎている場合の罰
    score -= 0.4 * max(0, top["len"] - mid["len"] - 1)
    score -= 0.4 * max(0, top["len"] - bot["len"] - 1)

    return score


def heuristic_rank_actions(state: OFCState, acts, keep: int = 64):
    scored = []
    for a in acts:
        st = _preview_state_after_action(state, a)
        scored.append((_heuristic_board_score(st), a))
    scored.sort(key=lambda x: x[0], reverse=True)
    if keep is not None and keep > 0 and len(scored) > keep:
        scored = scored[:keep]
    return [a for _, a in scored]

# ---------------------------
# 4) ε-greedy（可変行動：候補から選ぶ）
# ---------------------------
def select_action(env: OFCEnv, qnet: QNet, state_vec: np.ndarray, epsilon: float, max_branch=96, heuristic_keep=64):
    acts = env.legal_actions(max_samples=max_branch)
    acts = filter_by_foul_montecarlo(env, acts, samples=12, threshold=0.7, seed=777)  # early-foul mask
    if env.state is not None:
        keep = None
        if heuristic_keep is not None:
            keep = min(len(acts), heuristic_keep)
        acts = heuristic_rank_actions(env.state, acts, keep=keep)

    if not acts:
        return None, acts

    if random.random() < epsilon:
        a = random.choice(acts)
        return a, acts

    with torch.no_grad():
        qs = []
        for a in acts:
            a_vec = encode_action(env.state, a)
            sa_np = np.concatenate([np.asarray(state_vec, dtype=np.float32),
                                    np.asarray(a_vec,    dtype=np.float32)], axis=0)  # (SA_DIM,)
            sa = torch.tensor(sa_np, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            q = qnet(sa)  # (1,)
            qs.append(q.item())
        best_idx = int(np.argmax(qs))
        return acts[best_idx], acts

# ---------------------------
# 5) 1エピソード self-play 収集（ベクトルを即時格納）
# ---------------------------
def play_episode(env: OFCEnv, qnet: QNet, epsilon: float, gamma=0.99, max_branch=96):
    s = env.reset()
    done = False
    ep_trans = []
    total_r = 0.0

    # 初手
    s_vec = encode_state(s)
    a, _ = select_action(env, qnet, s_vec, epsilon, max_branch=max_branch)
    a_vec = encode_action(s, a)              # ミューテーション前にベクトル化
    s2, r, done, _ = env.step(a)
    s2_vec = encode_state(s2)
    total_r += r
    ep_trans.append((s_vec, a_vec, r, s2_vec, float(done)))

    # pineapple 4ターンまで
    while not done:
        s = s2
        s_vec = encode_state(s)
        a, _ = select_action(env, qnet, s_vec, epsilon, max_branch=max_branch)
        a_vec = encode_action(s, a)          # ここも直後にベクトル化
        s2, r, done, _ = env.step(a)
        s2_vec = encode_state(s2)
        total_r += r
        ep_trans.append((s_vec, a_vec, r, s2_vec, float(done)))

    return total_r, ep_trans

# ---------------------------
# 6) s_vec → OFCState 再構成（ターゲット計算用）
# ---------------------------
def rehydrate_state(s_vec: np.ndarray) -> OFCState:
    loc = s_vec[:312].reshape(52,6)
    turn_oh = s_vec[312:317]
    turn = int(np.argmax(turn_oh))

    def idx_to_card(i: int) -> str:
        r = i // 4
        su = i % 4
        return RANKS[r] + SUITS[su]

    def where(cat: int) -> List[str]:
        idxs = np.where(loc[:,cat] > 0.5)[0].tolist()
        return [idx_to_card(i) for i in idxs]

    s = OFCState()
    s.top.cards = where(1)
    s.mid.cards = where(2)
    s.bot.cards = where(3)
    s.dead      = where(4)
    s.incoming  = where(5)
    s.turn      = turn
    s.foul      = False
    return s

# ---------------------------
# 7) 学習ループ（Double DQNに変更）
# ---------------------------
def train(num_episodes=2000,
          start_learning=512,
          batch_size=256,
          gamma=0.995,
          lr=2e-4,
          target_sync=2000,
          epsilon_start=0.7,
          epsilon_end=0.05,
          epsilon_decay=20000,
          max_branch=128,
          seed=11,
          resume=False,
          save_path="./ckpt/ofc_qnet.pt"):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

    env = OFCEnv(seed=seed)
    qnet = QNet().to(DEVICE)
    tgt  = QNet().to(DEVICE)
    if resume and os.path.exists(save_path):
        qnet.load_state_dict(torch.load(save_path, map_location=DEVICE))
        print(f"✅ Resumed from {save_path}")
    tgt.load_state_dict(qnet.state_dict())

    opt = optim.Adam(qnet.parameters(), lr=lr)
    buf = Replay(cap=300_000)

    step = 0
    log_every = 200
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    for ep in range(1, num_episodes+1):
        epsilon = epsilon_end + (epsilon_start - epsilon_end) * math.exp(-(step)/epsilon_decay)
        ret, traj = play_episode(env, qnet, epsilon, gamma=gamma, max_branch=max_branch)

        # バッファにpush（すでにベクトル化済みで安全）
        for (s_vec, a_vec, r, s2_vec, done) in traj:
            buf.push(s_vec, a_vec, r, s2_vec, done)

        # 学習
        if len(buf) >= start_learning:
            batch = buf.sample(batch_size)
            s_batch  = torch.tensor(np.stack([b.s_vec  for b in batch]), dtype=torch.float32, device=DEVICE)
            a_batch  = torch.tensor(np.stack([b.a_vec  for b in batch]), dtype=torch.float32, device=DEVICE)
            r_batch  = torch.tensor(np.array([b.r      for b in batch], dtype=np.float32), device=DEVICE)
            s2_batch = torch.tensor(np.stack([b.s2_vec for b in batch]), dtype=torch.float32, device=DEVICE)
            d_batch  = torch.tensor(np.array([b.done   for b in batch], dtype=np.float32), device=DEVICE)

            # Q(s,a)
            sa = torch.cat([s_batch, a_batch], dim=1)  # (B, SA_DIM)
            q = qnet(sa)  # (B,)

            # Double DQN target
            with torch.no_grad():
                q_next = []
                for i in range(batch_size):
                    s2_np = s2_batch[i].cpu().numpy()
                    s2_obj = rehydrate_state(s2_np)
                    env_tmp = OFCEnv(seed=seed); env_tmp.state = s2_obj
                    acts2 = env_tmp.legal_actions(max_samples=max_branch)
                    if len(acts2) == 0:
                        q_next.append(0.0)
                        continue
                    # argmax は online
                    q_online = []
                    for a2 in acts2:
                        a2_vec = encode_action(s2_obj, a2)
                        sa2_np = np.concatenate([np.asarray(s2_np, dtype=np.float32),
                                                 np.asarray(a2_vec, dtype=np.float32)], axis=0)
                        sa2 = torch.tensor(sa2_np, dtype=torch.float32, device=DEVICE).unsqueeze(0)
                        q_online.append(qnet(sa2).item())
                    best_idx = int(np.argmax(q_online))
                    best_a2 = acts2[best_idx]
                    a2_vec = encode_action(s2_obj, best_a2)
                    sa2_np = np.concatenate([np.asarray(s2_np, dtype=np.float32),
                                             np.asarray(a2_vec, dtype=np.float32)], axis=0)
                    sa2 = torch.tensor(sa2_np, dtype=torch.float32, device=DEVICE).unsqueeze(0)
                    q_next.append(tgt(sa2).item())
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
            print(f"[EP {ep:5d}] return={ret:+.3f}  buf={len(buf)}  eps={epsilon:.3f}")

    torch.save(qnet.state_dict(), save_path)
    print("saved:", save_path)

# ---------------------------
# 試行（学習済みモデルで1エピソード）
# ---------------------------
def run_test(model_path="./ckpt/ofc_qnet.pt", seed=None, max_branch=128):
    env = OFCEnv(seed=seed)
    model = QNet().to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    print("=== 試行開始 ===")
    s = env.reset()
    done = False
    total_r = 0.0
    while not done:
        s_vec = encode_state(s)
        acts = env.legal_actions(max_samples=max_branch)
        acts = heuristic_rank_actions(env.state, acts, keep=None)
        qs = []
        for a in acts:
            a_vec = encode_action(s, a)
            sa_np = np.concatenate([np.asarray(s_vec, dtype=np.float32),
                                    np.asarray(a_vec, dtype=np.float32)], axis=0)
            sa = torch.tensor(sa_np, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            with torch.no_grad():
                qv = model(sa).item()
            qs.append(qv)
        a = acts[int(np.argmax(qs))]
        s, r, done, _ = env.step(a)
        total_r += r
    print("報酬:", total_r, "ファウル:", s.foul)
    print("Top役:", catname_3(*eval_3(s.top.cards)), " Mid役:", catname_5(*eval_5(s.mid.cards)), " Bot役:", catname_5(*eval_5(s.bot.cards)))
    print("Top:", s.top.cards)
    print("Mid:", s.mid.cards)
    print("Bot:", s.bot.cards)

# ---------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true", help="学習を再開")
    parser.add_argument("--test", action="store_true", help="学習済みモデルで試行のみ")
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=None, help="学習・試行で使う乱数シード")
    args = parser.parse_args()

    save_path = "./ckpt/ofc_qnet.pt"

    if args.test:
        run_test(model_path=save_path, seed=args.seed)
    else:
        train(num_episodes=args.episodes, resume=args.resume, save_path=save_path, seed=args.seed)
