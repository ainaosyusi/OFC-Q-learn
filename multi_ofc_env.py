from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple, Dict
import random
from collections import Counter
import itertools
from functools import lru_cache

# =====================
# 共通定義
# =====================
RANKS = "23456789TJQKA"
SUITS = "shdc"  # s:♠, h:♥, d:♦, c:♣
Card = str
DECK52: List[Card] = [r + s for r in RANKS for s in SUITS]
JOKER: Card = "X"  # ULT OFC joker（任意のカードに変化）

RANK2IDX = {r: i for i, r in enumerate(RANKS)}

def card_rank(c: Card) -> int:
    if c == JOKER:
        return -1
    return RANK2IDX[c[0]]

def card_suit(c: Card) -> str:
    if c == JOKER:
        return "?"
    return c[1]

# =====================
# 役判定（3枚/5枚）
#   - core: Joker なし
#   - eval_*: Joker ありも対応（総当たりで最良）
# =====================

# 5枚役カテゴリ
# 0:High 1:Pair 2:TwoPair 3:Trips 4:Straight 5:Flush 6:FullHouse 7:Quads 8:StraightFlush
def _is_straight(ranks_sorted: List[int]):
    # ranks_sorted: 降順ユニーク想定
    if len(ranks_sorted) < 5:
        return None
    # A2345
    wheel = [12, 3, 2, 1, 0]
    if ranks_sorted[:5] == wheel:
        return 3  # 5-high straight
    for i in range(4):
        if ranks_sorted[i] - 1 != ranks_sorted[i + 1]:
            return None
    return ranks_sorted[0]

def eval_5_core(cards: List[Card]) -> Tuple[int, List[int]]:
    ranks = [card_rank(c) for c in cards]
    suits = [card_suit(c) for c in cards]
    cnt = Counter(ranks)
    counts = sorted(cnt.items(), key=lambda x: (x[1], x[0]), reverse=True)
    is_flush = len(set(suits)) == 1
    uniq = sorted(set(ranks), reverse=True)

    st_top = _is_straight(sorted(ranks, reverse=True))
    if st_top is None and len(uniq) == 5:
        st_top = _is_straight(uniq)

    if is_flush and st_top is not None:
        return 8, [st_top]
    if counts[0][1] == 4:
        quad = counts[0][0]
        kicker = max([r for r in ranks if r != quad])
        return 7, [quad, kicker]
    if counts[0][1] == 3 and counts[1][1] == 2:
        trips = counts[0][0]
        pair = counts[1][0]
        return 6, [trips, pair]
    if is_flush:
        return 5, sorted(ranks, reverse=True)
    if st_top is not None and len(set(ranks)) == 5:
        return 4, [st_top]
    if counts[0][1] == 3:
        trips = counts[0][0]
        kickers = sorted([r for r in ranks if r != trips], reverse=True)
        return 3, [trips] + kickers
    if counts[0][1] == 2 and counts[1][1] == 2:
        p1, p2 = counts[0][0], counts[1][0]
        hi, lo = max(p1, p2), min(p1, p2)
        kicker = max([r for r in ranks if r != p1 and r != p2])
        return 2, [hi, lo, kicker]
    if counts[0][1] == 2:
        pair = counts[0][0]
        kickers = sorted([r for r in ranks if r != pair], reverse=True)
        return 1, [pair] + kickers
    return 0, sorted(ranks, reverse=True)

def eval_3_core(cards: List[Card]) -> Tuple[int, List[int]]:
    ranks = [card_rank(c) for c in cards]
    cnt = Counter(ranks)
    counts = sorted(cnt.items(), key=lambda x: (x[1], x[0]), reverse=True)
    if counts[0][1] == 3:
        return 2, [counts[0][0]]
    if counts[0][1] == 2:
        pair = counts[0][0]
        kicker = max([r for r in ranks if r != pair])
        return 1, [pair, kicker]
    return 0, sorted(ranks, reverse=True)

@lru_cache(maxsize=500000)
def _eval_5_cached(cards_t: Tuple[Card, ...]) -> Tuple[int, Tuple[int, ...]]:
    cat, tieb = eval_5_core(list(cards_t))
    return cat, tuple(tieb)

@lru_cache(maxsize=500000)
def _eval_3_cached(cards_t: Tuple[Card, ...]) -> Tuple[int, Tuple[int, ...]]:
    cat, tieb = eval_3_core(list(cards_t))
    return cat, tuple(tieb)

def eval_5(cards: List[Card]) -> Tuple[int, List[int]]:
    j = sum(1 for c in cards if c == JOKER)
    if j == 0:
        c, t = _eval_5_cached(tuple(cards))
        return c, list(t)

    fixed = [c for c in cards if c != JOKER]
    pool = [c for c in DECK52 if c not in fixed]

    best_cat = -1
    best_tb: List[int] = []
    for repl in itertools.combinations(pool, j):
        cand = fixed + list(repl)
        c, t = _eval_5_cached(tuple(sorted(cand)))
        t = list(t)
        if (c > best_cat) or (c == best_cat and t > best_tb):
            best_cat, best_tb = c, t
    return best_cat, best_tb

def eval_3(cards: List[Card]) -> Tuple[int, List[int]]:
    j = sum(1 for c in cards if c == JOKER)
    if j == 0:
        c, t = _eval_3_cached(tuple(cards))
        return c, list(t)

    fixed = [c for c in cards if c != JOKER]
    pool = [c for c in DECK52 if c not in fixed]

    best_cat = -1
    best_tb: List[int] = []
    for repl in itertools.combinations(pool, j):
        cand = fixed + list(repl)
        c, t = _eval_3_cached(tuple(sorted(cand)))
        t = list(t)
        if (c > best_cat) or (c == best_cat and t > best_tb):
            best_cat, best_tb = c, t
    return best_cat, best_tb

def catname_5(cat: int) -> str:
    return ["High","Pair","TwoPair","Trips","Straight","Flush","FullHouse","Quads","StraightFlush"][cat]

def catname_3(cat: int) -> str:
    return ["High","Pair","Trips"][cat]

# =====================
# 盤面
# =====================
@dataclass
class PlayerBoard:
    top: List[Card] = field(default_factory=list)    # 3
    mid: List[Card] = field(default_factory=list)    # 5
    bot: List[Card] = field(default_factory=list)    # 5
    foul: bool = False

    def is_full(self) -> bool:
        return len(self.top)==3 and len(self.mid)==5 and len(self.bot)==5

    def can_place(self, row: str) -> bool:
        if row=="T": return len(self.top) < 3
        if row=="M": return len(self.mid) < 5
        if row=="B": return len(self.bot) < 5
        return False

    def place(self, row: str, card: Card):
        if row=="T": self.top.append(card)
        elif row=="M": self.mid.append(card)
        elif row=="B": self.bot.append(card)
        else: raise ValueError("row must be T/M/B")

    def evaluate_foul(self):
        # 不完全な状態なら役判定できないので即フォール扱い
        if len(self.top) != 3 or len(self.mid) != 5 or len(self.bot) != 5:
            self.foul = True
            return

        t_cat, t_tb = eval_3(self.top)
        m_cat, m_tb = eval_5(self.mid)
        b_cat, b_tb = eval_5(self.bot)

        def key(cat, tb): return (cat, tb)
        if key(t_cat, t_tb) > key(m_cat, m_tb): self.foul = True
        if key(m_cat, m_tb) > key(b_cat, b_tb): self.foul = True

# =====================
# 状態
# =====================
@dataclass
class MultiOFCState:
    turn: int
    hand: List[Card]
    hero: PlayerBoard
    opps: List[PlayerBoard]

# =====================
# 環境
# =====================
class OFCMultiEnv:
    def __init__(self, n_players: int = 3, hero_idx: int = 0, seed: int = 0, n_jokers: int = 0):
        assert n_players >= 2
        self.n_players = n_players
        self.hero_idx = hero_idx
        self.rng = random.Random(seed)
        self.n_jokers = n_jokers

        self.players: List[PlayerBoard] = []
        self.turn = 0
        self.deck: List[Card] = []
        self.hand: List[Card] = []

    def _deal_init(self):
        need = min(5, 13)
        self.hand = [self.deck.pop() for _ in range(need)]

    def _deal_next(self):
        # 以降は 3 枚ずつ（残りスロットが足りない場合は不足分のみ）
        hero = self.players[self.hero_idx]
        remain = 13 - (len(hero.top) + len(hero.mid) + len(hero.bot))
        need = max(0, min(3, remain))
        self.hand = [self.deck.pop() for _ in range(need)] if need > 0 else []

    def reset(self) -> MultiOFCState:
        self.players = [PlayerBoard() for _ in range(self.n_players)]
        self.turn = 0
        self.deck = DECK52.copy() + [JOKER] * self.n_jokers
        self.rng.shuffle(self.deck)
        self._deal_init()
        return self._get_state()

    def _get_state(self) -> MultiOFCState:
        hero = self.players[self.hero_idx]
        opps = [p for i,p in enumerate(self.players) if i!=self.hero_idx]
        return MultiOFCState(turn=self.turn, hand=self.hand.copy(), hero=hero, opps=opps)

    def available_actions(self):
        acts = []
        hero = self.players[self.hero_idx]
        for i,_ in enumerate(self.hand):
            for row in ["T","M","B"]:
                if hero.can_place(row):
                    acts.append((i,row))
        return acts

    def step(self, action):
        idx,row = action
        card = self.hand.pop(idx)
        hero = self.players[self.hero_idx]
        hero.place(row, card)

        # 相手は簡易ランダム配置（本格化は後で）
        for i,p in enumerate(self.players):
            if i==self.hero_idx:
                continue
            rows = [r for r in ["T","M","B"] if p.can_place(r)]
            if not rows:
                continue
            used = set(p.top + p.mid + p.bot)
            pool = [c for c in DECK52 if c not in used]
            fake_card = self.rng.choice(pool)
            p.place(self.rng.choice(rows), fake_card)

        done = False
        reward = 0.0
        info: Dict = {}

        hero_cards = len(hero.top) + len(hero.mid) + len(hero.bot)
        if hero_cards >= 13:
            done = True
            reward = self._compute_hero_reward()
            info["hero_foul"] = self.players[self.hero_idx].foul
            return self._get_state(), reward, done, info

        if not self.hand:
            self.turn += 1
            self._deal_next()

        return self._get_state(), reward, done, info

    def _compute_hero_reward(self) -> float:
        hero = self.players[self.hero_idx]
        hero.evaluate_foul()
        if hero.foul:
            # 3pなら -40
            return -20.0 * (self.n_players - 1)

        # 簡易スコア（後で本格採点に差し替え）
        def strength(p: PlayerBoard) -> float:
            t = eval_3(p.top)
            m = eval_5(p.mid)
            b = eval_5(p.bot)
            return (t[0]*100 + sum(t[1])) + (m[0]*100 + sum(m[1])) + (b[0]*100 + sum(b[1]))

        hero_s = strength(hero)
        score = 0.0
        for i,p in enumerate(self.players):
            if i==self.hero_idx:
                continue
            p.evaluate_foul()
            if p.foul:
                score += 1.0
            else:
                score += 1.0 if hero_s >= strength(p) else -1.0
        return score
