# ofc_env.py (patched)
from dataclasses import dataclass, field
import random
import itertools
from collections import Counter
from typing import List, Tuple, Optional, Dict, Any, Union

# ===== 共通定義 =====
RANKS = "23456789TJQKA"
RANK2IDX = {r:i for i,r in enumerate(RANKS)}         # 2(0) ... A(12)
SUITS = "shdc"  # s=♠, h=♥, d=♦, c=♣
Card = str      # e.g., "As", "Td"
DECK52: List[Card] = [r+s for r in RANKS for s in SUITS]

def card_rank(c: Card) -> int:
    return RANK2IDX[c[0]]
def card_suit(c: Card) -> str:
    return c[1]

# ===== ロイヤリティ（学習用の代表テーブル：後で差し替え可）=====
ROY_TOP: Dict[str,int] = {
    "PAIR_66":1, "PAIR_77":2, "PAIR_88":3, "PAIR_99":4,
    "PAIR_TT":5, "PAIR_JJ":6, "PAIR_QQ":7, "PAIR_KK":8, "PAIR_AA":9,
    "SET_222":10, "SET_333":11, "SET_444":12, "SET_555":13, "SET_666":14,
    "SET_777":15, "SET_888":16, "SET_999":17, "SET_TTT":18, "SET_JJJ":19,
    "SET_QQQ":20, "SET_KKK":21, "SET_AAA":22
}
ROY_MID = { "STRAIGHT":4, "FLUSH":8, "FULLHOUSE":12, "QUADS":20, "SFLUSH":30, "RFLUSH":50 }
ROY_BOT = { "STRAIGHT":2, "FLUSH":4, "FULLHOUSE":6,  "QUADS":10, "SFLUSH":15, "RFLUSH":25 }

# ===== ハンド強度の序数 =====
FIVE_ORDER = {
    "SFLUSH": 8, "QUADS":7, "FULLHOUSE":6, "FLUSH":5, "STRAIGHT":4,
    "TRIPS":3, "TWO_PAIR":2, "PAIR":1, "HIGH":0
}
THREE_ORDER = { "SET":2, "PAIR":1, "HIGH":0 }

# ====== 役判定ユーティリティ ======
def _is_straight(ranks_sorted_desc: List[int]) -> Tuple[bool, int]:
    if len(ranks_sorted_desc) < 5:
        return (False, -1)
    rs = ranks_sorted_desc
    for i in range(len(rs)-4):
        window = rs[i:i+5]
        if all(window[j] - window[j+1] == 1 for j in range(4)):
            return True, window[0]
    # A2345
    need = {12,3,2,1,0}
    if need.issubset(set(ranks_sorted_desc)):
        return True, 3  # 5-high
    return (False, -1)

def eval_5(cards: List[Card]) -> Tuple[int, Tuple]:
    assert len(cards) == 5
    ranks = [card_rank(c) for c in cards]
    suits = [card_suit(c) for c in cards]
    ranks.sort(reverse=True)
    cnt = Counter(ranks)
    pairs = sorted([r for r,c in cnt.items() if c==2], reverse=True)
    trips = sorted([r for r,c in cnt.items() if c==3], reverse=True)
    quads = sorted([r for r,c in cnt.items() if c==4], reverse=True)
    singles = sorted([r for r,c in cnt.items() if c==1], reverse=True)

    is_flush = len(set(suits)) == 1
    uniq_desc = sorted(set(ranks), reverse=True)
    is_straight, hi_st = _is_straight(uniq_desc)

    if is_flush and is_straight:
        if set([12,11,10,9,8]).issubset(set(ranks)):
            return FIVE_ORDER["SFLUSH"], (12,11,10,9,8)
        return FIVE_ORDER["SFLUSH"], (hi_st,)

    if quads:
        q = quads[0]
        kicker = singles[0]
        return FIVE_ORDER["QUADS"], (q, kicker)

    if trips and pairs:
        return FIVE_ORDER["FULLHOUSE"], (trips[0], pairs[0])

    if is_flush:
        return FIVE_ORDER["FLUSH"], tuple(ranks)

    if is_straight:
        return FIVE_ORDER["STRAIGHT"], (hi_st,)

    if trips:
        t = trips[0]
        kickers = sorted([r for r in ranks if r != t], reverse=True)[:2]
        return FIVE_ORDER["TRIPS"], (t, *kickers)

    if len(pairs) >= 2:
        p1, p2 = pairs[0], pairs[1]
        kicker = max([r for r in ranks if r not in (p1, p2)])
        p_hi, p_lo = max(p1,p2), min(p1,p2)
        return FIVE_ORDER["TWO_PAIR"], (p_hi, p_lo, kicker)

    if len(pairs) == 1:
        p = pairs[0]
        kickers = sorted([r for r in ranks if r != p], reverse=True)[:3]
        return FIVE_ORDER["PAIR"], (p, *kickers)

    return FIVE_ORDER["HIGH"], tuple(ranks)

def eval_3(cards: List[Card]) -> Tuple[int, Tuple]:
    assert len(cards) == 3
    ranks = sorted([card_rank(c) for c in cards], reverse=True)
    cnt = Counter(ranks)
    if 3 in cnt.values():  # SET
        return THREE_ORDER["SET"], (ranks[0],)
    if 2 in cnt.values():  # PAIR
        pair_rank = max([r for r,c in cnt.items() if c==2])
        kicker = max([r for r,c in cnt.items() if c==1])
        return THREE_ORDER["PAIR"], (pair_rank, kicker)
    return THREE_ORDER["HIGH"], tuple(ranks)

def catname_5(weight: int, tiebreak: Tuple) -> str:
    inv = {v:k for k,v in FIVE_ORDER.items()}
    return inv[weight]
def catname_3(weight: int, tiebreak: Tuple) -> str:
    inv = {v:k for k,v in THREE_ORDER.items()}
    return inv[weight]

# ===== 盤面・環境 =====
@dataclass
class Line:
    cards: List[Card] = field(default_factory=list)
    cap: int = 5
    def full(self) -> bool:
        return len(self.cards) >= self.cap
    def space(self) -> int:
        return self.cap - len(self.cards)

@dataclass
class OFCState:
    top: Line = field(default_factory=lambda: Line(cap=3))
    mid: Line = field(default_factory=lambda: Line(cap=5))
    bot: Line = field(default_factory=lambda: Line(cap=5))
    turn: int = 0                       # 0: 初手5枚, 1..4: pineappleターン
    incoming: List[Card] = field(default_factory=list)
    dead: List[Card] = field(default_factory=list)  # 捨て札
    foul: bool = False

    def as_obs(self) -> Dict[str, Any]:
        return {
            "top": self.top.cards[:],
            "mid": self.mid.cards[:],
            "bot": self.bot.cards[:],
            "turn": self.turn,
            "incoming": self.incoming[:],
            "dead": self.dead[:],
            "foul": self.foul
        }

class OFCEnv:
    """
    単独プレイ学習用 Pineapple OFC 環境
    - reset(): 初手5枚配布
    - legal_actions(): 現在ターンの合法手（容量違反なし）を列挙
    - step(action): 
        turn==0  -> action は 5枚の行割当 (長さ5の 'T'/'M'/'B')
        turn>=1  -> action は (discard_idx, place_to_1, place_to_2)
    """
    def __init__(self, use_jokers=False, use_fantasy=False, seed=None):
        self.use_jokers = use_jokers
        self.use_fantasy = use_fantasy
        self.rng = random.Random(seed)
        self.deck: List[Card] = []
        self.state: Optional[OFCState] = None

    # ---- API ----
    def reset(self) -> OFCState:
        self.deck = DECK52.copy()
        self.rng.shuffle(self.deck)
        self.state = OFCState()
        self.state.incoming = [self.deck.pop() for _ in range(5)]
        self.state.turn = 0
        return self.state

    def legal_actions(self, max_samples: Optional[int] = None) -> List:
        """
        容量違反を除いた合法手を列挙。
        max_samples を指定するとサンプリング（行動空間の爆発抑制）。
        """
        s = self.state
        assert s is not None
        acts = []
        if s.turn == 0:
            for choice in itertools.product("TMB", repeat=5):  # 3^5=243
                t = choice.count("T")
                m = choice.count("M")
                b = choice.count("B")
                if t <= s.top.space() and m <= s.mid.space() and b <= s.bot.space():
                    acts.append(list(choice))
        else:
            for d in range(3):
                for p1 in "TMB":
                    for p2 in "TMB":
                        need_T = (p1=="T") + (p2=="T")
                        need_M = (p1=="M") + (p2=="M")
                        need_B = (p1=="B") + (p2=="B")
                        if need_T <= s.top.space() and need_M <= s.mid.space() and need_B <= s.bot.space():
                            acts.append((d, p1, p2))
        if max_samples is not None and len(acts) > max_samples:
            self.rng.shuffle(acts)
            acts = acts[:max_samples]
        return acts

    def step(self, action: Union[List[str], Tuple[int,str,str]]) -> Tuple[OFCState, float, bool, dict]:
        s = self.state
        assert s is not None

        if s.turn == 0:
            self._apply_init5(s, action)
            s.turn = 1
            s.incoming = [self.deck.pop() for _ in range(3)]
        else:
            self._apply_pineapple(s, action)
            if s.turn < 4:
                s.turn += 1
                s.incoming = [self.deck.pop() for _ in range(3)]
            else:
                pass  # 終局入力は次の判定で

        s.foul = self._is_foul(s)

        done = False
        reward = 0.0
        if s.foul:
            done = True
            reward = -6.0
        elif (s.top.full() and s.mid.full() and s.bot.full()) and s.turn >= 4:
            done = True
            reward = self._terminal_reward(s)
        else:
            reward += self._shaping_penalty(s)

        info = {}
        return s, reward, done, info

    # ---- アクション適用ロジック ----
    def _apply_init5(self, s: OFCState, action: List[str]) -> None:
        assert isinstance(action, (list, tuple)) and len(action) == 5
        for card, dst in zip(s.incoming, action):
            self._place_one(s, card, dst)
        s.incoming = []

    def _apply_pineapple(self, s: OFCState, action: Tuple[int,str,str]) -> None:
        assert isinstance(action, (list, tuple)) and len(action) == 3
        discard_idx, p1, p2 = action
        assert 0 <= discard_idx < 3
        cards = s.incoming
        discard_card = cards[discard_idx]
        keep = [cards[i] for i in range(3) if i != discard_idx]
        s.dead.append(discard_card)
        self._place_one(s, keep[0], p1)
        self._place_one(s, keep[1], p2)
        s.incoming = []

    def _place_one(self, s: OFCState, card: Card, dst: str) -> None:
        line = {"T": s.top, "M": s.mid, "B": s.bot}[dst]
        if line.full():
            s.foul = True
            return
        line.cards.append(card)

    # ---- 評価系 ----
    def _is_foul(self, s: OFCState) -> bool:
        if s.top.full() and s.mid.full() and s.bot.full():
            wt_top, tb_top = eval_3(s.top.cards)
            wt_mid, tb_mid = eval_5(s.mid.cards)
            wt_bot, tb_bot = eval_5(s.bot.cards)
            if not self._lt_3v5((wt_top,tb_top), (wt_mid,tb_mid)):
                return True
            if not self._lt_5v5((wt_mid,tb_mid), (wt_bot,tb_bot)):
                return True
        return s.foul

    def _lt_3v5(self, a: Tuple[int,Tuple], b: Tuple[int,Tuple]) -> bool:
        """
        Top(3枚) < Middle(5枚) を厳密に判定。
        """
        wt_top, tb_top = a  # 3枚役
        wt_mid, tb_mid = b  # 5枚役

        # Top HIGH → Middle は PAIR 以上必須
        if wt_top == THREE_ORDER["HIGH"]:
            return wt_mid >= FIVE_ORDER["PAIR"]

        # Top PAIR
        if wt_top == THREE_ORDER["PAIR"]:
            if wt_mid < FIVE_ORDER["PAIR"]:
                return False
            if wt_mid >= FIVE_ORDER["TWO_PAIR"]:
                return True
            # PAIR vs PAIR：ペア階級で Middle が上なら OK
            return tb_mid[0] > tb_top[0]

        # Top SET (Trips)
        if wt_top == THREE_ORDER["SET"]:
            if wt_mid < FIVE_ORDER["TRIPS"]:
                return False
            if wt_mid == FIVE_ORDER["TRIPS"]:
                return tb_mid[0] > tb_top[0]
            return True

        return False

    def _lt_5v5(self, a: Tuple[int,Tuple], b: Tuple[int,Tuple]) -> bool:
        wa, ta = a
        wb, tb = b
        if wa != wb: return wa < wb
        return ta < tb

    def _terminal_reward(self, s: OFCState) -> float:
        rt = self._roy_top(s.top.cards)
        rm = self._roy_mid(s.mid.cards)
        rb = self._roy_bot(s.bot.cards)
        return float(rt + rm + rb)

    def _roy_top(self, cards: List[Card]) -> int:
        wt, tb = eval_3(cards)
        if wt == THREE_ORDER["SET"]:
            r = tb[0]
            key = f"SET_{RANKS[r]*3}"
            return ROY_TOP.get(key, 0)
        if wt == THREE_ORDER["PAIR"]:
            pair_rank = tb[0]
            tag = RANKS[pair_rank]*2
            table = {
                "66":"PAIR_66","77":"PAIR_77","88":"PAIR_88","99":"PAIR_99",
                "TT":"PAIR_TT","JJ":"PAIR_JJ","QQ":"PAIR_QQ",
                "KK":"PAIR_KK","AA":"PAIR_AA"
            }
            return ROY_TOP.get(table.get(tag,""), 0)
        return 0

    def _roy_mid(self, cards: List[Card]) -> int:
        w, tb = eval_5(cards)
        name = catname_5(w, tb)
        return ROY_MID.get(name, 0)

    def _roy_bot(self, cards: List[Card]) -> int:
        w, tb = eval_5(cards)
        name = catname_5(w, tb)
        return ROY_BOT.get(name, 0)

    def _shaping_penalty(self, s: OFCState) -> float:
        """
        進展を促す小ボーナス + 危険配置の軽い罰。
        強すぎると本来の報酬を壊すので係数は小さめ。
        """
        bonus = 0.0

        # 埋まり具合ボーナス（Mid/Bot を優先）
        bonus += 0.01  * len(s.mid.cards)
        bonus += 0.012 * len(s.bot.cards)
        bonus += 0.004 * len(s.top.cards)   # Top は控えめ

        # 早期に Top を強くし過ぎる罰（Mid/Bot がまだ弱いとき）
        if len(s.top.cards) >= 2:
            wt_top, _ = eval_3(s.top.cards) if len(s.top.cards)==3 else (THREE_ORDER["HIGH"], ())
            if wt_top >= THREE_ORDER["PAIR"] and (len(s.mid.cards) < 3 or len(s.bot.cards) < 3):
                bonus -= 0.03

        # フラッシュ目の小ボーナス
        def flush_seed(line: Line):
            suits = {}
            for c in line.cards:
                suits[c[1]] = suits.get(c[1], 0) + 1
            return max(suits.values()) if suits else 0
        if flush_seed(s.mid) >= 2 and s.mid.space() >= 3:
            bonus += 0.01
        if flush_seed(s.bot) >= 2 and s.bot.space() >= 3:
            bonus += 0.012

        # Top だけ先行で埋めすぎる罰
        over_top = max(0, len(s.top.cards) - min(len(s.mid.cards), len(s.bot.cards)))
        bonus -= 0.02 * over_top

        return bonus
