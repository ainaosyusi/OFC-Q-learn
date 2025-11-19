# multi_ofc_env.py
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
import random
from collections import Counter
import itertools

# =====================
# 共通定義
# =====================
RANKS = "23456789TJQKA"
SUITS = "shdc"  # s:♠, h:♥, d:♦, c:♣
Card = str
DECK52: List[Card] = [r + s for r in RANKS for s in SUITS]

RANK2IDX = {r: i for i, r in enumerate(RANKS)}

def card_rank(c: Card) -> int:
    return RANK2IDX[c[0]]

def card_suit(c: Card) -> str:
    return c[1]

# ---- 役の強さ（5枚・3枚） ----
FIVE_ORDER = {
    "SFLUSH": 8, "QUADS": 7, "FULLHOUSE": 6, "FLUSH": 5,
    "STRAIGHT": 4, "TRIPS": 3, "TWO_PAIR": 2, "PAIR": 1, "HIGH": 0
}
THREE_ORDER = {
    "SET": 2, "PAIR": 1, "HIGH": 0
}

# ---- ロイヤリティ（ざっくり・JOPT風に近いテーブル） ----
ROY_TOP: Dict[str, int] = {
    "PAIR_66":1, "PAIR_77":2, "PAIR_88":3, "PAIR_99":4,
    "PAIR_TT":5, "PAIR_JJ":6, "PAIR_QQ":7, "PAIR_KK":8, "PAIR_AA":9,
    "SET_222":10, "SET_333":11, "SET_444":12, "SET_555":13, "SET_666":14,
    "SET_777":15, "SET_888":16, "SET_999":17, "SET_TTT":18, "SET_JJJ":19,
    "SET_QQQ":20, "SET_KKK":21, "SET_AAA":22
}
ROY_MID = {
    "STRAIGHT": 4, "FLUSH": 8, "FULLHOUSE": 12,
    "QUADS": 20, "SFLUSH": 30, "RFLUSH": 50
}
ROY_BOT = {
    "STRAIGHT": 2, "FLUSH": 4, "FULLHOUSE": 6,
    "QUADS": 10, "SFLUSH": 15, "RFLUSH": 25
}

# =====================
# 役評価（5枚・3枚）
# =====================
def _is_straight(ranks_sorted_desc: List[int]) -> Tuple[bool, int]:
    if len(ranks_sorted_desc) < 5:
        return False, -1
    rs = ranks_sorted_desc
    for i in range(len(rs) - 4):
        window = rs[i:i+5]
        if all(window[j] - window[j+1] == 1 for j in range(4)):
            return True, window[0]
    # Wheel (A2345)
    need = {12, 3, 2, 1, 0}
    if need.issubset(set(ranks_sorted_desc)):
        return True, 3  # 5-high
    return False, -1

def eval_5(cards: List[Card]) -> Tuple[int, Tuple]:
    assert len(cards) == 5
    ranks = [card_rank(c) for c in cards]
    suits = [card_suit(c) for c in cards]
    ranks.sort(reverse=True)
    cnt = Counter(ranks)
    pairs = sorted([r for r, c in cnt.items() if c == 2], reverse=True)
    trips = sorted([r for r, c in cnt.items() if c == 3], reverse=True)
    quads = sorted([r for r, c in cnt.items() if c == 4], reverse=True)
    singles = sorted([r for r, c in cnt.items() if c == 1], reverse=True)

    is_flush = len(set(suits)) == 1
    uniq_desc = sorted(set(ranks), reverse=True)
    is_straight, hi_st = _is_straight(uniq_desc)

    # Straight flush / Royal flush
    if is_flush and is_straight:
        if set([12, 11, 10, 9, 8]).issubset(set(ranks)):
            return FIVE_ORDER["SFLUSH"], (12, 11, 10, 9, 8)
        return FIVE_ORDER["SFLUSH"], (hi_st,)

    # Quads
    if quads:
        q = quads[0]
        kicker = singles[0]
        return FIVE_ORDER["QUADS"], (q, kicker)

    # Full house
    if trips and pairs:
        return FIVE_ORDER["FULLHOUSE"], (trips[0], pairs[0])

    # Flush
    if is_flush:
        return FIVE_ORDER["FLUSH"], tuple(ranks)

    # Straight
    if is_straight:
        return FIVE_ORDER["STRAIGHT"], (hi_st,)

    # Trips
    if trips:
        t = trips[0]
        kickers = sorted([r for r in ranks if r != t], reverse=True)[:2]
        return FIVE_ORDER["TRIPS"], (t, *kickers)

    # Two pair
    if len(pairs) >= 2:
        p1, p2 = pairs[0], pairs[1]
        kicker = max([r for r in ranks if r not in (p1, p2)])
        p_hi, p_lo = max(p1, p2), min(p1, p2)
        return FIVE_ORDER["TWO_PAIR"], (p_hi, p_lo, kicker)

    # One pair
    if len(pairs) == 1:
        p = pairs[0]
        kickers = sorted([r for r in ranks if r != p], reverse=True)[:3]
        return FIVE_ORDER["PAIR"], (p, *kickers)

    # High card
    return FIVE_ORDER["HIGH"], tuple(ranks)

def eval_3(cards: List[Card]) -> Tuple[int, Tuple]:
    assert len(cards) == 3
    ranks = sorted([card_rank(c) for c in cards], reverse=True)
    cnt = Counter(ranks)
    if 3 in cnt.values():
        return THREE_ORDER["SET"], (ranks[0],)
    if 2 in cnt.values():
        pair_rank = max([r for r, c in cnt.items() if c == 2])
        kicker = max([r for r, c in cnt.items() if c == 1])
        return THREE_ORDER["PAIR"], (pair_rank, kicker)
    return THREE_ORDER["HIGH"], tuple(ranks)

def catname_5(weight: int, tiebreak: Tuple) -> str:
    inv = {v: k for k, v in FIVE_ORDER.items()}
    return inv[weight]

def catname_3(weight: int, tiebreak: Tuple) -> str:
    inv = {v: k for k, v in THREE_ORDER.items()}
    return inv[weight]

# =====================
# プレイヤーボード・状態
# =====================
@dataclass
class Line:
    cards: List[Card] = field(default_factory=list)
    cap: int = 5
    def full(self) -> bool:
        return len(self.cards) >= self.cap
    def space(self) -> int:
        return self.cap - len(self.cards)

@dataclass
class PlayerBoard:
    top: Line = field(default_factory=lambda: Line(cap=3))
    mid: Line = field(default_factory=lambda: Line(cap=5))
    bot: Line = field(default_factory=lambda: Line(cap=5))
    dead: List[Card] = field(default_factory=list)
    incoming: List[Card] = field(default_factory=list)
    foul: bool = False

@dataclass
class MultiOFCState:
    players: List[PlayerBoard]
    turn: int           # 0: 初手5枚, 1..4: pineappleターン
    hero_idx: int

# =====================
# ロイヤリティ計算（1プレイヤー）
# =====================
def roy_top(cards: List[Card]) -> int:
    if len(cards) != 3:
        return 0
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
        return ROY_TOP.get(table.get(tag, ""), 0)
    return 0

def roy_mid(cards: List[Card]) -> int:
    if len(cards) != 5:
        return 0
    w, tb = eval_5(cards)
    return ROY_MID.get(catname_5(w, tb), 0)

def roy_bot(cards: List[Card]) -> int:
    if len(cards) != 5:
        return 0
    w, tb = eval_5(cards)
    return ROY_BOT.get(catname_5(w, tb), 0)

# =====================
# マルチプレイ環境
# =====================
class OFCMultiEnv:
    """
    2〜3人用 Pineapple OFC 環境（単一エージェント学習用）
    - hero_idx プレイヤーが Q学習対象
    - その他プレイヤーは簡易ポリシー（ランダム合法手）で自動行動
    - 各ストリートごとに
        * 初手: 各プレイヤーに5枚 → 全員ボードに置く
        * 以降4ターン: 各プレイヤーに3枚 → 1枚捨てて2枚配置
    - 終局時に hero vs 各相手のヘッズアップスコアリングで報酬計算
    """

    def __init__(self, n_players: int = 2, hero_idx: int = 0, seed: Optional[int] = None):
        assert 2 <= n_players <= 3, "2〜3人プレイに対応"
        assert 0 <= hero_idx < n_players
        self.n_players = n_players
        self.hero_idx = hero_idx
        self.rng = random.Random(seed)
        self.deck: List[Card] = []
        self.players: List[PlayerBoard] = []
        self.turn: int = 0

    # ---------- 内部ユーティリティ ----------
    def _new_players(self) -> List[PlayerBoard]:
        return [PlayerBoard() for _ in range(self.n_players)]

    def _draw(self, n: int) -> List[Card]:
        out = []
        for _ in range(n):
            if not self.deck:
                break
            out.append(self.deck.pop())
        return out

    def _make_state(self) -> MultiOFCState:
        # 参照を渡すが、encode時にコピーするのでOKという想定
        return MultiOFCState(players=self.players, turn=self.turn, hero_idx=self.hero_idx)

    # ---------- 公開API ----------
    def reset(self) -> MultiOFCState:
        self.deck = DECK52.copy()
        self.rng.shuffle(self.deck)
        self.players = self._new_players()
        self.turn = 0

        # 全員に初手5枚
        for p in self.players:
            p.incoming = self._draw(5)
        return self._make_state()

    # hero用の合法手生成
    def legal_actions(self) -> List:
        p = self.players[self.hero_idx]
        acts = []
        if self.turn == 0:
            # 初手5枚: 各カードの行を T/M/B で指定
            assert len(p.incoming) == 5
            for choice in itertools.product("TMB", repeat=5):  # 3^5 = 243
                t = choice.count("T")
                m = choice.count("M")
                b = choice.count("B")
                if t <= p.top.space() and m <= p.mid.space() and b <= p.bot.space():
                    acts.append(list(choice))
        else:
            # pineapple: (discard_idx, place1, place2)
            assert len(p.incoming) == 3
            for d in range(3):
                for p1 in "TMB":
                    for p2 in "TMB":
                        need_T = (p1 == "T") + (p2 == "T")
                        need_M = (p1 == "M") + (p2 == "M")
                        need_B = (p1 == "B") + (p2 == "B")
                        if (need_T <= p.top.space()
                            and need_M <= p.mid.space()
                            and need_B <= p.bot.space()):
                            acts.append((d, p1, p2))
        return acts

    def step(self, hero_action) -> Tuple[MultiOFCState, float, bool, dict]:
        """
        hero_action を適用 → 他プレイヤーも自動で配置 → 必要なら次ストリートへ
        戻り:
            state, reward, done, info
        reward は hero 視点のスカラー。
        """
        # 1. hero の行動を適用
        self._apply_action(self.hero_idx, hero_action)

        # 2. 他プレイヤーの行動（簡易ランダムポリシー）
        for idx in range(self.n_players):
            if idx == self.hero_idx:
                continue
            self._auto_play(idx)

        # 3. 次ストリート or 終局
        reward = 0.0
        done = False

        if self.turn == 0:
            # 初手5枚が終わった → 次は turn=1 の pineapple
            self.turn = 1
            for p in self.players:
                p.incoming = self._draw(3)
        elif 1 <= self.turn <= 3:
            # pineapple 中 → 次のストリートへ
            self.turn += 1
            for p in self.players:
                p.incoming = self._draw(3)
        else:
            # turn==4 まで来ている → 本来はこのターンの配置後が終局
            # （今の実装では turn=4 で step が呼ばれたタイミングを終局とみなす）
            pass

        # 終局判定：全員のボードが埋まっていれば終わり
        all_full = all(
            len(p.top.cards) == 3 and len(p.mid.cards) == 5 and len(p.bot.cards) == 5
            for p in self.players
        )
        if all_full:
            # foul フラグを更新 & スコアリング
            for p in self.players:
                p.foul = self._is_foul_board(p)

            reward = self._compute_hero_reward()
            done = True
            # 終局なので incoming はすべてクリア
            for p in self.players:
                p.incoming = []

        return self._make_state(), reward, done, {}

    # ---------- 1プレイヤーのアクション適用 ----------
    def _apply_action(self, idx: int, action) -> None:
        p = self.players[idx]
        if self.turn == 0:
            # 初手5枚: action は ['T','M',...]
            assert len(p.incoming) == 5
            assert isinstance(action, (list, tuple)) and len(action) == 5
            for card, dst in zip(p.incoming, action):
                self._place_one(p, card, dst)
            p.incoming = []
        else:
            # pineapple: action は (discard_idx, p1, p2)
            assert len(p.incoming) == 3
            d, p1, p2 = action
            cards = p.incoming
            discard_card = cards[d]
            keep = [cards[i] for i in range(3) if i != d]
            p.dead.append(discard_card)
            self._place_one(p, keep[0], p1)
            self._place_one(p, keep[1], p2)
            p.incoming = []

    def _place_one(self, p: PlayerBoard, card: Card, dst: str) -> None:
        line = {"T": p.top, "M": p.mid, "B": p.bot}[dst]
        if line.full():
            # オーバーフローしたら即 foul 扱い（安全側）
            p.foul = True
            return
        line.cards.append(card)

    def _auto_play(self, idx: int) -> None:
        """
        シンプルなランダムポリシー：合法手から一様ランダムに選ぶ。
        """
        p = self.players[idx]
        # incoming が無ければ何もしない
        if self.turn == 0 and len(p.incoming) != 5:
            return
        if self.turn > 0 and len(p.incoming) != 3:
            return

        # hero と同じロジックで合法手を作り、ランダムに1つ選ぶ
        acts = []
        if self.turn == 0:
            for choice in itertools.product("TMB", repeat=5):
                t = choice.count("T")
                m = choice.count("M")
                b = choice.count("B")
                if t <= p.top.space() and m <= p.mid.space() and b <= p.bot.space():
                    acts.append(list(choice))
        else:
            for d in range(3):
                for p1 in "TMB":
                    for p2 in "TMB":
                        need_T = (p1 == "T") + (p2 == "T")
                        need_M = (p1 == "M") + (p2 == "M")
                        need_B = (p1 == "B") + (p2 == "B")
                        if (need_T <= p.top.space()
                            and need_M <= p.mid.space()
                            and need_B <= p.bot.space()):
                            acts.append((d, p1, p2))

        if not acts:
            return
        a = random.choice(acts)
        self._apply_action(idx, a)

    # ---------- foul 判定（Top < Mid < Bot） ----------
    def _is_foul_board(self, p: PlayerBoard) -> bool:
        if len(p.top.cards) != 3 or len(p.mid.cards) != 5 or len(p.bot.cards) != 5:
            return True  # 埋まりきってないのは安全側で foul とみなす

        wt_top, tb_top = eval_3(p.top.cards)
        wt_mid, tb_mid = eval_5(p.mid.cards)
        wt_bot, tb_bot = eval_5(p.bot.cards)

        # Top < Mid ?
        if not self._lt_3v5((wt_top, tb_top), (wt_mid, tb_mid)):
            return True
        # Mid < Bot ?
        if not self._lt_5v5((wt_mid, tb_mid), (wt_bot, tb_bot)):
            return True
        return False

    def _lt_3v5(self, top: Tuple[int, Tuple], mid: Tuple[int, Tuple]) -> bool:
        wt_top, tb_top = top
        wt_mid, tb_mid = mid

        if wt_top == THREE_ORDER["HIGH"]:
            return wt_mid >= FIVE_ORDER["PAIR"]

        if wt_top == THREE_ORDER["PAIR"]:
            if wt_mid < FIVE_ORDER["PAIR"]:
                return False
            if wt_mid >= FIVE_ORDER["TWO_PAIR"]:
                return True
            # PAIR vs PAIR
            return tb_mid[0] > tb_top[0]

        if wt_top == THREE_ORDER["SET"]:
            if wt_mid < FIVE_ORDER["TRIPS"]:
                return False
            if wt_mid == FIVE_ORDER["TRIPS"]:
                return tb_mid[0] > tb_top[0]
            return True
        return False

    def _lt_5v5(self, a: Tuple[int, Tuple], b: Tuple[int, Tuple]) -> bool:
        wa, ta = a
        wb, tb = b
        if wa != wb:
            return wa < wb
        return ta < tb

    # ---------- ヒーロー報酬計算 ----------
    def _compute_hero_reward(self) -> float:
        hero = self.players[self.hero_idx]
        hero_foul = hero.foul
        n_opp = self.n_players - 1

        # --- ヒーローがファウルしたら即大きなマイナス ---
        if hero_foul:
            # 相手がどうであれ「自分のファウルは超痛い」
            return -20.0 * n_opp

        # ここから先はヒーローがノンファウルのときだけ
        h_top = eval_3(hero.top.cards)
        h_mid = eval_5(hero.mid.cards)
        h_bot = eval_5(hero.bot.cards)
        h_r_top = roy_top(hero.top.cards)
        h_r_mid = roy_mid(hero.mid.cards)
        h_r_bot = roy_bot(hero.bot.cards)
        hero_roy = h_r_top + h_r_mid + h_r_bot

        total_reward = 0.0

        for idx, opp in enumerate(self.players):
            if idx == self.hero_idx:
                continue

            opp_foul = opp.foul

            if opp_foul:
                # 相手だけファウル → 固定で +6
                total_reward += 6.0
                continue

            # 両方ノンファウル → ライン勝敗 + ロイヤリティ差
            o_top = eval_3(opp.top.cards)
            o_mid = eval_5(opp.mid.cards)
            o_bot = eval_5(opp.bot.cards)

            def line_score(a, b):
                wa, ta = a
                wb, tb = b
                if wa != wb:
                    return 1 if wa > wb else -1
                return 1 if ta > tb else (-1 if ta < tb else 0)

            score = 0
            score += line_score(h_top, o_top)
            score += line_score(h_mid, o_mid)
            score += line_score(h_bot, o_bot)

            o_roy = roy_top(opp.top.cards) + roy_mid(opp.mid.cards) + roy_bot(opp.bot.cards)
            score += (hero_roy - o_roy)

            total_reward += float(score)

        # ノンファウルで完走しただけでちょいボーナス
        total_reward += 2.0 * n_opp

        return total_reward
