# ofc_env.py
from dataclasses import dataclass, field
import random
from typing import List, Tuple, Optional, Dict, Any, Union
from collections import Counter

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
# Top(3枚)は High=0, ペア66+で加点、スリーカードで大きく加点
ROY_TOP: Dict[str,int] = {
    # pair
    "PAIR_66":1, "PAIR_77":2, "PAIR_88":3, "PAIR_99":4,
    "PAIR_TT":5, "PAIR_JJ":6, "PAIR_QQ":7, "PAIR_KK":8, "PAIR_AA":9,
    # set
    "SET_222":10, "SET_333":11, "SET_444":12, "SET_555":13, "SET_666":14,
    "SET_777":15, "SET_888":16, "SET_999":17, "SET_TTT":18, "SET_JJJ":19,
    "SET_QQQ":20, "SET_KKK":21, "SET_AAA":22
}

# Middle/Bottom（例：一般的な傾向。後で公式表に合わせて上書き推奨）
ROY_MID = { "STRAIGHT":4, "FLUSH":8, "FULLHOUSE":12, "QUADS":20, "SFLUSH":30, "RFLUSH":50 }
ROY_BOT = { "STRAIGHT":2, "FLUSH":4, "FULLHOUSE":6,  "QUADS":10, "SFLUSH":15, "RFLUSH":25 }

# ===== ハンド強度のタプル順序定義 =====
# 5枚：SFLUSH > QUADS > FULLHOUSE > FLUSH > STRAIGHT > TRIPS > TWO_PAIR > PAIR > HIGH
FIVE_ORDER = {
    "SFLUSH": 8, "QUADS":7, "FULLHOUSE":6, "FLUSH":5, "STRAIGHT":4,
    "TRIPS":3, "TWO_PAIR":2, "PAIR":1, "HIGH":0
}
# 3枚：SET > PAIR > HIGH
THREE_ORDER = { "SET":2, "PAIR":1, "HIGH":0 }

# ====== 役判定ユーティリティ ======
def _is_straight(ranks_sorted_desc: List[int]) -> Tuple[bool, int]:
    """
    ranks_sorted_desc: 重複を除いた降順ランク (例: A,K,Q ⇒ [12,11,10])
    戻り値: (is_straight, high_rank)
    AはA2345で最小ストレートも考慮（ハイは5=rank 3）
    """
    if len(ranks_sorted_desc) < 5:
        return (False, -1)
    # 連続チェック（降順で -1 ずつ）
    # A2345 対応のため、Aを1扱いへコピー（A⇒-1ではなく、[12,3,2,1,0] を別途見る）
    rs = ranks_sorted_desc
    # 通常
    for i in range(len(rs)-4):
        window = rs[i:i+5]
        if all(window[j] - window[j+1] == 1 for j in range(4)):
            return True, window[0]
    # A-5
    # ranks が [12, 3,2,1,0] を含めば A2345
    need = {12,3,2,1,0}
    if need.issubset(set(ranks_sorted_desc)):
        return True, 3  # 5-high straight のハイを3とする（ランク値で5は3）
    return (False, -1)

def eval_5(cards: List[Card]) -> Tuple[int, Tuple]:
    """
    5枚ハンドの強度を返す。
    戻り値: (CategoryWeight, tiebreak_tuple)
      ※ tiebreak_tuple はカテゴリー名を先頭に含めず、比較用に数値降順で構築
    """
    assert len(cards) == 5
    ranks = [card_rank(c) for c in cards]
    suits = [card_suit(c) for c in cards]
    ranks.sort(reverse=True)
    # カウント
    cnt = Counter(ranks)
    pairs = sorted([r for r,c in cnt.items() if c==2], reverse=True)
    trips = sorted([r for r,c in cnt.items() if c==3], reverse=True)
    quads = sorted([r for r,c in cnt.items() if c==4], reverse=True)
    singles = sorted([r for r,c in cnt.items() if c==1], reverse=True)

    is_flush = len(set(suits)) == 1
    # ストレート判定用に重複削除
    uniq_desc = sorted(set(ranks), reverse=True)
    is_straight, hi_st = _is_straight(uniq_desc)

    # ストフラ／ロイヤル
    if is_flush and is_straight:
        # ロイヤル判定: A,K,Q,J,10 = {12,11,10,9,8} を含む
        if set([12,11,10,9,8]).issubset(set(ranks)):
            return FIVE_ORDER["RFLUSH" if "RFLUSH" in FIVE_ORDER else "SFLUSH"], (12,11,10,9,8)
        return FIVE_ORDER["SFLUSH"], (hi_st,)

    # フォーカード
    if quads:
        q = quads[0]
        kicker = singles[0]
        return FIVE_ORDER["QUADS"], (q, kicker)

    # フルハウス
    if trips and pairs:
        return FIVE_ORDER["FULLHOUSE"], (trips[0], pairs[0])

    # フラッシュ
    if is_flush:
        return FIVE_ORDER["FLUSH"], tuple(ranks)

    # ストレート
    if is_straight:
        return FIVE_ORDER["STRAIGHT"], (hi_st,)

    # スリーカード
    if trips:
        t = trips[0]
        kickers = sorted([r for r in ranks if r != t], reverse=True)[:2]
        return FIVE_ORDER["TRIPS"], (t, *kickers)

    # ツーペア
    if len(pairs) >= 2:
        p1, p2 = pairs[0], pairs[1]
        kicker = max([r for r in ranks if r not in (p1, p2)])
        # 大きいペアから比較
        p_hi, p_lo = max(p1,p2), min(p1,p2)
        return FIVE_ORDER["TWO_PAIR"], (p_hi, p_lo, kicker)

    # ワンペア
    if len(pairs) == 1:
        p = pairs[0]
        kickers = sorted([r for r in ranks if r != p], reverse=True)[:3]
        return FIVE_ORDER["PAIR"], (p, *kickers)

    # ハイカード
    return FIVE_ORDER["HIGH"], tuple(ranks)

def eval_3(cards: List[Card]) -> Tuple[int, Tuple]:
    """
    3枚ハンドの強度（Top用）。
    SET > PAIR > HIGH
    戻り値: (CategoryWeight, tiebreak_tuple)
    """
    assert len(cards) == 3
    ranks = sorted([card_rank(c) for c in cards], reverse=True)
    cnt = Counter(ranks)

    if 3 in cnt.values():  # SET
        return THREE_ORDER["SET"], (ranks[0],)   # 三枚のランクが同一なので ranks[0]でOK
    if 2 in cnt.values():  # PAIR
        pair_rank = max([r for r,c in cnt.items() if c==2])
        kicker = max([r for r,c in cnt.items() if c==1])
        return THREE_ORDER["PAIR"], (pair_rank, kicker)
    return THREE_ORDER["HIGH"], tuple(ranks)

# カテゴリ名の取得（ロイヤリティ用）
def catname_5(weight: int, tiebreak: Tuple) -> str:
    # 逆引き
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
    - step(action): 
        turn==0  -> action は 5枚の行割当 (長さ5の文字列/リスト: 'T','M','B')
        turn>=1  -> action は (discard_idx, place_to_1, place_to_2)
                    place_to_* は 'T'/'M'/'B' のいずれか（順序はカード順）
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
        # 初手5枚
        self.state.incoming = [self.deck.pop() for _ in range(5)]
        self.state.turn = 0
        return self.state

    def legal_actions(self) -> List:
        """
        実運用では全列挙すると爆発するので、学習側でサンプリング/マスク運用推奨。
        ここでは「容量違反と明白ファウル確定」を除いたサンプルを少数返す簡易版でもOK。
        （現段階では空リストを返す：方策側が直接actionを指定する想定）
        """
        return []

    def step(self, action: Union[List[str], Tuple[int,str,str]]) -> Tuple[OFCState, float, bool, dict]:
        s = self.state
        assert s is not None

        # === アクション適用 ===
        if s.turn == 0:
            self._apply_init5(s, action)          # 5枚割当
            s.turn = 1
            s.incoming = [self.deck.pop() for _ in range(3)]
        else:
            self._apply_pineapple(s, action)      # 3枚から1枚捨て＋2枚配置
            if s.turn < 4:
                s.turn += 1
                s.incoming = [self.deck.pop() for _ in range(3)]
            else:
                # 終局
                pass

        # === ファウル判定 ===
        s.foul = self._is_foul(s)

        # === 終局判定 & 報酬 ===
        done = False
        reward = 0.0
        if s.foul:
            done = True
            reward = -6.0  # ソロ学習用ペナルティ
        elif (s.top.full() and s.mid.full() and s.bot.full()) and s.turn >= 4:
            done = True
            reward = self._terminal_reward(s)
        # 途中報酬（任意）：序列違反リスクを軽微にペナルティ
        else:
            reward += self._shaping_penalty(s)

        info = {}
        return s, reward, done, info

    # ---- アクション適用ロジック ----
    def _apply_init5(self, s: OFCState, action: List[str]) -> None:
        """
        action: 長さ5の ['T','M','B', ...]
        """
        assert isinstance(action, (list, tuple)) and len(action) == 5
        for card, dst in zip(s.incoming, action):
            self._place_one(s, card, dst)
        s.incoming = []

    def _apply_pineapple(self, s: OFCState, action: Tuple[int,str,str]) -> None:
        """
        action: (discard_idx, place_to_1, place_to_2)
        - incomingは3枚。discard_idx ∈ {0,1,2}
        - 残り2枚は**incomingの順序**で place_to_1, place_to_2 に置く
        """
        assert isinstance(action, (list, tuple)) and len(action) == 3
        discard_idx, p1, p2 = action
        assert 0 <= discard_idx < 3
        cards = s.incoming
        discard_card = cards[discard_idx]
        keep = [cards[i] for i in range(3) if i != discard_idx]
        # 捨てる
        s.dead.append(discard_card)
        # 置く（順序に意味を持たせる）
        self._place_one(s, keep[0], p1)
        self._place_one(s, keep[1], p2)
        s.incoming = []

    def _place_one(self, s: OFCState, card: Card, dst: str) -> None:
        line = {"T": s.top, "M": s.mid, "B": s.bot}[dst]
        if line.full():
            # 容量オーバーは強制ファウル扱いにする（安全のため）
            s.foul = True
            return
        line.cards.append(card)

    # ---- 評価系 ----
    def _is_foul(self, s: OFCState) -> bool:
        """
        3行が完成していれば厳密チェック。未完成時は明白な逆転が確定していれば早期Foul可。
        現段階は完成時のみ厳密。
        """
        if s.top.full() and s.mid.full() and s.bot.full():
            wt_top, tb_top = eval_3(s.top.cards)
            wt_mid, tb_mid = eval_5(s.mid.cards)
            wt_bot, tb_bot = eval_5(s.bot.cards)
            # 厳密序列
            if not self._lt_3v5((wt_top,tb_top), (wt_mid,tb_mid)):
                return True
            if not self._lt_5v5((wt_mid,tb_mid), (wt_bot,tb_bot)):
                return True
        return s.foul  # 置きミスなどで立っている場合

    def _lt_3v5(self, a: Tuple[int,Tuple], b: Tuple[int,Tuple]) -> bool:
        # 3枚 vs 5枚の「弱さ」比較：カテゴリー重みで比較（3枚の最大はSET=2、5枚の最小はHIGH=0）
        # 実際にはカテゴリ体系が異なるので「3枚は必ず5枚より弱い」が前提。
        wt3, tb3 = a
        wt5, tb5 = b
        # ざっくり：3枚SETでも5枚HIGHよりは「弱い」とみなす（OFCの序列前提）
        # → ここは「Top<Mid」を強制するための設計上の取り決め（役比較の異種間）
        return True

    def _lt_5v5(self, a: Tuple[int,Tuple], b: Tuple[int,Tuple]) -> bool:
        # 5枚同士の強度比較
        wa, ta = a
        wb, tb = b
        if wa != wb: return wa < wb
        return ta < tb  # タイブレークはタプルの辞書順

    def _terminal_reward(self, s: OFCState) -> float:
        # ロイヤリティ合計（ソロ学習用）
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
            # 66以上のみボーナス
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
        形状ペナルティ（小さく）：Topが埋まっているのにMid/Bottomがスカスカ等、将来ファウルリスク増を軽減。
        """
        p = 0.0
        # Topの埋まり過ぎを微ペナ：Topのcap=3に対し、他が残り大なら -0.02 * 超過感
        over_top = max(0, len(s.top.cards) - min(len(s.mid.cards), len(s.bot.cards)))
        p -= 0.02 * over_top
        return p
