#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import time
from datetime import datetime
from typing import Any, Dict, List, Tuple, Union

import numpy as np

from multi_ofc_env import OFCMultiEnv


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def safe_json(obj: Any) -> Any:
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, (list, tuple)):
        return [safe_json(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): safe_json(v) for k, v in obj.items()}
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return repr(obj)


def pretty_state(env: OFCMultiEnv) -> Dict[str, Any]:
    hero = env.players[env.hero_idx]
    return {
        "turn": getattr(env, "turn", None),
        "hero_idx": getattr(env, "hero_idx", None),
        "n_players": getattr(env, "n_players", None),
        "incoming": list(hero.incoming),
        "hero_top": list(hero.top.cards),
        "hero_mid": list(hero.mid.cards),
        "hero_bot": list(hero.bot.cards),
        "hero_dead": list(hero.dead),
        "hero_foul_flag": bool(hero.foul),
    }


def ask_place_for_each_card(cards: List[str]) -> List[str]:
    """
    cards の順番に対して、T/M/B を1枚ずつ質問して配置を作る
    返り値: ['T','M','B',...]
    """
    places = []
    for i, c in enumerate(cards):
        while True:
            ans = input(f"Card[{i}]={c} -> place? (T/M/B): ").strip().upper()
            if ans in ("T", "M", "B"):
                places.append(ans)
                break
            print("Please input T or M or B.")
    return places


def ask_pineapple_action(cards: List[str]) -> Tuple[int, str, str]:
    """
    Pineapple: 3枚から1枚捨てて、残り2枚をそれぞれ T/M/B に置く。
    返り値: (discard_index, place1, place2)
    """
    assert len(cards) == 3
    print("Incoming 3 cards:")
    for i, c in enumerate(cards):
        print(f"  {i}: {c}")
    while True:
        d = input("Choose discard index (0/1/2): ").strip()
        if d in ("0", "1", "2"):
            d = int(d)
            break
        print("Please input 0, 1, or 2.")

    keep = [cards[i] for i in range(3) if i != d]
    print(f"Keep cards: {keep[0]}, {keep[1]}")
    p1 = input(f"Place {keep[0]} to (T/M/B): ").strip().upper()
    while p1 not in ("T", "M", "B"):
        p1 = input("Please input T/M/B: ").strip().upper()
    p2 = input(f"Place {keep[1]} to (T/M/B): ").strip().upper()
    while p2 not in ("T", "M", "B"):
        p2 = input("Please input T/M/B: ").strip().upper()

    return (d, p1, p2)


def find_matching_legal_action(legal: List[Union[List[str], Tuple]], proposed) -> Union[List[str], Tuple]:
    """
    env.legal_actions() に対して、ユーザが入力した proposed が一致するものを返す。
    一致しなければ例外（＝その配置は不合法）でやり直しさせる。
    """
    # legal要素が list なら list同士で比較
    for a in legal:
        if a == proposed:
            return a
    raise ValueError("Proposed action is not legal under current rules.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_players", type=int, default=2)
    ap.add_argument("--hero_idx", type=int, default=0)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--outdir", type=str, default="demos")
    ap.add_argument("--prefix", type=str, default=None)
    args = ap.parse_args()

    ensure_dir(args.outdir)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = args.prefix or f"demo_{ts}_{args.n_players}p_s{args.seed}"
    out_path = os.path.join(args.outdir, f"{prefix}.jsonl")

    env = OFCMultiEnv(n_players=args.n_players, hero_idx=args.hero_idx, seed=args.seed)

    print("=== Human Demo Play (Ask T/M/B per card) ===")
    print(f"out: {out_path}")
    print(f"episodes: {args.episodes} | n_players={args.n_players} hero_idx={args.hero_idx} seed={args.seed}")
    print("形式：配られたカードを1枚ずつ、T/M/Bどこに置くか聞きます。\n")

    with open(out_path, "w", encoding="utf-8") as f:
        for ep in range(args.episodes):
            env.reset()
            done = False
            traj = []
            step = 0

            print(f"\n--- Episode {ep+1}/{args.episodes} ---")

            while not done:
                hero = env.players[env.hero_idx]
                state_raw = pretty_state(env)
                incoming = list(hero.incoming)
                turn = getattr(env, "turn", None)

                print("\n[STATE]")
                print(json.dumps(state_raw, ensure_ascii=False, indent=2))

                legal = env.legal_actions()

                # 入力 → proposed action を作る
                while True:
                    try:
                        if len(incoming) == 5:
                            print("\nInitial deal (5 cards). Decide T/M/B for each card in order.")
                            proposed = ask_place_for_each_card(incoming)  # ['T','M',...]
                        elif len(incoming) == 3:
                            print("\nPineapple deal (3 cards). Choose discard, then place 2 cards.")
                            proposed = ask_pineapple_action(incoming)      # (d,p1,p2)
                        else:
                            raise RuntimeError(f"Unexpected incoming length: {len(incoming)}")

                        # legal_actions の中に完全一致があるかチェック
                        action = find_matching_legal_action(legal, proposed)
                        break
                    except ValueError as e:
                        print(f"[ILLEGAL] {e}")
                        print("もう一度入力して。")
                    except KeyboardInterrupt:
                        print("\n[INFO] interrupted. Ending episode.")
                        done = True
                        action = None
                        break

                if done or action is None:
                    break

                # step
                obs2, r, done, info = env.step(action)

                rec = {
                    "episode": ep,
                    "step": step,
                    "timestamp": time.time(),
                    "n_players": args.n_players,
                    "hero_idx": args.hero_idx,
                    "seed": args.seed,
                    "turn": safe_json(turn),
                    "incoming": incoming,              # 入力時のカード
                    "action_raw": safe_json(action),   # envに渡した行動（list/tuple）
                    "legal_count": len(legal),
                    "reward_step": float(r),
                    "done": bool(done),
                    "info": safe_json(info),
                    "state_raw": safe_json(state_raw),
                }
                traj.append(rec)

                step += 1

            # episode end
            ret = sum(x["reward_step"] for x in traj)
            hero_foul = None
            try:
                hero_foul = bool(env.players[args.hero_idx].foul)
            except Exception:
                pass

            for rec in traj:
                rec["episode_return"] = float(ret)
                rec["hero_foul"] = hero_foul
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()

            print(f"[EP DONE] return={ret:.3f} foul={hero_foul} steps={len(traj)}")

    print("\nAll demos saved:", out_path)
    print("Next: python analyze_demos.py demos/demo_*.jsonl")


if __name__ == "__main__":
    main()
