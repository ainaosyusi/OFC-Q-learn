#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, glob, json
from collections import Counter, defaultdict
from typing import Any, Dict, List, Tuple


def load_jsonl(paths: List[str]) -> List[Dict[str, Any]]:
    rows = []
    for p in paths:
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    return rows


def is_initial_action(a) -> bool:
    return isinstance(a, list) and len(a) == 5 and all(x in ("T", "M", "B") for x in a)


def is_pineapple_action(a) -> bool:
    # (discard_idx, p1, p2)
    return isinstance(a, list) and len(a) == 3 and isinstance(a[0], int) and a[1] in ("T","M","B") and a[2] in ("T","M","B")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("patterns", nargs="+", help="e.g. demos/demo_*.jsonl")
    args = ap.parse_args()

    paths = []
    for pat in args.patterns:
        g = glob.glob(pat)
        paths += g if g else [pat]

    rows = load_jsonl(paths)
    if not rows:
        print("No rows.")
        return

    # episode grouping
    eps = defaultdict(list)
    for r in rows:
        eps[(r.get("seed"), r.get("episode"))].append(r)

    foul_known = []
    rets = []
    steps = []

    init_place_counter = Counter()   # (card, place)
    init_pattern_counter = Counter() # tuple(places)
    disc_counter = Counter()         # discarded card
    place_counter = Counter()        # (card, place) for pineapple kept cards

    for (seed, ep), traj in sorted(eps.items(), key=lambda x: (x[0][0], x[0][1])):
        traj = sorted(traj, key=lambda x: x.get("step", 0))
        ret = traj[-1].get("episode_return", None)
        foul = traj[-1].get("hero_foul", None)
        rets.append(ret)
        if foul is not None:
            foul_known.append(bool(foul))
        steps.append(len(traj))

        print(f"[EP] seed={seed} ep={ep} steps={len(traj)} ret={ret} foul={foul}")

        for rec in traj:
            incoming = rec.get("incoming", [])
            a = rec.get("action_raw", None)

            # 初手：incoming 5枚に places 5個
            if isinstance(a, list) and len(a) == 5 and len(incoming) == 5:
                init_pattern_counter[tuple(a)] += 1
                for c, p in zip(incoming, a):
                    init_place_counter[(c, p)] += 1

            # pineapple：incoming 3枚に discard + 2places
            # ログ保存の都合で tuple が list 化されてるので list 想定
            if isinstance(a, list) and len(a) == 3 and len(incoming) == 3:
                d, p1, p2 = a
                if isinstance(d, int):
                    discarded = incoming[d]
                    kept = [incoming[i] for i in range(3) if i != d]
                    disc_counter[discarded] += 1
                    place_counter[(kept[0], p1)] += 1
                    place_counter[(kept[1], p2)] += 1

    avg_ret = sum(x for x in rets if x is not None) / max(1, sum(1 for x in rets if x is not None))
    foul_rate = (sum(1 for x in foul_known if x) / len(foul_known)) if foul_known else None

    print("\n=== SUMMARY ===")
    print(f"episodes: {len(eps)}")
    print(f"avg_return: {avg_ret:.3f}")
    if foul_rate is not None:
        print(f"foul_rate: {foul_rate:.3f} (known {len(foul_known)}/{len(eps)})")
    print(f"avg_steps: {sum(steps)/len(steps):.2f}")

    print("\n=== Initial: top patterns ===")
    for pat, c in init_pattern_counter.most_common(10):
        print(f"{pat}  count={c}")

    print("\n=== Initial: card->place (top 20) ===")
    for (card, place), c in init_place_counter.most_common(20):
        print(f"{card} -> {place}  count={c}")

    print("\n=== Pineapple: discarded cards (top 20) ===")
    for card, c in disc_counter.most_common(20):
        print(f"discard {card}  count={c}")

    print("\n=== Pineapple: kept card->place (top 20) ===")
    for (card, place), c in place_counter.most_common(20):
        print(f"{card} -> {place}  count={c}")


if __name__ == "__main__":
    main()
