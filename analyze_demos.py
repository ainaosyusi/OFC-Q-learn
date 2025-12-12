#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import glob
import json
from collections import Counter, defaultdict
from typing import Dict, List, Any


def load_jsonl(paths: List[str]) -> List[Dict[str, Any]]:
    rows = []
    for p in paths:
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("patterns", nargs="+", help="jsonl path or glob (e.g., demos/demo_*.jsonl)")
    args = ap.parse_args()

    paths = []
    for pat in args.patterns:
        g = glob.glob(pat)
        paths += g if g else [pat]

    rows = load_jsonl(paths)
    if not rows:
        print("No records.")
        return

    # episode grouping
    eps = defaultdict(list)
    for r in rows:
        eps[(r.get("seed"), r.get("episode"))].append(r)

    ep_returns = []
    ep_fouls = []
    steps_per_ep = []
    action_counter = Counter()
    chosen_not_first = Counter()

    for (seed, ep), traj in sorted(eps.items(), key=lambda x: (x[0][0], x[0][1])):
        traj = sorted(traj, key=lambda x: x.get("step", 0))
        ret = traj[-1].get("episode_return", None)
        foul = traj[-1].get("hero_foul", None)
        ep_returns.append(ret)
        ep_fouls.append(foul)
        steps_per_ep.append(len(traj))

        for rec in traj:
            a = rec.get("action_id")
            action_counter[a] += 1

        # “人間が選んだのが legal の何番目か” をざっくり見たければ：
        for rec in traj:
            legal = rec.get("legal_actions", [])
            a = rec.get("action_id")
            if legal and a in legal:
                idx = legal.index(a)
                chosen_not_first[idx] += 1

        print(f"[EP] seed={seed} ep={ep} steps={len(traj)} ret={ret} foul={foul}")

    # summary
    foul_known = [x for x in ep_fouls if x is not None]
    foul_rate = None
    if foul_known:
        foul_rate = sum(1 for x in foul_known if x) / len(foul_known)

    avg_ret = sum(x for x in ep_returns if x is not None) / max(1, sum(1 for x in ep_returns if x is not None))

    print("\n=== SUMMARY ===")
    print(f"episodes: {len(eps)}")
    print(f"avg_return: {avg_ret:.3f}")
    if foul_rate is not None:
        print(f"foul_rate: {foul_rate:.3f} (known {len(foul_known)}/{len(ep_fouls)})")
    print(f"avg_steps: {sum(steps_per_ep)/len(steps_per_ep):.2f}")

    print("\n=== TOP ACTIONS (chosen) ===")
    for a, c in action_counter.most_common(20):
        print(f"action_id={a} count={c}")

    print("\n=== chosen index in legal list (rough) ===")
    for idx, c in chosen_not_first.most_common(10):
        print(f"legal_index={idx} count={c}")


if __name__ == "__main__":
    main()
