#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_bc.py
- demos JSONL から (state_vec, action_id, legal_actions) を読み込み
- QNet を「275クラス分類」として事前学習（Behavior Cloning）
- 合法以外のlogitを -inf にして、合法の中で正解を当てるように学習する（安全）
"""

import argparse
import glob
import json
import os
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# ---- 既存 QNet を優先利用。無ければ簡易MLPでフォールバック ----
def load_qnet_or_fallback(input_dim: int, action_dim: int) -> nn.Module:
    try:
        import dqn_ofc_multi as dqn_mod
        QNet = getattr(dqn_mod, "QNet", None)
        if QNet is not None:
            return QNet(input_dim, action_dim)
    except Exception:
        pass

    # fallback
    return nn.Sequential(
        nn.Linear(input_dim, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, action_dim),
    )


def read_jsonl(paths: List[str]) -> List[Dict[str, Any]]:
    rows = []
    for p in paths:
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
    return rows


def make_dataset(rows: List[Dict[str, Any]], action_dim: int) -> Tuple[np.ndarray, np.ndarray, List[List[int]]]:
    X = []
    y = []
    legals = []
    for r in rows:
        s = r.get("state_vec", None)
        a = r.get("action_id", None)
        legal = r.get("legal_actions", None)
        if s is None or a is None or legal is None:
            continue

        s = np.array(s, dtype=np.float32).reshape(-1)
        a = int(a)
        legal = [int(x) for x in legal]

        # action_dim の外は捨てる（環境によっては別IDが混じる可能性があるため）
        if not (0 <= a < action_dim):
            continue
        legal = [x for x in legal if 0 <= x < action_dim]
        if a not in legal:
            # まれにログの不整合があると困るのでスキップ
            continue

        X.append(s)
        y.append(a)
        legals.append(legal)

    if not X:
        raise RuntimeError("No valid demo records found (check state_vec/action_id/legal_actions).")

    # state dim を揃える：最頻の長さに合わせる（混在対策）
    lens = [len(v) for v in X]
    target_dim = max(set(lens), key=lens.count)
    X2, y2, L2 = [], [], []
    for s, a, legal in zip(X, y, legals):
        if len(s) != target_dim:
            continue
        X2.append(s)
        y2.append(a)
        L2.append(legal)

    return np.stack(X2, axis=0), np.array(y2, dtype=np.int64), L2


def masked_cross_entropy(logits: torch.Tensor, targets: torch.Tensor, legal_lists: List[List[int]]) -> torch.Tensor:
    """
    logits: [B, A]
    targets: [B]
    legal_lists: length B, each list of legal action ids
    """
    B, A = logits.shape
    mask = torch.full((B, A), float("-inf"), device=logits.device)
    for i, legal in enumerate(legal_lists):
        if len(legal) == 0:
            continue
        idx = torch.tensor(legal, dtype=torch.long, device=logits.device)
        mask[i, idx] = 0.0
    masked_logits = logits + mask
    return nn.CrossEntropyLoss()(masked_logits, targets)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demos", nargs="+", required=True, help="jsonl paths or globs (e.g. demos/demo_*.jsonl)")
    ap.add_argument("--out", required=True, help="output .pt")
    ap.add_argument("--action_dim", type=int, default=275)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    paths = []
    for pat in args.demos:
        g = glob.glob(pat)
        paths += g if g else [pat]

    rows = read_jsonl(paths)
    X, y, legals = make_dataset(rows, args.action_dim)

    input_dim = X.shape[1]
    action_dim = args.action_dim
    device = torch.device(args.device)

    model = load_qnet_or_fallback(input_dim, action_dim).to(device)
    opt = optim.Adam(model.parameters(), lr=args.lr)

    # shuffle index
    idxs = np.arange(len(X))

    print("=== BC TRAIN ===")
    print(f"records: {len(X)} | input_dim={input_dim} | action_dim={action_dim} | device={device}")
    print(f"out: {args.out}")

    for ep in range(args.epochs):
        np.random.shuffle(idxs)
        total_loss = 0.0
        total = 0

        model.train()
        for start in range(0, len(idxs), args.batch):
            batch_ids = idxs[start : start + args.batch]
            xb = torch.tensor(X[batch_ids], dtype=torch.float32, device=device)
            yb = torch.tensor(y[batch_ids], dtype=torch.long, device=device)
            legal_b = [legals[i] for i in batch_ids]

            logits = model(xb)
            loss = masked_cross_entropy(logits, yb, legal_b)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += float(loss.item()) * len(batch_ids)
            total += len(batch_ids)

        avg = total_loss / max(1, total)
        print(f"[epoch {ep+1}/{args.epochs}] loss={avg:.6f}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save(model.state_dict(), args.out)
    print("saved:", args.out)


if __name__ == "__main__":
    main()
