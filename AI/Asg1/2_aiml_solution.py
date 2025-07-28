# aiml_min_heap.py
"""
A minimal AIML pipeline that *learns* to classify binary trees as min-heaps.
The goal is instructional: compare an ML approach with the exact algorithm.

Steps:
  - make synthetic trees (positives = valid heaps; negatives = violated order or shape)
  - compute small, interpretable features
  - train logistic regression from scratch (NumPy)
  - evaluate and demo predictions
"""

from __future__ import annotations
from dataclasses import dataclass
from collections import deque
from typing import Optional, List, Tuple, Dict, Any
import numpy as np
import random
import json
import os


@dataclass
class Node:
    val: int
    left: Optional["Node"] = None
    right: Optional["Node"] = None


def build_tree_from_level_list(level: List[Optional[int]]) -> Optional[Node]:
    if not level or level[0] is None:
        return None
    nodes: List[Optional[Node]] = [Node(v) if v is not None else None for v in level]
    for i in range(len(level)):
        if nodes[i] is None: 
            continue
        li, ri = 2 * i + 1, 2 * i + 2
        nodes[i].left = nodes[li] if li < len(level) else None
        nodes[i].right = nodes[ri] if ri < len(level) else None
    return nodes[0]


def check_completeness(root: Optional[Node]) -> bool:
    if root is None: 
        return True
    q = deque([root])
    seen_gap = False
    while q:
        node = q.popleft()
        if node is None:
            seen_gap = True
        else:
            if seen_gap: 
                return False
            q.append(node.left)
            q.append(node.right)
    return True


def count_min_heap_violations(root: Optional[Node]) -> Tuple[int, int]:
    if root is None: 
        return 0, 0
    violations = 0
    pairs = 0
    def dfs(n: Optional[Node]):
        nonlocal violations, pairs
        if n is None: 
            return
        for c in (n.left, n.right):
            if c is not None:
                pairs += 1
                if n.val > c.val:
                    violations += 1
        dfs(n.left); dfs(n.right)
    dfs(root)
    return violations, pairs


def tree_height(root: Optional[Node]) -> int:
    if root is None: 
        return -1
    return 1 + max(tree_height(root.left), tree_height(root.right))


FEATURE_NAMES = [
    "n_nodes", "height", "complete", "violations", "parent_child_pairs",
    "violation_ratio", "array_left_packed", "val_min", "val_max", "val_mean"
]

def featurize_tree(level: List[Optional[int]]) -> Tuple[np.ndarray, Dict[str, float]]:
    root = build_tree_from_level_list(level)
    n_nodes = sum(v is not None for v in level)
    max_idx = max((i for i, v in enumerate(level) if v is not None), default=-1)
    array_left_packed = all(v is not None for v in level[:max_idx + 1]) if max_idx >= 0 else True

    complete = 1.0 if check_completeness(root) else 0.0
    viol, pairs = count_min_heap_violations(root)
    violation_ratio = (viol / pairs) if pairs else 0.0
    ht = float(tree_height(root))

    vals = [v for v in level if v is not None]
    vmin = float(min(vals)) if vals else 0.0
    vmax = float(max(vals)) if vals else 0.0
    vmean = float(sum(vals) / len(vals)) if vals else 0.0

    x = np.array([
        float(n_nodes), ht, complete, float(viol), float(pairs),
        float(violation_ratio), 1.0 if array_left_packed else 0.0,
        vmin, vmax, vmean
    ], dtype=float)
    named = {k: v for k, v in zip(FEATURE_NAMES, x)}
    return x, named


def make_complete_array(n: int) -> List[Optional[int]]:
    return [0] * n

def generate_min_heap_array(n: int, value_start: int = 0, max_step: int = 5) -> List[Optional[int]]:
    arr = make_complete_array(n)
    curr = value_start
    for i in range(n):
        step = 0 if i == 0 else random.randint(0, max_step)
        curr += step
        arr[i] = curr
    return arr

def introduce_order_violation(arr: List[Optional[int]], n_swaps: int = 1) -> List[Optional[int]]:
    arr = arr.copy(); n = len(arr)
    for _ in range(n_swaps):
        i = random.randrange(n)
        li, ri = 2*i+1, 2*i+2
        kids = [j for j in (li, ri) if j < n]
        if not kids: 
            continue
        j = random.choice(kids)
        arr[i], arr[j] = arr[j], arr[i]  
    return arr

def introduce_incompleteness(arr: List[Optional[int]]) -> List[Optional[int]]:
    arr = arr.copy(); n = len(arr)
    for i in range(n):
        li, ri = 2*i+1, 2*i+2
        if ri < n and arr[li] is not None and arr[ri] is not None:
            arr[li] = None
            return arr
    arr[random.randrange(max(1, n-1))] = None
    return arr

def label_is_min_heap(level: List[Optional[int]]) -> int:
    root = build_tree_from_level_list(level)
    return int(check_completeness(root) and count_min_heap_violations(root)[0] == 0)

def synthesize_dataset(num_pos: int = 200, num_neg: int = 200,
                       min_size: int = 3, max_size: int = 31) -> Tuple[np.ndarray, np.ndarray, List[List[Optional[int]]]]:
    rng = random.Random(1234); random.seed(1234)
    arrays: List[List[Optional[int]]] = []

    for _ in range(num_pos):
        n = rng.randint(min_size, max_size)
        arrays.append(generate_min_heap_array(n, value_start=rng.randint(0, 20), max_step=5))

    for _ in range(num_neg):
        n = rng.randint(min_size, max_size)
        base = generate_min_heap_array(n, value_start=rng.randint(0, 20), max_step=5)
        arr = introduce_order_violation(base, n_swaps=rng.randint(1, 3)) if rng.random() < 0.5 else introduce_incompleteness(base)
        arrays.append(arr)

    rng.shuffle(arrays)

    X_list, y_list = [], []
    for arr in arrays:
        x, _ = featurize_tree(arr)
        y = label_is_min_heap(arr)
        X_list.append(x); y_list.append(y)

    return np.vstack(X_list).astype(float), np.array(y_list).astype(float), arrays


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))

def train_logreg(X: np.ndarray, y: np.ndarray, lr: float = 0.2, epochs: int = 1500, l2: float = 1e-3) -> Dict[str, Any]:
    n, d = X.shape
    W = np.zeros(d, dtype=float); b = 0.0
    history = {"loss": [], "acc": []}
    for t in range(epochs):
        z = X @ W + b
        p = sigmoid(z)
        eps = 1e-9
        loss = -np.mean(y*np.log(p+eps) + (1-y)*np.log(1-p+eps)) + 0.5*l2*np.sum(W*W)
        gW = (X.T @ (p - y))/n + l2*W
        gb = float(np.mean(p - y))
        W -= lr*gW; b -= lr*gb
        if t % 50 == 0 or t == epochs-1:
            acc = float(np.mean((p >= 0.5).astype(float) == y))
            history["loss"].append(float(loss)); history["acc"].append(acc)
    return {"W": W, "b": b, "history": history}

def predict_proba(X: np.ndarray, model: Dict[str, Any]) -> np.ndarray:
    return sigmoid(X @ model["W"] + model["b"])


def train_and_eval() -> Dict[str, Any]:
    X, y, _ = synthesize_dataset()
    # 80/20 split
    n = len(y); idx = np.arange(n); np.random.seed(42); np.random.shuffle(idx)
    k = int(0.8*n); tr, te = idx[:k], idx[k:]
    Xtr, ytr, Xte, yte = X[tr], y[tr], X[te], y[te]

    model = train_logreg(Xtr, ytr, lr=0.2, epochs=1500, l2=1e-3)

    p_tr = predict_proba(Xtr, model); p_te = predict_proba(Xte, model)
    acc_tr = float(np.mean((p_tr >= 0.5).astype(float) == ytr))
    acc_te = float(np.mean((p_te >= 0.5).astype(float) == yte))
    return {"model": model, "train_acc": acc_tr, "test_acc": acc_te}

def predict_tree(level: List[Optional[int]], model: Dict[str, Any]) -> Tuple[float, int, Dict[str, float]]:
    x, feats = featurize_tree(level)
    p = float(predict_proba(x.reshape(1, -1), model)[0])
    return p, int(p >= 0.5), feats


if __name__ == "__main__":
    stats = train_and_eval()
    print(f"Train acc: {stats['train_acc']:.3f} | Test acc: {stats['test_acc']:.3f}")

    model = stats["model"]
    examples = [
        [1, 4, 7, 9, 10, 8, 11],    # valid heap
        [1, 2, 0, 9, 3, 8, 11],     # order violation
        [1, 4, 7, None, 10, 8, 11], # completeness violation
    ]
    for arr in examples:
        p, pred, feats = predict_tree(arr, model)
        print("\nArray:", arr)
        print("Features:", json.dumps(feats, indent=2))
        print(f"Predicted P(min-heap) = {p:.3f}  -> label = {bool(pred)}")
