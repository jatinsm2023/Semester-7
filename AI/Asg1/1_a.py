# algorithmic_min_heap.py
"""
Algorithmic solution to test whether a binary tree is a *min-heap*.

A min-heap must satisfy:
  (1) Completeness: it's a complete binary tree (levels filled left-to-right).
  (2) Order: parent.val <= child.val for every existing child.

We work from level-order arrays (None means "no node") and provide:
  - Node class
  - build_tree_from_level_list
  - check_completeness (BFS)
  - count_min_heap_violations (DFS)
  - is_min_heap: returns (bool, report)
  - a small __main__ demo
"""

from __future__ import annotations
from dataclasses import dataclass
from collections import deque
from typing import Optional, List, Tuple, Dict, Any

@dataclass
class Node:
    val: int
    left: Optional["Node"] = None
    right: Optional["Node"] = None

def build_tree_from_level_list(level: List[Optional[int]]) -> Optional[Node]:
    """
    Children of index i live at 2*i+1 and 2*i+2. None means "no node".
    """
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

def height(root: Optional[Node]) -> int:
    if root is None:
        return -1
    return 1 + max(height(root.left), height(root.right))

def check_completeness(root: Optional[Node]) -> Tuple[bool, Dict[str, Any]]:
    """
    Level-order traverse while keeping None placeholders.
    After the first None (gap), any later non-None ⇒ not complete.
    """
    if root is None:
        return True, {
            "size": 0,
            "height": -1,
            "first_gap_index": None,
            "explanation": "Empty tree is complete by convention.",
        }

    q = deque([root])
    seen_gap = False
    size = 0
    idx = 0
    first_gap_index = None

    while q:
        node = q.popleft()
        if node is None:
            if not seen_gap:
                seen_gap = True
                first_gap_index = idx
        else:
            size += 1
            if seen_gap:  # node after a gap means last level isn’t left-packed
                return False, {
                    "size": size,
                    "height": height(root),
                    "first_gap_index": first_gap_index,
                    "explanation": "Found a non-empty node after a gap in level-order.",
                }
            q.append(node.left)
            q.append(node.right)
        idx += 1

    return True, {
        "size": size,
        "height": height(root),
        "first_gap_index": first_gap_index,
        "explanation": "No nodes after a gap; shape is complete.",
    }


def count_min_heap_violations(root: Optional[Node]) -> Tuple[int, int]:
    """
    Return (#violations, #parent-child pairs checked).
    A violation is parent.val > child.val.
    """
    if root is None:
        return 0, 0

    violations = 0
    pairs = 0

    def dfs(node: Optional[Node]):
        nonlocal violations, pairs
        if node is None: return
        for child in (node.left, node.right):
            if child is not None:
                pairs += 1
                if node.val > child.val:
                    violations += 1
        dfs(node.left)
        dfs(node.right)

    dfs(root)
    return violations, pairs


def is_min_heap(root: Optional[Node]) -> Tuple[bool, Dict[str, Any]]:
    complete, c_report = check_completeness(root)
    violations, checked_pairs = count_min_heap_violations(root)
    order_ok = (violations == 0)
    report = {
        "complete": complete,
        "order_ok": order_ok,
        "violations": violations,
        "checked_pairs": checked_pairs,
        **c_report,
    }
    return (complete and order_ok), report


def to_level_list(root: Optional[Node]) -> List[Optional[int]]:
    if root is None: return []
    q = deque([root])
    out: List[Optional[int]] = []
    while q:
        node = q.popleft()
        if node is None:
            out.append(None)
        else:
            out.append(node.val)
            q.append(node.left)
            q.append(node.right)
    while out and out[-1] is None:
        out.pop()
    return out


if __name__ == "__main__":
    arr1 = [1, 4, 7, 9, 10, 8, 11]
    arr2 = [1, 2, 0, 9, 3, 8, 11]
    arr3 = [1, 4, 7, None, 10, 8, 11]

    for arr in (arr1, arr2, arr3):
        root = build_tree_from_level_list(arr)
        ok, rep = is_min_heap(root)
        print(f"\nArray: {arr}")
        print("Is min-heap? ", ok)
        print("Details:", rep)
