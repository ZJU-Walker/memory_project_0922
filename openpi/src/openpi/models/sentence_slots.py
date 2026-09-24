"""Deterministic template discovery from a training reference vocabulary (no task names)."""


def template_masks(rows: tuple[tuple[int, ...], ...], max_diff: int) -> tuple[tuple[bool, ...], ...]:
    adjacency = [set() for _ in rows]
    differences = {}
    for i, left in enumerate(rows):
        for j in range(i + 1, len(rows)):
            right = rows[j]
            if len(left) != len(right):
                continue
            changed = [p for p, (a, b) in enumerate(zip(left, right, strict=True)) if a != b]
            if 1 <= len(changed) <= max_diff:
                adjacency[i].add(j)
                adjacency[j].add(i)
                differences[i, j] = changed
    seen = set()
    masks = [()] * len(rows)
    for start in range(len(rows)):
        if start in seen:
            continue
        component, stack = set(), [start]
        while stack:
            node = stack.pop()
            if node in component:
                continue
            component.add(node)
            stack.extend(adjacency[node] - component)
        seen.update(component)
        variable = set()
        for (a, b), changed in differences.items():
            if a in component and b in component:
                variable.update(changed)
        for node in component:
            masks[node] = tuple(p not in variable for p in range(len(rows[node])))
    return tuple(masks)


def template_representatives(rows: tuple[tuple[int, ...], ...], max_diff: int = 2) -> tuple[int, ...]:
    """One reference-row index per distinct encoded template, in vocabulary order."""
    masks = template_masks(rows, max_diff)
    seen, representatives = set(), []
    for i, (row, mask) in enumerate(zip(rows, masks, strict=True)):
        signature = tuple(token for token, keep in zip(row, mask, strict=True) if keep)
        if not signature:
            raise ValueError("template discovery erased every token; refine the reference vocabulary or max_diff")
        if signature not in seen:
            seen.add(signature)
            representatives.append(i)
    return tuple(representatives)
