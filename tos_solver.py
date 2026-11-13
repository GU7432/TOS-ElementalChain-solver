"""
5x5 消珠問題：最大化「平均每刪一格消除的珠數」
=================================================
規則（可在下方常數修改）：
  - 盤面 N x N，C 種屬性，補珠各屬性機率 1/C
  - 每步選一格刪除 → 上方珠子掉落 → 頂端補隨機珠
  - 橫/直連 >=3 同屬性即消除（L、T、+ 因重疊自然合併），連鎖持續到穩定
  - 報酬 = 該步連鎖消除的總珠數（被刪的那格不計）

演算法：Sampled Expectimax + 啟發式盤面價值 h(s)
  Q(s,a) ≈ E[ r(s,a) + λ·h(s') ]，用 S 次蒙地卡羅抽樣估計期望
  深度 2 時：Q(s,a) ≈ E[ r + max_a' Q1(s',a') ]，只對深度 1 排名前 K 的動作展開
  執行 python3 tos_solver.py 會跑出各策略的長期平均值做比較。
"""
import random

N, C = 5, 5
CELLS = N * N


# ---------------- 遊戲引擎 ----------------
def find_matches(b):
    """回傳所有要消除的格子 index 集合（直線 >=3，重疊自然合併成 L/T/+）"""
    m = set()
    for r in range(N):
        row = r * N
        for c in range(N - 2):
            i = row + c
            if b[i] == b[i + 1] == b[i + 2]:
                m.update((i, i + 1, i + 2))
    for c in range(N):
        for r in range(N - 2):
            i = r * N + c
            if b[i] == b[i + N] == b[i + 2 * N]:
                m.update((i, i + N, i + 2 * N))
    return m


def drop(b, removed, rng):
    """移除 removed 中的格子，每欄往下掉，頂端補 rng 產生的新珠"""
    for c in range(N):
        col = [b[r * N + c] for r in range(N) if (r * N + c) not in removed]
        col = [rng.randrange(C) for _ in range(N - len(col))] + col
        for r in range(N):
            b[r * N + c] = col[r]


def settle(b, rng):
    """連鎖消除到穩定，回傳總消除珠數"""
    total = 0
    while True:
        m = find_matches(b)
        if not m:
            return total
        total += len(m)
        drop(b, m, rng)


def step(b, a, rng):
    """刪除格子 a，回傳該步消除數（就地修改 b）"""
    drop(b, {a}, rng)
    return settle(b, rng)


def new_board(rng):
    b = [rng.randrange(C) for _ in range(CELLS)]
    settle(b, rng)
    return b


# ---------------- 盤面價值 h(s) ----------------
def heuristic(b, w_pair=1.0, w_gap=0.5):
    """
    盤面潛力：越多「差一顆就能連三」的結構，未來越容易一刪就消。
      pair : 相鄰同色 (XX)
      gap  : 隔一格同色 (X_X)，刪掉中間或其上方就可能接起來
    """
    s = 0.0
    for r in range(N):
        for c in range(N):
            i = r * N + c
            v = b[i]
            if c + 1 < N and b[i + 1] == v: s += w_pair
            if r + 1 < N and b[i + N] == v: s += w_pair
            if c + 2 < N and b[i + 2] == v: s += w_gap
            if r + 2 < N and b[i + 2 * N] == v: s += w_gap
    return s


# ---------------- 決策 ----------------
def q_values(b, rng, samples, lam, actions=None):
    """Q(s,a) ≈ (1/S) Σ [ r + λ h(s') ]"""
    actions = range(CELLS) if actions is None else actions
    q = {}
    for a in actions:
        tot = 0.0
        for _ in range(samples):
            bb = b[:]
            r = step(bb, a, rng)
            tot += r + lam * heuristic(bb)
        q[a] = tot / samples
    return q


def choose_depth1(b, rng, samples=16, lam=0.5):
    q = q_values(b, rng, samples, lam)
    return max(q, key=q.get)


def choose_depth2(b, rng, samples1=16, top_k=5, samples2=4, inner_samples=4, lam=0.5):
    """
    先用深度 1 粗篩出前 K 個動作，再對每個動作抽樣下一個盤面 s'，
    在 s' 上再做一次深度 1 的最大化：Q2(s,a) = E[ r + max_a' Q1(s',a') ]
    """
    q1 = q_values(b, rng, samples1, lam)
    cand = sorted(q1, key=q1.get, reverse=True)[:top_k]
    best, best_v = cand[0], -1e9
    for a in cand:
        tot = 0.0
        for _ in range(samples2):
            bb = b[:]
            r = step(bb, a, rng)
            q_next = q_values(bb, rng, inner_samples, lam)
            tot += r + max(q_next.values())
        v = tot / samples2
        if v > best_v:
            best, best_v = a, v
    return best


# ---------------- 評估 ----------------
def evaluate(policy, steps, seed):
    rng = random.Random(seed)
    b = new_board(rng)
    total = 0
    for _ in range(steps):
        a = policy(b, rng)
        total += step(b, a, rng)
    return total / steps


if __name__ == "__main__":
    tests = [
        ("隨機", lambda b, g: g.randrange(CELLS), 100000),
        ("貪婪 (λ=0)", lambda b, g: choose_depth1(b, g, lam=0.0), 2000),
        ("深度1 + h(s)", lambda b, g: choose_depth1(b, g, lam=0.5), 2000),
        ("深度2 + h(s)", lambda b, g: choose_depth2(b, g), 300),
    ]
    for name, pol, n in tests:
        print(f"{name:14s} 平均每刪一格消除 ≈ {evaluate(pol, n, 42):.3f}  ({n} 步)")
