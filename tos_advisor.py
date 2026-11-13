"""
給定盤面 → 推薦「長期期望值」最高的刪除位置
=============================================
問題本質是一個「平均報酬 MDP」：目標是最大化長期的平均每步消除數，
而不是這一步消最多。所以每個動作要看

    Q(s,a) = E[ r(s,a) - ρ + V(s') ]

  r     ：這一步（含連鎖）消除的珠數
  ρ     ：最佳策略下的長期平均每步消除數
  V(s') ：盤面 s' 的「差分價值」= 從 s' 出發，未來總消除數比平均水準多出多少

  → Q(s,a1) - Q(s,a2) 就是「選 a1 比選 a2，長期下來總共多消幾顆」的期望。
    單步消 0 顆但把盤面整理好、讓之後多消 3 顆的動作，Q 會比單步消 3 顆但把盤面弄爛的高。

V(s) 用 n-tuple network 學（2048 AI 常用的方法）：
  - 盤面切成多個 2x3 / 3x2 / 整列 / 整行 的小區塊，每個區塊的「同色樣式」查表加總
  - 只看「哪些格子同色」而非實際顏色（補珠各色等機率 → 顏色可互換）
  - 左右鏡像共用權重（重力只在垂直方向）
  - 用差分 TD 學習（average-reward TD）自我對弈訓練

另外 advise --verify 會對前幾名動作做蒙地卡羅 rollout（真的往後模擬 H 步），
直接驗證長期總消除數，不完全依賴學出來的 V。

用法：
  python3 tos_advisor.py train --minutes 10          # 訓練價值函數（存到 tos_value.json）
  python3 tos_advisor.py eval  --steps 3000          # 和舊策略比較長期平均
  python3 tos_advisor.py advise board.txt            # 給盤面，輸出建議
  python3 tos_advisor.py advise --board "12345/23451/34512/45123/51234" --verify

盤面格式：5 行、每行 5 個符號（任意字元，例如 水火木光暗 或 RGBYP 或 01234），
不同符號 = 不同屬性；因為屬性對稱，符號對應到哪個數字不影響結果。
"""
import argparse
import json
import math
import os
import random
import sys
import time
from multiprocessing import Pool

from tos_solver import N, C, CELLS, find_matches, drop, settle, step, new_board, heuristic

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL = os.path.join(HERE, "tos_value.json")


# =====================================================================
# n-tuple 價值函數
# =====================================================================
def _canon_table(k):
    """長度 k 的顏色序列（base-C 編碼）→ 同色樣式編號（依首次出現重新編號）"""
    ids, table = {}, []
    for code in range(C ** k):
        seq, x = [], code
        for _ in range(k):
            seq.append(x % C)
            x //= C
        seq.reverse()
        relabel, pat = {}, []
        for v in seq:
            if v not in relabel:
                relabel[v] = len(relabel)
            pat.append(relabel[v])
        table.append(ids.setdefault(tuple(pat), len(ids)))
    return table, len(ids)


def _mirror(cells):
    return [(i // N) * N + (N - 1 - i % N) for i in cells]


def _build_tuples():
    """回傳 [(cells, group_key)]；同 group 共用一張權重表（鏡像共用）"""
    shapes = []
    for r in range(N - 1):                       # 2x3 橫向
        for c in range(N - 2):
            shapes.append([(r + dr) * N + c + dc for dr in range(2) for dc in range(3)])
    for r in range(N - 2):                       # 3x2 直向
        for c in range(N - 1):
            shapes.append([(r + dr) * N + c + dc for dr in range(3) for dc in range(2)])
    for r in range(N):                           # 整列
        shapes.append([r * N + c for c in range(N)])
    for c in range(N):                           # 整行
        shapes.append([r * N + c for r in range(N)])

    out, done, g = [], set(), 0
    for cells in shapes:
        key, mkey = tuple(cells), tuple(_mirror(cells))
        if key in done:
            continue
        done.update((key, mkey))
        out.append((cells, g))
        out.append((list(mkey), g))             # 鏡像共用同一張表（等於對稱化）
        g += 1
    return out


class ValueNet:
    def __init__(self):
        self.tuples = _build_tuples()
        sizes = sorted({len(c) for c, _ in self.tuples})
        self.canon = {}
        npat = {}
        for k in sizes:
            self.canon[k], npat[k] = _canon_table(k)
        # 每個 group 的權重 offset
        group_len = {}
        for cells, g in self.tuples:
            group_len[g] = len(cells)
        self.offset, off = {}, 0
        for g in sorted(group_len):
            self.offset[g] = off
            off += npat[group_len[g]]
        self.size = off
        self.W = [0.0] * self.size
        self.n_terms = len(self.tuples)
        self._compile()

    def _compile(self):
        """把 V(b) 與 features(b) 產生成展開的 Python 程式碼（快很多）"""
        idx_exprs = []
        for cells, g in self.tuples:
            k = len(cells)
            code = "+".join(f"b[{c}]*{C ** (k - 1 - j)}" for j, c in enumerate(cells))
            idx_exprs.append(f"{self.offset[g]}+P{k}[{code}]")
        env = {f"P{k}": t for k, t in self.canon.items()}
        src_v = "def V(b, W):\n    return (" + "\n        + ".join(f"W[{e}]" for e in idx_exprs) + ")\n"
        src_f = "def F(b):\n    return [" + ",\n        ".join(idx_exprs) + "]\n"
        exec(src_v, env)
        exec(src_f, env)
        self._V, self._F = env["V"], env["F"]

    def value(self, b):
        return self._V(b, self.W)

    def features(self, b):
        return self._F(b)

    def save(self, path, rho, meta=None):
        with open(path, "w") as f:
            json.dump({"N": N, "C": C, "rho": rho, "W": self.W, "meta": meta or {}}, f)

    @classmethod
    def load(cls, path):
        net = cls()
        with open(path) as f:
            d = json.load(f)
        if d["N"] != N or d["C"] != C or len(d["W"]) != net.size:
            raise ValueError("模型檔與目前 N/C 設定不符，請重新訓練")
        net.W = d["W"]
        return net, d["rho"], d.get("meta", {})


# =====================================================================
# 決策：Q(s,a) = E[r + V(s')]
#   刪除後該欄頂端補 1 顆新珠 → 5 種顏色「精確列舉」
#   若有連鎖，後續補珠再用 chain_samples 次抽樣
# =====================================================================
def delete_with_color(b, a, color):
    c, r = a % N, a // N
    for rr in range(r, 0, -1):
        b[rr * N + c] = b[(rr - 1) * N + c]
    b[c] = color


def q_values(b, net, rng, chain_samples=1, actions=None):
    """回傳 (Q 列表, 該步期望消除數列表)；Q 不含 -ρ（排序用不到）"""
    V, W = net._V, net.W
    actions = range(CELLS) if actions is None else actions
    q = [float("-inf")] * CELLS
    imm = [0.0] * CELLS
    for a in actions:
        tot = rt = 0.0
        for color in range(C):
            bb = b[:]
            delete_with_color(bb, a, color)
            m = find_matches(bb)
            if not m:
                tot += V(bb, W)
                continue
            for _ in range(chain_samples):
                b2 = bb[:]
                drop(b2, m, rng)
                r = len(m) + settle(b2, rng)
                tot += (r + V(b2, W)) / chain_samples
                rt += r / chain_samples
        q[a] = tot / C
        imm[a] = rt / C
    return q, imm


def best_action(b, net, rng, chain_samples=1):
    q, _ = q_values(b, net, rng, chain_samples)
    best = max(q)
    ties = [a for a in range(CELLS) if q[a] >= best - 1e-12]
    return rng.choice(ties), best


# =====================================================================
# 訓練（多程序：worker 用快照權重自我對弈產生 (features, target)，主程序做 SGD）
# =====================================================================
_W_NET = None
_W_BOARD = None
_W_RNG = None


def _worker_init(seed_base):
    global _W_NET, _W_RNG
    _W_NET = ValueNet()
    _W_RNG = random.Random(seed_base + os.getpid())


def _worker_play(args):
    global _W_BOARD
    W, steps = args
    net, rng = _W_NET, _W_RNG
    net.W = W
    if _W_BOARD is None:
        _W_BOARD = new_board(rng)
    b = _W_BOARD
    out = []
    for _ in range(steps):
        a, qbest = best_action(b, net, rng)
        feats = net.features(b)
        r = step(b, a, rng)
        out.append((feats, qbest, r))  # 目標 = max_a E[r + V(s')] - ρ（ρ 由主程序減）
    return out


def train(args):
    net = ValueNet()
    rho = 0.0
    if args.resume and os.path.exists(args.model):
        net, avg0, meta0 = ValueNet.load(args.model)
        rho = meta0.get("rho_td", 0.0)
        print(f"接續訓練 {args.model}")
    print(f"權重數 {net.size}, 每盤面 {net.n_terms} 個 tuple, workers={args.workers}")
    W = net.W
    n = net.n_terms
    deadline = time.time() + args.minutes * 60
    total = 0
    rewards_window = []
    alpha0 = args.alpha
    avg_r = 3.5 if not (args.resume and os.path.exists(args.model)) else avg0
    last_print = 0
    with Pool(args.workers, initializer=_worker_init, initargs=(args.seed,)) as pool:
        while time.time() < deadline:
            results = pool.map(_worker_play, [(W, args.chunk)] * args.workers)
            for out in results:
                for feats, qbest, r in out:
                    total += 1
                    avg_r += 1e-4 * (r - avg_r)
                    rewards_window.append(r)
                    v = 0.0
                    for f in feats:
                        v += W[f]
                    delta = (qbest - rho) - v
                    # 學習率隨時間衰減
                    lr = alpha0 / (1 + total / args.decay)
                    g = lr * delta / n
                    for f in feats:
                        W[f] += g
                    # 差分 TD（Wan et al. 2021）：ρ 用 TD 誤差更新，
                    # 吸收 max 帶來的偏差，避免 V 的常數項無限漂移
                    rho += args.beta * lr * delta
            if len(rewards_window) > 20000:
                rewards_window = rewards_window[-20000:]
            if time.time() - last_print > 15:
                last_print = time.time()
                avg = sum(rewards_window) / len(rewards_window)
                left = max(0, deadline - time.time())
                print(f"  樣本 {total:8d}  ρ_td={rho:.3f}  |V|≈{abs(net.value(new_board(random.Random(total)))):.1f}  近 {len(rewards_window)} 步平均={avg:.3f}  剩 {left/60:.1f} 分", flush=True)
                net.save(args.model, avg_r, {"samples": total, "rho_td": rho})
    net.save(args.model, avg_r, {"samples": total, "rho_td": rho})
    print(f"完成：{total} 樣本，訓練中平均每步消除≈{avg_r:.3f}，已存到 {args.model}")


# =====================================================================
# 評估：各策略長期平均（多程序，每個 worker 一條長軌跡）
# =====================================================================
def _eval_worker(args):
    name, steps, seed, model = args
    rng = random.Random(seed)
    net = None
    if name == "learned":
        net, _, _ = ValueNet.load(model)
    elif name == "greedy":
        net = ValueNet()  # W 全 0 → 純貪婪（用精確列舉的期望）
    b = new_board(rng)
    total = 0
    for _ in range(steps):
        if name == "random":
            a = rng.randrange(CELLS)
        elif name == "heur":
            from tos_solver import choose_depth1
            a = choose_depth1(b, rng, lam=0.5)
        else:
            a, _ = best_action(b, net, rng)
        total += step(b, a, rng)
    return total, steps


def evaluate(args):
    names = args.policies.split(",")
    per = max(1, args.steps // args.workers)
    with Pool(args.workers) as pool:
        for name in names:
            t0 = time.time()
            res = pool.map(_eval_worker, [(name, per, args.seed + i, args.model) for i in range(args.workers)])
            means = [t / s for t, s in res]
            mu = sum(means) / len(means)
            sd = math.sqrt(sum((x - mu) ** 2 for x in means) / max(1, len(means) - 1))
            se = sd / math.sqrt(len(means))
            print(f"{name:8s} 長期平均每步消除 ≈ {mu:.3f} ± {se:.3f}  ({per * args.workers} 步, {time.time()-t0:.0f}s)", flush=True)


# =====================================================================
# 建議：給盤面 → 排名
# =====================================================================
def parse_board(text):
    rows = [r for r in (x.replace(" ", "").replace("\t", "") for x in text.replace("/", "\n").splitlines()) if r]
    if len(rows) != N or any(len(r) != N for r in rows):
        raise ValueError(f"盤面需為 {N} 行、每行 {N} 個符號，收到：{rows}")
    sym = {}
    b = []
    for r in rows:
        for ch in r:
            if ch not in sym:
                sym[ch] = len(sym)
            b.append(sym[ch])
    if len(sym) > C:
        raise ValueError(f"盤面有 {len(sym)} 種符號，但 C={C}")
    return b, rows


_ROLLOUT_CACHE = {}


def _rollout_worker(args):
    b0, a, seed, horizon, model = args
    if model not in _ROLLOUT_CACHE:          # 每個子程序只載入一次模型
        _ROLLOUT_CACHE[model] = ValueNet.load(model)
    net, rho, _ = _ROLLOUT_CACHE[model]
    rng = random.Random(seed)
    b = b0[:]
    total = step(b, a, rng) - rho
    for _ in range(horizon):
        a2, _ = best_action(b, net, rng)
        total += step(b, a2, rng) - rho
    return a, total + net.value(b)


def rollout_verify(b, cand, model, rollouts=200, horizon=20, seed=1, pool=None, workers=None):
    """
    對候選動作做蒙地卡羅 rollout（同一組亂數 = common random numbers）。
    回傳依長期值排序的 [{"a", "diff"（與第一差）, "se"（成對標準誤，第一名為 None）}]
    """
    jobs = [(b, a, seed * 100003 + k, horizon, model) for k in range(rollouts) for a in cand]
    if pool is None:
        with Pool(workers or max(1, (os.cpu_count() or 2) - 2)) as p:
            res = p.map(_rollout_worker, jobs, chunksize=4)
    else:
        res = pool.map(_rollout_worker, jobs, chunksize=4)
    acc = {a: [] for a in cand}
    for a, v in res:          # jobs 依 (k, a) 順序，所以 acc[a][k] 對應同一組亂數
        acc[a].append(v)
    means = {a: sum(acc[a]) / len(acc[a]) for a in cand}
    lead = max(cand, key=means.get)

    def paired_se(a):
        d = [x - y for x, y in zip(acc[a], acc[lead])]
        mu = sum(d) / len(d)
        return math.sqrt(sum((x - mu) ** 2 for x in d) / max(1, len(d) - 1) / len(d))

    return [{"a": a, "diff": means[a] - means[lead], "se": None if a == lead else paired_se(a)}
            for a in sorted(cand, key=lambda a: -means[a])]


def advise(args):
    text = args.board if args.board else (open(args.file).read() if args.file else sys.stdin.read())
    b, rows = parse_board(text)
    if find_matches(b):
        print("注意：輸入盤面本身已有三連（遊戲中應已被消除），照樣計算。")
    net, rho, meta = ValueNet.load(args.model)
    rng = random.Random(args.seed)
    q, imm = q_values(b, net, rng, chain_samples=args.samples)
    order = sorted(range(CELLS), key=lambda a: -q[a])
    best = q[order[0]]

    print(f"模型：{os.path.basename(args.model)}（訓練樣本 {meta.get('samples', '?')}），訓練時長期平均每步消除≈{rho:.3f}\n")
    print("盤面：")
    for r in rows:
        print("   " + " ".join(r))

    def grid(title, vals, fmt):
        print(f"\n{title}")
        print("      " + " ".join(f"{c+1:>6d}" for c in range(N)))
        for r in range(N):
            print(f"   {r+1:>2d} " + " ".join(fmt(vals[r * N + c]) for c in range(N)))

    grid("該步期望消除數（短期）", imm, lambda x: f"{x:6.2f}")
    grid("長期優勢 Q(a) - Q(最佳)  （0 = 最佳，負數 = 長期少消幾顆）", [x - best for x in q], lambda x: f"{x:6.2f}")

    print(f"\n前 {args.top} 名（列, 行 從 1 起算，左上為 1,1）：")
    print("  名次  位置   該步期望消除   長期差距（相對最佳，總消除顆數）")
    for i, a in enumerate(order[: args.top]):
        print(f"  {i+1:>3d}   ({a//N+1},{a%N+1})   {imm[a]:9.2f}      {q[a]-best:8.2f}")
    greedy_a = max(range(CELLS), key=lambda a: imm[a])
    if greedy_a != order[0] and imm[greedy_a] > imm[order[0]] + 1e-9:
        print(f"\n  ※ 單步最多的是 ({greedy_a//N+1},{greedy_a%N+1})（期望 {imm[greedy_a]:.2f} 顆），"
              f"但長期來看比最佳少 {best - q[greedy_a]:.2f} 顆。")

    if args.verify:
        cand = order[: args.verify_top]
        print(f"\n蒙地卡羅驗證：前 {len(cand)} 名 × {args.rollouts} 次 rollout，每次往後模擬 {args.horizon} 步 + 尾端 V ...")
        t0 = time.time()
        res = rollout_verify(b, cand, args.model, args.rollouts, args.horizon, args.seed, workers=args.workers)
        print("  名次  位置   該步期望消除   與第一差（長期總消除）  ±成對標準誤   ({:.0f}s)".format(time.time() - t0))
        for i, x in enumerate(res):
            a = x["a"]
            se = "" if x["se"] is None else f"{x['se']:6.2f}"
            print(f"  {i+1:>3d}   ({a//N+1},{a%N+1})   {imm[a]:9.2f}      {x['diff']:10.2f}            {se}")
        print("  （差距小於約 2 倍標準誤時，兩者可視為差不多好）")


# =====================================================================
def main():
    p = argparse.ArgumentParser(description="TOS 消珠長期期望值顧問")
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train", help="訓練價值函數")
    t.add_argument("--minutes", type=float, default=10)
    t.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    t.add_argument("--chunk", type=int, default=100, help="每個 worker 每輪的步數")
    t.add_argument("--alpha", type=float, default=0.3)
    t.add_argument("--beta", type=float, default=0.5, help="ρ 相對於 α 的更新比例")
    t.add_argument("--decay", type=float, default=300000)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--resume", action="store_true")
    t.add_argument("--model", default=DEFAULT_MODEL)

    e = sub.add_parser("eval", help="比較各策略的長期平均")
    e.add_argument("--steps", type=int, default=4000)
    e.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    e.add_argument("--policies", default="greedy,heur,learned")
    e.add_argument("--seed", type=int, default=12345)
    e.add_argument("--model", default=DEFAULT_MODEL)

    a = sub.add_parser("advise", help="給盤面，輸出長期最佳動作")
    a.add_argument("file", nargs="?", help="盤面檔（不給則讀 --board 或 stdin）")
    a.add_argument("--board", help='例如 "12345/23451/34512/45123/51234"')
    a.add_argument("--samples", type=int, default=64, help="連鎖補珠的抽樣次數")
    a.add_argument("--top", type=int, default=8)
    a.add_argument("--verify", action="store_true", help="對前幾名做 rollout 驗證")
    a.add_argument("--verify-top", type=int, default=5)
    a.add_argument("--rollouts", type=int, default=200)
    a.add_argument("--horizon", type=int, default=20)
    a.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    a.add_argument("--seed", type=int, default=1)
    a.add_argument("--model", default=DEFAULT_MODEL)

    args = p.parse_args()
    {"train": train, "eval": evaluate, "advise": advise}[args.cmd](args)


if __name__ == "__main__":
    main()
