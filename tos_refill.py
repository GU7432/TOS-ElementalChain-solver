"""
天降（補珠）機率分析
=====================
用你實際遊玩的紀錄（刪除前盤面、刪除位置、穩定後盤面）估計每種屬性的天降機率。

為什麼不能直接數「新出現的珠」：
  連鎖時，有些天降珠掉下來馬上又被消掉，你根本看不到它的顏色；
  而且「沒被消掉」本身就跟顏色有關（容易湊成三連的顏色更常被吃掉），
  直接數會有偏差。

做法：對每一步，列舉所有「能從刪除前盤面變成你記錄的結果盤面」的天降序列
  （看得到的珠顏色固定，看不到的珠所有可能都列舉），
  得到該步的概似函數 P(結果 | p)，再用 EM 求整體最大概似估計 p̂，
  並用 bootstrap 給信賴區間、用概似比檢定判斷「是否為均等」。
  順便檢查：天降是否偏向「故意製造/避免」消除（跟機率模型的預測比較）。
  如果某一步「不可能」發生，代表紀錄可能打錯了，會列出行號。

紀錄格式（純文字，# 之後是註解）：
  屬性: 水火木光暗             ← 第一個非註解行，列出全部屬性符號（順序就是報告順序）

  水水木火暗                   ← 一局的起始盤面（5 行）
  水水火水暗
  光火火暗火
  光水光木木
  暗木光暗木
  > 2,4                        ← 刪除位置（列,行），左上為 1,1
                                 可選：「> 2,4 消 7」= 這一步（含連鎖）共消 7 顆，不含刪掉的那顆
  水水木?暗                    ← 刪除後「完全穩定」的盤面（同時是下一步的刪除前盤面）
  ...
  > 3,4
  ...（一直接下去）

  ---                          ← 換一局 / 中斷後重新開始：下一個盤面是新的起始盤面

用法：
  python3 tos_refill.py analyze refill_log.txt
  python3 tos_refill.py stats refill_log.txt [--detail]                # 統計每步消了幾顆
  python3 tos_refill.py simulate --steps 300 --probs 0.3,0.2,0.2,0.15,0.15 -o sample_log.txt   # 產生假資料驗證
"""
import argparse
import re
import math
import random
import sys
from itertools import product

from tos_solver import N, find_matches

CELLS = N * N


# =====================================================================
# 解析紀錄檔
# =====================================================================
class LogError(Exception):
    pass


def parse_log(path):
    """回傳 (symbols, steps)；steps = [(before, action, after, line_no, game_no, 消除數或None)]"""
    symbols = None
    steps = []
    board_lines, board_start = [], 0
    prev = None          # 上一個完整盤面
    game = 1
    pending = None       # (action, line_no, 消除數或 None) 等待結果盤面

    def flush_board(lineno):
        nonlocal board_lines, prev, pending
        rows = board_lines
        board_lines = []
        b = []
        for r in rows:
            for ch in r:
                if ch not in sym_idx:
                    raise LogError(f"第 {lineno} 行附近：符號「{ch}」不在屬性列表 {symbols} 中")
                b.append(sym_idx[ch])
        if pending is not None:
            a, aline, elim = pending
            steps.append((prev, a, b, aline, game, elim))
            pending = None
        prev = b

    with open(path, encoding="utf-8") as f:
        lines = f.readlines()
    for i, raw in enumerate(lines, 1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        if symbols is None:
            if not (line.startswith("屬性") and (":" in line or "：" in line)):
                raise LogError(f"第 {i} 行：第一行必須是「屬性: 水火木光暗」這種格式")
            symbols = list(line.replace("：", ":").split(":", 1)[1].replace(" ", ""))
            if len(set(symbols)) != len(symbols) or len(symbols) < 2:
                raise LogError(f"第 {i} 行：屬性符號重複或太少")
            sym_idx = {s: k for k, s in enumerate(symbols)}
            continue
        if line.startswith("---"):
            if board_lines:
                raise LogError(f"第 {i} 行：盤面行數不足 {N} 行就遇到 ---")
            if pending is not None:
                raise LogError(f"第 {pending[1]} 行：刪除位置之後沒有結果盤面")
            if prev is not None:
                game += 1
            prev = None
            continue
        if line.startswith(">"):
            if board_lines:
                raise LogError(f"第 {i} 行：盤面行數不足 {N} 行就遇到刪除位置")
            if prev is None:
                raise LogError(f"第 {i} 行：刪除位置之前沒有盤面")
            if pending is not None:
                raise LogError(f"第 {i} 行：連續兩個刪除位置，中間缺結果盤面")
            nums = [int(x) for x in re.findall(r"\d+", line)]
            if len(nums) not in (2, 3) or not all(1 <= x <= N for x in nums[:2]):
                raise LogError(f"第 {i} 行：刪除位置格式應為「> 列,行」或「> 列,行 消 顆數」（列、行 1~{N}）")
            elim = nums[2] if len(nums) == 3 else None
            if elim is not None and elim != 0 and elim < 3:
                raise LogError(f"第 {i} 行：消除數 {elim} 不合理（只能是 0 或 3 以上）")
            pending = ((nums[0] - 1) * N + (nums[1] - 1), i, elim)
            continue
        row = line.replace(" ", "").replace("\t", "")
        if len(row) != N:
            raise LogError(f"第 {i} 行：盤面每行要 {N} 個符號，收到「{row}」")
        if not board_lines:
            board_start = i
        board_lines.append(row)
        if len(board_lines) == N:
            flush_board(board_start)
    if board_lines:
        raise LogError("檔案結尾有不完整的盤面")
    if pending is not None:
        raise LogError(f"第 {pending[1]} 行：刪除位置之後沒有結果盤面")
    if symbols is None:
        raise LogError("檔案是空的")
    return symbols, steps


# =====================================================================
# 單步概似：列舉所有一致的天降序列
#   回傳 {各色天降數 tuple: 路徑數}；P(結果|p) = Σ 路徑數 × Π p_k^n_k
# =====================================================================
class TooComplex(Exception):
    """列舉量超過上限（只取決於刪除前盤面與位置，與天降結果無關 → 略過不造成偏差）"""


class Truncated(Exception):
    """在隱藏珠上限內找不到任何路徑：可能抄錯，也可能是極長連鎖"""


# 未知珠用負數表示（每格不同，find_matches 不會把它們當同色）
def _renumber(b):
    for i in range(CELLS):
        if b[i] < 0:
            b[i] = -1 - i


def _potential(b, u):
    """未知格 u 是否「可能」湊成三連：某個含 u 的三格窗口裡，已知的珠全部同色"""
    r, c = divmod(u, N)
    wins = []
    for s in range(max(0, c - 2), min(c, N - 3) + 1):
        wins.append([r * N + s + j for j in range(3)])
    for s in range(max(0, r - 2), min(r, N - 3) + 1):
        wins.append([(s + j) * N + c for j in range(3)])
    for w in wins:
        known = {b[i] for i in w if b[i] >= 0}
        if len(known) <= 1:
            return True
    return False


MAX_HIDDEN = 2   # 每步最多考慮幾顆「天降後又被連鎖吃掉、看不到」的珠；更多的機率極低


def step_likelihood(before, a, after, C, limit=300000, max_hidden=None, elim=None, info=None):
    """
    延遲決定：新珠只有在「可能湊成三連」時才展開顏色分支，
    其餘維持未知；到盤面穩定時直接從結果盤面讀出顏色。
    """
    if max_hidden is None:
        # 有消除數時，不符合的路徑會被提早剪掉，可以放寬隱藏珠上限
        max_hidden = MAX_HIDDEN if elim is None else MAX_HIDDEN + 3
    b = before[:]
    new = [False] * CELLS
    c0, r0 = a % N, a // N
    for rr in range(r0, 0, -1):
        b[rr * N + c0] = b[(rr - 1) * N + c0]
    b[c0] = -1
    new[c0] = True
    _renumber(b)
    leaves = {}
    nodes = [0]
    truncated = [False]

    def rec(b, new, counts, hidden, removed=0):
        nodes[0] += 1
        if nodes[0] > limit:
            raise TooComplex
        for u in range(CELLS):
            if b[u] < 0 and _potential(b, u):
                for k in range(C):
                    b2 = b[:]
                    b2[u] = k
                    cnt = list(counts)
                    cnt[k] += 1
                    rec(b2, new, cnt, hidden, removed)
                return
        m = find_matches(b)
        if not m:
            if elim is not None and removed != elim:
                return
            cnt = list(counts)
            for i in range(CELLS):
                if b[i] < 0:
                    cnt[after[i]] += 1
                elif b[i] != after[i]:
                    return
            t = tuple(cnt)
            leaves[t] = leaves.get(t, 0) + 1
            return
        removed += len(m)
        if elim is not None and removed > elim:
            return
        hidden += sum(1 for i in m if new[i])
        if hidden > max_hidden:
            truncated[0] = True
            return
        nb, nn = b[:], new[:]
        for c in range(N):
            keep = [r * N + c for r in range(N) if (r * N + c) not in m]
            k = N - len(keep)
            vals = [-1] * k + [b[i] for i in keep]
            flags = [True] * k + [new[i] for i in keep]
            for r in range(N):
                nb[r * N + c] = vals[r]
                nn[r * N + c] = flags[r]
        _renumber(nb)
        rec(nb, nn, counts, hidden, removed)

    rec(b, new, [0] * C, 0)
    if info is not None:
        info["truncated"] = truncated[0]
    if not leaves and truncated[0]:
        raise Truncated
    return leaves


def first_round_info(before, a, after, C):
    """
    只看「刪除那一欄頂端第一顆天降」：
      pred[k] = 天降顏色為 k 時，第一輪是否就會發生消除
      obs     = 實際上是否發生了消除（結果盤面 ≠ 單純下移+補一顆）
    """
    b = before[:]
    c0, r0 = a % N, a // N
    for rr in range(r0, 0, -1):
        b[rr * N + c0] = b[(rr - 1) * N + c0]
    pred = []
    for k in range(C):
        b[c0] = k
        pred.append(bool(find_matches(b)))
    b[c0] = after[c0]
    obs = not (b == after and not find_matches(b))
    return pred, obs



# =====================================================================
# 逐步處理：每一步的概似（或標記為略過 / 抄錯）
# =====================================================================
def process_steps(steps, C, limit):
    recs = []
    for before, a, after, line, game, elim in steps:
        rec = {"before": before, "a": a, "after": after, "line": line, "game": game, "elim": elim,
               "status": "ok", "leaves": None, "why": ""}
        recs.append(rec)
        if find_matches(after):
            rec["status"], rec["why"] = "bad", "結果盤面裡還有三連（不是穩定盤面？）"
            continue
        leaves = None
        for lim, mh in [(limit, MAX_HIDDEN)]:
            try:
                leaves = step_likelihood(before, a, after, C, lim, max_hidden=None if elim is not None else mh, elim=elim)
                break
            except TooComplex:
                continue
            except Truncated:
                try:   # 放寬隱藏珠上限再試一次
                    leaves = step_likelihood(before, a, after, C, lim * 3, max_hidden=MAX_HIDDEN + 5, elim=elim)
                except (TooComplex, Truncated):
                    rec["status"], rec["why"] = "bad", "疑似抄錯（或是非常長的連鎖，無法確認）"
                break
        if rec["status"] == "bad":
            continue
        if leaves is None:
            rec["status"] = "skip"
        elif not leaves:
            rec["status"], rec["why"] = "bad", ("不可能從刪除前盤面得到這個結果（可能抄錯盤面或位置）" if elim is None
                                                else f"盤面變化和「消 {elim} 顆」對不上（可能抄錯盤面、位置或消除數）")
        else:
            rec["leaves"] = list(leaves.items())
            if elim is not None:   # 檢查消除數是否「可能但很不尋常」
                try:
                    info = {}
                    free = step_likelihood(before, a, after, C, limit, max_hidden=MAX_HIDDEN + 3, info=info)
                    if free:
                        q = elim_dist(list(free.items()), [1.0 / C] * C).get(elim, 0.0)
                        # q=0 且有截斷：可能是超過上限的長連鎖，無法判斷 → 不提醒
                        if (0 < q < 0.01) or (q == 0 and not info["truncated"]):
                            likely = max(elim_dist(list(free.items()), [1.0 / C] * C).items(), key=lambda x: x[1])[0]
                            rec["warn"] = f"寫的是消 {elim}，但從盤面看幾乎一定是消 {likely}（請確認）"
                except (TooComplex, Truncated):
                    pass
    return recs


def problems_of(recs):
    return {
        "warn": [{"line": r["line"], "msg": r["warn"]} for r in recs if r.get("warn")],
        "bad": [{"line": r["line"], "msg": r["why"]} for r in recs if r["status"] == "bad"],
        "skipped": sum(1 for r in recs if r["status"] == "skip"),
    }


def report_problems(pr):
    if pr["warn"]:
        print(f"⚠ 有 {len(pr['warn'])} 步的消除數很不尋常（仍有採用），請確認：")
        for w in pr["warn"][:20]:
            print(f"   第 {w['line']} 行：{w['msg']}")
        print()
    if pr["bad"]:
        print(f"⚠ 有 {len(pr['bad'])} 步和規則不一致，已排除，請檢查：")
        for b in pr["bad"][:20]:
            print(f"   第 {b['line']} 行那一步（檢查它上方的刪除前盤面、刪除位置、下方的結果盤面）：{b['msg']}")
        if len(pr["bad"]) > 20:
            print(f"   ...另外 {len(pr['bad']) - 20} 筆")
        print()
    if pr["skipped"]:
        print(f"（{pr['skipped']} 步連鎖太長、可能情況太多，略過；可用 --limit 調高）\n")


def old_removed(before, a, after):
    """
    下限：每一欄最後 = 底部存活的舊珠（順序不變）+ 上方新珠，
    找出最多能存活幾顆舊珠 → 至少消了幾顆舊珠（被吃掉的新珠看不到，不算）
    """
    c0, r0 = a % N, a // N
    elim = 0
    for c in range(N):
        old = [before[r * N + c] for r in range(N) if not (c == c0 and r == r0)][::-1]
        fin = [after[r * N + c] for r in range(N)][::-1]
        s = 0
        for x in old:
            if s < len(fin) and x == fin[s]:
                s += 1
        elim += len(old) - s
    return elim


def elim_dist(leaves, p):
    """消除數的後驗分布 {消除數: 機率}"""
    lp = [math.log(max(x, 1e-300)) for x in p]
    terms = [(math.log(m) + sum(n * l for n, l in zip(cnt, lp)), sum(cnt) - 1) for cnt, m in leaves]
    mx = max(t for t, _ in terms)
    d = {}
    for t, e in terms:
        d[e] = d.get(e, 0.0) + math.exp(t - mx)
    z = sum(d.values())
    return {e: w / z for e, w in d.items()}


def elim_posterior(leaves, p):
    """消除數 = 天降數 - 1。回傳 (期望值, 最小, 最大)；有隱藏珠時用估計機率 p 加權"""
    lp = [math.log(max(x, 1e-300)) for x in p]
    terms = [(math.log(m) + sum(n * l for n, l in zip(cnt, lp)), sum(cnt) - 1) for cnt, m in leaves]
    mx = max(t for t, _ in terms)
    ws = [(math.exp(t - mx), e) for t, e in terms]
    z = sum(w for w, _ in ws)
    es = [e for _, e in ws]
    return sum(w * e for w, e in ws) / z, min(es), max(es)


# 在「5 屬性、均等天降」模型下，各策略的長期平均（tos_advisor.py eval 的結果）
BENCH = [("隨機亂刪", 0.86), ("只看這一步最多（貪婪）", 3.68), ("tos_advisor 建議", 4.83)]


def step_text(r):
    if r["status"] != "ok":
        return "略過（太複雜）" if r["status"] == "skip" else "⚠ " + r["why"]
    if r["kind"] == "bound":
        return f"至少 {r['lo']}，估計約 {r['e']:.0f}（大連鎖）"
    if r["lo"] == r["hi"]:
        return f"{r['lo']}"
    mode = max(r["dist"], key=r["dist"].get)
    others = "、".join(f"{e}（{q:.0%}）" for e, q in sorted(r["dist"].items()) if e != mode and q >= 0.005)
    return f"{mode}（{r['dist'][mode]:.0%}）" + (f"，也可能 {others}" if others else "")


STAT_BINS = [(0, 0, "0 顆"), (1, 3, "3 顆"), (4, 5, "4-5 顆"), (6, 8, "6-8 顆"), (9, 14, "9-14 顆"), (15, 999, "15 顆以上")]


def compute_stats(symbols, steps, limit=100000):
    """每步消除數統計；回傳 dict（網頁與命令列共用）"""
    C = len(symbols)
    recs = process_steps(steps, C, limit)
    ok = [r for r in recs if r["status"] == "ok"]
    p = em([r["leaves"] for r in ok], C)[0] if len(ok) >= 30 else [1.0 / C] * C
    for r in ok:
        r["e"], r["lo"], r["hi"] = elim_posterior(r["leaves"], p)
        r["kind"] = "exact" if r["lo"] == r["hi"] else "post"
        r["dist"] = elim_dist(r["leaves"], p)
    # 太複雜的大連鎖：不能略過（會嚴重低估），改用近似
    for r in recs:
        if r["status"] != "skip":
            continue
        r["status"] = "ok"
        if r["elim"] is not None:
            r["e"] = r["lo"] = r["hi"] = r["elim"]
            r["kind"] = "exact"
            continue
        try:
            L = step_likelihood(r["before"], r["a"], r["after"], C, limit * 5, max_hidden=1)
            r["e"], r["lo"], r["hi"] = elim_posterior(list(L.items()), p)
            r["dist"] = elim_dist(list(L.items()), p)
            r["kind"] = "post"
            continue
        except Truncated:
            extra, factor = 2, 1.15      # 至少還有 2 顆看不到的新珠被吃掉
        except TooComplex:
            extra, factor = 0, 1.3
        lb = old_removed(r["before"], r["a"], r["after"]) + extra
        r["lo"], r["hi"], r["e"], r["kind"] = lb, None, lb * factor, "bound"

    out = {"n_records": len(recs), "problems": problems_of(recs), "ok": False,
           "bench": [{"name": a, "value": v} for a, v in BENCH],
           "steps": [{"line": r["line"], "game": r["game"], "pos": [r["a"] // N + 1, r["a"] % N + 1],
                      "given": r["elim"], "e": r.get("e"), "text": step_text(r)} for r in recs]}
    ok = [r for r in recs if r["status"] == "ok"]
    if not ok:
        return out
    n = len(ok)
    es = [r["e"] for r in ok]
    tot = sum(es)
    mu = tot / n
    sd = math.sqrt(sum((x - mu) ** 2 for x in es) / max(1, n - 1))
    nb = sum(1 for r in ok if r["kind"] == "bound")
    npost = sum(1 for r in ok if r["kind"] == "post")
    games = []
    for g in sorted({r["game"] for r in ok}):
        gs = [r["e"] for r in ok if r["game"] == g]
        games.append({"game": g, "n": len(gs), "total": sum(gs), "mean": sum(gs) / len(gs)})
    out.update(
        ok=True, n=n, n_exact=n - npost - nb, n_post=npost, n_bound=nb,
        total=tot, total_lo=sum(r["lo"] for r in ok) if nb else None,
        mean=mu, ci95=1.96 * sd / math.sqrt(n),
        has=sum(1 for r in ok if r["e"] >= 0.5), max=max(es),
        bins=[{"name": name, "k": sum(1 for r in ok if lo <= round(r["e"]) <= hi)} for lo, hi, name in STAT_BINS],
        games=games,
    )
    return out


def stats(args):
    try:
        symbols, steps = parse_log(args.file)
    except LogError as e:
        print("紀錄格式錯誤：", e)
        sys.exit(1)
    r = compute_stats(symbols, steps, args.limit)
    report_problems(r["problems"])
    if not r["ok"]:
        print("沒有可用的資料。")
        return
    n = r["n"]
    print(f"共 {r['n_records']} 步紀錄，可計算 {n} 步：")
    print(f"  {r['n_exact']} 步精確；{r['n_post']} 步有看不到的天降珠，用期望值估計", end="")
    print(f"；{r['n_bound']} 步是很長的連鎖，只能估下限再推估" if r["n_bound"] else "")
    print(f"\n  總消除        ≈ {r['total']:.1f} 顆", end="")
    print(f"（確定至少 {r['total_lo']:.0f} 顆）" if r["n_bound"] else "")
    print(f"  平均每步消除  ≈ {r['mean']:.2f} 顆   （95% 區間約 ±{r['ci95']:.2f}）")
    print(f"  有消除的步數  ≈ {r['has']} / {n}（{100 * r['has'] / n:.0f}%）")
    print(f"  單步最多      ≈ {r['max']:.0f} 顆")
    print("\n消除數分布：")
    for b in r["bins"]:
        print(f"  {b['name']:>8s} {b['k']:5d} 步  {'█' * round(40 * b['k'] / n)}")
    if len(r["games"]) > 1:
        print("\n各局：")
        print("   局   步數   總消除   平均每步")
        for g in r["games"]:
            print(f"  {g['game']:3d}  {g['n']:5d}  {g['total']:7.1f}   {g['mean']:6.2f}")
    print("\n對照（模擬：5 屬性、均等天降下的長期平均）：")
    for b in r["bench"]:
        print(f"  {b['name']:<22s} {b['value']:5.2f}")
    print(f"  {'你的紀錄':<22s} {r['mean']:5.2f}")
    if n < 100:
        print(f"  （只有 {n} 步，運氣影響很大，至少 100 步以上比較準）")
    if args.detail:
        print("\n逐步明細：")
        print("   行號   局   位置    消除")
        for x in r["steps"]:
            print(f"  {x['line']:5d}  {x['game']:3d}   ({x['pos'][0]},{x['pos'][1]})   {x['text']}")

# =====================================================================
# EM 求最大概似
# =====================================================================
def loglik(data, p):
    lp = [math.log(x) if x > 0 else -1e300 for x in p]
    tot = 0.0
    for leaves in data:
        terms = [math.log(m) + sum(n * l for n, l in zip(cnt, lp)) for cnt, m in leaves]
        mx = max(terms)
        tot += mx + math.log(sum(math.exp(t - mx) for t in terms))
    return tot


def em(data, C, p=None, iters=500, tol=1e-10):
    p = list(p) if p else [1.0 / C] * C
    for _ in range(iters):
        lp = [math.log(max(x, 1e-300)) for x in p]
        exp_cnt = [0.0] * C
        for leaves in data:
            if len(leaves) == 1:
                cnt = leaves[0][0]
                for k in range(C):
                    exp_cnt[k] += cnt[k]
                continue
            terms = [math.log(m) + sum(n * l for n, l in zip(cnt, lp)) for cnt, m in leaves]
            mx = max(terms)
            ws = [math.exp(t - mx) for t in terms]
            z = sum(ws)
            for (cnt, _), w in zip(leaves, ws):
                for k in range(C):
                    exp_cnt[k] += w / z * cnt[k]
        tot = sum(exp_cnt)
        newp = [x / tot for x in exp_cnt]
        if max(abs(x - y) for x, y in zip(newp, p)) < tol:
            return newp, exp_cnt
        p = newp
    return p, exp_cnt


def chi2_sf(x, df):
    """卡方分布右尾機率（正則化上不完全 gamma）"""
    if x <= 0:
        return 1.0
    a, x2 = df / 2.0, x / 2.0
    if x2 < a + 1:      # 級數
        s = term = 1.0 / a
        n = a
        for _ in range(1000):
            n += 1
            term *= x2 / n
            s += term
            if term < s * 1e-15:
                break
        return max(0.0, 1.0 - s * math.exp(-x2 + a * math.log(x2) - math.lgamma(a)))
    # 連分數
    b0 = x2 + 1 - a
    c, d = 1e300, 1.0 / b0
    h = d
    for i in range(1, 1000):
        an = -i * (i - a)
        b0 += 2
        d = an * d + b0
        d = 1e-300 if abs(d) < 1e-300 else d
        c = b0 + an / c
        c = 1e-300 if abs(c) < 1e-300 else c
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1) < 1e-15:
            break
    return math.exp(-x2 + a * math.log(x2) - math.lgamma(a)) * h


# =====================================================================
# 分析
# =====================================================================
def compute_analyze(symbols, steps, limit=100000, bootstrap=300, seed=0):
    """天降機率分析；回傳 dict（網頁與命令列共用）"""
    C = len(symbols)
    recs = process_steps(steps, C, limit)
    out = {"symbols": symbols, "n_steps": len(steps), "problems": problems_of(recs), "ok": False}
    data = [r["leaves"] for r in recs if r["status"] == "ok"]
    fr = [first_round_info(r["before"], r["a"], r["after"], C) for r in recs if r["status"] == "ok"]
    if not data:
        return out

    p, exp_cnt = em(data, C)
    n_eff = sum(exp_cnt)
    rng = random.Random(seed)
    boots = []
    for _ in range(bootstrap):
        sample = [data[rng.randrange(len(data))] for _ in range(len(data))]
        boots.append(em(sample, C, p=p, iters=60, tol=1e-7)[0])

    def ci(k):
        if not boots:
            return p[k], p[k]
        xs = sorted(b[k] for b in boots)
        return xs[int(0.025 * len(xs))], xs[min(len(xs) - 1, int(0.975 * len(xs)))]

    G = 2 * (loglik(data, p) - loglik(data, [1.0 / C] * C))
    pv = chi2_sf(G, C - 1)
    if pv < 0.01:
        verdict = "有明顯證據「不是」均等機率。"
    elif pv < 0.05:
        verdict = "有些證據不是均等，建議再多收集一些資料確認。"
    else:
        verdict = "資料與「均等機率」相容（目前看不出偏差）。"

    exp_m = var_m = 0.0
    obs_m = cnt = 0
    for pred, obs in fr:
        q = sum(pk for pk, m in zip(p, pred) if m)
        if 0 < q < 1:
            cnt += 1
            exp_m += q
            var_m += q * (1 - q)
            obs_m += obs
    board_test = {"cnt": cnt, "ok": cnt >= 10}
    if cnt >= 10:
        z = (obs_m - exp_m) / math.sqrt(var_m)
        if abs(z) > 2.6:
            bt = f"天降似乎會刻意{'製造' if z > 0 else '避免'}消除，不是單純獨立隨機；這會影響最佳策略。"
        else:
            bt = "與「獨立隨機、不看盤面」相容。"
        board_test.update(exp=exp_m, obs=obs_m, z=z, verdict=bt)

    out.update(
        ok=True,
        used=len(data),
        exact=sum(1 for d in data if len(d) == 1),
        total_draws=sum(sum(d[0][0]) for d in data if len(d) == 1),
        n_eff=n_eff,
        probs=[{"sym": symbols[k], "p": p[k], "lo": ci(k)[0], "hi": ci(k)[1], "count": exp_cnt[k]} for k in range(C)],
        uniform=1.0 / C,
        G=G, pvalue=pv, verdict=verdict,
        err95=1.96 * math.sqrt((1 / C) * (1 - 1 / C) / max(n_eff, 1)),
        need_draws=math.ceil((1.96 / 0.03) ** 2 * (1 / C) * (1 - 1 / C)),
        board_test=board_test,
    )
    return out


def analyze(args):
    try:
        symbols, steps = parse_log(args.file)
    except LogError as e:
        print("紀錄格式錯誤：", e)
        sys.exit(1)
    C = len(symbols)
    print(f"屬性 {C} 種：{' '.join(symbols)}；共 {len(steps)} 步紀錄\n")
    r = compute_analyze(symbols, steps, args.limit, args.bootstrap, args.seed)
    report_problems(r["problems"])
    if not r["ok"]:
        print("沒有可用的資料。")
        return
    print(f"可用 {r['used']} 步，其中 {r['exact']} 步的天降可以唯一確定（共 {r['total_draws']} 顆），"
          f"其餘步驟有被連鎖吃掉、看不到的天降珠，用 EM 推估。")
    print(f"天降總數（含推估）≈ {r['n_eff']:.1f} 顆\n")
    print("  屬性   估計機率     95% 信賴區間        天降數(含推估)   均等時應為")
    for x in r["probs"]:
        print(f"   {x['sym']}     {x['p']:6.3f}     [{x['lo']:5.3f}, {x['hi']:5.3f}]        {x['count']:8.1f}         {r['uniform']:.3f}")
    print(f"\n均等檢定（概似比 G={r['G']:.2f}, 自由度 {C - 1}）：p 值 = {r['pvalue']:.4f}")
    print(f"  → {r['verdict']}")
    print(f"  目前每個機率的誤差約 ±{r['err95']:.3f}（95%）；"
          f"想分辨 ±0.03 的偏差，大約需要 {r['need_draws']} 顆天降。")
    bt = r["board_test"]
    if bt["ok"]:
        print(f"\n天降是否「看盤面」：有 {bt['cnt']} 步的第一顆天降可能湊成消除，")
        print(f"  依估計機率預期會消 {bt['exp']:.1f} 次，實際 {bt['obs']} 次（z = {bt['z']:+.2f}）")
        print(f"  → {bt['verdict']}")
    else:
        print(f"\n（第一顆天降可能湊成消除的步數只有 {bt['cnt']} 步，資料不足以檢查天降是否看盤面）")


# =====================================================================
# 產生模擬紀錄（驗證分析程式用，也可當格式範例）
# =====================================================================
def simulate(args):
    probs = [float(x) for x in args.probs.split(",")]
    C = len(probs)
    symbols = list(args.symbols)[:C]
    if len(symbols) < C:
        sys.exit("--symbols 數量不足")
    rng = random.Random(args.seed)
    draw = lambda: rng.choices(range(C), probs)[0]

    def settle(b):
        tot = 0
        while True:
            m = find_matches(b)
            if not m:
                return tot
            tot += len(m)
            for c in range(N):
                col = [b[r * N + c] for r in range(N) if (r * N + c) not in m]
                col = [draw() for _ in range(N - len(col))] + col
                for r in range(N):
                    b[r * N + c] = col[r]

    def fmt(b):
        return "\n".join("".join(symbols[b[r * N + c]] for c in range(N)) for r in range(N))

    out = [f"# 模擬資料，真實機率 {probs}", f"屬性: {''.join(symbols)}", ""]
    b = None
    for i in range(args.steps):
        if i % args.game_len == 0:
            if b is not None:
                out.append("---")
            b = [draw() for _ in range(CELLS)]
            settle(b)
            out.append(fmt(b))
        a = rng.randrange(CELLS)
        c0, r0 = a % N, a // N
        for rr in range(r0, 0, -1):
            b[rr * N + c0] = b[(rr - 1) * N + c0]
        b[c0] = draw()
        k = settle(b)
        out.append(f"> {a // N + 1},{a % N + 1}" + (f" 消 {k}" if rng.random() < args.count_rate else ""))
        out.append(fmt(b) + (f"   # 真實消除 {k}" if args.annotate else ""))
    with open(args.output, "w", encoding="utf-8") as f:
        f.write("\n".join(out) + "\n")
    print(f"已寫出 {args.output}（{args.steps} 步）")


def main():
    p = argparse.ArgumentParser(description="天降機率分析")
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("analyze", help="分析紀錄檔")
    a.add_argument("file")
    a.add_argument("--bootstrap", type=int, default=300)
    a.add_argument("--limit", type=int, default=100000, help="單步列舉上限")
    a.add_argument("--seed", type=int, default=0)
    st = sub.add_parser("stats", help="統計每步消了幾顆")
    st.add_argument("file")
    st.add_argument("--detail", action="store_true", help="列出每一步")
    st.add_argument("--limit", type=int, default=100000)
    s = sub.add_parser("simulate", help="產生模擬紀錄")
    s.add_argument("--steps", type=int, default=300)
    s.add_argument("--probs", default="0.2,0.2,0.2,0.2,0.2")
    s.add_argument("--symbols", default="水火木光暗心")
    s.add_argument("--game-len", type=int, default=50)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--annotate", action="store_true", help="在結果盤面後註記真實消除數（測試用）")
    s.add_argument("--count-rate", type=float, default=0.0, help="多少比例的步驟在刪除位置後寫上消除數")
    s.add_argument("-o", "--output", default="sample_log.txt")
    args = p.parse_args()
    {"analyze": analyze, "stats": stats, "simulate": simulate}[args.cmd](args)


if __name__ == "__main__":
    main()
