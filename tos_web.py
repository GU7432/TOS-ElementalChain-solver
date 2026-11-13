"""
TOS 消珠助手 — 網頁介面（後端）
================================
啟動：
  uv run tos_web.py                    # 只有本機能連：http://127.0.0.1:8000
  uv run tos_web.py --host 0.0.0.0     # 同一個 Wi-Fi 的手機也能連（會印出網址）

功能：
  建議：輸入盤面 → 長期期望值熱力圖、排名、rollout 驗證
  記錄：選刪除位置、填消除數（選填）、填結果盤面 → 檢查後寫入紀錄檔
  統計：每步消除數統計、天降機率分析
"""
import argparse
import os
import random
import socket
import threading
from contextlib import asynccontextmanager
from multiprocessing import Pool

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import tos_advisor as adv
import tos_refill as rf
from tos_solver import N, find_matches

HERE = os.path.dirname(os.path.abspath(__file__))
WEB_DIR = os.path.join(HERE, "web")
DEFAULT_SYMBOLS = "水火木光暗"

@asynccontextmanager
async def lifespan(app):
    yield
    if "pool" in _pool:            # 關閉伺服器時一併結束 rollout 子程序
        _pool["pool"].terminate()
        _pool.pop("pool")


app = FastAPI(title="TOS 消珠助手", lifespan=lifespan)
CFG = {"log": os.path.join(HERE, "refill_log.txt"), "model": adv.DEFAULT_MODEL}
_lock = threading.Lock()
_net = {}
_pool = {}


def get_net():
    if "net" not in _net:
        if not os.path.exists(CFG["model"]):
            raise HTTPException(400, f"找不到模型檔 {CFG['model']}，請先執行 tos_advisor.py train")
        _net["net"], _net["rho"], _net["meta"] = adv.ValueNet.load(CFG["model"])
    return _net["net"], _net["rho"], _net["meta"]


def get_pool():
    if "pool" not in _pool:
        _pool["pool"] = Pool(max(1, (os.cpu_count() or 2) - 2))
    return _pool["pool"]


# =====================================================================
# 紀錄檔
# =====================================================================
def ensure_log():
    if not os.path.exists(CFG["log"]):
        with open(CFG["log"], "w", encoding="utf-8") as f:
            f.write(f"# 天降紀錄檔（由網頁介面建立）\n\n屬性: {DEFAULT_SYMBOLS}\n")


def read_log():
    ensure_log()
    with open(CFG["log"], encoding="utf-8") as f:
        return f.read()


def write_log(text):
    with open(CFG["log"], "w", encoding="utf-8") as f:
        f.write(text)


def log_state():
    """回傳 (symbols, steps, 目前盤面 rows 或 None, 錯誤訊息, 目前局號)"""
    text = read_log()
    try:
        symbols, steps = rf.parse_log(CFG["log"])
    except rf.LogError as e:
        return None, [], None, str(e), 1
    # 目前盤面 = 檔案中最後一個完整盤面（若最後是 --- 則沒有）
    cur, rows, game, seen = None, [], 1, False
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or line.startswith("屬性"):
            continue
        if line.startswith("---"):
            if seen:
                game += 1
            cur, rows, seen = None, [], False
            continue
        if line.startswith(">"):
            rows = []
            continue
        rows.append(line.replace(" ", "").replace("\t", ""))
        if len(rows) == N:
            cur, rows, seen = rows, [], True
    return symbols, steps, cur, None, game


def rows_to_board(rows, symbols):
    if len(rows) != N or any(len(r) != N for r in rows):
        raise HTTPException(400, f"盤面需要 {N}x{N}")
    idx = {s: k for k, s in enumerate(symbols)}
    b = []
    for r in rows:
        for ch in r:
            if ch not in idx:
                raise HTTPException(400, f"盤面有未填或不認得的格子（{ch}）")
            b.append(idx[ch])
    return b


# =====================================================================
# API
# =====================================================================
class BoardReq(BaseModel):
    rows: list[str]


class VerifyReq(BaseModel):
    rows: list[str]
    top: int = 5
    rollouts: int = 200
    horizon: int = 20


class StepReq(BaseModel):
    pos: list[int]               # [列, 行]，1 起算
    rows: list[str]              # 結果盤面
    elim: int | None = None
    force: bool = False


@app.get("/api/state")
def api_state():
    with _lock:
        symbols, steps, cur, err, game = log_state()
    recent = [{"line": s[3], "game": s[4], "pos": [s[1] // N + 1, s[1] % N + 1], "elim": s[5]} for s in steps[-8:]]
    return {"symbols": symbols or list(DEFAULT_SYMBOLS), "board": cur, "n_steps": len(steps),
            "game": game, "recent": recent[::-1], "error": err,
            "log": os.path.basename(CFG["log"])}


@app.post("/api/advise")
def api_advise(req: BoardReq):
    net, rho, meta = get_net()
    try:
        b, _ = adv.parse_board("\n".join(req.rows))
    except ValueError as e:
        raise HTTPException(400, str(e))
    if "?" in "".join(req.rows):
        raise HTTPException(400, "盤面還有未填的格子（?）")
    q, imm = adv.q_values(b, net, random.Random(1), chain_samples=64)
    best = max(q)
    order = sorted(range(N * N), key=lambda a: -q[a])
    greedy = max(range(N * N), key=lambda a: imm[a])
    return {
        "imm": imm, "adv": [x - best for x in q], "order": order,
        "greedy": greedy if imm[greedy] > imm[order[0]] + 1e-9 else None,
        "has_match": bool(find_matches(b)),
        "model": {"samples": meta.get("samples"), "rho": rho},
    }


@app.post("/api/verify")
def api_verify(req: VerifyReq):
    net, _, _ = get_net()
    b, _ = adv.parse_board("\n".join(req.rows))
    q, _ = adv.q_values(b, net, random.Random(1), chain_samples=64)
    cand = sorted(range(N * N), key=lambda a: -q[a])[: max(2, min(req.top, 10))]
    res = adv.rollout_verify(b, cand, CFG["model"], max(20, min(req.rollouts, 2000)),
                             max(1, min(req.horizon, 60)), pool=get_pool())
    return {"results": res, "rollouts": req.rollouts, "horizon": req.horizon}


@app.post("/api/step")
def api_step(req: StepReq):
    with _lock:
        symbols, steps, cur, err, _ = log_state()
        if err:
            raise HTTPException(400, "紀錄檔格式錯誤：" + err)
        if cur is None:
            raise HTTPException(400, "目前沒有盤面，請先「開新局」")
        r, c = req.pos
        if not (1 <= r <= N and 1 <= c <= N):
            raise HTTPException(400, "刪除位置錯誤")
        if req.elim is not None and req.elim != 0 and req.elim < 3:
            raise HTTPException(400, "消除數只能是 0 或 3 以上")
        before = rows_to_board(cur, symbols)
        after = rows_to_board(req.rows, symbols)
        a = (r - 1) * N + (c - 1)
        rec = rf.process_steps([(before, a, after, 0, 1, req.elim)], len(symbols), 100000)[0]
        if rec["status"] == "bad" and not req.force:
            return {"ok": False, "error": rec["why"]}
        text = read_log()
        if not text.endswith("\n"):
            text += "\n"
        text += f"> {r},{c}" + (f" 消 {req.elim}" if req.elim is not None else "") + "\n"
        text += "\n".join(req.rows) + "\n"
        write_log(text)
        return {"ok": True, "warn": rec.get("warn"), "skipped": rec["status"] == "skip",
                "forced": rec["status"] == "bad"}


@app.post("/api/newgame")
def api_newgame(req: BoardReq):
    with _lock:
        symbols, steps, cur, err, _ = log_state()
        if err:
            raise HTTPException(400, "紀錄檔格式錯誤：" + err)
        b = rows_to_board(req.rows, symbols)
        if find_matches(b):
            raise HTTPException(400, "起始盤面裡有三連，請確認盤面已經靜止")
        text = read_log()
        if not text.endswith("\n"):
            text += "\n"
        text += ("---\n" if cur is not None else "\n") + "\n".join(req.rows) + "\n"
        write_log(text)
    return {"ok": True}


@app.post("/api/undo")
def api_undo():
    """刪掉最後一步（或最後一個起始盤面）"""
    with _lock:
        lines = read_log().split("\n")
        toks = []
        for i, raw in enumerate(lines):
            line = raw.split("#", 1)[0].strip()
            if line and not line.startswith("屬性"):
                toks.append((i, line))
        if len(toks) < N:
            raise HTTPException(400, "沒有可以復原的紀錄")
        cut_tok = len(toks) - N
        if cut_tok > 0 and (toks[cut_tok - 1][1].startswith(">") or toks[cut_tok - 1][1].startswith("---")):
            cut_tok -= 1
            what = "最後一步" if toks[cut_tok][1].startswith(">") else "最後一局的起始盤面"
        else:
            what = "起始盤面"
        cut = toks[cut_tok][0]
        new = "\n".join(lines[:cut]).rstrip("\n") + "\n"
        write_log(new)
    return {"ok": True, "removed": what}


@app.get("/api/stats")
def api_stats():
    with _lock:
        symbols, steps, _, err, _ = log_state()
    if err:
        raise HTTPException(400, "紀錄檔格式錯誤：" + err)
    return rf.compute_stats(symbols, steps)


@app.get("/api/analyze")
def api_analyze():
    with _lock:
        symbols, steps, _, err, _ = log_state()
    if err:
        raise HTTPException(400, "紀錄檔格式錯誤：" + err)
    return rf.compute_analyze(symbols, steps, bootstrap=200)


@app.get("/api/lograw")
def api_lograw():
    return {"text": read_log()}


@app.get("/")
def index():
    return FileResponse(os.path.join(WEB_DIR, "index.html"))


app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")


def lan_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("10.255.255.255", 1))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except OSError:
        return None


def main():
    import uvicorn
    p = argparse.ArgumentParser(description="TOS 消珠助手網頁介面")
    p.add_argument("--host", default="127.0.0.1", help="0.0.0.0 = 讓同網路的手機也能連")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--log", default=CFG["log"], help="紀錄檔路徑")
    p.add_argument("--model", default=CFG["model"])
    args = p.parse_args()
    CFG["log"], CFG["model"] = os.path.abspath(args.log), os.path.abspath(args.model)
    print(f"紀錄檔：{CFG['log']}")
    print(f"電腦開啟：http://127.0.0.1:{args.port}")
    if args.host == "0.0.0.0" and lan_ip():
        print(f"手機開啟（同一個 Wi-Fi）：http://{lan_ip()}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
