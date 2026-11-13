# TOS-ElementalChain-solver

《神魔之塔》(Tower of Saviors) 風格 5×5「刪珠 → 天降 → 連鎖」小遊戲的最佳化助手。
目標不是「這一步消最多」，而是**最大化長期平均每步消除的珠數**。

## 規則

- 盤面 5×5、5 種屬性（水火木光暗），補珠（天降）各屬性機率預設相等。
- 每步刪除一格 → 上方珠子掉落 → 頂端補隨機珠。
- 橫／直連 ≥3 同屬性即消除（L、T、+ 自然合併），連鎖持續到盤面穩定。
- 報酬 = 該步連鎖消掉的總珠數（被刪的那格不計）。

## 檔案

| 檔案 | 說明 |
| --- | --- |
| `tos_solver.py` | 遊戲引擎（消除、掉落、連鎖）＋ 啟發式 expectimax 基準策略 |
| `tos_advisor.py` | 長期期望值顧問：用 n-tuple network + 差分 TD 學習盤面價值 V(s)，推薦最佳刪除位置，並可 rollout 驗證 |
| `tos_refill.py` | 天降機率分析：從實際遊玩紀錄用 EM 估計各屬性天降機率，含 bootstrap 信賴區間與均等性檢定 |
| `tos_web.py` + `web/index.html` | FastAPI 網頁介面（建議熱力圖、記錄對局、統計） |
| `tos_value.json` | 已訓練好的價值函數權重 |
| `refill_log.txt` | 實際遊玩紀錄（天降分析的資料，格式說明見檔頭） |
| `board.txt`、`example_board.txt` | 盤面輸入範例 |

## 安裝

需要 Python ≥ 3.14 與 [uv](https://docs.astral.sh/uv/)：

```bash
uv sync
```

（引擎、顧問、分析腳本只用標準庫；僅網頁介面需要 fastapi / uvicorn。）

## 使用

### 1. 網頁介面

```bash
uv run tos_web.py                  # http://127.0.0.1:8000
uv run tos_web.py --host 0.0.0.0   # 同個 Wi-Fi 的手機也能連
```

- **建議**：輸入盤面 → 長期期望值熱力圖、排名、rollout 驗證
- **記錄**：選刪除位置、填消除數（選填）、填結果盤面 → 檢查後寫入紀錄檔
- **統計**：每步消除數統計、天降機率分析

### 2. 命令列顧問

```bash
uv run tos_advisor.py train --minutes 10      # 訓練價值函數（存到 tos_value.json，--resume 可續訓）
uv run tos_advisor.py eval --steps 3000       # 與 greedy / 啟發式策略比較長期平均
uv run tos_advisor.py advise example_board.txt --verify
uv run tos_advisor.py advise --board "12345/23451/34512/45123/51234"
```

盤面格式：5 行、每行 5 個符號（任意字元皆可，如 `水火木光暗`、`RGBYP`、`01234`），
不同符號代表不同屬性。

做法簡述：把問題視為平均報酬 MDP，`Q(s,a) = E[r(s,a) − ρ + V(s')]`，
`Q(s,a1) − Q(s,a2)` 即「選 a1 比選 a2 長期多消幾顆」。
`--verify` 會對前幾名動作做蒙地卡羅 rollout，直接驗證長期總消除數。

### 3. 天降機率分析

```bash
uv run tos_refill.py analyze refill_log.txt            # EM 估計各屬性天降機率 + 信賴區間 + 均等性檢定
uv run tos_refill.py stats refill_log.txt --detail     # 每步消除數統計
uv run tos_refill.py simulate --steps 300 --probs 0.3,0.2,0.2,0.15,0.15 -o sample_log.txt   # 產生假資料驗證
```

不能直接數「新出現的珠」：連鎖中掉下又馬上被消掉的珠看不到，且「沒被消掉」本身與顏色有關。
因此對每一步列舉所有可能的天降序列，建立概似函數後以 EM 求最大概似估計。
若某步不可能發生，會指出紀錄檔的行號（可能打錯）。

紀錄檔格式（`#` 後為註解）：

```
屬性: 水火木光暗        # 第一行列出所有屬性符號

水水木火暗              # 一局的起始盤面（5 行）
...
> 2,4 消 7              # 刪除位置「列,行」（左上=1,1）；「消 N」選填
水水木光暗              # 刪除後完全穩定的盤面（同時是下一步的刪除前盤面）
...
---                     # 換局或中斷：下一個盤面為新的起始盤面
```

## 基準策略

```bash
uv run tos_solver.py     # 比較 greedy、depth-1、depth-2 expectimax 的長期平均
```
