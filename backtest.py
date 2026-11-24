import numpy as np
import pandas as pd

SYMBOL = "GC=F"
INTERVAL = "1h"
PERIOD = "730d"
FAST, SLOW = 10, 30
SL, TP = 5.0, 10.0
LOT = 0.03
OZ_PER_LOT = 100
SPREAD = 0.30
START_BALANCE = 1000.0
MAX_DAILY_LOSS = 20.0


def run_backtest(df):
    o, h, l, c = (df[k].to_numpy(float) for k in ("Open", "High", "Low", "Close"))
    f = pd.Series(c).ewm(span=FAST, adjust=False).mean().to_numpy()
    s = pd.Series(c).ewm(span=SLOW, adjust=False).mean().to_numpy()
    days = df.index.date
    size = LOT * OZ_PER_LOT

    trades, pos = [], None
    day, day_pnl, signal = None, 0.0, 0

    for j in range(SLOW + 1, len(df)):
        if days[j] != day:
            day, day_pnl = days[j], 0.0

        if pos is None and signal != 0 and day_pnl > -MAX_DAILY_LOSS:
            entry = o[j] + signal * SPREAD / 2
            pos = dict(side=signal, entry=entry,
                       sl=entry - signal * SL, tp=entry + signal * TP,
                       time=df.index[j])

        if pos is not None:
            side = pos["side"]
            hit_sl = l[j] <= pos["sl"] if side == 1 else h[j] >= pos["sl"]
            hit_tp = h[j] >= pos["tp"] if side == 1 else l[j] <= pos["tp"]
            exit_price = pos["sl"] if hit_sl else (pos["tp"] if hit_tp else None)
            if exit_price is not None:
                pnl = (exit_price - pos["entry"]) * side * size
                trades.append(dict(ouverture=pos["time"], fermeture=df.index[j],
                                   sens="ACHAT" if side == 1 else "VENTE",
                                   entree=round(pos["entry"], 2),
                                   sortie=round(exit_price, 2),
                                   gain=round(pnl, 2)))
                day_pnl += pnl
                pos = None

        signal = 0
        if f[j - 1] <= s[j - 1] and f[j] > s[j]:
            signal = 1
        elif f[j - 1] >= s[j - 1] and f[j] < s[j]:
            signal = -1

    return pd.DataFrame(trades)


def summary(trades):
    if trades.empty:
        return "Aucun trade sur la période."
    equity = START_BALANCE + trades["gain"].cumsum()
    drawdown = (equity.cummax() - equity).max()
    wins = (trades["gain"] > 0).sum()
    gross_win = trades.loc[trades.gain > 0, "gain"].sum()
    gross_loss = -trades.loc[trades.gain < 0, "gain"].sum()
    pf = gross_win / gross_loss if gross_loss else float("inf")
    return "\n".join([
        f"Trades : {len(trades)}",
        f"Trades gagnants : {wins} ({100 * wins / len(trades):.0f} %)",
        f"Résultat total : {trades['gain'].sum():+.2f} $",
        f"Solde final (départ {START_BALANCE:.0f} $) : {equity.iloc[-1]:.2f} $",
        f"Pire baisse (drawdown) : {drawdown:.2f} $",
        f"Profit factor : {pf:.2f}  (>1 = gagnant, <1 = perdant)",
    ])


if __name__ == "__main__":
    import yfinance as yf
    data = yf.download(SYMBOL, interval=INTERVAL, period=PERIOD,
                       auto_adjust=True, progress=False)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    data = data.dropna()
    print(f"{len(data)} bougies de {data.index[0]} à {data.index[-1]}")
    result = run_backtest(data)
    text = summary(result)
    print(text)
    result.to_csv("trades.csv", index=False)
    with open("resultats.txt", "w", encoding="utf-8") as fh:
        fh.write(text)
