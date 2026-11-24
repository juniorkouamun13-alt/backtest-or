import numpy as np
import pandas as pd

LOT, OZ, SPREAD = 0.03, 100, 0.30
SIZE = LOT * OZ
TRAIN_FRAC = 0.6
WARM = 250
SLTP = [(1, 2), (1.5, 3), (2, 2), (2, 4), (1, 3)]


def ema(x, n):
    return pd.Series(x).ewm(span=n, adjust=False).mean().to_numpy()


def atr14(h, l, c):
    pc = np.roll(c, 1)
    pc[0] = c[0]
    tr = np.maximum(h - l, np.maximum(abs(h - pc), abs(l - pc)))
    return pd.Series(tr).ewm(alpha=1 / 14, adjust=False).mean().to_numpy()


def rsi(c, p):
    d = pd.Series(c).diff()
    up = d.clip(lower=0).ewm(alpha=1 / p, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / p, adjust=False).mean()
    return (100 - 100 / (1 + up / dn.replace(0, np.nan))).fillna(50).to_numpy()


def trend_filter(sig, c, on):
    if on:
        t = ema(c, 200)
        sig[(sig == 1) & (c < t)] = 0
        sig[(sig == -1) & (c > t)] = 0
    return sig


def sig_ema(c, f, s, trend):
    ef, es = ema(c, f), ema(c, s)
    sig = np.zeros(len(c))
    up = (ef[:-1] <= es[:-1]) & (ef[1:] > es[1:])
    dn = (ef[:-1] >= es[:-1]) & (ef[1:] < es[1:])
    sig[1:][up] = 1
    sig[1:][dn] = -1
    return trend_filter(sig, c, trend)


def sig_donchian(h, l, c, n, trend):
    hh = pd.Series(h).rolling(n).max().shift(1).to_numpy()
    ll = pd.Series(l).rolling(n).min().shift(1).to_numpy()
    sig = np.zeros(len(c))
    sig[c > hh] = 1
    sig[c < ll] = -1
    return trend_filter(sig, c, trend)


def sig_rsi(c, p, lo, hi, trend):
    r = rsi(c, p)
    sig = np.zeros(len(c))
    sig[r < lo] = 1
    sig[r > hi] = -1
    return trend_filter(sig, c, trend)


def backtest(o, h, l, atr, sig, slm, tpm, a, b):
    pnls, pos, entry, sl, tp = [], 0, 0.0, 0.0, 0.0
    for j in range(a, b):
        if pos == 0 and sig[j - 1] != 0 and atr[j - 1] > 0:
            pos = int(sig[j - 1])
            entry = o[j]
            d = atr[j - 1]
            sl, tp = entry - pos * slm * d, entry + pos * tpm * d
        if pos != 0:
            if pos == 1:
                hs, ht = l[j] <= sl, h[j] >= tp
            else:
                hs, ht = h[j] >= sl, l[j] <= tp
            if hs:
                pnls.append((sl - entry) * pos)
                pos = 0
            elif ht:
                pnls.append((tp - entry) * pos)
                pos = 0
    return np.array(pnls)


def stats(pnl_oz):
    if len(pnl_oz) == 0:
        return dict(n=0, pf=0.0, total=0.0, dd=0.0)
    p = (pnl_oz - SPREAD) * SIZE
    gw, gl = p[p > 0].sum(), -p[p < 0].sum()
    pf = min(gw / gl, 99.0) if gl > 0 else 99.0
    eq = np.concatenate([[0], np.cumsum(p)])
    dd = (np.maximum.accumulate(eq) - eq).max()
    return dict(n=len(p), pf=round(pf, 2), total=round(p.sum(), 2), dd=round(dd, 2))


def strategies(h, l, c):
    for f, s in [(10, 30), (20, 50), (50, 100), (20, 100)]:
        for t in (0, 1):
            yield f"EMA {f}/{s}" + (" +tendance" if t else ""), sig_ema(c, f, s, t)
    for n in (10, 20, 40, 55):
        for t in (0, 1):
            yield f"Cassure {n}" + (" +tendance" if t else ""), sig_donchian(h, l, c, n, t)
    for p, lo, hi in [(14, 30, 70), (14, 25, 75), (7, 20, 80)]:
        for t in (0, 1):
            yield f"RSI {p} ({lo}/{hi})" + (" +tendance" if t else ""), sig_rsi(c, p, lo, hi, t)


def evaluate(df, tf):
    o, h, l, c = (df[k].to_numpy(float) for k in ("Open", "High", "Low", "Close"))
    atr = atr14(h, l, c)
    n = len(df)
    split = int(WARM + (n - WARM) * TRAIN_FRAC)
    rows = []
    for name, sig in strategies(h, l, c):
        for slm, tpm in SLTP:
            tr = stats(backtest(o, h, l, atr, sig, slm, tpm, WARM, split))
            te = stats(backtest(o, h, l, atr, sig, slm, tpm, split, n))
            if tr["n"] < 25 or te["n"] < 10:
                continue
            rows.append(dict(tf=tf, strategie=name, sl_atr=slm, tp_atr=tpm,
                             tr_n=tr["n"], tr_pf=tr["pf"], tr_gain=tr["total"],
                             te_n=te["n"], te_pf=te["pf"], te_gain=te["total"],
                             te_dd=te["dd"], score=min(tr["pf"], te["pf"])))
    return rows


def ohlc(df, rule):
    return df.resample(rule).agg({"Open": "first", "High": "max", "Low": "min",
                                  "Close": "last"}).dropna()


def report(rows):
    res = pd.DataFrame(rows).sort_values("score", ascending=False)
    lines = [f"{len(res)} combinaisons testees (entrainement 60 % / test 40 %)", ""]
    for _, r in res.head(10).iterrows():
        lines.append(f"{r.tf} | {r.strategie} | SL {r.sl_atr}xATR TP {r.tp_atr}xATR\n"
                     f"   entrainement: {r.tr_n} trades, PF {r.tr_pf}, {r.tr_gain:+.0f}$\n"
                     f"   TEST (jamais vu): {r.te_n} trades, PF {r.te_pf}, "
                     f"{r.te_gain:+.0f}$, pire baisse {r.te_dd:.0f}$")
    best = res.iloc[0]["score"] if len(res) else 0
    lines.append("")
    if best >= 1.2:
        lines.append(f"VERDICT : meilleure candidate a score {best}. A tester en DEMO "
                     "plusieurs semaines, pas avec de l'argent reel.")
    else:
        lines.append(f"VERDICT : meilleur score {best}. Aucune strategie assez fiable "
                     "(il faut PF > 1.2 a la fois en entrainement ET en test).")
    return res, "\n".join(lines)


if __name__ == "__main__":
    import yfinance as yf

    def get(interval, period):
        d = yf.download("GC=F", interval=interval, period=period,
                        auto_adjust=True, progress=False)
        if isinstance(d.columns, pd.MultiIndex):
            d.columns = d.columns.get_level_values(0)
        return d.dropna()

    h1, d1 = get("1h", "730d"), get("1d", "10y")
    rows = []
    for tf, df in [("1h", h1), ("4h", ohlc(h1, "4h")), ("1j", d1)]:
        print(tf, len(df), "bougies")
        rows += evaluate(df, tf)
    res, text = report(rows)
    print(text)
    res.to_csv("tous_les_resultats.csv", index=False)
    open("resultats_optim.txt", "w", encoding="utf-8").write(text)
