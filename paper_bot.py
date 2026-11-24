import csv
import json
import os

import pandas as pd

from optimize import atr14, ohlc, sig_ema

LOT_OZ = 1.0
SPREAD = 0.30
SLM, TPM = 2.0, 4.0
START = 1000.0
STATE, LOG = "paper_state.json", "trades_log.csv"
HEAD = ["ouverture", "fermeture", "sens", "entree", "sortie", "gain_usd"]


def fetch():
    import yfinance as yf
    d = yf.download("GC=F", interval="1h", period="60d",
                    auto_adjust=True, progress=False)
    if isinstance(d.columns, pd.MultiIndex):
        d.columns = d.columns.get_level_values(0)
    return d.dropna()


def load():
    if os.path.exists(STATE):
        return json.load(open(STATE))
    return dict(balance=START, pos=None, last_bar=None, trades=0, wins=0,
                gross_win=0.0, gross_loss=0.0)


def run(fetch_fn=fetch, now=None):
    if not os.path.exists(LOG):
        csv.writer(open(LOG, "w", newline="")).writerow(HEAD)
    now = now or pd.Timestamp.now(tz="UTC")
    df = ohlc(fetch_fn(), "4h")
    df = df[df.index + pd.Timedelta(hours=4) <= now]
    if len(df) < 250:
        print("Pas assez de donnees.")
        return
    o, h, l, c = (df[k].to_numpy(float) for k in ("Open", "High", "Low", "Close"))
    atr, sig = atr14(h, l, c), sig_ema(c, 10, 30, 1)
    st = load()

    if st["last_bar"] is None:
        st["last_bar"] = df.index[-1].isoformat()
        print("Demarrage de la simulation. Premiers trades aux prochaines bougies.")
    else:
        last = pd.Timestamp(st["last_bar"])
        for j in range(1, len(df)):
            if df.index[j] <= last:
                continue
            t, pos = df.index[j], st["pos"]
            if pos is None and sig[j - 1] != 0 and atr[j - 1] > 0:
                side, d = int(sig[j - 1]), float(atr[j - 1])
                e = float(o[j])
                pos = dict(side=side, entry=e, sl=e - side * SLM * d,
                           tp=e + side * TPM * d, open=t.isoformat())
            if pos:
                s = pos["side"]
                hs = l[j] <= pos["sl"] if s == 1 else h[j] >= pos["sl"]
                ht = h[j] >= pos["tp"] if s == 1 else l[j] <= pos["tp"]
                ex = pos["sl"] if hs else (pos["tp"] if ht else None)
                if ex is not None:
                    pnl = ((ex - pos["entry"]) * s - SPREAD) * LOT_OZ
                    st["balance"] += pnl
                    st["trades"] += 1
                    st["wins"] += pnl > 0
                    st["gross_win"] += max(pnl, 0)
                    st["gross_loss"] += max(-pnl, 0)
                    csv.writer(open(LOG, "a", newline="")).writerow(
                        [pos["open"], t.isoformat(), "ACHAT" if s == 1 else "VENTE",
                         round(pos["entry"], 2), round(ex, 2), round(pnl, 2)])
                    pos = None
            st["pos"], st["last_bar"] = pos, t.isoformat()
    json.dump(st, open(STATE, "w"), indent=1)

    pf = f"{st['gross_win'] / st['gross_loss']:.2f}" if st["gross_loss"] else "-"
    lines = [f"Solde virtuel : {st['balance']:.2f} $ (depart {START:.0f} $)",
             f"Trades fermes : {st['trades']}, gagnants : {st['wins']}",
             f"Profit factor : {pf}",
             f"Derniere bougie 4h : {df.index[-1]}  prix {c[-1]:.2f}"]
    p = st["pos"]
    lines.append("Position ouverte : " + (
        f"{'ACHAT' if p['side'] == 1 else 'VENTE'} a {p['entry']:.2f} "
        f"(SL {p['sl']:.2f} / TP {p['tp']:.2f})" if p else "aucune"))
    text = "\n".join(lines)
    print(text)
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        open(os.environ["GITHUB_STEP_SUMMARY"], "a").write(text.replace("\n", "\n\n") + "\n")


if __name__ == "__main__":
    run()
