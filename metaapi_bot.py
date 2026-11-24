import asyncio
import json
import os

import pandas as pd

from optimize import atr14, ohlc, sig_ema

SYMBOL = os.environ.get("SYMBOL") or "XAUUSD"
LOT = min(float(os.environ.get("LOT") or 0.01), 0.02)
SLM, TPM = 2.0, 4.0
STATE = "bot_state.json"


def fetch():
    import yfinance as yf
    d = yf.download("GC=F", interval="1h", period="60d",
                    auto_adjust=True, progress=False)
    if isinstance(d.columns, pd.MultiIndex):
        d.columns = d.columns.get_level_values(0)
    return d.dropna()


def load():
    return json.load(open(STATE)) if os.path.exists(STATE) else {"last_bar": None}


def say(text):
    print(text)
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        open(os.environ["GITHUB_STEP_SUMMARY"], "a").write(text + "\n\n")


async def main(fetch_fn=fetch, now=None, api_factory=None):
    now = now or pd.Timestamp.now(tz="UTC")
    df = ohlc(fetch_fn(), "4h")
    df = df[df.index + pd.Timedelta(hours=4) <= now]
    if len(df) < 250:
        say("Pas assez de donnees.")
        return
    c = df["Close"].to_numpy(float)
    atr = atr14(df["High"].to_numpy(float), df["Low"].to_numpy(float), c)
    side, d = int(sig_ema(c, 10, 30, 1)[-1]), float(atr[-1])
    bar = df.index[-1]
    age_h = (now - (bar + pd.Timedelta(hours=4))).total_seconds() / 3600

    st = load()
    if st["last_bar"] == bar.isoformat():
        say(f"Bougie {bar} deja traitee. Rien a faire.")
        return
    if side == 0 or age_h > 3:
        st["last_bar"] = bar.isoformat()
        json.dump(st, open(STATE, "w"))
        say(f"Bougie {bar} : pas de signal. Aucun ordre.")
        return

    if api_factory is None:
        from metaapi_cloud_sdk import MetaApi
        api_factory = lambda: MetaApi(token=os.environ["METAAPI_TOKEN"])
    api = api_factory()
    account = await api.metatrader_account_api.get_account(os.environ["METAAPI_ACCOUNT_ID"])
    if account.state != "DEPLOYED":
        await account.deploy()
    await account.wait_connected()
    conn = account.get_rpc_connection()
    await conn.connect()
    await conn.wait_synchronized()

    info = await conn.get_account_information()
    say(f"Compte : serveur {info.get('server')}, solde {info.get('balance')}, "
        f"equity {info.get('equity')}")
    open_pos = [p for p in await conn.get_positions() if p.get("symbol") == SYMBOL]
    if open_pos:
        say(f"Signal {'ACHAT' if side == 1 else 'VENTE'} mais une position {SYMBOL} "
            "est deja ouverte : aucun nouvel ordre.")
        st["last_bar"] = bar.isoformat()
        json.dump(st, open(STATE, "w"))
        return

    await conn.subscribe_to_market_data(SYMBOL)
    price = await conn.get_symbol_price(SYMBOL)
    ask, bid = price["ask"], price["bid"]
    opts = {"comment": "bot4h", "clientId": f"x4h{int(bar.timestamp())}"}
    if side == 1:
        res = await conn.create_market_buy_order(
            SYMBOL, LOT, round(ask - SLM * d, 2), round(ask + TPM * d, 2), opts)
    else:
        res = await conn.create_market_sell_order(
            SYMBOL, LOT, round(bid + SLM * d, 2), round(bid - TPM * d, 2), opts)
    ok = res.get("stringCode") in ("TRADE_RETCODE_DONE", "ERR_NO_ERROR") \
        or res.get("numericCode") in (10009, 0)
    say(f"Ordre {'ACHAT' if side == 1 else 'VENTE'} {LOT} lot {SYMBOL} : "
        f"{'ACCEPTE' if ok else 'REFUSE'} -> {res}")
    if ok:
        st["last_bar"] = bar.isoformat()
        json.dump(st, open(STATE, "w"))


if __name__ == "__main__":
    asyncio.run(main())
