#!/usr/bin/env python3
"""
QQQ intraday "Noise Area" momentum bot, rules v0.1 (Alpaca PAPER account only).

Logic is identical to backtest.py (tested to reproduce its trades exactly).
Decisions happen only at 10:00, 10:30, ... 15:30 ET. Everything is closed at 15:59 ET.

Environment variables (set as GitHub Secrets / workflow env, never in the code):
    APCA_KEY, APCA_SECRET   Alpaca paper API keys
    LEVERAGE                position size = last_equity * LEVERAGE / today's open (default 1)
    DRY_RUN                 "true" (default) = only log, send no orders; "false" = send paper orders

Usage:
    python qqq_bot.py --check                         # connection + today's bands, no trading
    python qqq_bot.py --first 10:00 --last 12:30      # morning session
    python qqq_bot.py --first 13:00 --last 15:30 --close   # afternoon session + close at 15:59
"""
import argparse
import os
import sys
import time
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

NY = ZoneInfo("America/New_York")
SYMBOL = os.environ.get("SYMBOL", "QQQ")
LEVERAGE = float(os.environ.get("LEVERAGE", "1.0"))
DRY_RUN = os.environ.get("DRY_RUN", "true").strip().lower() != "false"
LOOKBACK = 14
VM = 1.0
N_MIN = 390
CHECKS = list(range(30, 361, 30))                       # minute offsets for 10:00..15:30
LABELS = [f"{9 + (30 + k) // 60:02d}:{(30 + k) % 60:02d}" for k in CHECKS]
LOG_FILE = "trade_log.csv"


# ======================================================== pure strategy logic
def prep_day(g):
    """One full day of 1-min bars (naive ET index) -> arrays, same as backtest.load_days."""
    g = g.sort_index()
    g = g[~g.index.duplicated(keep="first")].between_time("09:30", "15:59")
    mins = (g.index.hour * 60 + g.index.minute) - (9 * 60 + 30)
    g = g.set_index(mins).reindex(range(N_MIN))
    g[["open", "high", "low", "close"]] = g[["open", "high", "low", "close"]].ffill().bfill()
    g["volume"] = g["volume"].fillna(0.0)
    tp = (g["high"] + g["low"] + g["close"]) / 3.0
    return dict(O=g["open"].to_numpy(), C=g["close"].to_numpy(),
                cum_pv=(tp * g["volume"]).cumsum().to_numpy(),
                cum_v=g["volume"].cumsum().to_numpy())


def noise_sigma(hist):
    """Average |open(k)/open(0) - 1| over the last LOOKBACK full days, for each checkpoint."""
    moves = np.array([[abs(d["O"][k] / d["O"][0] - 1) for k in CHECKS] for d in hist[-LOOKBACK:]])
    return moves.mean(axis=0)


def band_levels(o0, prev_close, sigma_j):
    top, bot = max(o0, prev_close), min(o0, prev_close)
    return top * (1 + VM * sigma_j), bot * (1 - VM * sigma_j)


def decide(px, vwap, ub, lb, pos):
    """Returns (close_existing, new_side). Same rules as the backtest."""
    was = pos
    stop = (pos == 1 and px < max(ub, vwap)) or (pos == -1 and px > min(lb, vwap))
    cur = 0 if stop else pos
    new = 0
    if cur == 0:
        want = 1 if px > ub else (-1 if px < lb else 0)
        if was != 0 and want == was:                    # no same-direction re-entry on the stop bar
            want = 0
        new = want
    return stop, new


# ================================================================ utilities
def now_ny():
    return datetime.now(NY)


def log(msg):
    print(f"[{now_ny().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def log_row(action, side, qty, decision_px, fill_px, note=""):
    new = not os.path.exists(LOG_FILE)
    with open(LOG_FILE, "a") as f:
        if new:
            f.write("time_et,action,side,qty,decision_px,fill_px,note\n")
        f.write(f"{now_ny().isoformat(timespec='seconds')},{action},{side},{qty},{decision_px},{fill_px},{note}\n")


def sleep_until(t):
    while True:
        left = (t - now_ny()).total_seconds()
        if left <= 0:
            return
        time.sleep(min(left, 30))


def at(day, hhmm, sec=0):
    h, m = map(int, hhmm.split(":"))
    return datetime(day.year, day.month, day.day, h, m, sec, tzinfo=NY)


# ================================================================ Alpaca I/O
def clients():
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.trading.client import TradingClient
    key, secret = os.environ["APCA_KEY"], os.environ["APCA_SECRET"]
    return TradingClient(key, secret, paper=True), StockHistoricalDataClient(key, secret)


def to_naive_et(df):
    df = df.reset_index()
    ts = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert("America/New_York").dt.tz_localize(None)
    df.index = ts
    return df[["open", "high", "low", "close", "volume"]]


def load_history(data, today):
    """Last full days (>=380 bars) before today, from the consolidated (SIP) feed."""
    from alpaca.data.enums import Adjustment, DataFeed
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    start = datetime(today.year, today.month, today.day, tzinfo=NY) - timedelta(days=50)
    end = datetime(today.year, today.month, today.day, tzinfo=NY)
    req = StockBarsRequest(symbol_or_symbols=SYMBOL, timeframe=TimeFrame.Minute, start=start, end=end,
                           feed=DataFeed.SIP, adjustment=Adjustment.RAW)
    df = to_naive_et(data.get_stock_bars(req).df)
    days = []
    for _, g in df.groupby(df.index.date):
        g = g.between_time("09:30", "15:59")
        if len(g) >= 380:                               # skips half days, like the backtest
            days.append(prep_day(g))
    return days


def todays_bars(data, today):
    from alpaca.data.enums import DataFeed
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    start = datetime(today.year, today.month, today.day, 9, 30, tzinfo=NY)
    req = StockBarsRequest(symbol_or_symbols=SYMBOL, timeframe=TimeFrame.Minute, start=start,
                           end=now_ny(), feed=DataFeed.IEX)
    df = data.get_stock_bars(req).df
    return to_naive_et(df) if len(df) else df


def latest_price(data):
    from alpaca.data.enums import DataFeed
    from alpaca.data.requests import StockLatestTradeRequest
    r = data.get_stock_latest_trade(StockLatestTradeRequest(symbol_or_symbols=SYMBOL, feed=DataFeed.IEX))
    return float(r[SYMBOL].price)


def is_full_day(trading, today):
    from alpaca.trading.requests import GetCalendarRequest
    cal = trading.get_calendar(GetCalendarRequest(start=today, end=today))
    if not cal or cal[0].date != today:
        return False
    c = cal[0].close
    return c.hour == 16 and c.minute == 0


def get_pos(trading):
    try:
        p = trading.get_open_position(SYMBOL)
    except Exception:
        return 0, 0
    q = int(float(p.qty))
    return (1 if q > 0 else -1), abs(q)


def wait_fill(trading, order, tries=30):
    for _ in range(tries):
        o = trading.get_order_by_id(order.id)
        if str(o.status).lower().split(".")[-1] == "filled":
            return float(o.filled_avg_price)
        time.sleep(1)
    return float("nan")


def already_handled(trading, t):
    """True if any order for SYMBOL was submitted since checkpoint time t (duplicate-run guard)."""
    from alpaca.trading.enums import QueryOrderStatus
    from alpaca.trading.requests import GetOrdersRequest
    orders = trading.get_orders(GetOrdersRequest(status=QueryOrderStatus.ALL, symbols=[SYMBOL],
                                                 after=t, limit=50))
    return len(orders) > 0


def close_position(trading, px, note):
    if DRY_RUN:
        log(f"DRY_RUN would close position ({note})")
        log_row("close(dry)", 0, 0, px, "", note)
        return
    pos, qty = get_pos(trading)
    if pos == 0:
        return
    order = trading.close_position(SYMBOL)
    fill = wait_fill(trading, order)
    log(f"CLOSED {qty} {SYMBOL} fill={fill} decision_px={px} ({note})")
    log_row("close", pos, qty, px, fill, note)


def open_position(trading, side, px, o0):
    from alpaca.trading.enums import OrderSide, TimeInForce
    from alpaca.trading.requests import MarketOrderRequest
    equity = float(trading.get_account().last_equity)
    qty = int(equity * LEVERAGE / o0)
    if qty < 1:
        log("size < 1 share, skip")
        return
    if DRY_RUN:
        log(f"DRY_RUN would {'BUY' if side == 1 else 'SHORT'} {qty} {SYMBOL} @ ~{px}")
        log_row("open(dry)", side, qty, px, "", "")
        return
    req = MarketOrderRequest(symbol=SYMBOL, qty=qty, side=OrderSide.BUY if side == 1 else OrderSide.SELL,
                             time_in_force=TimeInForce.DAY)
    order = trading.submit_order(req)
    fill = wait_fill(trading, order)
    log(f"OPENED {'LONG' if side == 1 else 'SHORT'} {qty} {SYMBOL} fill={fill} decision_px={px}")
    log_row("open", side, qty, px, fill, "")


# ===================================================================== main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--first", default="10:00")
    ap.add_argument("--last", default="15:30")
    ap.add_argument("--close", action="store_true", help="flatten at 15:59 ET")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()

    trading, data = clients()
    today = now_ny().date()
    log(f"{SYMBOL} leverage={LEVERAGE} dry_run={DRY_RUN} paper=True")
    acct = trading.get_account()
    log(f"account equity={acct.equity} last_equity={acct.last_equity} buying_power={acct.buying_power}")

    if not is_full_day(trading, today):
        log("Not a full trading day (closed or early close) -> no trading today.")
        return
    hist = load_history(data, today)
    if len(hist) < LOOKBACK + 1:
        log(f"Only {len(hist)} history days loaded, need {LOOKBACK + 1}. Abort.")
        sys.exit(1)
    prev_close = float(hist[-1]["C"][-1])
    sigma = noise_sigma(hist)
    log(f"history days={len(hist)} prev_close={prev_close:.2f}")

    if a.check:
        bars = todays_bars(data, today)
        px = latest_price(data)
        o0 = float(bars["open"].iloc[0]) if len(bars) else px
        log(f"CHECK: today open~{o0:.2f} last price={px:.2f}")
        for j, k in enumerate(CHECKS):
            ub, lb = band_levels(o0, prev_close, sigma[j])
            log(f"  {LABELS[j]}  upper={ub:.2f}  lower={lb:.2f}  sigma={sigma[j]:.4%}")
        log("Check finished. No orders were sent.")
        return

    pos = 0                                              # only used for DRY_RUN
    for j, k in enumerate(CHECKS):
        if not (a.first <= LABELS[j] <= a.last):
            continue
        t = at(today, "09:30") + timedelta(minutes=k)
        if now_ny() > t + timedelta(minutes=3):
            log(f"{LABELS[j]} checkpoint missed (job started late), skipping")
            continue
        sleep_until(t + timedelta(seconds=3))
        try:
            if not DRY_RUN and already_handled(trading, t):
                log(f"{LABELS[j]} already handled by another run, skipping")
                continue
            bars = todays_bars(data, today)
            px = latest_price(data)
            if len(bars) == 0:
                log(f"{LABELS[j]} no bars yet, skipping")
                continue
            o0 = float(bars["open"].iloc[0])
            prior = bars[bars.index < pd.Timestamp(t.replace(tzinfo=None))]
            tp = (prior["high"] + prior["low"] + prior["close"]) / 3.0
            v = float(prior["volume"].sum())
            vwap = float((tp * prior["volume"]).sum() / v) if v > 0 else px
            ub, lb = band_levels(o0, prev_close, sigma[j])
            cur = pos if DRY_RUN else get_pos(trading)[0]
            stop, new = decide(px, vwap, ub, lb, cur)
            log(f"{LABELS[j]} px={px:.2f} vwap={vwap:.2f} upper={ub:.2f} lower={lb:.2f} pos={cur} -> stop={stop} new={new}")
            if stop:
                close_position(trading, px, f"stop {LABELS[j]}")
                cur = 0
            if new != 0:
                open_position(trading, new, px, o0)
                cur = new
            pos = cur
        except Exception as e:                           # never crash the day on one bad call
            log(f"ERROR at {LABELS[j]}: {e!r}")

    if a.close:
        sleep_until(at(today, "15:59", 15))
        try:
            close_position(trading, float("nan"), "end of day")
        except Exception as e:
            log(f"ERROR closing at end of day: {e!r}")
        log("Day finished.")


if __name__ == "__main__":
    main()
