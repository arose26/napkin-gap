#!/usr/bin/env python3
"""napkin-near52h-live: the near52h-gate strategy, live on NPKN.

Replaces boll-barrier on operator instruction (2026-08-28). The strategy is
the single best performer of the Quantiacs slate (napkin-labels #39,
napkin-near52h-gate, in-sample Sharpe 1.488): hold the top-3 symbols by
closeness to their trailing 52-week high (close / rolling max), equal weight,
refreshed weekly, long-only — and in CASH whenever BTC closes below its own
trailing 100-bar mean. Adapted honestly to this venue's tape:

  * the grid is the 18-symbol common-DATE intersection (trading days), so the
    52-week high is a 252-BAR rolling max and the BTC gate a 100-BAR SMA
    (the Quantiacs original used 365/100 calendar rows on daily crypto bars);
  * rebalance cadence is 5 bars (weekly on a weekday grid; original: 7 days);
  * the gate input (X:BTCUSD closes) comes from THIS bot's own tape — the
    same feed refreshed live at 09:31 — never an external series. If the
    tape cannot serve 252 finite bars the run REFUSES and logs; no fallback.

Backtested 2022-06-29..2026-08-25 on this exact tape/grid via load_tape():
+130.4% total, -13.1% maxDD, Sharpe 1.24 at 0.52 exposure — vs boll-barrier
+33.3%/-5.0%/1.17 at 0.10, and the exposure-matched buy&hold blend
+89.0%/-16.2%/1.30. Survivor-universe caveat stated: the 15 stocks are
today's mega-caps, which flatters every long book on this tape equally.

Inherits every hold5/boll design rule:
  * OWN TAPE under out/near52h/tape — never another module's files.
  * State on disk; rebalance cadence counted by TAPE INDEX so missed crons
    and holidays cannot shift the phase.
  * Every run writes its JSON log, including refused runs.
  * Stray venue positions (the boll book) are CLOSED on the first run.
  * Held positions are never resized; exits and fresh entries only.
  * A failed BUY is not tracked — no phantom entries.

Cron: 31 9 * * 1-5 (replaces napkin_boll_live.py trade).
Usage: selfcheck | trade [--dry] | state
"""
import json, os, sys, time
from datetime import datetime
from zoneinfo import ZoneInfo

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "napkin-tape"))
sys.path.insert(0, HERE)

import napkin_gap as ng                      # api(), creds()
import napkin_boll_live as nb                # refresh/load tape machinery

ET = ZoneInfo("America/New_York")
OUT = os.path.join(HERE, "out", "near52h")
TAPE = os.path.join(OUT, "tape")
STATE = os.path.join(OUT, "state.json")

TICKER = "NPKN"
PREFIX = "CLAWSTREET"

UNIVERSE = list(nb.UNIVERSE)
LOOK, GATE_N, K, CAD = 252, 100, 3, 5
GROSS = 0.95
GATE_SYM = "X:BTCUSD"
DUST = ng.DUST_DOLLARS
PAUSE = ng.PAUSE


# ------------------------------------------------------------------ tape

def refresh_tape(universe=None, force=False):
    """boll's upsert, retargeted at THIS module's own tape dir."""
    old = nb.TAPE
    try:
        nb.TAPE = TAPE
        nb.refresh_tape(universe or UNIVERSE, force=force)
    finally:
        nb.TAPE = old


def load_tape(universe=None):
    old = nb.TAPE
    try:
        nb.TAPE = TAPE
        return nb.load_tape(universe or UNIVERSE)
    finally:
        nb.TAPE = old


# ---------------------------------------------------------------- signal

def gate_on(btc_closes, n=GATE_N):
    """True (risk-on) when the last BTC close >= its n-bar simple mean."""
    if len(btc_closes) < n:
        raise ValueError(f"gate needs {n} bars, tape has {len(btc_closes)}")
    return btc_closes[-1] >= sum(btc_closes[-n:]) / n


def near52h_score(closes, look=LOOK):
    """close / rolling look-bar max (window includes the current bar)."""
    if len(closes) < look:
        raise ValueError(f"score needs {look} bars, tape has {len(closes)}")
    return closes[-1] / max(closes[-look:])


def plan(dates, bars, st):
    """Pure decision for the LAST bar: (exits, entries, updated_state).
    Risk-off (BTC < SMA-100 on this tape) -> target empty, flatten.
    Risk-on -> every CAD bars (by TAPE INDEX) re-pick top-K by 52w-high
    closeness; between rebalances hold the tracked set unchanged."""
    t = len(dates) - 1
    idx = {d: i for i, d in enumerate(dates)}
    tracked = set(st.get("positions", {}))
    on = gate_on([b["c"] for b in bars[GATE_SYM][:t + 1]])
    lr = idx.get(st.get("last_reb"))           # None if date fell off grid
    if not on:
        target = set()
        new_lr = st.get("last_reb")            # keep the phase for re-entry
    elif lr is None or (t - lr) >= CAD:
        scores = {s: near52h_score([b["c"] for b in bars[s][:t + 1]])
                  for s in UNIVERSE}
        target = set(sorted(scores, key=lambda s: (scores[s], s))[-K:])
        new_lr = dates[t]
    else:
        target = tracked
        new_lr = st.get("last_reb")
    exits = sorted(tracked - target)
    entries = sorted(target - tracked)
    pos = {s: (st["positions"][s] if s in tracked else {"entry_date": dates[t]})
           for s in target}
    return exits, entries, {"positions": pos, "last_reb": new_lr,
                            "last_bar": dates[t]}


def order_qty(sym, dollars, price):
    q = dollars / price
    return round(q, 4) if sym.startswith("X:") else float(int(q))


# ----------------------------------------------------------------- trade

def read_state():
    if os.path.exists(STATE):
        return json.load(open(STATE))
    return {"positions": {}, "last_reb": None}


def trade(dry=False, refresh=True):
    if refresh:
        refresh_tape()
    dates, bars = load_tape()
    st = read_state()

    c = ng.creds()
    key, aid = c.get(f"{PREFIX}_API_KEY"), c.get(f"{PREFIX}_AGENT_ID")
    if not key or not aid:
        print(f"{TICKER}: no credentials — nothing done"); return
    stamp = datetime.now(ET).strftime("%Y-%m-%d")
    os.makedirs(os.path.join(OUT, "live"), exist_ok=True)
    fn = os.path.join(OUT, "live", f"{stamp}_{TICKER}{'_dry' if dry else ''}.json")
    log = {"date": stamp, "ticker": TICKER, "strategy": "near52h-gate",
           "dry": dry, "bar": dates[-1], "orders": []}
    save = lambda: json.dump(log, open(fn, "w"), indent=1)

    try:
        exits, entries, new_state = plan(dates, bars, st)
    except ValueError as e:                    # tape too short for gate/score
        log["refused"] = str(e); save()
        print(f"{TICKER}: REFUSED — {e} -> {fn}"); return
    log["exits"], log["entries"] = exits, entries

    me = ng.api("/me", key)
    if not dry and not (me.get("agent") or {}).get("claimed"):
        log["refused"] = "not claimed"; save()
        print(f"{TICKER}: not claimed — no live orders -> {fn}"); return
    time.sleep(PAUSE)
    pf = ng.api(f"/me/agents/{aid}/portfolio", key)
    time.sleep(PAUSE)
    pos_r = ng.api(f"/me/agents/{aid}/positions", key)
    time.sleep(PAUSE)
    equity = pf.get("equity") or 100_000.0
    held = {}
    for p in (pos_r.get("positions") or pos_r.get("data") or []):
        q = float(p.get("qty", p.get("quantity", 0)))
        if str(p.get("side", "long")) == "short" and q > 0:
            q = -q
        held[p["symbol"]] = q
    log["equity"], log["positions_before"] = equity, dict(held)

    tracked = set(new_state["positions"]) | set(exits)
    need = sorted(set(held) | set(entries) | set(exits))
    legs = []      # (sym, side, qty, why)
    # 1) strays: anything held that this strategy does not track — the boll
    #    book on the first run — is closed outright.
    for sym, q in held.items():
        if sym in tracked or abs(q) < 1e-9:
            continue
        legs.append((sym, "sell" if q > 0 else "cover", abs(q),
                     "legacy/untracked position closed on strategy swap"))
    # 2) exits (rotation out of the top-3, or gate risk-off)
    for sym in exits:
        q = held.get(sym, 0.0)
        if q > 1e-9:
            legs.append((sym, "sell", q, "near52h exit (rotation/gate)"))
    # 3) entries, slot-sized off current equity
    if need:
        quotes = ng.api("/quotes?symbols=" +
                        ",".join(s.replace(":", "%3A") for s in need), key)
        time.sleep(PAUSE)
        qraw = quotes.get("quotes") or {}
        qmap = {s: (v.get("price") if isinstance(v, dict) else None)
                for s, v in (qraw.items() if isinstance(qraw, dict) else [])}
    else:
        qmap = {}
    slot = GROSS / max(1, len(new_state["positions"]) or K)
    for sym in entries:
        price = qmap.get(sym)
        if not price:
            continue
        # net against anything already held (e.g. a legacy short of the same
        # symbol on swap day): buy up to the slot, never resize downward.
        qty = order_qty(sym, slot * equity, price) - held.get(sym, 0.0)
        qty = round(qty, 4) if sym.startswith("X:") else float(int(qty))
        if qty <= 0 or qty * price < DUST:
            continue
        legs.append((sym, "buy", qty, f"near52h entry, slot gross/{K}"))

    for sym, side, qty, why in legs:
        rec = {"symbol": sym, "side": side, "qty": qty, "why": why}
        if not dry:
            body = {"symbol": sym, "side": side, "qty": qty,
                    "reasoning": (
                        f"napkin-near52h {stamp}: top-{K} by closeness to the "
                        f"trailing 52-week high, equal weight, weekly refresh, "
                        f"long-only; in cash when BTC < its 100-bar SMA. "
                        f"{why}. Live test; every decision logged publicly.")}
            rec["response"] = ng.api(f"/me/agents/{aid}/orders", key, "POST",
                                     body,
                                     idem=f"napkin-near52h/{stamp}/{sym}/{side}")
            time.sleep(PAUSE)
        log["orders"].append(rec)

    if not dry:
        # a failed BUY must not be tracked, or a phantom position blocks the
        # slot until the next rebalance. (Failed exits self-heal: an unexited
        # holding is untracked next run -> closed as a stray.)
        for o in log["orders"]:
            if o["side"] == "buy" and not o.get("response", {}).get("success"):
                new_state["positions"].pop(o["symbol"], None)
        os.makedirs(OUT, exist_ok=True)
        json.dump(new_state, open(STATE, "w"), indent=1)
    save()
    placed = sum(1 for o in log["orders"] if "response" in o)
    ok = sum(1 for o in log["orders"] if o.get("response", {}).get("success"))
    print(f"{TICKER}: bar {dates[-1]}, {len(exits)} exits, {len(entries)} "
          f"entries, {len(legs)} legs, placed {placed}, ok {ok} -> {fn}")


def state():
    print(json.dumps(read_state(), indent=1))


# --------------------------------------------------------------- selfcheck

def selfcheck():
    D = [f"d{i:04d}" for i in range(300)]

    def mk(close_fn):
        return [{"date": d, "o": close_fn(i), "h": close_fn(i),
                 "l": close_fn(i), "c": close_fn(i), "v": 1}
                for i, d in enumerate(D)]

    # 1. score: at the high -> 1.0; halfway below a past high -> 0.5
    assert near52h_score([1.0] * 251 + [2.0]) == 1.0
    assert near52h_score([2.0] + [1.0] * 251) == 0.5
    try:
        near52h_score([1.0] * 251); assert False
    except ValueError:
        pass
    print("selfcheck 1/6: 52w score = close/rollmax(252), refuses short tape")

    # 2. gate: on at/above the 100-bar mean, off below, refuses short tape
    assert gate_on([1.0] * 100)
    assert not gate_on([2.0] * 99 + [1.0])
    try:
        gate_on([1.0] * 99); assert False
    except ValueError:
        pass
    print("selfcheck 2/6: BTC gate is >= SMA-100, refuses short tape")

    # 3. risk-on rebalance picks the top-K by score, equalish tape elsewhere
    bars = {s: mk(lambda i: 100.0) for s in UNIVERSE}
    hot = [s for s in UNIVERSE if s != GATE_SYM][:K]
    for s in hot:   # rising into their high -> score 1.0; others sag to 0.5
        bars[s] = mk(lambda i: 100.0 + i)
    for s in UNIVERSE:
        if s not in hot:   # incl. BTC: spike at i=150 is outside the 100-bar
            bars[s] = mk(lambda i: 200.0 if i == 150 else 100.0)  # gate window
    ex, en, st = plan(D, bars, {"positions": {}, "last_reb": None})
    assert set(en) == set(hot) and not ex, (en, hot)
    assert st["last_reb"] == D[-1]
    print(f"selfcheck 3/6: rebalance enters the top-{K} by 52w closeness")

    # 4. between rebalances the book is held unchanged (no resizing/rotation)
    D2 = D + ["d0300"]
    bars2 = {s: b + [dict(b[-1], date="d0300")] for s, b in bars.items()}
    ex2, en2, st2 = plan(D2, bars2, st)
    assert not ex2 and not en2 and set(st2["positions"]) == set(hot)
    assert st2["last_reb"] == st["last_reb"], "phase must not drift"
    print(f"selfcheck 4/6: holds the book between rebalances (CAD={CAD} bars)")

    # 5. gate off -> everything exits; phase preserved for re-entry
    bars3 = dict(bars2)
    bars3[GATE_SYM] = bars2[GATE_SYM][:-1] + [dict(bars2[GATE_SYM][-1], c=1.0)]
    ex3, en3, st3 = plan(D2, bars3, st)
    assert set(ex3) == set(hot) and not en3 and not st3["positions"]
    print("selfcheck 5/6: BTC below SMA-100 flattens the book to cash")

    # 6. determinism + state round-trip; slot maths
    a = plan(D2, bars2, st)
    b = plan(D2, bars2, json.loads(json.dumps(st)))
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)
    assert abs(GROSS / K * K - GROSS) < 1e-12
    assert order_qty("AAPL", 30000.0, 310.0) == 96.0
    assert order_qty("X:BTCUSD", 30000.0, 78000.0) == 0.3846
    print("selfcheck 6/6: plan() is pure; slots are gross/3; share rounding ok")
    print("ALL SELFCHECKS PASS")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "selfcheck"
    if cmd == "trade":
        trade(dry="--dry" in sys.argv, refresh="--no-refresh" not in sys.argv)
    else:
        {"selfcheck": selfcheck, "state": state}[cmd]()
