#!/usr/bin/env python3
"""napkin-boll-live: the boll-barrier strategy, live on NPKN.

Replaces the 5-seed DQN on operator instruction (2026-08-26). The strategy is
the one validated through Quantiacs' own backtester in napkin-labels #30:
enter long when close drops 2 population-sd below its 20-bar mean, exit when
the bar's high/low touches entry*(1 +/- 2*vol) or after 10 bars, slot-sized,
long-only. On the venue's point-in-time NASDAQ panel it scored Sharpe 0.33 at
0.22 exposure; on our survivor tape the family sat at ~1.5 with the market.
Numbers face-up: this is a live test, not a claim of edge.

Inherits every hard-won hold5 design rule:
  * OWN TAPE under out/boll/tape — never napkin_gap's shared files.
  * State on disk, bars counted by TAPE INDEX so missed crons and holidays
    cannot shift barrier expiries.
  * Every run writes its JSON log, including refused runs.
  * Stray venue positions (the legacy DQN book: shorts included) are CLOSED —
    without this the old book is stranded forever.
  * Held positions are never resized; exits and fresh entries only.

Cron: 31 9 * * 1-5 (replaces napkin_gap.py trade).
Usage: selfcheck | trade [--dry] | state
"""
import json, math, os, sys, time
from datetime import datetime
from zoneinfo import ZoneInfo

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "napkin-tape"))
sys.path.insert(0, HERE)

import napkin_gap as ng                      # api(), creds()
import napkin_tape as nt                     # _yahoo_daily(), _coinbase_daily()

ET = ZoneInfo("America/New_York")
OUT = os.path.join(HERE, "out", "boll")
TAPE = os.path.join(OUT, "tape")
STATE = os.path.join(OUT, "state.json")

TICKER = "NPKN"
PREFIX = "CLAWSTREET"

UNIVERSE = list(nt.UNIVERSE)                  # NPKN keeps all 18 incl SOL
BOLL_N, BOLL_K = 20, 2.0
PT, SL, H = 2.0, 2.0, 10
VOL_SPAN = 20
GROSS = 0.95
SLOT_FRAC = GROSS / len(UNIVERSE)             # fixed slot: gross/18 of equity
DUST = ng.DUST_DOLLARS
PAUSE = ng.PAUSE


# ------------------------------------------------------------------ tape

def refresh_tape(universe=None, force=False):
    """Upsert into THIS module's own tape (hold5's function, retargeted)."""
    os.makedirs(TAPE, exist_ok=True)
    for sym in (universe or UNIVERSE):
        path = os.path.join(TAPE, sym.replace(":", "_") + ".jsonl")
        have = {}
        if os.path.exists(path):
            for line in open(path):
                r = json.loads(line)
                have[r["date"]] = r
        fresh = (nt._coinbase_daily(sym) if sym.startswith("X:")
                 else nt._yahoo_daily(sym))
        added = 0
        for r in fresh:
            r = {"date": r["date"], "o": r["o"], "h": r["h"], "l": r["l"],
                 "c": r["c"], "v": r["v"]}
            old = have.get(r["date"])
            if old is None:
                have[r["date"]] = r
                added += 1
            elif not force:
                for f in ("o", "c"):
                    assert abs(old[f] - r[f]) < 1e-6, \
                        f"{sym} {r['date']}: stored {f}={old[f]}, source {r[f]}"
        rows = [have[d] for d in sorted(have)]
        with open(path, "w") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
        print(f"tape {sym:10s} {len(rows)} bars (+{added}) -> {rows[-1]['date']}",
              flush=True)
        time.sleep(0.4)


def load_tape(universe=None):
    universe = universe or UNIVERSE
    by_sym, common = {}, None
    for sym in universe:
        path = os.path.join(TAPE, sym.replace(":", "_") + ".jsonl")
        rows = [json.loads(l) for l in open(path)]
        by_sym[sym] = {r["date"]: r for r in rows}
        ds = set(by_sym[sym])
        common = ds if common is None else (common & ds)
    dates = sorted(common)
    return dates, {s: [by_sym[s][d] for d in dates] for s in universe}


# ---------------------------------------------------------------- signal

def ewma_vol(closes, span=VOL_SPAN):
    if len(closes) < 3:
        return None
    alpha = 2.0 / (span + 1)
    var = None
    for i in range(1, len(closes)):
        r = math.log(closes[i] / closes[i - 1])
        var = r * r if var is None else (1 - alpha) * var + alpha * r * r
    return math.sqrt(var)


def boll_entry(closes, n=BOLL_N, k=BOLL_K):
    """True when the last close sits k population-sd below the n-bar mean."""
    if len(closes) < n:
        return False
    win = closes[-n:]
    m = sum(win) / n
    sd = math.sqrt(sum((c - m) ** 2 for c in win) / n)
    return sd > 0 and closes[-1] < m - k * sd


def plan(dates, bars, st):
    """Pure decision for the LAST bar: (exits, entries, updated_state).
    Exits: barrier touch on the last bar's h/l, or expiry by TAPE INDEX.
    Entries: boll fires, sym not already tracked. State carries entry_date,
    upper, lower; bars held are recomputed from tape indices every run."""
    t = len(dates) - 1
    idx = {d: i for i, d in enumerate(dates)}
    exits, entries = [], []
    pos = dict(st.get("positions", {}))
    for sym, tr in list(pos.items()):
        e = idx.get(tr["entry_date"])
        if e is None:                          # entry date fell off the tape
            exits.append((sym, "lost_anchor"))
            del pos[sym]
            continue
        if t <= e:
            continue                           # entry bar itself: never exit
        bar = bars[sym][t]
        hit_up, hit_dn = bar["h"] >= tr["upper"], bar["l"] <= tr["lower"]
        if hit_up or hit_dn or (t - e) >= H:
            exits.append((sym, "target" if hit_up and not hit_dn else
                          "stop" if hit_dn else "expiry"))
            del pos[sym]
    for sym in UNIVERSE:
        if sym in pos:
            continue
        closes = [b["c"] for b in bars[sym][:t + 1]]
        if not boll_entry(closes):
            continue
        vol = ewma_vol(closes[-(3 * VOL_SPAN + 1):])
        if not vol or vol <= 0:
            continue
        c = closes[-1]
        entries.append(sym)
        pos[sym] = {"entry_date": dates[t], "upper": c * (1 + PT * vol),
                    "lower": c * (1 - SL * vol)}
    return exits, entries, {"positions": pos, "last_bar": dates[t]}


def order_qty(sym, dollars, price):
    q = dollars / price
    return round(q, 4) if sym.startswith("X:") else float(int(q))


# ----------------------------------------------------------------- trade

def read_state():
    if os.path.exists(STATE):
        return json.load(open(STATE))
    return {"positions": {}}


def trade(dry=False, refresh=True):
    if refresh:
        refresh_tape()
    dates, bars = load_tape()
    st = read_state()
    exits, entries, new_state = plan(dates, bars, st)

    c = ng.creds()
    key, aid = c.get(f"{PREFIX}_API_KEY"), c.get(f"{PREFIX}_AGENT_ID")
    if not key or not aid:
        print(f"{TICKER}: no credentials — nothing done"); return
    stamp = datetime.now(ET).strftime("%Y-%m-%d")
    os.makedirs(os.path.join(OUT, "live"), exist_ok=True)
    fn = os.path.join(OUT, "live", f"{stamp}_{TICKER}{'_dry' if dry else ''}.json")
    log = {"date": stamp, "ticker": TICKER, "strategy": "boll-barrier",
           "dry": dry, "bar": dates[-1], "exits": exits, "entries": entries,
           "orders": []}
    save = lambda: json.dump(log, open(fn, "w"), indent=1)

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

    tracked = set(new_state["positions"]) | {s for s, _ in exits}
    need = sorted(set(held) | set(entries) | {s for s, _ in exits})
    legs = []      # (sym, side, qty, why)
    # 1) strays: anything held that this strategy does not track — the legacy
    #    DQN book, shorts included — is closed outright.
    for sym, q in held.items():
        if sym in tracked or abs(q) < 1e-9:
            continue
        legs.append((sym, "sell" if q > 0 else "cover", abs(q),
                     "legacy/untracked position closed on strategy swap"))
    # 2) exits
    for sym, why in exits:
        q = held.get(sym, 0.0)
        if q > 1e-9:
            legs.append((sym, "sell", q, f"barrier exit ({why})"))
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
    for sym in entries:
        price = qmap.get(sym)
        if not price:
            continue
        qty = order_qty(sym, SLOT_FRAC * equity, price)
        if qty * price < DUST:
            continue
        legs.append((sym, "buy", qty, "boll entry, slot gross/18"))

    for sym, side, qty, why in legs:
        rec = {"symbol": sym, "side": side, "qty": qty, "why": why}
        if not dry:
            body = {"symbol": sym, "side": side, "qty": qty,
                    "reasoning": (
                        f"napkin-boll {stamp}: Bollinger 20d/2sd oversold entries "
                        f"with volatility-anchored barrier exits "
                        f"(pt=sl=2*vol, 10-bar expiry), slot gross/18, long-only. "
                        f"{why}. Live test; every decision logged publicly.")}
            rec["response"] = ng.api(f"/me/agents/{aid}/orders", key, "POST",
                                     body,
                                     idem=f"napkin-boll/{stamp}/{sym}/{side}")
            time.sleep(PAUSE)
        log["orders"].append(rec)

    if not dry:
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
    # synthetic tape helpers
    def mk(dates_closes):
        dates = [d for d, _ in dates_closes]
        bars = [{"date": d, "o": c, "h": c, "l": c, "c": c, "v": 1}
                for d, c in dates_closes]
        return dates, bars

    # 1. boll direction: fires on a crash through the band, never on strength
    flat = [100.0] * 25
    assert not boll_entry(flat + [100.0])
    assert not boll_entry([100 + i for i in range(25)])
    crash = [100.0] * 24 + [99.0, 80.0]
    assert boll_entry(crash), "must fire on a 2-sigma break down"
    print("selfcheck 1/7: boll fires on the crash bar, never on flat or strength")

    # 2. entry creates barriers anchored at the entry close
    D = [f"2026-01-{i:02d}" for i in range(1, 27)]
    closes = [100.0] * 24 + [99.0, 80.0]
    dates = D[:26]
    bars = {s: [{"date": d, "o": c, "h": c, "l": c, "c": c, "v": 1}
                for d, c in zip(dates, closes)] for s in UNIVERSE}
    ex, en, st = plan(dates, bars, {"positions": {}})
    assert not ex and set(en) == set(UNIVERSE)
    tr = st["positions"][UNIVERSE[0]]
    vol = ewma_vol(closes[-(3 * VOL_SPAN + 1):])
    assert abs(tr["upper"] - 80.0 * (1 + 2 * vol)) < 1e-9
    assert abs(tr["lower"] - 80.0 * (1 - 2 * vol)) < 1e-9
    print("selfcheck 2/7: entry anchors upper/lower at entry_close*(1 +/- 2*vol)")

    # 3. no exit on the entry bar itself; target exit when high touches upper
    dates2 = dates + ["2026-01-27"]
    up = tr["upper"] + 1
    bars2 = {s: bars[s] + [{"date": "2026-01-27", "o": 80, "h": up, "l": 80,
                            "c": up, "v": 1}] for s in UNIVERSE}
    ex2, _, st2 = plan(dates2, bars2, st)
    assert (UNIVERSE[0], "target") in ex2
    assert UNIVERSE[0] not in st2["positions"]
    print("selfcheck 3/7: barrier target exit fires on the NEXT bar's high")

    # 4. expiry by TAPE INDEX: exactly H bars after entry, flat tape
    datesH = dates + [f"2026-02-{i:02d}" for i in range(1, H + 1)]
    barsH = {s: bars[s] + [{"date": f"2026-02-{i:02d}", "o": 80, "h": 80,
                            "l": 80, "c": 80, "v": 1} for i in range(1, H + 1)]
             for s in UNIVERSE}
    exH, _, stH = plan(datesH, barsH, st)
    assert (UNIVERSE[0], "expiry") in exH, exH[:2]
    exH2, _, _ = plan(datesH[:-1], {s: b[:-1] for s, b in barsH.items()}, st)
    assert not any(w == "expiry" for _, w in exH2), "expired one bar early"
    print(f"selfcheck 4/7: vertical expiry at exactly H={H} bars by tape index")

    # 5. both barriers hit on one bar counts as a STOP (conservative), matching
    #    the backtested rule
    wide = {"positions": {UNIVERSE[0]: {"entry_date": dates[-1],
                                         "upper": 81.0, "lower": 79.0}}}
    bars3 = {s: bars[s] + [{"date": "2026-01-27", "o": 80, "h": 82, "l": 78,
                            "c": 80, "v": 1}] for s in UNIVERSE}
    ex3, _, _ = plan(dates2, bars3, wide)
    assert (UNIVERSE[0], "stop") in ex3
    print("selfcheck 5/7: both-barriers-in-one-bar resolves to stop, as backtested")

    # 6. slot maths and share rounding
    assert order_qty("AAPL", 5013.9, 310.0) == 16.0          # whole shares
    assert order_qty("X:BTCUSD", 5013.9, 78000.0) == 0.0643  # fractional
    assert abs(SLOT_FRAC * 18 - GROSS) < 1e-12
    print("selfcheck 6/7: slots are gross/18; stocks round to whole shares")

    # 7. determinism + state round-trip
    a = plan(dates2, bars2, st)
    b = plan(dates2, bars2, json.loads(json.dumps(st)))
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)
    print("selfcheck 7/7: plan() is pure and state survives a JSON round-trip")
    print("ALL SELFCHECKS PASS")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "selfcheck"
    if cmd == "trade":
        trade(dry="--dry" in sys.argv, refresh="--no-refresh" not in sys.argv)
    else:
        {"selfcheck": selfcheck, "state": state}[cmd]()
