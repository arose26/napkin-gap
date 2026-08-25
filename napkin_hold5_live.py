#!/usr/bin/env python3
"""napkin-hold5-live: the hold5 cross-sectional strategy, deployed to ClawStreet.

Drop-in sibling of napkin_gap.py. Trades ONE agent (default NPKL / keel) and
leaves napkin_gap's NPKN agent completely alone — different credentials,
different state file, its OWN tape copy, no shared mutable data.

  Commands: train | selfcheck | decide | trade [--dry] | state

WHAT THIS STRATEGY IS (from arose26/napkin-labels, experiments #B3-#B6):
  One pooled logistic regression over the whole universe (not per-symbol),
  features = last 10 log returns, labels = triple-barrier (pt=sl=2*vol, h=10).
  Rank symbols by P(up), gate at 0.5, hold the top 3 equal-weighted at 0.95
  gross, RE-RANK ONLY EVERY 5 BARS, and trade only when the held SET changes.

WHAT THE EVIDENCE SAYS — read before trusting it:
  On 9 walk-forward windows covering dates never used in its development
  (2018-09 to 2023-05), this strategy returned -2.05% mean excess vs
  buy-and-hold at 5x venue costs (t -0.32, 4/9 windows). Its earlier +11.83%
  came from selecting the best of 36 backtest cells, and swung 17.5pp merely
  from dropping one symbol. In genuine bear regimes it was much worse than
  buy-and-hold (crypto winter -27.4pp, 2022 bear -6.8pp, COVID -18.0pp).
  The one robust finding is RELATIVE: re-ranking every bar is worse than
  holding 5 (-2.05% vs -9.27%), i.e. this loses less than the per-bar version.
  Deployed at the operator's explicit direction with that record on file.

DESIGN NOTES (deliberate choices, not accidents):
  * OWN TAPE. napkin_tape.bulk() SKIPS any symbol whose file already has
    >=1800 bars, so it does not advance an established tape. This module
    therefore maintains its own copy under out/hold5/tape and upserts new
    bars into it each run. napkin_gap's shared bulk files are never written,
    so NPKN's behaviour is bit-for-bit unaffected by installing this.
  * STATE ON DISK. The 5-bar cadence cannot be inferred from a daily cron, so
    the last re-rank BAR DATE and the intended holding set are persisted.
    Bars are counted by tape index, not calendar days, so a missed cron run
    or a holiday cannot silently change the strategy.
  * UNIVERSE EXITS ARE EXPLICIT. Removing a symbol (X:SOLUSD here) would
    otherwise strand any open position forever, because a symbol outside the
    universe never receives an order. Anything held outside TRADE_UNIVERSE is
    closed on the next run.
  * LONG ONLY. Targets are non-negative, so map_orders can only ever emit
    buy/sell legs; asserted in selfcheck.
  * Pure stdlib. No torch, no numpy — this module cannot be broken by the
    DQN stack's dependencies.
"""
import json, math, os, sys, time
from datetime import datetime
from zoneinfo import ZoneInfo

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "napkin-tape"))
import napkin_gap as ng                      # api(), creds(), map_orders()
import napkin_tape as nt                     # _yahoo_daily(), _coinbase_daily()

ET = ZoneInfo("America/New_York")
OUT = os.path.join(HERE, "out", "hold5")
TAPE = os.path.join(OUT, "tape")
MODEL = os.path.join(OUT, "model.json")
STATE = os.path.join(OUT, "state.json")

# ---- the agent this module trades. NPKN (napkin) is deliberately absent. ----
TICKER = "NPKL"
PREFIX = "CLAWSTREET_KEEL"

DROP = ("X:SOLUSD",)                          # removed from trading per operator
TRADE_UNIVERSE = [s for s in nt.UNIVERSE if s not in DROP]

K, GROSS, HOLD_BARS, THR = 3, 0.95, 5, 0.5    # the hold5 configuration
NFEAT, PT, SL, H, VOL_SPAN = 10, 2.0, 2.0, 10, 20
EPOCHS, LR, L2 = 300, 0.5, 1e-3
DUST = ng.DUST_DOLLARS
PAUSE = ng.PAUSE


# ------------------------------------------------------------------ tape

def refresh_tape(universe=None, force=False):
    """Upsert fresh daily bars into THIS module's own tape copy. Never touches
    napkin_gap's shared bulk files. Existing bars must match incoming ones —
    a silent history rewrite fails loudly, as in napkin_tape.merge_rows."""
    os.makedirs(TAPE, exist_ok=True)
    for sym in (universe or TRADE_UNIVERSE):
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
                        f"{sym} {r['date']}: stored {f}={old[f]}, source says {r[f]}"
        rows = [have[d] for d in sorted(have)]
        with open(path, "w") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
        print(f"tape {sym:10s} {len(rows)} bars (+{added}) -> {rows[-1]['date']}",
              flush=True)
        time.sleep(0.4)


def load_tape(universe=None):
    """Aligned bars over dates common to every traded symbol, oldest first."""
    universe = universe or TRADE_UNIVERSE
    by_sym, common = {}, None
    for sym in universe:
        path = os.path.join(TAPE, sym.replace(":", "_") + ".jsonl")
        rows = [json.loads(l) for l in open(path)]
        by_sym[sym] = {r["date"]: r for r in rows}
        ds = set(by_sym[sym])
        common = ds if common is None else (common & ds)
    dates = sorted(common)
    return dates, {s: [by_sym[s][d] for d in dates] for s in universe}


# ------------------------------------------------------------------ model

def features(closes):
    if len(closes) < NFEAT + 1:
        return None
    return [math.log(closes[i] / closes[i - 1]) for i in range(-NFEAT, 0)]


def ewma_vol(closes, span=VOL_SPAN):
    if len(closes) < 3:
        return None
    a, var = 2.0 / (span + 1), None
    for i in range(1, len(closes)):
        r = math.log(closes[i] / closes[i - 1])
        var = r * r if var is None else (1 - a) * var + a * r * r
    return math.sqrt(var)


def triple_barrier(bars, t0, vol):
    """+1 profit target, -1 stop, 0 vertical/ambiguous. High/low touch test."""
    if t0 + 1 >= len(bars):
        return None
    p0 = bars[t0]["c"]
    up, dn = p0 * (1 + PT * vol), p0 * (1 - SL * vol)
    last = min(t0 + H, len(bars) - 1)
    for t in range(t0 + 1, last + 1):
        hu, hd = bars[t]["h"] >= up, bars[t]["l"] <= dn
        if hu and hd:
            return 0, t
        if hu:
            return 1, t
        if hd:
            return -1, t
    return 0, last


def zstats(X):
    n, k = len(X), len(X[0])
    mu = [sum(x[j] for x in X) / n for j in range(k)]
    sd = [math.sqrt(sum((x[j] - mu[j]) ** 2 for x in X) / n) or 1.0 for j in range(k)]
    return mu, sd


def zapply(x, mu, sd):
    return [(x[j] - mu[j]) / sd[j] for j in range(len(x))]


def fit_logit(X, y):
    """Full-batch GD, zero init, fixed epochs: seed-free and byte-reproducible."""
    n, k = len(X), len(X[0])
    w, b = [0.0] * k, 0.0
    for _ in range(EPOCHS):
        gw, gb = [0.0] * k, 0.0
        for x, t in zip(X, y):
            z = b + sum(w[j] * x[j] for j in range(k))
            p = 1.0 / (1.0 + math.exp(-max(-30, min(30, z))))
            e = p - t
            for j in range(k):
                gw[j] += e * x[j]
            gb += e
        for j in range(k):
            w[j] -= LR * (gw[j] / n + L2 * w[j])
        b -= LR * gb / n
    return w, b


def predict(model, f):
    z = model["b"] + sum(wj * xj for wj, xj in
                         zip(model["w"], zapply(f, model["mu"], model["sd"])))
    return 1.0 / (1.0 + math.exp(-max(-30, min(30, z))))


def build_pool(dates, bars, upto=None):
    """Pooled training rows. Only events whose barrier window has fully
    resolved by `upto` are used, so no label can peek past the data."""
    hi = (len(dates) - 1) if upto is None else upto
    X, y = [], []
    for sym in bars:
        closes = [b["c"] for b in bars[sym]]
        for t0 in range(max(NFEAT, 21), hi + 1):
            if t0 + H > hi:
                break
            vol = ewma_vol(closes[max(0, t0 - 3 * VOL_SPAN):t0 + 1])
            if not vol or vol <= 0:
                continue
            out = triple_barrier(bars[sym], t0, vol)
            if out is None or out[0] == 0 or out[1] > hi:
                continue
            f = features(closes[:t0 + 1])
            if f is None:
                continue
            X.append(f)
            y.append(1 if out[0] > 0 else 0)
    return X, y


def train(refresh=True):
    if refresh:
        refresh_tape()
    dates, bars = load_tape()
    X, y = build_pool(dates, bars)
    assert len(X) >= 500 and 0 < sum(y) < len(y), f"degenerate training set ({len(X)})"
    mu, sd = zstats(X)
    w, b = fit_logit([zapply(x, mu, sd) for x in X], y)
    model = {"w": w, "b": b, "mu": mu, "sd": sd,
             "universe": TRADE_UNIVERSE, "rows": len(X),
             "up_frac": sum(y) / len(y), "data_end": dates[-1],
             "bars": len(dates), "config": {"K": K, "GROSS": GROSS,
             "HOLD_BARS": HOLD_BARS, "PT": PT, "SL": SL, "H": H, "NFEAT": NFEAT},
             "trained_utc": datetime.utcnow().isoformat() + "Z"}
    os.makedirs(OUT, exist_ok=True)
    json.dump(model, open(MODEL, "w"), indent=1)
    print(f"trained on {len(X)} pooled rows ({model['up_frac']:.1%} up) "
          f"over {len(dates)} bars of {len(TRADE_UNIVERSE)} symbols -> {dates[-1]}")


# ------------------------------------------------------------------ decide

def read_state():
    if os.path.exists(STATE):
        return json.load(open(STATE))
    return {"last_rerank_date": None, "held": [], "runs": 0}


def bars_since(dates, last_date):
    """Bars elapsed by TAPE INDEX, so missed cron runs and holidays cannot
    change the cadence. None (never ranked) forces a re-rank."""
    if last_date is None:
        return None
    if last_date not in dates:
        return None
    return (len(dates) - 1) - dates.index(last_date)


def rank(model, dates, bars):
    out = {}
    for sym in model["universe"]:
        closes = [b["c"] for b in bars[sym]]
        f = features(closes)
        if f is None:
            continue
        out[sym] = predict(model, f)
    return out


def select(probs, k=K, thr=THR):
    passers = sorted((s for s, p in probs.items() if p > thr),
                     key=lambda s: (-probs[s], s))
    return passers[:k]


def decide():
    model = json.load(open(MODEL))
    dates, bars = load_tape(model["universe"])
    st = read_state()
    n = bars_since(dates, st["last_rerank_date"])
    due = (n is None) or (n >= HOLD_BARS)
    probs = rank(model, dates, bars)
    plan = {"bar": dates[-1], "bars_since_rerank": n, "rerank_due": due,
            "held": list(st["held"]), "probs": {s: round(p, 4) for s, p in probs.items()}}
    plan["target"] = select(probs) if due else list(st["held"])
    return plan, model, dates, st


# ------------------------------------------------------------------ planning

def plan_orders(target, held, qmap, equity, universe, set_changed,
                k=K, gross=GROSS, dust=DUST):
    """Pure: (symbol -> legs) for this bar. THE rule that keeps live behaviour
    identical to the backtested strategy:

      * A holding is only ever RESIZED when the held SET changes. On a hold
        bar an existing position is left completely alone, however far equity
        or price has drifted. Rebalancing on hold bars is what turns hold5
        (-2.05% out of sample) back into the per-bar arm (-9.27%), so it is
        forbidden here rather than merely discouraged.
      * Exits are always allowed: a symbol outside `universe` (a dropped
        ticker such as X:SOLUSD) or one no longer selected must be closed,
        or the position is stranded forever.
      * A selected name that is NOT held at all is still established even on
        a hold bar — that is self-healing after a failed fill, not churn.
    """
    per = gross / max(1, len(target)) if target else 0.0
    plans = []
    need = sorted(set(target) | {s for s, q in held.items() if abs(q) > 0})
    for sym in need:
        price = qmap.get(sym)
        if not price:
            plans.append({"symbol": sym, "skip": "no quote"})
            continue
        cur = held.get(sym, 0.0)
        selected = sym in target and sym in universe
        if selected:
            tgt = per * equity / price
            if not sym.startswith("X:"):
                tgt = float(round(tgt))
        else:
            tgt = 0.0
        if sym not in universe:
            why = "exit: not in traded universe"
        elif tgt == 0.0:
            why = "exit: not in top-%d" % k
        elif abs(cur) > 0 and not set_changed:
            continue                      # HOLD BAR: never resize a live holding
        elif abs(cur) > 0:
            why = "resize: set changed"
        else:
            why = "enter: top-%d" % k
        if abs(tgt - cur) * price < dust:
            continue
        for side, qty in ng.map_orders(cur, tgt):
            assert side in ("buy", "sell"), f"long-only violated: {side}"
            plans.append({"symbol": sym, "side": side, "qty": qty, "quote": price,
                          "cur": cur, "tgt": tgt, "why": why})
    return plans


# ------------------------------------------------------------------ trade

def trade(dry=False, refresh=True):
    if refresh:
        refresh_tape()
    plan, model, dates, st = decide()
    c = ng.creds()
    key, aid = c.get(f"{PREFIX}_API_KEY"), c.get(f"{PREFIX}_AGENT_ID")
    if not key or not aid:
        print(f"{TICKER}: no credentials ({PREFIX}_*) — nothing done"); return
    stamp = datetime.now(ET).strftime("%Y-%m-%d")
    os.makedirs(os.path.join(OUT, "live"), exist_ok=True)
    log = {"date": stamp, "ticker": TICKER, "strategy": "hold5", "dry": dry,
           "plan": plan, "model_data_end": model["data_end"], "orders": []}
    # every run leaves an audit trail, including the ones that refuse to trade
    fn = os.path.join(OUT, "live", f"{stamp}_{TICKER}{'_dry' if dry else ''}.json")
    save = lambda: json.dump(log, open(fn, "w"), indent=1)

    me = ng.api("/me", key)
    if not dry and not (me.get("agent") or {}).get("claimed"):
        log["refused"] = "not claimed"
        save()
        print(f"{TICKER}: not claimed — no live orders -> {fn}"); return
    time.sleep(PAUSE)
    pf = ng.api(f"/me/agents/{aid}/portfolio", key)
    time.sleep(PAUSE)
    pos_r = ng.api(f"/me/agents/{aid}/positions", key)
    time.sleep(PAUSE)
    equity = pf.get("equity") or 100_000.0
    held_live = {}
    for p in (pos_r.get("positions") or pos_r.get("data") or []):
        q = float(p.get("qty", p.get("quantity", 0)))
        if str(p.get("side", "long")) == "short" and q > 0:
            q = -q
        held_live[p["symbol"]] = q
    log["equity"], log["positions_before"] = equity, dict(held_live)

    # symbols to price: the target set, everything currently held (so exits are
    # priced), and anything held outside the traded universe (SOL after the drop)
    need = sorted(set(plan["target"]) | {s for s, q in held_live.items() if abs(q) > 0})
    if not need:
        log["refused"] = "nothing held and nothing selected"
        save()
        print(f"{TICKER}: nothing held and nothing selected — no orders -> {fn}")
        return
    quotes = ng.api("/quotes?symbols=" + ",".join(s.replace(":", "%3A") for s in need), key)
    time.sleep(PAUSE)
    qraw = quotes.get("quotes") or {}
    qmap = {s: (q.get("price") if isinstance(q, dict) else None)
            for s, q in (qraw.items() if isinstance(qraw, dict) else [])}

    set_changed = set(plan["target"]) != set(st["held"])
    log["set_changed"] = set_changed
    for rec in plan_orders(plan["target"], held_live, qmap, equity,
                           model["universe"], set_changed):
        if "side" in rec:
            if not dry:
                body = {"symbol": rec["symbol"], "side": rec["side"],
                        "qty": rec["qty"],
                        "reasoning": (
                            f"napkin-hold5 {stamp}: pooled logistic model ranks "
                            f"{len(model['universe'])} symbols by P(up) from the last "
                            f"{NFEAT} log returns; hold the top {K} equal-weighted at "
                            f"{GROSS:.2f} gross, re-ranked every {HOLD_BARS} bars. "
                            f"{rec['why']}. Out-of-sample walk-forward showed this "
                            f"strategy BELOW buy-and-hold (-2.05% mean excess at 5x "
                            f"costs); deployed as a live test of that null, logged "
                            f"publicly.")}
                rec["response"] = ng.api(f"/me/agents/{aid}/orders", key, "POST", body,
                                         idem=f"napkin-hold5/{stamp}/{rec['symbol']}"
                                              f"/{rec['side']}")
                time.sleep(PAUSE)
        log["orders"].append(rec)

    # commit state only after orders are away, and only on a real re-rank
    if plan["rerank_due"] and not dry:
        st["last_rerank_date"] = plan["bar"]
        st["held"] = list(plan["target"])
        st["runs"] = st.get("runs", 0) + 1
        json.dump(st, open(STATE, "w"), indent=1)
    save()
    placed = sum(1 for o in log["orders"] if "response" in o)
    ok = sum(1 for o in log["orders"] if o.get("response", {}).get("success"))
    print(f"{TICKER}: bar {plan['bar']}, {'RE-RANK' if plan['rerank_due'] else 'hold'} "
          f"(n={plan['bars_since_rerank']}), target {plan['target']}, "
          f"{len(log['orders'])} legs, placed {placed}, ok {ok} -> {fn}")


def state():
    st = read_state()
    print(json.dumps(st, indent=1))
    if os.path.exists(MODEL):
        m = json.load(open(MODEL))
        print(f"model: {m['rows']} rows to {m['data_end']}, "
              f"{len(m['universe'])} symbols, trained {m['trained_utc'][:19]}Z")


# ------------------------------------------------------------------ selfcheck

def selfcheck():
    # 1. the traded universe drops exactly what was asked and nothing else.
    assert "X:SOLUSD" not in TRADE_UNIVERSE
    assert len(TRADE_UNIVERSE) == len(nt.UNIVERSE) - 1
    assert set(nt.UNIVERSE) - set(TRADE_UNIVERSE) == {"X:SOLUSD"}
    print(f"selfcheck 1/9: universe is {len(TRADE_UNIVERSE)} symbols, SOL removed, "
          f"nothing else changed")

    # 2. cadence by TAPE INDEX: a missed run must not shift the schedule, and
    #    a longer gap must still fire exactly once.
    dates = [f"2026-01-{i:02d}" for i in range(1, 21)]
    assert bars_since(dates, None) is None                 # never ranked -> due
    assert bars_since(dates, dates[-1]) == 0
    assert bars_since(dates, dates[-6]) == 5               # exactly due
    assert bars_since(dates, dates[-9]) == 8               # missed runs -> still due
    assert bars_since(dates, "1999-01-01") is None         # unknown -> re-rank
    for n, due in ((0, False), (4, False), (5, True), (8, True)):
        assert ((n is None) or (n >= HOLD_BARS)) == due, n
    print("selfcheck 2/9: cadence counted by tape index — due at 5, holds at 4, "
          "fires once after a missed run, unknown state re-ranks")

    # 3. selection: top-K above the gate, deterministic tie-break, gate respected.
    probs = {"A": 0.9, "B": 0.7, "C": 0.7, "D": 0.55, "E": 0.4}
    assert select(probs) == ["A", "B", "C"]                # B before C on ties (name)
    assert select({"A": 0.4, "B": 0.3}) == []              # nothing passes -> flat
    assert select({"A": 0.6}) == ["A"]                     # fewer passers than K
    print("selfcheck 3/9: top-K selection respects the gate, ties break by symbol")

    # 4. LONG ONLY: over random books, targets are never negative, so the
    #    venue mapping can only ever emit buy/sell — never short/cover.
    import random
    rng = random.Random(7)
    for _ in range(4000):
        cur = max(0.0, round(rng.uniform(-20, 60), 3))
        tgt = rng.choice([0.0, round(rng.uniform(0, 60), 3)])
        legs = ng.map_orders(cur, tgt)
        assert all(s in ("buy", "sell") for s, _ in legs), legs
        assert abs(ng.apply_legs(cur, legs) - tgt) < 1e-3
    print("selfcheck 4/9: long-only book can only produce buy/sell legs, and they "
          "reconcile to target exactly")

    # 5. sizing: K names share GROSS, stocks are whole shares, crypto fractional.
    eq, price = 100_000.0, 250.0
    per = GROSS / K
    q_stock = float(round(per * eq / price))
    assert abs(q_stock - 127.0) < 1e-9, q_stock            # .95/3*100k/250 = 126.67
    assert abs(K * per - GROSS) < 1e-12                    # fully invested when 3 held
    q_cr = per * eq / 60000.0
    assert q_cr != float(round(q_cr))                      # crypto keeps fractions
    print(f"selfcheck 5/9: {K} names x {per:.4f} = {GROSS} gross; stocks rounded "
          f"({q_stock:.0f} sh), crypto fractional")

    # 6. EXITS. A dropped symbol and a stale DQN holding must both be closed:
    #    anything held that is not in the target set gets target 0.
    universe, target = TRADE_UNIVERSE, ["AAPL", "MSFT", "X:BTCUSD"]
    live = {"X:SOLUSD": 3.0, "COST": 40.0, "AAPL": 10.0}
    need = sorted(set(target) | {s for s, q in live.items() if abs(q) > 0})
    exits = [s for s in need if s not in target]
    assert "X:SOLUSD" in exits and "COST" in exits and "AAPL" not in exits
    assert all(s in need for s in target)
    print("selfcheck 6/9: dropped SOL and stale non-target holdings are both priced "
          "and closed; kept names are not")

    # 7. label safety: no training event may use a barrier that resolves after
    #    the data cutoff (the live analogue of the leak guard).
    bars = [{"o": 100, "h": 101, "l": 99, "c": 100 + math.sin(i / 3),
             "v": 1e6} for i in range(80)]
    dts = [f"d{i}" for i in range(80)]
    X, y = build_pool(dts, {"S": bars}, upto=60)
    assert len(X) > 0
    Xf, _ = build_pool(dts, {"S": bars}, upto=79)
    assert len(Xf) > len(X), "cutoff had no effect — leak guard is not binding"
    print(f"selfcheck 7/9: training honours the data cutoff ({len(X)} rows to bar 60 "
          f"vs {len(Xf)} to bar 79)")

    # 8. determinism of the fit and of ranking.
    Xs = [[math.sin(i * 1.7 + j) for j in range(NFEAT)] for i in range(120)]
    ys = [1 if math.sin(i * 2.3) > 0 else 0 for i in range(120)]
    mu, sd = zstats(Xs)
    a = fit_logit([zapply(x, mu, sd) for x in Xs], ys)
    b = fit_logit([zapply(x, mu, sd) for x in Xs], ys)
    assert a == b
    m = {"w": a[0], "b": a[1], "mu": mu, "sd": sd}
    assert predict(m, Xs[0]) == predict(m, Xs[0])
    print("selfcheck 8/9: fit and prediction byte-deterministic (zero init, "
          "fixed epochs, no seeds)")

    # 9. THE HOLD-BAR RULE, through the shipped planner. On a hold bar an
    #    existing holding must not be resized even when equity has moved a
    #    long way; exits must still happen; a missing selected name is still
    #    established (self-healing after a failed fill).
    uni, tgt3 = TRADE_UNIVERSE, ["AAPL", "MSFT", "X:BTCUSD"]
    qm = {"AAPL": 200.0, "MSFT": 400.0, "X:BTCUSD": 60000.0, "COST": 900.0,
          "X:SOLUSD": 150.0}
    live = {"AAPL": 158.0, "MSFT": 79.0, "X:BTCUSD": 0.52,
            "COST": 40.0, "X:SOLUSD": 3.0}
    hold = plan_orders(tgt3, live, qm, 200_000.0, uni, set_changed=False)
    syms = {p["symbol"] for p in hold if "side" in p}
    assert syms == {"COST", "X:SOLUSD"}, syms          # equity DOUBLED: still no resize
    assert all(p["side"] == "sell" for p in hold if "side" in p)
    moved = plan_orders(tgt3, live, qm, 200_000.0, uni, set_changed=True)
    assert {p["symbol"] for p in moved if "side" in p} >= {"AAPL", "COST", "X:SOLUSD"}
    gone = plan_orders(tgt3, {k2: v for k2, v in live.items() if k2 != "MSFT"},
                       qm, 100_000.0, uni, set_changed=False)
    assert any(p["symbol"] == "MSFT" and p["side"] == "buy" for p in gone), \
        "failed fill was not re-established"
    flat = plan_orders([], live, qm, 100_000.0, uni, set_changed=True)
    assert all(p["side"] == "sell" for p in flat if "side" in p)
    assert {p["symbol"] for p in flat if "side" in p} == set(live)
    print("selfcheck 9/9: hold bars never resize a live holding (even on doubled "
          "equity), always exit dropped names, and re-establish a missing one")
    print("ALL SELFCHECKS PASS")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "selfcheck"
    if cmd == "trade":
        trade(dry="--dry" in sys.argv, refresh="--no-refresh" not in sys.argv)
    elif cmd == "train":
        train(refresh="--no-refresh" not in sys.argv)
    elif cmd == "decide":
        print(json.dumps(decide()[0], indent=1))
    else:
        {"selfcheck": selfcheck, "state": state}[cmd]()
