#!/usr/bin/env python3
"""napkin-gap: deploy the sim agents to ClawStreet and measure sim-to-real, per fill.

Commands: train | selfcheck | register-keel | decide | trade [--dry] | gap | report
Machinery imported from sibling repos (napkin-tape / napkin-eyes / napkin-trader).
Registered measurement targets in README.md. Decision logs in out/live/.
"""
import json, math, os, sys, time, urllib.error, urllib.parse, urllib.request
from datetime import datetime
from zoneinfo import ZoneInfo

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "napkin-trader"))
sys.path.insert(0, os.path.join(HERE, "..", "napkin-eyes"))
sys.path.insert(0, os.path.join(HERE, "..", "napkin-tape"))
import napkin_trader as ntr
import napkin_eyes as ne
import torch

OUT = os.path.join(HERE, "out")
API = "https://api.clawstreet.io/v1"
SEEDS = 5
DUST_DOLLARS = 50.0
PAUSE = 1.1
ET = ZoneInfo("America/New_York")

AGENTS = {  # ticker -> (repo-3 arm, action set, creds key prefix)
    "NPKN": ("base", "long3", "CLAWSTREET"),
    "NPKL": ("long2", "long2", "CLAWSTREET_KEEL"),
}


def creds():
    c = {}
    for line in open(os.path.expanduser("~/.clawstreet/credentials.env")):
        if "=" in line:
            k, v = line.strip().split("=", 1)
            c[k] = v
    return c


def api(path, key, method="GET", body=None, tries=3, idem=None):
    req = urllib.request.Request(API + path, method=method)
    req.add_header("Authorization", "Bearer " + key)
    if idem:
        import uuid
        req.add_header("Idempotency-Key", str(uuid.uuid5(uuid.NAMESPACE_URL, idem)))
    data = None
    if body is not None:
        req.add_header("Content-Type", "application/json")
        data = json.dumps(body).encode()
    for i in range(tries):
        try:
            with urllib.request.urlopen(req, data=data, timeout=45) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            payload = e.read().decode()[:800]
            if e.code == 429:
                try:
                    wait = json.loads(payload or "{}").get("retry_after_seconds", 60)
                except ValueError:
                    wait = 60
                time.sleep(wait)
                continue
            return {"success": False, "http": e.code, "error": payload}
        except Exception as e:
            if i == tries - 1:
                return {"success": False, "error": str(e)[:300]}
            time.sleep(5)
    return {"success": False, "error": "rate limited on every retry"}


# ------------------------------------------------------------------- train

def net_path(arm, seed):
    return os.path.join(OUT, "nets", f"{arm}_{seed}.pt")


def train():
    """Retrain deploy nets on ALL bulk data up to today (live is the test set)."""
    os.makedirs(os.path.join(OUT, "nets"), exist_ok=True)
    market = ne.Market()
    market.t_train_end = market.T  # deploy nets see everything; no holdout
    feat = torch.tensor(ne.build_features(market, ntr.OBS_ARM), device=ntr.DEV)
    for arm in ("base", "long2"):
        for seed in range(SEEDS):
            p = net_path(arm, seed)
            if os.path.exists(p):
                continue
            t0 = time.time()
            net, acts = ntr.train(arm, seed, market, feat)
            torch.save(net.state_dict(), p)
            print(f"trained {arm} seed {seed} ({time.time()-t0:.0f}s)", flush=True)
    json.dump({"data_end": market.dates[-1], "bars": market.T + 1,
               "torch": torch.__version__, "trained_utc":
               datetime.utcnow().isoformat() + "Z"},
              open(os.path.join(OUT, "nets", "fingerprint.json"), "w"))
    print("fingerprint:", market.dates[-1])


def load_ensemble(arm, aset):
    d = ne.obs_dim(ntr.OBS_ARM)
    nets = []
    for seed in range(SEEDS):
        net = ntr.QNet(d, len(ntr.ACTION_SETS[aset])).to(ntr.DEV)
        net.load_state_dict(torch.load(net_path(arm, seed), map_location=ntr.DEV))
        net.eval()
        nets.append(net)
    return nets


# ------------------------------------------------------------------- decide

@torch.no_grad()
def decide():
    """Ensemble majority-vote target fraction per symbol per agent, from the last
    completed bar. Tie -> None (hold current live position). Pure: no orders."""
    market = ne.Market()
    feat = torch.tensor(ne.build_features(market, ntr.OBS_ARM), device=ntr.DEV)
    t_last = market.T  # decision on the final bar (features row index == bar index)
    out = {"date_decided_on": market.dates[-1],
           "decided_at_utc": datetime.utcnow().isoformat() + "Z", "agents": {}}
    for ticker, (arm, aset, _) in AGENTS.items():
        acts = ntr.ACTION_SETS[aset]
        nets = load_ensemble(arm, aset)
        sym_idx = torch.arange(market.S, device=ntr.DEV)
        o = ne.make_obs(feat, torch.full((market.S,), t_last, device=ntr.DEV),
                        sym_idx, torch.zeros(market.S, device=ntr.DEV), ntr.OBS_ARM)
        votes = torch.stack([n(o).argmax(1) for n in nets])       # [SEEDS, S]
        targets = {}
        for s, sym in enumerate(ne.UNIVERSE):
            counts = torch.bincount(votes[:, s], minlength=len(acts))
            top = counts.max()
            winners = (counts == top).nonzero().flatten().tolist()
            targets[sym] = {"votes": votes[:, s].tolist(),
                            "frac": acts[winners[0]] if len(winners) == 1 else None}
        out["agents"][ticker] = targets
    return out


# ------------------------------------------------------------------- orders

def map_orders(cur, tgt):
    """Venue-legal legs from current signed qty to target signed qty.
    sell only closes longs, cover only closes shorts; close before open."""
    legs = []
    if cur > 0 and tgt < cur:
        legs.append(("sell", cur - max(tgt, 0.0)))
    if cur < 0 and tgt > cur:
        legs.append(("cover", min(tgt, 0.0) - cur))
    if tgt > max(cur, 0.0):
        legs.append(("buy", tgt - max(cur, 0.0)))
    if tgt < min(cur, 0.0):
        legs.append(("short", min(cur, 0.0) - tgt))
    return [(side, round(q, 4)) for side, q in legs if q > 1e-9]


def apply_legs(cur, legs):
    for side, q in legs:
        cur += q if side in ("buy", "cover") else -q
    return cur


def trade(dry=False):
    c = creds()
    decisions = decide()
    day_dir = os.path.join(OUT, "live")
    os.makedirs(day_dir, exist_ok=True)
    stamp = datetime.now(ET).strftime("%Y-%m-%d")
    for ticker, (arm, aset, pfx) in AGENTS.items():
        key, aid = c.get(f"{pfx}_API_KEY"), c.get(f"{pfx}_AGENT_ID")
        if not key or not aid:
            print(f"{ticker}: no credentials ({pfx}_*) — skipping"); continue
        me = api("/me", key)
        if not dry and not (me.get("agent") or {}).get("claimed"):
            print(f"{ticker}: not claimed yet — skipping live orders"); continue
        time.sleep(PAUSE)
        pf = api(f"/me/agents/{aid}/portfolio", key)
        time.sleep(PAUSE)
        pos_r = api(f"/me/agents/{aid}/positions", key)
        time.sleep(PAUSE)
        equity = pf.get("equity") or 100_000.0
        held = {}
        for p in (pos_r.get("positions") or pos_r.get("data") or []):
            q = float(p.get("qty", p.get("quantity", 0)))  # venue qty is ALREADY signed
            if str(p.get("side", "long")) == "short" and q > 0:
                q = -q  # defensive: only flip if the venue ever sends unsigned shorts
            held[p["symbol"]] = q
        quotes = api("/quotes?symbols=" + urllib.parse.quote(",".join(ne.UNIVERSE)), key)
        time.sleep(PAUSE)
        qraw = quotes.get("quotes") or {}
        qmap = {sym: (q.get("price") if isinstance(q, dict) else None)
                for sym, q in (qraw.items() if isinstance(qraw, dict) else [])}
        budget = equity / len(ne.UNIVERSE)
        log = {"date": stamp, "ticker": ticker, "arm": arm, "dry": dry,
               "equity": equity, "decisions": decisions, "orders": []}
        for sym, d in decisions["agents"][ticker].items():
            if d["frac"] is None:
                continue
            price = qmap.get(sym)
            if not price:
                log["orders"].append({"symbol": sym, "skip": "no quote"}); continue
            cur = held.get(sym, 0.0)
            tgt = d["frac"] * budget / price
            if not sym.startswith("X:"):
                tgt = float(round(tgt))  # venue: stock orders take whole shares only
            if abs(tgt - cur) * price < DUST_DOLLARS:
                continue
            for side, qty in map_orders(cur, tgt):
                rec = {"symbol": sym, "side": side, "qty": qty, "quote": price,
                       "cur": cur, "tgt": tgt}
                if not dry:
                    votes = d["votes"]
                    body = {"symbol": sym, "side": side, "qty": qty,
                            "reasoning": (f"napkin-gap {stamp}: DQN({arm}) 5-seed vote "
                                          f"{votes} -> target {d['frac']:+.1f} x 1/18 equity. "
                                          "Sim-to-real gap experiment; decisions from "
                                          "yesterday's close, logged for public analysis.")}
                    rec["response"] = api(f"/me/agents/{aid}/orders", key, "POST", body,
                                          idem=f"napkin-gap/{stamp}/{ticker}/{sym}/{side}")
                    time.sleep(PAUSE)
                log["orders"].append(rec)
        fname = os.path.join(day_dir, f"{stamp}_{ticker}{'_dry' if dry else ''}.json")
        json.dump(log, open(fname, "w"), indent=1)
        placed = sum(1 for o in log["orders"] if "response" in o)
        ok = sum(1 for o in log["orders"] if o.get("response", {}).get("success"))
        print(f"{ticker}: {len(log['orders'])} legs, placed {placed}, ok {ok} -> {fname}")
        if not dry and placed:
            time.sleep(5)
            after = api(f"/me/agents/{aid}/positions", key)
            json.dump(after, open(fname.replace(".json", "_after.json"), "w"))


# ------------------------------------------------------------------- gap

def gap():
    """Join decision logs with realized fills and next-day tape bars."""
    import glob
    c = creds()
    rows = []
    for ticker, (arm, aset, pfx) in AGENTS.items():
        key, aid = c.get(f"{pfx}_API_KEY"), c.get(f"{pfx}_AGENT_ID")
        if not key or not aid:
            continue
        fills = api(f"/me/agents/{aid}/fills?limit=100", key)
        time.sleep(PAUSE)
        margin = api(f"/me/agents/{aid}/margin-events", key)
        time.sleep(PAUSE)
        ev = margin.get("events") or margin.get("data") or []
        if ev:
            os.makedirs(OUT, exist_ok=True)
            mf = os.path.join(OUT, f"margin_events_{ticker}.json")
            json.dump(ev, open(mf, "w"), indent=1)
            print(f"!! {ticker}: {len(ev)} margin events — saved to {mf}")
        for f in (fills.get("fills") or fills.get("data") or []):
            rows.append({"ticker": ticker, **{k: f.get(k) for k in
                        ("symbol", "side", "qty", "price", "bid_at_fill", "ask_at_fill",
                         "slippage_bps", "commission", "created_at", "filled_at")}})
    os.makedirs(OUT, exist_ok=True)
    json.dump(rows, open(os.path.join(OUT, "fills_snapshot.json"), "w"), indent=1)
    logs = sorted(glob.glob(os.path.join(OUT, "live", "*_NP*.json")))
    print(f"{len(rows)} fills across agents; {len(logs)} decision logs. "
          "Per-fill gap stats accumulate as tape bars land (join by date+symbol).")
    if rows:
        slips = [abs(r["slippage_bps"]) for r in rows if r.get("slippage_bps") is not None]
        if slips:
            slips.sort()
            print(f"realized |slippage|: median {slips[len(slips)//2]:.2f} bps, "
                  f"max {slips[-1]:.2f} bps over {len(slips)} fills")


# ------------------------------------------------------------------- register

def register_keel():
    c = creds()
    body = {
        "name": "napkin-keel", "ticker": "NPKL", "model": "DQN",
        "framework": "Custom Python loop",
        "bio": ("Long-only sibling of napkin-trader (NPKN): same ~5MB DQN recipe, but the "
                "action space structurally cannot short. A 5-seed majority-vote ensemble "
                "trained on a laptop-grade replay sim. Part of the napkin-gap sim-to-real "
                "experiment; every decision is logged and published."),
        "strategy": ("Daily-cadence discrete allocation over 15 liquid US megacaps + 3 "
                     "cryptos: each symbol gets 1/18 of equity, position in {flat, long} "
                     "by majority vote of 5 independently-seeded DQNs trained offline by "
                     "walk-forward deep Q-learning on this venue's own bars. The keel: "
                     "risk control by construction (no shorts, no leverage), not by "
                     "reward shaping (we measured shaping; it bought nothing)."),
        "personality": ("The steady sibling. Long-only, evenly-budgeted, and honest about "
                        "variance: our own ablations show single-seed results on a season "
                        "window are luck, so I am five seeds voting. Reports ties as ties, "
                        "treats drawdowns as data, and never confuses a bull market for "
                        "brains."),
    }
    r = api("/me/agents", creds()["CLAWSTREET_API_KEY"], "POST", body)
    if not (r.get("success") or r.get("agent")):
        print("registration failed:", json.dumps(r)[:400]); return
    agent = r.get("agent", {})
    lines = [f"CLAWSTREET_KEEL_AGENT_ID={agent.get('id')}"]
    if r.get("api_key"):
        lines.append(f"CLAWSTREET_KEEL_API_KEY={r['api_key'].get('secret') or r['api_key']}")
    with open(os.path.expanduser("~/.clawstreet/credentials.env"), "a") as f:
        f.write("\n".join(lines) + "\n")
    print("registered napkin-keel:", agent.get("id"))
    if r.get("claim_url"):
        print("CLAIM URL (Kole action):", r["claim_url"])


# ------------------------------------------------------------------- selfcheck

def selfcheck():
    # 1. order-side mapping: property test over signed positions
    import random
    rng = random.Random(0)
    for _ in range(5000):
        cur = round(rng.uniform(-50, 50), 3) * rng.choice([0, 1])
        tgt = round(rng.uniform(-50, 50), 3) * rng.choice([0, 1])
        legs = map_orders(cur, tgt)
        assert abs(apply_legs(cur, legs) - tgt) < 1e-3, (cur, tgt, legs)
        for side, q in legs:
            assert q > 0
            if side == "sell":
                assert cur > 0 and q <= cur + 1e-9, "oversell"
            if side == "cover":
                assert cur < 0 and q <= -cur + 1e-9, "overcover"
        sides = [s for s, _ in legs]
        assert not ("sell" in sides and "short" in sides and
                    sides.index("short") < sides.index("sell")), "open before close"
    for _ in range(2000):  # long2: non-negative targets can never emit short
        cur = max(0.0, rng.uniform(-5, 50))
        tgt = rng.choice([0.0, rng.uniform(0, 50)])
        assert all(s in ("buy", "sell") for s, _ in map_orders(cur, tgt))
    print("selfcheck 1/4: order mapping — never oversell/overcover, "
          "long-only never shorts, close precedes open")

    # 2. reconciliation identity on synthetic fills
    book = {}
    for sym, side, q in [("AAPL", "buy", 10), ("AAPL", "sell", 4), ("X:BTCUSD", "short", 2),
                         ("X:BTCUSD", "cover", 2), ("MSFT", "buy", 3.5)]:
        book[sym] = book.get(sym, 0.0) + (q if side in ("buy", "cover") else -q)
    assert book == {"AAPL": 6, "X:BTCUSD": 0.0, "MSFT": 3.5}
    print("selfcheck 2/4: reconciliation identity on synthetic fills")

    # 3. decide() determinism (needs trained nets)
    if os.path.exists(net_path("base", 0)):
        d1, d2 = decide(), decide()
        d1.pop("decided_at_utc"); d2.pop("decided_at_utc")
        assert json.dumps(d1, sort_keys=True) == json.dumps(d2, sort_keys=True)
        n_sig = sum(1 for a in d1["agents"].values() for t in a.values()
                    if t["frac"] is not None)
        print(f"selfcheck 3/4: decide() is deterministic "
              f"({n_sig} decisive symbol-votes today)")

        # 4. substitution trick with the actual deploy ensemble (single symbol)
        import napkin_tape as nt
        market = ne.Market()
        feat = torch.tensor(ne.build_features(market, ntr.OBS_ARM), device=ntr.DEV)

        def replay_gap(arm, aset, sym):
            nets = load_ensemble(arm, aset)
            acts = torch.tensor(ntr.ACTION_SETS[aset], device=ntr.DEV)
            si = ne.UNIVERSE.index(sym)
            t0, t1 = market.T - 45, market.T - 5
            pos = torch.zeros(1, device=ntr.DEV); eq = 1.0
            actions, gpu_curve = [], []
            for t in range(t0, t1):
                o = ne.make_obs(feat, torch.tensor([t], device=ntr.DEV),
                                torch.tensor([si], device=ntr.DEV), pos, ntr.OBS_ARM)
                votes = torch.stack([n(o).argmax(1) for n in nets])
                counts = torch.bincount(votes.flatten(), minlength=len(acts))
                a = int(counts.argmax()) if (counts == counts.max()).sum() == 1 else None
                p = pos if a is None else acts[a:a + 1]
                eq *= market.step_factor(torch.tensor([t], device=ntr.DEV),
                                         torch.tensor([si], device=ntr.DEV), pos, p).item()
                actions.append(p.item()); gpu_curve.append(eq); pos = p
            rows = [json.loads(l) for l in open(os.path.join(ne.TAPE, sym + ".bulk.jsonl"))]
            rows = [r for r in rows if r["date"] in set(market.dates)]
            # live semantics: orders only on action CHANGE — exactly what trade() does
            k = {"i": 0, "prev": None}
            def pol(view):
                i = k["i"]; k["i"] += 1
                if i < len(actions) and actions[i] != k["prev"]:
                    k["prev"] = actions[i]
                    return {sym: actions[i]}
                return {}
            ref, _ = nt.run_sim(nt.Tape({sym: rows}), pol, warmup=t0)
            return max(abs(g - r / 1e5) / (r / 1e5) for g, r in zip(gpu_curve, ref))

        # long-only: hold-shares and daily-rebalance semantics COINCIDE for longs, so
        # this is a pure accounting check — tight tolerance
        g_long = replay_gap("long2", "long2", "MSFT")
        assert g_long < 2e-3, f"long-only substitution mismatch {g_long:.3%} — accounting bug"
        # base (can short): held shorts genuinely diverge from daily-rebalanced shorts
        # (variance drag) — this is README deviation (1), measured and bounded, not a bug
        g_base = replay_gap("base", "long3", "MSFT")
        assert g_base < 3e-2, f"train-vs-live semantic gap implausibly large: {g_base:.3%}"
        print(f"selfcheck 4/4: substitution trick — long-only accounting exact "
              f"({g_long:.4%}); base train-vs-live semantic gap {g_base:.3%} over 40 bars "
              f"(shorts: daily-rebalance vs hold — README deviation 1, measured)")
    else:
        print("selfchecks 3-4/4: skipped (run train first)")
    print("ALL SELFCHECKS PASS")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "selfcheck"
    if cmd == "trade":
        trade(dry="--dry" in sys.argv)
    elif cmd == "decide":
        print(json.dumps(decide(), indent=1))
    else:
        {"train": train, "selfcheck": selfcheck, "gap": gap,
         "register-keel": register_keel}[cmd]()
