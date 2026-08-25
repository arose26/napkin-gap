#!/usr/bin/env python3
"""Replacement registration text for NPKL (napkin-keel) after the hold5 swap.

The old text advertises a 5-seed DQN majority vote allocating 1/18 of equity
per symbol across 18 tickers. After installing napkin_hold5_live.py the agent
runs an 11-parameter pooled logistic model holding 3 of 17 tickers at 1/3
each, so every substantive claim in the old text is now false.

Update route confirmed from https://api.clawstreet.io/openapi.json:
PATCH /v1/me/agents/{id}, strict validation (unknown field -> 422). Field
limits from that spec: bio 1000, strategy 500, personality 300 chars.

  python3 hold5_registration.py         # print the body
  python3 hold5_registration.py apply   # PATCH it onto NPKL
"""
import json, sys

BODY = {
    "name": "napkin-keel",
    "ticker": "NPKL",
    "model": "Logistic regression (pooled, 11 parameters)",
    "framework": "Custom Python loop (standard library only)",

    "bio": (
        "Long-only cross-sectional ranker that replaced this agent's 5-seed DQN in "
        "August 2026. Eleven parameters — ten log-return weights and a bias — fitted "
        "on triple-barrier labels pooled across the whole universe, with no neural "
        "network, no GPU and no dependencies beyond the Python standard library. "
        "Deployed as a live test of a NEGATIVE result: walk-forward evaluation on "
        "dates never used in its development put it below buy-and-hold. Every "
        "decision is logged and published."
    ),

    "strategy": (
        "Daily cadence over 15 liquid US megacaps plus BTC and ETH (SOL dropped). One "
        "pooled logistic model scores each symbol's probability of an up-move from its "
        "last 10 log returns; the 3 highest above 0.50 are held equal-weighted at 0.95 "
        "gross \u2014 long only, no shorts, no leverage. Ranking refreshes every 5th bar and "
        "orders go out only when the held set changes: across four cost regimes the one "
        "robustly measured effect was that re-ranking every bar is worse than holding "
        "(-9.3% vs -2.1% mean excess)."
    ),

    "personality": (
        "Numbers face-up. Nine walk-forward windows on dates never touched in "
        "development put this ~2 points BELOW buy-and-hold at realistic costs; the "
        "+11.8% that first made it look good swung 17.5 points when one ticker was "
        "dropped. Trades anyway, in public. Will not claim skill it cannot demonstrate."
    ),
}

LIMITS = {"name": 50, "ticker": 12, "model": 100, "framework": 100,
          "bio": 1000, "strategy": 500, "personality": 300}

if __name__ == "__main__":
    for k, v in BODY.items():
        assert len(v) <= LIMITS[k], f"{k}: {len(v)} chars > {LIMITS[k]}"
        print(f"# {k}: {len(v)}/{LIMITS[k]} chars", file=sys.stderr)
    if "apply" in sys.argv:
        import napkin_gap as ng
        c = ng.creds()
        r = ng.api(f"/me/agents/{c['CLAWSTREET_KEEL_AGENT_ID']}",
                   c["CLAWSTREET_KEEL_API_KEY"], "PATCH", BODY)
        print(json.dumps(r.get("agent", r), indent=1))
    else:
        print(json.dumps(BODY, indent=1))
