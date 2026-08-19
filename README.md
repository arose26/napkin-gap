# napkin-gap

Repo 4 of the **napkin-trader series** ([napkin-tape](https://github.com/arose26/napkin-tape)
→ [napkin-eyes](https://github.com/arose26/napkin-eyes) →
[napkin-trader](https://github.com/arose26/napkin-trader) → this → napkin-wallstreet).
Repos 1–3 built a calibrated replay sim and trained agents in it, with every claim ending
in "…on the sim." This repo deploys those agents to
[ClawStreet](https://www.clawstreet.io)'s live paper-trading arena and asks the only
question that decides whether the whole series means anything:

> **Does the sim survive contact with reality? Measured per fill — realized vs
> sim-predicted prices, cost-model error, PnL divergence over matched decision logs —
> not vibes.**

## The deployment

Two agents, both disclosed on their public profiles:

- **napkin-trader (NPKN)** — the `base` arm from repo 3: {short, flat, long} per symbol,
  raw-PnL objective. The board-chaser profile.
- **napkin-keel (NPKL)** — the `long2` arm: {flat, long} only. The structural-risk
  flagship profile (repo 3 showed reward *shaping* bought nothing measurable; a policy
  that cannot be short is a risk story enforced by the action space, not the reward).

Each agent is a **majority-vote ensemble of 5 seeds** (repo 2 measured a −9%..+6%
single-seed spread on a season window; deploying one seed would be entering a lottery,
and the ensemble is disclosed). Nets are retrained fresh on all bulk data up to deploy
date — walk-forward consistent: live *is* the test set. Ties in the vote hold the current
position.

**The daily loop** (cron, 9:31 ET, trading days): refresh the bulk tape → compute
yesterday-close features → greedy ensemble vote per symbol → diff against live positions
→ market orders (venue-mapped sides: buy/sell for longs, short/cover for shorts) → log
the full decision record (features hash, votes, targets, order responses) → run the SAME
decisions through the napkin-tape sim → store both. Reconcile against `GET /portfolio`
after fills.

**Known deviations, logged not hidden:** (1) live rebalances only on action change, not
the sim's daily drift re-target (avoids dust-order spam on a public feed). For longs the
two semantics coincide (selfcheck: 0.0003% over 40 bars — accounting is exact); for
*held shorts* they genuinely diverge (variance drag of a daily-rebalanced short vs a held
short) — measured at 1.4% over a 40-bar stretch on the deploy ensemble. NPKL never
shorts, so this applies to NPKN only;
(2) crypto "opens" are UTC-midnight bars but orders fill at 9:31 ET (~6.5 h later) —
the gap analysis buckets crypto separately; (3) day-one orders filled mid-session, not
at open (timestamped, excluded from open-fill stats).

## Registered measurement targets (2026-08-19, before the first live fill)

These are predictions about the *gap*, the thing this repo exists to measure:

1. **Fill-price error**: median |realized fill − sim-predicted fill| < **20 bps** for
   stocks at our sizes (venue publishes its own execution formula; the residual should
   be spread + timing, and our megacaps are tight). Crypto bucketed separately —
   predicted worse (deviation 2 above): < 60 bps.
2. **Cost-model error**: realized per-fill `slippage_bps` + commission within **2×** of
   the formula's prediction for ≥ 80% of fills (we mirror the venue's own published
   numbers, so big misses mean the formula description is wrong — worth knowing).
3. **PnL divergence**: after 10 trading days, |live equity − matched-sim equity| < **1%**
   of starting equity per agent.
4. **Rank order** (the honest question): whether NPKN-vs-NPKL relative ranking is
   preserved between matched sim and live. Registered as an **open question, not a
   prediction** — repo 2/3 showed differences this size are seed noise on this window;
   we expect the *sign* of the gap to be informative about execution, not about which
   agent is "better".

## Selfchecks (`python3 napkin_gap.py selfcheck`)

- **Substitution trick on the actual deploy nets**: ensemble decisions replayed through
  napkin-tape's independent accounting.
- **Order-side mapping**: property-tested over signed current/target positions — never
  oversell, never overcover, long2 can never emit `short`, two-leg flips ordered
  close-then-open.
- **Dry-run coherence**: `trade --dry` produces a decision log byte-identical to
  `decide` on the same inputs, and places zero orders.
- **Reconciliation identity**: synthetic fill responses reproduce the local position
  book to the share.

## Results

*(gap statistics accumulate daily in `out/live/`; analysis lands here after ~2–4 weeks
of paper trading — the series' registered calendar bottleneck)*

## Run it

```bash
python3 napkin_gap.py train        # retrain 2 arms x 5 seeds on all data to date
python3 napkin_gap.py selfcheck
python3 napkin_gap.py trade --dry  # full pipeline, no orders
python3 napkin_gap.py trade        # the real thing (cron runs this 9:31 ET weekdays)
python3 napkin_gap.py gap          # accumulated sim-vs-real report
```

## What's deliberately not here

No new learning machinery (nets and env are repos 2–3's, imported); no intraday
decisions (cadence matches training, daily); no automatic recalibration loop yet —
the doc says *fit* the cost model from real fills, and that fit happens once there are
enough fills to fit to (single manual `gap --refit` pass, not a daemon).
