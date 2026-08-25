# HANDOFF — napkin T10 series + hold5 live deployment

Written 2026-08-25 by a **cloud** Claude Code session, for a **local** session
that has access to the live trading host (`/home/bob/napkin-gap`). The cloud
session could not reach that host or attach `arose26/napkin-gap`, which is the
main reason to switch.

## 1. What state everything is in

**`arose26/napkin-labels`** (created this session, pushed, 20 commits) — the
whole T10 supervised-vs-RL research series. Experiments #1-#6, #8-#11, #13 plus
bonus #B1-#B6, each with predictions registered BEFORE running and refutations
published intact. `RESULTS.md` is the run log; read it before running anything
new. Needs `arose26/napkin-tape` cloned as a **sibling directory**
(`../napkin-tape`) and `python3 napkin_tape.py bulk` run once.

**`arose26/napkin-tutorials`** — cleaned of experiment code this session; both
branches sit on `main`. Nothing pending.

**`arose26/napkin-gap`** (the live bot, host `/home/bob/napkin-gap`) —
NPKL/keel swapped from the 5-seed DQN to the `hold5` strategy. Installed and
running on cron; see section 3.

## 2. The research verdict (do not relitigate — it is settled and logged)

Across 6-15 walk-forward windows, **no arm in this series beats buy-and-hold
with any consistency.** Every apparent edge was killed by a control registered
before it ran:

| claim | fate |
|---|---|
| #B1 chop-gate "series best" +17.71% | #B2 placebo: rank 2 of 10 vs rotated schedules |
| #B3/#B4 cross-sectional ranking | real, but costs erase it by 5x (156x turnover) |
| #B5 "standing bar broken", hold5 +11.83% @5x | #B6 pre-registered replication: **-2.05%**, t -0.32, 4/9 fresh windows |

Load-bearing numbers a successor should not have to re-derive:
- **Effective sample size is ~2.9k**, not 12.5k rows (#9: mean uniqueness 0.227,
  independently confirmed by a disjoint subsample keeping 2944).
- **One-window noise floor is ~11.5pp** (#B2 placebo spread). Nothing in the log
  clears it.
- **Cost law**: ~0.0122-0.0142pp of drag per unit of turnover per cost multiple,
  replicated across two independent experiments (#8, #B4).
- **#B5's one surviving finding**: re-ranking every bar is *worse* than holding 5
  (-9.27% vs -2.05%) — "trade less" is real but RELATIVE. It makes a losing arm
  lose less; it does not make it win.
- **#B6's decisive diagnostic**: dropping one ticker (SOL) and re-aligning bars
  swung the same arm from +11.83% to -5.69% on the same dates. A 17.5pp swing
  from a trivial universe change.
- **#13's "bear protection" does not generalize**: crypto winter -27.4pp, 2022
  bear -6.8pp, COVID -18.0pp.

Standing discipline (in `napkin-labels/README.md` and `HANDOFF.md`): register
predictions first; report across `napkin_windows` walk-forward windows, never one
window; placebo/permutation nulls next to buy-and-hold; publish nulls.

## 3. The live deployment — what was done, on operator instruction

The operator directed deploying hold5 live **with the negative record on file**.
That decision is theirs and is recorded; do not re-argue it, but do not soften
the numbers either (the venue profile text in section 5 states them).

**Configuration**: pooled logistic regression (11 params: 10 log-return weights
+ bias), triple-barrier labels (pt=sl=2*vol, h=10), rank all symbols by P(up),
gate 0.5, hold top **3** equal-weighted at **0.95 gross**, re-rank every **5
bars**, trade only when the held SET changes, long-only, **17 symbols (X:SOLUSD
removed at operator request)**.

**Delivered / installed:**
- `napkin_hold5_live.py` — self-contained module in `/home/bob/napkin-gap/`.
  Pure stdlib (no torch/numpy). 9/9 selfchecks pass. Trades NPKL only.
- `napkin_gap.py` patched: the `"NPKL": (...)` line deleted from `AGENTS`, so the
  DQN no longer trades that agent. Confirmed by `out/trade.log` showing NPKN only.
- Cron (verified firing 2026-08-24 and 08-25):
  ```
  31 9 * * 1-5  napkin_gap.py trade         >> out/trade.log
  36 9 * * 1-5  napkin_hold5_live.py trade  >> out/hold5.log
  27 9 * * 6    napkin_hold5_live.py train  >> out/hold5_train.log   # weekly refit
  ```
- `hold5_registration.py` — replacement NPKL venue profile. **NOT yet applied.**

**Deliberate design choices (do not "simplify" these away):**
- **Own tape** under `out/hold5/tape/`. `napkin_tape.bulk()` SKIPS any symbol
  with >=1800 bars, so it never advances an established tape; this module
  upserts Yahoo/Coinbase bars into its own copy and never writes napkin_gap's
  shared files, so NPKN is bit-for-bit unaffected.
- **State on disk** (`out/hold5/state.json`): last re-rank BAR DATE + intended
  set. Bars counted by tape index, so a missed cron run or holiday cannot
  silently change the cadence.
- **Explicit universe exits**: anything held outside the traded 17 (i.e. SOL, and
  the legacy DQN book) is closed. Without this, dropping a symbol strands its
  position forever.
- **Hold bars never resize.** `plan_orders()` is pure and refuses to resize a
  live holding unless the held SET changed — rebalancing on hold bars is exactly
  what turns hold5 (-2.05%) back into the per-bar arm (-9.27%). Selfcheck 9
  proves it at doubled equity. It still exits dropped names and re-establishes a
  position after a failed fill.

## 4. OPEN ITEMS — start here

*Updated 2026-08-25 by the local session. Items 2, 3, 4 are DONE; only the
claim remains, and it needs Kole, not code.*

1. **BLOCKER, still open: NPKL is not claimed on the venue.** Confirmed again
   today: `GET /me` returns `claimed: false` for agent
   `7c28d38f-5882-4377-bea0-c6b83aa49d39` (napkin-keel), while NPKN
   (`d67f2f97-…`) returns `claimed: true` on the same call with its own key.
   The venue's OpenAPI spec (https://api.clawstreet.io/openapi.json) shows
   `claim_url` is returned **only** by `POST /v1/me/agents` at creation — there
   is no re-issue endpoint. **Kole must claim napkin-keel from the ClawStreet
   dashboard.** Do NOT re-run `register_keel()`; it would create a second agent.
   Until then every cron run correctly refuses and now logs the refusal.

2. **DONE — NPKN stale-tape bug fixed at the root.** `napkin_tape.bulk()` no
   longer skips symbols with >=1800 bars; it fetches full history and upserts via
   `merge_rows(sym, rows, bulk_path(sym))` (which now takes a path). First run
   caught up +3 equity bars and +5 crypto bars; the shared tape now reads 2515 /
   3749 / 3892, matching hold5's independently-built tape exactly. All 2515x18
   pre-existing bars passed the o/c match assertion, so no silent history
   rewrite. `napkin_tape.py selfcheck` is 7/7 (new check 7 covers the upsert
   offline). Known ceiling, marked with a `ponytail:` comment: a genuine split
   makes Yahoo re-adjust history and will trip that assertion, halting the
   caller — add a force/re-fetch path the first time it actually fires.

3. **DONE — every hold5 run leaves an audit trail.** `trade()` builds the log
   path once and saves before returning on `not claimed` (`"refused": "not
   claimed"`) and on the nothing-held path. That second path also used to write
   an un-suffixed filename, so a dry run overwrote the real log; both now use the
   `_dry` suffix. 9/9 selfchecks pass.

4. **DONE — NPKL venue profile replaced.** Route is `PATCH /v1/me/agents/{id}`
   (found in the OpenAPI spec, not invented). `hold5_registration.py` now applies
   itself with `python3 hold5_registration.py apply` and asserts the spec's length
   limits first. `strategy` (646) and `personality` (629) were over the 500/300
   caps, so both were tightened — every number kept (-9.3% vs -2.1%, ~2 points
   below buy-and-hold, +11.8% swinging 17.5 points). Applied and confirmed in the
   PATCH response.

5. **Once it trades**, verify on the first real run: SOL exit legs present, old
   DQN book unwound, then `hold (n=1..4)` with 0 legs on the following 4 days.
   `ok` should equal `placed`; a gap usually means whole-share rounding or
   buying power.

## 5. Health-check commands (run on the host)

```bash
cd /home/bob/napkin-gap
tail -30 out/hold5.log
PYTHONPATH=.deps /usr/bin/python3.10 napkin_hold5_live.py state
PYTHONPATH=.deps /usr/bin/python3.10 napkin_hold5_live.py trade --dry   # plan, places nothing
ls -lt out/hold5/live/ | head
tail -5 out/trade.log                                                   # NPKN unaffected?
```

Healthy looks like: `state` shows a recent `last_rerank_date` and 3 names in
`held`; tape lines show yesterday's date; hold-bar days log 0 legs. If
`last_rerank_date` stays `null` after a non-dry run that placed orders, the
state commit is broken and the strategy is silently re-ranking daily.

## 6. Replacement NPKL venue profile (ready to apply)

In `hold5_registration.py` as a dict; `python3 hold5_registration.py` prints the
JSON. It states the negative evidence in every field on purpose — the old text
advertises a "5-seed DQN majority vote" allocating "1/18 of equity per symbol"
across 18 tickers, all four claims now false. The series publishes nulls; a
profile claiming skill the agent cannot demonstrate is the one indefensible part
of this deployment.

## 7. If you continue the research instead

Remaining from the original task list: #12 ensembles (needs stochastic models),
#14 GBDT baseline, #26 calibration (prereq for #25 sizing), #7 bar types (needs
intraday), and the bigger swings #15-#23. But #3/#9/#10/#11 together say the
ceiling is **information, not model** — so the honest open axes are DATA
(intraday bars, more symbols, longer history) or **#15 RL for execution**, which
is where the literature says RL actually pays.
