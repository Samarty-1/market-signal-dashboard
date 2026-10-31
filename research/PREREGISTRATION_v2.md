# Pre-registration v2: raising Sharpe *and* profit of the point-in-time book

Written 2026-10-06, **before any of the experiments below were run.**

## Starting point (frozen, from `results_confirm_pit.json`)

The `FIXED full (blend, rank, hl42, band)` book on the point-in-time universe:

| | DEV (≤ 2022) | CONFIRM (2023-01 → 2026-10) |
|---|---|---|
| net Sharpe | 0.51 | 0.71 |
| net return / yr | 3.8% | 2.2% |
| vol | 7.4% | 3.1% |

Diagnosis that motivates each hypothesis:

* **Gross is now the bottleneck, not cost.** Confirmation cost is 0.18 bps/day
  against 1.05 bps/day gross. Further construction tweaks cannot help much.
* **The features are all *total*-return technicals.** Each stock's return is mostly
  market + common-factor moves. Reversal and momentum measured on total returns
  are contaminated by those moves. The literature finds the *residual* versions
  much stronger and less crash-prone: residual momentum (Blitz, Huij & Martens
  2011) roughly doubles the Sharpe of total-return momentum, and residual /
  industry-adjusted short-term reversal (Da, Liu & Schaumburg 2014) is far
  stronger than raw reversal.
* **The book is dollar-neutral, not beta-neutral.** Its standing tilt is long
  small / illiquid / high-volatility and short large / liquid / low-volatility.
  High-volatility names are high-beta names, so the book carries net market
  beta. That is unpaid volatility for a book whose job is relative selection.
* **The book runs at 3–7% vol.** Profit in percentage terms is a choice of
  leverage, not of signal. Volatility targeting raises return at roughly
  constant Sharpe, and in factor portfolios it has often raised Sharpe too
  (Moreira & Muir 2017; the evidence for long-short books is mixed).
* **The simulator charges no borrow cost on the short leg.** That flatters every
  number above and matters more once the book is levered.

## Hypotheses

| id | change | expected effect |
|---|---|---|
| H0 | add 50 bp/yr borrow on short notional (accounting fix, always adopted) | Sharpe slightly down |
| H1 | beta-neutral construction: project target weights orthogonal to each name's trailing 252-day beta to the equal-weight universe | vol down, Sharpe up |
| H2 | residual-return features: residual reversal (1d/5d/21d), residual momentum 12-1 scaled by residual vol, residual vol; retrain the same 4-horizon LightGBM blend with them added | IC up, gross up |
| H3 | volatility-target overlay: scale the book to 10% annual vol from trailing 63-day realised vol (lagged one day), leverage capped at 4×, cost charged on the scale changes | return up ~2–3×, Sharpe ≈ unchanged |

## Decision rule (fixed now)

* All choices are made on **DEV only** (dates ≤ 2022-12-31).
* H1 and H2 are each adopted only if DEV net Sharpe rises by **≥ 0.10** over the
  baseline **and** the change does not lower net Sharpe in more than one of the
  five DEV calendar years 2018–2022.
* H3 is judged on a different bar, because its purpose is profit, not Sharpe. It is adopted if
  DEV net Sharpe does not fall by more than 0.05 and max drawdown stays above −20%.
* No parameter in H1–H3 is tuned. The values above (252, 63, 10%, 4×, 50 bp) are
  conventional, not searched.
* The stacked, adopted configuration is then scored **once** on CONFIRM.

## Honest limits, stated in advance

* CONFIRM is **not pristine**. It has been scored before, on two universes and four
  books. A pass there is weaker evidence than it was the first time. The only
  fully clean test left is forward paper performance after 2026-10-06.
* Trials are counted below. Four hypotheses, each a single fixed configuration.

## Addendum A (written 2026-10-06, after H1–H3, before H5/H6 were run)

H1–H3 used price data only. H5 and H6 add one new piece of information that free
data supports point-in-time: each company's **GICS sector** as listed in the same
pinned Wikipedia revisions that define membership. A name's sector on date t comes
from the latest revision at or before t, never a later one.

| id | change | why |
|---|---|---|
| H5 | sector-neutral construction: demean target weights within each sector every day, restore gross | A rank book drifts into sector bets that carry risk but no forecast. Stat-arb practice removes them. |
| H6 | sector-relative features: return_1d/5d/21d and mom_12_1 minus their same-day sector mean, plus the sector's own 21d and 126d return; retrain the same blend | Industry-adjusted short-term reversal is far stronger than raw reversal (Da, Liu & Schaumburg 2014). Industry momentum is a separate documented effect (Moskowitz & Grinblatt 1999). |

The same rule applies: adopt only if DEV net Sharpe rises ≥ 0.10 over H0 (0.47)
and is not lower in more than one DEV year. If both pass, the H5+H6 stack is
scored once on CONFIRM. **Caveat stated in advance:** CONFIRM has now been
scored many times, so a pass there is weak evidence and needs forward paper
results before anyone trades on it.

### Addendum A results (DEV; CONFIRM not scored because nothing qualified)

| # | hypothesis | DEV net Sharpe | Δ vs H0 | adopted? |
|---|---|---|---|---|
| 6 | H5 sector-neutral | 0.34 | −0.13 | **no** |
| 7 | H6 sector-relative features | 0.31 | −0.16 | **no** (blend IC 0.0144 → 0.0027) |
| 8 | H5+H6 (diagnostic) | 0.41 | −0.06 | no |

**Why, from `research/feature_ic.py` (single-feature DEV rank IC vs 5-day return,
every 5th date):** in this universe and period the literature's adjustments make
reversal *weaker*. Raw 1-day reversal has IC −0.024 (t −2.4); the sector-relative
version −0.016 and the market-residual version −0.015. The sector's own 21-day
move is itself a reversal signal (IC −0.023, t −2.2), so sector-neutralising
deletes part of the edge. That is the same mechanism by which beta-neutralising
deleted the beta premium. 12-1 momentum carries nothing in any form (|IC| < 0.004).

**Total trials in v2: 8. Adopted: none.**

## Addendum B (written 2026-10-06, after `feature_ic.py`, before H7 was run)

**Motivation, from DEV:** the fitted LightGBM blend has a 5-day IC of +0.0144. That
is *below* the |IC| of single raw inputs it already sees: return_1d −0.024 and
sector_mom_21d −0.023. A fitted model scoring under its own best input is the low-SNR
signature of the trees fitting noise. The standard response is an **unfitted**
composite. There are no weights to learn, so there is nothing to overfit.

| id | signal | construction |
|---|---|---|
| H7a | composite alone: mean within-date rank of (−return_1d, −return_5d, −mom_21d, −sector_mom_21d). These are the four features with the largest DEV \|IC\| and t ≤ −1.6. Signs come from the literature (short-term reversal), not from fitting. | unchanged: rank weights, EMA-42, band 0.5, 10bp, 50bp borrow |
| H7b | 50/50 within-date rank average of H7a and the ML blend | unchanged |

Same rule: adopt only if DEV net Sharpe ≥ 0.57 (H0 + 0.10) and not lower in more
than one DEV year. If both qualify, the higher DEV Sharpe wins and is the only one
scored on CONFIRM. Trials 9 and 10.

**Selection caveat:** the four features were picked by looking at DEV ICs, so
DEV flatters H7 somewhat. CONFIRM is the check, with the standing caveat that it
has been viewed before for *other* books (never for this signal).

### Addendum B results (DEV; CONFIRM not scored because nothing qualified)

| # | signal | DEV 5d IC | DEV net Sharpe | adopted? |
|---|---|---|---|---|
| — | ML blend (H0) | 0.0144 | 0.47 | — |
| 9 | H7a unfitted reversal composite | **0.0215** | 0.19 | **no** |
| 10 | H7b 50/50 | 0.0205 | 0.32 | **no** |

The composite is a better *forecast* (+49% IC) and a worse *book*. Reversal is a
fast signal. The only construction that survives costs (EMA-42 + band) holds
positions for weeks, and by then the reversal has played out. Trading it faster
costs more than it earns: hard-decile daily rebalancing in the main README scored
−1.70. The signal is real and smaller than the cost of harvesting it. That is the
same wall `quantpulse` (hourly RSI) and `pairs-trading-audit` hit.

**Total v2 trials: 10. Adopted: 0.** On price/volume data, the binding constraint
is not model fit, construction, hedging or leverage. It is that the
cost-surviving horizon (weeks) has almost no price-based signal, and the horizon
with signal (days) can't pay its costs. Only information that predicts at a
horizon of weeks to months can move this. Fundamentals (H4) are the candidate.

## Addendum C (written 2026-10-06, after H7, before H8 was run)

H7 showed the gap: the price signal pays over days, but only a book holding for
weeks survives costs. H8 does not hold the fast signal. It uses it to **time the
trades the slow book already makes** (alpha-aware execution).

| id | change | fixed parameters |
|---|---|---|
| H8 | Same slow targets as H0. When a name's gap to target exceeds the band, trade only if the H7a reversal composite agrees with the trade direction (buy only when composite > 0, i.e. the name has recently fallen relative to peers; sell only when < 0). Otherwise defer. A gap above 2× the band trades regardless, so no position can drift without limit. | band 0.5 (as H0), force at 1.0, composite exactly as H7a |

Expected mechanism: turnover about unchanged, gross up by the reversal earned on
each executed trade. Same rule: adopt only if DEV net Sharpe ≥ 0.57 and not lower
in more than one DEV year. If it qualifies, CONFIRM is scored once. Trial 11.

### Addendum C result

| # | | DEV net Sharpe | ret | adopted? |
|---|---|---|---|---|
| 11 | H8 reversal-timed execution | 0.47 (H0 0.47) | 3.4% (3.4%) | **no** |

No effect. The slow book trades about 2.3% of gross a day, so even a real per-trade
timing edge adds only hundredths of a percent a year. Execution alpha scales
with turnover, and this book barely turns over. **Total v2: 11 trials, 0 adopted.**

## Deferred: H4, point-in-time fundamentals (not run)

Gross profitability (Novy-Marx 2013), book-to-market, earnings yield and asset
growth (Cooper, Gulen & Schill 2008), taken from SEC EDGAR XBRL `companyfacts` and
usable only from each filing's `filed` date. That makes them a structurally
different source of information from every price/volume feature here. They also
change slowly, which suits an EMA-42 book. The fixed form, if run: an unfitted
equal rank-average of the four with literature signs, blended 50/50 with the ML
signal, judged by the same rule as H1/H2.

**Blocked on access, not on evidence.** EDGAR returns 403 unless the User-Agent
carries a real contact email. That needs the repo owner's consent before any
request is sent.

## Trial ledger

DEV net Sharpe. Baseline reproduced the published figure exactly (0.51).

| # | hypothesis | DEV net Sharpe | Δ vs H0 | years worse | adopted? |
|---|---|---|---|---|---|
| 1 | H0 borrow 50 bp/yr | 0.47 | — | — | yes (accounting) |
| 2 | H1 beta-neutral | 0.30 | −0.17 | 4 of 5 | **no** |
| 3 | H2 residual features | 0.22 | −0.25 | 3 of 5 | **no** (IC 0.0144 → 0.0043) |
| 4 | H1+H2 (diagnostic, not a candidate) | 0.19 | −0.28 | — | no |
| 5 | H3 vol-target 10% | **0.68** | **+0.21** | n/a (H3 rule) | **yes** (ret 3.4% → 7.5%, dd −12.1%, avg gross 2.28×) |

Decision recorded in `research/adopted_v2.json` before CONFIRM was scored.

## CONFIRM (scored once, 2026-10-06)

| book | net Sharpe | net ret/yr | vol | max DD |
|---|---|---|---|---|
| H0 baseline + borrow | 0.27 | 0.9% | 3.4% | −6.5% |
| **adopted stack (H0 + H3 vol-target)** | **−0.01** | −0.2% | 10.3% | **−19.6%** |

**H3 failed confirmation.** On DEV it added +0.21 Sharpe. On CONFIRM it removed
0.28, and it nearly reached the −20% drawdown limit. Confirmation-period vol was
low (3.4%), so the overlay ran at 3.29× average gross and levered up a signal with
no reliable edge. Vol-targeting multiplies whatever edge exists. Here that was
close to nothing, so it mostly multiplied noise. **Nothing from v2 is adopted.**

## Reconciliation: why H0 scored 0.27 and not the published 0.71

`research/reconcile_v2.py`. The published `confirm.py` passes the blend through
`decompose.split()`. Its `keep` mask silently drops every (date, name) with fewer
than 120 prior predictions, which mostly means names recently added to the index
(67,842 of 923,993 rows). `improve.py` used the blend as designed. No borrow on
either row:

| | DEV | CONFIRM |
|---|---|---|
| blend, every member ranked | 0.51 | **0.34** |
| blend + incidental 120-day mask (published) | 0.52 | **0.71** |

The mask is causal, so it is not a leak. But it was never a design decision. On
DEV, where decisions are made, it is worth +0.01. Its +0.37 on CONFIRM therefore
cannot be claimed as validated. **The honest confirmation figure for the book as
designed is 0.34 gross of borrow and 0.27 net of it**, not 0.71. That is
t ≈ 0.5 over 3.7 years.
