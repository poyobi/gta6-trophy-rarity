#!/usr/bin/env python3
"""Forecast GTA VI PS5 trophy earn rates on Exophase (make.fun gta6-trophy-rarity-forecast-01).

Reads every <id>.in case in $MAKEFUN_INPUT_DIR (else /input) and writes
$MAKEFUN_OUTPUT_DIR/<id>.out: for each of the case's trophies, five quantiles
(5/25/50/75/95%) of its earn rate at the case's target, as percentages.

Method. An earn rate is a ratio, earned / tracked, and both sides move on
their own clocks: new players join (the denominator jumps whenever
Exophase refreshes) while players keep earning (the numerator climbs and
saturates as the early crowd finishes the early content). So I split the
rate into its two counts and forecast each:

- Tracked players N(h) grow toward a ceiling like N_first + A*(1-exp(-h/tau));
  tau is picked by a grid search over the case's reads.
- Each trophy's earned count E_i(h) saturates like A_i*(1-exp(-h/s_i));
  s_i is picked per trophy by a grid search, and the spread of E_i at the
  target across near-best s values is kept as that trophy's model
  uncertainty. Shrinkage: a trophy's eventual share A_i / N_inf is pulled
  toward its tier's median share when its own fit is weak.
- The rate forecast is 100 * E_i(target) / N(target); the spread combines
  the E and N extrapolation uncertainties (delta method), the pooled fit
  residuals, and a horizon term.

One exception: when the whole history is one frozen page (launch night, no
refresh yet), the next refresh is imminent and it can only push rates down
as the tracked-player count jumps, so the median shifts down and the spread
widens a lot.

Pure standard library, deterministic, offline, no model calls.
"""
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.6448536269514722, -0.6744897501960817, 0.0, 0.6744897501960817, 1.6448536269514722]

# Launch-night frozen page: the next refresh is imminent and the tracked
# count jumps, so rates fall. Median shift and spread on the logit scale.
FLAT_SHIFT = -1.10
FLAT_SIGMA = 1.05
# Normal regime: spread floor/cap, horizon term per sqrt day, median-move cap.
SIGMA_FLOOR = 0.55
SIGMA_CAP = 1.8
HORIZON_SIGMA = 0.15
MEDIAN_MOVE_CAP = 2.5
# Grid searches for the two saturating curves (hours).
TAU_GRID = [12.0, 18.0, 24.0, 36.0, 48.0, 72.0, 96.0, 144.0, 216.0, 336.0, 504.0, 720.0, 1080.0]
S_GRID = [6.0, 12.0, 24.0, 48.0, 96.0, 192.0, 384.0, 768.0]


def parse_ts(text):
    s = text.strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    return datetime.fromisoformat(s)


def parse_day(text):
    return datetime.strptime(text.strip(), "%Y-%m-%d").replace(tzinfo=timezone.utc)


def clamp_pct(x):
    return min(99.99, max(0.01, x))


def logit(pct):
    p = clamp_pct(pct)
    return math.log(p / (100.0 - p))


def sigmoid(y):
    if y > 500:
        return 100.0
    if y < -500:
        return 0.0
    return 100.0 / (1.0 + math.exp(-y))


def round4(x):
    """x rounded to 4 decimals, half up, as JavaScript's Math.round does it."""
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value == int(value) else value


def median(vals):
    s = sorted(vals)
    n = len(s)
    if n == 0:
        return 0.0
    if n % 2:
        return s[n // 2]
    return 0.5 * (s[n // 2 - 1] + s[n // 2])


def quantiles_from_logit(med_logit, sigma):
    qs = []
    for z in Z:
        q = sigmoid(med_logit + z * sigma)
        qs.append(round4(min(99.99, max(0.01, q))))
    return sorted(qs)


def lin_scale(xs, ys):
    """Best scale A for y ~= A*x (least squares), and its SSE."""
    sxx = sum(x * x for x in xs)
    if sxx <= 1e-12:
        return 0.0, sum(y * y for y in ys)
    a = sum(x * y for x, y in zip(xs, ys)) / sxx
    sse = sum((y - a * x) ** 2 for x, y in zip(xs, ys))
    return a, sse


def fit_saturating(hours, vals, base, grid, h_target, d_grid=(0.0,)):
    """Fit vals(h) ~= base + A*(1-exp(-max(h-d,0)/tau)) for tau in grid and
    d in d_grid (a delay lets the curve start slow, S-shaped, the way earned
    counts do while players are still reaching the content). Returns
    (mean_at_target, sd_at_target, best_sse, n) where the mean/sd are over
    the grid points weighted by how well each explains the data."""
    n = len(hours)
    if n == 0:
        return base, 0.0, 0.0, 0
    scored = []
    for d in d_grid:
        for tau in grid:
            g = [1.0 - math.exp(-max(h - d, 0.0) / tau) for h in hours]
            a, sse = lin_scale(g, [v - base for v in vals])
            g_t = 1.0 - math.exp(-max(h_target - d, 0.0) / tau)
            scored.append((tau, a, sse, base + a * g_t))
    best_sse = min(s[2] for s in scored)
    dof = max(n - 2, 1)
    var = best_sse / dof
    # Likelihood-style weights; fall back to uniform if the fit is perfect.
    weights = []
    for _tau, _a, sse, _v in scored:
        if var <= 1e-12:
            w = 1.0 if sse <= best_sse + 1e-9 else 0.0
        else:
            w = math.exp(-max(sse - best_sse, 0.0) / (2.0 * var))
        weights.append(w)
    wmax = max(weights) if weights else 0.0
    # A small uniform floor keeps the spread honest when one grid point
    # dominates the weights (the grid is coarse, the truth between points).
    weights = [w + 0.05 * wmax for w in weights]
    wsum = sum(weights)
    if wsum <= 0:
        weights = [1.0] * len(scored)
        wsum = float(len(scored))
    mean = sum(w * s[3] for w, s in zip(weights, scored)) / wsum
    var_t = sum(w * (s[3] - mean) ** 2 for w, s in zip(weights, scored)) / wsum
    return mean, math.sqrt(max(var_t, 0.0)), best_sse, n


def forecast_case(case):
    trophies = case["trophies"]
    history = case["history"]
    if not history:
        raise ValueError("case has no history")
    release = parse_day((case.get("game") or {}).get("release_date") or "2026-11-19")
    target = parse_ts(case["target"])
    h_target = max((target - release).total_seconds() / 3600.0, 0.0)
    horizon_days = case["horizon_minutes"] / 1440.0

    # Per read: hours since release, tracked players, rates per trophy.
    reads = []
    for rd in history:
        at = parse_ts(rd["at"])
        h = max((at - release).total_seconds() / 3600.0, 0.0)
        reads.append((h, float(rd.get("players_tracked") or 0), rd.get("rates") or []))
    hours = [r[0] for r in reads]
    n_first = reads[0][1]

    last_rates = []
    for i in range(len(trophies)):
        last_rates.append(reads[-1][2][i] if i < len(reads[-1][2]) and reads[-1][2][i] is not None else 0.0)

    # Frozen-page regime: every read identical and the tracked count constant.
    flat = len(history) >= 2
    if flat:
        if len({r[1] for r in reads}) > 1:
            flat = False
        else:
            for i in range(len(trophies)):
                ys = [logit(r[2][i]) for r in reads if i < len(r[2]) and r[2][i] is not None]
                if ys and max(ys) - min(ys) > 1e-9:
                    flat = False
                    break
    if flat:
        rows = []
        for i, tr in enumerate(trophies):
            med = logit(last_rates[i]) + FLAT_SHIFT
            rows.append({"id": tr["id"], "q": quantiles_from_logit(med, FLAT_SIGMA)})
        return {"trophies": rows}

    # Tracked players: saturating growth toward a ceiling.
    n_vals = [r[1] for r in reads]
    n_mean, n_sd, _n_sse, _n = fit_saturating(hours, n_vals, n_first, TAU_GRID, h_target)
    n_target = max(n_mean, n_first, 1.0)
    # Relative uncertainty of N at the target (floor: at least 5%).
    n_rel_sd = max(n_sd / n_target if n_target > 0 else 0.0, 0.05)
    n_inf = max(n_target, n_first)

    # Earned counts per trophy: E_i(h) = rate * N / 100, saturating.
    e_mean = []
    e_rel_sd = []
    for i in range(len(trophies)):
        hs = []
        es = []
        for (h, n, rates) in reads:
            if i < len(rates) and rates[i] is not None:
                hs.append(h)
                es.append(float(rates[i]) / 100.0 * n)
        if not hs:
            e_mean.append(0.0)
            e_rel_sd.append(1.0)
            continue
        m, sd, _sse, _nn = fit_saturating(hs, es, 0.0, S_GRID, h_target,
                                         d_grid=(0.0, 12.0, 24.0, 48.0))
        e_mean.append(max(m, 0.0))
        e_rel_sd.append(max(sd / m if m > 0 else 1.0, 0.05))
    # Pooled residual spread in log space: the page's 2-decimal rounding and
    # the refresh steps put a floor under how precisely any smooth curve can
    # track a trophy, so the spread never trusts the fits alone.
    sig_res = 0.18

    # Shrinkage prior: a trophy's eventual share is pulled toward its tier's
    # median share, weighted by how uncertain its own extrapolation is.
    shares = []
    for i in range(len(trophies)):
        shares.append(e_mean[i] / n_inf if n_inf > 0 else 0.0)
    by_tier = {}
    for tr, sh in zip(trophies, shares):
        by_tier.setdefault(tr["tier"], []).append(sh)
    tier_med = {t: median([s for s in v if s > 0]) for t, v in by_tier.items()}
    global_med = median([s for s in shares if s > 0])
    rows = []
    sig_h = HORIZON_SIGMA * math.sqrt(horizon_days)
    for k, tr in enumerate(trophies):
        tm = tier_med.get(tr["tier"], global_med)
        if tm and tm > 0 and shares[k] > 0:
            w_own = 1.0 / (1.0 + e_rel_sd[k])
            share = math.exp(w_own * math.log(shares[k]) + (1.0 - w_own) * math.log(tm))
            e_adj = share * n_inf
            # Only let shrinkage move the target E part-way (the target is
            # short of the eventual share); blend in log space.
            if e_mean[k] > 0:
                e_use = math.exp(0.5 * math.log(e_mean[k]) + 0.5 * math.log(max(e_adj, 1e-9)))
            else:
                e_use = e_adj
        else:
            e_use = e_mean[k]
        rate = 100.0 * e_use / n_target if n_target > 0 else 0.0
        y_hat = logit(rate)
        last_lg = logit(last_rates[k])
        y_hat = min(max(y_hat, last_lg - MEDIAN_MOVE_CAP), last_lg + MEDIAN_MOVE_CAP)
        sig = math.sqrt(e_rel_sd[k] ** 2 + n_rel_sd ** 2 + sig_res ** 2 + sig_h ** 2)
        sig = min(max(sig, SIGMA_FLOOR), SIGMA_CAP)
        rows.append({"id": tr["id"], "q": quantiles_from_logit(y_hat, sig)})
    return {"trophies": rows}


def fallback_rows(case):
    """Persistence with a random-walk spread (house baseline shape). Valid
    by construction; used only if the main forecast raises."""
    last = case["history"][-1]
    spread = 0.5 * math.sqrt(case["horizon_minutes"] / 1440.0)
    rows = []
    for i, tr in enumerate(case["trophies"]):
        rates = last.get("rates") or []
        r = rates[i] if i < len(rates) and rates[i] is not None else 0.0
        rows.append({"id": tr["id"], "q": quantiles_from_logit(logit(r), spread)})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        try:
            fc = forecast_case(case)
        except Exception as exc:  # never crash a case: fall back valid
            print(f"{path.name}: forecast failed ({exc}); writing fallback", file=sys.stderr)
            fc = fallback_rows(case)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(fc, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
