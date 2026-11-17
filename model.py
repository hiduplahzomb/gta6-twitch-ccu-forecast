"""Forecast model for gta6-twitch-ccu-forecast-01.

Approach (validated by leave-one-out cross-validation over the 37 comparable
launch curves shipped in the kit's practice cases; see
gta6-twitch-experiment.md in the agent's working notes):

  - The forecast is anchored to the case's hour-1 print p1. For each
    comparable launch we build its launch-day curve (hourly viewers, hours
    1..24 relative to its day_start) and its log-ratio curve
    x_h = ln(viewers_h / viewers_1).
  - The log-ratio at each hour depends strongly on the absolute hour-1
    level: corr(ln(p1), x_h) reaches -0.94 at h=20. Launches already hot
    in hour 1 stay flat; launches still pre-unlock in hour 1 surge.
    So per hour we fit a PIECEWISE-LINEAR regression of x_h on ln(p1)
    with a knot at the comparables' median ln(p1): one slope below the
    knot (the pre-unlock surge regime) and one above it (the already-hot
    regime), continuous at the knot. A global quadratic was tried first
    (LOO loss 0.377); the piecewise fit is better (0.365) because the
    relation has a kink, not smooth curvature. The target's ln(p1) is
    clamped to the comparables' observed range so a bigger-than-ever
    launch never extrapolates wildly.
  - The median is shifted so the residuals' median is zero (the pinball
    loss at tau=0.5 wants the conditional median, not the mean).
  - Spreads come from the fit's EMPIRICAL residual quantiles per hour
    (no normality assumption), rescaled by ONE heteroscedastic factor
    pooled over all hours: |residual| regressed on ln(p1) across every
    (comparable, hour) pair, so low-p1 targets (whose surge is genuinely
    uncertain) get wider spreads and high-p1 targets tighter ones. The
    scale is clamped to [0.3, 2.5]. Pooling stabilises the slope: a
    per-hour fit from 36 points is noisy, the pooled fit uses ~850.
  - Quantiles are p1 * exp(mu + residual_quantile * scale), floored at 1
    and capped at 1e9, forced non-decreasing.

LOO-CV (37 pseudo-cases, official pinball loss in log space):
  persistence baseline (kit starter): 0.930
  pooled empirical quantiles:         0.714
  quadratic regression (first version): 0.377
  this model (piecewise + pooled scale): 0.365  (0.206 on high-p1 targets,
                                              0.287 on the mid-p1 band
                                              like the practice cases)
  The house persistence baseline's published practice loss is 0.387.

Standard library only, no network, deterministic: the same case file
always produces the same bytes.
"""

import math
import statistics
from datetime import datetime, timedelta

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0, 0.674, 1.645]
PERSISTENCE_SIGMA = 0.15
SCALE_MIN = 0.3
SCALE_MAX = 2.5
Q_FLOOR = 1.0
Q_CAP = 1e9


def _parse_time(text):
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def _quantile_sorted(xs, p):
    """Linear-interpolation quantile of an already-sorted list."""
    n = len(xs)
    if n == 1:
        return xs[0]
    i = (n - 1) * p
    lo = int(math.floor(i))
    hi = int(math.ceil(i))
    if lo == hi:
        return xs[lo]
    return xs[lo] + (xs[hi] - xs[lo]) * (i - lo)


def _ols(xs, ys):
    """Ordinary least squares for y ~ a + b*x. Returns (a, b)."""
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx == 0:
        return my, 0.0
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    return my - b * mx, b


def _piecewise_fit(xs, ys):
    """Least squares for y ~ b0 + b1*x + b2*max(0, x - knot), knot at the
    median x: one slope below the knot, one above, continuous at the knot.

    Returns (b0, b1, b2, knot). Falls back to a linear fit if the system is
    singular (fewer than 3 distinct x values, or nothing on one side).
    """
    knot = statistics.median(xs)
    if len(set(xs)) < 3 or all(x <= knot for x in xs) or all(x >= knot for x in xs):
        a, b = _ols(xs, ys)
        return (a, b, 0.0, knot)
    rows = [[1.0, x, max(0.0, x - knot)] for x in xs]
    ata = [[sum(r[i] * r[j] for r in rows) for j in range(3)] for i in range(3)]
    atb = [sum(r[i] * y for r, y in zip(rows, ys)) for i in range(3)]
    m = [ata[i][:] + [atb[i]] for i in range(3)]
    for i in range(3):
        piv = max(range(i, 3), key=lambda r: abs(m[r][i]))
        if abs(m[piv][i]) < 1e-12:
            a, b = _ols(xs, ys)
            return (a, b, 0.0, knot)
        m[i], m[piv] = m[piv], m[i]
        for r in range(i + 1, 3):
            f = m[r][i] / m[i][i]
            for cc in range(i, 4):
                m[r][cc] -= f * m[i][cc]
    beta = [0.0, 0.0, 0.0]
    for i in (2, 1, 0):
        beta[i] = (m[i][3] - sum(m[i][j] * beta[j] for j in range(i + 1, 3))) / m[i][i]
    if any(not math.isfinite(v) for v in beta):
        a, b = _ols(xs, ys)
        return (a, b, 0.0, knot)
    return (beta[0], beta[1], beta[2], knot)


def _launch_curve(comparable):
    """Hourly viewers for hours 1..24 relative to the comparable's day_start.

    Returns a list of 24 numbers (or None for a missing hour). A comparable
    with no hour-1 point is not a launch curve at all (the target's own
    pre-release history) and yields None.
    """
    day_start = _parse_time(comparable["day_start"])
    points = {}
    for p in comparable.get("hourly", []):
        try:
            points[_parse_time(p["from"])] = p["viewers"]
        except (KeyError, ValueError):
            continue
    curve = [points.get(day_start + timedelta(hours=h - 1)) for h in range(1, 25)]
    if curve[0] is None:
        return None
    return curve


def _comparable_curves(case):
    """All usable comparable launch curves, sorted by name for determinism."""
    curves = []
    for comp in sorted(case.get("comparables", []), key=lambda c: c.get("name", "")):
        curve = _launch_curve(comp)
        if curve is not None:
            curves.append(curve)
    return curves


def _persistence_quantiles(p1, h):
    return [max(Q_FLOOR, p1 * math.exp(z * PERSISTENCE_SIGMA * math.sqrt(h - 1)))
            for z in Z]


def _pooled_quantiles(p1, h, curves):
    xs = sorted(math.log(c[h - 1] / c[0]) for c in curves
                if c[0] is not None and c[h - 1] is not None and c[0] > 0 and c[h - 1] > 0)
    if not xs:
        return _persistence_quantiles(p1, h)
    return [max(Q_FLOOR, p1 * math.exp(_quantile_sorted(xs, tau))) for tau in LEVELS]


def _fit_all_hours(p1, hours, curves):
    """Fit every requested hour and return (per_hour, scale).

    per_hour[h] = (mu, sorted_residuals) with mu the median-shifted fitted
    log-ratio at the target's ln(p1), clamped to that hour's comparables'
    observed range. Hours with fewer than 5 comparables are left out; the
    caller fills them from pooled empirical quantiles.

    scale is ONE heteroscedastic factor pooled over all fitted hours:
    |residual| regressed on ln(p1) across every (comparable, hour) pair,
    evaluated at the target's ln(p1) (clamped to the global comparables'
    range) and clamped to [SCALE_MIN, SCALE_MAX]. A per-hour fit from ~36
    points is noisy; the pooled fit uses ~850.
    """
    lp1 = math.log(max(1.0, p1))
    all_lps = [math.log(c[0]) for c in curves
               if c[0] is not None and c[0] > 0]
    global_lo, global_hi = (min(all_lps), max(all_lps)) if all_lps else (lp1, lp1)
    lp1g = min(max(lp1, global_lo), global_hi)
    per_hour = {}
    all_abs, all_lp = [], []
    for h in hours:
        sample = [(c[0], c[h - 1]) for c in curves
                  if c[0] is not None and c[h - 1] is not None
                  and c[0] > 0 and c[h - 1] > 0]
        if len(sample) < 5:
            continue
        lps = [math.log(v1) for v1, _ in sample]
        ys = [math.log(vh / v1) for v1, vh in sample]
        lp1c = min(max(lp1, min(lps)), max(lps))
        b0, b1, b2, knot = _piecewise_fit(lps, ys)
        mu = b0 + b1 * lp1c + b2 * max(0.0, lp1c - knot)
        resid = [y - (b0 + b1 * x + b2 * max(0.0, x - knot))
                 for x, y in zip(lps, ys)]
        # Median shift: q50 must be the conditional median, not the mean.
        med = statistics.median(resid)
        mu += med
        resid = [r - med for r in resid]
        per_hour[h] = (mu, sorted(resid))
        all_abs.extend(abs(r) for r in resid)
        all_lp.extend(lps)
    scale = 1.0
    if all_abs:
        mean_abs = statistics.mean(all_abs)
        if mean_abs > 0:
            ah, bh = _ols(all_lp, all_abs)
            scale = min(SCALE_MAX, max(SCALE_MIN, (ah + bh * lp1g) / mean_abs))
    return per_hour, scale


def _hour_quantiles(p1, h, curves, per_hour, scale):
    """The five quantiles for hour h from a fitted model, with fallbacks."""
    if h in per_hour:
        mu, rs = per_hour[h]
        qs = [p1 * math.exp(mu + _quantile_sorted(rs, tau) * scale)
              for tau in LEVELS]
    else:
        return _pooled_quantiles(p1, h, curves)
    qs = [min(Q_CAP, max(Q_FLOOR, q)) for q in qs]
    for i in range(1, 5):
        qs[i] = max(qs[i], qs[i - 1])
    return qs


def forecast_hour(p1, h, curves):
    """The five quantiles for hour h given hour-1 print p1 and the curves.

    Fits all hours so the pooled heteroscedastic scale matches what
    forecast_case uses; the per-hour fits are cheap.
    """
    per_hour, scale = _fit_all_hours(p1, list(range(2, 25)), curves)
    return _hour_quantiles(p1, h, curves, per_hour, scale)


def forecast_case(case):
    """Forecast a whole case: {"hours": [{"hour": h, "q": [...]}, ...]}."""
    if case["levels"] != LEVELS:
        raise ValueError(f"case asks for levels {case['levels']}, not {LEVELS}")
    first = next((k for k in case["known"] if k["hour"] == 1), None)
    if first is None:
        raise ValueError("case has no hour-1 print")
    p1 = max(1.0, float(first["viewers"]))
    curves = _comparable_curves(case)
    hours = [row["hour"] for row in case["forecast"]]
    per_hour, scale = _fit_all_hours(p1, hours, curves)
    return {"hours": [{"hour": h, "q": _hour_quantiles(p1, h, curves, per_hour, scale)}
                      for h in hours]}
