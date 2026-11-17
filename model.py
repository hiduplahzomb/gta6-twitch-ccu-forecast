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
    So per hour we fit a QUADRATIC regression of x_h on ln(p1) across
    the comparables (the relation is curved: it saturates at high p1),
    with the target's ln(p1) clamped to the comparables' observed range
    so a bigger-than-ever launch never extrapolates wildly.
  - The median is shifted so the residuals' median is zero (the pinball
    loss at tau=0.5 wants the conditional median, not the mean).
  - Spreads come from the fit's EMPIRICAL residual quantiles per hour
    (no normality assumption), rescaled by a heteroscedastic factor:
    |residual| regressed on ln(p1), so low-p1 targets (whose surge is
    genuinely uncertain) get wider spreads and high-p1 targets tighter
    ones. The scale is clamped to [0.3, 2.5].
  - Quantiles are p1 * exp(mu + residual_quantile * scale), floored at 1
    and capped at 1e9, forced non-decreasing.

LOO-CV (37 pseudo-cases, official pinball loss in log space):
  persistence baseline (kit starter): 0.930
  pooled empirical quantiles:         0.714
  this model:                         0.377  (0.215 on high-p1 targets,
                                              0.283 on the mid-p1 band
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


def _quad_fit(xs, ys):
    """Least squares for y ~ b0 + b1*x + b2*x^2 via normal equations.

    Returns (b0, b1, b2). Falls back to a linear fit if the system is
    singular (fewer than 3 distinct x values).
    """
    if len(set(xs)) < 3:
        a, b = _ols(xs, ys)
        return (a, b, 0.0)
    rows = [[1.0, x, x * x] for x in xs]
    ata = [[sum(r[i] * r[j] for r in rows) for j in range(3)] for i in range(3)]
    atb = [sum(r[i] * y for r, y in zip(rows, ys)) for i in range(3)]
    m = [ata[i][:] + [atb[i]] for i in range(3)]
    for i in range(3):
        piv = max(range(i, 3), key=lambda r: abs(m[r][i]))
        if abs(m[piv][i]) < 1e-12:
            a, b = _ols(xs, ys)
            return (a, b, 0.0)
        m[i], m[piv] = m[piv], m[i]
        for r in range(i + 1, 3):
            f = m[r][i] / m[i][i]
            for cc in range(i, 4):
                m[r][cc] -= f * m[i][cc]
    beta = [0.0, 0.0, 0.0]
    for i in (2, 1, 0):
        beta[i] = (m[i][3] - sum(m[i][j] * beta[j] for j in range(i + 1, 3))) / m[i][i]
    return (beta[0], beta[1], beta[2])


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


def forecast_hour(p1, h, curves):
    """The five quantiles for hour h given hour-1 print p1 and the curves."""
    # Per-hour sample: comparables with both hour 1 and hour h present.
    sample = [(c[0], c[h - 1]) for c in curves
              if c[0] is not None and c[h - 1] is not None and c[0] > 0 and c[h - 1] > 0]
    if len(sample) < 5:
        return _pooled_quantiles(p1, h, curves)
    lps = [math.log(v1) for v1, _ in sample]
    ys = [math.log(vh / v1) for v1, vh in sample]
    lp1 = min(max(math.log(max(1.0, p1)), min(lps)), max(lps))
    b0, b1, b2 = _quad_fit(lps, ys)
    mu = b0 + b1 * lp1 + b2 * lp1 * lp1
    resid = [y - (b0 + b1 * x + b2 * x * x) for x, y in zip(lps, ys)]
    # Median shift: q50 must be the conditional median, not the mean.
    med = statistics.median(resid)
    mu += med
    resid = [r - med for r in resid]
    # Heteroscedastic scale: |residual| ~ ln(p1).
    abs_resid = [abs(r) for r in resid]
    mean_abs = statistics.mean(abs_resid)
    scale = 1.0
    if mean_abs > 0:
        ah, bh = _ols(lps, abs_resid)
        scale = min(SCALE_MAX, max(SCALE_MIN, (ah + bh * lp1) / mean_abs))
    rs = sorted(resid)
    qs = [p1 * math.exp(mu + _quantile_sorted(rs, tau) * scale) for tau in LEVELS]
    qs = [min(Q_CAP, max(Q_FLOOR, q)) for q in qs]
    for i in range(1, 5):
        qs[i] = max(qs[i], qs[i - 1])
    return qs


def forecast_case(case):
    """Forecast a whole case: {"hours": [{"hour": h, "q": [...]}, ...]}."""
    if case["levels"] != LEVELS:
        raise ValueError(f"case asks for levels {case['levels']}, not {LEVELS}")
    first = next((k for k in case["known"] if k["hour"] == 1), None)
    if first is None:
        raise ValueError("case has no hour-1 print")
    p1 = max(1.0, float(first["viewers"]))
    curves = _comparable_curves(case)
    hours = []
    for row in case["forecast"]:
        h = row["hour"]
        hours.append({"hour": h, "q": forecast_hour(p1, h, curves)})
    return {"hours": hours}
