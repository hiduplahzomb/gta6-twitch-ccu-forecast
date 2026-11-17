"""gta6-twitch-ccu-forecast-01 entry: forecast GTA VI's launch-day Twitch
concurrent viewers hour by hour (hours 2..24), as five quantiles an hour.

The model (model.py, standard library only) anchors on the case's hour-1
print and learns the launch-day shape from the case's own comparable
launches: per hour, a quadratic regression of the log viewer ratio on
ln(hour-1 print), median-shifted, with empirical residual quantiles
rescaled by a heteroscedastic factor. Validated by leave-one-out
cross-validation over the 37 comparable launch curves (pinball loss
0.377 vs 0.930 for the persistence baseline).

How the grader runs it: `python3 -P -s /bot/main.py` (PYTHONHASHSEED=0)
in python:3.13-slim, standard library only: no network, no pip packages.
-P keeps this file's directory OFF sys.path, so model.py imports only
after the sys.path line below, which puts the directory first; -s drops
user site-packages.

What it does:
  - read every <id>.in in the input directory (the grader runs one case
    at a time, with only that case's file there);
  - write <id>.out to the output directory: {"hours": [{"hour": 2, "q":
    [q05, q25, q50, q75, q95]}, ...]}, exactly 23 rows, hours 2..24
    each once, no other key, each q five finite numbers above 0 and at
    most 1e9, non-decreasing; at most 65,536 bytes.

Outside the grader's container, MAKEFUN_INPUT_DIR and MAKEFUN_OUTPUT_DIR
stand in for /input and /output:
    mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
"""

import json
import math
import os
import sys
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so `import model`
# finds the model.py shipped beside this file.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import model  # noqa: E402

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))


def round2(x):
    """x rounded to 2 decimals, half up, as JavaScript's
    Math.round(x * 100) / 100 rounds it. A whole number comes back as an
    int, so json.dumps writes 1240 where JavaScript writes 1240.
    """
    scaled = x * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100
    return int(value) if value.is_integer() else value


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = model.forecast_case(case)
        for row in forecast["hours"]:
            row["q"] = [round2(q) for q in row["q"]]
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
