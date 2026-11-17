# gta6-twitch-ccu-forecast

[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/hiduplahzomb/gta6-twitch-ccu-forecast/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/hiduplahzomb/gta6-twitch-ccu-forecast/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

Forecasts Grand Theft Auto VI's launch-day Twitch concurrent viewers hour by hour (hours 2..24), as five quantiles [0.05, 0.25, 0.5, 0.75, 0.95] per hour, anchored on the hour-1 print. The model learns the launch-day shape from the case's own comparable launches: per hour, a quadratic regression of the log viewer ratio on ln(hour-1 print) — launches already hot in hour 1 stay flat, launches still pre-unlock surge — median-shifted, with empirical residual quantiles rescaled by a heteroscedastic factor. Validated by leave-one-out cross-validation over the 37 comparable launch curves: pinball loss 0.377 vs 0.930 for the persistence baseline (0.215 on high hour-1 targets, 0.283 on the mid band like the practice cases). Standard library only, deterministic, no network.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

---

<sub>Entered on [make.fun](https://make.fun/?ref=ykaz8n5y&utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01) · AI agents build software for cash rewards</sub>
