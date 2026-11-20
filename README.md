# gta6-trophy-rarity

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/poyobi/gta6-trophy-rarity/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/poyobi/gta6-trophy-rarity/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Forecasts every Grand Theft Auto VI PlayStation 5 trophy's earn rate on Exophase at 2026-11-26T00:00:00Z, as five quantiles (5/25/50/75/95%) per trophy.

An earn rate is a ratio, earned / tracked, and both sides move on their own clocks: new players join (the denominator jumps whenever Exophase refreshes) while players keep earning (the numerator climbs and saturates as the early crowd finishes the early content). So `main.py` splits the rate into its two counts and forecasts each: tracked players grow toward a ceiling like N_first + A*(1-exp(-h/tau)), and each trophy's earned count saturates like A_i*(1-exp(-max(h-d,0)/s_i)), with the delay letting the curve start slow the way earned counts do while players are still reaching the content. The rate forecast is 100 * E_i(target) / N(target); the spread combines the E and N extrapolation uncertainties, the pooled fit residuals, and a horizon term. One exception: when the whole history is one frozen page (launch night, no refresh yet), the next refresh is imminent and it can only push rates down as the tracked-player count jumps, so the median shifts down and the spread widens a lot.

Pure Python standard library, deterministic, offline, no model calls.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

---

<sub>Entered on [make.fun](https://make.fun/?ref=sfcm9bvc&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
