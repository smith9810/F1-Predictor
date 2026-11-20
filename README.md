# F1 race predictor

Forecasts every Formula 1 Grand Prix twice - before qualifying and after it - logs each forecast before the
race starts, scores it afterwards, and changes the model only when a walk-forward test says the change
helps. Win, podium, points and retirement probabilities for every driver, plus fair (margin-free) odds.

## What the backtest found (2013-2026, 291 races, every forecast made with earlier data only)

Ranked probability score of each driver's finishing place (lower is better):

| Forecast | After qualifying | Before qualifying |
|---|---|---|
| Every driver equal | 0.1749 | 0.1749 |
| Starting grid only (history of each slot) | 0.1238 | - |
| Recent qualifying form only | - | 0.1298 |
| **Model** | **0.1165** | **0.1259** |

- After qualifying the model beats the grid-only forecast by 0.0073 (95% interval 0.0057 to 0.0089) and
  is better in 71% of races; before qualifying it beats recent form by 0.0040 (0.0027 to 0.0053), better in
  65% of races. 2026 (new rules): 0.1091 against the grid's 0.1212.
- Raw simulated win chances were under-confident about favourites (drivers given 30-40% won 55% of the
  time). A recalibration fitted each season on the previous six seasons fixes most of it and is part of the
  live model: winner log loss after qualifying 1.399 -> 1.291 (p < 0.001), before qualifying 1.748 -> 1.696.
- **Not tested against bookmakers**: no free source of historical F1 odds was found, and The Odds API (used
  by the Premier League model) does not cover F1. The `fair_win_odds` column is what to compare with a
  bookmaker's price by hand.

### Is a race less predictable once the champion is decided?
Tested on every season since 1991 where the drivers' title was settled with races to spare (49 races in
1996-2026 with forecasts, against 157 late-season races where the title was still open):
- No. If anything the reverse: titles are settled early in dominant seasons, and the winner of the
  remaining races was **easier** to predict (winner log loss 1.26 v 1.57, p = 0.006).
- Relative to the model's own expectations, results after the title was decided were no more random (strength
  scale 1.09 v 0.96; 1 = as the model expects; difference -0.10 to +0.40).
- The new champion did not finish measurably worse than expected (-1.68 v -1.65 places, p = 0.93) and did
  not retire more often (2.1% v 5.7%).
- So the model has no "title decided" adjustment. Sample sizes are small; the test reruns monthly and
  2026 adds three more post-title races (Las Vegas, Qatar, Abu Dhabi).

## How it works
1. **Strengths.** Plackett-Luce (rank-ordered logit) fitted to finishing orders: driver skill + car pace for
   each team in each season (linked to the team's previous season, weakly when the technical rules change:
   2009, 2014, 2017, 2022, 2026), qualifying orders (fitted as more orderly than races) and sprints. Older
   sessions count less (half-life 365 days). Team renames are followed (Toro Rosso -> Racing Bulls, Sauber ->
   Audi, etc.).
2. **Track position.** A better starting slot adds strength (`gamma`), learned from the data.
3. **Retirements.** Logistic regression on the field's recent DNF rate and the team's, driver's and
   circuit's histories (and grid slot after qualifying), refitted each January on the previous 10 seasons.
4. **Simulation.** 20,000 races: retirements drawn first, finishers ordered by strength plus random noise.
   Before qualifying, qualifying is simulated first to get the grid.
5. **Recalibration.** Win, podium and points probabilities recalibrated with parameters fitted on the previous
   six seasons of out-of-sample forecasts.
6. **No look-ahead.** A test rewrites later results and checks that earlier forecasts do not change.
   Settings were chosen on 2005-2012 only (`models/tuning_2005_2012.json`), then frozen for the 2013-2026 test.

Challengers tested and **not** adopted: a driver skill per season linked across seasons (no difference:
+0.00008, p = 0.28), shorter or longer memory (flat between 180 and 730 days).

## Running it
GitHub Actions (`.github/workflows/f1.yml`, UTC): Thursday-Sunday every 3 hours it refreshes F1DB, forecasts
the next race (before qualifying, then again once qualifying is in F1DB) and commits; Monday/Tuesday it
scores the last race; on the 3rd of each month it reruns the backtest and recalibration.
Manual run: Actions > f1-predictor > Run workflow.

```
pip install -r requirements.txt
python -m pytest -q                       # 21 tests
python -m f1 update --f1db path/to/f1db/src/data
python -m f1 predict                      # next race, stage chosen automatically
python -m f1 score
python -m f1 backtest                     # about 2 minutes
```

Outputs: `logs/predictions.csv` (append-only; refuses rows after the race start), `logs/dists/` (full
finishing-place distributions for scoring), `logs/scores.csv`, `out/dashboard/*.csv`.

## Limits
- Grid penalties are not known before the official grid; after qualifying the qualifying order is used.
- Practice pace, tyre strategy, weather and team orders are not modelled.
- The model is slow to recognise a driver's step change (example: Antonelli's rookie 2025 still weighs on his
  2026 rating, so it rates him below Russell despite his title).
- The Las Vegas 2026 pre-qualifying forecast was logged twice: by the first version (uncalibrated, `pl-v1`)
  and after recalibration was adopted the same morning (`pl-v1-cal`). Both are scored.
- Nothing here places bets.

Data: [F1DB](https://github.com/f1db/f1db) (CC BY 4.0).
