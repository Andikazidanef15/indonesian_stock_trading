# autoresearch — trading

This is an experiment to have the LLM do its own trading strategy research.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `apr16`). The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current main.
3. **Read the in-scope files**: Read these files for full context:
   - `README.md` — repository context.
   - `prepare.py` — fixed data loading and evaluation oracle. Do not modify.
   - `train.py` — the file you modify. Strategy functions and `get_signal()`.
4. **Verify data exists**: Check that `data/` contains `train_BBCA_JK.csv`, `test_BBCA_JK.csv`, and `ihsg_close.csv`. If not, tell the human to run `python prepare.py`.
5. **Initialize results.tsv**: Create `results.tsv` with just the header row:
   ```
   commit	oos_sharpe	max_drawdown	status	description
   ```
6. **Confirm and go**: Confirm setup looks good, then kick off experimentation.

## Experimentation

**What you CAN do:**
- Modify `train.py` — this is the only file you edit. Everything in `get_signal()` and the strategy functions above it is fair game: regime detector, entry/exit conditions, indicator combinations, parameter search on `train_data`, ensembling, etc.

**What you CANNOT do:**
- Modify `prepare.py`. It is read-only. It contains the fixed evaluation oracle (`backtest_strategy`), data loading, and output format.
- Install new packages. Only use what is already imported in `train.py`.
- Modify `transaction_cost` in the `backtest_strategy` call — it is fixed at 0.003 (0.3% per side, calibrated for Indonesian retail brokers).

**The goal is simple: get the highest `oos_sharpe`** on the held-out test set. The train set is available for parameter fitting (e.g. Optuna search) — use it. The test set is the ground truth and must never be used for fitting.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Removing code and getting equal or better results is a great outcome. When evaluating whether to keep a change, weigh the complexity cost against the improvement. A 0.01 Sharpe improvement from 20 lines of hacky code? Probably not worth it. A 0.01 improvement from deleting code? Definitely keep.

**The first run**: Always establish the baseline first — run `train.py` as-is without any changes.

**Ideas to try** (not exhaustive):
- RSI / stochastic filter on mean-reversion entry to reduce false signals
- Volume confirmation for trend-following entries
- Tune `entry_z`, `exit_z`, `vol_ratio_threshold`, `sma_trend` via Optuna on train set
- Add a short-term momentum filter (e.g. ROC > 0)
- Alternative regime detectors: rolling Hurst, Variance Ratio test
- Ensemble of trend + mean-reversion with dynamic weighting
- Tighter stop-loss via ATR-based exit rule
- Sector rotation: use IHSG regime to weight exposure

## Output format

Once `python train.py` finishes it prints a summary like this:

```
---
oos_sharpe:       0.723400
max_drawdown:    -0.082100
ann_return:       0.123400
ann_volatility:   0.145600
win_rate:         0.523400
profit_factor:    1.234500
n_trades:         45
avg_exposure:     0.623400
```

Extract the key metrics:

```bash
grep "^oos_sharpe:\|^max_drawdown:" run.log
```

## Logging results

Log each experiment to `results.tsv` (tab-separated — do NOT use commas):

```
commit	oos_sharpe	max_drawdown	status	description
```

1. git commit hash (short, 7 chars)
2. `oos_sharpe` value (e.g. `0.723400`) — use `0.000000` for crashes
3. `max_drawdown` value (e.g. `-0.082100`) — use `0.000000` for crashes
4. status: `keep`, `discard`, or `crash`
5. short description of what this experiment tried

Example:

```
commit	oos_sharpe	max_drawdown	status	description
a1b2c3d	0.723400	-0.082100	keep	baseline
b2c3d4e	0.801200	-0.071300	keep	RSI(14)<35 filter on MR entry
c3d4e5f	0.698000	-0.091000	discard	volume filter — degraded OOS
d4e5f6g	0.000000	0.000000	crash	Optuna on test data (data leak bug)
```

Do not commit `results.tsv` — leave it untracked.

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/apr16`).

LOOP FOREVER:

1. Check git state: current branch and commit.
2. Modify `train.py` with an experimental idea — directly edit the code.
3. `git commit -am "brief description of change"`
4. Run: `python train.py > run.log 2>&1`
5. Read results: `grep "^oos_sharpe:\|^max_drawdown:" run.log`
6. If grep is empty, the run crashed. Read `tail -n 50 run.log` for the traceback. Attempt a fix. If unfixable after 2 attempts, log as `crash` and revert.
7. Record in `results.tsv`.
8. If `oos_sharpe` improved → **keep** the commit, advance the branch.
9. If `oos_sharpe` equal or worse → **discard**, `git reset --hard HEAD~1`.

**NEVER STOP**: Once the loop begins, do NOT pause to ask the human whether to continue. The human may be asleep. You are autonomous. If you run out of ideas, re-read the in-scope files, revisit near-misses, try more radical changes. The loop runs until the human interrupts you, period.
