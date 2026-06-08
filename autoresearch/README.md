# autoresearch — trading

Autonomous trading strategy research, inspired by [@karpathy's autoresearch](https://github.com/karpathy/autoresearch).

The idea: give an AI agent a backtesting setup and let it experiment autonomously. It modifies the strategy code, runs a backtest, checks if `oos_sharpe` improved, keeps or discards, and repeats. You wake up to a log of experiments and (hopefully) a better strategy.

## How it works

Three files matter:

- **`prepare.py`** — fixed oracle. Data loading, IHSG alignment, `backtest_strategy()` evaluator, standardized output format. **Do not modify.**
- **`train.py`** — the single file the agent edits. Contains all strategy signal functions and `get_signal()`. Everything is fair game: regime detector, entry/exit rules, parameter search, ensembling. **This file is edited and iterated on by the agent.**
- **`program.md`** — instructions for the agent. **Edited by the human.**

The metric is **`oos_sharpe`** (out-of-sample Sharpe ratio on the held-out test set) — higher is better.

## Quick start

```bash
# 1. Install dependencies
pip install -r ../requirements.txt

# 2. Download and save data (one-time, ~1 min)
python prepare.py

# 3. Run a single experiment
python train.py > run.log 2>&1
grep "^oos_sharpe:\|^max_drawdown:" run.log
```

## Running the agent

Point Claude Code at this repo and prompt:

```
Have a look at program.md and let's kick off a new autoresearch experiment.
```

## Project structure

```
prepare.py      — data loading + evaluation oracle (do not modify)
train.py        — strategy signal functions + get_signal() (agent modifies this)
program.md      — agent instructions (human modifies this)
pyproject.toml  — dependencies
data/           — CSVs populated by prepare.py (gitignored)
results.tsv     — experiment ledger (gitignored, stays local)
run.log         — last run output (gitignored)
```

## Experiment ledger (`results.tsv`)

Tab-separated, 5 columns:

```
commit	oos_sharpe	max_drawdown	status	description
a1b2c3d	0.723400	-0.082100	keep	baseline
b2c3d4e	0.801200	-0.071300	keep	RSI(14)<35 filter on MR entry
c3d4e5f	0.698000	-0.091000	discard	volume filter — degraded OOS
```

`results.tsv` is intentionally untracked by git.
