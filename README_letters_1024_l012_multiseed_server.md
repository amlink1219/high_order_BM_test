# EMNIST 1024 L012 Multi-Seed Server Run

Purpose: reproduce the L012 1024 p-bit EMNIST20 + ISOLET400 two-port BM result over multiple model seeds.

## What This Runs

- Fixed data seed: `123`
- Fixed train pairing seed: `20260610`
- Fixed test pairing seed: `20260611`
- Varied model seeds: `124, 125, 126, 127, 128`
- Runs:
  - `L018_l012_modelseed124`
  - `L019_l012_modelseed125`
  - `L020_l012_modelseed126`
  - `L021_l012_modelseed127`
  - `L022_l012_modelseed128`

The training script now supports `--model_seed`, so the EMNIST/ISOLET split stays fixed while model initialization and training randomness vary.

## Submit

From the uploaded folder on the server:

```bash
sbatch sbatch_letters_1024_l012_multiseed.sh
```

Monitor:

```bash
squeue
tail -f logs/emnist1024_l012_multiseed_<JOBID>.out
```

## Outputs

Each run saves:

```text
runs_letters_isolet_L018_l012_modelseed124/
  config.json
  history.json
  best.pt
  last.pt
  full_eval_best_3000.json
```

Aggregate files:

```text
letters_1024_l012_multiseed_server_summary.json
letters_1024_l012_multiseed_server_summary.md
```

Per-run stdout/stderr:

```text
logs/L018_l012_modelseed124_stdout.log
logs/L018_l012_modelseed124_stderr.log
...
```

## Notes

The scripts use `--auto_download` for EMNIST and ISOLET. If the server cannot access the network, copy the existing `data_letters_isolet/` cache from the local machine or a previous server run into the same directory before submitting.
