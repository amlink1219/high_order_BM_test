#!/usr/bin/env bash
set -euo pipefail
shopt -s nullglob

cd /home/Hongjie_Zeng/high_order_BM

RUN_DIR=runs_vggsound_full_AV012_twoport_videolstm4096_audioaf031_lstm4096_h8_g115_lc5_e320

git add \
  eval_vggsound_twoport_checkpoint.py \
  sbatch_eval_vggsound_av012_best.sh \
  push_vggsound_av012_eval_results.sh \
  README_vggsound_av012_eval.md \
  vggsound_full_experiment_status.md || true

git add \
  "$RUN_DIR/config.json" \
  "$RUN_DIR/history.json" \
  "$RUN_DIR/full_eval_best_evalonly_3000.json" \
  "$RUN_DIR"_stdout.log \
  "$RUN_DIR"_stderr.log || true

git add logs/vggsound_av012_eval_*.out logs/vggsound_av012_eval_*.err || true

git status --short

if git diff --cached --quiet; then
  echo "No staged AV012 eval files to commit."
else
  git commit -m "Add AV012 eval-only full Gibbs result"
fi

git push -u origin main
