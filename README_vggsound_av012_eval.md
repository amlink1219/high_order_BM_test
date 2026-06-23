# AV012 Eval-Only Full Gibbs Test

This package evaluates the partially trained AV012 two-port BM checkpoint without running training on the login node.

Target checkpoint on the server:

```text
runs_vggsound_full_AV012_twoport_videolstm4096_audioaf031_lstm4096_h8_g115_lc5_e320/best.pt
```

Evaluation settings:

```text
eval_steps=3000
eval_burn_in=500
eval_thin=2
label_init=random_onehot
label_update=binary
eval_batch_size=16
```

Output:

```text
runs_vggsound_full_AV012_twoport_videolstm4096_audioaf031_lstm4096_h8_g115_lc5_e320/full_eval_best_evalonly_3000.json
logs/vggsound_av012_eval_JOBID.out
logs/vggsound_av012_eval_JOBID.err
```

Submit on the server:

```bash
cd /home/Hongjie_Zeng/high_order_BM
unzip -o vggsound_av012_eval_code_20260623.zip
chmod +x sbatch_eval_vggsound_av012_best.sh push_vggsound_av012_eval_results.sh
sbatch sbatch_eval_vggsound_av012_best.sh
```

Upload result after completion:

```bash
cd /home/Hongjie_Zeng/high_order_BM
bash push_vggsound_av012_eval_results.sh
```

If this eval still hits a CUDA launch timeout, rerun the same sbatch after editing `--eval_batch_size 16` to `--eval_batch_size 8`.
