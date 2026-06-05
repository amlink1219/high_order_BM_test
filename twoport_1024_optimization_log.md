# Two-Port 1024 Optimization Log

This log records reproducible experiments for improving the 1024 p-bit two-port BM.
Final recognition accuracy must be full test `test_label_gibbs_acc`; `short_cd_label_acc`
is diagnostic only.

## Baseline Facts Before New Runs

- Previous best conditional two-port BM: `94.59%` full test label-Gibbs accuracy.
- Source run: `runs_joint_cond_twoport_wsd_coupled_lc5_p1000_20x20_mnist20crop_time40_gamma110_lr2e4`.
- Previous best config used `total_pbits=1000`, `image_dim=400`, `label_dim=50`, `hidden_dim=550`, `gamma_h=1.1`, `gamma_l=1.1`, `lr=2e-4`, `cd_k=3`.
- New optimization target uses strict `1024` p-bit resource: `image_dim=400`, `label_dim=50`, `hidden_dim=574`.
- Existing late-fusion diagnostic result: `98.61%` at `lambda_audio=0.5`; this can be a teacher for training, but final inference must remain p-bit Gibbs.

## Implementation Notes

- New training entrypoint: `train_twoport_1024_optimization_wsd.py`.
- The script saves `config.json`, `history.json`, `summary.json`, `best.pt`, and `last.pt`.
- `last.pt` includes optimizer state by default.
- Quick eval fields are recorded as `quick_test_label_gibbs_acc`; final claims must use full `test_label_gibbs_acc`.
- Distillation requires train-set teacher probabilities in the supplied NPZ. The existing `runs_fusion_diagnostics_p1000_20x20/predictions_and_scores.npz` contains test-set teacher ingredients only, so it must not be used for training distillation unless a train teacher NPZ is generated.

## SMOKE attempted - 2026-06-04

- Purpose: verify the new 1024 optimization entry with `max_train=100`, `max_test=40`, `epochs=1`, quick eval only.
- Result: not executed in this Codex shell because no torch-enabled Python is available.
- Environment details: bundled Codex Python can compile the scripts but does not include `torch`; system `python` is not on PATH; Windows `py.exe` reports no installed Python.
- Static check: `py_compile` passed for `train_twoport_1024_optimization_wsd.py` and `make_late_fusion_teacher_wsd.py`.
- Note: this was the restricted Codex tool environment only; the same smoke test later succeeded with the user's normal system Python environment, as recorded below.
- Next: run smoke/E001 from the same Python environment that produced the existing training results.
## SMOKE started - 2026-06-04 23:00:10
- Purpose: Smoke test for new 1024 optimization entry
- Change: max_train=100,max_test=40,quick eval only
- Output: `./runs_twoport1024_smoke`
- Config: total=1024, hidden=574, gamma_h=1.1, gamma_l=1.1, lr=0.0002, cd_k=3, distill_weight=0.0
- Command: `train_twoport_1024_optimization_wsd.py --out_dir ./runs_twoport1024_smoke --experiment_id SMOKE --purpose Smoke test for new 1024 optimization entry --change_note max_train=100,max_test=40,quick eval only --next_note Delete/ignore smoke run; start E001 with full data --max_train 100 --max_test 40 --epochs 1 --batch_size 20 --eval_batch_size 20 --num_workers 0 --quick_eval_steps 20 --quick_eval_burn_in 5 --quick_eval_thin 2 --eval_every 1 --early_stop_patience 0 --no_full_eval_final --cpu`

## SMOKE completed - 2026-06-04 23:00:10
- Output: `./runs_twoport1024_smoke`
- Best selection metric: 0.100000 at epoch 1
- Final epoch: 1
- Final full `test_label_gibbs_acc`: not_run
- Next: Delete/ignore smoke run; start E001 with full data

## E001 started - 2026-06-04 23:07:18
- Purpose: E001_1024_baseline
- Change: total_pbits_1024_only_from_best_twoport_config
- Output: `./runs_twoport1024_E001_baseline`
- Config: total=1024, hidden=574, gamma_h=1.1, gamma_l=1.1, lr=0.0002, cd_k=3, distill_weight=0.0
- Command: `train_twoport_1024_optimization_wsd.py --out_dir ./runs_twoport1024_E001_baseline --experiment_id E001 --purpose E001_1024_baseline --change_note total_pbits_1024_only_from_best_twoport_config --next_note review_E001_then_gamma_sweep --epochs 80 --early_stop_patience 8 --eval_every 2 --quick_eval_steps 800 --quick_eval_burn_in 100 --quick_eval_thin 2 --full_eval_on_best --full_eval_steps 3000 --full_eval_burn_in 500 --full_eval_thin 2 --batch_size 50 --eval_batch_size 128 --cd_k 3 --lr 0.0002 --momentum 0.6 --weight_decay 0.0 --gamma_h 1.1 --gamma_l 1.1 --label_inhibit 0.3 --label_update binary --label_init random_onehot --audio_layout time40_fold --audio_scale zscore_sigmoid --num_workers 2`

## E001 completed - 2026-06-04 23:55:32
- Output: `./runs_twoport1024_E001_baseline`
- Best selection metric: 0.942700 at epoch 34
- Final epoch: 50
- Final full `test_label_gibbs_acc`: 0.942
- Next: review_E001_then_gamma_sweep

## E002-E005 compact gamma sweep queued - 2026-06-05 14:58:20

- Mode: single-process sequential queue
- Runs: E002 gamma 1.05/1.05; E003 gamma 1.15/1.15; E004 gamma 1.20/1.20; E005 gamma_h/gamma_l 1.10/1.20
- Queue log: runs_twoport1024_gamma_sweep_E002_E005_queue.log

## E002 started - 2026-06-05 14:58:43
- Purpose: compact_gamma_sweep
- Change: gamma_h=1.05,gamma_l=1.05
- Output: `./runs_twoport1024_E002_gamma105`
- Config: total=1024, hidden=574, gamma_h=1.05, gamma_l=1.05, lr=0.0002, cd_k=3, distill_weight=0.0
- Command: `train_twoport_1024_optimization_wsd.py --out_dir ./runs_twoport1024_E002_gamma105 --experiment_id E002 --purpose compact_gamma_sweep --change_note gamma_h=1.05,gamma_l=1.05 --next_note compare_compact_gamma_sweep --epochs 80 --early_stop_patience 8 --eval_every 2 --quick_eval_steps 800 --quick_eval_burn_in 100 --quick_eval_thin 2 --full_eval_on_best --full_eval_steps 3000 --full_eval_burn_in 500 --full_eval_thin 2 --batch_size 50 --eval_batch_size 128 --cd_k 3 --lr 0.0002 --momentum 0.6 --weight_decay 0.0 --gamma_h 1.05 --gamma_l 1.05 --label_inhibit 0.3 --label_update binary --label_init random_onehot --audio_layout time40_fold --audio_scale zscore_sigmoid --num_workers 2`

## E002 completed - 2026-06-05 16:46:31
- Output: `./runs_twoport1024_E002_gamma105`
- Best selection metric: 0.941800 at epoch 34
- Final epoch: 50
- Final full `test_label_gibbs_acc`: 0.9413
- Next: compare_compact_gamma_sweep

## E002 queue summary - 2026-06-05 16:46:32

- Output: ./runs_twoport1024_E002_gamma105
- Best selection metric: 0.9418
- Best epoch: 34
- Final epoch: 50
- Final full test_label_gibbs_acc: 0.9413
- stderr bytes: 0

## E003 started - 2026-06-05 16:46:36
- Purpose: compact_gamma_sweep
- Change: gamma_h=1.15,gamma_l=1.15
- Output: `./runs_twoport1024_E003_gamma115`
- Config: total=1024, hidden=574, gamma_h=1.15, gamma_l=1.15, lr=0.0002, cd_k=3, distill_weight=0.0
- Command: `train_twoport_1024_optimization_wsd.py --out_dir ./runs_twoport1024_E003_gamma115 --experiment_id E003 --purpose compact_gamma_sweep --change_note gamma_h=1.15,gamma_l=1.15 --next_note compare_compact_gamma_sweep --epochs 80 --early_stop_patience 8 --eval_every 2 --quick_eval_steps 800 --quick_eval_burn_in 100 --quick_eval_thin 2 --full_eval_on_best --full_eval_steps 3000 --full_eval_burn_in 500 --full_eval_thin 2 --batch_size 50 --eval_batch_size 128 --cd_k 3 --lr 0.0002 --momentum 0.6 --weight_decay 0.0 --gamma_h 1.15 --gamma_l 1.15 --label_inhibit 0.3 --label_update binary --label_init random_onehot --audio_layout time40_fold --audio_scale zscore_sigmoid --num_workers 2`

## E003 completed - 2026-06-05 17:44:57
- Output: `./runs_twoport1024_E003_gamma115`
- Best selection metric: 0.943800 at epoch 34
- Final epoch: 50
- Final full `test_label_gibbs_acc`: 0.9422
- Next: compare_compact_gamma_sweep

## E003 queue summary - 2026-06-05 17:44:58

- Output: ./runs_twoport1024_E003_gamma115
- Best selection metric: 0.9438
- Best epoch: 34
- Final epoch: 50
- Final full test_label_gibbs_acc: 0.9422
- stderr bytes: 0

## E004 started - 2026-06-05 17:45:02
- Purpose: compact_gamma_sweep
- Change: gamma_h=1.20,gamma_l=1.20
- Output: `./runs_twoport1024_E004_gamma120`
- Config: total=1024, hidden=574, gamma_h=1.2, gamma_l=1.2, lr=0.0002, cd_k=3, distill_weight=0.0
- Command: `train_twoport_1024_optimization_wsd.py --out_dir ./runs_twoport1024_E004_gamma120 --experiment_id E004 --purpose compact_gamma_sweep --change_note gamma_h=1.20,gamma_l=1.20 --next_note compare_compact_gamma_sweep --epochs 80 --early_stop_patience 8 --eval_every 2 --quick_eval_steps 800 --quick_eval_burn_in 100 --quick_eval_thin 2 --full_eval_on_best --full_eval_steps 3000 --full_eval_burn_in 500 --full_eval_thin 2 --batch_size 50 --eval_batch_size 128 --cd_k 3 --lr 0.0002 --momentum 0.6 --weight_decay 0.0 --gamma_h 1.20 --gamma_l 1.20 --label_inhibit 0.3 --label_update binary --label_init random_onehot --audio_layout time40_fold --audio_scale zscore_sigmoid --num_workers 2`

## E004 completed - 2026-06-05 19:17:22
- Output: `./runs_twoport1024_E004_gamma120`
- Best selection metric: 0.941400 at epoch 28
- Final epoch: 44
- Final full `test_label_gibbs_acc`: 0.9418
- Next: compare_compact_gamma_sweep

## E004 queue summary - 2026-06-05 19:17:23

- Output: ./runs_twoport1024_E004_gamma120
- Best selection metric: 0.9414
- Best epoch: 28
- Final epoch: 44
- Final full test_label_gibbs_acc: 0.9418
- stderr bytes: 0

## E005 started - 2026-06-05 19:17:27
- Purpose: compact_gamma_sweep
- Change: gamma_h=1.10,gamma_l=1.20
- Output: `./runs_twoport1024_E005_gamma_h110_l120`
- Config: total=1024, hidden=574, gamma_h=1.1, gamma_l=1.2, lr=0.0002, cd_k=3, distill_weight=0.0
- Command: `train_twoport_1024_optimization_wsd.py --out_dir ./runs_twoport1024_E005_gamma_h110_l120 --experiment_id E005 --purpose compact_gamma_sweep --change_note gamma_h=1.10,gamma_l=1.20 --next_note compare_compact_gamma_sweep --epochs 80 --early_stop_patience 8 --eval_every 2 --quick_eval_steps 800 --quick_eval_burn_in 100 --quick_eval_thin 2 --full_eval_on_best --full_eval_steps 3000 --full_eval_burn_in 500 --full_eval_thin 2 --batch_size 50 --eval_batch_size 128 --cd_k 3 --lr 0.0002 --momentum 0.6 --weight_decay 0.0 --gamma_h 1.10 --gamma_l 1.20 --label_inhibit 0.3 --label_update binary --label_init random_onehot --audio_layout time40_fold --audio_scale zscore_sigmoid --num_workers 2`

## server next-experiment analysis - 2026-06-05 19:44:39
- Best completed compact run: E003
- Best final_full_test_label_gibbs_acc: 0.9422
- Best selection metric: 0.9438
- Checkpoint: E:\BaiduSyncdisk\p-bit\python\high_order_DBM\runs_twoport1024_E003_gamma115\best.pt
- Next default: teacher generation, then distillation weights 0.02/0.05/0.10

## E005 completed - 2026-06-05 21:10:08
- Output: `./runs_twoport1024_E005_gamma_h110_l120`
- Best selection metric: 0.943600 at epoch 34
- Final epoch: 50
- Final full `test_label_gibbs_acc`: 0.9422
- Next: compare_compact_gamma_sweep

## E005 queue summary - 2026-06-05 21:10:09

- Output: ./runs_twoport1024_E005_gamma_h110_l120
- Best selection metric: 0.9436
- Best epoch: 34
- Final epoch: 50
- Final full test_label_gibbs_acc: 0.9422
- stderr bytes: 0

## E002-E005 compact gamma sweep completed - 2026-06-05 21:10:09

- Next: compare E001-E005 and choose best checkpoint for teacher generation or expanded gamma search.

## server next-experiment analysis - 2026-06-05 21:46:57
- Best completed compact run: E003
- Best final_full_test_label_gibbs_acc: 0.9422
- Best selection metric: 0.9438
- Checkpoint: /home/Hongjie_Zeng/high_order_BM/runs_twoport1024_E003_gamma115/best.pt
- Next default: teacher generation, then distillation weights 0.02/0.05/0.10

## TEACHER_LF05 teacher generation started - 2026-06-05 21:47:00
- Output: `/home/Hongjie_Zeng/high_order_BM/runs_twoport1024_teacher_latefusion_lam05/latefusion_teacher_lam05_train_test.npz`
- Image checkpoint: `./runs_rbm_wsd_lc5_p1000_20x20_mnist20crop_e100/best.pt`
- Audio checkpoint: `./runs_audioonly_mlp_raw507_zsig/best.pt`
- lambda_audio: 0.5
- eval: steps=3000, burn_in=500, thin=2

## TEACHER_LF05 teacher generation completed - 2026-06-05 21:50:10
- Output: `/home/Hongjie_Zeng/high_order_BM/runs_twoport1024_teacher_latefusion_lam05/latefusion_teacher_lam05_train_test.npz`
- Train teacher acc: 0.996367
- Test teacher acc: 0.986100
- Train image/audio acc: 0.929050 / 0.995750
- Test image/audio acc: 0.930200 / 0.833800

## E006 started - 2026-06-05 21:50:16
- Purpose: teacher_distillation_from_best_compact_gamma
- Change: warm_start=E003,distill_weight=0.02,lr=0.0001
- Output: `/home/Hongjie_Zeng/high_order_BM/runs_twoport1024_E006_distill_w0p02`
- Config: total=1024, hidden=574, gamma_h=1.15, gamma_l=1.15, lr=0.0001, cd_k=3, distill_weight=0.02
- Command: `train_twoport_1024_optimization_wsd.py --out_dir /home/Hongjie_Zeng/high_order_BM/runs_twoport1024_E006_distill_w0p02 --experiment_id E006 --purpose teacher_distillation_from_best_compact_gamma --change_note warm_start=E003,distill_weight=0.02,lr=0.0001 --next_note compare_distillation_runs --warm_start_ckpt /home/Hongjie_Zeng/high_order_BM/runs_twoport1024_E003_gamma115/best.pt --teacher_scores_npz /home/Hongjie_Zeng/high_order_BM/runs_twoport1024_teacher_latefusion_lam05/latefusion_teacher_lam05_train_test.npz --teacher_temperature 1.0 --distill_weight 0.02 --distill_start_epoch 1 --epochs 40 --early_stop_patience 6 --eval_every 2 --quick_eval_steps 800 --quick_eval_burn_in 100 --quick_eval_thin 2 --full_eval_on_best --full_eval_steps 3000 --full_eval_burn_in 500 --full_eval_thin 2 --batch_size 50 --eval_batch_size 128 --cd_k 3 --lr 0.0001 --momentum 0.6 --weight_decay 0.0 --gamma_h 1.15 --gamma_l 1.15 --label_inhibit 0.3 --label_update binary --label_init random_onehot --audio_layout time40_fold --audio_scale zscore_sigmoid --num_workers 2`

## E006 completed - 2026-06-05 22:00:08
- Output: `/home/Hongjie_Zeng/high_order_BM/runs_twoport1024_E006_distill_w0p02`
- Best selection metric: 0.942000 at epoch 4
- Final epoch: 16
- Final full `test_label_gibbs_acc`: 0.942
- Next: compare_distillation_runs

## E007 started - 2026-06-05 22:00:15
- Purpose: teacher_distillation_from_best_compact_gamma
- Change: warm_start=E003,distill_weight=0.05,lr=0.0001
- Output: `/home/Hongjie_Zeng/high_order_BM/runs_twoport1024_E007_distill_w0p05`
- Config: total=1024, hidden=574, gamma_h=1.15, gamma_l=1.15, lr=0.0001, cd_k=3, distill_weight=0.05
- Command: `train_twoport_1024_optimization_wsd.py --out_dir /home/Hongjie_Zeng/high_order_BM/runs_twoport1024_E007_distill_w0p05 --experiment_id E007 --purpose teacher_distillation_from_best_compact_gamma --change_note warm_start=E003,distill_weight=0.05,lr=0.0001 --next_note compare_distillation_runs --warm_start_ckpt /home/Hongjie_Zeng/high_order_BM/runs_twoport1024_E003_gamma115/best.pt --teacher_scores_npz /home/Hongjie_Zeng/high_order_BM/runs_twoport1024_teacher_latefusion_lam05/latefusion_teacher_lam05_train_test.npz --teacher_temperature 1.0 --distill_weight 0.05 --distill_start_epoch 1 --epochs 40 --early_stop_patience 6 --eval_every 2 --quick_eval_steps 800 --quick_eval_burn_in 100 --quick_eval_thin 2 --full_eval_on_best --full_eval_steps 3000 --full_eval_burn_in 500 --full_eval_thin 2 --batch_size 50 --eval_batch_size 128 --cd_k 3 --lr 0.0001 --momentum 0.6 --weight_decay 0.0 --gamma_h 1.15 --gamma_l 1.15 --label_inhibit 0.3 --label_update binary --label_init random_onehot --audio_layout time40_fold --audio_scale zscore_sigmoid --num_workers 2`

## E007 completed - 2026-06-05 22:10:02
- Output: `/home/Hongjie_Zeng/high_order_BM/runs_twoport1024_E007_distill_w0p05`
- Best selection metric: 0.942100 at epoch 4
- Final epoch: 16
- Final full `test_label_gibbs_acc`: 0.9419
- Next: compare_distillation_runs

## E008 started - 2026-06-05 22:10:08
- Purpose: teacher_distillation_from_best_compact_gamma
- Change: warm_start=E003,distill_weight=0.1,lr=0.0001
- Output: `/home/Hongjie_Zeng/high_order_BM/runs_twoport1024_E008_distill_w0p1`
- Config: total=1024, hidden=574, gamma_h=1.15, gamma_l=1.15, lr=0.0001, cd_k=3, distill_weight=0.1
- Command: `train_twoport_1024_optimization_wsd.py --out_dir /home/Hongjie_Zeng/high_order_BM/runs_twoport1024_E008_distill_w0p1 --experiment_id E008 --purpose teacher_distillation_from_best_compact_gamma --change_note warm_start=E003,distill_weight=0.1,lr=0.0001 --next_note compare_distillation_runs --warm_start_ckpt /home/Hongjie_Zeng/high_order_BM/runs_twoport1024_E003_gamma115/best.pt --teacher_scores_npz /home/Hongjie_Zeng/high_order_BM/runs_twoport1024_teacher_latefusion_lam05/latefusion_teacher_lam05_train_test.npz --teacher_temperature 1.0 --distill_weight 0.1 --distill_start_epoch 1 --epochs 40 --early_stop_patience 6 --eval_every 2 --quick_eval_steps 800 --quick_eval_burn_in 100 --quick_eval_thin 2 --full_eval_on_best --full_eval_steps 3000 --full_eval_burn_in 500 --full_eval_thin 2 --batch_size 50 --eval_batch_size 128 --cd_k 3 --lr 0.0001 --momentum 0.6 --weight_decay 0.0 --gamma_h 1.15 --gamma_l 1.15 --label_inhibit 0.3 --label_update binary --label_init random_onehot --audio_layout time40_fold --audio_scale zscore_sigmoid --num_workers 2`

## E008 completed - 2026-06-05 22:20:03
- Output: `/home/Hongjie_Zeng/high_order_BM/runs_twoport1024_E008_distill_w0p1`
- Best selection metric: 0.941600 at epoch 4
- Final epoch: 16
- Final full `test_label_gibbs_acc`: 0.9416
- Next: compare_distillation_runs

## server distillation batch completed - 2026-06-05 22:20:04
- Warm start compact run: E003
- Teacher NPZ: /home/Hongjie_Zeng/high_order_BM/runs_twoport1024_teacher_latefusion_lam05/latefusion_teacher_lam05_train_test.npz
- Completed distillation IDs: E006/E007/E008

