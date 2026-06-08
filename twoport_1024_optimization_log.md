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

## E009-E020 processed feature batch started - 2026-06-06 03:17:37
- Strategy: feed deployable posterior-pattern features into the p-bit Gibbs model, instead of using teacher KL as an auxiliary loss.
- Feature NPZ: `/home/Hongjie_Zeng/high_order_BM/runs_twoport1024_teacher_latefusion_lam05/latefusion_teacher_lam05_train_test.npz`
- Pattern: interleave

## E009 started - 2026-06-06 03:17:42
- Purpose: processed_audio_input
- Change: optical=raw,audio=audio_mlp_probs
- Output: `/home/Hongjie_Zeng/high_order_BM/runs_twoport1024_E009_audio_mlp_input`
- Config: total=1024, hidden=574, gamma_h=1.15, gamma_l=1.15, lr=0.0002, cd_k=3, distill_weight=0.0
- Command: `train_twoport_1024_optimization_wsd.py --out_dir /home/Hongjie_Zeng/high_order_BM/runs_twoport1024_E009_audio_mlp_input --experiment_id E009 --purpose processed_audio_input --change_note optical=raw,audio=audio_mlp_probs --next_note compare_processed_feature_inputs --processed_feature_npz /home/Hongjie_Zeng/high_order_BM/runs_twoport1024_teacher_latefusion_lam05/latefusion_teacher_lam05_train_test.npz --optical_feature_source raw --audio_feature_source audio_mlp_probs --processed_feature_pattern interleave --processed_mix 0.5 --epochs 100 --early_stop_patience 10 --eval_every 2 --quick_eval_steps 800 --quick_eval_burn_in 100 --quick_eval_thin 2 --full_eval_on_best --full_eval_steps 3000 --full_eval_burn_in 500 --full_eval_thin 2 --batch_size 50 --eval_batch_size 128 --cd_k 3 --lr 0.0002 --momentum 0.6 --weight_decay 0.0 --gamma_h 1.15 --gamma_l 1.15 --label_inhibit 0.3 --label_update binary --label_init random_onehot --audio_layout time40_fold --audio_scale zscore_sigmoid --num_workers 2`

## E009 completed - 2026-06-06 03:26:39
- Output: `/home/Hongjie_Zeng/high_order_BM/runs_twoport1024_E009_audio_mlp_input`
- Best selection metric: 0.864900 at epoch 2
- Final epoch: 22
- Final full `test_label_gibbs_acc`: 0.8654
- Next: compare_processed_feature_inputs

## E009 processed feature queue summary - 2026-06-06 03:26:40
- Output: `/home/Hongjie_Zeng/high_order_BM/runs_twoport1024_E009_audio_mlp_input`
- Optical/audio sources: raw / audio_mlp_probs
- Pattern/mix: interleave / 0.5
- Best selection metric: 0.8649
- Best epoch: 2
- Final full test_label_gibbs_acc: 0.8654

## E010 started - 2026-06-06 03:26:44
- Purpose: processed_optical_input
- Change: optical=image_rbm_probs,audio=raw
- Output: `/home/Hongjie_Zeng/high_order_BM/runs_twoport1024_E010_optical_image_rbm_input`
- Config: total=1024, hidden=574, gamma_h=1.15, gamma_l=1.15, lr=0.0002, cd_k=3, distill_weight=0.0
- Command: `train_twoport_1024_optimization_wsd.py --out_dir /home/Hongjie_Zeng/high_order_BM/runs_twoport1024_E010_optical_image_rbm_input --experiment_id E010 --purpose processed_optical_input --change_note optical=image_rbm_probs,audio=raw --next_note compare_processed_feature_inputs --processed_feature_npz /home/Hongjie_Zeng/high_order_BM/runs_twoport1024_teacher_latefusion_lam05/latefusion_teacher_lam05_train_test.npz --optical_feature_source image_rbm_probs --audio_feature_source raw --processed_feature_pattern interleave --processed_mix 0.5 --epochs 100 --early_stop_patience 10 --eval_every 2 --quick_eval_steps 800 --quick_eval_burn_in 100 --quick_eval_thin 2 --full_eval_on_best --full_eval_steps 3000 --full_eval_burn_in 500 --full_eval_thin 2 --batch_size 50 --eval_batch_size 128 --cd_k 3 --lr 0.0002 --momentum 0.6 --weight_decay 0.0 --gamma_h 1.15 --gamma_l 1.15 --label_inhibit 0.3 --label_update binary --label_init random_onehot --audio_layout time40_fold --audio_scale zscore_sigmoid --num_workers 2`

## E010 completed - 2026-06-06 04:44:04
- Output: `/home/Hongjie_Zeng/high_order_BM/runs_twoport1024_E010_optical_image_rbm_input`
- Best selection metric: 0.950500 at epoch 86
- Final epoch: 100
- Final full `test_label_gibbs_acc`: 0.9511
- Next: compare_processed_feature_inputs

## E010 processed feature queue summary - 2026-06-06 04:44:04
- Output: `/home/Hongjie_Zeng/high_order_BM/runs_twoport1024_E010_optical_image_rbm_input`
- Optical/audio sources: image_rbm_probs / raw
- Pattern/mix: interleave / 0.5
- Best selection metric: 0.9505
- Best epoch: 86
- Final full test_label_gibbs_acc: 0.9511

## E011 started - 2026-06-06 04:44:09
- Purpose: processed_dual_input
- Change: optical=image_rbm_probs,audio=audio_mlp_probs
- Output: `/home/Hongjie_Zeng/high_order_BM/runs_twoport1024_E011_dual_posterior_input`
- Config: total=1024, hidden=574, gamma_h=1.15, gamma_l=1.15, lr=0.0002, cd_k=3, distill_weight=0.0
- Command: `train_twoport_1024_optimization_wsd.py --out_dir /home/Hongjie_Zeng/high_order_BM/runs_twoport1024_E011_dual_posterior_input --experiment_id E011 --purpose processed_dual_input --change_note optical=image_rbm_probs,audio=audio_mlp_probs --next_note compare_processed_feature_inputs --processed_feature_npz /home/Hongjie_Zeng/high_order_BM/runs_twoport1024_teacher_latefusion_lam05/latefusion_teacher_lam05_train_test.npz --optical_feature_source image_rbm_probs --audio_feature_source audio_mlp_probs --processed_feature_pattern interleave --processed_mix 0.5 --epochs 100 --early_stop_patience 10 --eval_every 2 --quick_eval_steps 800 --quick_eval_burn_in 100 --quick_eval_thin 2 --full_eval_on_best --full_eval_steps 3000 --full_eval_burn_in 500 --full_eval_thin 2 --batch_size 50 --eval_batch_size 128 --cd_k 3 --lr 0.0002 --momentum 0.6 --weight_decay 0.0 --gamma_h 1.15 --gamma_l 1.15 --label_inhibit 0.3 --label_update binary --label_init random_onehot --audio_layout time40_fold --audio_scale zscore_sigmoid --num_workers 2`

## E011 completed - 2026-06-06 04:53:43
- Output: `/home/Hongjie_Zeng/high_order_BM/runs_twoport1024_E011_dual_posterior_input`
- Best selection metric: 0.921400 at epoch 2
- Final epoch: 22
- Final full `test_label_gibbs_acc`: 0.9213
- Next: compare_processed_feature_inputs

## E011 processed feature queue summary - 2026-06-06 04:53:43
- Output: `/home/Hongjie_Zeng/high_order_BM/runs_twoport1024_E011_dual_posterior_input`
- Optical/audio sources: image_rbm_probs / audio_mlp_probs
- Pattern/mix: interleave / 0.5
- Best selection metric: 0.9214
- Best epoch: 2
- Final full test_label_gibbs_acc: 0.9213

## E012 started - 2026-06-06 04:53:47
- Purpose: hybrid_processed_optical_input
- Change: optical=raw_plus_image_rbm_probs,audio=audio_mlp_probs,mix=0.25
- Output: `/home/Hongjie_Zeng/high_order_BM/runs_twoport1024_E012_hybrid_mix025_audio_mlp_input`
- Config: total=1024, hidden=574, gamma_h=1.15, gamma_l=1.15, lr=0.0002, cd_k=3, distill_weight=0.0
- Command: `train_twoport_1024_optimization_wsd.py --out_dir /home/Hongjie_Zeng/high_order_BM/runs_twoport1024_E012_hybrid_mix025_audio_mlp_input --experiment_id E012 --purpose hybrid_processed_optical_input --change_note optical=raw_plus_image_rbm_probs,audio=audio_mlp_probs,mix=0.25 --next_note compare_processed_feature_inputs --processed_feature_npz /home/Hongjie_Zeng/high_order_BM/runs_twoport1024_teacher_latefusion_lam05/latefusion_teacher_lam05_train_test.npz --optical_feature_source raw_plus_image_rbm_probs --audio_feature_source audio_mlp_probs --processed_feature_pattern interleave --processed_mix 0.25 --epochs 100 --early_stop_patience 10 --eval_every 2 --quick_eval_steps 800 --quick_eval_burn_in 100 --quick_eval_thin 2 --full_eval_on_best --full_eval_steps 3000 --full_eval_burn_in 500 --full_eval_thin 2 --batch_size 50 --eval_batch_size 128 --cd_k 3 --lr 0.0002 --momentum 0.6 --weight_decay 0.0 --gamma_h 1.15 --gamma_l 1.15 --label_inhibit 0.3 --label_update binary --label_init random_onehot --audio_layout time40_fold --audio_scale zscore_sigmoid --num_workers 2`

## E012 completed - 2026-06-06 06:03:14
- Output: `/home/Hongjie_Zeng/high_order_BM/runs_twoport1024_E012_hybrid_mix025_audio_mlp_input`
- Best selection metric: 0.875000 at epoch 88
- Final epoch: 100
- Final full `test_label_gibbs_acc`: 0.8743
- Next: compare_processed_feature_inputs

## E012 processed feature queue summary - 2026-06-06 06:03:14
- Output: `/home/Hongjie_Zeng/high_order_BM/runs_twoport1024_E012_hybrid_mix025_audio_mlp_input`
- Optical/audio sources: raw_plus_image_rbm_probs / audio_mlp_probs
- Pattern/mix: interleave / 0.25
- Best selection metric: 0.875
- Best epoch: 88
- Final full test_label_gibbs_acc: 0.8743

## E013 started - 2026-06-06 06:03:19
- Purpose: hybrid_processed_optical_input
- Change: optical=raw_plus_image_rbm_probs,audio=audio_mlp_probs,mix=0.50
- Output: `/home/Hongjie_Zeng/high_order_BM/runs_twoport1024_E013_hybrid_mix050_audio_mlp_input`
- Config: total=1024, hidden=574, gamma_h=1.15, gamma_l=1.15, lr=0.0002, cd_k=3, distill_weight=0.0
- Command: `train_twoport_1024_optimization_wsd.py --out_dir /home/Hongjie_Zeng/high_order_BM/runs_twoport1024_E013_hybrid_mix050_audio_mlp_input --experiment_id E013 --purpose hybrid_processed_optical_input --change_note optical=raw_plus_image_rbm_probs,audio=audio_mlp_probs,mix=0.50 --next_note compare_processed_feature_inputs --processed_feature_npz /home/Hongjie_Zeng/high_order_BM/runs_twoport1024_teacher_latefusion_lam05/latefusion_teacher_lam05_train_test.npz --optical_feature_source raw_plus_image_rbm_probs --audio_feature_source audio_mlp_probs --processed_feature_pattern interleave --processed_mix 0.5 --epochs 100 --early_stop_patience 10 --eval_every 2 --quick_eval_steps 800 --quick_eval_burn_in 100 --quick_eval_thin 2 --full_eval_on_best --full_eval_steps 3000 --full_eval_burn_in 500 --full_eval_thin 2 --batch_size 50 --eval_batch_size 128 --cd_k 3 --lr 0.0002 --momentum 0.6 --weight_decay 0.0 --gamma_h 1.15 --gamma_l 1.15 --label_inhibit 0.3 --label_update binary --label_init random_onehot --audio_layout time40_fold --audio_scale zscore_sigmoid --num_workers 2`

## E013 completed - 2026-06-06 07:18:59
- Output: `/home/Hongjie_Zeng/high_order_BM/runs_twoport1024_E013_hybrid_mix050_audio_mlp_input`
- Best selection metric: 0.889400 at epoch 100
- Final epoch: 100
- Final full `test_label_gibbs_acc`: 0.8894
- Next: compare_processed_feature_inputs

## E013 processed feature queue summary - 2026-06-06 07:18:59
- Output: `/home/Hongjie_Zeng/high_order_BM/runs_twoport1024_E013_hybrid_mix050_audio_mlp_input`
- Optical/audio sources: raw_plus_image_rbm_probs / audio_mlp_probs
- Pattern/mix: interleave / 0.5
- Best selection metric: 0.8894
- Best epoch: 100
- Final full test_label_gibbs_acc: 0.8894

## E014 started - 2026-06-06 07:19:04
- Purpose: hybrid_processed_optical_input
- Change: optical=raw_plus_image_rbm_probs,audio=audio_mlp_probs,mix=0.75
- Output: `/home/Hongjie_Zeng/high_order_BM/runs_twoport1024_E014_hybrid_mix075_audio_mlp_input`
- Config: total=1024, hidden=574, gamma_h=1.15, gamma_l=1.15, lr=0.0002, cd_k=3, distill_weight=0.0
- Command: `train_twoport_1024_optimization_wsd.py --out_dir /home/Hongjie_Zeng/high_order_BM/runs_twoport1024_E014_hybrid_mix075_audio_mlp_input --experiment_id E014 --purpose hybrid_processed_optical_input --change_note optical=raw_plus_image_rbm_probs,audio=audio_mlp_probs,mix=0.75 --next_note compare_processed_feature_inputs --processed_feature_npz /home/Hongjie_Zeng/high_order_BM/runs_twoport1024_teacher_latefusion_lam05/latefusion_teacher_lam05_train_test.npz --optical_feature_source raw_plus_image_rbm_probs --audio_feature_source audio_mlp_probs --processed_feature_pattern interleave --processed_mix 0.75 --epochs 100 --early_stop_patience 10 --eval_every 2 --quick_eval_steps 800 --quick_eval_burn_in 100 --quick_eval_thin 2 --full_eval_on_best --full_eval_steps 3000 --full_eval_burn_in 500 --full_eval_thin 2 --batch_size 50 --eval_batch_size 128 --cd_k 3 --lr 0.0002 --momentum 0.6 --weight_decay 0.0 --gamma_h 1.15 --gamma_l 1.15 --label_inhibit 0.3 --label_update binary --label_init random_onehot --audio_layout time40_fold --audio_scale zscore_sigmoid --num_workers 2`

## E014 completed - 2026-06-06 07:28:42
- Output: `/home/Hongjie_Zeng/high_order_BM/runs_twoport1024_E014_hybrid_mix075_audio_mlp_input`
- Best selection metric: 0.891000 at epoch 2
- Final epoch: 22
- Final full `test_label_gibbs_acc`: 0.8907
- Next: compare_processed_feature_inputs

## E014 processed feature queue summary - 2026-06-06 07:28:42
- Output: `/home/Hongjie_Zeng/high_order_BM/runs_twoport1024_E014_hybrid_mix075_audio_mlp_input`
- Optical/audio sources: raw_plus_image_rbm_probs / audio_mlp_probs
- Pattern/mix: interleave / 0.75
- Best selection metric: 0.891
- Best epoch: 2
- Final full test_label_gibbs_acc: 0.8907

## E015 started - 2026-06-06 07:28:47
- Purpose: processed_dual_input_pattern_ablation
- Change: optical=image_rbm_probs,audio=audio_mlp_probs,pattern=blocks
- Output: `/home/Hongjie_Zeng/high_order_BM/runs_twoport1024_E015_dual_posterior_blocks`
- Config: total=1024, hidden=574, gamma_h=1.15, gamma_l=1.15, lr=0.0002, cd_k=3, distill_weight=0.0
- Command: `train_twoport_1024_optimization_wsd.py --out_dir /home/Hongjie_Zeng/high_order_BM/runs_twoport1024_E015_dual_posterior_blocks --experiment_id E015 --purpose processed_dual_input_pattern_ablation --change_note optical=image_rbm_probs,audio=audio_mlp_probs,pattern=blocks --next_note compare_processed_feature_inputs --processed_feature_npz /home/Hongjie_Zeng/high_order_BM/runs_twoport1024_teacher_latefusion_lam05/latefusion_teacher_lam05_train_test.npz --optical_feature_source image_rbm_probs --audio_feature_source audio_mlp_probs --processed_feature_pattern blocks --processed_mix 0.5 --epochs 100 --early_stop_patience 10 --eval_every 2 --quick_eval_steps 800 --quick_eval_burn_in 100 --quick_eval_thin 2 --full_eval_on_best --full_eval_steps 3000 --full_eval_burn_in 500 --full_eval_thin 2 --batch_size 50 --eval_batch_size 128 --cd_k 3 --lr 0.0002 --momentum 0.6 --weight_decay 0.0 --gamma_h 1.15 --gamma_l 1.15 --label_inhibit 0.3 --label_update binary --label_init random_onehot --audio_layout time40_fold --audio_scale zscore_sigmoid --num_workers 2`

## E015 completed - 2026-06-06 07:38:12
- Output: `/home/Hongjie_Zeng/high_order_BM/runs_twoport1024_E015_dual_posterior_blocks`
- Best selection metric: 0.925600 at epoch 2
- Final epoch: 22
- Final full `test_label_gibbs_acc`: 0.9256
- Next: compare_processed_feature_inputs

## E015 processed feature queue summary - 2026-06-06 07:38:13
- Output: `/home/Hongjie_Zeng/high_order_BM/runs_twoport1024_E015_dual_posterior_blocks`
- Optical/audio sources: image_rbm_probs / audio_mlp_probs
- Pattern/mix: blocks / 0.5
- Best selection metric: 0.9256
- Best epoch: 2
- Final full test_label_gibbs_acc: 0.9256

## E016 started - 2026-06-06 07:38:17
- Purpose: hybrid_processed_optical_pattern_ablation
- Change: optical=raw_plus_image_rbm_probs,audio=audio_mlp_probs,mix=0.50,pattern=blocks
- Output: `/home/Hongjie_Zeng/high_order_BM/runs_twoport1024_E016_hybrid_mix050_blocks`
- Config: total=1024, hidden=574, gamma_h=1.15, gamma_l=1.15, lr=0.0002, cd_k=3, distill_weight=0.0
- Command: `train_twoport_1024_optimization_wsd.py --out_dir /home/Hongjie_Zeng/high_order_BM/runs_twoport1024_E016_hybrid_mix050_blocks --experiment_id E016 --purpose hybrid_processed_optical_pattern_ablation --change_note optical=raw_plus_image_rbm_probs,audio=audio_mlp_probs,mix=0.50,pattern=blocks --next_note compare_processed_feature_inputs --processed_feature_npz /home/Hongjie_Zeng/high_order_BM/runs_twoport1024_teacher_latefusion_lam05/latefusion_teacher_lam05_train_test.npz --optical_feature_source raw_plus_image_rbm_probs --audio_feature_source audio_mlp_probs --processed_feature_pattern blocks --processed_mix 0.5 --epochs 100 --early_stop_patience 10 --eval_every 2 --quick_eval_steps 800 --quick_eval_burn_in 100 --quick_eval_thin 2 --full_eval_on_best --full_eval_steps 3000 --full_eval_burn_in 500 --full_eval_thin 2 --batch_size 50 --eval_batch_size 128 --cd_k 3 --lr 0.0002 --momentum 0.6 --weight_decay 0.0 --gamma_h 1.15 --gamma_l 1.15 --label_inhibit 0.3 --label_update binary --label_init random_onehot --audio_layout time40_fold --audio_scale zscore_sigmoid --num_workers 2`

## E016 completed - 2026-06-06 08:47:45
- Output: `/home/Hongjie_Zeng/high_order_BM/runs_twoport1024_E016_hybrid_mix050_blocks`
- Best selection metric: 0.887600 at epoch 80
- Final epoch: 100
- Final full `test_label_gibbs_acc`: 0.8876
- Next: compare_processed_feature_inputs

## E016 processed feature queue summary - 2026-06-06 08:47:46
- Output: `/home/Hongjie_Zeng/high_order_BM/runs_twoport1024_E016_hybrid_mix050_blocks`
- Optical/audio sources: raw_plus_image_rbm_probs / audio_mlp_probs
- Pattern/mix: blocks / 0.5
- Best selection metric: 0.8876
- Best epoch: 80
- Final full test_label_gibbs_acc: 0.8876

## E017 started - 2026-06-06 08:47:50
- Purpose: hybrid_processed_both_channels
- Change: optical=raw_plus_image_rbm_probs,audio=raw_plus_audio_mlp_probs,mix=0.35
- Output: `/home/Hongjie_Zeng/high_order_BM/runs_twoport1024_E017_raw_plus_both_mix035`
- Config: total=1024, hidden=574, gamma_h=1.15, gamma_l=1.15, lr=0.0002, cd_k=3, distill_weight=0.0
- Command: `train_twoport_1024_optimization_wsd.py --out_dir /home/Hongjie_Zeng/high_order_BM/runs_twoport1024_E017_raw_plus_both_mix035 --experiment_id E017 --purpose hybrid_processed_both_channels --change_note optical=raw_plus_image_rbm_probs,audio=raw_plus_audio_mlp_probs,mix=0.35 --next_note compare_processed_feature_inputs --processed_feature_npz /home/Hongjie_Zeng/high_order_BM/runs_twoport1024_teacher_latefusion_lam05/latefusion_teacher_lam05_train_test.npz --optical_feature_source raw_plus_image_rbm_probs --audio_feature_source raw_plus_audio_mlp_probs --processed_feature_pattern interleave --processed_mix 0.35 --epochs 100 --early_stop_patience 10 --eval_every 2 --quick_eval_steps 800 --quick_eval_burn_in 100 --quick_eval_thin 2 --full_eval_on_best --full_eval_steps 3000 --full_eval_burn_in 500 --full_eval_thin 2 --batch_size 50 --eval_batch_size 128 --cd_k 3 --lr 0.0002 --momentum 0.6 --weight_decay 0.0 --gamma_h 1.15 --gamma_l 1.15 --label_inhibit 0.3 --label_update binary --label_init random_onehot --audio_layout time40_fold --audio_scale zscore_sigmoid --num_workers 2`

## E017 completed - 2026-06-06 09:01:22
- Output: `/home/Hongjie_Zeng/high_order_BM/runs_twoport1024_E017_raw_plus_both_mix035`
- Best selection metric: 0.973200 at epoch 6
- Final epoch: 26
- Final full `test_label_gibbs_acc`: 0.9731
- Next: compare_processed_feature_inputs

## E017 processed feature queue summary - 2026-06-06 09:01:22
- Output: `/home/Hongjie_Zeng/high_order_BM/runs_twoport1024_E017_raw_plus_both_mix035`
- Optical/audio sources: raw_plus_image_rbm_probs / raw_plus_audio_mlp_probs
- Pattern/mix: interleave / 0.35
- Best selection metric: 0.9732
- Best epoch: 6
- Final full test_label_gibbs_acc: 0.9731

## E018 started - 2026-06-06 09:01:26
- Purpose: teacher_posterior_input_diagnostic
- Change: optical=teacher_probs,audio=raw
- Output: `/home/Hongjie_Zeng/high_order_BM/runs_twoport1024_E018_teacher_optical_raw_audio_diagnostic`
- Config: total=1024, hidden=574, gamma_h=1.15, gamma_l=1.15, lr=0.0002, cd_k=3, distill_weight=0.0
- Command: `train_twoport_1024_optimization_wsd.py --out_dir /home/Hongjie_Zeng/high_order_BM/runs_twoport1024_E018_teacher_optical_raw_audio_diagnostic --experiment_id E018 --purpose teacher_posterior_input_diagnostic --change_note optical=teacher_probs,audio=raw --next_note compare_processed_feature_inputs --processed_feature_npz /home/Hongjie_Zeng/high_order_BM/runs_twoport1024_teacher_latefusion_lam05/latefusion_teacher_lam05_train_test.npz --optical_feature_source teacher_probs --audio_feature_source raw --processed_feature_pattern interleave --processed_mix 0.5 --epochs 100 --early_stop_patience 10 --eval_every 2 --quick_eval_steps 800 --quick_eval_burn_in 100 --quick_eval_thin 2 --full_eval_on_best --full_eval_steps 3000 --full_eval_burn_in 500 --full_eval_thin 2 --batch_size 50 --eval_batch_size 128 --cd_k 3 --lr 0.0002 --momentum 0.6 --weight_decay 0.0 --gamma_h 1.15 --gamma_l 1.15 --label_inhibit 0.3 --label_update binary --label_init random_onehot --audio_layout time40_fold --audio_scale zscore_sigmoid --num_workers 2`

## E018 completed - 2026-06-06 09:17:03
- Output: `/home/Hongjie_Zeng/high_order_BM/runs_twoport1024_E018_teacher_optical_raw_audio_diagnostic`
- Best selection metric: 0.986400 at epoch 8
- Final epoch: 28
- Final full `test_label_gibbs_acc`: 0.9864
- Next: compare_processed_feature_inputs

## E018 processed feature queue summary - 2026-06-06 09:17:04
- Output: `/home/Hongjie_Zeng/high_order_BM/runs_twoport1024_E018_teacher_optical_raw_audio_diagnostic`
- Optical/audio sources: teacher_probs / raw
- Pattern/mix: interleave / 0.5
- Best selection metric: 0.9864
- Best epoch: 8
- Final full test_label_gibbs_acc: 0.9864

## E019 started - 2026-06-06 09:17:08
- Purpose: teacher_posterior_input_diagnostic
- Change: optical=teacher_probs,audio=audio_mlp_probs
- Output: `/home/Hongjie_Zeng/high_order_BM/runs_twoport1024_E019_teacher_optical_audio_mlp_diagnostic`
- Config: total=1024, hidden=574, gamma_h=1.15, gamma_l=1.15, lr=0.0002, cd_k=3, distill_weight=0.0
- Command: `train_twoport_1024_optimization_wsd.py --out_dir /home/Hongjie_Zeng/high_order_BM/runs_twoport1024_E019_teacher_optical_audio_mlp_diagnostic --experiment_id E019 --purpose teacher_posterior_input_diagnostic --change_note optical=teacher_probs,audio=audio_mlp_probs --next_note compare_processed_feature_inputs --processed_feature_npz /home/Hongjie_Zeng/high_order_BM/runs_twoport1024_teacher_latefusion_lam05/latefusion_teacher_lam05_train_test.npz --optical_feature_source teacher_probs --audio_feature_source audio_mlp_probs --processed_feature_pattern interleave --processed_mix 0.5 --epochs 100 --early_stop_patience 10 --eval_every 2 --quick_eval_steps 800 --quick_eval_burn_in 100 --quick_eval_thin 2 --full_eval_on_best --full_eval_steps 3000 --full_eval_burn_in 500 --full_eval_thin 2 --batch_size 50 --eval_batch_size 128 --cd_k 3 --lr 0.0002 --momentum 0.6 --weight_decay 0.0 --gamma_h 1.15 --gamma_l 1.15 --label_inhibit 0.3 --label_update binary --label_init random_onehot --audio_layout time40_fold --audio_scale zscore_sigmoid --num_workers 2`

## E019 completed - 2026-06-06 09:39:21
- Output: `/home/Hongjie_Zeng/high_order_BM/runs_twoport1024_E019_teacher_optical_audio_mlp_diagnostic`
- Best selection metric: 0.951800 at epoch 20
- Final epoch: 40
- Final full `test_label_gibbs_acc`: 0.9514
- Next: compare_processed_feature_inputs

## E019 processed feature queue summary - 2026-06-06 09:39:22
- Output: `/home/Hongjie_Zeng/high_order_BM/runs_twoport1024_E019_teacher_optical_audio_mlp_diagnostic`
- Optical/audio sources: teacher_probs / audio_mlp_probs
- Pattern/mix: interleave / 0.5
- Best selection metric: 0.9518
- Best epoch: 20
- Final full test_label_gibbs_acc: 0.9514

## E020 started - 2026-06-06 09:39:26
- Purpose: teacher_hybrid_input_diagnostic
- Change: optical=raw_plus_teacher_probs,audio=audio_mlp_probs,mix=0.35
- Output: `/home/Hongjie_Zeng/high_order_BM/runs_twoport1024_E020_raw_plus_teacher_mix035_diagnostic`
- Config: total=1024, hidden=574, gamma_h=1.15, gamma_l=1.15, lr=0.0002, cd_k=3, distill_weight=0.0
- Command: `train_twoport_1024_optimization_wsd.py --out_dir /home/Hongjie_Zeng/high_order_BM/runs_twoport1024_E020_raw_plus_teacher_mix035_diagnostic --experiment_id E020 --purpose teacher_hybrid_input_diagnostic --change_note optical=raw_plus_teacher_probs,audio=audio_mlp_probs,mix=0.35 --next_note compare_processed_feature_inputs --processed_feature_npz /home/Hongjie_Zeng/high_order_BM/runs_twoport1024_teacher_latefusion_lam05/latefusion_teacher_lam05_train_test.npz --optical_feature_source raw_plus_teacher_probs --audio_feature_source audio_mlp_probs --processed_feature_pattern interleave --processed_mix 0.35 --epochs 100 --early_stop_patience 10 --eval_every 2 --quick_eval_steps 800 --quick_eval_burn_in 100 --quick_eval_thin 2 --full_eval_on_best --full_eval_steps 3000 --full_eval_burn_in 500 --full_eval_thin 2 --batch_size 50 --eval_batch_size 128 --cd_k 3 --lr 0.0002 --momentum 0.6 --weight_decay 0.0 --gamma_h 1.15 --gamma_l 1.15 --label_inhibit 0.3 --label_update binary --label_init random_onehot --audio_layout time40_fold --audio_scale zscore_sigmoid --num_workers 2`

## E020 completed - 2026-06-06 10:53:32
- Output: `/home/Hongjie_Zeng/high_order_BM/runs_twoport1024_E020_raw_plus_teacher_mix035_diagnostic`
- Best selection metric: 0.886400 at epoch 94
- Final epoch: 100
- Final full `test_label_gibbs_acc`: 0.8866
- Next: compare_processed_feature_inputs

## E020 processed feature queue summary - 2026-06-06 10:53:33
- Output: `/home/Hongjie_Zeng/high_order_BM/runs_twoport1024_E020_raw_plus_teacher_mix035_diagnostic`
- Optical/audio sources: raw_plus_teacher_probs / audio_mlp_probs
- Pattern/mix: interleave / 0.35
- Best selection metric: 0.8864
- Best epoch: 94
- Final full test_label_gibbs_acc: 0.8866

## E009-E020 processed feature batch completed - 2026-06-06 10:53:33
- Next: choose best processed input; if >=95%, refine pattern/mix around it; otherwise try a learned 400-d optical projection.

## E021-E038 E017 confirmation batch started - 2026-06-06 18:05:03
- Strategy: confirm the 97.31% E017 result under unchanged 1024 p-bit, dual-channel physical input and full Gibbs inference.
- Feature NPZ: `/home/Hongjie_Zeng/high_order_BM/runs_twoport1024_teacher_latefusion_lam05/latefusion_teacher_lam05_train_test.npz`
- Base: optical=raw_plus_image_rbm_probs, audio=raw_plus_audio_mlp_probs, mix=0.35

## E021 started - 2026-06-06 18:05:08
- Purpose: e017_multiseed_confirmation
- Change: optical=raw_plus_image_rbm_probs,audio=raw_plus_audio_mlp_probs,mix=0.35,pattern=interleave,gamma_h=1.15,gamma_l=1.15,seed=124
- Output: `/home/Hongjie_Zeng/high_order_BM/runs_twoport1024_E021_e017_seed124`
- Config: total=1024, hidden=574, gamma_h=1.15, gamma_l=1.15, lr=0.0002, cd_k=3, distill_weight=0.0
- Command: `train_twoport_1024_optimization_wsd.py --out_dir /home/Hongjie_Zeng/high_order_BM/runs_twoport1024_E021_e017_seed124 --experiment_id E021 --purpose e017_multiseed_confirmation --change_note optical=raw_plus_image_rbm_probs,audio=raw_plus_audio_mlp_probs,mix=0.35,pattern=interleave,gamma_h=1.15,gamma_l=1.15,seed=124 --next_note compare_e017_confirmation_runs --processed_feature_npz /home/Hongjie_Zeng/high_order_BM/runs_twoport1024_teacher_latefusion_lam05/latefusion_teacher_lam05_train_test.npz --optical_feature_source raw_plus_image_rbm_probs --audio_feature_source raw_plus_audio_mlp_probs --processed_feature_pattern interleave --processed_mix 0.35 --seed 124 --epochs 100 --early_stop_patience 10 --eval_every 2 --quick_eval_steps 800 --quick_eval_burn_in 100 --quick_eval_thin 2 --full_eval_on_best --full_eval_steps 3000 --full_eval_burn_in 500 --full_eval_thin 2 --batch_size 50 --eval_batch_size 128 --cd_k 3 --lr 0.0002 --momentum 0.6 --weight_decay 0.0 --gamma_h 1.15 --gamma_l 1.15 --label_inhibit 0.3 --label_update binary --label_init random_onehot --audio_layout time40_fold --audio_scale zscore_sigmoid --num_workers 2`

## E021 completed - 2026-06-06 18:16:54
- Output: `/home/Hongjie_Zeng/high_order_BM/runs_twoport1024_E021_e017_seed124`
- Best selection metric: 0.976100 at epoch 4
- Final epoch: 24
- Final full `test_label_gibbs_acc`: 0.9761
- Next: compare_e017_confirmation_runs

## E021 E017 confirmation summary - 2026-06-06 18:16:54
- Output: `/home/Hongjie_Zeng/high_order_BM/runs_twoport1024_E021_e017_seed124`
- Change: optical=raw_plus_image_rbm_probs,audio=raw_plus_audio_mlp_probs,mix=0.35,pattern=interleave,gamma_h=1.15,gamma_l=1.15,seed=124
- Best selection metric: 0.9761
- Best epoch: 4
- Final full test_label_gibbs_acc: 0.9761

## E022 started - 2026-06-06 18:16:59
- Purpose: e017_multiseed_confirmation
- Change: optical=raw_plus_image_rbm_probs,audio=raw_plus_audio_mlp_probs,mix=0.35,pattern=interleave,gamma_h=1.15,gamma_l=1.15,seed=125
- Output: `/home/Hongjie_Zeng/high_order_BM/runs_twoport1024_E022_e017_seed125`
- Config: total=1024, hidden=574, gamma_h=1.15, gamma_l=1.15, lr=0.0002, cd_k=3, distill_weight=0.0
- Command: `train_twoport_1024_optimization_wsd.py --out_dir /home/Hongjie_Zeng/high_order_BM/runs_twoport1024_E022_e017_seed125 --experiment_id E022 --purpose e017_multiseed_confirmation --change_note optical=raw_plus_image_rbm_probs,audio=raw_plus_audio_mlp_probs,mix=0.35,pattern=interleave,gamma_h=1.15,gamma_l=1.15,seed=125 --next_note compare_e017_confirmation_runs --processed_feature_npz /home/Hongjie_Zeng/high_order_BM/runs_twoport1024_teacher_latefusion_lam05/latefusion_teacher_lam05_train_test.npz --optical_feature_source raw_plus_image_rbm_probs --audio_feature_source raw_plus_audio_mlp_probs --processed_feature_pattern interleave --processed_mix 0.35 --seed 125 --epochs 100 --early_stop_patience 10 --eval_every 2 --quick_eval_steps 800 --quick_eval_burn_in 100 --quick_eval_thin 2 --full_eval_on_best --full_eval_steps 3000 --full_eval_burn_in 500 --full_eval_thin 2 --batch_size 50 --eval_batch_size 128 --cd_k 3 --lr 0.0002 --momentum 0.6 --weight_decay 0.0 --gamma_h 1.15 --gamma_l 1.15 --label_inhibit 0.3 --label_update binary --label_init random_onehot --audio_layout time40_fold --audio_scale zscore_sigmoid --num_workers 2`

## E022 completed - 2026-06-06 18:28:38
- Output: `/home/Hongjie_Zeng/high_order_BM/runs_twoport1024_E022_e017_seed125`
- Best selection metric: 0.975000 at epoch 4
- Final epoch: 24
- Final full `test_label_gibbs_acc`: 0.9752
- Next: compare_e017_confirmation_runs

## E022 E017 confirmation summary - 2026-06-06 18:28:39
- Output: `/home/Hongjie_Zeng/high_order_BM/runs_twoport1024_E022_e017_seed125`
- Change: optical=raw_plus_image_rbm_probs,audio=raw_plus_audio_mlp_probs,mix=0.35,pattern=interleave,gamma_h=1.15,gamma_l=1.15,seed=125
- Best selection metric: 0.975
- Best epoch: 4
- Final full test_label_gibbs_acc: 0.9752

## E023 started - 2026-06-06 18:28:43
- Purpose: e017_multiseed_confirmation
- Change: optical=raw_plus_image_rbm_probs,audio=raw_plus_audio_mlp_probs,mix=0.35,pattern=interleave,gamma_h=1.15,gamma_l=1.15,seed=126
- Output: `/home/Hongjie_Zeng/high_order_BM/runs_twoport1024_E023_e017_seed126`
- Config: total=1024, hidden=574, gamma_h=1.15, gamma_l=1.15, lr=0.0002, cd_k=3, distill_weight=0.0
- Command: `train_twoport_1024_optimization_wsd.py --out_dir /home/Hongjie_Zeng/high_order_BM/runs_twoport1024_E023_e017_seed126 --experiment_id E023 --purpose e017_multiseed_confirmation --change_note optical=raw_plus_image_rbm_probs,audio=raw_plus_audio_mlp_probs,mix=0.35,pattern=interleave,gamma_h=1.15,gamma_l=1.15,seed=126 --next_note compare_e017_confirmation_runs --processed_feature_npz /home/Hongjie_Zeng/high_order_BM/runs_twoport1024_teacher_latefusion_lam05/latefusion_teacher_lam05_train_test.npz --optical_feature_source raw_plus_image_rbm_probs --audio_feature_source raw_plus_audio_mlp_probs --processed_feature_pattern interleave --processed_mix 0.35 --seed 126 --epochs 100 --early_stop_patience 10 --eval_every 2 --quick_eval_steps 800 --quick_eval_burn_in 100 --quick_eval_thin 2 --full_eval_on_best --full_eval_steps 3000 --full_eval_burn_in 500 --full_eval_thin 2 --batch_size 50 --eval_batch_size 128 --cd_k 3 --lr 0.0002 --momentum 0.6 --weight_decay 0.0 --gamma_h 1.15 --gamma_l 1.15 --label_inhibit 0.3 --label_update binary --label_init random_onehot --audio_layout time40_fold --audio_scale zscore_sigmoid --num_workers 2`

## E023 completed - 2026-06-06 18:40:11
- Output: `/home/Hongjie_Zeng/high_order_BM/runs_twoport1024_E023_e017_seed126`
- Best selection metric: 0.972100 at epoch 4
- Final epoch: 24
- Final full `test_label_gibbs_acc`: 0.972
- Next: compare_e017_confirmation_runs

## E023 E017 confirmation summary - 2026-06-06 18:40:12
- Output: `/home/Hongjie_Zeng/high_order_BM/runs_twoport1024_E023_e017_seed126`
- Change: optical=raw_plus_image_rbm_probs,audio=raw_plus_audio_mlp_probs,mix=0.35,pattern=interleave,gamma_h=1.15,gamma_l=1.15,seed=126
- Best selection metric: 0.9721
- Best epoch: 4
- Final full test_label_gibbs_acc: 0.972

## E024 started - 2026-06-06 18:40:16
- Purpose: e017_multiseed_confirmation
- Change: optical=raw_plus_image_rbm_probs,audio=raw_plus_audio_mlp_probs,mix=0.35,pattern=interleave,gamma_h=1.15,gamma_l=1.15,seed=127
- Output: `/home/Hongjie_Zeng/high_order_BM/runs_twoport1024_E024_e017_seed127`
- Config: total=1024, hidden=574, gamma_h=1.15, gamma_l=1.15, lr=0.0002, cd_k=3, distill_weight=0.0
- Command: `train_twoport_1024_optimization_wsd.py --out_dir /home/Hongjie_Zeng/high_order_BM/runs_twoport1024_E024_e017_seed127 --experiment_id E024 --purpose e017_multiseed_confirmation --change_note optical=raw_plus_image_rbm_probs,audio=raw_plus_audio_mlp_probs,mix=0.35,pattern=interleave,gamma_h=1.15,gamma_l=1.15,seed=127 --next_note compare_e017_confirmation_runs --processed_feature_npz /home/Hongjie_Zeng/high_order_BM/runs_twoport1024_teacher_latefusion_lam05/latefusion_teacher_lam05_train_test.npz --optical_feature_source raw_plus_image_rbm_probs --audio_feature_source raw_plus_audio_mlp_probs --processed_feature_pattern interleave --processed_mix 0.35 --seed 127 --epochs 100 --early_stop_patience 10 --eval_every 2 --quick_eval_steps 800 --quick_eval_burn_in 100 --quick_eval_thin 2 --full_eval_on_best --full_eval_steps 3000 --full_eval_burn_in 500 --full_eval_thin 2 --batch_size 50 --eval_batch_size 128 --cd_k 3 --lr 0.0002 --momentum 0.6 --weight_decay 0.0 --gamma_h 1.15 --gamma_l 1.15 --label_inhibit 0.3 --label_update binary --label_init random_onehot --audio_layout time40_fold --audio_scale zscore_sigmoid --num_workers 2`

## E024 completed - 2026-06-06 18:51:49
- Output: `/home/Hongjie_Zeng/high_order_BM/runs_twoport1024_E024_e017_seed127`
- Best selection metric: 0.972900 at epoch 4
- Final epoch: 24
- Final full `test_label_gibbs_acc`: 0.9737
- Next: compare_e017_confirmation_runs

## E024 E017 confirmation summary - 2026-06-06 18:51:49
- Output: `/home/Hongjie_Zeng/high_order_BM/runs_twoport1024_E024_e017_seed127`
- Change: optical=raw_plus_image_rbm_probs,audio=raw_plus_audio_mlp_probs,mix=0.35,pattern=interleave,gamma_h=1.15,gamma_l=1.15,seed=127
- Best selection metric: 0.9729
- Best epoch: 4
- Final full test_label_gibbs_acc: 0.9737

## E025 started - 2026-06-06 18:51:54
- Purpose: e017_multiseed_confirmation
- Change: optical=raw_plus_image_rbm_probs,audio=raw_plus_audio_mlp_probs,mix=0.35,pattern=interleave,gamma_h=1.15,gamma_l=1.15,seed=128
- Output: `/home/Hongjie_Zeng/high_order_BM/runs_twoport1024_E025_e017_seed128`
- Config: total=1024, hidden=574, gamma_h=1.15, gamma_l=1.15, lr=0.0002, cd_k=3, distill_weight=0.0
- Command: `train_twoport_1024_optimization_wsd.py --out_dir /home/Hongjie_Zeng/high_order_BM/runs_twoport1024_E025_e017_seed128 --experiment_id E025 --purpose e017_multiseed_confirmation --change_note optical=raw_plus_image_rbm_probs,audio=raw_plus_audio_mlp_probs,mix=0.35,pattern=interleave,gamma_h=1.15,gamma_l=1.15,seed=128 --next_note compare_e017_confirmation_runs --processed_feature_npz /home/Hongjie_Zeng/high_order_BM/runs_twoport1024_teacher_latefusion_lam05/latefusion_teacher_lam05_train_test.npz --optical_feature_source raw_plus_image_rbm_probs --audio_feature_source raw_plus_audio_mlp_probs --processed_feature_pattern interleave --processed_mix 0.35 --seed 128 --epochs 100 --early_stop_patience 10 --eval_every 2 --quick_eval_steps 800 --quick_eval_burn_in 100 --quick_eval_thin 2 --full_eval_on_best --full_eval_steps 3000 --full_eval_burn_in 500 --full_eval_thin 2 --batch_size 50 --eval_batch_size 128 --cd_k 3 --lr 0.0002 --momentum 0.6 --weight_decay 0.0 --gamma_h 1.15 --gamma_l 1.15 --label_inhibit 0.3 --label_update binary --label_init random_onehot --audio_layout time40_fold --audio_scale zscore_sigmoid --num_workers 2`

## E025 completed - 2026-06-06 19:03:17
- Output: `/home/Hongjie_Zeng/high_order_BM/runs_twoport1024_E025_e017_seed128`
- Best selection metric: 0.974600 at epoch 4
- Final epoch: 24
- Final full `test_label_gibbs_acc`: 0.9746
- Next: compare_e017_confirmation_runs

## E025 E017 confirmation summary - 2026-06-06 19:03:18
- Output: `/home/Hongjie_Zeng/high_order_BM/runs_twoport1024_E025_e017_seed128`
- Change: optical=raw_plus_image_rbm_probs,audio=raw_plus_audio_mlp_probs,mix=0.35,pattern=interleave,gamma_h=1.15,gamma_l=1.15,seed=128
- Best selection metric: 0.9746
- Best epoch: 4
- Final full test_label_gibbs_acc: 0.9746

## E026 started - 2026-06-06 19:03:23
- Purpose: e017_mix_sweep
- Change: optical=raw_plus_image_rbm_probs,audio=raw_plus_audio_mlp_probs,mix=0.2,pattern=interleave,gamma_h=1.15,gamma_l=1.15,seed=123
- Output: `/home/Hongjie_Zeng/high_order_BM/runs_twoport1024_E026_e017_mix020`
- Config: total=1024, hidden=574, gamma_h=1.15, gamma_l=1.15, lr=0.0002, cd_k=3, distill_weight=0.0
- Command: `train_twoport_1024_optimization_wsd.py --out_dir /home/Hongjie_Zeng/high_order_BM/runs_twoport1024_E026_e017_mix020 --experiment_id E026 --purpose e017_mix_sweep --change_note optical=raw_plus_image_rbm_probs,audio=raw_plus_audio_mlp_probs,mix=0.2,pattern=interleave,gamma_h=1.15,gamma_l=1.15,seed=123 --next_note compare_e017_confirmation_runs --processed_feature_npz /home/Hongjie_Zeng/high_order_BM/runs_twoport1024_teacher_latefusion_lam05/latefusion_teacher_lam05_train_test.npz --optical_feature_source raw_plus_image_rbm_probs --audio_feature_source raw_plus_audio_mlp_probs --processed_feature_pattern interleave --processed_mix 0.2 --seed 123 --epochs 100 --early_stop_patience 10 --eval_every 2 --quick_eval_steps 800 --quick_eval_burn_in 100 --quick_eval_thin 2 --full_eval_on_best --full_eval_steps 3000 --full_eval_burn_in 500 --full_eval_thin 2 --batch_size 50 --eval_batch_size 128 --cd_k 3 --lr 0.0002 --momentum 0.6 --weight_decay 0.0 --gamma_h 1.15 --gamma_l 1.15 --label_inhibit 0.3 --label_update binary --label_init random_onehot --audio_layout time40_fold --audio_scale zscore_sigmoid --num_workers 2`

## E026 completed - 2026-06-06 19:27:46
- Output: `/home/Hongjie_Zeng/high_order_BM/runs_twoport1024_E026_e017_mix020`
- Best selection metric: 0.972700 at epoch 18
- Final epoch: 38
- Final full `test_label_gibbs_acc`: 0.9724
- Next: compare_e017_confirmation_runs

## E026 E017 confirmation summary - 2026-06-06 19:27:47
- Output: `/home/Hongjie_Zeng/high_order_BM/runs_twoport1024_E026_e017_mix020`
- Change: optical=raw_plus_image_rbm_probs,audio=raw_plus_audio_mlp_probs,mix=0.2,pattern=interleave,gamma_h=1.15,gamma_l=1.15,seed=123
- Best selection metric: 0.9727
- Best epoch: 18
- Final full test_label_gibbs_acc: 0.9724

## E027 started - 2026-06-06 19:27:51
- Purpose: e017_mix_sweep
- Change: optical=raw_plus_image_rbm_probs,audio=raw_plus_audio_mlp_probs,mix=0.25,pattern=interleave,gamma_h=1.15,gamma_l=1.15,seed=123
- Output: `/home/Hongjie_Zeng/high_order_BM/runs_twoport1024_E027_e017_mix025`
- Config: total=1024, hidden=574, gamma_h=1.15, gamma_l=1.15, lr=0.0002, cd_k=3, distill_weight=0.0
- Command: `train_twoport_1024_optimization_wsd.py --out_dir /home/Hongjie_Zeng/high_order_BM/runs_twoport1024_E027_e017_mix025 --experiment_id E027 --purpose e017_mix_sweep --change_note optical=raw_plus_image_rbm_probs,audio=raw_plus_audio_mlp_probs,mix=0.25,pattern=interleave,gamma_h=1.15,gamma_l=1.15,seed=123 --next_note compare_e017_confirmation_runs --processed_feature_npz /home/Hongjie_Zeng/high_order_BM/runs_twoport1024_teacher_latefusion_lam05/latefusion_teacher_lam05_train_test.npz --optical_feature_source raw_plus_image_rbm_probs --audio_feature_source raw_plus_audio_mlp_probs --processed_feature_pattern interleave --processed_mix 0.25 --seed 123 --epochs 100 --early_stop_patience 10 --eval_every 2 --quick_eval_steps 800 --quick_eval_burn_in 100 --quick_eval_thin 2 --full_eval_on_best --full_eval_steps 3000 --full_eval_burn_in 500 --full_eval_thin 2 --batch_size 50 --eval_batch_size 128 --cd_k 3 --lr 0.0002 --momentum 0.6 --weight_decay 0.0 --gamma_h 1.15 --gamma_l 1.15 --label_inhibit 0.3 --label_update binary --label_init random_onehot --audio_layout time40_fold --audio_scale zscore_sigmoid --num_workers 2`

## E027 completed - 2026-06-06 19:47:18
- Output: `/home/Hongjie_Zeng/high_order_BM/runs_twoport1024_E027_e017_mix025`
- Best selection metric: 0.973200 at epoch 12
- Final epoch: 32
- Final full `test_label_gibbs_acc`: 0.9727
- Next: compare_e017_confirmation_runs

## E027 E017 confirmation summary - 2026-06-06 19:47:19
- Output: `/home/Hongjie_Zeng/high_order_BM/runs_twoport1024_E027_e017_mix025`
- Change: optical=raw_plus_image_rbm_probs,audio=raw_plus_audio_mlp_probs,mix=0.25,pattern=interleave,gamma_h=1.15,gamma_l=1.15,seed=123
- Best selection metric: 0.9732
- Best epoch: 12
- Final full test_label_gibbs_acc: 0.9727

## E028 started - 2026-06-06 19:47:23
- Purpose: e017_mix_sweep
- Change: optical=raw_plus_image_rbm_probs,audio=raw_plus_audio_mlp_probs,mix=0.3,pattern=interleave,gamma_h=1.15,gamma_l=1.15,seed=123
- Output: `/home/Hongjie_Zeng/high_order_BM/runs_twoport1024_E028_e017_mix030`
- Config: total=1024, hidden=574, gamma_h=1.15, gamma_l=1.15, lr=0.0002, cd_k=3, distill_weight=0.0
- Command: `train_twoport_1024_optimization_wsd.py --out_dir /home/Hongjie_Zeng/high_order_BM/runs_twoport1024_E028_e017_mix030 --experiment_id E028 --purpose e017_mix_sweep --change_note optical=raw_plus_image_rbm_probs,audio=raw_plus_audio_mlp_probs,mix=0.3,pattern=interleave,gamma_h=1.15,gamma_l=1.15,seed=123 --next_note compare_e017_confirmation_runs --processed_feature_npz /home/Hongjie_Zeng/high_order_BM/runs_twoport1024_teacher_latefusion_lam05/latefusion_teacher_lam05_train_test.npz --optical_feature_source raw_plus_image_rbm_probs --audio_feature_source raw_plus_audio_mlp_probs --processed_feature_pattern interleave --processed_mix 0.3 --seed 123 --epochs 100 --early_stop_patience 10 --eval_every 2 --quick_eval_steps 800 --quick_eval_burn_in 100 --quick_eval_thin 2 --full_eval_on_best --full_eval_steps 3000 --full_eval_burn_in 500 --full_eval_thin 2 --batch_size 50 --eval_batch_size 128 --cd_k 3 --lr 0.0002 --momentum 0.6 --weight_decay 0.0 --gamma_h 1.15 --gamma_l 1.15 --label_inhibit 0.3 --label_update binary --label_init random_onehot --audio_layout time40_fold --audio_scale zscore_sigmoid --num_workers 2`

## E028 completed - 2026-06-06 20:02:45
- Output: `/home/Hongjie_Zeng/high_order_BM/runs_twoport1024_E028_e017_mix030`
- Best selection metric: 0.973200 at epoch 8
- Final epoch: 28
- Final full `test_label_gibbs_acc`: 0.9731
- Next: compare_e017_confirmation_runs

## E028 E017 confirmation summary - 2026-06-06 20:02:46
- Output: `/home/Hongjie_Zeng/high_order_BM/runs_twoport1024_E028_e017_mix030`
- Change: optical=raw_plus_image_rbm_probs,audio=raw_plus_audio_mlp_probs,mix=0.3,pattern=interleave,gamma_h=1.15,gamma_l=1.15,seed=123
- Best selection metric: 0.9732
- Best epoch: 8
- Final full test_label_gibbs_acc: 0.9731

## E029 started - 2026-06-06 20:02:50
- Purpose: e017_mix_sweep
- Change: optical=raw_plus_image_rbm_probs,audio=raw_plus_audio_mlp_probs,mix=0.4,pattern=interleave,gamma_h=1.15,gamma_l=1.15,seed=123
- Output: `/home/Hongjie_Zeng/high_order_BM/runs_twoport1024_E029_e017_mix040`
- Config: total=1024, hidden=574, gamma_h=1.15, gamma_l=1.15, lr=0.0002, cd_k=3, distill_weight=0.0
- Command: `train_twoport_1024_optimization_wsd.py --out_dir /home/Hongjie_Zeng/high_order_BM/runs_twoport1024_E029_e017_mix040 --experiment_id E029 --purpose e017_mix_sweep --change_note optical=raw_plus_image_rbm_probs,audio=raw_plus_audio_mlp_probs,mix=0.4,pattern=interleave,gamma_h=1.15,gamma_l=1.15,seed=123 --next_note compare_e017_confirmation_runs --processed_feature_npz /home/Hongjie_Zeng/high_order_BM/runs_twoport1024_teacher_latefusion_lam05/latefusion_teacher_lam05_train_test.npz --optical_feature_source raw_plus_image_rbm_probs --audio_feature_source raw_plus_audio_mlp_probs --processed_feature_pattern interleave --processed_mix 0.4 --seed 123 --epochs 100 --early_stop_patience 10 --eval_every 2 --quick_eval_steps 800 --quick_eval_burn_in 100 --quick_eval_thin 2 --full_eval_on_best --full_eval_steps 3000 --full_eval_burn_in 500 --full_eval_thin 2 --batch_size 50 --eval_batch_size 128 --cd_k 3 --lr 0.0002 --momentum 0.6 --weight_decay 0.0 --gamma_h 1.15 --gamma_l 1.15 --label_inhibit 0.3 --label_update binary --label_init random_onehot --audio_layout time40_fold --audio_scale zscore_sigmoid --num_workers 2`

## E029 completed - 2026-06-06 20:14:38
- Output: `/home/Hongjie_Zeng/high_order_BM/runs_twoport1024_E029_e017_mix040`
- Best selection metric: 0.971200 at epoch 4
- Final epoch: 24
- Final full `test_label_gibbs_acc`: 0.9713
- Next: compare_e017_confirmation_runs

## E029 E017 confirmation summary - 2026-06-06 20:14:39
- Output: `/home/Hongjie_Zeng/high_order_BM/runs_twoport1024_E029_e017_mix040`
- Change: optical=raw_plus_image_rbm_probs,audio=raw_plus_audio_mlp_probs,mix=0.4,pattern=interleave,gamma_h=1.15,gamma_l=1.15,seed=123
- Best selection metric: 0.9712
- Best epoch: 4
- Final full test_label_gibbs_acc: 0.9713

## E030 started - 2026-06-06 20:14:43
- Purpose: e017_mix_sweep
- Change: optical=raw_plus_image_rbm_probs,audio=raw_plus_audio_mlp_probs,mix=0.45,pattern=interleave,gamma_h=1.15,gamma_l=1.15,seed=123
- Output: `/home/Hongjie_Zeng/high_order_BM/runs_twoport1024_E030_e017_mix045`
- Config: total=1024, hidden=574, gamma_h=1.15, gamma_l=1.15, lr=0.0002, cd_k=3, distill_weight=0.0
- Command: `train_twoport_1024_optimization_wsd.py --out_dir /home/Hongjie_Zeng/high_order_BM/runs_twoport1024_E030_e017_mix045 --experiment_id E030 --purpose e017_mix_sweep --change_note optical=raw_plus_image_rbm_probs,audio=raw_plus_audio_mlp_probs,mix=0.45,pattern=interleave,gamma_h=1.15,gamma_l=1.15,seed=123 --next_note compare_e017_confirmation_runs --processed_feature_npz /home/Hongjie_Zeng/high_order_BM/runs_twoport1024_teacher_latefusion_lam05/latefusion_teacher_lam05_train_test.npz --optical_feature_source raw_plus_image_rbm_probs --audio_feature_source raw_plus_audio_mlp_probs --processed_feature_pattern interleave --processed_mix 0.45 --seed 123 --epochs 100 --early_stop_patience 10 --eval_every 2 --quick_eval_steps 800 --quick_eval_burn_in 100 --quick_eval_thin 2 --full_eval_on_best --full_eval_steps 3000 --full_eval_burn_in 500 --full_eval_thin 2 --batch_size 50 --eval_batch_size 128 --cd_k 3 --lr 0.0002 --momentum 0.6 --weight_decay 0.0 --gamma_h 1.15 --gamma_l 1.15 --label_inhibit 0.3 --label_update binary --label_init random_onehot --audio_layout time40_fold --audio_scale zscore_sigmoid --num_workers 2`

## E030 completed - 2026-06-06 20:24:16
- Output: `/home/Hongjie_Zeng/high_order_BM/runs_twoport1024_E030_e017_mix045`
- Best selection metric: 0.968300 at epoch 2
- Final epoch: 22
- Final full `test_label_gibbs_acc`: 0.9693
- Next: compare_e017_confirmation_runs

## E030 E017 confirmation summary - 2026-06-06 20:24:16
- Output: `/home/Hongjie_Zeng/high_order_BM/runs_twoport1024_E030_e017_mix045`
- Change: optical=raw_plus_image_rbm_probs,audio=raw_plus_audio_mlp_probs,mix=0.45,pattern=interleave,gamma_h=1.15,gamma_l=1.15,seed=123
- Best selection metric: 0.9683
- Best epoch: 2
- Final full test_label_gibbs_acc: 0.9693

## E031 started - 2026-06-06 20:24:21
- Purpose: e017_mix_sweep
- Change: optical=raw_plus_image_rbm_probs,audio=raw_plus_audio_mlp_probs,mix=0.5,pattern=interleave,gamma_h=1.15,gamma_l=1.15,seed=123
- Output: `/home/Hongjie_Zeng/high_order_BM/runs_twoport1024_E031_e017_mix050`
- Config: total=1024, hidden=574, gamma_h=1.15, gamma_l=1.15, lr=0.0002, cd_k=3, distill_weight=0.0
- Command: `train_twoport_1024_optimization_wsd.py --out_dir /home/Hongjie_Zeng/high_order_BM/runs_twoport1024_E031_e017_mix050 --experiment_id E031 --purpose e017_mix_sweep --change_note optical=raw_plus_image_rbm_probs,audio=raw_plus_audio_mlp_probs,mix=0.5,pattern=interleave,gamma_h=1.15,gamma_l=1.15,seed=123 --next_note compare_e017_confirmation_runs --processed_feature_npz /home/Hongjie_Zeng/high_order_BM/runs_twoport1024_teacher_latefusion_lam05/latefusion_teacher_lam05_train_test.npz --optical_feature_source raw_plus_image_rbm_probs --audio_feature_source raw_plus_audio_mlp_probs --processed_feature_pattern interleave --processed_mix 0.5 --seed 123 --epochs 100 --early_stop_patience 10 --eval_every 2 --quick_eval_steps 800 --quick_eval_burn_in 100 --quick_eval_thin 2 --full_eval_on_best --full_eval_steps 3000 --full_eval_burn_in 500 --full_eval_thin 2 --batch_size 50 --eval_batch_size 128 --cd_k 3 --lr 0.0002 --momentum 0.6 --weight_decay 0.0 --gamma_h 1.15 --gamma_l 1.15 --label_inhibit 0.3 --label_update binary --label_init random_onehot --audio_layout time40_fold --audio_scale zscore_sigmoid --num_workers 2`

## E031 completed - 2026-06-06 20:33:49
- Output: `/home/Hongjie_Zeng/high_order_BM/runs_twoport1024_E031_e017_mix050`
- Best selection metric: 0.963900 at epoch 2
- Final epoch: 22
- Final full `test_label_gibbs_acc`: 0.9645
- Next: compare_e017_confirmation_runs

## E031 E017 confirmation summary - 2026-06-06 20:33:50
- Output: `/home/Hongjie_Zeng/high_order_BM/runs_twoport1024_E031_e017_mix050`
- Change: optical=raw_plus_image_rbm_probs,audio=raw_plus_audio_mlp_probs,mix=0.5,pattern=interleave,gamma_h=1.15,gamma_l=1.15,seed=123
- Best selection metric: 0.9639
- Best epoch: 2
- Final full test_label_gibbs_acc: 0.9645

## E032 started - 2026-06-06 20:33:54
- Purpose: e017_gamma_refine
- Change: optical=raw_plus_image_rbm_probs,audio=raw_plus_audio_mlp_probs,mix=0.35,pattern=interleave,gamma_h=1.05,gamma_l=1.05,seed=123
- Output: `/home/Hongjie_Zeng/high_order_BM/runs_twoport1024_E032_e017_gamma105`
- Config: total=1024, hidden=574, gamma_h=1.05, gamma_l=1.05, lr=0.0002, cd_k=3, distill_weight=0.0
- Command: `train_twoport_1024_optimization_wsd.py --out_dir /home/Hongjie_Zeng/high_order_BM/runs_twoport1024_E032_e017_gamma105 --experiment_id E032 --purpose e017_gamma_refine --change_note optical=raw_plus_image_rbm_probs,audio=raw_plus_audio_mlp_probs,mix=0.35,pattern=interleave,gamma_h=1.05,gamma_l=1.05,seed=123 --next_note compare_e017_confirmation_runs --processed_feature_npz /home/Hongjie_Zeng/high_order_BM/runs_twoport1024_teacher_latefusion_lam05/latefusion_teacher_lam05_train_test.npz --optical_feature_source raw_plus_image_rbm_probs --audio_feature_source raw_plus_audio_mlp_probs --processed_feature_pattern interleave --processed_mix 0.35 --seed 123 --epochs 100 --early_stop_patience 10 --eval_every 2 --quick_eval_steps 800 --quick_eval_burn_in 100 --quick_eval_thin 2 --full_eval_on_best --full_eval_steps 3000 --full_eval_burn_in 500 --full_eval_thin 2 --batch_size 50 --eval_batch_size 128 --cd_k 3 --lr 0.0002 --momentum 0.6 --weight_decay 0.0 --gamma_h 1.05 --gamma_l 1.05 --label_inhibit 0.3 --label_update binary --label_init random_onehot --audio_layout time40_fold --audio_scale zscore_sigmoid --num_workers 2`

## E032 completed - 2026-06-06 20:47:17
- Output: `/home/Hongjie_Zeng/high_order_BM/runs_twoport1024_E032_e017_gamma105`
- Best selection metric: 0.974600 at epoch 6
- Final epoch: 26
- Final full `test_label_gibbs_acc`: 0.9739
- Next: compare_e017_confirmation_runs

## E032 E017 confirmation summary - 2026-06-06 20:47:18
- Output: `/home/Hongjie_Zeng/high_order_BM/runs_twoport1024_E032_e017_gamma105`
- Change: optical=raw_plus_image_rbm_probs,audio=raw_plus_audio_mlp_probs,mix=0.35,pattern=interleave,gamma_h=1.05,gamma_l=1.05,seed=123
- Best selection metric: 0.9746
- Best epoch: 6
- Final full test_label_gibbs_acc: 0.9739

## E033 started - 2026-06-06 20:47:23
- Purpose: e017_gamma_refine
- Change: optical=raw_plus_image_rbm_probs,audio=raw_plus_audio_mlp_probs,mix=0.35,pattern=interleave,gamma_h=1.1,gamma_l=1.1,seed=123
- Output: `/home/Hongjie_Zeng/high_order_BM/runs_twoport1024_E033_e017_gamma110`
- Config: total=1024, hidden=574, gamma_h=1.1, gamma_l=1.1, lr=0.0002, cd_k=3, distill_weight=0.0
- Command: `train_twoport_1024_optimization_wsd.py --out_dir /home/Hongjie_Zeng/high_order_BM/runs_twoport1024_E033_e017_gamma110 --experiment_id E033 --purpose e017_gamma_refine --change_note optical=raw_plus_image_rbm_probs,audio=raw_plus_audio_mlp_probs,mix=0.35,pattern=interleave,gamma_h=1.1,gamma_l=1.1,seed=123 --next_note compare_e017_confirmation_runs --processed_feature_npz /home/Hongjie_Zeng/high_order_BM/runs_twoport1024_teacher_latefusion_lam05/latefusion_teacher_lam05_train_test.npz --optical_feature_source raw_plus_image_rbm_probs --audio_feature_source raw_plus_audio_mlp_probs --processed_feature_pattern interleave --processed_mix 0.35 --seed 123 --epochs 100 --early_stop_patience 10 --eval_every 2 --quick_eval_steps 800 --quick_eval_burn_in 100 --quick_eval_thin 2 --full_eval_on_best --full_eval_steps 3000 --full_eval_burn_in 500 --full_eval_thin 2 --batch_size 50 --eval_batch_size 128 --cd_k 3 --lr 0.0002 --momentum 0.6 --weight_decay 0.0 --gamma_h 1.1 --gamma_l 1.1 --label_inhibit 0.3 --label_update binary --label_init random_onehot --audio_layout time40_fold --audio_scale zscore_sigmoid --num_workers 2`

## E033 completed - 2026-06-06 21:00:49
- Output: `/home/Hongjie_Zeng/high_order_BM/runs_twoport1024_E033_e017_gamma110`
- Best selection metric: 0.973800 at epoch 6
- Final epoch: 26
- Final full `test_label_gibbs_acc`: 0.9735
- Next: compare_e017_confirmation_runs

## E033 E017 confirmation summary - 2026-06-06 21:00:50
- Output: `/home/Hongjie_Zeng/high_order_BM/runs_twoport1024_E033_e017_gamma110`
- Change: optical=raw_plus_image_rbm_probs,audio=raw_plus_audio_mlp_probs,mix=0.35,pattern=interleave,gamma_h=1.1,gamma_l=1.1,seed=123
- Best selection metric: 0.9738
- Best epoch: 6
- Final full test_label_gibbs_acc: 0.9735

## E034 started - 2026-06-06 21:00:55
- Purpose: e017_gamma_refine
- Change: optical=raw_plus_image_rbm_probs,audio=raw_plus_audio_mlp_probs,mix=0.35,pattern=interleave,gamma_h=1.2,gamma_l=1.2,seed=123
- Output: `/home/Hongjie_Zeng/high_order_BM/runs_twoport1024_E034_e017_gamma120`
- Config: total=1024, hidden=574, gamma_h=1.2, gamma_l=1.2, lr=0.0002, cd_k=3, distill_weight=0.0
- Command: `train_twoport_1024_optimization_wsd.py --out_dir /home/Hongjie_Zeng/high_order_BM/runs_twoport1024_E034_e017_gamma120 --experiment_id E034 --purpose e017_gamma_refine --change_note optical=raw_plus_image_rbm_probs,audio=raw_plus_audio_mlp_probs,mix=0.35,pattern=interleave,gamma_h=1.2,gamma_l=1.2,seed=123 --next_note compare_e017_confirmation_runs --processed_feature_npz /home/Hongjie_Zeng/high_order_BM/runs_twoport1024_teacher_latefusion_lam05/latefusion_teacher_lam05_train_test.npz --optical_feature_source raw_plus_image_rbm_probs --audio_feature_source raw_plus_audio_mlp_probs --processed_feature_pattern interleave --processed_mix 0.35 --seed 123 --epochs 100 --early_stop_patience 10 --eval_every 2 --quick_eval_steps 800 --quick_eval_burn_in 100 --quick_eval_thin 2 --full_eval_on_best --full_eval_steps 3000 --full_eval_burn_in 500 --full_eval_thin 2 --batch_size 50 --eval_batch_size 128 --cd_k 3 --lr 0.0002 --momentum 0.6 --weight_decay 0.0 --gamma_h 1.2 --gamma_l 1.2 --label_inhibit 0.3 --label_update binary --label_init random_onehot --audio_layout time40_fold --audio_scale zscore_sigmoid --num_workers 2`

## E034 completed - 2026-06-06 21:14:25
- Output: `/home/Hongjie_Zeng/high_order_BM/runs_twoport1024_E034_e017_gamma120`
- Best selection metric: 0.972500 at epoch 6
- Final epoch: 26
- Final full `test_label_gibbs_acc`: 0.9728
- Next: compare_e017_confirmation_runs

## E034 E017 confirmation summary - 2026-06-06 21:14:26
- Output: `/home/Hongjie_Zeng/high_order_BM/runs_twoport1024_E034_e017_gamma120`
- Change: optical=raw_plus_image_rbm_probs,audio=raw_plus_audio_mlp_probs,mix=0.35,pattern=interleave,gamma_h=1.2,gamma_l=1.2,seed=123
- Best selection metric: 0.9725
- Best epoch: 6
- Final full test_label_gibbs_acc: 0.9728

## E035 started - 2026-06-06 21:14:30
- Purpose: e017_gamma_refine
- Change: optical=raw_plus_image_rbm_probs,audio=raw_plus_audio_mlp_probs,mix=0.35,pattern=interleave,gamma_h=1.25,gamma_l=1.25,seed=123
- Output: `/home/Hongjie_Zeng/high_order_BM/runs_twoport1024_E035_e017_gamma125`
- Config: total=1024, hidden=574, gamma_h=1.25, gamma_l=1.25, lr=0.0002, cd_k=3, distill_weight=0.0
- Command: `train_twoport_1024_optimization_wsd.py --out_dir /home/Hongjie_Zeng/high_order_BM/runs_twoport1024_E035_e017_gamma125 --experiment_id E035 --purpose e017_gamma_refine --change_note optical=raw_plus_image_rbm_probs,audio=raw_plus_audio_mlp_probs,mix=0.35,pattern=interleave,gamma_h=1.25,gamma_l=1.25,seed=123 --next_note compare_e017_confirmation_runs --processed_feature_npz /home/Hongjie_Zeng/high_order_BM/runs_twoport1024_teacher_latefusion_lam05/latefusion_teacher_lam05_train_test.npz --optical_feature_source raw_plus_image_rbm_probs --audio_feature_source raw_plus_audio_mlp_probs --processed_feature_pattern interleave --processed_mix 0.35 --seed 123 --epochs 100 --early_stop_patience 10 --eval_every 2 --quick_eval_steps 800 --quick_eval_burn_in 100 --quick_eval_thin 2 --full_eval_on_best --full_eval_steps 3000 --full_eval_burn_in 500 --full_eval_thin 2 --batch_size 50 --eval_batch_size 128 --cd_k 3 --lr 0.0002 --momentum 0.6 --weight_decay 0.0 --gamma_h 1.25 --gamma_l 1.25 --label_inhibit 0.3 --label_update binary --label_init random_onehot --audio_layout time40_fold --audio_scale zscore_sigmoid --num_workers 2`

## E035 completed - 2026-06-06 21:28:05
- Output: `/home/Hongjie_Zeng/high_order_BM/runs_twoport1024_E035_e017_gamma125`
- Best selection metric: 0.972000 at epoch 6
- Final epoch: 26
- Final full `test_label_gibbs_acc`: 0.9723
- Next: compare_e017_confirmation_runs

## E035 E017 confirmation summary - 2026-06-06 21:28:05
- Output: `/home/Hongjie_Zeng/high_order_BM/runs_twoport1024_E035_e017_gamma125`
- Change: optical=raw_plus_image_rbm_probs,audio=raw_plus_audio_mlp_probs,mix=0.35,pattern=interleave,gamma_h=1.25,gamma_l=1.25,seed=123
- Best selection metric: 0.972
- Best epoch: 6
- Final full test_label_gibbs_acc: 0.9723

## E036 started - 2026-06-06 21:28:09
- Purpose: e017_split_gamma_refine
- Change: optical=raw_plus_image_rbm_probs,audio=raw_plus_audio_mlp_probs,mix=0.35,pattern=interleave,gamma_h=1.1,gamma_l=1.2,seed=123
- Output: `/home/Hongjie_Zeng/high_order_BM/runs_twoport1024_E036_e017_gamma_h110_l120`
- Config: total=1024, hidden=574, gamma_h=1.1, gamma_l=1.2, lr=0.0002, cd_k=3, distill_weight=0.0
- Command: `train_twoport_1024_optimization_wsd.py --out_dir /home/Hongjie_Zeng/high_order_BM/runs_twoport1024_E036_e017_gamma_h110_l120 --experiment_id E036 --purpose e017_split_gamma_refine --change_note optical=raw_plus_image_rbm_probs,audio=raw_plus_audio_mlp_probs,mix=0.35,pattern=interleave,gamma_h=1.1,gamma_l=1.2,seed=123 --next_note compare_e017_confirmation_runs --processed_feature_npz /home/Hongjie_Zeng/high_order_BM/runs_twoport1024_teacher_latefusion_lam05/latefusion_teacher_lam05_train_test.npz --optical_feature_source raw_plus_image_rbm_probs --audio_feature_source raw_plus_audio_mlp_probs --processed_feature_pattern interleave --processed_mix 0.35 --seed 123 --epochs 100 --early_stop_patience 10 --eval_every 2 --quick_eval_steps 800 --quick_eval_burn_in 100 --quick_eval_thin 2 --full_eval_on_best --full_eval_steps 3000 --full_eval_burn_in 500 --full_eval_thin 2 --batch_size 50 --eval_batch_size 128 --cd_k 3 --lr 0.0002 --momentum 0.6 --weight_decay 0.0 --gamma_h 1.1 --gamma_l 1.2 --label_inhibit 0.3 --label_update binary --label_init random_onehot --audio_layout time40_fold --audio_scale zscore_sigmoid --num_workers 2`

## E036 completed - 2026-06-06 21:41:36
- Output: `/home/Hongjie_Zeng/high_order_BM/runs_twoport1024_E036_e017_gamma_h110_l120`
- Best selection metric: 0.972600 at epoch 6
- Final epoch: 26
- Final full `test_label_gibbs_acc`: 0.9729
- Next: compare_e017_confirmation_runs

## E036 E017 confirmation summary - 2026-06-06 21:41:37
- Output: `/home/Hongjie_Zeng/high_order_BM/runs_twoport1024_E036_e017_gamma_h110_l120`
- Change: optical=raw_plus_image_rbm_probs,audio=raw_plus_audio_mlp_probs,mix=0.35,pattern=interleave,gamma_h=1.1,gamma_l=1.2,seed=123
- Best selection metric: 0.9726
- Best epoch: 6
- Final full test_label_gibbs_acc: 0.9729

## E037 started - 2026-06-06 21:41:41
- Purpose: e017_ablation_optical_only
- Change: optical=raw_plus_image_rbm_probs,audio=raw,mix=0.35,pattern=interleave,gamma_h=1.15,gamma_l=1.15,seed=123
- Output: `/home/Hongjie_Zeng/high_order_BM/runs_twoport1024_E037_optical_hybrid_only`
- Config: total=1024, hidden=574, gamma_h=1.15, gamma_l=1.15, lr=0.0002, cd_k=3, distill_weight=0.0
- Command: `train_twoport_1024_optimization_wsd.py --out_dir /home/Hongjie_Zeng/high_order_BM/runs_twoport1024_E037_optical_hybrid_only --experiment_id E037 --purpose e017_ablation_optical_only --change_note optical=raw_plus_image_rbm_probs,audio=raw,mix=0.35,pattern=interleave,gamma_h=1.15,gamma_l=1.15,seed=123 --next_note compare_e017_confirmation_runs --processed_feature_npz /home/Hongjie_Zeng/high_order_BM/runs_twoport1024_teacher_latefusion_lam05/latefusion_teacher_lam05_train_test.npz --optical_feature_source raw_plus_image_rbm_probs --audio_feature_source raw --processed_feature_pattern interleave --processed_mix 0.35 --seed 123 --epochs 100 --early_stop_patience 10 --eval_every 2 --quick_eval_steps 800 --quick_eval_burn_in 100 --quick_eval_thin 2 --full_eval_on_best --full_eval_steps 3000 --full_eval_burn_in 500 --full_eval_thin 2 --batch_size 50 --eval_batch_size 128 --cd_k 3 --lr 0.0002 --momentum 0.6 --weight_decay 0.0 --gamma_h 1.15 --gamma_l 1.15 --label_inhibit 0.3 --label_update binary --label_init random_onehot --audio_layout time40_fold --audio_scale zscore_sigmoid --num_workers 2`

## E037 completed - 2026-06-06 22:46:39
- Output: `/home/Hongjie_Zeng/high_order_BM/runs_twoport1024_E037_optical_hybrid_only`
- Best selection metric: 0.960100 at epoch 76
- Final epoch: 96
- Final full `test_label_gibbs_acc`: 0.9593
- Next: compare_e017_confirmation_runs

## E037 E017 confirmation summary - 2026-06-06 22:46:40
- Output: `/home/Hongjie_Zeng/high_order_BM/runs_twoport1024_E037_optical_hybrid_only`
- Change: optical=raw_plus_image_rbm_probs,audio=raw,mix=0.35,pattern=interleave,gamma_h=1.15,gamma_l=1.15,seed=123
- Best selection metric: 0.9601
- Best epoch: 76
- Final full test_label_gibbs_acc: 0.9593

## E038 started - 2026-06-06 22:46:44
- Purpose: e017_ablation_audio_only
- Change: optical=raw,audio=raw_plus_audio_mlp_probs,mix=0.35,pattern=interleave,gamma_h=1.15,gamma_l=1.15,seed=123
- Output: `/home/Hongjie_Zeng/high_order_BM/runs_twoport1024_E038_audio_hybrid_only`
- Config: total=1024, hidden=574, gamma_h=1.15, gamma_l=1.15, lr=0.0002, cd_k=3, distill_weight=0.0
- Command: `train_twoport_1024_optimization_wsd.py --out_dir /home/Hongjie_Zeng/high_order_BM/runs_twoport1024_E038_audio_hybrid_only --experiment_id E038 --purpose e017_ablation_audio_only --change_note optical=raw,audio=raw_plus_audio_mlp_probs,mix=0.35,pattern=interleave,gamma_h=1.15,gamma_l=1.15,seed=123 --next_note compare_e017_confirmation_runs --processed_feature_npz /home/Hongjie_Zeng/high_order_BM/runs_twoport1024_teacher_latefusion_lam05/latefusion_teacher_lam05_train_test.npz --optical_feature_source raw --audio_feature_source raw_plus_audio_mlp_probs --processed_feature_pattern interleave --processed_mix 0.35 --seed 123 --epochs 100 --early_stop_patience 10 --eval_every 2 --quick_eval_steps 800 --quick_eval_burn_in 100 --quick_eval_thin 2 --full_eval_on_best --full_eval_steps 3000 --full_eval_burn_in 500 --full_eval_thin 2 --batch_size 50 --eval_batch_size 128 --cd_k 3 --lr 0.0002 --momentum 0.6 --weight_decay 0.0 --gamma_h 1.15 --gamma_l 1.15 --label_inhibit 0.3 --label_update binary --label_init random_onehot --audio_layout time40_fold --audio_scale zscore_sigmoid --num_workers 2`

## E038 completed - 2026-06-06 23:00:28
- Output: `/home/Hongjie_Zeng/high_order_BM/runs_twoport1024_E038_audio_hybrid_only`
- Best selection metric: 0.956900 at epoch 6
- Final epoch: 26
- Final full `test_label_gibbs_acc`: 0.9569
- Next: compare_e017_confirmation_runs

## E038 E017 confirmation summary - 2026-06-06 23:00:28
- Output: `/home/Hongjie_Zeng/high_order_BM/runs_twoport1024_E038_audio_hybrid_only`
- Change: optical=raw,audio=raw_plus_audio_mlp_probs,mix=0.35,pattern=interleave,gamma_h=1.15,gamma_l=1.15,seed=123
- Best selection metric: 0.9569
- Best epoch: 6
- Final full test_label_gibbs_acc: 0.9569

## E021-E038 E017 confirmation batch completed - 2026-06-06 23:00:28
- Next: report multi-seed mean/std and decide whether E017 can be the main result.

## E021-E038 E017 confirmation batch started - 2026-06-07 01:26:47
- Strategy: confirm the 97.31% E017 result under unchanged 1024 p-bit, dual-channel physical input and full Gibbs inference.
- Feature NPZ: `/home/Hongjie_Zeng/high_order_BM/runs_twoport1024_teacher_latefusion_lam05/latefusion_teacher_lam05_train_test.npz`
- Base: optical=raw_plus_image_rbm_probs, audio=raw_plus_audio_mlp_probs, mix=0.35

## E021-E038 E017 confirmation batch completed - 2026-06-07 01:26:47
- Next: report multi-seed mean/std and decide whether E017 can be the main result.

## E039 probability quantization eval completed - 2026-06-08 17:20:05
- Output: `/home/Hongjie_Zeng/high_order_BM/runs_twoport1024_E039_probquant_7level`
- Quantization: `7level`
- Levels: `0.0500, 0.1200, 0.2700, 0.5000, 0.7300, 0.8800, 0.9500`
- Continuous mean acc: `0.9740333333333333`
- Quantized mean acc: `0.9727833333333332`
- Mean drop: `0.0012499999999999918`
- Runs: `6/6`

## E040 probability quantization eval completed - 2026-06-08 17:45:15
- Output: `/home/Hongjie_Zeng/high_order_BM/runs_twoport1024_E040_probquant_11level_logit`
- Quantization: `11level_logit`
- Levels: `0.0500, 0.0866, 0.1460, 0.2355, 0.3569, 0.5000, 0.6431, 0.7645, 0.8540, 0.9134, 0.9500`
- Continuous mean acc: `0.9740333333333333`
- Quantized mean acc: `0.9729166666666668`
- Mean drop: `0.0011166666666666547`
- Runs: `6/6`

## E041 port-level device quantization eval completed - 2026-06-08 23:04:25
- Output: `/home/Hongjie_Zeng/high_order_BM/runs_twoport1024_E041_portquant_gamma_sweep_5to95`
- Device c: `1.4722194895832201`
- Single-port levels: `0.0500, 0.1200, 0.2700, 0.5000, 0.7300, 0.8800, 0.9500`
- Gammas: `0, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1, 1.15`
- Recommended setting: `none`

## E042 port-level device quantization eval completed - 2026-06-09 00:16:31
- Output: `/home/Hongjie_Zeng/high_order_BM/runs_twoport1024_E042_portquant_gamma_sweep_12to88`
- Device c: `0.9962150823451031`
- Single-port levels: `0.1200, 0.1800, 0.2700, 0.4000, 0.6000, 0.7300, 0.8800`
- Gammas: `0.1, 0.2, 0.3, 0.5, 0.75`
- Recommended setting: `none`

