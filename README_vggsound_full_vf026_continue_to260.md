# VGGSound Full VF026 Continue To Epoch260

## 目的

VF026 是 8192 visible video LSTM feature + h6 standard BM：

```text
input_dim = 8192
label_dim = 309 * 5 = 1545
hidden_dim = 49152
total_pbits = 58889
```

JobID 302 因为整个 job 的 24h time limit 停在 epoch 149。JobID 309 对当前 best checkpoint 做了 eval-only full eval：

```text
best checkpoint epoch = 145
quick selection acc = 42.12%
full eval acc = 42.10%
current best video-only BM VF024 = 42.74%
```

VF026 尚未超过 VF024，但停止时仍在上升。这个包从 `last.pt` 继续训练到 epoch 260，并做最终 full eval。

## 提交命令

```bash
cd /home/Hongjie_Zeng/high_order_BM
unzip -o vggsound_full_vf026_continue_to260_code_20260621.zip
chmod +x sbatch_vggsound_full_vf026_continue_to260.sh push_vggsound_full_vf026_continue_to260_results.sh
sbatch sbatch_vggsound_full_vf026_continue_to260.sh
```

## 查看进度

```bash
squeue
tail -f /home/Hongjie_Zeng/high_order_BM/logs/vggsound_vf026_continue_to260_JOBID.out
tail -f /home/Hongjie_Zeng/high_order_BM/runs_vggsound_full_VF026_continue_to260_stdout.log
```

把 `JOBID` 换成 `sbatch` 返回的编号。

## 上传结果

```bash
cd /home/Hongjie_Zeng/high_order_BM
./push_vggsound_full_vf026_continue_to260_results.sh
```

上传内容包括 `summary.json`、`full_eval_best_3000.json`、`history.json`、precheck 和日志。不会上传 `best.pt` 或 `last.pt`。

## 注意

续跑前脚本会把 309 的 eval-only 结果备份为：

```text
summary_eval_only_epoch149.json
full_eval_best_3000_eval_only_epoch149.json
```

这样训练结束后新的 `summary.json` 可以覆盖为 epoch260 的正式结果，同时保留 epoch149 的 eval-only 记录。
