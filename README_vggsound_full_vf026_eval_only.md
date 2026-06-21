# VGGSound Full VF026 Eval-Only补评估

## 目的

JobID 302 被 Slurm time limit 截断，VF026 只训练到 epoch 149，没有生成正式 `summary.json`。这个包只在计算节点上对当前 `best.pt` 做一次 full Gibbs eval，不在登录节点运行，也不继续训练。

## 当前已知状态

```text
VF026 branch:
input_dim = 8192
hidden_factor = 6
hidden_dim = 49152
total_pbits = 58889
completed epoch = 149
best quick epoch = 145
best quick acc = 42.12%
```

注意：`42.12%` 是 quick eval，不是正式 full accuracy。

## 提交命令

```bash
cd /home/Hongjie_Zeng/high_order_BM
unzip -o vggsound_full_vf026_eval_only_code_20260621.zip
chmod +x sbatch_vggsound_full_vf026_eval_only.sh push_vggsound_full_vf026_eval_only_results.sh
sbatch sbatch_vggsound_full_vf026_eval_only.sh
```

## 查看任务

```bash
squeue
tail -f /home/Hongjie_Zeng/high_order_BM/logs/vggsound_vf026_eval_only_JOBID.out
```

把 `JOBID` 替换成 `sbatch` 返回的编号。

## 上传结果

```bash
cd /home/Hongjie_Zeng/high_order_BM
./push_vggsound_full_vf026_eval_only_results.sh
```

上传内容包括 `summary.json`、`full_eval_best_3000.json`、`history.json` 和日志。不会上传 `best.pt` 或 `last.pt`。

## 后续判断

如果 VF026 full eval 超过或接近当前最佳 VF024 `42.74%`，再考虑从 `last.pt` 续跑到 epoch 260。按 VF026 实际速度估算，从 epoch 149 续到 260 大约还需要 7-8 小时训练，加上 full eval 时间。
