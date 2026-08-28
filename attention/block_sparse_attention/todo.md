# 任务：刷新 MXFP4/MXFP8 设计对照 + S=4096 keep=1 仿真可视

## 目标
1. 按 9.1 HEAD（QK 一拍 128×128、G=16、V1_C1 在 QK 前）刷新设计对照画布。
2. `msprof op simulator` 采 S=4096 keep=1：FIA MXFP8（CANN 包）、BSA MXFP4、BSA MXFP8；不因耗时长主动 kill。

## 待办事项
- [x] 刷新 BSA-MXFP4-vs-MXFP8-design 画布
- [x] 仿真 FIA MXFP8（CANN，sparse=0）EXIT 0 · 31.27 µs
- [x] 仿真 BSA MXFP4（cann_bsa_9.1）EXIT 0 · 22.15 µs
- [x] 仿真 BSA MXFP8（cann_bsa_9.1）EXIT 0 · 33.05 µs
- [x] extract_pipe_lanes + 画三方泳道图

## 进度
5/5

## 产物
- 设计：`BSA-MXFP4-vs-MXFP8-design.canvas.tsx`
- 泳道：`S4096-keep1-fia-bsa-sim.canvas.tsx`
- 数据：`tests/mxfp8_pytest/bsa_msprof_sim_s4096_keep1/`

## 不做
- 不 push；不主动 kill 仿真
