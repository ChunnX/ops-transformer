# 任务：新增 BSA MXFP8 qs64 处理

## 目标
qs64/kvs128 进官方比对（≥99.5%）；qs128 过线 case 不回退。QK 将 qs pad 到 128，softmax/pscale 走 qs128 VF；align64 仍为 64 供 PV/rescale。

## 待办事项
- [x] tiling：MXFP8 允许 blockShapeX=64 或 128
- [x] CopyPUBToPL1：qs64+kvs128 搬满 64 个 TLA 行（K-outer）
- [x] QK：L1 Q/Q-scale 清零，mmad/fixpipe N 用 align128
- [x] softmax/pscale：pad 后走 qs128 VF；align64 保持 64
- [x] 重编安装 cann_bsa_0817（root，先不 chown）
- [x] 上板 qs64/kvs128；抽测 qs128/kvs128 不回退
- [x] 更新 0817 设计文档 §15

## 进度
7/7

## 板上结果（NPU 1，dump-off）
- qs64/kvs128：100.00% PASSED
- qs128/kvs128：99.74% PASSED（未回退）
- 官方 7 case 全过
