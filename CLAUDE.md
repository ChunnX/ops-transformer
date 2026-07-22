# CLAUDE.md — ops-transformer

此文件为 CANN 算子项目 `ops-transformer` 子目录的说明。父仓库为 `lightning_indexer_kl_loss`，整体背景详见根目录 `CLAUDE.md`。

## 项目结构

本目录是昇腾 CANN 算子的标准四层架构源码树，包含多个模块（attention, moe, ffn, mc2, mamba 等）。

## 关键模块

当前开发的核心算子在 `attention/lightning_indexer_kl_loss/` 下：

```
attention/lightning_indexer_kl_loss/
├── CMakeLists.txt
├── examples/test_aclnn_lightning_indexer_kl_loss.cpp  # aclnn API 集成测试
├── op_host/
│   ├── CMakeLists.txt
│   ├── lightning_indexer_kl_loss_def.cpp               # 算子定义
│   ├── lightning_indexer_kl_loss_infershape.cpp        # Shape/DataType 推导
│   ├── lightning_indexer_kl_loss_tiling.cpp            # Tiling 策略
│   └── op_api/                                         # aclnn API (未查看是否有文件)
├── op_kernel/
│   ├── lightning_indexer_kl_loss.cpp                   # AscendC kernel 入口
│   ├── lightning_indexer_kl_loss.h                     # AscendC kernel 主逻辑
│   ├── lightning_indexer_kl_loss_tiling_data.h          # Tiling 数据结构体
│   └── lightning_indexer_kl_loss_tiling_key.h           # TilingKey 模板参数
└── tests/ut/.gitkeep
```

## TilingKey 说明

- **TilingKey 0**: 使用 `atomicAdd` 进行跨核累加
- **TilingKey 1**: 使用 per-core workspace 实现确定性累加

## 构建与测试

- 编译在远程昇腾环境（80.5.5.86）上使用 CMake 完成
- 集成测试通过 `test_aclnn_lightning_indexer_kl_loss.cpp` 执行
- Torch 插件位于 `torch_extension/cann_ops_transformer/ops/` 下

### 测试流程

1. `git add/commit` 本地修改
2. 远程环境拉取代码：
   ```
   ssh 80.5.5.86
   cd /root/data_nvme0n1/jzh/ops-transformer && git pull
   ```
3. `conda activate jzh-ops`
4. `sh test.sh`
