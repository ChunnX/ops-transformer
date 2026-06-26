# moe_distribute_dispatch_ccu

CCU 引擎实现的 MoE **dispatch** 算子（`<<<>>>` 直调工程，无 PTA）。

> 设计依据：`docs/CCU_DISPATCH_COMBINE_IMPL_DESIGN.md` v4 / `docs/CCU_HALF_RTT_DISPATCH_COMBINE_LEARNING_PATH.md` v4
> 当前状态：**P0 骨架 + host 编排入口已完成**；CCU setup kernel 与 AICore teardown 核函数为骨架 stub。

## 算子形态（三件套）

| 件 | 文件 | 引擎 | 说明 |
|---|---|---|---|
| host 直调入口 | `moe_distribute_dispatch_ccu_host.{h,cpp}` | Host | 建链 + `remote_va_table` + 注册/启动 CCU kernel + 双流编排 |
| setup（CCU kernel） | `dispatch_setup_ccu.{h,cpp}` | CCU | `hcomm::CcuKernel` 子类，把本卡 token 按 topk 散发到各 expert 卡 HBM |
| teardown（AICore 核函数） | `moe_distribute_dispatch_ccu.{h,cpp}` | AICore | `<<<>>>` 核函数，等数据落地、组装 `expand_x`、回写收讫 ack |

## 文件结构

```
moe_distribute_dispatch_ccu/
├── moe_distribute_dispatch_ccu_host.{h,cpp}   # [HOST]   ★ 编排入口 MoeDistributeDispatchCcu（已完成）
├── dispatch_setup_ccu.{h,cpp}                 # [HOST]   CCU setup kernel（CcuKernel 子类；Algorithm 为 stub）
├── moe_distribute_dispatch_ccu.{h,cpp}        # [DEVICE] teardown 核函数 + <<<>>> 启动包装（核体为 stub）
├── dispatch_ccu_common.h                      # [共用]   tiling 结构 + Per-Rank HBM GM 布局常量
├── test_moe_distribute_dispatch_ccu.cpp       # [HOST]   <<<>>> 直调测试驱动
├── CMakeLists.txt / run.sh
└── scripts/{data_generate,data_compare}.py    # 数据生成 / 精度对比（P0 步骤 6 待补）
```

## 编译

```bash
# A6 仿真器环境（沿用 arch35 工具链）
./run.sh -c ON
# 如 hcomm 不在默认路径，cmake 时加 -DHCOMM_HOME=<hcomm 仓路径>
```

## 待办（见 progress.md）

- P0 步骤 4：`DispatchSetupCcu::Algorithm()` —— 照 `hccl/src/ops/scatter/template/ccu/` 实现
- P0 步骤 5：`dispatch_teardown_aicore` 核函数 —— 照设计文档 §9
- P0 步骤 6：数据生成 + 精度对齐现存 `moe_distribute_dispatch_{setup,teardown}`
