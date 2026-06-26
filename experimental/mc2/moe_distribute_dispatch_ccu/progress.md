# Progress: CCU dispatch 算子（moe_distribute_dispatch_ccu）

- **任务目标**：在 `ops-transformer/experimental/mc2/` 下实现 CCU 引擎 dispatch 算子
- **任务级别**：L2
- **设计依据**：`docs/CCU_DISPATCH_COMBINE_IMPL_DESIGN.md` v4（§14 阶段计划 P0）
- **工程模板**：扁平式 `<<<>>>` 直调工程，照 `experimental/mc2/quant_all_reduce/`

## 已完成阶段

- [x] **P0 步骤 1**：目录骨架 + `DispatchCcuTilingData` / GM 布局常量（`dispatch_ccu_common.h`）
- [x] **P0 步骤 3**：host 直调入口 `MoeDistributeDispatchCcu`（`moe_distribute_dispatch_ccu_host.cpp`）
      —— 建 CCU thread/channel、构造 `remote_va_table`、注册/启动 CCU kernel、双流编排、启动 teardown 全链路

## 当前阶段

- [ ] **P0 步骤 4**：`DispatchSetupCcu::Algorithm()` —— CCU 通信算法主体（散发 token）

## 待办

- P0 步骤 4：`Algorithm()` / `GetKernelSignature()` —— 照 `hccl/src/ops/scatter/template/ccu/` 改造
- P0 步骤 5：`dispatch_teardown_aicore` 核函数 —— 照设计文档 §9（等 READY_FLAG、组装 expand_x、回写 ack）
- P0 步骤 6：`scripts/data_generate.py` + `data_compare.py`；精度对齐现存 `dispatch_{setup,teardown}`

## 关键状态

- **未编译验证**：本开发环境无 CANN 工具链；需在 A6 仿真器环境 `./run.sh -c ON` 编译。
- **待验证项**（设计文档 §13）：
  - V3 ：CCU channel `COMM_PROTOCOL_UBC_CTP` 在 A6 仿真器的支持性
  - V9 ：`HcclThreadAcquireWithStream` 对 CCU 引擎是否可用（当前用 `HcclThreadAcquire`+`ResGetInfo`）
  - V10：算子 host 层直接裸调 `HcclCcuKernelRegister`（仓库内仅框架调过，无算子直调先例）
- **CMake 待确认**：`HCOMM_HOME` 指向 hcomm 仓；部署态 CANN 包内 CCU 头文件确切路径待 A6 环境核对。
