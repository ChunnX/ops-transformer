# 自定义 CCU 算子工程 — 进度（2026-06-09）

## 目标
一套可复用的自定义 CCU 算子工程：能新增全新 CCU 算子并在 Ascend950PR(dav-3510) 4 卡跑通、功能符合预期。

## 环境
- box ops0609; CANN 9.1.0 @ /home/0609; conda mc2_py310; 4x Ascend950PR(dav-3510)
- 激活: source /root/miniconda3/etc/profile.d/conda.sh && conda activate mc2_py310 && source /home/0609/env/ascend-toolkit/set_env.sh
- 工程: /home/developer/ops-transformer/examples/ccu_custom_ops (克隆自 fast_kernel_launch_example)
- 构建: export NPU_ARCH=ascend950; python3 -m build --wheel --no-isolation; pip install dist/*.whl --force-reinstall --no-deps

## Phase 1 [DONE ✅] 工程骨架在 950 打通
- csrc/add/ascend950 (dav-3510) 编译+上板+golden 通过 (allclose True, maxdiff 0)
- 证明整条流水(建目录→bisheng 编 dav-3510→wheel→torch_npu binding→上板→golden)在 950 可用
- 这是该机器上第一个能跑的 ascend950 自定义算子工程，可复用骨架坐实

## Phase 2 [BLOCKED] 自定义 <<<>>> 跨卡 AllGather —— 平台硬限
目标: 让一个自定义 <<<>>> AICore kernel 内 DataCopy 直接读/写对端 rank HBM。
四条通用机制全部上板坐实在 950PR 不可用:
1. HCCL 对称内存窗口 SymWin (HcclCommSymWinRegister): 代码硬门 deviceType_!=DEV_TYPE_910_93 → NOT_SUPPORT (hccl_communicator_host.cc:9320/9344); ascend950 框架树零实现。
2. MC2 op-res-ctx (HcclCreateOpResCtx): 上板实测 ret:5=HCCL_E_NOT_SUPPORT (910/A2A3 路)。
3. ACL VMM 跨进程 IPC (aclrtMemExportToShareableHandleV2 FABRIC + Import): export/setpid/交换均成功, 但 import 上板 ret=207001=ACL_ERROR_RT_MEMORY_ALLOCATION (与 910-only 的 symmetric_memory.cc 用法逐字一致, 是平台不通)。
4. 进程内多卡 P2P (aclrtDeviceEnablePeerAccess, canAccessPeer=1): put kernel 跨卡写对端 VA → aicore error 507035。P2P 只让 runtime memcpy 跨卡, AICore GM 地址空间不含对端 HBM。
结论: 950 上让 kernel 能寻址对端 HBM 的唯一途径是 CCU 硬件引擎(经 channel/jetty 映射 + RemoteAddr/ReadNb/WriteNb)。

## 已交付代码
- csrc/add/ascend950/ : 工程骨架示范(纯计算), 上板 golden 通过
- csrc/all_gather_ccu/ascend950/all_gather_ccu.cpp : 自定义 AllGather (ACL-IPC FABRIC 版), 编译通过、host 侧 alloc/export/setpid/交换全通, 卡在平台门控的 import
- csrc/p2p_ctx/ascend950/ : 进程内 P2P context builder, 验证 P2P DataCopy 撞 507035
- ascend_ops/all_gather_ccu.py, tests/all_gather_ccu/test_all_gather.py, tests/p2p/test_p2p.py

## 下一步(若继续): CCU 引擎路 (effort L)
- host: 从 torch 进程组取 HcclComm(HcclGetCommName/HcclCommGetHandleWithName) → HcclRankGraphGetLinks+HcclChannelAcquire(COMM_PROTOCOL_UBC_CTP, COMM_ENGINE_CCU) 建 channel → HcclThreadAcquireWithStream → HcclCcuKernelRegister/RegisterFinish → HcclCcuKernelLaunch
- kernel: 派生 hcomm::CcuKernel, Algorithm() 用 CcuRep DSL 写 ReadNb/WriteNb/CreateRemoteAddr/NotifyRecord/NotifyWait (Variable 恒 8 字节, 见 CCU_VARIABLE_OPS_CHEATSHEET); 可复用 pto::comm::ccu::CcuReduceMesh1D 模板
- 参考: /home/developer/workspace/hcomm-test/src/ccu_kernel_test/{hcomm_ccu_kernel_demo_test.cc, ccu_kernel_impl/ccu_demo.h}
- 风险: CcuRep 指令图首跑难一次过(项目史 v8→v20.5), 多 bug 仅上板暴露
