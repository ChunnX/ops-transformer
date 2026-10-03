# SageAttention2 设备核 camodel 仿真 harness（B2 门禁）

本目录是设备核第一次获得本地数值证据的仿真工具链：在 CANN 9.2.0-beta.1 容器的
Ascend950PR camodel（dav-3510，1C2V 拓扑）上跑通 `op_kernel/sageattention2.cpp`
canonical 注册内核，并与宿主孪生真值（`sageattn_fp8/ascendc/twin_reference.h`）
做端到端逐位数值比对。

## 文件清单

| 文件 | 作用 |
| --- | --- |
| `sim_case_common.hpp` | 共享用例定义（形状/模式/种子/输入生成/key_mean/tiling 单点/bin IO） |
| `sim_reference.cpp` | golden 参考程序（g++ 宿主编译）：生成输入 + tiling bin，孪生流水算 O/LSE |
| `sim_main.cpp` | 设备 harness（bisheng `--run-mode=sim` 混编）：读 bin → GM → chevron 发射 → 回读落盘 |
| `compare_outputs.py` | numpy 比对：O 位级一致率/ulp 分布、LSE 相对差、结论判定 |
| `run_sim_e2e.sh` | 一键编排：参考编译运行 → sim 编译 → camodel 运行 → 比对 |

## 用例（任务书冻结小 case）

B=1、Hq=Hkv=1、Sq=Sk=128、D=64、fp16、per_thread、pv_accum=fp32+fp16、非因果、
smooth_k=1、sm_scale=1/8；输入 `std::mt19937(20261003±{1,2,3})` 生成 fp16 [-2,2)
（整数域确定性映射，跨编译器位级一致）。

## 复现步骤（容器内一次性命令形态）

```bash
SNAP=<快照仓根>            # 含 experimental/attention/sageattention2 的目录
WORK=<可写工作目录>
docker run --rm \
  -v <宿主输出目录>:/work/out \
  -e SAGE_ATTN_SIM_OUT=/work/out/sa2_sim_out \
  sageattn2-dev:latest \
  bash $SNAP/experimental/attention/sageattention2/tools/sim/run_sim_e2e.sh \
       $SNAP/ops-transformer /work/out/sa2_sim_e2e
```

四步：g++ 参考程序 → bisheng 设备 sim 编译 → camodel 仿真运行 → 数值比对。
退出码 0 = 数值判定 PASS。

## 启动机制结论（2026-10-03 实证）

1. **编译**：`bisheng --run-mode=sim -xasc --npu-arch=dav-3510 -std=c++17
   <kernel + harness> -L<$ASCEND_HOME_PATH>/aarch64-linux/simulator/Ascend950PR_9599/lib
   -lruntime_camodel -o sim_app`。混编 TU 内 chevron 语法
   `sage_attention2<<<blockDim, nullptr, stream>>>(...)` 被 lowering 成
   `aclrtBinaryLoadFromData / aclrtBinaryGetFunctionByEntry / aclrtLaunchKernelWithHostArgs`
   调用链（真机 .o 的未定义符号在 sim 形态由 libascendcl 桩 + camodel runtime 承接）。
   canonical 三件套（REGISTER_TILING_DEFAULT / GET_TILING_DATA_WITH_STRUCT /
   KERNEL_TASK_TYPE_DEFAULT(MIX_AIC_1_2)）在 sim 形态原样生效，内核源文件零改动。
2. **运行栈**：`LD_LIBRARY_PATH` 须指向 cann-9.2.0-beta.1 侧 camodel 目录
   （`/usr/local/Ascend/cann-9.2.0-beta.1/tools/simulator/Ascend950PR_9589/camodel`，
   内含 libruntime_camodel/libascend_hal/libstars_wrapper 全家桶）。
   ascend-toolkit 侧 simulator 目录仅可作链接（`-L`），两栈混用会在
   libstars_wrapper.so 报 `STARS_TOP::ext_write_ffts_plus_context` 符号缺失。
   `msprof op simulator`（缺 libascend_hal）与 `npusim record -s Ascend950`（同样
   落到 9589 栈）均可用，但直接以 env 驱动 `./sim_app` 最小可控，本 harness 采用之。
3. **blockDim 语义**：MIX_AIC_1_2 下 blockDim = AIC 任务数（tiling.core_num_used）；
   每任务硬件展开 1 AIC + 2 AIV，AIV 侧 `GetBlockIdx() = 2*aic_blk + sub`、
   `GetSubBlockIdx() = sub`，与内核 `Init` 的 `core_idx_ = GetBlockIdx()/2` 约定一致
   （探针实证：blockDim=1 时 AIV 两上下文为 (0,0)/(1,1)）。
4. **rt 宿主 API**：camodel 的 `libruntime_camodel.so` 导出全量 `rt*` C ABI
   （rtSetDevice/rtMalloc/rtMemcpy/rtMemset/rtStreamCreate/rtStreamSynchronize/
   rtDeviceReset），harness 用 `extern "C"` 自声明直连，不依赖 acl 框 compelled 头。

## 数值结论

（见交付报告；本节随每次运行由 compare_outputs.py 输出刷新）

## 已知边界

- smooth_k=1 用例的 LSE 含 `q·km × sm_scale` 校正项；设备分支 `lse_corr_` 基线恒 0
  （`LoadAndQuantizeQ` 步骤 4 仅宿主/KT 分支实现），该差属设备核已知未实现域，
  比对脚本单独标注，不算 harness 失真。
- 快照运行：并发会话在迭代 op_kernel/ 既有文件，端到端验证须对 git 快照（HEAD
  archive）执行，避免工作树中间态污染编译。
