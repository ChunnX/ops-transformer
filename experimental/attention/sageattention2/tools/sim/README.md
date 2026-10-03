# SageAttention2 设备核 camodel 仿真 harness（B2 门禁）

本目录是设备核第一次获得本地数值证据的仿真工具链：在 CANN 9.2.0-beta.1 容器的
Ascend950PR camodel（dav-3510，1C2V 拓扑）上跑通 `op_kernel/sageattention2.cpp`
canonical 注册内核，并与宿主孪生真值（`sageattn_fp8/ascendc/twin_reference.h`）
做端到端逐位数值比对。

## 文件清单

| 文件 | 作用 |
| --- | --- |
| `sim_case_common.hpp` | 共享用例定义（形状/模式/种子/输入生成/key_mean/tiling 单点/bin IO） |
| `sim_reference.cpp` | golden 参考程序（bisheng 宿主模式编译）：生成输入 + tiling bin，孪生流水算 O/LSE |
| `sim_main.cpp` | 设备 harness（bisheng `--run-mode=sim` 混编）：读 bin → GM → chevron 发射 → 回读落盘 |
| `compare_outputs.py` | numpy 比对：O 位级一致率/ulp 分布、LSE 相对差、结论判定 |
| `run_sim_e2e.sh` | 一键编排：参考编译运行 → sim 编译 → camodel 运行 → 比对 |

## 用例（任务书冻结小 case）

B=1、Hq=Hkv=1、Sq=Sk=128、D=64、fp16、per_thread、pv_accum=fp32+fp16、非因果、
smooth_k=1、sm_scale=1/8；输入 `std::mt19937(20261003±{1,2,3})` 生成 fp16 [-2,2)
（整数域确定性映射，跨编译器位级一致）。

## 复现步骤（算子门禁容器内一次性命令形态）

```bash
cd ops-transformer/experimental/attention/sageattention2   # 算子目录
# 前置：bash docker/init.sh（CATLASS 置备）；产物落算子 docker/out/（挂载为 /work/out）
SAGE_ATTN_SIM_OUT=/work/out/sa2_sim_out \
./docker/sage run \
  bash tools/sim/run_sim_e2e.sh /work/ops-transformer /work/out/sa2_sim_e2e
# 快照形态（避免工作树中间态污染编译，见"已知边界"）：
#   git archive HEAD --prefix=ops-transformer/ | tar -x -C <快照父目录>
# 并把 <算子仓根> 参数换成快照里的 /work/ops-transformer 对应挂载路径, 其余参数不变。
```

四步：bisheng 宿主模式参考程序 → bisheng 设备 sim 编译 → camodel 仿真运行 → 数值比对
（全部编译统一采用 BiSheng 编译器）。
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

（判定输出由 compare_outputs.py 运行时打印至 stdout；交付报告与本节摘要按每次运行人工回填）

## 已知边界

- smooth_k=1 用例的 LSE 含 `q·km × sm_scale` 校正项；核内不算该项（桥接层契约，
  见 op_kernel LoadAndQuantizeQ 注释），参考程序与设备 harness 经 `lseCorrection`
  GM 入参两侧同源传入、V1 FinalizeAndStoreOutput 直读；LSE 超差即属数值失真，
  比对脚本单独标注。
- 快照运行：并发会话在迭代 op_kernel/ 既有文件，端到端验证须对 git 快照（HEAD
  archive）执行，避免工作树中间态污染编译。
