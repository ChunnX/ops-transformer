> 原仿真仓说明保留供协议参考。本次 A/B 的配置、标记范围和验证状态以 [shared-start A/B](README-shared-start-ab.md) 为准。

# MegaMoE 仿真工程约定

入口是仓库根目录 `./mega-sim`，默认配置是 `sim/configs/bs72.json`。

## 修改入口

| 要改的内容 | 文件 / 命令 |
|---|---|
| kernel | `mc2/mega_moe/op_kernel/arch35/`，修改后 `./mega-sim prepare` |
| shape、核数、专家数、seed | 复制 `configs/bs72.json`，使用 `--config` |
| 输入构造、路由、量化原始字节 | `tensors.py` |
| host tiling 参数适配 | `tiling_driver.cpp` |
| CANN 编译入口、dtype/format 分支选择 | `build.py` |
| debug task/SQ、地址分配、TensorList | `case.py` |
| case 与输出检查 | `validate.py` |
| ELF 双核入口提取 | `elf.py` |
| 默认 COND 打点、编号和后续维护 | [README-marks.md](README-marks.md)、`marks.py`、kernel `common/mega_moe_sim_marks.h` |

当前适配一组明确的编译协议：FP8 E4M3 输入直通、MoE FP4 NZ_C0_32、共享 FP8 E4M3 ND、SwiGLU、BF16 topk weights/BF16 输出、MTE、无 prefetch、key 730152。shape 可以在这个协议的约束内变化。切换量化类型、格式、激活算法或 tiling key 需要一起扩展编译分支选择、tiling 和输入编码；不接受只改数字后静默跑错分支。

共享专家选择 ND 是本仿真配置的明确选择；tiling key 本身不编码 ND/NZ，`build.py` 会额外匹配格式，避免从同一个 key 的多个产物中随意取第一个。

## 编译与源码绑定

编译调用上游 CMake 产生正式 wrapper/参数文件，再由 `asc_opc --tiling_key=730152` 编译唯一的 dtype/format 分支。输出在 `sim_build/`；改变源码、helper 或生成逻辑后，旧二进制不能用来生成新 case。case 参数文件不参与源码 hash，因此仅修改 bs/shape/seed 不必重新编译 kernel。

`tiling_driver.cpp` 与当前 `op_host/op_tiling/arch35/mega_moe_tiling.cpp` 在同一个 C++ translation unit 编译，直接调用上游 adaptive buffer、wave、L1 和 workspace 函数。它不是调用完整的 ACLNN host tiling：仅适配上面列出的固定协议，shape 约束由 `common.py` 检查。上游改动函数签名或 kernel ABI 时应编译/检查失败并修改适配器。

默认 `max_recv_tokens=0` 沿用上游容量规则，当前为 1728/rank；实际均衡接收为 432/rank。二者不能混淆。若研究紧缩 workspace，可显式设置 `max_recv_tokens=432`；生成器拒绝小于实际路由接收量的容量。

## 输入协议

| 槽位 | 内容 | 数据布局 |
|---|---|---|
| 0 | Mc2MoeContext | C++ `sizeof`/`offsetof` 校验；8 个真实 peer 地址 |
| 1 | x | `[72,5120]`，FP8 E4M3 原始字节 |
| 2 | topkIds | `[72,6]`，int32；expert id 仅 0..23 |
| 3 | topkWeights | `[72,6]`，BF16，默认每项为 BF16(1/6) |
| 4/5 | MoE W1/W2 | TensorList 中一个堆叠 tensor；FP4 NZ_C0_32 |
| 6/7 | MoE scale1/2 | `[3,4608,80,2]` / `[3,5120,36,2]`，E8M0 ND |
| 8/9/10 | bias1/bias2/activeMask | 显式 null |
| 11 | 输入 scale | `[72,80,2]`，E8M0 ND |
| 12/13 | shared W1/W2 | TensorList 中一个堆叠 tensor；FP8 E4M3 ND |
| 14/15 | shared scale1/2 | `[1,4608,80,2]` / `[1,5120,36,2]`，E8M0 ND |
| 16/17/18 | shared bias1/bias2/maskBuffer | 显式 null |
| 19 | y | `[72,5120]` BF16 |
| 20 | expertTokenNums | `[3]` int32，预期 `[144,144,144]` |
| 21/22 | workspace / tiling | 从当前 C++ 源码计算并序列化 |

FP4 先把逻辑 `[E,N,K]` 排成 `[E,K/32,N/16,16,32]`，再将相邻 K 的偶数元素放低半字节、奇数元素放高半字节。scale 为每 32 个 K 元素一个 E8M0 字节，逻辑上用 `[K/64,2]` 表达。TensorList 的第一个 uint64 是 pointer table 的字节偏移 8，第二个 uint64 是同一分配中 `base+512` 的实际 tensor 地址。

`fixture.kind=sparse` 默认生成非零、有限、可重复的稀疏权重；`random` 生成有限离散随机输入/权重。原始 FP8/FP4 编码取明确的 ±0.5、±1、±2，不把随机 uint8 当浮点数据。`scale_exponent=-2` 对应 E8M0 字节 125，即 scale=0.25。路由按全局 token/route 序号循环到全部 24 个 MoE 专家，默认每源卡向每专家发送 18 个 token，总计 144；共享专家处理本卡 72 个 token，不进入 topk。

## 混合核 task 与地址

arch35 kernel 声明 `KERNEL_TYPE_MIX_AIC_1_2`。SQ 为 AIC task；task 使用 `sqe_type=AIC, mix=1, ratio=2, block_dim=32`，与 64 AIV 对应。`BIN.name=aic.bin`，`BIN.sub_name=aiv.bin`，分别装载，AIC/AIV 的 `hbm_para_addr` 相同。`group_dim=1, group_slice=32` 等字段沿用 arch35 参考 MIX 配置；用户已报告默认配置在内部 debug 包跑通，其他工具版本仍需验证。

ELF 解析只接受最终 ET_EXEC 和唯一一组非空 MIX 符号；按 section 地址保留 `.text`、`.rodata`、`.data` 等所有 ALLOC section 的相对位置，再从对应入口切出镜像。不能只用 objcopy 提取 `.text`，因为本 kernel 有 `g_winRankAddr_` 等 GM 全局数据。源码/二进制/配置的 hash 和符号偏移随 case 保存。

每卡虚拟地址跨度 `0x10000000000` 是 **1 TiB**，不是参考 dispatch 文档中注释的 64 GB。各区都加 rank 偏移：

| 区域 | rank 内偏移 / 预留 |
|---|---|
| AIC / AIV 镜像 | `0x40000000` / `0x41000000`，各 16 MiB |
| 通信窗口 | `0x8000000000`，C++ 计算尺寸，含 60 KiB 异常 dump 前缀 |
| 参数表 | `0x8100000000`，预留 16 MiB |
| stack | `0x8200000000`，预留 4 GiB |
| SQ/CQ/swap | `0x9010000000` / `0x9018000000` / `0x9028000000` |
| tensor/workspace arena | `[0x9100000000,0xA000000000)`，分配按 64 KiB 对齐 |

通信窗口显式装载全零文件，避免依赖仿真器新内存初值。workspace 同样清零。每项数据先补齐 512B，再放 512B guard（0xD3）。输出数据先填 BF16 NaN/int32 -1。默认不配置 `ast_output_array`，与 A5 参考生成器相同；仅当内部仿真器支持该字段且配置 `simulation.dump_outputs=true` 时才生成输出/guard 抓取项并运行输出检查。

按照 A5 参考，kernel 的全部 0..22 参数（含输出 19/20）都由 `input_para_array` 装载，`output_para_array=[]`。额外第 23 槽是**仿真器装载专用尾项**：初始化通信窗口及其 guard；kernel 的原始 0..22 槽保持不变，不读取这个尾指针。参数区预留 16 MiB，24 项指针只占 192B。该默认装载流程已获用户实跑成功反馈；CPU 测试不能替代新工具版本的实际装载验证。

`validate` 检查输入 hash、字节数、槽位、TensorList 指针、真实跨卡窗口、路由均衡、AIC/AIV 双文件及地址区间。`check-output` 检查 8 卡 count、输出均已写入且有限非零、guard 完整；**不等同于数值精度通过，也不能证明 workspace 内部任意两个逻辑区域不会被 kernel 错写**。

## 参考来源和验证范围

基线：`cann/ops-transformer` upstream/master，`b8c0215b466ed5d2d8aa292273320820d3550d9e`，2026-09-24 拉取。跟进上游时显式 `git fetch upstream master` 后 merge/rebase，再重新编译与生成，不在运行时自动更新源码。

- [ops-transformer-sims demo/megamoe-profile](https://gitcode.com/macech/ops-transformer-sims/tree/demo/megamoe-profile)，`e451315c64a497d262cc6ff3119552c2ffdeaa04`：参考原生 wrapper 构建、MIX 配置、ELF 入口和 TensorList 装载。它的 release 仿真命令不是本仓默认入口。
- [temp_storage](https://gitcode.com/macech/temp_storage)，`fbd547f7986d327c4d27783e864d9b5ff7d66c09`：参考内部 debug 多卡 JSON 格式和 rank 地址分段。dispatch 的 HcclOpParam、tiling 和纯 AIV task 不适用于 MegaMoE。
- [Wchun/megamoe_camodel](https://gitcode.com/Wchun/megamoe_camodel)，`810cd4073bffd0bb14eed88f2cc3f6dc5740d48c`：参考混合核主/子入口与地址检查方法；其 1c1v、TOML、参数槽位和地址空间不沿用。
- [Wchun A5 分支](https://gitcode.com/Wchun/megamoe_camodel/tree/feat/a5-camodel-case-generation)，`f15ce60`：确认 arch35 的 AIC MIX 1c2v、双 bin、全部参数放 input_para_array、无默认 AST 抓取的 JSON 格式。该分支明确不支持 W4/NZ/预量化输入，不能直接替代本仓 730152 的输入和 tiling 适配。其按 FUNC size 截取的方式只适用于自包含代码；本仓保留实际 ELF 的全局数据和相对布局。

本机已验证：上游 host/native wrapper 构建、唯一 dtype/format 分支匹配、真实 C++ tiling/helper、CPU 布局/序列化和负例检查。默认 shape 的 helper 输出：432B tiling，48,495,616B workspace，4,939,264B window，16,400B context，L1 A/B buffer=3/4、scaleK=2048。

完整默认 shape 的 8 卡 CPU 序列化检查总数据量约 1.18 GB（含初始化区），确认全体 24 个 MoE 专家各收 144 tokens；离线依赖包在新构建目录、无网络的沙箱中完成了 host/native wrapper/helper 构建。

2026-09-28 用户报告：使用本仓流程在内部服务器完成默认 8p 仿真，耗时约三天。
内部工具精确版本、完整运行日志、guard/数值精度结果尚未收集，不据此宣称数值验证通过。

本机 CANN 9.2.0-beta.2 编译 kernel 时缺少上游要求的 `ReadGmBypassDCache`/`WriteGmBypassDCache` 与 Hcomm `BatchHandle` 等 API，本机尚不能复现设备编译及内部 debug 8p 仿真。请使用兼容的内部工具包。失败会保留 `sim_build/logs/kernel.log`，不会产生成功 manifest 或用旧 kernel 冒充本次编译产物。
