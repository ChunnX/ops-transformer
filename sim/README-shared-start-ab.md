# 共享 GMM 提前启动：camodel A/B

## 两版源码

- 基线分支：`sim/mega-moe-shared-start-baseline`，起点 `56b62be90`。
- 优化分支：`sim/mega-moe-shared-start-optimized`，起点 `6c8a5eddc`，包含 PR !12907。
- 仿真工具来源：`macech/mega-moe-sim` 的 `b15137e4`。迁移 `sim/`、`mega-sim`、`mega-sim-deploy`，未迁入其 W4 预取 2→4 的实验修改。
- 两版使用同一组输入生成、校验和标记工具；kernel 保留各自同步实现。

## 目标配置

显式指定 `--config sim/configs/shared_start_bs128.json`：

| 参数 | 值 |
|---|---|
| EP / 每卡核数 | 8 / 28 AIC + 56 AIV |
| BS / H / GMM1 输出 | 128 / 7168 / 4096 |
| intermediate | 2048，SwiGLU 后宽度 |
| topk / 每卡专家 | 6 / MoE 3 + shared 1 |
| 输入 | 预量化 E4M3 + E8M0 scale，`[128,7168]` + `[128,112,2]` |
| MoE / shared | A8W4 NZ_C0_32 / A8W8 ND |
| topk 权重 / 输出 | FP32 / BF16 |
| 最大输入 token 数 / max_recv | 128 / 0（host 默认容量） |
| key / prefetch / combine quant | 730152 / 0 / 0 |
| fixture | random，seed 56；有限离散 FP8/FP4 数据，均衡循环路由 |
| 编译并发 | 8 |

实机脚本 `expert_scales_dtype=torch.float32`；`topk_weights_type=0` 是属性开关，不是数据类型编码。
每源卡对每专家发送32个 token，每专家全局接收256个，每卡总接收768个。
本用例复现 shape、类型、分核和均衡负载，不承诺与 torch 脚本 seed=56 的随机字节、路由排列相同。
两版输入的逐字节一致性由 `compare_cases.py` 验证。

## 环境与构建

需要兼容当前主线 API 的编译 CANN，以及内部 debug 版 `npusim -c`。
分别在两版源码目录运行以下命令，所有 case 路径使用不同的新目录。
从没有 source 另一套 CANN 的终端启动：

```bash
# 先做协议检查，再构建和生成；此步骤不启动仿真。
./mega-sim test
python3 sim/marks.py --check
source /path/to/compiler/cann/set_env.sh
./mega-sim prepare --config sim/configs/shared_start_bs128.json --case /data/megamoe/baseline-bs128
```

优化版将最后一个路径改成 `/data/megamoe/optimized-bs128`。
构建产物和输入有源码 hash 绑定；不要用另一版 kernel 冒充当前构建。
编译脚本按 FP32/BF16 区分唯一输入签名，同时验证构建 dtype 与 case 一致。
首次配置需要下载固定依赖；离线环境使用 `--dependencies` 指向独立缓存，参照 `sim/README.md`。

确认两版可启动且输入相同：

```bash
python3 sim/compare_cases.py /data/megamoe/baseline-bs128 /data/megamoe/optimized-bs128
```

在分别加载仿真环境的独立终端顺序运行：

```bash
source /path/to/debug/cann/set_env.sh
cd /data/megamoe/baseline-bs128
npusim -c /data/megamoe/baseline-bs128
# 完成并保存日志后，再运行 optimized-bs128。
```

也可使用 `mega-sim-deploy --build-env ... --sim-env ... --config sim/configs/shared_start_bs128.json --case NEW_DIR` 一键执行。
该入口没有 wrapper 超时；目标配置给 `mega-sim run/all` 设定7天上限。此前默认8卡用例约三天仅是原仓用户反馈。
SoC 默认沿用仓库 `Ascend950PR_9589`；执行前应确认目标内部包支持这一型号及28/56核任务。

## 只生成输入的本地准备

```bash
source /path/to/compiler/cann/set_env.sh
./mega-sim build --host-only --config sim/configs/shared_start_bs128.json
./mega-sim inputs --config sim/configs/shared_start_bs128.json --case /data/megamoe/baseline-inputs
./mega-sim validate --inputs-only --case /data/megamoe/baseline-inputs
```

input-only 目录只有 `top.pending.json`，不能直接仿真。服务器端用同一份源码和配置 `build` 后执行
`./mega-sim attach --case INPUT_DIR`，会重新比对真实 tiling 和内存尺寸，再生成 `top.json`。

## 观测口径

使用原仓的 `set_cond` 协议，记录 Scalar 调用边界，未新增同步或 GM/UB 缓冲区。
两版相同 ID 保持相同含义。`python3 sim/marks.py --list` 是本次实际编号表。
只保留本次 A/B 的19项标记；未迁入的原仓编号保留空缺，不重用于别的含义。

| 阶段 | 主要角色 | END 的证据与含义 |
|---|---|---|
| RESET_WORKSPACE / PACK_INPUT / SEND_INDEX / SEND_COUNT | AIV | MTE/V指令已下发，END不证明异步搬运完成 |
| INPUT_PREPARE | AIV | 函数返回；两版等待条件不同，应看MTE3和跨核事件 |
| WAIT_INPUT_READY | AIC | 等待函数返回；基线含MTE2_S，优化版直读输入延后准备等待 |
| SHARED_GMM1 | AIC、AIV0 | GMM1/激活调用范围；MMAD、FIX和V/MTE完成看对应指令 |
| WAIT_REMOTE_COUNT | AIV1 | WaitForCountTable返回，整表轮次校验完成 |
| COUNT_PREPARE | AIV | count准备调用范围，后续发布依赖看具体指令 |
| DISPATCH_FIRST / DISPATCH_SLICE | AIV1 | 首wave/后续slice的搬运指令下发范围 |
| WAIT_DISPATCH_DATA | AIC / 参与等待的AIV | Scalar轮询返回，ready已观察到 |
| MOE_ENTRY_SYNC | 各角色 | 优化版独立入口同步；基线为空边界，原有同步留在原位 |
| GMM1_SLICE / GMM2_WAVE / MOE_STAGES | 各角色 | slice/wave/完整MoE调用范围，不是各流水完成栅栏 |

第一轮比较共享MMAD/FIX完成、共享激活完成、首批dispatch ready及MoE首条MMAD。
若提前量延续到MoE，再沿后续wave、共享GMM2和出口追踪。
不能把不同rank/核的BEGIN和END配对；不同运行以各自kernel起点对齐。
set_cond会引入少量Scalar指令，A/B保持相同粒度；绝对仿真耗时不直接等于实机耗时。

## 输入协议审查

- 两版完整kernel ABI均为23个参数和MIX 1C2V；生成器的窗口初始化尾项保持不变。
- 参数槽位、TensorList、预量化拼接、定址count区、workspace、peermem和主线固定依赖在A/B间一致。
- host helper直接编入当前host源码；按combine_comm_mode=0请求dispatch去重并执行当前host的容量/收益门，再算UB、wave和workspace。
- FP32权重变化覆盖编译签名、输入字节、Unpermute UB预算和校验；其他量化编码沿用原工具。
- 各rank窗口和workspace显式初始化，保留guard、真实peer地址、TensorList内部指针检查。
- 输出检查只验证count、有限值和guard；不等于数值精度验证。默认仍不启用AST输出抓取。

本地验证状态见 [SIMULATION_VALIDATION.md](../SIMULATION_VALIDATION.md)。设备编译及内部debug多卡运行须在服务器确认。
