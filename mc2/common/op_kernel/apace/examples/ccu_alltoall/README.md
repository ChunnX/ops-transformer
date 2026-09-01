# CCU AllToAll 纯通信验证（阶段2 上板精度 PASS ✅）

> **阶段状态**：阶段1（编码及编译验证）完成——2026-08-29 用户服务器编译通过
> （commits：`e255a5121` 工程初始+检视修复 → `1a80ff575` 编译修复1 → `1e013df40` 编译修复2）。
> **阶段2（NPU 上板精度验证）完成——2026-09-01 标卡4P 服务器 4 rank 全 PASS**
> （commit `1f888526c` 文件重命名 + 跟随 asc-comm 正式版 host launch API，同 commit 复验编译零警告）。
> 下一步：阶段3 通算融合（叠 matmul，Blaze 恢复点见文末）。

2 卡（可扩展）均匀 AllToAll + CCU 直调（FP16，strideCount 型，commTurn 多轮事件桥）。
通信全部运行在 `__ccu_host__` CCU kernel 内（streamCcu），AIC 桥 kernel（streamAicore）
预写 GM 共享内存参数区（本工程称"参数桥"，术语说明见 docs/CODE_GUIDE.md §2.1）
+ wake + 轮询 round done——双流架构与 `../ccu_all_gather_matmul` 同款。

> 设计文档见 `docs/DESIGN_ANALYSIS.md`（含 CANN `ccu_kernel_all_to_all_mesh1d.h` 代码搬运映射表）。
> **本工程 Blaze-free**：纯通信阶段不依赖 ops-tensor/Blaze，编译只需 CANN 环境。

## 目录结构

```
ccu_alltoall/
├── CMakeLists.txt              # 工程根（Blaze-free）
├── README.md                   # 本文件
├── docs/
│   └── DESIGN_ANALYSIS.md      # 设计文档（CANN 代码搬运映射表）
├── scripts/
│   └── run.sh                  # 一键编译 + 运行（唯一入口脚本）
└── src/                        # 单翻译单元：.asc 经相对 include 引入其余三个文件
    ├── alltoall_ccu_kernel_impl.asc # CCU kernel（__ccu_host__）+ AIC 桥 kernel + launch 入口
    ├── alltoall_aicore_kernel_impl.h     # AIC 侧：参数区预写 + wake + per-round done 轮询
    ├── alltoall_host_impl.h         # Host 编排：AllToAllContext + PrepareAllToAll
    └── main.cpp         # Host 入口：溯源编码数据（元素级可区分）+ 断言校验 + 双模式
```

> 注意：CMakeLists 只编译 `src/alltoall_ccu_kernel_impl.asc` 一个源文件——ASC 编译模型下
> host 侧 `.h`/`.cpp` 由 `.asc` 单翻译单元 include 引入（对齐 `../ccu_all_gather_matmul` 结构），
> 不要拆成独立编译单元。

## 编译 + 运行（在用户的编译/运行服务器执行）

```bash
# 确认 CANN 环境
echo $ASCEND_HOME_PATH   # 应指向 CANN 安装目录
source $ASCEND_HOME_PATH/set_env.sh

cd <本目录>
bash scripts/run.sh
```

`scripts/run.sh` 自动完成：source 环境 → `export HCCL_OP_EXPANSION_MODE=CCU_SCHED` → cmake 编译 →
运行默认 precision 用例（每块 65536 元素 FP16 = 128KB，commTurn=1）。运行产物（`output/`、`build/`）
统一落地在工程根目录。

### 手动分步（调试用，在工程根目录执行）

```bash
source $ASCEND_HOME_PATH/set_env.sh
export HCCL_OP_EXPANSION_MODE=CCU_SCHED

rm -rf build
cmake -S . -B build
cmake --build build --target ccu_alltoall --parallel 4

./build/ccu_alltoall precision              # 默认：128KB 块，commTurn=1
./build/ccu_alltoall precision 65536 2      # commTurn=2 多轮事件桥验证
./build/ccu_alltoall precision 16385 1      # 32770 字节块：触发 goSize 尾部分支
./build/ccu_alltoall perf 65536 1 10        # perf 模式：循环 10 次不落地
```

## 验证环境与验证方法（2026-09-01 阶段2 实录）

### 验证环境

| 项 | 值 |
|---|---|
| 服务器 | 标卡4P（4 卡 Ascend950PR），共享服务器（多开发者共用 NPU） |
| CANN | `cann-9.2.0`（`~/Ascend/cann` 符号链接指向；**须 ≥ 9.2.0**，ccu_host_launch 依赖） |
| 环境变量 | `source ~/Ascend/cann/set_env.sh` + `export HCCL_OP_EXPANSION_MODE=CCU_SCHED` |
| NPU 排队 | **必须走 `qrun` 排队工具**，禁止直接跑（见下） |

### 编译验证方法

```bash
source ~/Ascend/cann/set_env.sh   # 或在预配环境的 tmux 会话（canon）内执行
cd <本目录>
rm -rf build
cmake -S . -B build
cmake --build build --target ccu_alltoall --parallel 4
# 成功标志：Built target ccu_alltoall；工程开 -Werror，零警告才算通过
```

编译不占 NPU，无需排队。产物仅依赖 CANN 自带 `libhcomm.so` + `libasccomm_ccu.so`
（`readelf -d build/ccu_alltoall | grep asccomm` 可复核）。

### ⚠️ 精度验证必须走 qrun 排队（共享服务器铁律）

标卡4P 是多开发者共享服务器，**任何占用 NPU 的运行（precision/perf）都必须通过
`qrun` 排队工具提交**，直接裸跑会与他人任务抢卡冲突。`qrun -h` 可看完整帮助。

> **🚫 铁律：禁止使用 `qrun clear`**——该命令会清除队列中**所有人**的排队任务
> （含他人正在等待的任务），属于破坏共享秩序的危险操作，任何情况下不得使用。
> 只允许 `qrun list`（查队列）/ `qrun stats`（查占用看板）等只读操作；
> 自己的任务误提交时等它执行完或超时，不得动全局队列。

**排队的两个坑（2026-09-01 实测）：**

1. **qrun 的执行 shell 不继承你的环境**——它起的是干净 shell，CANN 环境
   （`set_env.sh`）和 `HCCL_OP_EXPANSION_MODE=CCU_SCHED` 都**必须写进任务命令里**。
   否则报 `libhccl_alg.so: cannot open shared object file`（缺 CANN 环境）。

2. **管道会让 qrun 误报成功**——命令里用 `... | tee log` 时，qrun 的退出码取的是
   管道最后一个命令（tee）的，二进制失败也会显示"执行成功"。改用重定向
   `> log 2>&1`，让退出码真实反映测试结果。

**正确提交姿势（可直接复制）：**

```bash
qrun -n c00697382_a2a_precision -t 600 \
  "source /home/c00697382/Ascend/cann/set_env.sh && export HCCL_OP_EXPANSION_MODE=CCU_SCHED && cd <本目录绝对路径> && ./build/ccu_alltoall precision > /tmp/a2a_precision_qrun.log 2>&1"
# -n 任务名：工号前缀（共享看板归属）；-t 超时秒数（默认 600s）
# 提交后 qrun 自动排队并执行；结果看 /tmp/a2a_precision_qrun.log
# 查队列：qrun list    查看板：qrun stats
```

### 阶段2 验证结果（2026-09-01）

- 用例：precision 默认参数——4 rank，FP16，128KB/块（65536 元素），commTurn=1
- 结果：**4 个 rank 全部 PASS**（每 rank 262144 元素逐块逐元素断言零错位），
  `=== ALL RANKS PASS ===`，退出码 0，耗时 1min7s（含排队）
- 同时验证了 asc-comm 跟随改动的运行时正确性：动态 dieId（`ENDPOINT_ATTR_DIE_ID`
  查询）→ `phyDieMask`/Variable/Event 分配，4 字段 `asccomm_ccu_schd` 正式版启动路径，
  CANN 自带 `libasccomm_ccu.so` 端到端跑通

## 精度校验说明

程序内置 host 侧逐块逐元素断言（无需外部 golden）：

- **数据设计（溯源编码）**：每个 uint16 元素位型 = `srcRank<<12 | dstBlock<<8 | tPos`，
  其中 `tPos = (t ^ (t>>8)) & 0xFF`（块内位置摘要）——**每个 slice、每个元素都不同**；
  纯字节搬运验证，位型不解释为 FP16 数值（CCU Write/GroupCopy 均为字节级拷贝）
- **发送侧**：rank i 的 sendBuf 第 j 块元素 t = `Encode(src=i, dst=j, t)`
- **期望结果**（AllToAll 语义 `recv_i[r] = send_r[i]`）：rank i 的 recvBuf 第 r 块元素 t
  = `Encode(src=r, dst=i, t)`
- **输出**：每 rank 打印 `AllToAll PASS/FAIL`；FAIL 时打印首错定位
  （块号/元素偏移/期望 vs 实际位型 + 位型直接反解 srcRank/dstBlock，可分类
  src 混淆 / 块混淆 / 块内偏移 / 数据损坏；tPos 非线性混合使"Δ 为 256 整数倍"的
  块内错位亦可检出）
- 落地 `output/<rank>/npu_out.bin`（uint16 位型），可用 python 复核：

```python
import numpy as np
R = 4  # rank 数（阶段2 实测 4 卡；改卡数时同步修改）
elems = 65536
MY_RANK = 0  # 复核哪个 rank 的输出
t = np.arange(elems)
raw = np.fromfile(f"output/{MY_RANK}/npu_out.bin", dtype=np.uint16)
for r in range(R):
    expect = (r << 12) | (MY_RANK << 8) | ((t ^ (t >> 8)) & 0xFF)
    assert np.all(raw[r*elems:(r+1)*elems] == expect), f"block {r} mismatch"
print("PASS")
```

## 关键编译参数

| 参数 | 值 | 说明 |
|---|---|---|
| `--npu-arch` | `dav-3510` | Ascend 950 |
| `HCCL_OP_EXPANSION_MODE` | `CCU_SCHED` | CCU 调度模式（运行时必须） |
| 链接库 | `hcomm asccomm_ccu c_sec ascendcl acl_rt ascendc_runtime pthread dl` | 无 Blaze 三件套（platform/tiling_api/register） |
| CANN 版本 | ≥ 9.2.0 | `ccu_host_launch.h` 头 + `libasccomm_ccu.so` 库需 CANN 9.2.0+ 自带（asc-comm 正式版 host launch，见下） |

> **asc-comm 跟随说明（2026-09-01）**：CCU kernel 启动已从 asc-comm 草稿 API
> （`ccu/ccu_launch_kernel.h` + `asccomm_ccu_dataplane`，仅存在于本地仓草稿 commit `bb45144`）
> 迁移到 **asc-comm 正式版**（master `7149d33 add hostlaunch`）：
> - 头文件 `ccu_host_launch.h`（CANN 9.2.0+ 自带于 `asc/include/comm_api/ccu/`，与 asc-comm master 逐字节一致）
> - 库 `asccomm_ccu`（CANN 9.2.0+ 自带 `libasccomm_ccu.so`）
> - 调度描述符 3 字段 → 4 字段（新增 `binaryCacheTag`，传 0），`phyDieMask` 由硬编码 `0x01` 改为
>   `1<<dieId` 动态查询（`HcclRankGraphGetEndpointInfo` 查 `ENDPOINT_ATTR_DIE_ID`）
> - Variable/Event 分配的 DIE ID 由硬编码 `1` 改为同一动态 `dieId`
> - 参考实现：asc-comm `examples/ccu/ccu_direct/02_allgather_add`（同为参数桥双流架构）

## 运行环境要求

- `HCCL_OP_EXPANSION_MODE=CCU_SCHED` 必须设置（`run.sh` 已包含）
- `ASCEND_HOME_PATH` 指向 CANN 安装目录
- 至少 2 张 NPU 卡（2 rank 单进程多线程）

## 与 ccu_all_gather_matmul 的差异速览

| 项 | AG 示例 | 本工程 |
|---|---|---|
| 通信语义 | AllGather（一份源广播，recvBuf 膨胀 R 倍） | AllToAll（逐对换块，recvBuf 等量） |
| 参数桥 | (commTurn+1)×10 槽（含 scale 轮） | commTurn×11 槽（新增 srcStride） |
| CCU kernel src | 每 round 固定（input+inputOffset） | 逐 rank 递推（input+srcOffset+j*srcStride） |
| 远端 dst 落点 | output+outputOffset（轮次决定） | output+myRank*blockSize+r*sub（发送方 rankId 决定，host 预乘烧入 dstOffset） |
| 本地拷贝 | sendBuf+inputOff → recvBuf+outputOff | 独立累加构造 localSrc（本 rank 块）后 GroupCopy（与远端 j 循环互不复用） |
| dtype | INT8（FP8 复用） | FP16 |
| Blaze 依赖 | 需要（ops-tensor submodule） | 无（阶段3 叠 matmul 时恢复） |

## Blaze 恢复点（阶段3 叠加通算融合时）

`CMakeLists.txt` 恢复以下三项（对照 `../ccu_all_gather_matmul/CMakeLists.txt`）：
1. `OPS_TENSOR_ROOT` 检查与 3 个 include 路径（L25-33, L57-60 对应段）
2. 链接库 `platform tiling_api register`
3. `platform_ascendc.h` 相关 include 路径（GetCoreNumAic 查询，替换 `AIC_BLOCK_NUM=8` 常量）

## 已知问题 / 风险

- **AIC 桥 kernel 固定 8 核**（Blaze-free 无法查询 platform 核数），阶段3 恢复查询
- ~~**token 机制**：仅对 sendBuf 获取 token~~（阶段2 已上板验证无问题，4 rank 全 PASS）
- **CCU kernel 性能不可被 msprof --ai-core 采集**（沿用 AG 实测结论），perf 模式仅作烟囱
- **共享服务器必须 qrun 排队**（见上"精度验证必须走 qrun 排队"），裸跑抢占会干扰他人任务；
  **禁止 `qrun clear`**（清除全队列含他人任务，铁律）
