# Add Custom — MoeEpDispatch ping-buffer 穿刺验证

## 概述

验证 MoeEpDispatch 方案中最关键的 **Host↔Device 零拷贝 Ping-Buffer** 机制：

```
[Host] aclrtMallocHost → RegisterV2(MAPPED) → syncHost (host-pinned, device-mapped)
  ↓
[OP1] add_custom_with_sync: kernel 计算完成后将 output size 写入 syncHost (零拷贝直写)
  ↓
[Host] while(*syncHost == 0) {}  ← 零拷贝自旋等待 (无需 aclrtMemcpy D2H)
  ↓
[Host] 读到 size → 动态 malloc → 分配 OP2 缓冲区
  ↓
[OP2] add_custom: 使用动态分配的 buffer 完成计算
  ↓
[Host] 验证 OP1 / OP2 输出精度
```

## 对应 dispatch 方案

| 本例 | 真实 dispatch |
|------|---------------|
| `add_custom_with_sync` 写 sync | Dispatch K1 写 host-pinned actualA |
| `while(*syncHost==0){}` | Host spin-wait `num_recv_tokens` |
| `malloc(op2BufSize)` + `aclrtMalloc(bufDevB)` | `torch.empty({actualA, hidden})` |
| `add_custom` → bufDevB | Epilogue K2: Window → recv_x 紧凑拷贝 |

## 目录结构

```
add_custom/
├── add_custom.asc          # kernel 实现 + main (两段式 + 零拷贝轮询)
├── data_utils.h            # 文件读写工具 (ReadFile / WriteFile)
├── CMakeLists.txt           # 编译工程 (dav-2201 / dav-3510)
├── run.sh                   # 一键编译运行脚本
├── README.md                # 本文档
└── scripts/
    ├── gen_data.py          # 生成随机输入 + golden (shape [8,2048], float16)
    └── verify_result.py     # 精度比对 (rtol=1e-3, atol=1e-3)
```

## 运行

### 前置条件

- CANN ≥ 9.0.0
- NPU: 910B / 910C / 910_93 (dav-2201) 或 950PR / 950DT / 910PR (dav-3510)

### 配置环境

```bash
source ${CANN_PATH}/set_env.sh
```

### 一键运行

```bash
cd ops-transformer/mc2/add_custom

# 910 系列
./run.sh dav-2201

# 950 系列 (默认)
./run.sh

# CPU 调试模式 (无需 NPU)
mkdir -p build && cd build
cmake -DCMAKE_ASC_RUN_MODE=cpu -DCMAKE_ASC_ARCHITECTURES=dav-2201 .. && make -j
```

### 分步执行

```bash
# 1. 生成数据
python3 scripts/gen_data.py

# 2. 编译
mkdir -p build && cd build
cmake -DCMAKE_ASC_ARCHITECTURES=dav-2201 .. && make -j

# 3. 执行
./demo

# 4. 验证
cd ..
python3 scripts/verify_result.py \
    output/output_op1.bin output/golden.bin \
    output/output_op2.bin output/golden.bin
```

## 关键输出解读

```
[INFO] aclrtMallocHost(sync) ret=0 ptr=0x...     ← ① host-pinned 申请成功
[INFO] RegisterV2(MAPPED) ret=0 (0=success)       ← ② 零拷贝映射成功
[INFO] GetDevicePointer ret=0 syncDev=0x...        ← ③ device 端地址已获取
[INFO] Final: useZeroCopySync=1                    ← ④ 零拷贝通道已就绪
[INFO] === Launching OP1 ===
[INFO] Host polling sync (before SynchronizeStream) ...  ← ⑤ 先 ping 再 sync
[INFO] OP1 done, N polls, reported size = 32768 bytes     ← ⑥ N>0 说明自旋等待
[INFO] bufHostB=... bufDevB=... (dynamic size=32768)      ← ⑦ 动态分配成功
[INFO] === Running OP2 ===
[INFO] Done.

>>> Verifying output/output_op1.bin vs output/golden.bin
error ratio: 0.0000      ← ⑧ OP1 精度通过
>>> Verifying output/output_op2.bin vs output/golden.bin
error ratio: 0.0000      ← ⑨ OP2 精度通过
all test pass!
```

## 关键发现

| 内存申请方式 | RegisterV2(MAPPED) | GetDevicePointer |
|-------------|:---:|:---:|
| 普通 `malloc` | ❌ ret=507899 | nil |
| `aclrtMallocHost` | ✅ ret=0 | 有效地址 |

**910_93 + CANN 9.1.0** 上必须用 `aclrtMallocHost` 申请 host-pinned 内存，才能成功 `RegisterV2(MAPPED)` 并拿到 device 端可用地址。

## 规格

| 参数 | 值 |
|------|-----|
| shape | [8, 2048] |
| dtype | float16 (half) |
| numBlocks | 8 |
| 双缓冲 | BUFFER_NUM=2, tileNum=8 |
| 精度容忍 | rtol=1e-3, atol=1e-3 |
