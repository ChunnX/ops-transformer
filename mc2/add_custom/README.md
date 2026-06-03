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
[INFO] syncHost=0x12c0014e1000  syncDev=0x3fff8200000
       ↑ MallocHost+RegisterV2(MAPPED)+GetDevicePointer 三步完成
       Host/Device 两侧地址不同, 指向同一物理页
[INFO] Launch OP1 ...
[INFO] Polling syncHost (zero-copy) ...
       ↑ while(*syncHost==0){}, 等 kernel 写入, 无 memcpy
[INFO] OP1 done, size=32768 bytes
       ↑ 8 × 2048 × 2B(half) = 32768 ✓
[INFO] RegisterV2(bufB PINNED) host=... size=32768
       ↑ 按 OP1 返回的 size 动态 malloc, RegisterV2(PINNED) 锁页
[INFO] Launch OP2 ...
[INFO] Done.

>>> Verifying output/output_op1.bin vs output/golden.bin
error ratio: 0.0000      ← OP1 精度通过
>>> Verifying output/output_op2.bin vs output/golden.bin
error ratio: 0.0000      ← OP2 精度通过
all test pass!
```

## 平台结果

| 平台 | RegisterV2(MAPPED) | kernel 写 mapped | Host 零拷贝轮询 | 结论 |
|------|:---:|:---:|:---:|------|
| A3 (910_93 + CANN 9.1.0) | ✅ | ✅ | ✅ `while(*syncHost==0){}` | **通过** |
| A5 (950PR + CANN 9.x) | ✅ | ❌ 挂死 | ❌ 死等 | **不支持** |

## RegisterV2 使用

| 位置 | 申请方式 | Flag | 目的 |
|------|---------|------|------|
| sync 通道 | `aclrtMallocHost` → `RegisterV2` → `GetDevicePointer` | `MAPPED` | kernel 直写 host-pinned 内存, Host 零拷贝读 |
| OP2 输出 bufB | `malloc` → `RegisterV2` | `PINNED` | 锁定物理页, 防 OS swap |

## 关键发现

- 必须使用 `aclrtMallocHost` 申请内存才能 `RegisterV2(MAPPED)` 成功；普通 `malloc` 返回 `507899`
- 910 系列零拷贝通路完整可用
- 950PR RegisterV2 API 返回成功但 kernel 写 mapped 内存挂死

## 规格

| 参数 | 值 |
|------|-----|
| shape | [8, 2048] |
| dtype | float16 (half) |
| numBlocks | 8 |
| 双缓冲 | BUFFER_NUM=2, tileNum=8 |
| 精度容忍 | rtol=1e-3, atol=1e-3 |
