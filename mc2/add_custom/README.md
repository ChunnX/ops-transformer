# Add Custom — aclrtHostRegister V2 零拷贝穿刺验证

## 概述

验证 `aclrtHostRegisterV2` + `aclrtHostGetDevicePointer` 实现 Host↔Device 零拷贝信号通路。

```
[Host] aclrtMallocHost → aclrtHostRegisterV2(MAPPED) → aclrtHostGetDevicePointer → syncHost/syncDev
[OP1] add_custom_with_sync: x+y→bufA, *(uint32_t*)syncDev = 32768
[Host] while(*syncHost == 0) {}  ← 零拷贝自旋
[Host] malloc(op2Bytes) → OP2 动态分配 → 执行
[Host] 精度验证
```

## 目录

```
add_custom/
├── add_custom.asc          # kernel + main (183 行)
├── data_utils.h            # 文件读写
├── CMakeLists.txt           # dav-2201 / dav-3510
├── run.sh                   # 一键运行
├── README.md
├── A5_RESULT.md             # A3/A5 对比报告
└── scripts/
    ├── gen_data.py          # [8,2048] float16
    └── verify_result.py     # 精度比对
```

## 运行

```bash
source ${CANN_PATH}/set_env.sh
cd ops-transformer/mc2/add_custom

# A3
rm -rf build && ./run.sh dav-2201

# A5
rm -rf build && ./run.sh
```

## 关键输出

```
[INFO] syncHost=0x12c0014e1000  syncDev=0x3fff8200000  ← V2 两步拿到
[INFO] Launch OP1 ...
[INFO] Polling syncHost (zero-copy) ...
[INFO] OP1 done, size=32768 bytes
[INFO] Launch OP2 ...
[INFO] Done.
error ratio: 0.0000
```

## 核心 API

| 步骤 | API | 说明 |
|------|-----|------|
| 申请 host-pinned | `aclrtMallocHost` | 必须 MallocHost, malloc 不行 |
| 映射+拿dev地址 | `aclrtHostRegisterV2` + `aclrtHostGetDevicePointer` | V2: 注册与取地址分离 |
| kernel 写 | `*(volatile __gm__ uint32_t*)syncDev = val` | block0 写入 |
| Host 读 | `while(*syncHost == 0){}` | 零拷贝, 无 memcpy |
| 解注册 | `aclrtHostUnregister` | 必须先 Unregister 再 FreeHost |

## 平台结果

| | A3 (910_93 + CANN 9.1.0) | A5 (950PR) |
|------|:---:|:---:|
| V2 RegisterV2(MAPPED) | ✅ | ✅ |
| GetDevicePointer | ✅ | ✅ |
| kernel 写 mapped | ✅ | ✅ |

## 规格

| 参数 | 值 |
|------|-----|
| shape | [8, 2048] |
| dtype | float16 |
| numBlocks | 8 |
| tileNum × buffer | 8 × 2 |
| sync 值 | 32768 (8×2048×2) |
