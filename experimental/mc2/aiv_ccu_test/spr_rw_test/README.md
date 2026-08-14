# SPR 读写测试

## 样例介绍

本样例实现 `background.md` 中描述的 **SPR→vaddr 映射访问例子**，验证 AIV 核通过 `halResMap` 映射的 vaddr 访问 SPR（Special Purpose Register）的能力。

### 功能点

1. Host 侧通过 `halResMap` 映射所有 AICORE 的 SPR vaddr（含 AIV）
2. 打印获取的 SPR 地址
3. 将 vaddr 表拷贝到 GM
4. AIV kernel 通过 `st_dev` 写 SPR、`get_data_main_base()` 读 SPR（先写后读）
5. 验证 SPR 寄存器可写可读

### 目录结构

```text
├── CMakeLists.txt                      # 编译配置
├── kernel/
│   └── spr_rw_kernel.h                 # AIV 侧 SPR 读写 kernel 核心逻辑
├── spr_rw.cpp                          # Kernel 入口 + <<<>>> 封装
├── spr_rw.h                            # 对外接口
├── test_spr_rw.cpp                     # Host 侧测试代码 (halResMap + <<<>>> 调用)
├── run_spr_rw.sh                       # 编译运行脚本
└── README.md
```

## 一、环境准备

- Ascend 950PR / 950DT
- CANN Toolkit (与 local_ms_rw_test 相同版本)
- 设置环境变量：`source /usr/local/Ascend/cann/set_env.sh`
- 设置 `RANK_TABLE_FILE` 环境变量指向 rank table 文件

## 二、编译与运行

### 编译并运行

```bash
bash run_spr_rw.sh -v ascend950pr_9599 -b Release -c ON -r ON -t 18 -n 36 -a 0 -d 1
```

### 参数说明

| 参数 | 含义 | 默认值 |
|------|------|--------|
| `-v` | 芯片版本 | ascend950pr_9599 |
| `-b` | 编译类型 Release/Debug | Debug |
| `-c` | 是否重新编译 ON/OFF | OFF |
| `-r` | 是否运行 ON/OFF | OFF |
| `-t` | 目标核 res_id | 18 (DIE0 AIV0) |
| `-n` | halResMap 映射核总数 | 36 |
| `-a` | rank_id | 0 |
| `-d` | ranksize | 1 |

## 三、原理说明

### 3.1 SPR→vaddr 映射 (background.md 2.1-2.2)

```
Host:  halResMap(devId, {PROCESS_CP1, RES_AICORE, res_id}, &va, &len)
         → 返回该核 SPR 寄存器区域的 vaddr (mmap 设备文件)
         → 内核页表建立 VA → 设备总线地址映射

AIV:   st_dev(value, (__gm__ uint32_t*)(sprVaddr + 0x00D0), 0)   // 写 SPR
       get_data_main_base()                                       // 读本核 SPR (cce 指令)
```

### 3.2 访问流程 (background.md 2.4)

```cpp
// 1. 解锁寄存器
st_dev(0x19101920, (__gm__ uint32_t*)(sprVaddr + 0x4078), 0);
// 2. CrossCore 同步 (提高成功率)
CrossCoreSetFlag<0, PIPE_ALL>(11);
CrossCoreWaitFlag<0, PIPE_ALL>(11);
// 3. 写 SPR
st_dev(testVal, (__gm__ uint32_t*)(sprVaddr + 0x00D0), 0);
// 4. 读 SPR
uint64_t val = get_data_main_base();
```

### 3.3 已知限制 (background.md 2.4)

- 部分核（11/17/20/32）写不了，默认用 res_id=18 (DIE0 AIV0)
- 读前解锁序列不一定必需，但加上可提高成功率
- `get_data_main_base()` 读取的是 DATA_MAIN_BASE 寄存器，`st_dev` 写的是 +0x00D0 偏移子寄存器，两者可能不是同一字段

## 四、输出示例

```
[INFO] ===== SPR 读写测试 =====
[INFO] rank_id=0, ranksize=1, target_core_id=18, res_id_count=36
[INFO] 通信域初始化成功。

[INFO] ===== halResMap 映射 SPR vaddr =====
[INFO] halResMap res_id=0 -> va=0x..., len=...
...
[INFO] halResMap res_id=18 -> va=0x4a01100000, len=...

[INFO] 目标核 res_id=18:
[INFO]   SPR base vaddr  = 0x4a01100000
[INFO]   SPR +0x00D0     = 0x4a011000d0 (DATA_MAIN_BASE 访问地址)

[INFO] ===== 启动 Kernel (blockDim=1, targetCoreId=18) =====
[INFO] Kernel 执行完成。

===== SPR 访问结果 =====
  目标核 res_id:      18
  sprVaddr (回显):    0x4a011000d0
  写前读值:           0x00000000xxxxxxxx
  写后读值:           0x00000000yyyyyyyy
  verifyResult:       0 (0=通过/有变化, 1=失败, 2=skip)
  aivCoreId:          0
========================

[INFO] SPR 访问验证: 通过 (写后读值与写前不同，SPR 寄存器可写可读)
```
