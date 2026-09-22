# prep_cache 数据生成基准报告 (M2+M3, V2/V3/V8)

- 生成时间: 2026-09-22 08:23:15
- 数据规格: dtype=fp8_e4m3fn, use_fp8=1, sf_dtype=e8m0, world_size=4 (total%4 为 sliced 整除性, 仅报告)

## 规格结果

| 规格 | total | hidden | 存储GB | sfGB | 冷生成s | 命中s | load s | 抽样校验 | total%4 |
|---|---|---|---|---|---|---|---|---|---|
| functional | 1024 | 576 | 0.00 | 0.00 | hit(0.00) | 0.00 | 0.00 | PASS | 0 |
| vocab_small | 5138000 | 576 | 2.76 | 0.09 | 17.13 | 0.00 | 0.00 | PASS | 0 |
| vocab_medium | 10276000 | 576 | 5.51 | 0.17 | 36.37 | 0.00 | 0.00 | PASS | 0 |
| vocab_medium2 | 20552000 | 576 | 11.02 | 0.34 | 72.54 | 0.00 | 0.00 | PASS | 0 |
| vocab_large1 | 41104000 | 576 | 22.05 | 0.69 | 151.54 | 0.00 | 0.00 | PASS | 0 |
| vocab_large2 | 82208000 | 576 | 44.10 | 1.38 | 281.90 | 0.00 | 0.00 | PASS | 0 |
| tensor_1x | 5138000 | 576 | 2.76 | 0.09 | hit(0.00) | 0.00 | 0.00 | PASS | 0 |
| tensor_4x | 12845000 | 2304 | 27.56 | 0.86 | 176.64 | 0.00 | 0.00 | PASS | 0 |
| tensor_16x | 321128 | 9216 | 2.76 | 0.09 | 17.60 | 0.00 | 0.00 | PASS | 0 |

## 环境

- host: ascend195
- cpu_count: 256
- cache_dir: /home/qianlong/workspace/cann/ops-transformer.worktrees/engram-local/examples/mc2/engram_test/prep_cache (文件系统: ext4, 可用 755.45 GiB / 共 7096.47 GiB)
- /dev/shm 占用: 运行前 0.8 MiB, 运行后 0.8 MiB (差值 +0.0 MiB, 应基本不变)
- 峰值 RSS (ru_maxrss, RUSAGE_SELF): 1.69 GB

## 结论

- 冷生成: 7 次未命中, 总耗时 753.72s, 写入总量 119.38 GiB, 平均生成速度 0.16 GiB/s
- 缓存复用: 2 个规格首次 generate 即命中; tensor_1x 与 vocab_small 同 key (total/hidden/seed 相同), 已验证复用命中
- /dev/shm 前后占用基本不变 (未使用共享内存); 峰值 RSS 1.69 GB (分块生成, 无整表 fp32 中间)

ALL SPECS PASS
