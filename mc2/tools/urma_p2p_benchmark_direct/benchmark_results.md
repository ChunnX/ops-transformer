# URMA P2P Benchmark 数据整理

## 字段说明

| 字段 | 说明 |
|------|------|
| benchMode | bench_mode 参数值: batch_write = 攒一批 Write 后统一 Commit + Drain; write_drain_loop = 每次 Write 后立即 Drain |
| sqe_bytes | 单次 Write 传输的数据量 (字节) |
| loop | loop_count 参数值, 即 Write 次数 |
| assemble | avg 单次 Write 时间 (组装 SQE + DataCopy 到 SQ ring) |
| commit/issue | benchMode=batch_write 时为 commit (敲一次 DoorBell); benchMode=write_drain_loop 时为 issue (Write 内含 commit, = assemble) |
| drain | avg 单次 Drain 时间 (poll CQ 等硬件完成 + 更新 PICI) |
| total | avg 单次总时间 (assemble + commit/issue + drain) |
| bw | avg 平均带宽 |
| verify | 数据正确性验证结果 (PASS/FAIL), 验证 sender 数据是否被 receiver 正确接收 |

## 表1: `-m 2` (hcomm)

| benchMode | sqe_bytes | loop | assemble(us) | commit/issue(us) | drain(us) | total(us) | bw(GB/s) | verify |
|-----------|-----------|------|--------------|------------------|-----------|-----------|----------|--------|
| batch_write | 32 | 1 | | | | | | |
| batch_write | 32 | 10 | | | | | | |
| batch_write | 32 | 100 | | | | | | |
| batch_write | 32 | 500 | | | | | | |
| batch_write | 65535 | 1 | | | | | | |
| batch_write | 65535 | 10 | | | | | | |
| batch_write | 65535 | 100 | | | | | | |
| batch_write | 65535 | 500 | | | | | | |
| write_drain_loop | 32 | 1 | | | | | | |
| write_drain_loop | 32 | 10 | | | | | | |
| write_drain_loop | 32 | 100 | | | | | | |
| write_drain_loop | 32 | 500 | | | | | | |
| write_drain_loop | 65535 | 1 | | | | | | |
| write_drain_loop | 65535 | 10 | | | | | | |
| write_drain_loop | 65535 | 100 | | | | | | |
| write_drain_loop | 65535 | 500 | | | | | | |

## 表2: `-m 3` (mc2_native)

| benchMode | sqe_bytes | loop | assemble(us) | commit/issue(us) | drain(us) | total(us) | bw(GB/s) | verify |
|-----------|-----------|------|--------------|------------------|-----------|-----------|----------|--------|
| batch_write | 32 | 1 | | | | | | |
| batch_write | 32 | 10 | | | | | | |
| batch_write | 32 | 100 | | | | | | |
| batch_write | 32 | 500 | | | | | | |
| batch_write | 65535 | 1 | | | | | | |
| batch_write | 65535 | 10 | | | | | | |
| batch_write | 65535 | 100 | | | | | | |
| batch_write | 65535 | 500 | | | | | | |
| write_drain_loop | 32 | 1 | | | | | | |
| write_drain_loop | 32 | 10 | | | | | | |
| write_drain_loop | 32 | 100 | | | | | | |
| write_drain_loop | 32 | 500 | | | | | | |
| write_drain_loop | 65535 | 1 | | | | | | |
| write_drain_loop | 65535 | 10 | | | | | | |
| write_drain_loop | 65535 | 100 | | | | | | |
| write_drain_loop | 65535 | 500 | | | | | | |
