#!/usr/bin/env python3
# ----------------------------------------------------------------------------
# ffts_profile.data 解析脚本（STARS/FFTS 每核 PMU 记录）
#
# 用途:
#   解析 msprof PROF 目录下 device_*/data/ffts_profile.data.*，输出每核 PMU 表。
#   FFTS(Fast Function Task Scheduler) 即 STARS 调度器实现，任务在核上
#   启动/结束时由硬件锁存 SYS_CNT 绝对时刻，是核级执行起止的权威记录:
#
#     pmu_start_cnt: 该核开始执行任务的 SYS_CNT（与 kernel 内 GetSystemCycle 同域）
#     pmu_end_cnt:   该核结束执行任务的 SYS_CNT（覆盖 kernel 末指令之后的
#                    MTE 异步搬运排空段——kernel 内打点无法测到的硬件尾）
#     ai_total_cycle: 该核计算单元(AIV/AIC) cycles
#     各流水 busy cycles: vec/cube/scalar/mte1/mte2/mte3/icache/fixp
#
#   典型用法（配合 parse_stars_log.py + kernel T0/T2 打点做全链路分解）:
#     dispatch -> pmu_start = 核启动延迟
#     T2 -> pmu_end         = 硬件尾(MTE 排空)
#     T2-stars + 硬件尾 == dispatch -> 最慢核结束 (自洽校验闭环)
#
# 用法:
#   python3 parse_ffts_pmu.py -p <PROF目录>/PROF_xxx/device_0/data/
#   python3 parse_ffts_pmu.py -p <data目录> --output pmu.csv
#
# 字段说明（AST_STARS_PMU_FMT, 256 bytes/条）:
#   func_type: 42 = task pmu, 41 = blk pmu
#   core_type: 0 = AIC(Cube), 1 = AIV(Vector)
#   sys cnt 频率: A5 实测 1000 MHz（SoC 全局计数器, AIV/AIC 同频）
# ----------------------------------------------------------------------------
import os
import argparse
import struct
import csv
import statistics

# MsprofAdditonalInfo: 256 byte
AST_STARS_PMU_FMT = "HH" + "HH" + "Q" + "I" * 4 + "Q" * 12
STRUCT_SIZE = struct.calcsize(AST_STARS_PMU_FMT)

TASK_PMU = 42  # task 级记录
BLK_PMU = 41   # block 级记录


class AstPmuStructBean:
    """单条 FFTS PMU 记录（每核一条）"""

    def __init__(self, fields):
        self.func_type = fields[0] & 63
        self.cnt = (fields[0] >> 6) & 15
        self.sqe_type = (fields[0] >> 10) & 63
        self.head = fields[1]
        self.stream_id = fields[2]
        self.task_id = fields[3]
        self.ai_total_cycle = fields[4]
        self.core_type = fields[6] & 1          # 0:AIC, 1:AIV
        self.core_id = (fields[6] >> 8) & 255
        self.block_id = fields[7] >> 16
        self.sub_block_id = fields[7] & 65535
        self.pmu_start_cnt = fields[19]
        self.pmu_end_cnt = fields[20]

    @property
    def core_name(self):
        return "AIV" if self.core_type else "AIC"

    def to_row(self):
        return [
            self.func_type, self.cnt, self.task_id, self.stream_id,
            self.core_name, self.core_id, self.block_id, self.sub_block_id,
            self.ai_total_cycle, self.pmu_start_cnt, self.pmu_end_cnt,
        ]


def parse(file_path: str) -> list:
    bean_list = []
    for name in sorted(os.listdir(file_path)):
        # 只解析 ffts_profile.data.*，过滤 .done 等标记文件
        if "ffts_profile.data" not in name or name.endswith(".done"):
            continue
        file_name = os.path.join(file_path, name)
        file_size = os.path.getsize(file_name)
        if file_size == 0:
            continue
        n_records = file_size // STRUCT_SIZE
        if n_records == 0:
            print(f"[WARN] {name}: 大小 {file_size}B 不是 {STRUCT_SIZE}B 整数倍，跳过")
            continue
        print(f"[INFO] 解析 {name}: {n_records} 条记录")
        with open(file_name, "rb") as fd:
            context = fd.read()
            for index in range(n_records):
                data = context[index * STRUCT_SIZE:(index + 1) * STRUCT_SIZE]
                bean_list.append(AstPmuStructBean(struct.unpack(AST_STARS_PMU_FMT, data)))
    return bean_list


def main():
    parser = argparse.ArgumentParser(description="解析 msprof ffts_profile.data（STARS 每核 PMU）")
    parser.add_argument("-p", "--path", type=str, required=True,
                        help="要解析的 data 目录（PROF_xxx/device_N/data/）")
    parser.add_argument("--output", default="ffts_pmu.csv",
                        help="输出 CSV 路径（默认 ./ffts_pmu.csv）")
    parser.add_argument("--freq", type=float, default=1000.0,
                        help="SYS_CNT 频率 MHz（默认 1000，A5 实测；A3 需上板标定）")
    args = parser.parse_args()

    print(f"start parse: {args.path}")
    bean_list = parse(args.path)
    print(f"end parse: {len(bean_list)} 条记录")

    header = ["func type", "cnt", "task id", "stream id", "core type", "core id",
              "block id", "sub block id", "ai total cycle", "pmu start cnt", "pmu end cnt"]
    with open(args.output, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)
        for bean in bean_list:
            writer.writerow(bean.to_row())
    print(f"CSV 输出: {args.output}")

    # 简要统计: 单核执行时长 (pmu_end - pmu_start)
    task_pmu = [b for b in bean_list if b.func_type == TASK_PMU]
    durs = [b.pmu_end_cnt - b.pmu_start_cnt for b in task_pmu if b.pmu_end_cnt > b.pmu_start_cnt > 0]
    if durs:
        print(f"per-core exec (pmu_end - pmu_start): task_pmu={len(task_pmu)} 条, 有效={len(durs)}")
        print(f"  median={statistics.median(durs):.0f} cyc ({statistics.median(durs)/args.freq:.3f} us)  "
              f"min={min(durs)}  max={max(durs)}")
    ends = [b.pmu_end_cnt for b in task_pmu if b.pmu_end_cnt > 0]
    starts = [b.pmu_start_cnt for b in task_pmu if b.pmu_start_cnt > 0]
    if ends:
        print(f"pmu_end max = {max(ends)} (末拍最慢核结束时刻, 用于 dispatch->核结束 校验)")


if __name__ == "__main__":
    main()
