#!/usr/bin/env python3
# ----------------------------------------------------------------------------
# stars_soc.data 解析脚本（STARS 硬件任务调度日志）
#
# 用途:
#   解析 msprof PROF 目录下 device_*/data/stars_soc.data.*，输出调度事件表。
#   STARS 每个 task 派发/完成时向该文件写入一条记录（含 SYS_CNT 绝对时刻），
#   与 kernel 内 AscendC::GetSystemCycle() 读的是同一个计数器，可直接相减：
#
#     头开销 = T1(kernel 内 GetSystemCycle) - STARS_dispatch(func_type=0 的 sys cnt)
#
# 用法:
#   python3 parse_stars_log.py -p <PROF目录>/PROF_xxx/device_0/data/
#   python3 parse_stars_log.py -p <data目录> --output stars.csv
#
# 字段说明（AST_STARS_LOG_FMT, 28 bytes/条）:
#   func_type: 0 = STARS dispatch（task 派发）, 1 = STARS complete（task 完成回执）
#   cnt:       全局计数器（dispatch/complete 成对递增）
#   head:      固定 head 标识（如 27603）
#   task_id / stream_id: 任务与流编号
#   sys cnt:   SYS_CNT 寄存器绝对值（A5 实测 1000MHz, 1 cyc = 1ns）
# ----------------------------------------------------------------------------
import os
import argparse
import struct
import csv

# MsprofAdditonalInfo: 256 byte; 本工具只解析前 28 字节有效字段
AST_STARS_LOG_FMT = "I" + "HH" + "Q" + "I" * 4
STRUCT_SIZE = struct.calcsize(AST_STARS_LOG_FMT)


class AstLogStructBean:
    """单条 STARS 调度日志记录"""

    def __init__(self, fields):
        self.func_type = fields[0] & 63
        self.cnt = (fields[0] >> 6) & 15
        self.head = (fields[0] >> 16) & 65535
        self.stream_id = fields[1]
        self.task_id = fields[2]
        self.sys_cycle = fields[3]

    def to_row(self):
        return [self.func_type, self.cnt, self.head, self.task_id, self.stream_id, self.sys_cycle]


def parse(file_path: str) -> list:
    bean_list = []
    for name in sorted(os.listdir(file_path)):
        # 只解析 stars_soc.data.*，过滤 .done 等标记文件（空文件/非 28 字节整数倍）
        if "stars_soc.data" not in name or name.endswith(".done"):
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
                bean_list.append(AstLogStructBean(struct.unpack(AST_STARS_LOG_FMT, data)))
    return bean_list


def main():
    parser = argparse.ArgumentParser(description="解析 msprof stars_soc.data STARS 调度日志")
    parser.add_argument("-p", "--path", type=str, required=True,
                        help="要解析的 data 目录（PROF_xxx/device_N/data/）")
    parser.add_argument("--output", default="stars_log.csv",
                        help="输出 CSV 路径（默认 ./stars_log.csv）")
    parser.add_argument("--freq", type=float, default=1000.0,
                        help="SYS_CNT 频率 MHz（默认 1000，A5 实测；A3 需上板标定）")
    args = parser.parse_args()

    print(f"start parse: {args.path}")
    bean_list = parse(args.path)
    print(f"end parse: {len(bean_list)} 条记录")

    dispatches = [b.sys_cycle for b in bean_list if b.func_type == 0]
    completes = [b.sys_cycle for b in bean_list if b.func_type == 1]
    print(f"dispatch={len(dispatches)}  complete={len(completes)}")

    header = ["func type", "cnt", "head", "task id", "stream id", "sys cnt"]
    with open(args.output, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)
        for bean in bean_list:
            writer.writerow(bean.to_row())
    print(f"CSV 输出: {args.output}")

    # 简要统计: 成对 dispatch/complete 的 task 跨度
    if dispatches and completes:
        pairs = sorted(dispatches + completes)
        if len(dispatches) == len(completes):
            spans = [c - d for d, c in zip(sorted(dispatches), sorted(completes)) if c >= d]
            if spans:
                import statistics
                print(f"task span (dispatch->complete): mean={statistics.mean(spans):.0f} cyc "
                      f"({statistics.mean(spans)/args.freq*1000:.3f} us)  min={min(spans)}  max={max(spans)}")


if __name__ == "__main__":
    main()
