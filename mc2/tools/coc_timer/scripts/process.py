import sys
import pandas as pd
import json
from pathlib import Path
import re

def parse_event_name(column):
    """从列名解析事件名称"""
    for suffix in ['_start(us)', '_end(us)', '_duration(us)', '_start_cycle', '_end_cycle', '_duration_us']:
        if column.endswith(suffix):
            return column[:-len(suffix)]
    return None

def is_cycle_column(col):
    """判断是否为 cycle 列"""
    return '_start_cycle' in col or '_end_cycle' in col

def detect_pipeline_mode(df):
    """根据时间戳自动检测 pipeline 模式：aic_first 还是 aiv_first"""
    aic_starts = []
    aiv_starts = []

    for col in df.columns:
        if '_start(us)' in col or '_start_cycle' in col:
            event_name = parse_event_name(col)
            if event_name is None:
                continue
            parts = event_name.rsplit('_', 1)
            if len(parts) != 2:
                continue
            try:
                iter_num = int(parts[1])
            except ValueError:
                continue

            if parts[0] == 'AIC':
                for val in df[col]:
                    if pd.notna(val) and val > 0:
                        aic_starts.append((iter_num, val))
            elif parts[0] == 'AIV':
                for val in df[col]:
                    if pd.notna(val) and val > 0:
                        aiv_starts.append((iter_num, val))

    if not aic_starts or not aiv_starts:
        return "aic_first"

    aic_by_iter = {}
    for iter_num, start in aic_starts:
        if iter_num not in aic_by_iter:
            aic_by_iter[iter_num] = []
        aic_by_iter[iter_num].append(start)

    aiv_by_iter = {}
    for iter_num, start in aiv_starts:
        if iter_num not in aiv_by_iter:
            aiv_by_iter[iter_num] = []
        aiv_by_iter[iter_num].append(start)

    aic_first_count = 0
    aiv_first_count = 0

    common_iters = set(aic_by_iter.keys()) & set(aiv_by_iter.keys())
    for iter_num in common_iters:
        aic_avg = sum(aic_by_iter[iter_num]) / len(aic_by_iter[iter_num])
        aiv_avg = sum(aiv_by_iter[iter_num]) / len(aiv_by_iter[iter_num])
        if aic_avg < aiv_avg:
            aic_first_count += 1
        else:
            aiv_first_count += 1

    return "aic_first" if aic_first_count >= aiv_first_count else "aiv_first"

def collect_events_from_row(row, group_id):
    """从一行数据中收集所有事件及其时间信息（严格屏蔽干扰事件）"""
    events = []
    core_type = row['core_type'].strip() if isinstance(row['core_type'], str) else row['core_type']
    sub_id = int(row['sub_group_id'])

    for col in row.index:
        if '_start(us)' in col or '_start_cycle' in col:
            event_name = parse_event_name(col)
            if event_name is None:
                continue
            
            if not (event_name.startswith('AIC') or event_name.startswith('AIV')):
                continue

            # 构建 end_col 和 duration_col
            if '_start_cycle' in col:
                end_col = col.replace('_start_cycle', '_end_cycle')
                duration_col = col.replace('_start_cycle', '_duration_us')
            else:
                end_col = col.replace('_start(us)', '_end(us)')
                duration_col = col.replace('_start(us)', '_duration(us)')

            start_time = float(row[col]) if pd.notna(row[col]) else 0.0
            end_time = float(row[end_col]) if end_col in row.index and pd.notna(row[end_col]) else 0.0
            duration = float(row[duration_col]) if duration_col in row.index and pd.notna(row[duration_col]) else 0.0

            # 如果是 cycle 列，需要转换为 us
            if is_cycle_column(col):
                if start_time > 0:
                    start_time = start_time / 50.0
                if end_time > 0:
                    end_time = end_time / 50.0
                # duration 为 0 时用 end-start 重新计算
                if duration == 0.0 and start_time > 0 and end_time > start_time:
                    duration = end_time - start_time

            if start_time > 0 or end_time > 0:
                events.append({
                    'name': event_name,
                    'start': start_time,
                    'end': end_time,
                    'duration': duration,
                    'core_type': core_type,
                    'sub_id': sub_id,
                    'group_id': group_id
                })

    return events

def generate_iron_law_timeline_json(csv_path, output_json_path, op_name="unknown_op", pipeline_type="auto"):
    """基于时间戳和流水线强依赖（铁律）生成对齐的时间线"""
    df = pd.read_csv(csv_path)

    df.columns = df.columns.str.strip()
    if 'core_type' in df.columns:
        df['core_type'] = df['core_type'].str.strip()

    detected_mode = detect_pipeline_mode(df)
    if pipeline_type == "auto":
        pipeline_type = detected_mode
    print(f"  [Pipeline] 确定流水线模式: {pipeline_type}")

    final_json = []
    all_events = []
    grouped = df.groupby('group_id')

    for group_id, group_df in grouped:
        group_events = []
        for _, row in group_df.iterrows():
            group_events.extend(collect_events_from_row(row, group_id))

        step_events = {}
        for e in group_events:
            parts = e['name'].rsplit('_', 1)
            if len(parts) != 2 or not parts[1].isdigit():
                continue
            iter_num = int(parts[1])

            if pipeline_type == "aic_first":
                if e['name'].startswith('AIC'):
                    step = iter_num       # AIC_0 属于 Step 0, AIC_1 属于 Step 1
                else: 
                    step = iter_num + 1   # AIV_0 属于 Step 1
            else: # aiv_first
                if e['name'].startswith('AIV'):
                    step = iter_num
                else: 
                    step = iter_num + 1

            if step not in step_events:
                step_events[step] = []
            step_events[step].append(e)

        current_barrier_time = 0.0
        max_step = max(step_events.keys()) if step_events else -1

        for step in range(max_step + 1):
            if step not in step_events:
                continue
            
            events_in_step = step_events[step]
            
            # 按执行轨道(tid)把当前批次的事件分组
            events_by_tid = {}
            for e in events_in_step:
                tid = f"group_{int(e['group_id'])}_{e['core_type']}_{e['sub_id']}"
                if tid not in events_by_tid:
                    events_by_tid[tid] = []
                events_by_tid[tid].append(e)

            step_max_end = current_barrier_time # 初始化本批次的“木桶短板”

            # 遍历每个轨道 (比如 AIC核轨道, AIV核0轨道, AIV核1轨道)
            for tid, tid_events in events_by_tid.items():
                tid_min_raw_start = min(e['start'] for e in tid_events)
                
                for e in tid_events:
                    # 计算相对轨道起点的偏移（比如 AIV_RS_0 相对于 AIV_0 晚了多少）
                    local_offset = e['start'] - tid_min_raw_start
                    
                    # 铁律生效：主事件的 local_offset 为 0，完美卡在 current_barrier_time 起跑！
                    ideal_start = current_barrier_time + local_offset
                    ideal_end = ideal_start + e['duration']

                    final_json.append({
                        "ph": "X",
                        "cat": f"{e['core_type'].lower()}_event",
                        "pid": f"rank_{op_name}",
                        "tid": tid,
                        "name": e['name'],
                        "ts": round(ideal_start, 6),
                        "dur": round(e['duration'], 6),
                        "cname": "good" if e['name'].startswith('AIC') else "thread_state_uninterruptible" 
                    })

                    # 更新所有轨道里最晚结束的时间
                    step_max_end = max(step_max_end, ideal_end)

            current_barrier_time = step_max_end

    print(f"  生成 {len(final_json)} 个严格对齐的流水线事件")
    if len(final_json) == 0:
        print("  [WARNING] 未收集到任何流水线事件！")
        
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(final_json, f, indent=2, ensure_ascii=False, allow_nan=False)
    print(f"  输出: {output_json_path}\n")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        base_dir = Path(sys.argv[1])
    else:
        base_dir = Path("./output_timer")

    if not base_dir.exists():
        print(f"Directory {base_dir} does not exist.")
        sys.exit(1)

    CURRENT_OP_NAME = "UnknownOp"
    PIPELINE_MODE = "auto"

    if len(sys.argv) > 2:
        CURRENT_OP_NAME = sys.argv[2]
        # 根据算子名称静态配置优先级，未命中的走 'auto' 根据真实时间戳推断
        PIPELINE_MAP = {
            "agmm": "aiv_first", "agmmwg": "aiv_first", "agmmrdma": "aiv_first",
            "agmmdq": "aiv_first", "agmmdqbs": "aiv_first", "atagmm": "aiv_first",
            "mmar": "aic_first", "mmrs": "aic_first", "gmmata": "aic_first"
        }
        PIPELINE_MODE = PIPELINE_MAP.get(CURRENT_OP_NAME, "auto")

    data_dir = base_dir / "data"
    processed_count = 0
    if data_dir.exists():
        print(f"=== 开始处理 {CURRENT_OP_NAME} (流水线模式: {PIPELINE_MODE}) ===")
        for csv_path in data_dir.glob("timer_*_timeline.csv"):
            match = re.search(r'(rank_\d+)', csv_path.stem)
            # 生成干净漂亮的输出文件名
            json_name = f"{match.group(1) if match else csv_path.stem}_{CURRENT_OP_NAME}_timeline.json"
            json_path = base_dir / json_name
            
            print(f"-> 正在处理: {csv_path.name}")
            # 调用带铁律同步机制的时间线生成器
            generate_iron_law_timeline_json(
                csv_path=str(csv_path), 
                output_json_path=str(json_path), 
                op_name=CURRENT_OP_NAME, 
                pipeline_type=PIPELINE_MODE
            )
            processed_count += 1
            
    if processed_count == 0:
        print("[WARNING] 没有在 data/ 目录下找到 timer_*_timeline.csv 文件。请检查上游输出。")
    else:
        print("[Done] 所有 Timeline JSON 生成完毕！请导入 chrome://tracing 查看流水线排布。")