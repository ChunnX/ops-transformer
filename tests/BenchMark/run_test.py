import re
import os
import sys
import csv
import json
import subprocess
import numpy as np

STARS_FREQ_GHZ = 1.1
AIC_FREQ_GHZ = 1.55

def parse_stars_log_for_cycles(stars_log_path):
    start_cycle = None
    end_cycle = None
    with open(stars_log_path, 'r') as f:
        for line in f:
            m = re.search(r'\[(\d+)\]', line)
            if m and start_cycle is None:
                start_cycle = int(m.group(1))
            if '[TASK_DONE]' in line:
                m2 = re.search(r'\[(\d+)\]', line)
                if m2:
                    end_cycle = int(m2.group(1))
    if start_cycle is None:
        print(f"Error: no start cycle found in {stars_log_path}")
        sys.exit(1)
    if end_cycle is None:
        print(f"Error: no [TASK_DONE] found in {stars_log_path}")
        sys.exit(1)
    return start_cycle, end_cycle

def run_peon(esl_path, case_name):
    result = subprocess.run(
        ['python3', 'peon_v121_v122.py', '-i', esl_path, '-o', 'peon', '-t', '32', '--case_name', case_name],
        capture_output=True, text=True
    )

    if result.stderr:
        print("Error output from peon_v121_v122.py:")
        print(result.stderr)

    rd_bw, l2_bw, wr_bw, total_bw = 0, 0, 0, 0
    for line in result.stdout.splitlines():
        if 'peon_result:' in line:
            parts = line.split('peon_result:')[1].strip().split(',')
            if len(parts) >= 5:
                rd_bw = float(parts[1])
                l2_bw = float(parts[2])
                wr_bw = float(parts[3])
                total_bw = float(parts[4])
            break
    return rd_bw, l2_bw, wr_bw, total_bw

def calc_data_size_bytes(core_num, enable_core_num):
    total_bytes = enable_core_num * 2 * 8 *128
    return total_bytes

def update_csv_output(csv_file, case_name, json_content):
    rows = []
    fieldnames = []
    with open(csv_file, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for row in reader:
            if row.get('key') == case_name or row.get('id') == case_name:
                row['output'] = json_content
            rows.append(row)
    with open(csv_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

def main():
    if len(sys.argv) < 5:
        print("Usage: run_test.py <csv_file> <esl_path> <case_name> <result_csv_path>")
        sys.exit(1)

    csv_file = sys.argv[1]
    esl_path = sys.argv[2]
    case_name = sys.argv[3]
    result_csv = sys.argv[4]

    stars_log_path = os.path.join(esl_path, 'log_ca', 'stars_log0_0.dump')
    if not os.path.exists(stars_log_path):
        stars_log_path = os.path.join(esl_path, 'stars_log0_0.dump')

    start_cycle, end_cycle = 0, 0
    if os.path.exists(stars_log_path):
        start_cycle, end_cycle = parse_stars_log_for_cycles(stars_log_path)
    delta_cycles = end_cycle - start_cycle
    duration_us = delta_cycles / (STARS_FREQ_GHZ * 1000) if delta_cycles > 0 else 0

    rd_bw, l2_bw, wr_bw, total_bw = run_peon(esl_path, case_name)

    result_row = {
        'case_name': case_name,
        'start_cycle': start_cycle,
        'end_cycle': end_cycle,
        'delta_cycles': delta_cycles,
        'duration_us': round(duration_us, 2),
        'peak_rd_bw(GB/s)': round(rd_bw, 4),
        'peak_l2_bw(GB/s)': round(l2_bw, 4),
        'peak_wr_bw(GB/s)': round(wr_bw, 4),
        'peak_total_bw(GB/s)': round(total_bw, 4),
    }

    file_exists = os.path.exists(result_csv)
    fieldnames = list(result_row.keys())
    with open(result_csv, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(result_row)

    # json_content = json.dumps({
    #     "duration(us)": round(duration_us, 2),
        # "peak_rd_bw(GB/s)": round(rd_bw, 4),
        # "peak_l2_bw(GB/s)": round(l2_bw, 4),
        # "peak_wr_bw(GB/s)": round(wr_bw, 4),
        # "peak_total_bw(GB/s)": round(total_bw, 4),
        # "effective_bw(GB/s)": round(effective_bw_gbs, 4),
    # })
    # update_csv_output(csv_file, case_name, json_content)

    print(f"result: case={case_name}, cycles={delta_cycles}, duration={duration_us:.2f}us")


if __name__ == '__main__':
    main()