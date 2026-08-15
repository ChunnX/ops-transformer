import os
import glob
import re
import subprocess
import sys
import pandas as pd
from typing import Optional, Dict, List
from datetime import datetime

# ===================== 1. 可灵活修改的配置项 =====================
class Config:
    # 路径配置(请与本机 bsa_multicase_run_perf.sh 中的 ROOT_DIR1/ROOT_DIR2 保持一致)
    MY_CODE_DIR = os.environ.get("BSA_MXFP4_DIR", "./mxfp4_dir")      # mxfp4 结果目录(由 bsa_multicase_run_perf.sh 传入)
    OTHERS_CODE_DIR = os.environ.get("BSA_BASE_DIR", "./bf16_dir")    # bf16/half 结果目录(对照)
    OUTPUT_CSV = f"./bsa_perf_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    LOG_FILE = "./bsa_task_duration_extract.log"

    # 列名(与 bsa_mxfp4 性能测试结果记录.xlsx 保持一致)
    COL_NAME_MY_CODE = "A5性能（us）mxfp4"     # mxfp4 耗时
    COL_NAME_OTHERS_CODE = "A5性能（us）half"  # half/bf16 耗时(对照)
    COL_NAME_OPT_EFFECT = "优化效果"           # 优化效果 = 对照 / mxfp4

    # 其他可配置项
    CASE_PREFIX = "case_"  # Case文件夹前缀
    KEEP_DECIMAL = 6       # 优化效果保留小数位数

# ===================== 2. 日志工具函数 =====================
def log_info(msg: str):
    print(msg)
    with open(Config.LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"{msg}\n")

# ===================== 3. 解析Case名称提取参数列 =====================
CASE_NAME_REGEX = re.compile(
    r"case_B(?P<B>\d+)_qS(?P<qS>\d+)_kvS(?P<kvS>\d+)_qN(?P<qN>\d+)_kvN(?P<kvN>\d+)"
    r"_D(?P<D>\d+)_sbX(?P<sbX>\d+)_sbY(?P<sbY>\d+)_sp(?P<sp>[\d.]+)"
    r"_innerP(?P<innerP>\d+)_qL(?P<qL>\w+)_kvL(?P<kvL>\w+)"
    r"_isVL(?P<isVL>\d+)_isVS(?P<isVS>\d+)_isActS(?P<isActS>\d+)_dev(?P<dev>\d+)"
)

def parse_case_name(case_name: str) -> Dict:
    """从Case名称解析出与xlsx一致的参数列"""
    m = CASE_NAME_REGEX.search(case_name)
    if not m:
        return {}
    d = m.groupdict()
    return {
        "B": int(d["B"]),
        "qN": int(d["qN"]),
        "kvN": int(d["kvN"]),
        "qS": int(d["qS"]),
        "kvS": int(d["kvS"]),
        "D": int(d["D"]),
        "blockShapeX": int(d["sbX"]),
        "blockShapeY": int(d["sbY"]),
        "稀疏率（舍弃的块的占比）": float(d["sp"]),
        "innerPrec": int(d["innerP"]),
    }

# ===================== 4. 核心函数：提取单个case的Task Duration(us) =====================
def extract_task_duration(case_dir: str) -> Optional[float]:
    """递归查找case目录下的OpBasicInfo.csv，提取Task Duration(us)"""
    csv_paths = glob.glob(os.path.join(case_dir, "**", "OpBasicInfo.csv"), recursive=True)

    if not csv_paths:
        log_info(f"[无文件] {case_dir} 下未找到OpBasicInfo.csv（任意子目录）")
        return None

    csv_path = csv_paths[0]
    log_info(f"找到OpBasicInfo.csv: {csv_path}")

    try:
        df = pd.read_csv(csv_path, encoding="utf-8", on_bad_lines="skip", engine="python")

        # 模糊匹配Task Duration(us)列
        target_col = None
        for col in df.columns:
            if re.search(r"task\s*duration\(\s*us\s*\)", col.lower().replace(" ", "")):
                target_col = col
                break

        if target_col is None:
            log_info(f"[无列名] {csv_path} 中未找到Task Duration(us)列，表头列名：{df.columns.tolist()}")
            return None

        duration_series = df[target_col].dropna()
        if duration_series.empty:
            log_info(f"[无数据] {csv_path} 中{target_col}列无有效数据")
            return None

        duration_str = str(duration_series.iloc[0]).strip().replace(",", "")
        if not re.match(r"^[0-9.]+$", duration_str):
            log_info(f"[格式错误] {csv_path} 中{target_col}列值非数字：{duration_str}")
            return None

        duration = float(duration_str)
        log_info(f"提取成功：{case_dir} → Task Duration = {duration} us")
        return duration

    except Exception as e:
        log_info(f"[解析失败] {csv_path} → 错误信息：{str(e)}")
        return None

# ===================== 5. 遍历所有case，收集对比数据 =====================
def collect_all_case_data() -> List[Dict]:
    """收集所有case的耗时对比数据，按xlsx列结构整理"""
    case_data = []

    # 预加载Others目录下的所有case（原始Case名称 → 完整路径）
    others_case_paths = glob.glob(os.path.join(Config.OTHERS_CODE_DIR, f"{Config.CASE_PREFIX}*"))
    others_case_to_path = {}
    for others_case in others_case_paths:
        others_case_name = os.path.basename(others_case)
        if others_case_name not in others_case_to_path:
            others_case_to_path[others_case_name] = others_case
        else:
            log_info(f"[重复Case名称] Others目录中多个case名称为 {others_case_name}，仅保留第一个")

    # 处理MyCode目录下的case
    my_code_case_paths = glob.glob(os.path.join(Config.MY_CODE_DIR, f"{Config.CASE_PREFIX}*"))
    log_info(f"\n=== 开始处理MyCode目录 ===")
    log_info(f"MyCode目录路径：{Config.MY_CODE_DIR}")
    log_info(f"找到MyCode case数量：{len(my_code_case_paths)}")

    processed_others_cases = set()
    for my_code_case in my_code_case_paths:
        case_name = os.path.basename(my_code_case)
        log_info(f"\n=====================================")
        log_info(f"处理Case：{case_name}")

        my_code_duration = extract_task_duration(my_code_case)

        others_duration = None
        others_case = others_case_to_path.get(case_name)
        if others_case:
            processed_others_cases.add(case_name)
            log_info(f"匹配到Others case（名称一致）：{others_case}")
            others_duration = extract_task_duration(others_case)
        else:
            log_info(f"[无匹配] Others目录下无名称为 {case_name} 的case")

        # 计算优化效果：对照 / mxfp4
        opt_effect = None
        if my_code_duration is not None and others_duration is not None and my_code_duration != 0:
            opt_effect = round(others_duration / my_code_duration, Config.KEEP_DECIMAL)

        # 判定数据状态
        if my_code_duration is None and others_duration is None:
            status = "双方均无有效数据"
        elif my_code_duration is None:
            status = "MyCode无有效数据"
        elif others_duration is None:
            status = "Others无有效数据"
        else:
            status = "数据完整"

        row = parse_case_name(case_name)
        row[Config.COL_NAME_MY_CODE] = my_code_duration
        row[Config.COL_NAME_OTHERS_CODE] = others_duration
        row[Config.COL_NAME_OPT_EFFECT] = opt_effect
        row["Case名称"] = case_name
        row["数据状态"] = status
        case_data.append(row)

    # 补充Others目录下独有的case
    log_info(f"\n=== 补充Others独有Case ===")
    for others_case_name, others_case in others_case_to_path.items():
        if others_case_name not in processed_others_cases:
            log_info(f"处理Others独有Case：{others_case_name}")
            others_duration = extract_task_duration(others_case)
            row = parse_case_name(others_case_name)
            row[Config.COL_NAME_MY_CODE] = None
            row[Config.COL_NAME_OTHERS_CODE] = others_duration
            row[Config.COL_NAME_OPT_EFFECT] = None
            row["Case名称"] = others_case_name
            row["数据状态"] = "MyCode目录无此名称Case"
            case_data.append(row)

    return case_data

# ===================== 6. 生成CSV报告（列结构与xlsx一致） =====================
def generate_report(case_data: List[Dict]):
    """生成CSV报告，列顺序与 bsa_mxfp4 性能测试结果记录.xlsx 一致"""
    df = pd.DataFrame(case_data)

    # 目标列顺序：前13列与xlsx完全一致，末尾追加 Case名称/数据状态 便于追溯
    xlsx_cols = [
        "B", "qN", "kvN", "qS", "kvS", "D",
        "blockShapeX", "blockShapeY", "稀疏率（舍弃的块的占比）", "innerPrec",
        Config.COL_NAME_MY_CODE, Config.COL_NAME_OTHERS_CODE, Config.COL_NAME_OPT_EFFECT,
    ]
    # 仅保留实际存在的列（parse失败时参数列可能缺失）
    cols = [c for c in xlsx_cols if c in df.columns]
    for extra in ["Case名称", "数据状态"]:
        if extra in df.columns:
            cols.append(extra)
    df = df[cols]

    # 整数列: 缺失填NA字符串, 有值则保持整数输出(避免1.0)
    int_cols = ["B", "qN", "kvN", "qS", "kvS", "D", "blockShapeX", "blockShapeY", "innerPrec"]
    for c in int_cols:
        if c in df.columns:
            df[c] = df[c].apply(lambda x: "NA" if pd.isna(x) else str(int(x)))

    # 其余列填充NA
    for c in df.columns:
        if c not in int_cols:
            df[c] = df[c].fillna("NA")

    df.to_csv(Config.OUTPUT_CSV, index=False, encoding="utf-8-sig")
    log_info(f"\nCSV报告已保存：{os.path.abspath(Config.OUTPUT_CSV)}")
    log_info(f"总Case数：{len(df)}")

# ===================== 7. 自动安装依赖 =====================
def install_dependencies():
    required_packages = ["pandas"]
    missing_packages = []
    for pkg in required_packages:
        try:
            __import__(pkg)
        except ImportError:
            missing_packages.append(pkg)

    if missing_packages:
        log_info(f"缺少依赖包：{missing_packages}，正在自动安装...")
        try:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "-i", "https://pypi.tuna.tsinghua.edu.cn/simple"] + missing_packages
            )
            log_info("依赖包安装成功！")
        except subprocess.CalledProcessError as e:
            log_info(f"依赖包安装失败：{str(e)}")
            sys.exit(1)

# ===================== 8. 主函数 =====================
def main():
    with open(Config.LOG_FILE, "w", encoding="utf-8") as f:
        f.write(f"=== Task Duration提取日志 | 运行时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n")

    if not os.path.exists(Config.MY_CODE_DIR):
        log_info(f"MyCode目录不存在：{Config.MY_CODE_DIR}")
        return
    if not os.path.exists(Config.OTHERS_CODE_DIR):
        log_info(f"OthersCode目录不存在：{Config.OTHERS_CODE_DIR}")
        return

    install_dependencies()
    case_data = collect_all_case_data()
    if case_data:
        generate_report(case_data)
    else:
        log_info("未收集到任何Case数据！")

    log_info(f"\n=== 任务完成 ===")
    log_info(f"日志文件：{os.path.abspath(Config.LOG_FILE)}")

if __name__ == "__main__":
    main()
