import sys
import pandas as pd


if __name__ == '__main__':
    case_file, sheet_name, start_idx, end_idx = sys.argv[1:5]
    start_idx = int(start_idx)
    end_idx = int(end_idx)
    xlsx_df = pd.read_excel(case_file, sheet_name=sheet_name)
    for invalid_title in ["备注", "用例来源"]:
        if invalid_title in xlsx_df.columns:
            xlsx_df = xlsx_df.drop(columns=[invalid_title])
    i = 1
    case_result = []
    for row in xlsx_df.to_dict(orient='records'):
        line = list(row.values())
        if start_idx <= i <= end_idx:
            line_str = str(line).replace(" ", "")[1:-1] + "\n"
            # print(line_str)
            case_result.append(line_str)
        if i > end_idx:
            break
        i += 1
    with open("case.txt", "w", encoding="utf-8") as f:
        f.writelines(case_result)

