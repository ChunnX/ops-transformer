import yaml
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

from src.core.op_registry import OperatorRegistry
from src.core.parser_registry import ParserRegistry
from src.parsers import *
from src.operators import *
from src.analysis.encoder import MlaPrologCaseEncoder
from src.analysis.coverage import CoverageAnalyzer

def parse_arguments():
    """定义并解析命令行参数"""
    parser = argparse.ArgumentParser(description="算子白盒测试用例分析工具")
    parser.add_argument( '-op', '--operator', type=str, help='指定要分析的算子名称 (区分大小写，例如: mlaProlog)。', default='mlaProlog')
    parser.add_argument('-aiv', '--aivNum', type=int, help='指定aiv的数量。', default=48)
    parser.add_argument('-aic', '--aicNum', type=int, help='指定aic的数量。', default=24)
    parser.add_argument('-cv', '--chipversion', type=str, help='芯片版本 (A2, A3, A5)', default='A5')
    parser.add_argument('-tf', '--testFrame', type=str, help='测试框架 (aclnn, pytest)', default='aclnn')

    return parser.parse_args()

def main():
    # 1. 解析命令行参数
    args = parse_arguments()
    op_name = args.operator
    aiv_num = args.aivNum
    aic_num = args.aicNum
    chipversion = args.chipversion
    test_frame = args.testFrame

    # 2. 获取所有用例
    all_files = [f.resolve() for f in Path('data', f'{op_name}_cases').iterdir() if f.is_file()]
    if test_frame == 'aclnn':
        cases = [f for f in all_files if f.suffix != '.xlsx']
    else:
        xlsx_files = [f for f in all_files if f.suffix == '.xlsx']
        cases = []
        for file_path in xlsx_files:
            dataframe = pd.read_excel(file_path).replace({np.nan: None, pd.NA: None})
            cases.extend([row.to_dict() for _, row in dataframe.iterrows()])

    # 存储分析结果
    analysis_results = {}
    for case in cases:
        # 3. 获取算子json用例解析器
        ParserClass = ParserRegistry.get_parser(op_name)
        try:
            parser = ParserClass(op_name = op_name, json_path = case)
        except Exception as e:
            print(f"[*** ERROR ***] op_name={op_name}, ParserClass failed: {e}")
            continue
        params = parser.params
        # # print("解析结果:")
        # for param, info in params.items():
        #     print(f"{param}: {info}")

        case_name = params.get('case_name', 'unknown_case')

        # 4. 获取对应的算子执行器
        OperatorClass = OperatorRegistry.get_operator(op_name)
        operator = OperatorClass(op_name, chipversion)

        # 5. 执行白盒逻辑分析
        try:
            dimensions_state = operator.simulate_whitebox_logic(aic_num, aiv_num, params)
        except Exception as e:
            print(f"[*** ERROR ***] case_name={case_name}, file={case} failed: {e}")
            continue
        if op_name not in analysis_results:
            # 根据算子类型创建相应的覆盖率分析器
            if op_name == "mlaProlog":
                from src.analysis.mla_prolog_coverage import MlaPrologCoverageAnalyzer
                analysis_results[op_name] = MlaPrologCoverageAnalyzer(chipversion)
            elif op_name == "QLI":
                from src.analysis.qli_coverage import QLICoverageAnalyzer
                analysis_results[op_name] = QLICoverageAnalyzer()
            elif op_name == "LI":
                from src.analysis.li_coverage import LICoverageAnalyzer
                analysis_results[op_name] = LICoverageAnalyzer()
            elif op_name == "SFA":
                from src.analysis.sfa_coverage import SFACoverageAnalyzer
                analysis_results[op_name] = SFACoverageAnalyzer(chipversion)
            elif op_name == "QSFA":
                from src.analysis.qsfa_coverage import QSFACoverageAnalyzer
                analysis_results[op_name] = QSFACoverageAnalyzer(chipversion)
            else:
                # 对于其他算子，使用通用分析器
                analysis_results[op_name] = CoverageAnalyzer(op_name)
                # 这里需要为其他算子定义参数，可以根据需要扩展
        
        # 添加用例到覆盖率分析器
        if op_name == "mlaProlog":
            analysis_results[op_name].add_case(case_name, dimensions_state)
        elif op_name == "QLI":
            analysis_results[op_name].add_case(case_name, dimensions_state, params)
        elif op_name == "LI":
            analysis_results[op_name].add_case(case_name, dimensions_state, params)
        elif op_name == "SFA":
            analysis_results[op_name].add_case(case_name, dimensions_state, params)
        elif op_name == "QSFA":
            analysis_results[op_name].add_case(case_name, dimensions_state, params)
        else:
            # 对于其他算子，需要提取参数（这里需要根据具体算子实现）
            case_parameters = self._extract_parameters_for_other_ops(dimensions_state)
            analysis_results[op_name].add_case(case_name, case_parameters)
    
    for op_type, analyzer in analysis_results.items():
        report_path = analyzer.generate_report(".")
        if report_path:
            print(f"成功生成 {op_type} 算子报告: {report_path}")
            
            # 输出详细的覆盖率信息
            if hasattr(analyzer, 'get_all_coverages'):
                coverages = analyzer.get_all_coverages()
                overall = analyzer.get_overall_coverage()
                
                print(f"\n{op_type}算子详细覆盖率:")
                for param_name, stats in coverages.items():
                    if stats['total_possible'] > 0:  # 只显示有定义可能值的参数
                        print(f"  {param_name}: {stats['coverage_rate']}%")
                        
                        # 显示重要的未覆盖值
                        if stats['missing_values'] and len(stats['missing_values']) <= 5:
                            print(f"    未覆盖: {stats['missing_values']}")
        else:
            print(f"生成报告失败: {op_type}")

if __name__ == "__main__":
    main()