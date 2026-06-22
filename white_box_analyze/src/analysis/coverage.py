import os
import pandas as pd
from typing import Dict, List, Any, Set, Optional
from collections import defaultdict
import itertools

class ParameterCoverage:
    """参数维度覆盖率分析"""
    
    def __init__(self, param_name: str, possible_values: List[Any]):
        self.param_name = param_name
        self.possible_values = possible_values
        self.covered_values = set()
        self.total_cases = 0
    
    def add_case(self, value: Any) -> None:
        """添加用例覆盖情况"""
        self.total_cases += 1
        if value in self.possible_values:
            self.covered_values.add(value)
        else:
            # 如果值不在可能值列表中，也记录下来（但不算入覆盖率）
            pass
    
    def get_coverage_stats(self) -> Dict[str, Any]:
        """获取覆盖率统计"""
        total_possible = len(self.possible_values)
        covered_count = len(self.covered_values)
        missing_values = set(self.possible_values) - self.covered_values
        
        coverage_rate = 0
        if total_possible > 0:
            coverage_rate = round(covered_count / total_possible * 100, 2)
        
        return {
            'parameter': self.param_name,
            'total_possible': total_possible,
            'covered_count': covered_count,
            'covered_values': list(self.covered_values),
            'missing_values': list(missing_values),
            'coverage_rate': coverage_rate,
            'possible_values': self.possible_values  # 保存可能值列表供后续使用
        }

class CoverageAnalyzer:
    """通用的覆盖率分析器，与具体算子解耦"""
    
    def __init__(self, op_name: str):
        self.op_name = op_name
        self.parameter_coverages: Dict[str, ParameterCoverage] = {}
        self.cases = []
        self.case_info_map = {}
    
    def define_parameter(self, param_name: str, possible_values: List[Any]) -> None:
        """定义需要分析的参数及其可能取值"""
        self.parameter_coverages[param_name] = ParameterCoverage(param_name, possible_values)
    
    def add_case(self, case_name: str, case_parameters: Dict[str, Any]) -> None:
        """添加用例到分析器"""
        case_info = {
            'case_name': case_name,
            'parameters': case_parameters
        }
        self.cases.append(case_info)
        self.case_info_map[case_name] = case_info
        
        # 更新各参数的覆盖率
        for param_name, param_value in case_parameters.items():
            if param_name in self.parameter_coverages:
                self.parameter_coverages[param_name].add_case(param_value)
    
    def get_parameter_coverage(self, param_name: str) -> Optional[Dict[str, Any]]:
        """获取指定参数的覆盖率"""
        if param_name in self.parameter_coverages:
            return self.parameter_coverages[param_name].get_coverage_stats()
        return None
    
    def get_all_coverages(self) -> Dict[str, Dict[str, Any]]:
        """获取所有参数的覆盖率"""
        coverages = {}
        for param_name in self.parameter_coverages:
            coverages[param_name] = self.get_parameter_coverage(param_name)
        return coverages
    
    def get_overall_coverage(self) -> Dict[str, Any]:
        """获取整体覆盖率统计"""
        total_parameters = len(self.parameter_coverages)
        if total_parameters == 0:
            return {'average_coverage': 0, 'total_cases': len(self.cases)}
        
        total_coverage = 0
        for param_name in self.parameter_coverages:
            stats = self.get_parameter_coverage(param_name)
            total_coverage += stats['coverage_rate']
        
        return {
            'average_coverage': round(total_coverage / total_parameters, 2),
            'total_cases': len(self.cases),
            'total_parameters': total_parameters
        }
    
    def generate_excel_report(self, base_output_dir: str = ".") -> str:
        """生成Excel格式的覆盖率报告"""
        # 构建目标目录路径
        target_dir = os.path.join(base_output_dir, "result", self.op_name)
        os.makedirs(target_dir, exist_ok=True)
        
        # 构建完整的文件路径
        output_filename = f"coverage_analysis_{self.op_name}.xlsx"
        output_path = os.path.join(target_dir, output_filename)
        
        try:
            with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                # 1. 总体统计表
                overall_stats = self.get_overall_coverage()
                overall_df = pd.DataFrame([overall_stats])
                overall_df.to_excel(writer, sheet_name='总体统计', index=False)
                
                # 2. 参数覆盖率详情表（新增"覆盖到的取值"列）
                coverage_data = []
                all_coverages = self.get_all_coverages()
                
                for param_name, stats in all_coverages.items():
                    # 获取已覆盖的值
                    covered_values = stats.get('covered_values', [])
                    if not covered_values and 'missing_values' in stats:
                        # 如果没有直接提供covered_values，计算它
                        all_possible = stats.get('total_possible', 0)
                        missing_values = stats.get('missing_values', [])
                        if all_possible and 'possible_values' in stats:
                            possible_values = stats.get('possible_values', [])
                            covered_values = [v for v in possible_values if v not in missing_values]
                    
                    coverage_data.append({
                        '参数名称': stats['parameter'],
                        '可能取值总数': stats['total_possible'],
                        '已覆盖取值数': stats['covered_count'],
                        '覆盖到的取值': ', '.join(map(str, covered_values)) if covered_values else '无',
                        '未覆盖取值': ', '.join(map(str, stats['missing_values'])) if stats['missing_values'] else '无',
                        '覆盖率(%)': stats['coverage_rate']
                    })
                
                coverage_df = pd.DataFrame(coverage_data)
                coverage_df.to_excel(writer, sheet_name='参数覆盖率', index=False)
                
                # 3. 用例详情表
                case_data = []
                for case in self.cases:
                    case_row = {'用例名': case['case_name']}
                    for param_name, param_value in case['parameters'].items():
                        case_row[param_name] = param_value
                    case_data.append(case_row)
                
                # 对用例详情进行排序，使相同参数的用例相邻
                case_df = pd.DataFrame(case_data)
                
                # 生成一个用于排序的键：将所有参数值连接起来
                if not case_df.empty:
                    # 获取所有参数列（排除用例名列）
                    param_columns = [col for col in case_df.columns if col != '用例名']
                    
                    # 创建排序键
                    case_df['sort_key'] = case_df[param_columns].apply(
                        lambda row: ''.join(str(row[col]) for col in param_columns), axis=1
                    )
                    
                    # 按排序键排序
                    case_df = case_df.sort_values('sort_key')
                    
                    # 删除临时列
                    case_df = case_df.drop(columns=['sort_key'])
                
                # 新增：添加code列，将所有参数值连接起来
                if not case_df.empty:
                    # 获取所有参数列（排除用例名列）
                    param_columns = [col for col in case_df.columns if col != '用例名']
                    
                    # 创建code列
                    case_df['code'] = case_df[param_columns].apply(
                        lambda row: '_'.join(str(row[col]) for col in param_columns), axis=1
                    )
                    
                    # 新增：添加if_reserve列
                    # 初始化所有行都保留
                    case_df['if_reserve'] = True
                    
                    # 遍历每一行，如果当前行和下一行的code相同，则下一行不保留
                    for i in range(len(case_df) - 1):
                        if case_df.iloc[i]['code'] == case_df.iloc[i + 1]['code']:
                            # 当前行保留，下一行不保留
                            case_df.at[case_df.index[i + 1], 'if_reserve'] = False
                    
                    # 调整列顺序，将code和if_reserve放在前面
                    columns_order = ['用例名', 'code', 'if_reserve'] + param_columns
                    case_df = case_df[columns_order]
                
                case_df.to_excel(writer, sheet_name='用例详情', index=False)
            
            # 打印统计信息
            self._print_coverage_report(overall_stats, output_path)
            
            return output_path
            
        except Exception as e:
            print(f"生成Excel报告时出错: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _print_coverage_report(self, overall_stats: Dict, output_path: str) -> None:
        """打印覆盖率报告"""
        print(f"\n=== {self.op_name}算子覆盖率分析报告 ===")
        print(f"用例总数: {overall_stats['total_cases']}")
        print(f"分析参数数: {overall_stats['total_parameters']}")
        print(f"平均覆盖率: {overall_stats['average_coverage']}%")
        
        # 打印各参数覆盖率
        all_coverages = self.get_all_coverages()
        for param_name, stats in all_coverages.items():
            if stats['total_possible'] > 0:
                print(f"  {param_name}: {stats['covered_count']}/{stats['total_possible']} ({stats['coverage_rate']}%)")
        
        print(f"\n报告已保存至: {output_path}")