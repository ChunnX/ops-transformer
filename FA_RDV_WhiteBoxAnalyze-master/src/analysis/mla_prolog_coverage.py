import os
import pandas as pd
from typing import Dict, List, Any, Optional
from .coverage import CoverageAnalyzer

class MlaPrologCoverageAnalyzer:
    """mlaProlog算子的专用覆盖率分析器"""
    
    def __init__(self, chipversion: str = "A5"):
        """
        初始化mlaProlog覆盖率分析器
        
        参数:
            chipversion: 芯片版本，可选值为 "A2"、"A3"、"A5"
        """
        self.chipversion = chipversion.upper()
        if self.chipversion not in ["A2", "A3", "A5"]:
            print(f"警告: 未知的芯片版本 {chipversion}，默认使用 A5")
            self.chipversion = "A5"
            
        self.analyzer = CoverageAnalyzer("mlaProlog")
        self._define_parameters()
        
        print(f"初始化 mlaProlog 覆盖率分析器 - 芯片版本: {self.chipversion}")
    
    def _define_parameters(self):
        """根据芯片版本定义mlaProlog算子需要分析的参数"""
        
        # 1. cache_mode参数（根据芯片版本确定）
        if self.chipversion in ["A2", "A3"]:
            cache_modes = ["PA_BSND", "PA_NZ", "PA_BLK_BSND", "PA_BLK_NZ", "BSND", "TND"]
        else:  # A5
            cache_modes = ["PA_BSND", "PA_NZ"]
        self.analyzer.define_parameter("cache_mode", cache_modes)
        
        # 2. 量化模式相关参数（根据芯片版本确定）
        if self.chipversion in ["A2", "A3"]:
            # A2/A3: int8 和 bf16
            a2_a3_dtypes = ["int8", "bf16"]
            self.analyzer.define_parameter("x_dtype", a2_a3_dtypes)
            self.analyzer.define_parameter("w_uq_qr_dtype", a2_a3_dtypes)
            self.analyzer.define_parameter("out_q_dtype", a2_a3_dtypes)
            self.analyzer.define_parameter("kv_cache_dtype", a2_a3_dtypes)
            self.analyzer.define_parameter("kr_cache_dtype", a2_a3_dtypes)
        else:  # A5
            # A5: bf16 和 float8_e4m3fn，但kr_cache_dtype只有bf16
            a5_dtypes = ["bf16", "float8_e4m3fn"]
            self.analyzer.define_parameter("x_dtype", a5_dtypes)
            self.analyzer.define_parameter("w_uq_qr_dtype", a5_dtypes)
            self.analyzer.define_parameter("out_q_dtype", a5_dtypes)
            self.analyzer.define_parameter("kv_cache_dtype", a5_dtypes)
            self.analyzer.define_parameter("kr_cache_dtype", ["bf16"])  # 只有bf16
        
        # 3. 优化标志参数
        self.analyzer.define_parameter("dequantOpt", [True, False])
        self.analyzer.define_parameter("GroupComputeOpt", [False])  # 只有False
        
        # 4. matmul切分参数
        # matmul1和matmul2只有一种组合
        self.analyzer.define_parameter("matmul1", ["212212"])
        self.analyzer.define_parameter("matmul2", ["212212"])
        # matmul3有三种组合
        self.analyzer.define_parameter("matmul3", ["202202", "212212", "222222"])
        # matmul4只有一种组合
        self.analyzer.define_parameter("matmul4", ["222222"])

        # 5. 切M参数：1：BS/T > 128 | 0: BS/T <= 128
        split_m_combinations = [0, 1]
        self.analyzer.define_parameter("split_M", split_m_combinations)

        # 6. vector核间切分参数: True:分核均匀 | False:分核不均匀
        self.analyzer.define_parameter("vector_core_split", [True, False])
    
    def add_case(self, case_name: str, dimensions_state: Dict[str, Any]) -> None:
        """添加mlaProlog用例"""
        # 从dimensions_state中提取需要分析的参数
        case_parameters = {}
        
        # 提取基础参数
        case_parameters['cache_mode'] = dimensions_state.get('cache_mode', 'unknown')
        
        # 提取量化模式参数
        quant_info = dimensions_state.get('quantization_info', {})
        case_parameters['x_dtype'] = quant_info.get('x_dtype', 'unknown')
        case_parameters['w_uq_qr_dtype'] = quant_info.get('w_uq_qr_dtype', 'unknown')
        case_parameters['out_q_dtype'] = quant_info.get('out_q_dtype', 'unknown')
        case_parameters['kv_cache_dtype'] = quant_info.get('kv_cache_dtype', 'unknown')
        case_parameters['kr_cache_dtype'] = quant_info.get('kr_cache_dtype', 'unknown')
        
        # 提取优化标志
        case_parameters['dequantOpt'] = dimensions_state.get('enableDequantOpt', False)
        case_parameters['GroupComputeOpt'] = dimensions_state.get('enableGroupComputeOpt', False)
        
        # 提取matmul切分结果
        case_parameters['matmul1'] = dimensions_state.get('matmul1', 'unknown')
        case_parameters['matmul2'] = dimensions_state.get('matmul2', 'unknown')
        case_parameters['matmul3'] = dimensions_state.get('matmul3', 'unknown')
        case_parameters['matmul4'] = dimensions_state.get('matmul4', 'unknown')

        # 提取切M结果
        case_parameters['split_M'] = dimensions_state.get('m_axis_gt_128', 'unknown')

        # 提取vector核间切分结果
        case_parameters['vector_core_split'] = dimensions_state.get('vector_core_split', 'unknown')
        
        self.analyzer.add_case(case_name, case_parameters)
    
    def generate_report(self, base_output_dir: str = ".") -> str:
        """生成mlaProlog专用报告"""
        return self._generate_enhanced_excel_report(base_output_dir) 
    
    def _generate_enhanced_excel_report(self, base_output_dir: str = ".") -> str:
        """生成增强的Excel格式覆盖率报告，返回文件路径"""
        # 构建目标目录路径
        target_dir = os.path.join(base_output_dir, "result", "mlaProlog")
        os.makedirs(target_dir, exist_ok=True)
        
        # 构建完整的文件路径
        output_filename = f"coverage_analysis_mlaProlog_{self.chipversion}.xlsx"
        output_path = os.path.join(target_dir, output_filename)
        
        try:
            with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                # 1. 总体统计表
                overall_stats = self.analyzer.get_overall_coverage()
                overall_df = pd.DataFrame([overall_stats])
                overall_df.to_excel(writer, sheet_name='总体统计', index=False)
                
                # 2. 参数覆盖率详情表（新增"覆盖到的取值"列）
                coverage_data = []
                all_coverages = self.analyzer.get_all_coverages()
                
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
                for case in self.analyzer.cases:
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
        print(f"\n=== mlaProlog算子覆盖率分析报告 (芯片版本: {self.chipversion}) ===")
        print(f"用例总数: {overall_stats['total_cases']}")
        print(f"分析参数数: {overall_stats['total_parameters']}")
        print(f"平均覆盖率: {overall_stats['average_coverage']}%")
        
        # 打印各参数覆盖率
        all_coverages = self.analyzer.get_all_coverages()
        for param_name, stats in all_coverages.items():
            if stats['total_possible'] > 0:
                print(f"  {param_name}: {stats['covered_count']}/{stats['total_possible']} ({stats['coverage_rate']}%)")
        
        print(f"\n报告已保存至: {output_path}")
    
    def get_parameter_coverage(self, param_name: str) -> Dict[str, Any]:
        """获取指定参数的覆盖率详情"""
        return self.analyzer.get_parameter_coverage(param_name)
    
    def get_all_coverages(self) -> Dict[str, Dict[str, Any]]:
        """获取所有参数的覆盖率"""
        return self.analyzer.get_all_coverages()
    
    def get_overall_coverage(self) -> Dict[str, Any]:
        """获取整体覆盖率统计"""
        return self.analyzer.get_overall_coverage()