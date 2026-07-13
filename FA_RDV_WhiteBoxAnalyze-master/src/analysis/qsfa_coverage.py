from .coverage import CoverageAnalyzer
from typing import Dict, List, Any

# 0:整块 1:尾块 2:整块+尾块
class QSFACoverageAnalyzer:
    """QSFA算子的专用覆盖率分析器"""
    
    def __init__(self, chipversion: str = "A3"):
        self.analyzer = CoverageAnalyzer("QSFA")
        self.chipversion = chipversion
        self._define_parameters()
    
    def _define_parameters(self):
        """定义QSFA算子需要分析的参数"""
        # 1. layout_query参数
        layout_query = ["BSND", "TND"]
        self.analyzer.define_parameter("layout_query", layout_query)
        # 2. layout_KEY参数
        layout_kv = ["BSND", "TND", "PA_BSND"]
        self.analyzer.define_parameter("layout_kv", layout_kv)
        # 3. dtypes参数
        dtypes_query = ["bf16", "fp16"]
        dtypes_key = ["int8", "hifloat8", "float8_e4m3fn"]
        dtypes_value = ["int8", "hifloat8", "float8_e4m3fn"]
        self.analyzer.define_parameter("query_dtype", dtypes_query)
        self.analyzer.define_parameter("key_dtype", dtypes_key)
        self.analyzer.define_parameter("value_dtype", dtypes_value)
        # 4. sparsemode参数
        sparsemode = [0, 3]
        self.analyzer.define_parameter("sparsemode", sparsemode)
        # 5. PA_block
        PA_block = ['0', '1', '2']
        if self.chipversion == 'A5':
            self._define_parameters_A5()
        else:
            self._define_parameters_A2A3()

    def _define_parameters_A2A3(self):
        # 6. inter_core_split
        inter_core_split = ['00', '01', '02', '10', '11', '12', '20', '21', '22']
        self.analyzer.define_parameter("inter-core-split", inter_core_split)
        # 7. intra-core-vec0
        intra_core_vec0 = ['00', '10', '20']
        self.analyzer.define_parameter("intra-core-vec0", intra_core_vec0)
        # 8. intra-core-mm1_L1
        self.analyzer.define_parameter("intra-core-mm1_L1", self.IntraSplitMM())
        # 9. intra-core-mm1_L0
        self.analyzer.define_parameter("intra-core-mm1_L0", self.IntraSplitMM())
        # 10. intra-core-vec1
        intra_core_vec1 = ['000', '010', '020', '100', '110', '120', '200', '210', '220']
        self.analyzer.define_parameter("intra-core-vec1", intra_core_vec1)
        # 11. intra-core-mm2_L1
        self.analyzer.define_parameter("intra-core-mm2_L1", self.IntraSplitMM())
        # 12. intra-core-mm2_L0
        self.analyzer.define_parameter("intra-core-mm2_L0", self.IntraSplitMM())
        # 13. intra-core-vec2
        intra_core_vec2 = ['00', '01', '02', '10', '11', '12', '20', '21', '22']
        self.analyzer.define_parameter("intra-core-vec2", intra_core_vec2)

    def _define_parameters_A5(self):
        # 6. inter-core-split
        inter_core_split = ['00', '01', '02']
        self.analyzer.define_parameter("inter-core-split", inter_core_split)
        # 7. intra-core-vec0
        intra_core_vec0 = ['00', '01', '02']
        self.analyzer.define_parameter("intra-core-vec0", intra_core_vec0)
        # 8. intra-core-mm1_L1
        intra_core_mm1_L1 = ['000', '010', '020']
        self.analyzer.define_parameter("intra-core-mm1_L1", intra_core_mm1_L1)
        # 9. intra-core-mm1_L0
        intra_core_mm1_L0 = ['002', '012', '022']
        self.analyzer.define_parameter("intra-core-mm1_L0", intra_core_mm1_L0)
        # 10. intra-core-vec1
        intra_core_vec1 = ['000', '011', '012', '020', '023', '024']
        self.analyzer.define_parameter("intra-core-vec1", intra_core_vec1)
        # 11. intra-core-mm2_L1
        intra_core_mm2_L1 = ['000', '001', '002', '010', '011', '012', '020', '021', '022']
        self.analyzer.define_parameter("intra-core-mm2_L1", intra_core_mm2_L1)
        # 12. intra-core-mm2_L0
        intra_core_mm2_L0 = ['000', '001', '002', '010', '011', '012', '020', '021', '022']
        self.analyzer.define_parameter("intra-core-mm2_L0", intra_core_mm2_L0)
        # 13. intra-core-vec2
        intra_core_vec2 = ['00', '01', '02']
        self.analyzer.define_parameter("intra-core-vec2", intra_core_vec2)

    def IntraSplitMM(self) -> List[str]:
        """
        生成所有三位数的全排列，每位可取 '0','1','2'。
        格式：abc (3位)
        """
        combinations = []
        digits = ['0', '1', '2']
        
        for a in digits:
            for b in digits:
                for c in digits:
                    comb = f"{a}{b}{c}"
                    combinations.append(comb)
        return combinations
    
    def add_case(self, case_name: str, dimensions_state: Dict[str, Any], params) -> None:
        """添加LI用例"""
        # 从dimensions_state中提取需要分析的参数
        case_parameters = {}
        # 1. 提取layout_query参数
        print(params)
        case_parameters['layout_query'] = params.get('layout_query')
        # 2. layout_key参数
        case_parameters['layout_kv'] = params.get('layout_kv')
        # 3. dtypes参数
        case_parameters['query_dtype'] = params.get('query', {}).get('dtype')
        case_parameters['key_dtype'] = params.get('key', {}).get('dtype')
        case_parameters['value_dtype'] = params.get('value', {}).get('dtype')
        # 4. sparsemode参数
        case_parameters['sparsemode'] = params.get('sparsemode')
        # 5. PA_block
        case_parameters['PA_block'] = dimensions_state.get('stage_codes', {}).get('PA_block')
        # 6. sinter-core-split
        case_parameters['inter-core-split'] = dimensions_state.get('stage_codes', {}).get('inter-core-split')
        # 7. intra-core-vec0
        case_parameters['intra-core-vec0'] = dimensions_state.get('stage_codes', {}).get('intra-core-vec0')
        # 8. intra-core-mm1_L1
        case_parameters['intra-core-mm1_L1'] = dimensions_state.get('stage_codes', {}).get('intra-core-mm1_L1')
        # 9. intra-core-mm1_L0
        case_parameters['intra-core-mm1_L0'] = dimensions_state.get('stage_codes', {}).get('intra-core-mm1_L0')
        # 10. intra-core-vec1
        case_parameters['intra-core-vec1'] = dimensions_state.get('stage_codes', {}).get('intra-core-vec1')
        # 11. intra-core-mm2_L1
        case_parameters['intra-core-mm2_L1'] = dimensions_state.get('stage_codes', {}).get('intra-core-mm2_L1')
        # 12. intra-core-mm2_L0
        case_parameters['intra-core-mm2_L0'] = dimensions_state.get('stage_codes', {}).get('intra-core-mm2_L0')
        # 12. intra-core-vec2
        case_parameters['intra-core-vec2'] = dimensions_state.get('stage_codes', {}).get('intra-core-vec2')

        self.analyzer.add_case(case_name, case_parameters)
    
    def generate_report(self, base_output_dir: str = ".") -> str:
        """生成mlaProlog专用报告"""
        return self.analyzer.generate_excel_report(base_output_dir)
    
    def get_parameter_coverage(self, param_name: str) -> Dict[str, Any]:
        """获取指定参数的覆盖率详情"""
        return self.analyzer.get_parameter_coverage(param_name)
    
    def get_all_coverages(self) -> Dict[str, Dict[str, Any]]:
        """获取所有参数的覆盖率"""
        return self.analyzer.get_all_coverages()
    
    def get_overall_coverage(self) -> Dict[str, Any]:
        """获取整体覆盖率统计"""
        return self.analyzer.get_overall_coverage()