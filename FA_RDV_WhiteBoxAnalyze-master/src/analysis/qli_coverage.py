from .coverage import CoverageAnalyzer
from typing import Dict, List, Any

# 0:整块 1:尾块 2:整块+尾块
class QLICoverageAnalyzer:
    """qli算子的专用覆盖率分析器"""
    
    def __init__(self):
        self.analyzer = CoverageAnalyzer("QLI")
        self._define_parameters()
    
    def _define_parameters(self):
        """定义qli算子需要分析的参数"""
        # 1. layout_query参数
        layout_query = ["BSND", "TND"]
        self.analyzer.define_parameter("layout_query", layout_query)
        # 2. layout_KEY参数
        layout_key = ["BSND", "TND", "PA_BSND"]
        self.analyzer.define_parameter("layout_key", layout_key)
        # 3. dtypes参数
        dtypesQK = ["int8"]
        self.analyzer.define_parameter("query_dtype", dtypesQK)
        self.analyzer.define_parameter("key_dtype", dtypesQK)
        dtypesW = ["fp16"]
        self.analyzer.define_parameter("weights_dtype", dtypesW)
        # 4. sparsemode参数
        sparse_mode = [0, 3]
        self.analyzer.define_parameter("sparse_mode", sparse_mode)
        # 5. 量化模式：当前仅支持一种量化模式pertokenhead
        # 6. 核间切分参数
        InterSplitCore = ['00', '01', '02', '10', '11', '12', '20', '21', '22']
        self.analyzer.define_parameter("inter-core-split", InterSplitCore)

        # 7. Cube核内切分参数
        self.analyzer.define_parameter("intra-core-mm1_L1", self.IntraSplitMM())
        self.analyzer.define_parameter("intra-core-mm1_L0", self.IntraSplitMM())
        self.analyzer.define_parameter("intra-core-mm2_L1", self.IntraSplitMM())
        self.analyzer.define_parameter("intra-core-mm2_L0", self.IntraSplitMM())

        # 8. Vec核内切分参数
        IntraSplitVec = ['00', '01', '02', '10', '11', '12', '20', '21', '22']
        self.analyzer.define_parameter("intra-core-Vec", IntraSplitVec)

        # 9. pa-tail-block
        pa_tail_block = ['0', '1', '2']
        self.analyzer.define_parameter("pa-tail-block", pa_tail_block)


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
        case_parameters['layout_query'] = params.get('layout_query')
        # 2. layout_key参数
        case_parameters['layout_key'] = params.get('layout_key')
        # 3. dtypes参数
        case_parameters['query_dtype'] = params.get('query', {}).get('dtype')
        case_parameters['key_dtype'] = params.get('key', {}).get('dtype')
        case_parameters['weights_dtype'] = params.get('weights', {}).get('dtype')
        # 4. sparsemode参数
        case_parameters['sparse_mode'] = params.get('sparse_mode')
        # 6. 核间切分参数
        case_parameters['inter-core-split'] = dimensions_state.get('inter-core-split')
        # 7. MM核内切分参数
        case_parameters['intra-core-mm1_L1'] = dimensions_state.get('intra-core-mm1_L1')
        case_parameters['intra-core-mm1_L0'] = dimensions_state.get('intra-core-mm1_L0')

        case_parameters['intra-core-mm2_L1'] = dimensions_state.get('intra-core-mm2_L1')
        case_parameters['intra-core-mm2_L0'] = dimensions_state.get('intra-core-mm2_L0')
        # 8. Vec核内切分参数
        case_parameters['intra-core-Vec'] = dimensions_state.get('intra-core-Vec')
        # 9. pa-tail-block
        case_parameters['pa-tail-block'] = dimensions_state.get('pa-tail-block')
        
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