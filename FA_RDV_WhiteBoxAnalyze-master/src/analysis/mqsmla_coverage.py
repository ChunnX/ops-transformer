from .coverage import CoverageAnalyzer
from typing import Dict, List, Any


class MQSMLACoverageAnalyzer:
    """MQSMLA算子的专用覆盖率分析器"""

    def __init__(self, chipversion: str = "A5"):
        self.analyzer = CoverageAnalyzer("MQSMLA")
        self.chipversion = chipversion
        self._define_parameters()

    def _define_parameters(self):
        """定义MQSMLA算子需要分析的参数"""
        layout_q = ["BSND", "TND"]
        self.analyzer.define_parameter("layout_q", layout_q)
        layout_kv = ["BSND", "TND", "PA_BBND"]
        self.analyzer.define_parameter("layout_kv", layout_kv)

        dtypes_query = ["bf16"]
        self.analyzer.define_parameter("query_dtype", dtypes_query)
        dtypes_kv = ["float8_e4m3fn"]
        self.analyzer.define_parameter("ori_kv_dtype", dtypes_kv)
        self.analyzer.define_parameter("cmp_kv_dtype", dtypes_kv)

        template_mode = ['0', '1', '2']
        self.analyzer.define_parameter("template_mode", template_mode)

        split_g = ['0', '1']
        self.analyzer.define_parameter("split_g", split_g)

        quant_mode = [1, 2]
        self.analyzer.define_parameter("quant_mode", quant_mode)

        cmp_ratio = [1, 4, 128]
        self.analyzer.define_parameter("cmp_ratio", cmp_ratio)

        PA_ori_block = ['0', '1', '2']
        self.analyzer.define_parameter("PA_ori_block", PA_ori_block)
        PA_cmp_block = ['0', '1', '2']
        self.analyzer.define_parameter("PA_cmp_block", PA_cmp_block)

        self._define_parameters_A5()

    def _define_parameters_A5(self):
        inter_core_split = ['00', '01', '02']
        self.analyzer.define_parameter("inter-core-split", inter_core_split)

        intra_core_vec0 = ['00', '01', '02']
        self.analyzer.define_parameter("intra-core-vec0", intra_core_vec0)

        intra_core_mm1_L1 = ['000', '010', '020']
        self.analyzer.define_parameter("intra-core-mm1_L1", intra_core_mm1_L1)

        intra_core_mm1_L0 = ['002', '012', '022']
        self.analyzer.define_parameter("intra-core-mm1_L0", intra_core_mm1_L0)

        intra_core_vec1 = ['000', '011', '012', '020', '023', '024']
        self.analyzer.define_parameter("intra-core-vec1", intra_core_vec1)

        intra_core_mm2_L1 = ['000', '001', '002', '010', '011', '012', '020', '021', '022']
        self.analyzer.define_parameter("intra-core-mm2_L1", intra_core_mm2_L1)

        intra_core_mm2_L0 = ['000', '001', '002', '010', '011', '012', '020', '021', '022']
        self.analyzer.define_parameter("intra-core-mm2_L0", intra_core_mm2_L0)

        intra_core_vec2 = ['00', '01', '02']
        self.analyzer.define_parameter("intra-core-vec2", intra_core_vec2)

    def add_case(self, case_name: str, dimensions_state: Dict[str, Any], params) -> None:
        """添加MQSMLA用例"""
        case_parameters = {}
        case_parameters['layout_q'] = params.get('layout_q')
        case_parameters['layout_kv'] = params.get('layout_kv')
        case_parameters['query_dtype'] = params.get('query', {}).get('dtype')
        case_parameters['ori_kv_dtype'] = params.get('ori_kv', {}).get('dtype')
        case_parameters['cmp_kv_dtype'] = params.get('cmp_kv', {}).get('dtype')
        case_parameters['template_mode'] = dimensions_state.get('stage_codes', {}).get('template_mode')
        case_parameters['split_g'] = dimensions_state.get('stage_codes', {}).get('split_g')
        case_parameters['quant_mode'] = params.get('quant_mode')
        case_parameters['cmp_ratio'] = params.get('cmp_ratio')
        case_parameters['PA_ori_block'] = dimensions_state.get('stage_codes', {}).get('PA_ori_block')
        case_parameters['PA_cmp_block'] = dimensions_state.get('stage_codes', {}).get('PA_cmp_block')
        case_parameters['inter-core-split'] = dimensions_state.get('stage_codes', {}).get('inter-core-split')
        case_parameters['intra-core-vec0'] = dimensions_state.get('stage_codes', {}).get('intra-core-vec0')
        case_parameters['intra-core-mm1_L1'] = dimensions_state.get('stage_codes', {}).get('intra-core-mm1_L1')
        case_parameters['intra-core-mm1_L0'] = dimensions_state.get('stage_codes', {}).get('intra-core-mm1_L0')
        case_parameters['intra-core-vec1'] = dimensions_state.get('stage_codes', {}).get('intra-core-vec1')
        case_parameters['intra-core-mm2_L1'] = dimensions_state.get('stage_codes', {}).get('intra-core-mm2_L1')
        case_parameters['intra-core-mm2_L0'] = dimensions_state.get('stage_codes', {}).get('intra-core-mm2_L0')
        case_parameters['intra-core-vec2'] = dimensions_state.get('stage_codes', {}).get('intra-core-vec2')

        self.analyzer.add_case(case_name, case_parameters)

    def generate_report(self, base_output_dir: str = ".") -> str:
        return self.analyzer.generate_excel_report(base_output_dir)

    def get_parameter_coverage(self, param_name: str) -> Dict[str, Any]:
        return self.analyzer.get_parameter_coverage(param_name)

    def get_all_coverages(self) -> Dict[str, Dict[str, Any]]:
        return self.analyzer.get_all_coverages()

    def get_overall_coverage(self) -> Dict[str, Any]:
        return self.analyzer.get_overall_coverage()
