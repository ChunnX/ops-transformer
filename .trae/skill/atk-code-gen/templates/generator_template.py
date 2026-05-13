from atk.case_generator.generator.generate_types import GENERATOR_REGISTRY
from atk.case_generator.generator.base_generator import CaseGenerator
from atk.configs.case_config import CaseConfig
import os

rank_size = int(os.getenv("RANK_SIZE", 2))

@GENERATOR_REGISTRY.register("generate_{{GENERATOR_NAME}}")
class {{GENERATOR_CLASS_NAME}}(CaseGenerator):

    def after_case_config(self, case_config: CaseConfig) -> CaseConfig:
        '''
        用例参数约束修改入口
        :param case_config:  生成的用例信息，可能不满足参数间约束，导致用例无效
        :return: 返回修改后符合参数间约束关系的用例，需要用例保障用例有效
        根据文档对各个位置上的参数进行修改
        '''
        # M必须要为rank_size的倍数
        case_config.inputs[0].shape[0] = int(case_config.inputs[0].shape[0] / rank_size) * rank_size
        case_config.inputs[0].shape[0] = max(case_config.inputs[0].shape[0], rank_size)

        # 每个inputs包含以下属性（仅列出需要修改的）
        # dtype、shape、range_values
        # 注意！没有value这个属性！

        {{PARAMETER_CONSTRAINTS}}

        return case_config
