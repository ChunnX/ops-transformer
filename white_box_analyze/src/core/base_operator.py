from abc import ABC, abstractmethod
from typing import Dict, Any

class BaseOperator(ABC):
    def __init__(self, op_name):
        self.op_name = op_name

    @abstractmethod
    def simulate_whitebox_logic(self, aic_num: int, aiv_num: int, input_params: Dict) -> Dict[str, Any]:
        """
        核心逻辑：模拟C++算子内部的判断逻辑。
        输入：从文件解析出的 input_params (如 shape, dtype)
        输出：白盒维度状态
        """
        pass