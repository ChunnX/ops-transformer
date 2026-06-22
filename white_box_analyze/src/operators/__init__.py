from .mlaProlog_SplitCore import mlaPrologOperator
from .LI_SplitCore import LIOperator
from .QLI_SplitCore import QLIOperator
from .SFA_SplitCore import SFAOperator
from .QSFA_SplitCore import QSFAOperator

# 自动注册算子
__all__ = ['mlaPrologOperator', 'LIOperator', 'QLIOperator', 'SFAOperator', 'QSFAOperator']