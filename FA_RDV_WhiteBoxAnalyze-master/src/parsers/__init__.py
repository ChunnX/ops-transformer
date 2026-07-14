from .mlaProlog_parser import MLAPrologParser
from .LI_parser import LIParser
from .QLI_parser import QLIParser
from .SFA_parser import SFAParser
from .QSFA_parser import QSFAParser
from .MQSMLA_parser import MQSMLAParser

# 自动注册算子
__all__ = ['MLAPrologParser', 'LIParser', 'QLIParser', 'SFAParser', 'QSFAParser', 'MQSMLAParser']