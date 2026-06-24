# 用例json文件的注册器
class ParserRegistry:
    _registry = {}

    @classmethod
    def register(cls, op_type):
        def decorator(subclass):
            cls._registry[op_type] = subclass
            return subclass
        return decorator

    @classmethod
    def get_parser(cls, op_type):
        if op_type not in cls._registry:
            raise ValueError(f"Unknown parser type: {op_type}")
        return cls._registry[op_type]