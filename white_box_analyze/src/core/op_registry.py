class OperatorRegistry:
    _registry = {}

    @classmethod
    def register(cls, op_type):
        def decorator(subclass):
            cls._registry[op_type] = subclass
            return subclass
        return decorator

    @classmethod
    def get_operator(cls, op_type):
        if op_type not in cls._registry:
            raise ValueError(f"Unknown Operator type: {op_type}")
        return cls._registry[op_type]