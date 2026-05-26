class_registry_func = {}


def class_registry(*names):
    def decorator(cls):
        for name in names:
            class_registry_func[name] = cls
        return cls

    return decorator
