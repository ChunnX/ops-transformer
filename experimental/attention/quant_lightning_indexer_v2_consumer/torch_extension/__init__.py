def __getattr__(name):
    if name in ("quant_lightning_indexer_v2_consumer_candidate", "quant_lightning_indexer_v2_consumer"):
        from . import quant_lightning_indexer_consumer as _m
        return getattr(_m, name)
    raise AttributeError(name)
