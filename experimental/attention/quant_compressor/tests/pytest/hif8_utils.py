import numpy as np

HIF8_POSITIVE_ZERO = 0x00
HIF8_NAN = 0x80
HIF8_POSITIVE_INF = 0x6F
HIF8_NEGATIVE_INF = 0xEF
HIF8_MAX_POSITIVE_NORMAL = 0x6E
HIF8_MAX_NEGATIVE_NORMAL = 0xEE
HIF8_OVERFLOW_THRESHOLD = (2.0 ** 15) * 1.25

def _as_numpy_uint8_codes(codes):
    arr = np.asarray(codes)
    if arr.dtype == np.bool_ or np.issubdtype() 