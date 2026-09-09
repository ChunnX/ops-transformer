import threading
_kernel = None
_lock = threading.Lock()
def set_aicpu_kernel(kernel):
    global _kernel
    with _lock: _kernel = kernel
def get_aicpu_kernel():
    with _lock: return _kernel
