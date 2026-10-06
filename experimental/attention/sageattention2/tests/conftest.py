import os


def pytest_sessionfinish(session, exitstatus):
    """Safely terminate process to avoid torch_npu driver teardown SIGSEGV."""
    os._exit(exitstatus)
