import os


def configure_runtime_env():
    os.environ.setdefault("MNE_USE_NUMBA", "false")
    os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")
    os.makedirs(os.environ["NUMBA_CACHE_DIR"], exist_ok=True)
