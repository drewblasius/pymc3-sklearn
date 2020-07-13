import os

def set_mathlib_threads(max_threads: int):
    if not isinstance(max_threads, int) or max_threads <= 0:
        raise ValueError(f"`max_threads` must be a positive integer (got {max_threads}).")
    
    max_threads = str(max_threads)
    os.environ["OMP_NUM_THREADS"] = max_threads
    os.environ["OPENBLAS_NUM_THREADS"] = max_threads
    os.environ["MKL_NUM_THREADS"] = max_threads
