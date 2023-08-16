try:
    import cupy as cp
except ModuleNotFoundError:
    print('[ERROR] CuPy module not found.')
    exit(1)
except ImportError:
    print('[ERROR] Error occured when importing CuPy. Please check your CUDA environment or GPU card.')
    exit(1)
