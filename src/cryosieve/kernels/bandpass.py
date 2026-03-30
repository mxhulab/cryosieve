import cupy as cp
from cupy.fft import rfftn, irfftn
from numpy.typing import ArrayLike
from . import ceil_div

ker_bandpass2d = cp.RawKernel(r'''
extern "C" __global__ void bandpass2d(
    double* f_stack,
    int m,
    int n,
    double threshold_low,
    double threshold_high)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int n_ = n / 2 + 1;
    if (tid < m * n * n_) {
        int x = tid % n_;
        int y = tid / n_ % n; y = y < n_ ? y : y - n;

        double f = hypot((double)x / n, (double)y / n);
        if (f < threshold_low || f > threshold_high) {
            f_stack[tid * 2    ] = 0;
            f_stack[tid * 2 + 1] = 0;
        }
    }
}''', 'bandpass2d')

def bandpass2d(stack : ArrayLike, threshold_low : float, threshold_high : float) -> cp.ndarray:
    '''Bandpass filter

    Parameters
    ----------
    stack : ArrayLike
        shape (m, n, n), dtype float64
    threshold_low : float
        dimensionless, in range [0, 1]
    threshold_high : float
        dimensionless, in range [0, 1]

    Returns
    -------
    new_stack : cupy.ndarray
        shape (m, n, n), dtype float64
    '''
    stack = cp.asarray(stack, dtype = cp.float64)
    m = stack.shape[0]
    n = stack.shape[1]
    assert stack.shape == (m, n, n)

    f_stack = rfftn(stack, axes = (1, 2))
    ker_bandpass2d((ceil_div(f_stack.size, 256), ), (256, ), (f_stack, m, n, threshold_low, threshold_high))
    return irfftn(f_stack, axes = (1, 2))

def lowpass2d(stack : cp.ndarray, threshold : float) -> cp.ndarray:
    '''Lowpass filter

    Parameters
    ----------
    stack : cupy.ndarray
        shape (m, n, n), dtype float64
    threshold : float
        dimensionless, in range [0, 1]

    Returns
    -------
    new_stack : cupy.ndarray
        shape (m, n, n), dtype float64
    '''
    stack = cp.asarray(stack, dtype = cp.float64)
    m = stack.shape[0]
    n = stack.shape[1]
    assert stack.shape == (m, n, n)

    f_stack = rfftn(stack, axes = (1, 2))
    ker_bandpass2d((ceil_div(f_stack.size, 256), ), (256, ), (f_stack, m, n, 0., threshold))
    return irfftn(f_stack, axes = (1, 2))

def highpass2d(stack : cp.ndarray, threshold : float) -> cp.ndarray:
    '''Highpass filter

    Parameters
    ----------
    stack : cupy.ndarray
        shape (m, n, n), dtype float64
    threshold : float
        dimensionless, in range [0, 1]

    Returns
    -------
    new_stack : cupy.ndarray
        shape (m, n, n), dtype float64
    '''
    stack = cp.asarray(stack, dtype = cp.float64)
    m = stack.shape[0]
    n = stack.shape[1]
    assert stack.shape == (m, n, n)

    f_stack = rfftn(stack, axes = (1, 2))
    ker_bandpass2d((ceil_div(f_stack.size, 256), ), (256, ), (f_stack, m, n, threshold, 1.))
    return irfftn(f_stack, axes = (1, 2))
