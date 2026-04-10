import cupy as cp
from cupy.fft import rfftn, irfftn
from numpy.typing import ArrayLike
from . import ceil_div

ker_translate = cp.RawKernel(r'''
extern "C" __global__ void translate(
    double* f_stack,
    int m,
    int n,
    const double* trans)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int n_ = n / 2 + 1;
    if (tid < m * n * n_) {
        int x = tid % n_;
        int y = tid / n_ % n; y = y < n_ ? y : y - n;
        int z = tid / n_ / n;

        double pi = 3.1415926535897932384626;
        double ax = f_stack[2 * tid    ];
        double ay = f_stack[2 * tid + 1];
        double tx = trans[2 * z    ];
        double ty = trans[2 * z + 1];
        double phi = -2 * pi * (tx * x / n + ty * y / n);
        f_stack[2 * tid    ] = ax * cos(phi) - ay * sin(phi);
        f_stack[2 * tid + 1] = ax * sin(phi) + ay * cos(phi);
    }
}''', 'translate')

def translate(stack : ArrayLike, trans : ArrayLike) -> cp.ndarray:
    '''In-plane translation (in Fourier space)

    Parameters
    ----------
    stack : ArrayLike
        shape (m, n, n), dtype float64
    trans : ArrayLike
        shape (m, 2), dtype float64

    Returns
    -------
    new_stack : cupy.ndarray
        shape (m, n, n), dtype float64
    '''
    stack = cp.asarray(stack, dtype = cp.float64)
    trans = cp.asarray(trans, dtype = cp.float64)
    m = stack.shape[0]
    n = stack.shape[1]
    assert stack.shape == (m, n, n) and trans.shape == (m, 2)

    f_stack = rfftn(stack, axes = (1, 2))
    ker_translate((ceil_div(f_stack.size, 256), ), (256, ), (f_stack, m, n, trans))
    return irfftn(f_stack, axes = (1, 2))
