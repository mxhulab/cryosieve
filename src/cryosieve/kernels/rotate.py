import cupy as cp
from numpy.typing import ArrayLike
from . import ceil_div

ker_rotate2d = cp.RawKernel(r'''
extern "C" __global__ void rotate2d(
    const double* stack,
    int m,
    int n,
    const double* psi,
    double* new_stack)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < m * n * n) {
        int x = tid % n     - n / 2;
        int y = tid / n % n - n / 2;
        int z = tid / n / n;

        double vx = x *  cos(psi[z]) + y * sin(psi[z]) + n / 2;
        double vy = x * -sin(psi[z]) + y * cos(psi[z]) + n / 2;
        x = floor(vx);
        y = floor(vy);
        double dx = vx - x;
        double dy = vy - y;

        if (0 <= x     && x     < n && 0 <= y     && y     < n) new_stack[tid] += stack[(z * n + y    ) * n + x    ] * (1 - dx) * (1 - dy);
        if (0 <= x + 1 && x + 1 < n && 0 <= y     && y     < n) new_stack[tid] += stack[(z * n + y    ) * n + x + 1] * (    dx) * (1 - dy);
        if (0 <= x     && x     < n && 0 <= y + 1 && y + 1 < n) new_stack[tid] += stack[(z * n + y + 1) * n + x    ] * (1 - dx) * (    dy);
        if (0 <= x + 1 && x + 1 < n && 0 <= y + 1 && y + 1 < n) new_stack[tid] += stack[(z * n + y + 1) * n + x + 1] * (    dx) * (    dy);
    }
}''', 'rotate2d')

def rotate2d(stack : ArrayLike, angles : ArrayLike) -> cp.ndarray:
    '''Rotate 2D images (counter clock-wise) in real space

    Parameters
    ----------
    stack : ArrayLike
        shape (m, n, n), dtype float64
    angles : ArrayLike
        shape (m, ), dtype float64, angles in radians

    Returns
    -------
    new_stack : cupy.ndarray
        shape (m, n, n), dtype float64
    '''
    stack = cp.asarray(stack, dtype = cp.float64)
    angles = cp.asarray(angles, dtype = cp.float64)
    m = stack.shape[0]
    n = stack.shape[1]
    assert stack.shape == (m, n, n) and angles.shape == (m, )

    new_stack = cp.zeros((m, n, n), dtype = cp.float64)
    ker_rotate2d((ceil_div(stack.size, 256), ), (256, ), (stack, m, n, angles, new_stack))
    return new_stack
