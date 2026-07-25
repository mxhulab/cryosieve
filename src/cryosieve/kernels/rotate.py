import cupy as cp
from numpy.typing import ArrayLike

from . import ceil_div


_rotate_2d_kernel = cp.RawKernel(r'''
extern "C" __global__ void rotate_2d(
    const double* stack,
    int m,
    int n,
    const double* angles,
    double* rotated_stack)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < m * n * n) {
        int b = tid / n / n;
        int j = tid / n % n;
        int i = tid % n;
        int y = j - n / 2;
        int x = i - n / 2;

        double xp = x *  cos(angles[b]) + y * sin(angles[b]);
        double yp = x * -sin(angles[b]) + y * cos(angles[b]);
        double xpi = xp + n / 2;
        double ypi = yp + n / 2;
        int ip = floor(xpi);
        int jp = floor(ypi);
        double dx = xpi - ip;
        double dy = ypi - jp;

        if (0 <= ip     && ip     < n && 0 <= jp     && jp     < n) rotated_stack[tid] += stack[(b * n + jp    ) * n + ip    ] * (1 - dx) * (1 - dy);
        if (0 <= ip + 1 && ip + 1 < n && 0 <= jp     && jp     < n) rotated_stack[tid] += stack[(b * n + jp    ) * n + ip + 1] * (    dx) * (1 - dy);
        if (0 <= ip     && ip     < n && 0 <= jp + 1 && jp + 1 < n) rotated_stack[tid] += stack[(b * n + jp + 1) * n + ip    ] * (1 - dx) * (    dy);
        if (0 <= ip + 1 && ip + 1 < n && 0 <= jp + 1 && jp + 1 < n) rotated_stack[tid] += stack[(b * n + jp + 1) * n + ip + 1] * (    dx) * (    dy);
    }
}''', 'rotate_2d')


def rotate_2d(stack: ArrayLike, angles: ArrayLike) -> cp.ndarray:
    '''Rotate real 2D images counterclockwise.

    Parameters
    ----------
    stack : ArrayLike
        Shape (m, n, n), dtype float64.
    angles : ArrayLike
        Shape (m,), dtype float64, angles in radians equivalent to RELION psi
        Euler angles.

    Returns
    -------
    rotated_stack : cupy.ndarray
        Shape (m, n, n), dtype float64. Out-of-bounds samples contribute zero.
    '''
    stack = cp.asarray(stack, dtype=cp.float64)
    angles = cp.asarray(angles, dtype=cp.float64)
    m, n = stack.shape[:2]
    assert stack.shape == (m, n, n) and angles.shape == (m,)

    rotated_stack = cp.zeros((m, n, n), dtype=cp.float64)
    _rotate_2d_kernel(
        (ceil_div(stack.size, 256),),
        (256,),
        (stack, m, n, angles, rotated_stack),
    )
    return rotated_stack
