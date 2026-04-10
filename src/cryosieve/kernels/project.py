import cupy as cp
from numpy.typing import ArrayLike
from . import ceil_div

ker_project = cp.RawKernel(r'''
extern "C" __global__ void project(
    const double* volume,
    const double* rots,
    double* image,
    int n)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < n * n * n) {
        int x = tid % n     - n / 2;
        int y = tid / n % n - n / 2;
        int z = tid / n / n - n / 2;

        double vx = rots[0] * x + rots[1] * y + rots[2] * z + n / 2;
        double vy = rots[3] * x + rots[4] * y + rots[5] * z + n / 2;
        x = floor(vx);
        y = floor(vy);
        double dx = vx - x;
        double dy = vy - y;

        double voxel = volume[tid];
        if (0 <= x     && x     < n && 0 <= y     && y     < n) atomicAdd(image + (y    ) * n + (x    ), voxel * (1 - dx) * (1 - dy));
        if (0 <= x + 1 && x + 1 < n && 0 <= y     && y     < n) atomicAdd(image + (y    ) * n + (x + 1), voxel * (    dx) * (1 - dy));
        if (0 <= x     && x     < n && 0 <= y + 1 && y + 1 < n) atomicAdd(image + (y + 1) * n + (x    ), voxel * (1 - dx) * (    dy));
        if (0 <= x + 1 && x + 1 < n && 0 <= y + 1 && y + 1 < n) atomicAdd(image + (y + 1) * n + (x + 1), voxel * (    dx) * (    dy));
    }
}''', 'project')

def project(volume : ArrayLike, quats : ArrayLike) -> cp.ndarray:
    '''Project along given spatial rotations (in unit quaternion description)

    Parameters
    ----------
    volume : ArrayLike
        shape (n, n, n), dtype float64
    quats : ArrayLike
        shape (m, 4), dtype float64

    Returns
    -------
    stack : cupy.ndarray
        shape (m, n, n), dtype float64
    '''
    volume = cp.asarray(volume, dtype = cp.float64)
    quats = cp.asarray(quats, dtype = cp.float64)
    n = volume.shape[0]
    m = quats.shape[0]
    assert volume.shape == (n, n, n) and quats.shape == (m, 4)

    w = quats[:, 0]
    x = quats[:, 1]
    y = quats[:, 2]
    z = quats[:, 3]
    rots = cp.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)
    ], axis = 1)

    stack = cp.zeros((m, n, n), dtype = cp.float64)
    for i in range(m):
        ker_project((ceil_div(volume.size, 128), ), (128, ), (volume, rots[i], stack[i], n))
    return stack
