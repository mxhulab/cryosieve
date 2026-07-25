import cupy as cp
from numpy.typing import ArrayLike

from . import ceil_div


_project_kernel = cp.RawKernel(r'''
extern "C" __global__ void project(
    const double* volume,
    const double* rotations,
    double* image,
    int n)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < n * n * n) {
        int k = tid / n / n;
        int j = tid / n % n;
        int i = tid % n;
        int z = k - n / 2;
        int y = j - n / 2;
        int x = i - n / 2;

        double xp = rotations[0] * x + rotations[1] * y + rotations[2] * z;
        double yp = rotations[3] * x + rotations[4] * y + rotations[5] * z;
        double xpi = xp + n / 2;
        double ypi = yp + n / 2;
        int ip = floor(xpi);
        int jp = floor(ypi);
        double dx = xpi - ip;
        double dy = ypi - jp;

        double voxel = volume[tid];
        if (0 <= ip     && ip     < n && 0 <= jp     && jp     < n) atomicAdd(image + (jp    ) * n + (ip    ), voxel * (1 - dx) * (1 - dy));
        if (0 <= ip + 1 && ip + 1 < n && 0 <= jp     && jp     < n) atomicAdd(image + (jp    ) * n + (ip + 1), voxel * (    dx) * (1 - dy));
        if (0 <= ip     && ip     < n && 0 <= jp + 1 && jp + 1 < n) atomicAdd(image + (jp + 1) * n + (ip    ), voxel * (1 - dx) * (    dy));
        if (0 <= ip + 1 && ip + 1 < n && 0 <= jp + 1 && jp + 1 < n) atomicAdd(image + (jp + 1) * n + (ip + 1), voxel * (    dx) * (    dy));
    }
}''', 'project')


def project(volume: ArrayLike, quaternions: ArrayLike) -> cp.ndarray:
    '''Project a volume at unit-quaternion rotations.

    Parameters
    ----------
    volume : ArrayLike
        Shape (n, n, n), dtype float64.
    quaternions : ArrayLike
        Shape (m, 4), dtype float64, in w-x-y-z order.

    Returns
    -------
    stack : cupy.ndarray
        Shape (m, n, n), dtype float64.
    '''
    volume = cp.asarray(volume, dtype=cp.float64)
    quaternions = cp.asarray(quaternions, dtype=cp.float64)
    n = volume.shape[0]
    m = quaternions.shape[0]
    assert volume.shape == (n, n, n) and quaternions.shape == (m, 4)

    qw = quaternions[:, 0]
    qx = quaternions[:, 1]
    qy = quaternions[:, 2]
    qz = quaternions[:, 3]
    rotations = cp.stack([
        1 - 2 * (qy * qy + qz * qz),
        2 * (qx * qy - qw * qz),
        2 * (qx * qz + qw * qy),
        2 * (qx * qy + qw * qz),
        1 - 2 * (qx * qx + qz * qz),
        2 * (qy * qz - qw * qx),
    ], axis=1)

    stack = cp.zeros((m, n, n), dtype=cp.float64)
    for b in range(m):
        _project_kernel(
            (ceil_div(volume.size, 128),),
            (128,),
            (volume, rotations[b], stack[b], n),
        )
    return stack
