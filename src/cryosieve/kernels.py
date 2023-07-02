__all__ = [
    'convolute_ctf',
    'highpass2d',
    'project',
    'translate'
]

import cupy as cp
from cupy.fft import rfftn, irfftn

BLOCKSIZE = 1024
BLOCKDIM = lambda x : (x - 1) // BLOCKSIZE + 1

def convolute_ctf(stack, ctfs, order = 1):
    ker_convolute_ctf = cp.RawKernel(r'''
extern "C" __global__ void convolute_ctf(
    double* data,
    int m,
    int n,
    const double* ctfs,
    int order)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int n_ = n / 2 + 1;
    if (tid < m * n * n_) {
        int x = tid % n_;
        int y = tid / n_ % n; y = y < n / 2 ? y : y - n;
        int z = tid / n_ / n;

        double voltage           = ctfs[z * 8    ];
        double defocusU          = ctfs[z * 8 + 1];
        double defocusV          = ctfs[z * 8 + 2];
        double astigmatism       = ctfs[z * 8 + 3];
        double Cs                = ctfs[z * 8 + 4];
        double amplitudeContrast = ctfs[z * 8 + 5];
        double phaseShift        = ctfs[z * 8 + 6];
        double pixelSize         = ctfs[z * 8 + 7];

        double pi = 3.1415926535897932384626;
        double waveLength = 12.2643247 / sqrt(voltage * (1 + voltage * 0.978466e-6));
        double f = hypot(x / (pixelSize * n), y / (pixelSize * n));
        double alpha = atan2((double)y, (double)x) - astigmatism;
        double defocus = -(defocusU + defocusV + (defocusU - defocusV) * cos(2 * alpha)) / 2;
        double chi = pi * waveLength * defocus * pow(f, 2.) + pi / 2 * Cs * pow(waveLength, 3.) * pow(f, 4.) - phaseShift;
        double u = pow(-sqrt(1 - pow(amplitudeContrast, 2.)) * sin(chi) + amplitudeContrast * cos(chi), (double)order);

        data[tid * 2    ] *= u;
        data[tid * 2 + 1] *= u;
    }
}''', 'convolute_ctf')

    assert isinstance(stack, cp.ndarray) and stack.ndim == 3 and stack.shape[1] == stack.shape[2]
    m, n, _ = stack.shape
    ctfs = cp.array(ctfs, dtype = cp.float64)
    assert ctfs.shape == (m, 8)

    f_stack = rfftn(stack, axes = (1, 2))
    ker_convolute_ctf((BLOCKDIM(f_stack.size), ), (BLOCKSIZE, ), (f_stack, m, n, ctfs, order))
    return irfftn(f_stack, axes = (1, 2))

def highpass2d(stack, threshold):
    ker_highpass2d = cp.RawKernel(r'''
extern "C" __global__ void highpass2d(
    double* data,
    int m,
    int n,
    double threshold)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int n_ = n / 2 + 1;
    if (tid < m * n * n_) {
        int x = tid % n_;
        int y = tid / n_ % n; y = y < n / 2 ? y : y - n;

        double f = hypot((double)x / n, (double)y / n);
        if (f < threshold) {
            data[tid * 2    ] = 0;
            data[tid * 2 + 1] = 0;
        }
    }
}''', 'highpass2d')

    assert isinstance(stack, cp.ndarray) and stack.ndim == 3 and stack.shape[1] == stack.shape[2]
    m, n, _ = stack.shape

    f_stack = rfftn(stack, axes = (1, 2))
    ker_highpass2d((BLOCKDIM(f_stack.size), ), (BLOCKSIZE, ), (f_stack, m, n, threshold))
    return irfftn(f_stack, axes = (1, 2))

def project(volume, quats):
    ker_project = cp.RawKernel(r'''
extern "C" __global__ void project(
    const double* vol,
    int n,
    double qw,
    double qx,
    double qy,
    double qz,
    double* img)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < n * n * n) {
        int x = tid % n     - n / 2;
        int y = tid / n % n - n / 2;
        int z = tid / n / n - n / 2;

        double vx = x - 2 * (qy * qy + qz * qz) * x + 2 * (qx * qy - qw * qz) * y + 2 * (qx * qz + qw * qy) * z + n / 2;
        double vy = y + 2 * (qx * qy + qw * qz) * x - 2 * (qx * qx + qz * qz) * y + 2 * (qy * qz - qw * qx) * z + n / 2;
        // double vz = z + 2 * (qx * qz - qw * qy) * x + 2 * (qy * qz + qw * qx) * y - 2 * (qx * qx + qy * qy) * z + n / 2;
        x = floor(vx);
        y = floor(vy);
        double dx = vx - x;
        double dy = vy - y;

        double voxel = vol[tid];
        if (0 <= x     && x     < n && 0 <= y     && y     < n) atomicAdd(img + (y    ) * n + (x    ), voxel * (1 - dx) * (1 - dy));
        if (0 <= x + 1 && x + 1 < n && 0 <= y     && y     < n) atomicAdd(img + (y    ) * n + (x + 1), voxel * (    dx) * (1 - dy));
        if (0 <= x     && x     < n && 0 <= y + 1 && y + 1 < n) atomicAdd(img + (y + 1) * n + (x    ), voxel * (1 - dx) * (    dy));
        if (0 <= x + 1 && x + 1 < n && 0 <= y + 1 && y + 1 < n) atomicAdd(img + (y + 1) * n + (x + 1), voxel * (    dx) * (    dy));
    }
}''', 'project')

    assert isinstance(volume, cp.ndarray) and volume.ndim == 3
    m, n = len(quats), volume.shape[0]
    assert volume.shape == (n, n, n)

    stack = cp.zeros((m, n, n), dtype = cp.float64)
    for i in range(m):
        qw, qx, qy, qz = quats[i]
        ker_project((BLOCKDIM(volume.size), ), (BLOCKSIZE, ), (volume, n, qw, qx, qy, qz, stack[i]))
    return stack

def translate(stack, trans):
    ker_translate = cp.RawKernel(r'''
extern "C" __global__ void translate(
    double* data,
    int m,
    int n,
    const double* trans)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int n_ = n / 2 + 1;
    if (tid < m * n * n_) {
        int x = tid % n_;
        int y = tid / n_ % n; y = y < n / 2 ? y : y - n;
        int z = tid / n_ / n;

        double pi = 3.1415926535897932384626;
        double ax = data[2 * tid    ];
        double ay = data[2 * tid + 1];
        double tx = trans[2 * z    ];
        double ty = trans[2 * z + 1];
        double phi = -2 * pi * (tx * x / n + ty * y / n);
        double cosphi = cos(phi);
        double sinphi = sin(phi);
        data[2 * tid    ] = ax * cosphi - ay * sinphi;
        data[2 * tid + 1] = ax * sinphi + ay * cosphi;
    }
}''', 'translate')

    assert isinstance(stack, cp.ndarray) and stack.ndim == 3 and stack.shape[1] == stack.shape[2]
    m, n, _ = stack.shape
    trans = cp.array(trans, dtype = cp.float64)
    assert trans.shape == (m, 2)

    f_stack = rfftn(stack, axes = (1, 2))
    ker_translate((BLOCKDIM(f_stack.size), ), (BLOCKSIZE, ), (f_stack, m, n, trans))
    return irfftn(f_stack, axes = (1, 2))
