import cupy as cp
from numpy.typing import ArrayLike
from typing import Tuple

from . import ceil_div


_backproject_ft_kernel = cp.RawKernel(r'''
extern "C" __global__ void backproject_ft(
    double* volume_ft,
    double* weight_ft,
    const double* stack_ft,
    const double* weights,
    const double* rotations,
    int m,
    int n,
    int nh,
    int p,
    int ph,
    long long size)
{
    long long tid = blockDim.x * (long long)blockIdx.x + threadIdx.x;
    if (tid >= size) return;

    int b = tid / ((long long)n * nh);
    int j = tid / nh % n;
    int i = tid % nh;

    int y = j < nh ? j : j - n;
    int x = i;
    if ((x == 0 && y < 0) || x * x + y * y >= (n / 2) * (n / 2)) return;

    const double* rotation = rotations + 9 * b;
    double xp = 2.0 * (rotation[0] * x + rotation[3] * y);
    double yp = 2.0 * (rotation[1] * x + rotation[4] * y);
    double zp = 2.0 * (rotation[2] * x + rotation[5] * y);
    double real = stack_ft[2 * tid];
    double imag = stack_ft[2 * tid + 1];
    if (xp < 0.0) {
        xp = -xp;
        yp = -yp;
        zp = -zp;
        imag = -imag;
    }

    int ip = floor(xp);
    int jp = floor(yp);
    int kp = floor(zp);

    double dx = xp - ip;
    double dy = yp - jp;
    double dz = zp - kp;

    int ip1 = ip + 1;
    int jp1 = jp + 1;
    int kp1 = kp + 1;

    if (jp < 0) jp += p;
    if (jp1 < 0) jp1 += p;
    if (kp < 0) kp += p;
    if (kp1 < 0) kp1 += p;
    if (kp < 0 || kp1 >= p || jp < 0 || jp1 >= p || ip1 >= ph) return;

    double mdx = 1.0 - dx;
    double mdy = 1.0 - dy;
    double mdz = 1.0 - dz;

    double w000 = mdz * mdy * mdx;
    double w001 = mdz * mdy *  dx;
    double w010 = mdz *  dy * mdx;
    double w011 = mdz *  dy *  dx;
    double w100 =  dz * mdy * mdx;
    double w101 =  dz * mdy *  dx;
    double w110 =  dz *  dy * mdx;
    double w111 =  dz *  dy *  dx;

    long long t000 = ((long long)kp  * p + jp ) * ph + ip;
    long long t001 = ((long long)kp  * p + jp ) * ph + ip1;
    long long t010 = ((long long)kp  * p + jp1) * ph + ip;
    long long t011 = ((long long)kp  * p + jp1) * ph + ip1;
    long long t100 = ((long long)kp1 * p + jp ) * ph + ip;
    long long t101 = ((long long)kp1 * p + jp ) * ph + ip1;
    long long t110 = ((long long)kp1 * p + jp1) * ph + ip;
    long long t111 = ((long long)kp1 * p + jp1) * ph + ip1;

    atomicAdd(volume_ft + 2 * t000,     w000 * real);
    atomicAdd(volume_ft + 2 * t000 + 1, w000 * imag);
    atomicAdd(volume_ft + 2 * t001,     w001 * real);
    atomicAdd(volume_ft + 2 * t001 + 1, w001 * imag);
    atomicAdd(volume_ft + 2 * t010,     w010 * real);
    atomicAdd(volume_ft + 2 * t010 + 1, w010 * imag);
    atomicAdd(volume_ft + 2 * t011,     w011 * real);
    atomicAdd(volume_ft + 2 * t011 + 1, w011 * imag);
    atomicAdd(volume_ft + 2 * t100,     w100 * real);
    atomicAdd(volume_ft + 2 * t100 + 1, w100 * imag);
    atomicAdd(volume_ft + 2 * t101,     w101 * real);
    atomicAdd(volume_ft + 2 * t101 + 1, w101 * imag);
    atomicAdd(volume_ft + 2 * t110,     w110 * real);
    atomicAdd(volume_ft + 2 * t110 + 1, w110 * imag);
    atomicAdd(volume_ft + 2 * t111,     w111 * real);
    atomicAdd(volume_ft + 2 * t111 + 1, w111 * imag);

    double weight = weights[tid];
    atomicAdd(weight_ft + t000, w000 * weight);
    atomicAdd(weight_ft + t001, w001 * weight);
    atomicAdd(weight_ft + t010, w010 * weight);
    atomicAdd(weight_ft + t011, w011 * weight);
    atomicAdd(weight_ft + t100, w100 * weight);
    atomicAdd(weight_ft + t101, w101 * weight);
    atomicAdd(weight_ft + t110, w110 * weight);
    atomicAdd(weight_ft + t111, w111 * weight);
}''', 'backproject_ft')


_enforce_hermitian_ft_kernel = cp.RawKernel(r'''
extern "C" __global__ void enforce_hermitian_ft(
    double* volume_ft,
    double* weight_ft,
    int p,
    int ph)
{
    long long tid = blockDim.x * (long long)blockIdx.x + threadIdx.x;
    if (tid >= (long long)p * ph) return;

    int k = tid / ph;
    int j = tid % ph;

    int z = k < ph ? k : k - p;
    int y = j;
    if (y == 0 && z <= 0) return;

    int kp = -z >= 0 ? -z : -z + p;
    int jp = -y >= 0 ? -y : -y + p;
    long long a = ((long long)k * p + j) * ph;
    long long b = ((long long)kp * p + jp) * ph;

    double real = volume_ft[2 * a] + volume_ft[2 * b];
    double imag = volume_ft[2 * a + 1] - volume_ft[2 * b + 1];
    volume_ft[2 * a] = real;
    volume_ft[2 * a + 1] = imag;
    volume_ft[2 * b] = real;
    volume_ft[2 * b + 1] = -imag;

    double weight = weight_ft[a] + weight_ft[b];
    weight_ft[a] = weight;
    weight_ft[b] = weight;
}''', 'enforce_hermitian_ft')


_symmetrize_complex_ft_kernel = cp.RawKernel(r'''
extern "C" __global__ void symmetrize_complex_ft(
    const double* input,
    double* output,
    const double* rotations,
    int nsym,
    int p,
    int ph,
    long long size)
{
    long long tid = blockDim.x * (long long)blockIdx.x + threadIdx.x;
    if (tid >= size) return;

    int k = tid / ((long long)p * ph);
    int j = tid / ph % p;
    int i = tid % ph;

    int z = k < ph ? k : k - p;
    int y = j < ph ? j : j - p;
    int x = i;

    int max_radius = p / 2 - 1;
    if (x * x + y * y + z * z > max_radius * max_radius) return;

    double sum_real = output[2 * tid];
    double sum_imag = output[2 * tid + 1];
    for (int s = 1; s < nsym; ++s) {
        const double* rotation = rotations + 9 * s;
        double xp = rotation[0] * x + rotation[3] * y + rotation[6] * z;
        double yp = rotation[1] * x + rotation[4] * y + rotation[7] * z;
        double zp = rotation[2] * x + rotation[5] * y + rotation[8] * z;

        bool conjugate = xp < 0.0;
        if (conjugate) {
            xp = -xp;
            yp = -yp;
            zp = -zp;
        }
        int ip = floor(xp);
        int jp = floor(yp);
        int kp = floor(zp);

        double dx = xp - ip;
        double dy = yp - jp;
        double dz = zp - kp;

        int ip1 = ip + 1;
        int jp1 = jp + 1;
        int kp1 = kp + 1;

        if (jp < 0) jp += p;
        if (jp1 < 0) jp1 += p;
        if (kp < 0) kp += p;
        if (kp1 < 0) kp1 += p;
        if (kp < 0 || kp1 >= p || jp < 0 || jp1 >= p || ip1 >= ph) continue;

        long long s000 = ((long long)kp  * p + jp ) * ph + ip;
        long long s001 = ((long long)kp  * p + jp ) * ph + ip1;
        long long s010 = ((long long)kp  * p + jp1) * ph + ip;
        long long s011 = ((long long)kp  * p + jp1) * ph + ip1;
        long long s100 = ((long long)kp1 * p + jp ) * ph + ip;
        long long s101 = ((long long)kp1 * p + jp ) * ph + ip1;
        long long s110 = ((long long)kp1 * p + jp1) * ph + ip;
        long long s111 = ((long long)kp1 * p + jp1) * ph + ip1;

        double rz00 = (1.0 - dz) * input[2 * s000] + dz * input[2 * s100];
        double rz01 = (1.0 - dz) * input[2 * s001] + dz * input[2 * s101];
        double rz10 = (1.0 - dz) * input[2 * s010] + dz * input[2 * s110];
        double rz11 = (1.0 - dz) * input[2 * s011] + dz * input[2 * s111];
        double rzy0 = (1.0 - dy) * rz00 + dy * rz10;
        double rzy1 = (1.0 - dy) * rz01 + dy * rz11;
        double value_real = (1.0 - dx) * rzy0 + dx * rzy1;

        double iz00 = (1.0 - dz) * input[2 * s000 + 1] + dz * input[2 * s100 + 1];
        double iz01 = (1.0 - dz) * input[2 * s001 + 1] + dz * input[2 * s101 + 1];
        double iz10 = (1.0 - dz) * input[2 * s010 + 1] + dz * input[2 * s110 + 1];
        double iz11 = (1.0 - dz) * input[2 * s011 + 1] + dz * input[2 * s111 + 1];
        double izy0 = (1.0 - dy) * iz00 + dy * iz10;
        double izy1 = (1.0 - dy) * iz01 + dy * iz11;
        double value_imag = (1.0 - dx) * izy0 + dx * izy1;

        sum_real += value_real;
        sum_imag += conjugate ? -value_imag : value_imag;
    }

    output[2 * tid] = sum_real;
    output[2 * tid + 1] = sum_imag;
}''', 'symmetrize_complex_ft')


_symmetrize_real_ft_kernel = cp.RawKernel(r'''
extern "C" __global__ void symmetrize_real_ft(
    const double* input,
    double* output,
    const double* rotations,
    int nsym,
    int p,
    int ph,
    long long size)
{
    long long tid = blockDim.x * (long long)blockIdx.x + threadIdx.x;
    if (tid >= size) return;

    int k = tid / ((long long)p * ph);
    int j = tid / ph % p;
    int i = tid % ph;

    int z = k < ph ? k : k - p;
    int y = j < ph ? j : j - p;
    int x = i;

    int max_radius = p / 2 - 1;
    if (x * x + y * y + z * z > max_radius * max_radius) return;

    double sum = output[tid];
    for (int s = 1; s < nsym; ++s) {
        const double* rotation = rotations + 9 * s;
        double xp = rotation[0] * x + rotation[3] * y + rotation[6] * z;
        double yp = rotation[1] * x + rotation[4] * y + rotation[7] * z;
        double zp = rotation[2] * x + rotation[5] * y + rotation[8] * z;

        if (xp < 0.0) {
            xp = -xp;
            yp = -yp;
            zp = -zp;
        }
        int ip = floor(xp);
        int jp = floor(yp);
        int kp = floor(zp);

        double dx = xp - ip;
        double dy = yp - jp;
        double dz = zp - kp;

        int ip1 = ip + 1;
        int jp1 = jp + 1;
        int kp1 = kp + 1;

        if (jp < 0) jp += p;
        if (jp1 < 0) jp1 += p;
        if (kp < 0) kp += p;
        if (kp1 < 0) kp1 += p;
        if (kp < 0 || kp1 >= p || jp < 0 || jp1 >= p || ip1 >= ph) continue;

        long long s000 = ((long long)kp  * p + jp ) * ph + ip;
        long long s001 = ((long long)kp  * p + jp ) * ph + ip1;
        long long s010 = ((long long)kp  * p + jp1) * ph + ip;
        long long s011 = ((long long)kp  * p + jp1) * ph + ip1;
        long long s100 = ((long long)kp1 * p + jp ) * ph + ip;
        long long s101 = ((long long)kp1 * p + jp ) * ph + ip1;
        long long s110 = ((long long)kp1 * p + jp1) * ph + ip;
        long long s111 = ((long long)kp1 * p + jp1) * ph + ip1;

        double vz00 = (1.0 - dz) * input[s000] + dz * input[s100];
        double vz01 = (1.0 - dz) * input[s001] + dz * input[s101];
        double vz10 = (1.0 - dz) * input[s010] + dz * input[s110];
        double vz11 = (1.0 - dz) * input[s011] + dz * input[s111];
        double vzy0 = (1.0 - dy) * vz00 + dy * vz10;
        double vzy1 = (1.0 - dy) * vz01 + dy * vz11;
        double vzyx = (1.0 - dx) * vzy0 + dx * vzy1;

        sum += vzyx;
    }

    output[tid] = sum;
}''', 'symmetrize_real_ft')


_cut_sphere_ft_kernel = cp.RawKernel(r'''
extern "C" __global__ void cut_sphere_ft(
    double* data,
    int complex_data,
    int p,
    int ph,
    long long size)
{
    long long tid = blockDim.x * (long long)blockIdx.x + threadIdx.x;
    if (tid >= size) return;

    int k = tid / ((long long)p * ph);
    int j = tid / ph % p;
    int i = tid % ph;

    int z = k < ph ? k : k - p;
    int y = j < ph ? j : j - p;
    int x = i;

    int radius = p / 2 - 1;
    if (x * x + y * y + z * z > radius * radius) {
        data[(1 + complex_data) * tid] = 0.0;
        if (complex_data) data[2 * tid + 1] = 0.0;
    }
}''', 'cut_sphere_ft')


_weight_shells_ft_kernel = cp.RawKernel(r'''
extern "C" __global__ void weight_shells_ft(
    const double* weights,
    double* sums,
    unsigned long long* counts,
    int p,
    int ph,
    int shells,
    long long size)
{
    long long tid = blockDim.x * (long long)blockIdx.x + threadIdx.x;
    if (tid >= size) return;

    int k = tid / ((long long)p * ph);
    int j = tid / ph % p;
    int i = tid % ph;

    int z = k < ph ? k : k - p;
    int y = j < ph ? j : j - p;
    int x = i;

    int shell = floor(sqrt(
        (double)z * z + (double)y * y + (double)x * x) / 2.0);
    if (shell < shells) {
        atomicAdd(sums + shell, weights[tid]);
        atomicAdd(counts + shell, 1ULL);
    }
}''', 'weight_shells_ft')


_divide_ft_kernel = cp.RawKernel(r'''
extern "C" __global__ void divide_ft(
    double* volume,
    const double* weights,
    const double* floors,
    int p,
    int ph,
    int shells,
    long long size)
{
    long long tid = blockDim.x * (long long)blockIdx.x + threadIdx.x;
    if (tid >= size) return;

    int k = tid / ((long long)p * ph);
    int j = tid / ph % p;
    int i = tid % ph;

    int z = k < ph ? k : k - p;
    int y = j < ph ? j : j - p;
    int x = i;

    int shell = floor(sqrt(
        (double)z * z + (double)y * y + (double)x * x) / 2.0);
    if (shell >= shells) shell = shells - 1;

    double weight = fmax(weights[tid], floors[shell]);
    volume[2 * tid] /= weight;
    volume[2 * tid + 1] /= weight;
}''', 'divide_ft')


_initialize_gridding_weight_ft_kernel = cp.RawKernel(r'''
extern "C" __global__ void initialize_gridding_weight_ft(
    double* output,
    int p,
    int ph,
    long long size)
{
    long long tid = blockDim.x * (long long)blockIdx.x + threadIdx.x;
    if (tid >= size) return;

    int k = tid / ((long long)p * ph);
    int j = tid / ph % p;
    int i = tid % ph;

    int z = k < ph ? k : k - p;
    int y = j < ph ? j : j - p;
    int x = i;

    int radius = (p - 3) / 2;
    output[tid] = x * x + y * y + z * z < radius * radius ? 1.0 : 0.0;
}''', 'initialize_gridding_weight_ft')


_multiply_ftblob_3d_kernel = cp.RawKernel(r'''
extern "C" __global__ void multiply_ftblob_3d(
    double* volume,
    const double* table,
    int table_size,
    int p,
    int ph,
    long long size)
{
    long long tid = blockDim.x * (long long)blockIdx.x + threadIdx.x;
    if (tid >= size) return;

    int k = tid / ((long long)p * p);
    int j = tid / p % p;
    int i = tid % p;

    int z = k < ph ? k : k - p;
    int y = j < ph ? j : j - p;
    int x = i < ph ? i : i - p;

    int n = (p - 3) / 2;
    double radius = sqrt((double)(z * z + y * y + x * x)) / (2.0 * n);
    int index = radius / (0.5 / table_size);
    volume[tid] = index < table_size ? volume[tid] * table[index] / table[0] : 0.0;
}''', 'multiply_ftblob_3d')


_update_gridding_weight_ft_kernel = cp.RawKernel(r'''
extern "C" __global__ void update_gridding_weight_ft(
    double* weight,
    const double* convolved,
    int p,
    int ph,
    long long size)
{
    long long tid = blockDim.x * (long long)blockIdx.x + threadIdx.x;
    if (tid >= size) return;

    int k = tid / ((long long)p * ph);
    int j = tid / ph % p;
    int i = tid % ph;

    int z = k < ph ? k : k - p;
    int y = j < ph ? j : j - p;
    int x = i;

    int radius = (p - 3) / 2;
    if (x * x + y * y + z * z < radius * radius) {
        double magnitude = hypot(convolved[2 * tid], convolved[2 * tid + 1]);
        weight[tid] /= fmax(1e-6, magnitude);
    }
}''', 'update_gridding_weight_ft')


_window_ft_kernel = cp.RawKernel(r'''
extern "C" __global__ void window_ft(
    const double* input,
    double* output,
    int p,
    int ph,
    int n,
    int nh,
    long long size)
{
    long long tid = blockDim.x * (long long)blockIdx.x + threadIdx.x;
    if (tid >= size) return;

    int k = tid / ((long long)n * nh);
    int j = tid / nh % n;
    int i = tid % nh;

    int z = k < nh ? k : k - n;
    int y = j < nh ? j : j - n;

    int kp = z >= 0 ? z : z + p;
    int jp = y >= 0 ? y : y + p;
    long long source = ((long long)kp * p + jp) * ph + i;

    output[2 * tid] = input[2 * source];
    output[2 * tid + 1] = input[2 * source + 1];
}''', 'window_ft')


_translate_center_3d_ft_kernel = cp.RawKernel(r'''
extern "C" __global__ void translate_center_3d_ft(
    double* volume,
    int n,
    int nh,
    long long size)
{
    long long tid = blockDim.x * (long long)blockIdx.x + threadIdx.x;
    if (tid >= size) return;

    int k = tid / ((long long)n * nh);
    int j = tid / nh % n;
    int i = tid % nh;

    if ((k + j + i) & 1) {
        volume[2 * tid] = -volume[2 * tid];
        volume[2 * tid + 1] = -volume[2 * tid + 1];
    }
}''', 'translate_center_3d_ft')


_soft_mask_background_3d_kernel = cp.RawKernel(r'''
extern "C" __global__ void soft_mask_background_3d(
    const double* volume,
    double* statistics,
    double radius,
    double width,
    int n,
    long long size)
{
    long long tid = blockDim.x * (long long)blockIdx.x + threadIdx.x;
    if (tid >= size) return;

    int k = tid / ((long long)n * n);
    int j = tid / n % n;
    int i = tid % n;

    double z = k - n / 2;
    double y = j - n / 2;
    double x = i - n / 2;
    double r = sqrt(z * z + y * y + x * x);

    if (r < radius) return;

    double radius_outer = radius + width;
    double weight = r > radius_outer
        ? 1.0
        : 0.5 + 0.5 * cos(
            3.14159265358979323846 * (radius_outer - r) / width);

    atomicAdd(statistics, weight * volume[tid]);
    atomicAdd(statistics + 1, weight);
}''', 'soft_mask_background_3d')


_soft_mask_outside_3d_kernel = cp.RawKernel(r'''
extern "C" __global__ void soft_mask_outside_3d(
    double* volume,
    const double* statistics,
    double radius,
    double width,
    int n,
    long long size)
{
    long long tid = blockDim.x * (long long)blockIdx.x + threadIdx.x;
    if (tid >= size) return;

    int k = tid / ((long long)n * n);
    int j = tid / n % n;
    int i = tid % n;

    double z = k - n / 2;
    double y = j - n / 2;
    double x = i - n / 2;
    double r = sqrt(z * z + y * y + x * x);

    double radius_outer = radius + width;
    double background = statistics[0] / statistics[1];
    if (r > radius_outer) {
        volume[tid] = background;
    } else if (r >= radius) {
        double blend = 0.5 + 0.5 * cos(
            3.14159265358979323846 * (radius_outer - r) / width);
        volume[tid] = (1.0 - blend) * volume[tid] + blend * background;
    }
}''', 'soft_mask_outside_3d')


_gridding_correct_3d_kernel = cp.RawKernel(r'''
extern "C" __global__ void gridding_correct_3d(
    double* volume,
    int n,
    long long size)
{
    long long tid = blockDim.x * (long long)blockIdx.x + threadIdx.x;
    if (tid >= size) return;

    int k = tid / ((long long)n * n);
    int j = tid / n % n;
    int i = tid % n;

    double z = k - n / 2;
    double y = j - n / 2;
    double x = i - n / 2;

    double u = sqrt(z * z + y * y + x * x) / (2.0 * n);
    double sinc = u == 0.0
        ? 1.0
        : sin(3.14159265358979323846 * u)
            / (3.14159265358979323846 * u);
    volume[tid] /= sinc * sinc;
}''', 'gridding_correct_3d')


def backproject_ft(
    volume_ft: cp.ndarray,
    volume_weight_ft: cp.ndarray,
    stack_ft: cp.ndarray,
    weights: ArrayLike,
    rotations: ArrayLike,
) -> Tuple[cp.ndarray, cp.ndarray]:
    '''Insert only the independent positive half of 2D Fourier slices.

    Parameters
    ----------
    volume_ft : cupy.ndarray
        Shape (p, p, p // 2 + 1), dtype complex128. Modified in place.
    volume_weight_ft : cupy.ndarray
        Same shape, dtype float64. Modified in place.
    stack_ft : cupy.ndarray
        Shape (m, n, n // 2 + 1), dtype complex128, rfftn half-spectra over
        axes (1, 2).
    weights : ArrayLike
        Same shape as stack_ft, dtype float64.
    rotations : ArrayLike
        Shape (m, 3, 3), dtype float64.

    Returns
    -------
    volume_ft, volume_weight_ft : tuple of cupy.ndarray
        The modified input accumulators.

    Notes
    -----
    After all calls complete, callers must invoke ``enforce_hermitian_ft``
    exactly once before symmetry, normalization, or inverse transformation.
    '''
    m, n, nh = stack_ft.shape
    p, _, ph = volume_ft.shape
    assert volume_ft.dtype == cp.complex128 and volume_weight_ft.dtype == cp.float64
    weights = cp.asarray(weights, dtype=cp.float64)
    rotations = cp.asarray(rotations, dtype=cp.float64)
    _backproject_ft_kernel(
        (ceil_div(stack_ft.size, 256),),
        (256,),
        (volume_ft, volume_weight_ft, stack_ft, weights, rotations,
         m, n, nh, p, ph, stack_ft.size),
    )
    return volume_ft, volume_weight_ft


def enforce_hermitian_ft(
    volume_ft: cp.ndarray,
    volume_weight_ft: cp.ndarray,
) -> Tuple[cp.ndarray, cp.ndarray]:
    '''Complete the x=0 Hermitian plane in place.

    Parameters
    ----------
    volume_ft : cupy.ndarray
        Shape (p, p, p // 2 + 1), dtype complex128.
    volume_weight_ft : cupy.ndarray
        Same shape, dtype float64.

    Returns
    -------
    volume_ft, volume_weight_ft : tuple of cupy.ndarray
        The modified input arrays.
    '''
    p, _, ph = volume_ft.shape
    _enforce_hermitian_ft_kernel(
        (ceil_div(p * ph, 256),),
        (256,),
        (volume_ft, volume_weight_ft, p, ph),
    )
    return volume_ft, volume_weight_ft


def symmetrize_ft(
    volume_ft: cp.ndarray,
    rotations: ArrayLike,
) -> cp.ndarray:
    '''Sum symmetry-related samples of a Fourier volume.

    Parameters
    ----------
    volume_ft : cupy.ndarray
        Shape (p, p, p // 2 + 1), dtype float64 or complex128. The input is
        not modified.
    rotations : ArrayLike
        Shape (m, 3, 3), dtype float64.

    Returns
    -------
    symmetrized_ft : cupy.ndarray
        The input array when only the identity rotation is present; otherwise,
        a new array with the same shape and dtype as volume_ft.
    '''
    p, _, ph = volume_ft.shape
    assert volume_ft.shape == (p, p, ph) and ph == p // 2 + 1
    assert volume_ft.dtype in (cp.float64, cp.complex128)

    rotations = cp.asarray(rotations, dtype=cp.float64)
    assert rotations.ndim == 3 and rotations.shape[1:] == (3, 3)
    if len(rotations) == 1:
        return volume_ft

    output = volume_ft.copy()
    kernel = (
        _symmetrize_complex_ft_kernel
        if volume_ft.dtype == cp.complex128
        else _symmetrize_real_ft_kernel
    )
    kernel(
        (ceil_div(volume_ft.size, 256),),
        (256,),
        (volume_ft, output, rotations, len(rotations), p, ph, volume_ft.size),
    )
    return output


def cut_sphere_ft(data: cp.ndarray) -> cp.ndarray:
    '''Zero a Fourier volume outside its support sphere in place.

    Parameters
    ----------
    data : cupy.ndarray
        Shape (p, p, p // 2 + 1), dtype float64 or complex128.

    Returns
    -------
    data : cupy.ndarray
        The modified input array.
    '''
    p, _, ph = data.shape
    assert data.dtype in (cp.float64, cp.complex128)
    complex_data = int(data.dtype == cp.complex128)
    _cut_sphere_ft_kernel(
        (ceil_div(data.size, 256),),
        (256,),
        (data, complex_data, p, ph, data.size),
    )
    return data


def stable_divide_ft(
    volume_ft: cp.ndarray,
    weights: cp.ndarray,
) -> cp.ndarray:
    '''Divide a Fourier volume by shell-floored weights in place.

    Parameters
    ----------
    volume_ft : cupy.ndarray
        Shape (p, p, p // 2 + 1), dtype complex128. Modified in place.
    weights : cupy.ndarray
        Same shape, dtype float64.

    Returns
    -------
    volume_ft : cupy.ndarray
        The modified input Fourier array.
    '''
    p, _, ph = weights.shape
    shells = p // 4
    sums = cp.zeros(shells, dtype=cp.float64)
    counts = cp.zeros(shells, dtype=cp.uint64)
    _weight_shells_ft_kernel(
        (ceil_div(weights.size, 256),),
        (256,),
        (weights, sums, counts, p, ph, shells, weights.size),
    )
    floors = sums / counts / 1000.0
    _divide_ft_kernel(
        (ceil_div(volume_ft.size, 256),),
        (256,),
        (volume_ft, weights, floors, p, ph, shells, volume_ft.size),
    )
    return volume_ft


def initialize_gridding_weight_ft(p: int) -> cp.ndarray:
    '''Create iterative gridding weights for a padded side length.

    Parameters
    ----------
    p : int
        Padded real-space side length.

    Returns
    -------
    output : cupy.ndarray
        Shape (p, p, p // 2 + 1), dtype float64.
    '''
    ph = p // 2 + 1
    output = cp.empty((p, p, ph), dtype=cp.float64)
    _initialize_gridding_weight_ft_kernel(
        (ceil_div(output.size, 256),),
        (256,),
        (output, p, ph, output.size),
    )
    return output


def multiply_ftblob_3d(
    volume: cp.ndarray,
    table: ArrayLike,
) -> cp.ndarray:
    '''Multiply an unshifted real volume by a radial blob table in place.'''
    p = volume.shape[0]
    ph = p // 2 + 1
    table = cp.asarray(table, dtype=cp.float64)
    _multiply_ftblob_3d_kernel(
        (ceil_div(volume.size, 256),),
        (256,),
        (volume, table, len(table), p, ph, volume.size),
    )
    return volume


def update_gridding_weight_ft(
    weight: cp.ndarray,
    convolved: cp.ndarray,
) -> cp.ndarray:
    '''Update iterative Fourier gridding weights in place.'''
    p, _, ph = weight.shape
    _update_gridding_weight_ft_kernel(
        (ceil_div(weight.size, 256),),
        (256,),
        (weight, convolved, p, ph, weight.size),
    )
    return weight


def window_ft(volume_ft: cp.ndarray, n: int) -> cp.ndarray:
    '''Crop an unshifted Fourier volume to side length n.'''
    p, _, ph = volume_ft.shape
    nh = n // 2 + 1
    output = cp.zeros((n, n, nh), dtype=cp.complex128)
    _window_ft_kernel(
        (ceil_div(output.size, 256),),
        (256,),
        (volume_ft, output, p, ph, n, nh, output.size),
    )
    return output


def translate_center_3d_ft(volume_ft: cp.ndarray) -> cp.ndarray:
    '''Move the real-space origin by half a box in place.'''
    n, _, nh = volume_ft.shape
    _translate_center_3d_ft_kernel(
        (ceil_div(volume_ft.size, 256),),
        (256,),
        (volume_ft, n, nh, volume_ft.size),
    )
    return volume_ft


def soft_mask_outside_3d(
    volume: cp.ndarray,
    radius: float,
    width: float,
) -> cp.ndarray:
    '''Blend a centered real volume into its exterior background in place.'''
    n = volume.shape[0]
    statistics = cp.zeros(2, dtype=cp.float64)
    _soft_mask_background_3d_kernel(
        (ceil_div(volume.size, 256),),
        (256,),
        (volume, statistics, radius, width, n, volume.size),
    )
    _soft_mask_outside_3d_kernel(
        (ceil_div(volume.size, 256),),
        (256,),
        (volume, statistics, radius, width, n, volume.size),
    )
    return volume


def gridding_correct_3d(volume: cp.ndarray) -> cp.ndarray:
    '''Correct trilinear interpolation attenuation in place.'''
    n = volume.shape[0]
    _gridding_correct_3d_kernel(
        (ceil_div(volume.size, 256),),
        (256,),
        (volume, n, volume.size),
    )
    return volume
