import cupy as cp
from cupy.fft import irfftn, rfftn
from numpy.typing import ArrayLike

from . import ceil_div


_get_fsc_ft_kernel = cp.RawKernel(r'''
extern "C" __global__ void get_fsc_ft(
    const double* volume1_ft,
    const double* volume2_ft,
    double* numerator,
    double* denominator1,
    double* denominator2,
    int n,
    int nh,
    long long size)
{
    long long tid = blockDim.x * (long long)blockIdx.x + threadIdx.x;
    if (tid < size) {
        int k = tid / ((long long)n * nh);
        int j = tid / nh % n;
        int i = tid % nh;
        int z = k < nh ? k : k - n;
        int y = j < nh ? j : j - n;
        int x = i;
        int shell = floor(sqrt((double)(z * z + y * y + x * x)) + 0.5);

        if (shell < nh) {
            double real1 = volume1_ft[2 * tid];
            double imag1 = volume1_ft[2 * tid + 1];
            double real2 = volume2_ft[2 * tid];
            double imag2 = volume2_ft[2 * tid + 1];
            atomicAdd(numerator + shell, real1 * real2 + imag1 * imag2);
            atomicAdd(denominator1 + shell, real1 * real1 + imag1 * imag1);
            atomicAdd(denominator2 + shell, real2 * real2 + imag2 * imag2);
        }
    }
}''', 'get_fsc_ft')


_randomize_phases_3d_ft_kernel = cp.RawKernel(r'''
extern "C" __global__ void randomize_phases_3d_ft(
    double* volume_ft,
    const double* phases,
    int radius2,
    int n,
    int nh,
    long long size)
{
    long long tid = blockDim.x * (long long)blockIdx.x + threadIdx.x;
    if (tid < size) {
        int k = tid / ((long long)n * nh);
        int j = tid / nh % n;
        int i = tid % nh;
        int z = k < nh ? k : k - n;
        int y = j < nh ? j : j - n;
        int x = i;

        if (z * z + y * y + x * x >= radius2) {
            double real = volume_ft[2 * tid];
            double imag = volume_ft[2 * tid + 1];
            double magnitude = hypot(real, imag);
            double phase = phases[tid];
            volume_ft[2 * tid] = magnitude * cos(phase);
            volume_ft[2 * tid + 1] = magnitude * sin(phase);
        }
    }
}''', 'randomize_phases_3d_ft')


_fsc_weight_3d_ft_kernel = cp.RawKernel(r'''
extern "C" __global__ void fsc_weight_3d_ft(
    double* volume_ft,
    const double* fsc,
    int maximum_shell,
    int n,
    int nh,
    long long size)
{
    long long tid = blockDim.x * (long long)blockIdx.x + threadIdx.x;
    if (tid < size) {
        int k = tid / ((long long)n * nh);
        int j = tid / nh % n;
        int i = tid % nh;
        int z = k < nh ? k : k - n;
        int y = j < nh ? j : j - n;
        int x = i;
        int shell = floor(sqrt((double)(z * z + y * y + x * x)) + 0.5);

        if (shell <= maximum_shell && fsc[shell] > 0.0) {
            double weight = sqrt(2.0 * fsc[shell] / (1.0 + fsc[shell]));
            volume_ft[2 * tid] *= weight;
            volume_ft[2 * tid + 1] *= weight;
        } else {
            volume_ft[2 * tid] = 0.0;
            volume_ft[2 * tid + 1] = 0.0;
        }
    }
}''', 'fsc_weight_3d_ft')


_radial_amplitude_3d_ft_kernel = cp.RawKernel(r'''
extern "C" __global__ void radial_amplitude_3d_ft(
    const double* volume_ft,
    double* amplitude,
    unsigned long long* count,
    int n,
    int nh,
    long long size)
{
    long long tid = blockDim.x * (long long)blockIdx.x + threadIdx.x;
    if (tid < size) {
        int k = tid / ((long long)n * nh);
        int j = tid / nh % n;
        int i = tid % nh;
        int z = k < nh ? k : k - n;
        int y = j < nh ? j : j - n;
        int x = i;
        int shell = floor(sqrt((double)(z * z + y * y + x * x)) + 0.5);

        if (shell < nh) {
            double real = volume_ft[2 * tid];
            double imag = volume_ft[2 * tid + 1];
            atomicAdd(amplitude + shell, hypot(real, imag));
            atomicAdd(count + shell, 1ULL);
        }
    }
}''', 'radial_amplitude_3d_ft')


_bfactor_sharpen_ft_kernel = cp.RawKernel(r'''
extern "C" __global__ void bfactor_sharpen_ft(
    double* volume_ft,
    double angpix,
    double bfactor,
    int n,
    int nh,
    long long size)
{
    long long tid = blockDim.x * (long long)blockIdx.x + threadIdx.x;
    if (tid < size) {
        int k = tid / ((long long)n * nh);
        int j = tid / nh % n;
        int i = tid % nh;
        int z = k < nh ? k : k - n;
        int y = j < nh ? j : j - n;
        int x = i;
        double frequency = sqrt((double)(z * z + y * y + x * x)) / (n * angpix);
        double nyquist = 1.0 / (2.0 * angpix);

        if (frequency <= nyquist) {
            double weight = exp(-0.25 * bfactor * frequency * frequency);
            volume_ft[2 * tid] *= weight;
            volume_ft[2 * tid + 1] *= weight;
        } else {
            volume_ft[2 * tid] = 0.0;
            volume_ft[2 * tid + 1] = 0.0;
        }
    }
}''', 'bfactor_sharpen_ft')


def _ft_shape(volume_ft):
    n = volume_ft.shape[0]
    nh = n // 2 + 1
    assert volume_ft.shape == (n, n, nh)
    assert volume_ft.dtype == cp.complex128
    assert volume_ft.flags.c_contiguous
    return n, nh


def get_fsc_ft(volume1_ft: cp.ndarray, volume2_ft: cp.ndarray) -> cp.ndarray:
    '''Calculate the Fourier shell correlation of two volumes.

    Parameters
    ----------
    volume1_ft, volume2_ft : cupy.ndarray
        Shape (n, n, n // 2 + 1), dtype complex128, rfftn half-spectra over
        axes (0, 1, 2). The inputs are not modified.

    Returns
    -------
    fsc : cupy.ndarray
        Shape (n // 2 + 1,), dtype float64.
    '''
    n, nh = _ft_shape(volume1_ft)
    assert volume2_ft.shape == volume1_ft.shape
    assert volume2_ft.dtype == cp.complex128
    assert volume2_ft.flags.c_contiguous

    numerator = cp.zeros(nh, dtype=cp.float64)
    denominator1 = cp.zeros(nh, dtype=cp.float64)
    denominator2 = cp.zeros(nh, dtype=cp.float64)
    _get_fsc_ft_kernel(
        (ceil_div(volume1_ft.size, 256),),
        (256,),
        (volume1_ft, volume2_ft, numerator, denominator1, denominator2,
         n, nh, volume1_ft.size),
    )
    fsc = numerator / cp.sqrt(denominator1 * denominator2)
    fsc[0] = 1.0
    return fsc


def get_fsc(volume1: ArrayLike, volume2: ArrayLike) -> cp.ndarray:
    '''Calculate the Fourier shell correlation of two real volumes.

    Parameters
    ----------
    volume1, volume2 : ArrayLike
        Shape (n, n, n), dtype float64.

    Returns
    -------
    fsc : cupy.ndarray
        Shape (n // 2 + 1,), dtype float64.
    '''
    volume1 = cp.asarray(volume1, dtype=cp.float64)
    volume2 = cp.asarray(volume2, dtype=cp.float64)
    n = volume1.shape[0]
    assert volume1.shape == (n, n, n) and volume2.shape == volume1.shape
    return get_fsc_ft(
        rfftn(volume1, axes=(0, 1, 2), norm='forward'),
        rfftn(volume2, axes=(0, 1, 2), norm='forward'),
    )


def randomize_phases_3d_ft(
    volume_ft: cp.ndarray,
    radius: int,
    phases: ArrayLike,
) -> cp.ndarray:
    '''Randomize high-frequency phases in place.

    Parameters
    ----------
    volume_ft : cupy.ndarray
        Shape (n, n, n // 2 + 1), dtype complex128, an rfftn half-spectrum
        over axes (0, 1, 2). Modified in place.
    radius : int
        Fourier shell at which phase randomization starts.
    phases : ArrayLike
        Same shape as volume_ft, dtype float64, phases in radians.

    Returns
    -------
    volume_ft : cupy.ndarray
        The modified input Fourier array.
    '''
    n, nh = _ft_shape(volume_ft)
    phases = cp.asarray(phases, dtype=cp.float64)
    assert phases.shape == volume_ft.shape and phases.flags.c_contiguous
    _randomize_phases_3d_ft_kernel(
        (ceil_div(volume_ft.size, 256),),
        (256,),
        (volume_ft, phases, radius * radius, n, nh, volume_ft.size),
    )
    return volume_ft


def randomize_phases_3d(
    volume: ArrayLike,
    radius: int,
    phases: ArrayLike,
) -> cp.ndarray:
    '''Randomize high-frequency phases of a real volume.

    Parameters
    ----------
    volume : ArrayLike
        Shape (n, n, n), dtype float64.
    radius : int
        Fourier shell at which phase randomization starts.
    phases : ArrayLike
        Shape (n, n, n // 2 + 1), dtype float64, phases in radians.

    Returns
    -------
    randomized_volume : cupy.ndarray
        Shape (n, n, n), dtype float64.
    '''
    volume = cp.asarray(volume, dtype=cp.float64)
    n = volume.shape[0]
    assert volume.shape == (n, n, n)
    volume_ft = rfftn(volume, axes=(0, 1, 2), norm='forward')
    randomize_phases_3d_ft(volume_ft, radius, phases)
    return irfftn(volume_ft, s=volume.shape, axes=(0, 1, 2), norm='forward')


def fsc_weight_3d_ft(volume_ft: cp.ndarray, fsc: ArrayLike) -> cp.ndarray:
    '''Apply FSC weights to a Fourier volume in place.

    Parameters
    ----------
    volume_ft : cupy.ndarray
        Shape (n, n, n // 2 + 1), dtype complex128, an rfftn half-spectrum
        over axes (0, 1, 2). Modified in place.
    fsc : ArrayLike
        Shape (n // 2 + 1,), dtype float64.

    Returns
    -------
    volume_ft : cupy.ndarray
        The modified input Fourier array.
    '''
    n, nh = _ft_shape(volume_ft)
    fsc = cp.asarray(fsc, dtype=cp.float64)
    assert fsc.shape == (nh,)
    maximum_shell = 0
    for shell, value in enumerate(cp.asnumpy(fsc[1:]), start=1):
        if value < 0.0001:
            break
        maximum_shell = shell
    _fsc_weight_3d_ft_kernel(
        (ceil_div(volume_ft.size, 256),),
        (256,),
        (volume_ft, fsc, maximum_shell, n, nh, volume_ft.size),
    )
    return volume_ft


def radial_amplitude_3d_ft(volume_ft: cp.ndarray) -> cp.ndarray:
    '''Calculate mean amplitudes in rounded Fourier shells.

    Parameters
    ----------
    volume_ft : cupy.ndarray
        Shape (n, n, n // 2 + 1), dtype complex128, an rfftn half-spectrum
        over axes (0, 1, 2). The input is not modified.

    Returns
    -------
    amplitude : cupy.ndarray
        Shape (n // 2 + 1,), dtype float64.
    '''
    n, nh = _ft_shape(volume_ft)
    amplitude = cp.zeros(nh, dtype=cp.float64)
    count = cp.zeros(nh, dtype=cp.uint64)
    _radial_amplitude_3d_ft_kernel(
        (ceil_div(volume_ft.size, 256),),
        (256,),
        (volume_ft, amplitude, count, n, nh, volume_ft.size),
    )
    return amplitude / count


def bfactor_sharpen_ft(
    volume_ft: cp.ndarray,
    angpix: float,
    bfactor: float,
) -> cp.ndarray:
    '''Apply B-factor sharpening to a Fourier volume in place.

    Parameters
    ----------
    volume_ft : cupy.ndarray
        Shape (n, n, n // 2 + 1), dtype complex128, an rfftn half-spectrum
        over axes (0, 1, 2). Modified in place.
    angpix : float
        Pixel size in Angstrom.
    bfactor : float
        B-factor in square Angstrom.

    Returns
    -------
    volume_ft : cupy.ndarray
        The modified input Fourier array.
    '''
    n, nh = _ft_shape(volume_ft)
    _bfactor_sharpen_ft_kernel(
        (ceil_div(volume_ft.size, 256),),
        (256,),
        (volume_ft, angpix, bfactor, n, nh, volume_ft.size),
    )
    return volume_ft


def bfactor_sharpen(
    volume: ArrayLike,
    angpix: float,
    bfactor: float,
) -> cp.ndarray:
    '''Apply B-factor sharpening to a real volume.

    Parameters
    ----------
    volume : ArrayLike
        Shape (n, n, n), dtype float64.
    angpix : float
        Pixel size in Angstrom.
    bfactor : float
        B-factor in square Angstrom.

    Returns
    -------
    sharpened_volume : cupy.ndarray
        Shape (n, n, n), dtype float64.
    '''
    volume = cp.asarray(volume, dtype=cp.float64)
    n = volume.shape[0]
    assert volume.shape == (n, n, n)
    volume_ft = rfftn(volume, axes=(0, 1, 2), norm='forward')
    bfactor_sharpen_ft(volume_ft, angpix, bfactor)
    return irfftn(volume_ft, s=volume.shape, axes=(0, 1, 2), norm='forward')
