import cupy as cp
from cupy.fft import irfftn, rfftn
from numpy.typing import ArrayLike

from . import ceil_div


_get_ctf_kernel = cp.RawKernel(r'''
extern "C" __global__ void get_ctf(
    double* ctf_ft,
    int m,
    int n,
    const double* ctfs,
    int order)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int nh = n / 2 + 1;
    if (tid < m * n * nh) {
        int b = tid / nh / n;
        int j = tid / nh % n;
        int i = tid % nh;
        int y = j < nh ? j : j - n;
        int x = i;

        double voltage           = ctfs[b * 8    ];
        double defocus_u         = ctfs[b * 8 + 1];
        double defocus_v         = ctfs[b * 8 + 2];
        double astigmatism       = ctfs[b * 8 + 3];
        double cs                = ctfs[b * 8 + 4];
        double amplitude_contrast = ctfs[b * 8 + 5];
        double phase_shift       = ctfs[b * 8 + 6];
        double pixel_size        = ctfs[b * 8 + 7];

        double pi = 3.1415926535897932384626;
        double wavelength = 12.2643247 / sqrt(voltage * (1 + voltage * 0.978466e-6));
        double frequency = hypot(x / (pixel_size * n), y / (pixel_size * n));
        double alpha = atan2((double)y, (double)x) - astigmatism;
        double defocus = -(defocus_u + defocus_v + (defocus_u - defocus_v) * cos(2 * alpha)) / 2;
        double chi = pi * wavelength * defocus * pow(frequency, 2.) + pi / 2 * cs * pow(wavelength, 3.) * pow(frequency, 4.) - phase_shift;
        ctf_ft[tid] = pow(-sqrt(1 - pow(amplitude_contrast, 2.)) * sin(chi) + amplitude_contrast * cos(chi), (double)order);
    }
}''', 'get_ctf')


def get_ctf(ctfs: ArrayLike, n: int, order: int = 1) -> cp.ndarray:
    '''Get CTF in Fourier space.

    Parameters
    ----------
    ctfs : ArrayLike
        Shape (m, 8), dtype float64: voltage, defocus U, defocus V,
        astigmatism angle, Cs, amplitude contrast, phase shift, and pixel size.
    n : int
        Real-space image side length.
    order : int
        CTF exponent, 1 by default.

    Returns
    -------
    ctf_ft : cupy.ndarray
        Shape (m, n, n // 2 + 1), dtype float64, containing CTF ** order.
    '''
    ctfs = cp.asarray(ctfs, dtype=cp.float64)
    m = ctfs.shape[0]
    assert ctfs.shape == (m, 8)

    ctf_ft = cp.empty((m, n, n // 2 + 1), dtype=cp.float64)
    _get_ctf_kernel(
        (ceil_div(ctf_ft.size, 256),),
        (256,),
        (ctf_ft, m, n, ctfs, order),
    )
    return ctf_ft


def convolute_ctf_ft(
    stack_ft: cp.ndarray,
    ctfs: ArrayLike,
    order: int = 1,
) -> cp.ndarray:
    '''Convolute a 2D stack in place in Fourier space.

    Parameters
    ----------
    stack_ft : cupy.ndarray
        Shape (m, n, n // 2 + 1), dtype complex128, the rfftn half-spectrum
        over axes (1, 2). Modified in place.
    ctfs : ArrayLike
        Shape (m, 8), dtype float64.
    order : int
        CTF exponent, 1 by default.

    Returns
    -------
    stack_ft : cupy.ndarray
        The modified input Fourier array.
    '''
    m = stack_ft.shape[0]
    n = stack_ft.shape[1]
    assert stack_ft.shape == (m, n, n // 2 + 1)
    assert stack_ft.dtype == cp.complex128

    stack_ft *= get_ctf(ctfs, n, order)
    return stack_ft


def convolute_ctf(
    stack: ArrayLike,
    ctfs: ArrayLike,
    order: int = 1,
) -> cp.ndarray:
    '''Convolute a real 2D image stack with its CTF.

    Parameters
    ----------
    stack : ArrayLike
        Shape (m, n, n), dtype float64.
    ctfs : ArrayLike
        Shape (m, 8), dtype float64.
    order : int
        CTF exponent, 1 by default.

    Returns
    -------
    convolved_stack : cupy.ndarray
        Shape (m, n, n), dtype float64.
    '''
    stack = cp.asarray(stack, dtype=cp.float64)
    m, n = stack.shape[:2]
    assert stack.shape == (m, n, n)

    stack_ft = rfftn(stack, axes=(1, 2))
    convolute_ctf_ft(stack_ft, ctfs, order)
    return irfftn(stack_ft, s=(n, n), axes=(1, 2))
