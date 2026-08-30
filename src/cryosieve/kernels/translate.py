import cupy as cp
from cupy.fft import irfftn, rfftn
from numpy.typing import ArrayLike

from . import ceil_div


_translate_2d_ft_kernel = cp.RawKernel(r'''
extern "C" __global__ void translate_2d_ft(
    double* stack_ft,
    int m,
    int n,
    const double* translations)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int nh = n / 2 + 1;
    if (tid < m * n * nh) {
        int b = tid / nh / n;
        int j = tid / nh % n;
        int i = tid % nh;
        int y = j < nh ? j : j - n;
        int x = i;

        double real = stack_ft[2 * tid    ];
        double imag = stack_ft[2 * tid + 1];
        double tx = translations[2 * b    ];
        double ty = translations[2 * b + 1];
        double pi = 3.1415926535897932384626;
        double phi = -2 * pi * (tx * x / n + ty * y / n);
        stack_ft[2 * tid    ] = real * cos(phi) - imag * sin(phi);
        stack_ft[2 * tid + 1] = real * sin(phi) + imag * cos(phi);
    }
}''', 'translate_2d_ft')


def translate_2d_ft(
    stack_ft: cp.ndarray,
    translations: ArrayLike,
) -> cp.ndarray:
    '''Translate a 2D stack in place in Fourier space.

    Parameters
    ----------
    stack_ft : cupy.ndarray
        Shape (m, n, n // 2 + 1), dtype complex128, the rfftn half-spectrum
        over axes (1, 2). Modified in place.
    translations : ArrayLike
        Shape (m, 2), dtype float64, in x-y pixel order.

    Returns
    -------
    stack_ft : cupy.ndarray
        The modified input Fourier array.
    '''
    translations = cp.asarray(translations, dtype=cp.float64)
    m = stack_ft.shape[0]
    n = stack_ft.shape[1]
    assert stack_ft.shape == (m, n, n // 2 + 1)
    assert stack_ft.dtype == cp.complex128
    assert translations.shape == (m, 2)

    _translate_2d_ft_kernel(
        (ceil_div(stack_ft.size, 256),),
        (256,),
        (stack_ft, m, n, translations),
    )
    return stack_ft


def translate_2d(stack: ArrayLike, translations: ArrayLike) -> cp.ndarray:
    '''Translate a real 2D image stack in plane.

    Parameters
    ----------
    stack : ArrayLike
        Shape (m, n, n), dtype float64.
    translations : ArrayLike
        Shape (m, 2), dtype float64, in x-y pixel order.

    Returns
    -------
    translated_stack : cupy.ndarray
        Shape (m, n, n), dtype float64.
    '''
    stack = cp.asarray(stack, dtype=cp.float64)
    m, n = stack.shape[:2]
    assert stack.shape == (m, n, n)

    stack_ft = rfftn(stack, axes=(1, 2))
    translate_2d_ft(stack_ft, translations)
    return irfftn(stack_ft, s=(n, n), axes=(1, 2))
