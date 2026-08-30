import cupy as cp
from cupy.fft import irfftn, rfftn
from numpy.typing import ArrayLike

from . import ceil_div


_bandpass_2d_ft_kernel = cp.RawKernel(r'''
extern "C" __global__ void bandpass_2d_ft(
    double* stack_ft,
    int m,
    int n,
    double threshold_low,
    double threshold_high)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int nh = n / 2 + 1;
    if (tid < m * n * nh) {
        int b = tid / nh / n;
        int j = tid / nh % n;
        int i = tid % nh;
        int y = j < nh ? j : j - n;
        int x = i;

        double frequency = hypot((double)x / n, (double)y / n);
        if (frequency < threshold_low || frequency > threshold_high) {
            stack_ft[2 * tid    ] = 0;
            stack_ft[2 * tid + 1] = 0;
        }
    }
}''', 'bandpass_2d_ft')


def bandpass_2d_ft(
    stack_ft: cp.ndarray,
    threshold_low: float,
    threshold_high: float,
) -> cp.ndarray:
    '''Bandpass filter a 2D stack in place in Fourier space.

    Parameters
    ----------
    stack_ft : cupy.ndarray
        Shape (m, n, n // 2 + 1), dtype complex128, the rfftn half-spectrum
        over axes (1, 2). Modified in place.
    threshold_low : float
        Dimensionless lower cutoff in range [0, 1].
    threshold_high : float
        Dimensionless upper cutoff in range [0, 1].

    Returns
    -------
    stack_ft : cupy.ndarray
        The modified input Fourier array.
    '''
    m = stack_ft.shape[0]
    n = stack_ft.shape[1]
    assert stack_ft.shape == (m, n, n // 2 + 1)
    assert stack_ft.dtype == cp.complex128

    _bandpass_2d_ft_kernel(
        (ceil_div(stack_ft.size, 256),),
        (256,),
        (stack_ft, m, n, threshold_low, threshold_high),
    )
    return stack_ft


def bandpass_2d(
    stack: ArrayLike,
    threshold_low: float,
    threshold_high: float,
) -> cp.ndarray:
    '''Bandpass filter a real 2D image stack.

    Parameters
    ----------
    stack : ArrayLike
        Shape (m, n, n), dtype float64.
    threshold_low : float
        Dimensionless lower cutoff in range [0, 1].
    threshold_high : float
        Dimensionless upper cutoff in range [0, 1].

    Returns
    -------
    filtered_stack : cupy.ndarray
        Shape (m, n, n), dtype float64.
    '''
    stack = cp.asarray(stack, dtype=cp.float64)
    m, n = stack.shape[:2]
    assert stack.shape == (m, n, n)

    stack_ft = rfftn(stack, axes=(1, 2))
    bandpass_2d_ft(stack_ft, threshold_low, threshold_high)
    return irfftn(stack_ft, s=(n, n), axes=(1, 2))


def lowpass_2d_ft(stack_ft: cp.ndarray, threshold: float) -> cp.ndarray:
    '''Lowpass filter a 2D stack in place in Fourier space.

    Parameters
    ----------
    stack_ft : cupy.ndarray
        Shape (m, n, n // 2 + 1), dtype complex128, the rfftn half-spectrum
        over axes (1, 2). Modified in place.
    threshold : float
        Dimensionless cutoff in range [0, 1].

    Returns
    -------
    stack_ft : cupy.ndarray
        The modified input Fourier array.
    '''
    return bandpass_2d_ft(stack_ft, 0.0, threshold)


def lowpass_2d(stack: ArrayLike, threshold: float) -> cp.ndarray:
    '''Lowpass filter a real 2D image stack.

    Parameters
    ----------
    stack : ArrayLike
        Shape (m, n, n), dtype float64.
    threshold : float
        Dimensionless cutoff in range [0, 1].

    Returns
    -------
    filtered_stack : cupy.ndarray
        Shape (m, n, n), dtype float64.
    '''
    stack = cp.asarray(stack, dtype=cp.float64)
    m, n = stack.shape[:2]
    assert stack.shape == (m, n, n)

    stack_ft = rfftn(stack, axes=(1, 2))
    lowpass_2d_ft(stack_ft, threshold)
    return irfftn(stack_ft, s=(n, n), axes=(1, 2))


def highpass_2d_ft(stack_ft: cp.ndarray, threshold: float) -> cp.ndarray:
    '''Highpass filter a 2D stack in place in Fourier space.

    Parameters
    ----------
    stack_ft : cupy.ndarray
        Shape (m, n, n // 2 + 1), dtype complex128, the rfftn half-spectrum
        over axes (1, 2). Modified in place.
    threshold : float
        Dimensionless cutoff in range [0, 1].

    Returns
    -------
    stack_ft : cupy.ndarray
        The modified input Fourier array.
    '''
    return bandpass_2d_ft(stack_ft, threshold, 1.0)


def highpass_2d(stack: ArrayLike, threshold: float) -> cp.ndarray:
    '''Highpass filter a real 2D image stack.

    Parameters
    ----------
    stack : ArrayLike
        Shape (m, n, n), dtype float64.
    threshold : float
        Dimensionless cutoff in range [0, 1].

    Returns
    -------
    filtered_stack : cupy.ndarray
        Shape (m, n, n), dtype float64.
    '''
    stack = cp.asarray(stack, dtype=cp.float64)
    m, n = stack.shape[:2]
    assert stack.shape == (m, n, n)

    stack_ft = rfftn(stack, axes=(1, 2))
    highpass_2d_ft(stack_ft, threshold)
    return irfftn(stack_ft, s=(n, n), axes=(1, 2))
