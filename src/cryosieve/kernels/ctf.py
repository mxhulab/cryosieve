import cupy as cp
from cupy.fft import rfftn, irfftn
from numpy.typing import ArrayLike
from . import ceil_div

ker_get_ctf = cp.RawKernel(r'''
extern "C" __global__ void get_ctf(
    double* f_ctf,
    int m,
    int n,
    const double* ctfs,
    int order)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int n_ = n / 2 + 1;
    if (tid < m * n * n_) {
        int x = tid % n_;
        int y = tid / n_ % n; y = y < n_ ? y : y - n;
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
        f_ctf[tid] = pow(-sqrt(1 - pow(amplitudeContrast, 2.)) * sin(chi) + amplitudeContrast * cos(chi), (double)order);
    }
}''', 'get_ctf')

def get_ctf(ctfs : ArrayLike, n : int, order : int = 1) -> cp.ndarray:
    '''Get CTF in Fourier domain

    Parameters
    ----------
    ctfs : ArrayLike
        shape (m, 8), dtype float64,
        (voltage, defocus 1, defocus 2, astimatism angle, Cs, amplitude contrast, phase shift, pixelsize)
    order : int

    Returns
    -------
    f_ctf : cupy.ndarray
        shape (m, n, n), dtype float64,
        CTF ** order
    '''
    ctfs = cp.asarray(ctfs, dtype = cp.float64)
    m = ctfs.shape[0]
    assert ctfs.shape == (m, 8)

    f_ctf = cp.empty((m, n, n // 2 + 1), dtype = cp.float64)
    ker_get_ctf((ceil_div(f_ctf.size, 256), ), (256, ), (f_ctf, m, n, ctfs, order))
    return f_ctf

def convolute_ctf(stack : cp.ndarray, ctfs : ArrayLike, order : int = 1) -> cp.ndarray:
    '''Convolute CTF

    Parameters
    ----------
    stack : cupy.ndarray
        shape (m, n, n), dtype float64
    ctfs : ArrayLike
        shape (m, 8), dtype float64,
        (voltage, defocus 1, defocus 2, astimatism angle, Cs, amplitude contrast, phase shift, pixelsize)
    order : int
        1 by default,
        how many times the CTF function is convoluted

    Returns
    -------
    new_stack : cupy.ndarray
        shape (m, n, n), dtype float64,
        convolute(stack, CTF ** order)
    '''
    stack = cp.asarray(stack, dtype = cp.float64)
    ctfs = cp.asarray(ctfs, dtype = cp.float64)
    m = stack.shape[0]
    n = stack.shape[1]
    assert stack.shape == (m, n, n) and ctfs.shape == (m, 8)

    f_stack = rfftn(stack, axes = (1, 2))
    f_ctf = get_ctf(ctfs, n, order)
    f_stack *= f_ctf
    return irfftn(f_stack, axes = (1, 2))
