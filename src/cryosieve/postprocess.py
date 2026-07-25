from __future__ import annotations

import argparse
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import mrcfile
import numpy as np
import pandas as pd
import starfile
from .logger import configure_logging, logger


POSTPROCESS_LOG_NAME = 'postprocess.log'


@dataclass(frozen=True)
class PostprocessResult:
    processed_map: Path
    masked_map: Optional[Path]
    metadata_star: Path
    fsc_plot: Path
    final_resolution: float
    bfactor: float
    randomize_from: Optional[float]


def _postprocess_logger(output_root, log_path=None):
    job_dir = os.environ.get('COCO_JOB_DIR')
    if log_path is not None:
        path = Path(log_path)
    elif job_dir:
        path = Path(job_dir).absolute() / POSTPROCESS_LOG_NAME
    else:
        path = Path(output_root).parent / POSTPROCESS_LOG_NAME
    target = logging.getLogger('CryoSieve.Postprocess')
    for handler in list(target.handlers):
        target.removeHandler(handler)
        handler.close()
    target.setLevel(logging.INFO)
    target.propagate = False
    path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(str(path), mode='a', encoding='utf-8')
    handler.setFormatter(logging.Formatter('[%(asctime)s][%(levelname)s] %(message)s'))
    target.addHandler(handler)
    return target


def read_map(path):
    '''Read a cubic MRC map in canonical z-y-x order and its X voxel size.'''
    with mrcfile.mmap(path, permissive=True) as mrc:
        volume = np.array(mrc.data, dtype=np.float64, order='C', copy=True)
        voxel_size = np.array([
            float(mrc.voxel_size.x),
            float(mrc.voxel_size.y),
            float(mrc.voxel_size.z),
        ])

    if not np.allclose(voxel_size, voxel_size[0]):
        logger.warning(f'Anisotropic voxel size in {path}: {voxel_size}; using X value')

    n = volume.shape[0]
    if volume.ndim != 3 or volume.shape != (n, n, n):
        raise ValueError(f'Map must be cubic, got {volume.shape}: {path}')
    return volume, float(voxel_size[0])


def last_passing_shell(fsc, threshold):
    shell = -1
    for index, value in enumerate(fsc):
        if value < threshold:
            break
        shell = index
    return shell


def _guinier_plot_ft(volume_ft):
    import cupy as cp
    from .kernels import radial_amplitude_3d_ft

    amplitude = cp.asnumpy(radial_amplitude_3d_ft(volume_ft))
    log_amplitude = np.full(len(amplitude), -99.0, dtype=np.float64)
    positive = amplitude > 0.0
    log_amplitude[positive] = np.log(amplitude[positive])
    return log_amplitude


def fit_bfactor(log_amplitude, angpix, autob_lowres=10.0):
    n = 2 * (len(log_amplitude) - 1)
    shells = np.arange(len(log_amplitude), dtype=np.float64)
    frequency2 = (shells / (n * angpix)) ** 2
    resolutions = np.empty(len(shells), dtype=np.float64)
    resolutions[0] = np.inf
    resolutions[1:] = n * angpix / shells[1:]
    selected = ((resolutions <= autob_lowres)
                & (resolutions >= 0.0)
                & (log_amplitude > -99.0))

    x = frequency2[selected]
    y = log_amplitude[selected]
    if len(x) < 2:
        raise RuntimeError('Too few nonzero Fourier shells for automatic B-factor fitting')
    mean_x = np.mean(x)
    mean_y = np.mean(y)
    ss_xx = np.sum(x * x) - len(x) * mean_x * mean_x
    ss_yy = np.sum(y * y) - len(y) * mean_y * mean_y
    ss_xy = np.sum(x * y) - len(x) * mean_x * mean_y
    slope = ss_xy / ss_xx
    intercept = mean_y - slope * mean_x
    correlation = ss_xy * ss_xy / (ss_xx * ss_yy)
    return slope, intercept, correlation, 4.0 * slope


def _save_fsc_plot(path, curves, angpix, final_resolution):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    nh = len(curves[0][0])
    n = 2 * (nh - 1)
    shells = np.arange(nh, dtype=np.float64)
    final_shell = n * angpix / final_resolution
    fig = plt.figure(figsize=(15, 10), dpi=150)
    ax = fig.subplots()
    for spine in ax.spines.values():
        spine.set_linewidth(5)
    ax.tick_params(axis='both', length=10, width=5, labelsize=26,
                   direction='in', pad=10)
    for fsc, label in curves:
        ax.plot(shells, fsc, linewidth=6, label=label)
    ax.axhline(0.143, linewidth=4, linestyle='dashed', color='gray')
    ax.axvline(final_shell, linewidth=3, linestyle=':', color='black')
    ax.set_xlim(0.0, shells[-1])
    ax.set_ylim(-0.1, 1.05)
    ax.xaxis.set_major_formatter(FuncFormatter(
        lambda x, _: 'inf A' if x == 0.0 else f'{n * angpix / x:.2f} A'
    ))
    ax.set_title(f'Final resolution = {n * angpix / final_shell:.2f} A',
                 fontsize=30, fontweight='bold')
    ax.grid(alpha=0.4)
    ax.legend(frameon=True, prop={'size': 22, 'weight': 'bold'},
              loc='upper right')
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _randomized_masked_fsc(volume_ft, mask, radius, rng):
    import cupy as cp
    from cupy.fft import irfftn, rfftn
    from .kernels import randomize_phases_3d_ft

    randomized_ft = volume_ft.copy()
    phases = rng.random_sample(randomized_ft.shape, dtype=cp.float64)
    phases *= 2.0 * cp.pi
    randomize_phases_3d_ft(randomized_ft, radius, phases)
    del phases
    randomized = irfftn(
        randomized_ft,
        s=mask.shape,
        axes=(0, 1, 2),
        norm='forward',
    )
    del randomized_ft
    return rfftn(randomized * mask, axes=(0, 1, 2), norm='forward')


def postprocess(
    half1_path,
    half2_path,
    output_root,
    *,
    angpix=None,
    mask_path=None,
    random_seed=None,
    log_path=None,
):
    '''Postprocess two independently reconstructed half maps.

    Parameters
    ----------
    half1_path, half2_path : path-like
        Input half-map paths.
    output_root : path-like
        Output root used for MRC, STAR, and FSC plot files.
    angpix : float, optional
        Pixel size in Angstrom; read from MRC headers when omitted.
    mask_path : path-like, optional
        Solvent mask with values in [0, 1].
    random_seed : int, optional
        Seed for phase randomization.
    log_path : path-like, optional
        Explicit postprocess log path.

    Returns
    -------
    result : PostprocessResult
        Output paths and fitted postprocessing values.
    '''
    import cupy as cp
    from cupy.fft import irfftn, rfftn
    from .kernels import (
        bfactor_sharpen_ft,
        fsc_weight_3d_ft,
        get_fsc,
        get_fsc_ft,
    )

    output_root = Path(output_root)
    output_root.parent.mkdir(parents=True, exist_ok=True)
    pp_logger = _postprocess_logger(output_root, log_path)
    pp_logger.info(f'Reading half maps: {half1_path}, {half2_path}')
    volume1_cpu, voxel_size1 = read_map(half1_path)
    volume2_cpu, voxel_size2 = read_map(half2_path)
    if volume1_cpu.shape != volume2_cpu.shape:
        raise ValueError(f'Half-map shapes differ: {volume1_cpu.shape} != {volume2_cpu.shape}')

    if angpix is None:
        if voxel_size1 <= 0.0 or voxel_size2 <= 0.0:
            raise ValueError(f'Half maps have invalid voxel sizes: {voxel_size1}, {voxel_size2}')
        if not np.isclose(voxel_size1, voxel_size2):
            raise ValueError(f'Half-map voxel sizes differ: {voxel_size1} != {voxel_size2}')
        angpix = voxel_size1
    elif angpix <= 0.0:
        raise ValueError('angpix must be positive')

    n = volume1_cpu.shape[0]
    volume1 = cp.asarray(volume1_cpu)
    volume2 = cp.asarray(volume2_cpu)
    volume1_ft = rfftn(volume1, axes=(0, 1, 2), norm='forward')
    volume2_ft = rfftn(volume2, axes=(0, 1, 2), norm='forward')
    fsc_unmasked = cp.asnumpy(get_fsc_ft(volume1_ft, volume2_ft))

    mask = None
    mask_cpu = None
    mask_fraction = None
    randomize_at = None
    fsc_masked = None
    fsc_random_masked = None
    fsc_part = None
    if mask_path is None:
        fsc_final = fsc_unmasked.copy()
    else:
        mask_cpu, _ = read_map(mask_path)
        if mask_cpu.shape != volume1_cpu.shape:
            raise ValueError(f'Mask and half-map shapes differ: {mask_cpu.shape} != {volume1_cpu.shape}')
        if np.min(mask_cpu) < 0.0 or np.max(mask_cpu) > 1.0:
            raise ValueError(f'Mask values are outside [0, 1]: min={np.min(mask_cpu)}, max={np.max(mask_cpu)}')
        occupied = np.count_nonzero(mask_cpu > 0.5) / mask_cpu.size
        if occupied == 0.0:
            raise ValueError('Mask contains no voxels above 0.5')
        mask_fraction = 0.476 / occupied
        mask = cp.asarray(mask_cpu)
        fsc_masked = cp.asnumpy(get_fsc(volume1 * mask, volume2 * mask))
        randomize_at = last_passing_shell(fsc_unmasked, 0.8) + 1
        if randomize_at >= len(fsc_unmasked):
            raise RuntimeError('The unmasked FSC never drops below 0.8')

        rng = cp.random.RandomState(random_seed)
        randomized1_ft = _randomized_masked_fsc(volume1_ft, mask, randomize_at, rng)
        randomized2_ft = _randomized_masked_fsc(volume2_ft, mask, randomize_at, rng)
        fsc_random_masked = cp.asnumpy(get_fsc_ft(randomized1_ft, randomized2_ft))
        del randomized1_ft, randomized2_ft
        fsc_final = fsc_masked.copy()
        start = randomize_at + 2
        if start < len(fsc_final):
            fsc_final[start:] = ((fsc_masked[start:] - fsc_random_masked[start:])
                                 / (1.0 - fsc_random_masked[start:]))
        fsc_part = (mask_fraction * fsc_unmasked
                    / (1.0 + (mask_fraction - 1.0) * fsc_unmasked))

    global_shell = last_passing_shell(fsc_final, 0.143)
    unmasked_shell = last_passing_shell(fsc_unmasked, 0.143)
    if mask is not None and unmasked_shell > global_shell:
        pp_logger.warning('Unmasked FSC extends beyond corrected FSC; using unmasked FSC')
        fsc_final = fsc_unmasked.copy()
        global_shell = unmasked_shell
    global_resolution = 999.0 if global_shell == 0 else n * angpix / global_shell

    merged_ft = rfftn((volume1 + volume2) / 2.0,
                      axes=(0, 1, 2), norm='forward')
    guinier_original = _guinier_plot_ft(merged_ft)
    frequency2 = (np.arange(len(guinier_original), dtype=np.float64)
                  / (n * angpix)) ** 2
    fsc_weight_3d_ft(merged_ft, fsc_final)
    guinier_weighted = _guinier_plot_ft(merged_ft)
    slope, intercept, correlation, bfactor = fit_bfactor(guinier_weighted, angpix)
    bfactor_sharpen_ft(merged_ft, angpix, bfactor)
    guinier_sharpened = _guinier_plot_ft(merged_ft)
    processed = cp.asnumpy(irfftn(
        merged_ft,
        s=volume1.shape,
        axes=(0, 1, 2),
        norm='forward',
    ))

    output_map = Path(str(output_root) + '.mrc')
    masked_map = Path(str(output_root) + '_masked.mrc') if mask is not None else None
    star_path = Path(str(output_root) + '.star')
    plot_path = Path(str(output_root) + '_fsc.png')
    mrcfile.write(output_map, processed.astype('f4'), overwrite=True,
                  voxel_size=angpix)
    if masked_map is not None:
        mrcfile.write(masked_map, (processed * mask_cpu).astype('f4'),
                      overwrite=True, voxel_size=angpix)

    general = {
        'rlnFinalResolution': global_resolution,
        'rlnBfactorUsedForSharpening': bfactor,
        'rlnUnfilteredMapHalf1': str(half1_path),
        'rlnUnfilteredMapHalf2': str(half2_path),
        'rlnPostprocessedMap': str(output_map),
        'rlnFittedSlopeGuinierPlot': slope,
        'rlnFittedInterceptGuinierPlot': intercept,
        'rlnCorrelationFitGuinierPlot': correlation,
    }
    if mask is not None:
        general.update({
            'rlnParticleBoxFractionSolventMask': mask_fraction,
            'rlnMaskName': str(mask_path),
            'rlnRandomiseFrom': n * angpix / randomize_at,
            'rlnPostprocessedMapMasked': str(masked_map),
        })

    shells = np.arange(len(fsc_final), dtype=np.int64)
    resolutions = np.full(len(shells), 999.0, dtype=np.float64)
    resolutions[1:] = n * angpix / shells[1:]
    fsc_data = {
        'rlnSpectralIndex': shells,
        'rlnResolution': 1.0 / resolutions,
        'rlnAngstromResolution': resolutions,
    }
    if mask is None:
        fsc_data['rlnFourierShellCorrelationUnmaskedMaps'] = fsc_unmasked
    else:
        fsc_data.update({
            'rlnFourierShellCorrelationCorrected': fsc_final,
            'rlnFourierShellCorrelationParticleMaskFraction': fsc_part,
            'rlnFourierShellCorrelationUnmaskedMaps': fsc_unmasked,
            'rlnFourierShellCorrelationMaskedMaps': fsc_masked,
            'rlnCorrectedFourierShellCorrelationPhaseRandomizedMaskedMaps': fsc_random_masked,
        })
    guinier = pd.DataFrame({
        'rlnResolutionSquared': frequency2,
        'rlnLogAmplitudesOriginal': guinier_original,
        'rlnLogAmplitudesWeighted': guinier_weighted,
        'rlnLogAmplitudesSharpened': guinier_sharpened,
        'rlnLogAmplitudesIntercept': np.full(len(frequency2), intercept),
    })
    starfile.write({
        'general': pd.DataFrame([general]),
        'fsc': pd.DataFrame(fsc_data),
        'guinier': guinier,
    }, star_path, overwrite=True)

    curves = [(fsc_unmasked, 'No mask')]
    if mask is not None:
        curves.extend([(fsc_masked, 'Masked'), (fsc_final, 'Corrected')])
    _save_fsc_plot(plot_path, curves, angpix, global_resolution)
    randomize_from = None if randomize_at is None else n * angpix / randomize_at
    pp_logger.info(f'Final resolution: {global_resolution:g} Angstrom')
    pp_logger.info(f'B-factor: {bfactor:g} square Angstrom')
    pp_logger.info(f'Metadata file: {star_path}')

    del volume1, volume2, volume1_ft, volume2_ft, merged_ft
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()
    return PostprocessResult(
        processed_map=output_map,
        masked_map=masked_map,
        metadata_star=star_path,
        fsc_plot=plot_path,
        final_resolution=global_resolution,
        bfactor=bfactor,
        randomize_from=randomize_from,
    )


def parse_argument():
    parser = argparse.ArgumentParser(
        description='Postprocess two independently reconstructed half maps.'
    )
    parser.add_argument('--i', required=True, help='input half1 map')
    parser.add_argument('--i2', required=True, help='input half2 map')
    parser.add_argument('--o', default='postprocess', help='output root name')
    parser.add_argument('--angpix', type=float, help='pixel size in Angstrom')
    parser.add_argument('--mask', help='solvent mask with values in [0, 1]')
    parser.add_argument('--random_seed', type=int, help='phase randomization seed')
    return parser.parse_args()


def main():
    configure_logging(source='cryosieve-postprocess')
    args = parse_argument()

    from .utility import check_cupy
    check_cupy()

    postprocess(
        args.i,
        args.i2,
        args.o,
        angpix=args.angpix,
        mask_path=args.mask,
        random_seed=args.random_seed,
    )


if __name__ == '__main__':
    main()
