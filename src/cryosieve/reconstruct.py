from __future__ import annotations

import argparse
import mrcfile
import sys
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from .logger import configure_logging, logger


def _collate_particles(batch):
    images, parameters = zip(*batch)
    return np.stack(images), np.stack(parameters)


def _rfftn_separable(data):
    '''Apply a low-workspace 3D real FFT and return C-contiguous storage.'''
    import cupy as cp
    plan_cache = cp.fft.config.get_plan_cache()
    plan_cache_size = plan_cache.get_size()
    plan_cache.set_size(0)
    try:
        transformed = cp.fft.rfft(data, axis=2, norm='forward')
        transformed = cp.fft.fft(transformed, axis=1, norm='forward')
        return cp.ascontiguousarray(cp.fft.fft(transformed, axis=0, norm='forward'))
    finally:
        plan_cache.set_size(plan_cache_size)


def _irfftn_separable(data, shape):
    '''Apply a low-workspace 3D inverse real FFT as three 1D transforms.'''
    import cupy as cp
    plan_cache = cp.fft.config.get_plan_cache()
    plan_cache_size = plan_cache.get_size()
    plan_cache.set_size(0)
    try:
        transformed = cp.fft.ifft(data, n=shape[0], axis=0, norm='forward')
        transformed = cp.fft.ifft(transformed, n=shape[1], axis=1, norm='forward')
        return cp.ascontiguousarray(cp.fft.irfft(transformed, n=shape[2], axis=2, norm='forward'))
    finally:
        plan_cache.set_size(plan_cache_size)


@dataclass(frozen=True)
class ReconstructionResult:
    output_map: Path
    pixel_size: float
    particle_count: int
    subset: Optional[int]
    symmetry: str


def _default_ftblob():
    radius = 3.8
    alpha = 15.0
    size = 10000
    frequency = np.arange(size, dtype=np.float64) * (0.5 / size)
    aw = 2.0 * np.pi * radius * frequency
    sigma = np.sqrt(np.abs(alpha * alpha - aw * aw))
    ratio = np.empty_like(sigma)
    below = aw <= alpha
    nonzero = sigma != 0.0
    selected = below & nonzero
    ratio[selected] = np.sqrt(2.0 / (np.pi * sigma[selected])) * (
        np.cosh(sigma[selected]) - np.sinh(sigma[selected]) / sigma[selected]
    ) / sigma[selected] ** 1.5
    selected = ~below & nonzero
    ratio[selected] = np.sqrt(2.0 / (np.pi * sigma[selected])) * (
        np.sin(sigma[selected]) / sigma[selected] - np.cos(sigma[selected])
    ) / sigma[selected] ** 1.5
    ratio[~nonzero] = 1.0 / (2.0 ** 1.5 * (3.0 * np.sqrt(np.pi) / 4.0))
    return (2.0 * np.pi) ** 1.5 * radius ** 3 * ratio / np.i0(alpha)


def _iterative_gridding(weight_ft, iterations):
    import cupy as cp
    from .kernels.reconstruct import (
        initialize_gridding_weight_ft,
        multiply_ftblob_3d,
        update_gridding_weight_ft,
    )

    p = weight_ft.shape[0]
    gridding_weight = initialize_gridding_weight_ft(p)
    table = cp.asarray(_default_ftblob())
    logger.info(f'Start gridding correction ({iterations} iterations)')
    for _ in range(iterations):
        convolved = _irfftn_separable(gridding_weight * weight_ft, (p, p, p))
        multiply_ftblob_3d(convolved, table)
        convolved_ft = _rfftn_separable(convolved)
        update_gridding_weight_ft(gridding_weight, convolved_ft)
        del convolved
        del convolved_ft
    logger.info(f'Completed gridding correction ({iterations} iterations)')
    return gridding_weight


def reconstruct(
    dataset,
    symmetry='C1',
    *,
    apply_ctf=True,
    skip_gridding=False,
    gridding_iterations=10,
    batch_size=16,
):
    '''Reconstruct a centered volume from an oriented particle dataset.

    Parameters
    ----------
    dataset : ParticleDataset
        Non-empty particle dataset with one image size and pixel size.
    symmetry : str
        RELION point-group symmetry, C1 by default.
    apply_ctf : bool
        Apply CTF correction, True by default.
    skip_gridding : bool
        Skip iterative gridding preweighting and use stable division.
    gridding_iterations : int
        Number of gridding iterations, 10 by default.
    batch_size : int
        Number of particle images transformed per GPU batch.

    Returns
    -------
    volume : cupy.ndarray
        Shape (n, n, n), dtype float64.
    '''
    import cupy as cp
    from cupy.fft import rfftn
    from torch.utils.data import DataLoader
    from .kernels import get_ctf, translate_2d_ft
    from .kernels.project import quaternions_to_rotations
    from .kernels.reconstruct import (
        backproject_ft,
        cut_sphere_ft,
        enforce_hermitian_ft,
        gridding_correct_3d,
        soft_mask_outside_3d,
        stable_divide_ft,
        symmetrize_ft,
        translate_center_3d_ft,
        window_ft,
    )
    from .symmetry import get_point_group

    if len(dataset) == 0:
        raise ValueError('The particle dataset is empty')
    if batch_size < 1:
        raise ValueError('batch_size must be positive')
    if gridding_iterations < 0:
        raise ValueError('iter must be non-negative')
    image, _ = dataset[0]
    n = image.shape[0]
    if image.ndim != 2 or image.shape != (n, n) or n % 2:
        raise ValueError(f'Particle images must be even square arrays, got {image.shape}')
    p = 2 * n + 3
    ph = p // 2 + 1
    volume_ft = cp.zeros((p, p, ph), dtype=cp.complex128)
    weight_ft = cp.zeros((p, p, ph), dtype=cp.float64)

    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=_collate_particles)
    n_batch = len(loader)
    log_interval = min(max(1, (n_batch + 4) // 5), (10000 + batch_size - 1) // batch_size)
    for i_batch, (images, parameters) in enumerate(loader):
        if images.ndim != 3 or images.shape[1:] != (n, n):
            raise ValueError(
                f'Particle image batch shape differs: {images.shape[1:]} != {(n, n)}'
            )
        stack_ft = rfftn(
            cp.asarray(images, dtype=cp.float64),
            axes=(1, 2),
            norm='forward',
        )
        stack_ft[:, 0, 0] = 0.0
        translations = cp.asarray(parameters[:, :2] + n / 2.0)
        translate_2d_ft(stack_ft, translations)
        if apply_ctf:
            ctf = get_ctf(parameters[:, 6:14], n)
            stack_ft *= ctf
            weights = ctf * ctf
        else:
            weights = cp.ones(stack_ft.shape, dtype=cp.float64)
        rotations = quaternions_to_rotations(parameters[:, 2:6])
        backproject_ft(volume_ft, weight_ft, stack_ft, weights, rotations)
        if (i_batch + 1) % log_interval == 0 or i_batch + 1 == n_batch:
            stop = min((i_batch + 1) * batch_size, len(dataset))
            logger.info(
                f'[Batch {i_batch + 1}/{n_batch}]'
                f'[Particle {stop}/{len(dataset)}] Backprojected particles'
            )

    del stack_ft, weights, translations
    if apply_ctf:
        del ctf
    enforce_hermitian_ft(volume_ft, weight_ft)
    rotations = quaternions_to_rotations(get_point_group(symmetry))
    volume_ft = symmetrize_ft(volume_ft, rotations)
    weight_ft = symmetrize_ft(weight_ft, rotations)
    cut_sphere_ft(volume_ft)
    cut_sphere_ft(weight_ft)

    if not skip_gridding and gridding_iterations > 0:
        volume_ft *= _iterative_gridding(weight_ft, gridding_iterations)
    else:
        stable_divide_ft(volume_ft, weight_ft)

    del weight_ft
    volume_ft = window_ft(volume_ft, 2 * n)
    translate_center_3d_ft(volume_ft)
    volume = _irfftn_separable(volume_ft, (2 * n, 2 * n, 2 * n))
    center = slice(n // 2, 3 * n // 2)
    volume = volume[center, center, center] / (8.0 * n)
    soft_mask_outside_3d(volume, n / 2.0, 3.0)
    gridding_correct_3d(volume)
    return volume


def reconstruct_to_mrc(
    star_path,
    output_path,
    *,
    directory=None,
    pixel_size=None,
    symmetry='C1',
    subset=None,
    apply_ctf=True,
    skip_gridding=False,
    gridding_iterations=10,
    batch_size=16,
):
    '''Reconstruct particles from a STAR file and write an MRC map.'''
    import cupy as cp
    from .ParticleDataset import ParticleDataset

    dataset = ParticleDataset(star_path, directory, pixel_size)
    pixel_size = dataset.pixel_size
    if subset is not None and subset > 0:
        dataset = dataset.get_random_subset(subset)
    volume = None
    try:
        volume = reconstruct(
            dataset,
            symmetry,
            apply_ctf=apply_ctf,
            skip_gridding=skip_gridding,
            gridding_iterations=gridding_iterations,
            batch_size=batch_size,
        )
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        mrcfile.write(output_path, cp.asnumpy(volume).astype('f4'), overwrite=True,
                      voxel_size=pixel_size)
        particle_count = len(dataset)
        return ReconstructionResult(
            output_path, pixel_size, particle_count, subset, symmetry
        )
    finally:
        volume = None
        cp.fft.config.get_plan_cache().clear()
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()


def parse_argument():
    parser = argparse.ArgumentParser(
        description='Reconstruct a 3D map from particles in a RELION STAR file.'
    )
    parser.add_argument('--i', required=True, help='input particle STAR file')
    parser.add_argument('--o', default='reconstruct.mrc', help='output MRC map')
    parser.add_argument('--directory', help='particle stack directory')
    parser.add_argument('--sym', default='C1', help='point-group symmetry')
    parser.add_argument('--subset', type=int, default=-1, help='random subset')
    parser.add_argument('--angpix', type=float, help='output pixel size')
    parser.add_argument('--ctf', action='store_true', help='apply CTF correction')
    parser.add_argument('--skip_gridding', action='store_true', help='skip iterative gridding correction')
    parser.add_argument('--iter', type=int, default=10, help='number of gridding iterations')
    parser.add_argument('--batch_size', type=int, default=16, help='GPU particle batch size')
    if len(sys.argv) == 1:
        parser.print_help()
        raise SystemExit(0)
    return parser.parse_args()


def main():
    configure_logging(source='cryosieve-reconstruct')
    args = parse_argument()
    from .utility import check_cupy
    check_cupy()
    reconstruct_to_mrc(
        args.i, args.o, directory=args.directory, pixel_size=args.angpix,
        symmetry=args.sym, subset=args.subset, apply_ctf=args.ctf,
        skip_gridding=args.skip_gridding, gridding_iterations=args.iter,
        batch_size=args.batch_size,
    )


if __name__ == '__main__':
    main()
