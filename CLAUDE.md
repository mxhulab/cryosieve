# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development commands

- This project needs a CUDA-capable Python environment. The core CLIs call `cryosieve.utility.check_cupy()` and exit early if CuPy/CUDA is unavailable.
- Python requirement: `>=3.7`.
- Install for local development:
  - `python -m pip install -e .`
- Build the package artifacts used by the release workflow:
  - `python -m pip install -U build wheel twine`
  - `python -m build`
- Verify the installed entrypoints / inspect CLI arguments:
  - `cryosieve -h`
  - `cryosieve-core -h`
  - `cryosieve-csrefine -h`
  - `cryosieve-csrhbfactor -h`
- There is no automated test suite or lint configuration in this repo. No `pytest`/`unittest` tests or `ruff`/`black`/`mypy` config are checked in, so there is no repo-native “run all tests” or “run a single test” command.
- Validation is currently by building the package and running the CLIs on real STAR/MRC inputs. The README points users to the external `mxhulab/cryosieve-demos` repository for toy/demo datasets.
- Runtime fixes called out in the README:
  - `export MKL_THREADING_LAYER=GNU`
  - `export OMP_NUM_THREADS=1`

## High-level architecture

### Entry points

- `src/cryosieve/__main__.py` implements `cryosieve`, the top-level iterative workflow.
  - It creates `iter0.star`, then for each iteration runs two half-map reconstructions with an external RELION-style command, optionally postprocesses the half maps, and invokes `cryosieve-core` to produce the next filtered STAR.
  - Frequency thresholds are scheduled from `--frequency_start` to `--frequency_end` across `--num_iters`.
- `src/cryosieve/core.py` implements `cryosieve-core`, the actual sieving command.
  - It reads one STAR file, one masked volume per random subset, and writes the retained output STAR plus a sibling `<name>_sieved.star` for discarded particles.
- `src/cryosieve/cs_refine.py` implements `cryosieve-csrefine`.
  - It automates CryoSPARC particle import, ab-initio or reference import, homogeneous/non-uniform refinement, optional local refinement, and CSV summary generation.
- `src/cryosieve/cs_rhbfactor.py` implements `cryosieve-csrhbfactor`.
  - It generates progressively halved STAR subsets, calls `cryosieve-csrefine`, and fits Rosenthal-Henderson B-factors from the resulting resolution/particle-count data.

### Core data path

- `src/cryosieve/ParticleDataset.py` is the format bridge between cryo-EM metadata and the GPU code.
  - It supports both pre-RELION-3.1 STAR files and RELION 3.1+ `optics`/`particles` STAR files.
  - It parses translations, converts Euler angles to quaternions, extracts CTF parameters, lazily mmaps `.mrcs` slices, and writes filtered STAR outputs.
- `src/cryosieve/sieve.py` is the batch scoring loop.
  - For each particle batch it translates images, projects the masked 3D volume into 2D using the particle quaternions, applies the particle CTF, high-pass filters both images and residuals, computes a score, and retains the lowest-scoring particles according to `--retention_ratio`.
- `src/cryosieve/kernels/` contains the CuPy/CUDA primitives used by the scoring path:
  - `project.py` for volume projection
  - `ctf.py` for CTF generation/application
  - `bandpass.py` for Fourier-space filtering
  - `translate.py` / `rotate.py` for image-space transforms
- The GPU path mixes CuPy and PyTorch. `sieve.py` uses DLPack bridges when moving tensors between them, so changes in this area need to preserve device placement and tensor/array compatibility.

### Multi-GPU and external-tool assumptions

- Multi-GPU support is single-node only and implemented inside one `cryosieve-core` process with Python threads, not `torchrun`/DDP.
- `cryosieve-core --num_gpus N` checks the requested count against `cupy.cuda.runtime.getDeviceCount()`, then `sieve.py` starts one worker thread per GPU.
- Each GPU worker calls `cp.cuda.runtime.setDevice(device_id)`, processes a contiguous shard of particles, writes its scores into a shared CPU NumPy array, and the main thread sorts the combined scores to decide retained particles.
- `cryosieve-core` expects the number of input `--volume` arguments to match the number of particle random subsets (`rlnRandomSubset`).
- The top-level `cryosieve` wrapper assumes the reconstruction command understands RELION-style flags such as `--subset 1/2`, `--ctf true`, `--angpix`, and `--sym`.
- The CryoSPARC helper scripts expect a real CryoSPARC environment and IDs (`--user`, `--project`, `--workspace`, `--lane`) plus the command-core environment variables used by `cryosparcm env`.

## Packaging and release

- Packaging metadata and console-script registration live in `pyproject.toml`.
- The GitHub Actions release workflow in `.github/workflows/pippublish.yml` builds with `python -m build` and publishes to PyPI when a GitHub release is published.
