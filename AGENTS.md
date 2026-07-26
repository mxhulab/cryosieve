# CryoSieve Development Guide

## Commands and environment

CryoSieve requires Python `>=3.7` and a CUDA-capable environment. Core commands
call `cryosieve.utility.check_cupy()` and exit when CuPy/CUDA is unavailable.

```bash
python -m pip install -e .
python -m pip install -U build wheel twine
python -m build
python -m compileall src/cryosieve
cryosieve -h
cryosieve-core -h
cryosieve-reconstruct -h
cryosieve-postprocess -h
cryosieve-csrefine -h
cryosieve-csrhbfactor -h
```

There is no configured automated test, linter, formatter, or type-checker.
Validate builds and CLIs, then run focused workflows on representative STAR/MRC
inputs. External toy data lives in the `mxhulab/cryosieve-demos` repository.
Some environments require `MKL_THREADING_LAYER=GNU` and `OMP_NUM_THREADS=1`.

## Entry points and workflow

- `src/cryosieve/__main__.py` implements the iterative `cryosieve` workflow. It
  creates `iter0.star`, reconstructs two half maps per iteration with
  `src/cryosieve/reconstruct.py`, postprocesses them with
  `src/cryosieve/postprocess.py`, and calls `core.process()` in-process for the
  next STAR file.
- `src/cryosieve/core.py` implements scoring/sieving and writes both retained
  output and sibling `<name>_sieved.star` discarded particles.
- `src/cryosieve/cs_refine.py` drives CryoSPARC import/refinement and summary
  output. `src/cryosieve/cs_rhbfactor.py` builds subsets and estimates the
  Rosenthal-Henderson B-factor.
- `src/cryosieve/reconstruct.py` implements single-GPU Fourier reconstruction.
  `src/cryosieve/reconstruct_runner.py` remains for external adapter users.

Keep the six `pyproject.toml` console entry points and the commands listed in
`module.yaml` stable because the CoCo parent generates scripts and parses their
artifacts/logs. Use the explicit file logger and stable job-local filenames; do
not fall back to the default stderr logging handler.

## GPU and metadata constraints

`ParticleDataset.py` bridges RELION STAR metadata and mmap-backed MRCS images.
Preserve pre-3.1 and optics/particles STAR support, random subset semantics,
Euler-to-quaternion convention, translation direction/units, CTF ordering, and
filtered STAR structure.

Kernel changes under `src/cryosieve/kernels/` require numerical comparison for
projection, CTF, Fourier filters, rotation, translation, reconstruction, and
postprocessing rather than only a successful run. Follow the authoritative
conventions in `src/cryosieve/kernels/AGENTS.md`.

Multi-GPU sieving is single-node, one process, and one Python worker thread per
GPU. This is a deliberate performance contract: do not add multiprocessing,
`torch.distributed`, NCCL collectives, DDP, process spawning, rank contexts, or
barriers to `cryosieve`, `cryosieve-core`, or `sieve.py`. External launch policy
belongs to the caller, not this package. Each worker must select its CUDA device
before any CuPy allocation, handle one contiguous particle shard, and write only
its slice of the shared CPU score array; the main thread globally sorts after
all workers join. Preserve worker exception propagation.

Reconstruction and postprocessing are single-GPU and sequential. Keep the
explicit lifetime cleanup in `reconstruct_to_mrc()`: clear the CuPy FFT plan
cache and free default device and pinned memory pools after every half map. Do
not retain batch Fourier arrays or reconstruction volumes across half maps;
these releases prevent iteration-to-iteration GPU OOM failures.

The requested sieving GPU count must remain between 1 and the available device
count. The number of `--volume` inputs must match the particle random subsets.

The top-level reconstruction command uses RELION-style flags such as `--subset`,
`--ctf`, `--angpix`, and `--sym`. CryoSPARC helpers require a real CryoSPARC
environment and valid user/project/workspace/lane identifiers.

Packaging is defined by `pyproject.toml`; PyPI release automation is under
`.github/workflows/`. The CoCo superproject owns mapping fragments outside this
checkout. Coordinate command, artifact, capability, or job-type changes with
`module.yaml` and the parent instead of adding parent metadata here.
