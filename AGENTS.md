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
cryosieve-csrefine -h
cryosieve-csrhbfactor -h
```

There is no configured automated test, linter, formatter, or type-checker.
Validate builds and CLIs, then run focused workflows on representative STAR/MRC
inputs. External toy data lives in the `mxhulab/cryosieve-demos` repository.
Some environments require `MKL_THREADING_LAYER=GNU` and `OMP_NUM_THREADS=1`.

## Entry points and workflow

- `src/cryosieve/__main__.py` implements the iterative `cryosieve` workflow. It
  creates `iter0.star`, reconstructs two half maps per iteration, optionally
  postprocesses them, and invokes `cryosieve-core` for the next STAR file.
- `src/cryosieve/core.py` implements scoring/sieving and writes both retained
  output and sibling `<name>_sieved.star` discarded particles.
- `src/cryosieve/cs_refine.py` drives CryoSPARC import/refinement and summary
  output. `src/cryosieve/cs_rhbfactor.py` builds subsets and estimates the
  Rosenthal-Henderson B-factor.
- `src/cryosieve/reconstruct_runner.py` adapts reconstruction execution. Keep
  its command and output contracts synchronized with the top-level workflow.

Keep the four console entry points and `module.yaml` commands stable because the
CoCo parent generates scripts and parses their artifacts/logs. Use the existing
explicit file logger and stable job-local filenames; do not fall back to the
default stderr logging handler.

## GPU and metadata constraints

`ParticleDataset.py` bridges RELION STAR metadata and mmap-backed MRCS images.
Preserve pre-3.1 and optics/particles STAR support, random subset semantics,
Euler-to-quaternion convention, translation direction/units, CTF ordering, and
filtered STAR structure.

`sieve.py` combines CuPy and PyTorch through DLPack. Preserve device ownership,
contiguity, dtype, and lifetime across the bridge. Kernel changes under
`src/cryosieve/kernels/` require numerical comparison for projection, CTF,
Fourier filters, rotation, and translation rather than only a successful run.
All Python and CUDA C++ code under that directory must follow the authoritative
kernel conventions in `src/cryosieve/kernels/AGENTS.md`. New code follows those
conventions immediately; existing kernels may be migrated incrementally during
the v2.0.0 development cycle.

Multi-GPU execution is single-node, one process, and one Python worker thread
per GPU. Each worker selects its CUDA device, handles a contiguous particle
shard, and writes to a shared CPU score array before global sorting. Do not
replace this contract with `torchrun`/DDP incidentally. The requested GPU count
must remain bounded by available devices, and the number of `--volume` inputs
must match the particle random subsets.

The top-level reconstruction command uses RELION-style flags such as `--subset`,
`--ctf`, `--angpix`, and `--sym`. CryoSPARC helpers require a real CryoSPARC
environment and valid user/project/workspace/lane identifiers.

Packaging is defined by `pyproject.toml`; PyPI release automation is under
`.github/workflows/`. The CoCo superproject owns mapping fragments outside this
checkout. Coordinate command, artifact, capability, or job-type changes with
`module.yaml` and the parent instead of adding parent metadata here.
